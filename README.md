# EchoMind

腾讯云实时字幕（试用）：在“语音 → 识别引擎”选择“腾讯云（在线 / 上传音频）”，配置独立的腾讯云凭证并确认上传后使用。不需要本地模型/GPU；默认仍保留本地识别。[开通、配置、隐私与测试说明](docs/tencent-asr.md)。

EchoMind is a Windows conversation assistant. It captures audio from a selected Tencent Meeting process and shows live Chinese captions in a terminal. MVP 2 adds Chinese question detection, optional Jev semantic confirmation, and streaming answers. Knowledge retrieval, advanced routing, and the desktop overlay are later milestones.

## MVP 1 setup

Requirements: Windows 10 build 20348 or newer, .NET 9 SDK, Python 3.11, and an NVIDIA GPU for the recommended CUDA path. Capturing a meeting requires consent from its participants. Audio is processed in memory; this MVP does not save recordings or transcripts.

From the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".\services\transcription[dev]"
dotnet build .\native\audio-capture\EchoMind.Capture.csproj -c Release
.\scripts\download-model.ps1 -Model large-v3-turbo
.\scripts\install-cuda-runtime.ps1
.\scripts\run-captions.ps1
```

The last command lists Tencent Meeting processes. Select the process producing remote audio, or pass `-TargetPid 12345`. For a quick pipeline smoke test, download the smaller `tiny` model and run `run-captions.ps1 -Model tiny`; it is not intended for production Chinese captions. Use `-Device cpu` if CUDA is unavailable (the CUDA runtime step can then be skipped).

The model download script uses official Hugging Face model files and stores them under the ignored `.models/` directory. The CUDA script downloads the Windows CUDA 12/cuDNN 9 libraries linked by [faster-whisper](https://github.com/SYSTRAN/faster-whisper#gpu), checks the archive hash, and stores them under the ignored `.runtime/` directory. The first model download is large and may take time.

Run logic tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests services\transcription\tests
```

This version uses Windows application process loopback: it captures the selected PID and its child processes, not the entire system output. Tencent Meeting may play audio in a child or separate process; if captions stay silent, select the actual audio process. Partial captions can change before they become final. The current lightweight energy gate may miss quiet speech; replacing it with Silero VAD is a follow-up after baseline measurements.

## Layout

- `native/audio-capture/`: Windows process loopback to 16 kHz mono PCM.
- `services/transcription/`: Chinese faster-whisper captions, question detection, streaming answers, and CLI.
- `scripts/`: local model and runtime setup, plus the launch command.

## Windows 窗口版

Qwen3-ASR-1.7B 本地接入版：`dist/light-v0.1.3-qwen17/EchoMind/EchoMind.exe`。在“语音 → 识别引擎”选择“Qwen3-ASR-1.7B（本地 GPU）”，复用本机独立部署的模型与 GPU 运行环境；已接入语音检测与异常拦截，不传热词。它不是通用便携组件，也不自动下载。[部署要求、测试结果及限制](docs/qwen17-local-test-2026-10-12.md)。

本地 ASR 对比版：`dist/light-v0.1.3-asr-comparison/EchoMind/EchoMind.exe`。“语音 → 识别引擎”新增 SenseVoiceSmall 与流式 Paraformer CPU 实验选项，Whisper 默认不变。两种候选在当前录音上更快，但技术词更差；[实测报告与复现步骤](docs/asr-comparison-2026-10-10.md)给出了限制、耗时与具体错误。开发/打包需安装 `services/transcription[dev,desktop-build,asr-onnx]`；模型独立保存在 `.models/`，不捆入主程序。

轻量窗口版入口：`dist/light-v0.1.3-local-components/EchoMind/EchoMind.exe`（旧版保留在原目录）。双击即可运行，无需打开命令行。请保留整个 `EchoMind` 文件夹，不能只移动 EXE；主程序只含 Python 运行时和音频采集组件，不再捆绑模型与独立 CUDA，压缩后约 129 MiB。

首次启动会打开“语音组件”窗口，选择 GPU 或 CPU 后点击“下载并准备”。中文识别模型下载约 1.39 GiB，GPU 组件另需约 918 MiB；CPU 不下载 GPU 组件。下载来自已发布的 EchoMind v0.1.1 GitHub Release，固定 SHA-256 校验后安装；网络不通可稍后重试，取消或中断的下载支持续传（服务器不支持续传时重新下载）。选择 GPU 仍需支持 CUDA 的 NVIDIA 显卡和系统驱动，此功能不安装显卡驱动。组件保存在 `%LOCALAPPDATA%\EchoMind\components`，应用更新后复用；模型和 GPU 顺序安装，需额外空间容纳压缩包与解压暂存（首次 GPU 全量准备建议缓存盘至少空闲 5 GiB）。

可关闭准备窗口，先进行文字/截图回答；开始采集前会再次检查缺失组件。左下角“设置 + → 组件管理”可查看路径、切换设备并重试。完整旧版程序旁的 `models/large-v3-turbo` 和 `cuda` 优先复用，不重复下载；下载只写用户缓存，不改旧版程序、配置或密钥。失败的安装不启用临时文件，不完整的旧缓存保留为 `.backup-*` 供恢复。

本地开发打包的 EXE 还会复用附近 EchoMind 项目的 `.models/large-v3-turbo` 与 `.runtime/cuda12`。查找优先级：程序旁的完整组件 → 用户缓存的完整组件 → 附近项目中的完整开发组件；每个组件独立判断，不完整时才提示下载。仅检查程序目录及上方最多四层、且存在 `services/transcription/pyproject.toml` 的项目目录，不使用当前工作目录，不扫描磁盘，不硬编码开发者路径，也不复制或删除已有组件。程序移到项目外时仍按便携目录/用户缓存规则工作。

窗口采用 Ant Design 风格的蓝色操作、浅灰背景、白色卡片、分层标题与标签页，统一输入框、列表选中和键盘焦点状态。主操作按钮使用较深蓝色以保持白色文字可读性。当前仍是 Tkinter 原生桌面实现，不包含 antd React 组件，也不更改语音识别、纠错和配置流程。

页签在同组内等宽等高，切换时不再因原生主题的选中边距产生错位；设置页签与回答页签按各自标题长度分组定宽。滚动条统一使用无箭头、无纹理的细滑块，悬停或拖动时加深颜色，保留原生拖动、滚轮及键盘滚动行为。

1. 先选择“离线演练”，在窗口底部问题编辑框输入主题或问题，点击“提交回答”，确认界面工作正常。演练内容是固定示例，不访问云服务。
2. 在“基本”页选择“DeepSeek 在线回答”，在“模型”页填写 API 地址、模型及密钥。点击“保存配置（含加密密钥）”，下次启动会自动读取，无需重复输入。
3. 打开腾讯会议，点击“刷新”，选择会议进程；确认已取得参与者授权后，在设置区勾选授权确认，点击“开始采集”。设置区可以滚动；点击“停止”结束采集。
4. “Jev”页的语义补判是可选项，需要单独的 TypeSafe 密钥；启用后本地规则未命中的确定字幕及有限上下文会发送到 TypeSafe。规则命中明确提问时直接回答，不调用 Jev；疑似误听仍先复核。在线回答会将问题和有限问答上下文发送至所配置的回答服务，不上传原始音频。

回答按接口返回的文本增量持续追加，不等待整篇完成；生成期间显示“正在流式生成回答”及模型首字耗时。界面每轮最多处理 4 个回答增量或约 8 ms，然后让出绘制和输入处理；问题记录也追加增量，回看时不因每段文字替换全文而跳到底部。离线演练以 120 ms 间隔展示固定示例段落，仅用于检查流式界面，不代表在线模型性能。在线回答不人为拆字或延迟；如果服务或代理将完整内容集中返回，客户端无法提前展示尚未收到的内容，真实网络效果仍需实测。

### 不开会议，使用本机麦克风测试

在窗口上方把“输入来源”切换为“本机麦克风”，点击“开始测试”，对着电脑麦克风说话。使用 Windows 系统默认输入设备，无需打开腾讯会议或选择 PID，也无需勾选会议采集授权；仅在有权处理现场声音时使用。仍需本地 Whisper 模型及相应 CPU/GPU 运行组件。

麦克风以设备支持的采样率采集，转换为 16 kHz 单声道 PCM，送入现有 ASR / 问题判断 / 回答 / 问题记录流程。输入音量显示在右上方；不播放声音、不保存录音。点击“停止”释放麦克风并停止生成。设备不可用、断开、音频丢失或识别跟不上时会提示检查输入设备、权限或系统负载。首次使用若权限未开启，请在 Windows 设置中手动允许桌面应用访问麦克风；应用不会自动更改权限。

“仅字幕”不生成回答；“离线演练”只显示固定回答，适合先验证识别和触发；真实回答需选择在线模式及填写密钥。只有在线模式会发送问题文本与有限上下文，启用 Jev 时另发送有限转写文本；原始音频不发送。问题规则支持“讲解一下 React Fiber 原理”、“讲解一下”后接主题（8 秒内）以及“五乘六等于多少”，仍可能漏判其他表达。本次不提供音频文件读取入口。

右侧“问题记录”页保留本次运行中已触发回答的最近 100 条问题，包含时间、生成状态和对应回答；点击或用方向键选择即可回看。新问题不会打断正在回看的旧记录。停止再开始采集仍保留记录，退出应用后清空；不写入磁盘、不自动上传历史记录。单条回答最多保留 20,000 字符，正文显示沿用 400 行限制。仅字幕模式、被规则漏判或被 Jev 忽略的发言不进入该记录。

当前是 Tkinter 普通桌面窗口基础版，字幕与回答分栏显示；尚不是 Tauri 置顶透明 Overlay，也没有安装器或代码签名。真实会议采集、云服务回答质量和端到端延迟仍需实测。

麦克风待处理缓冲上限为约 30 秒（原为 10 秒），可以容忍短时识别卡顿；这不是录音时长限制。持续发言仍会持续采集，但积压音频会增加字幕延迟，缓冲满后会提示并停止，不静默丢弃音频。加长缓冲不会提高识别速度。

### 简体字幕、问题状态与配置

字幕和问题检测入口统一用 OpenCC 转为简体，避免繁体“解釋一下”漏过简体提问规则。最终识别使用 beam 5 与确定性解码，不再注入可能被回显的自然语言提示词；纯零音频不解码，同时过滤高静音概率且低可信的片段。其余低可信文本保留并标记待复核，不保证技术词完全准确。右侧显示提问检测状态，包括仅字幕模式、问题待补全、普通发言、重复提问及 Jev 忽略；既有回答不会被这些状态清空。

### 疑似误听问题复核

“设计一个／实现一个／制定一套方案”等完整任务也可直接触发回答。任务开头或以“方案”结尾的主题，会保留最多 12 秒的衔接上下文；后续以“支持／要求／需要／并且”等开头的条件可接上，最多合并 3 段、200 字。完整任务先回答，后续条件到达后按完整内容更新回答，不额外等待 12 秒；无关的新发言、超时或手动提交会断开衔接。“嗯／OK”等语气词不作为任务内容。并非任意停顿或任意多段都会自动合并。

例如“这几个AI前端挥布发布方案”后接“支持按用户ID设备类型地理位置逐步放量”，会保留两段原文并提示复核；不会直接硬改为“设计一个AI前端灰度发布方案”。在线模式可展示修正建议，仍需核对后手动提交；离线无法可靠还原时请编辑原文。Jev 未命中提示会显示对应发言，避免把“OK”的低概率判断误认为前一道题的结果。原字幕保持不变。

“1加1等于几”现可直接识别为问题；“1加1等于进”等疑似错字、部分技术词疑似误听或低可信提问，会先复核再决定是否回答。窗口版在线模式复用回答模型的接口与密钥，额外发送该句和最多 4 段近期字幕（每段最多 200 字），可能增加费用与延迟；明确且无疑似错误的问题不增加复核调用。离线演练及命令行仅做本地保守检查，不为纠错调用云服务；仅字幕模式不复核问题。

任何含义变化的修正建议都不会自动生成答案：右侧展示原识别、建议及原因，点击“使用建议编辑”，核对或修改后点击“提交回答”。原字幕、已有回答和未主动选择替换的草稿保持不变；确认后才生成新答案与问题记录。无法可靠还原、复核超时或返回无效时，不猜测原意，请手动编辑后提交。低可信标记仅是复核信号，不证明识别有误；尚未用真实麦克风数据量化准确率改善。

配置文件位于 `%LOCALAPPDATA%\EchoMind\settings.json`，保存接口地址、模型、回答模式、识别设备、输入来源和 Jev 开关；密钥使用 Windows DPAPI 加密，不以明文写入。仅相同 Windows 用户环境能正常解密，请勿将此文件作为跨电脑密钥备份。更换密钥后再次保存；清空密钥后保存可从配置移除。会议授权确认不保存，每次需重新确认。环境变量如已设置，优先于配置文件。读取失败会提示，损坏配置不会被自动覆盖；不读取/上传录音、字幕或问题记录。

### 选中字幕提问与问题编辑

默认快捷键：问题输入框按 Enter（或小键盘 Enter）提交回答；Ctrl+R 开始当前输入来源的音频采集；Ctrl+S 停止会话并停止生成。旧的 Ctrl+Enter 提交仍保留在问题输入框中，字幕区域 Ctrl+Enter 仍是选中字幕加入问题。快捷键仅在 EchoMind 主窗口有效，不是系统全局热键；截图核对弹窗和其他输入框的 Enter 不会提交问题。采集不保存录音文件。

Ctrl+F 切换窗口跟随：采集中也可切换，退出跟随不停止采集。仅在 EchoMind 主窗口（含紧凑界面）有焦点时生效，弹窗、关闭中及麦克风模式不切换。键位可在“设置 → 快捷键设置”修改并保存，下次启动读取；不得与其他操作重复。旧配置若已把 Ctrl+F 用于其他操作，会保留原键位并为跟随选择未占用的 Ctrl+Alt+F 或功能键。快捷键切换只改变本次跟随状态，持久启用请在窗口跟随设置保存。

输入来源仅保留一个选择框：“腾讯会议 / 本机麦克风”，移除重复的“应用”选择框，不显示 PID 列表；后台自动选择 `wemeetapp.exe` 主进程并捕获其子进程。不将崩溃处理器或辅助进程当作主进程。尚未启动或重启腾讯会议后可点击刷新重新连接。当前基础版仅支持腾讯会议，不表示已支持任意应用音频捕获。

顶部“收起设置 / 展开设置”按钮可隐藏或恢复左侧运行设置栏，字幕与回答自动使用释放的空间。密钥、草稿、授权勾选和设置页签不清空，采集中也可操作。默认展开，不额外写入配置；窗口跟随紧凑模式结束后保留本次设置栏的展开状态。收起时要更改配置或授权，请先点“展开设置”。

主界面采用紧凑两层控制区：输入来源在左，刷新/开始/停止同排靠右；快捷键与跟随状态在下方自动换行。麦克风模式隐藏会议刷新按钮。字幕“加入问题”与标题同排，问题编辑固定在底部；卡片内边距统一为 12，外侧留白 16。新增可用空间优先分配给回答区（字幕/回答伸展权重 2:3），仍可拖动分隔线调整。小窗口及设置栏收起时保留主要按钮和文本区可见。

### 检查版本与一键更新

当前程序版本为 `0.1.1`。窗口版启动后后台检查一次，也可以从左下角“设置 → 检查更新”手动检查。发现新版时展示版本与更新说明，点击“立即更新”并确认后下载、校验，停止当前会话，退出替换程序并重新启动。下载失败不修改当前程序；替换失败或新进程在启动后 5 秒内退出，更新助手尝试恢复旧程序。源码运行只检查版本，不自动覆盖开发目录。

更新来源固定为 `xzmybyg/EchoMind` 的公开 GitHub Releases，使用最新正式版；私有仓库、预发布版本、没有更新包或缺少 GitHub SHA-256 digest 时不安装。HTTPS 与 SHA-256 用于传输和完整性校验，不替代代码签名。检查请求不发送会议内容、模型密钥或本地配置。应用目录需要当前用户有写入权限，不自动请求管理员权限。

开发者发布流程（本次实现不会自动创建 Release）：

1. 修改 `services/transcription/echomind/version.py` 的 `VERSION` 与 `services/transcription/pyproject.toml` 版本，再打包到新目录。
2. 运行 `.\.venv\Scripts\python.exe scripts/package-update.py dist/light-v0.1.3/EchoMind dist/light-v0.1.3-update` 生成 `EchoMind-update.zip`；脚本核对 EXE 内版本，拒绝覆盖已有 ZIP。
3. 在 GitHub 创建对应的正式 Release，例如 `v0.1.3`，附上更新说明，并上传该 ZIP。GitHub 接口应提供资产的 `sha256:...` digest。语音组件复用 v0.1.1 的模型/GPU 资产，请保留该 Release 和组件文件。
4. 下次发布更高版本（例如 `v0.1.4`）后，本次 `0.1.3` 应用才能检测到更新。

更新 ZIP 仅包含 `EchoMind.exe`、`_internal`、`capture` 和 `update.json`，不重复下载模型和独立 CUDA 目录，也不覆盖配置与密钥。下载缓存位于 `%LOCALAPPDATA%\EchoMind\updates`；安装目录内 `.update-backup-*` 保留旧程序供恢复，失败的替换文件保留在 `.update-failed-*`。更新准备副本先放到应用所在盘，支持 C 盘下载、D 盘安装。当前不会自动删除旧版备份；更新需要足够空间容纳下载、准备副本和备份。这是目录版基础更新器，不支持安装器、跨 Python 大版本运行时迁移或完整的系统安装事务。

点击左下角蓝色“设置 ▾”按钮，展开 Ant Design 风格的下拉浮层：圆角白色面板、轻阴影、统一菜单行和悬停高亮，不挤压侧栏内容。底部空间不足时自动向上展开，窗口缩放时位置跟随按钮。保留快捷键设置、检查更新、组件管理与窗口跟随入口；选择一项先收起菜单再打开对应窗口。再次点击、点击菜单外、窗口失焦、最小化或按 Esc 收起；方向键/Tab 切换菜单项，Enter 选择，Esc 返回设置按钮。在快捷键窗口修改提交、开始、停止键位，点击“保存”立即生效并在下次启动读取；“取消”、Esc 或关闭窗口不改变原快捷键。基本、模型与 Jev 配置仍在原位置。采集或生成期间设置入口禁用，请先停止会话。支持 Enter、Ctrl/Alt/Shift 组合键、F1–F12，例如 Alt+Enter、F8、F9；开始/停止不接受单独 Enter 或字母，避免输入时误触。重复键位和复制、粘贴等保留键会显示错误，不覆盖配置或原快捷键。按钮禁用、停止中或关闭中时快捷键也不触发，不绕过会议授权、接口配置与模式检查。

截图或复制图片后，点击问题编辑区“粘贴题目截图”，或在问题编辑框中按 Ctrl+V。剪贴板图片直接在内存中识别，不写临时图片文件，不清空或覆盖剪贴板；只在主动粘贴时读取，不持续监听。剪贴板为文字时 Ctrl+V 保留普通文字粘贴，模型/密钥等其他输入框不触发截图识别。“选择图片”仍可选 PNG、JPG/JPEG 或 BMP 文件（最多 20 MB）。Windows 简体中文 OCR 在本机识别，图片不上传、不复制到项目，也不需要 API 密钥；需本机已安装简体中文 OCR 语言组件。图片尺寸超出系统 OCR 上限时会提示裁剪为单题。识别在后台进行，不阻塞字幕采集。

识别完成后在弹窗中核对、修改文字。多道题可拖选一题后点击“选中题目加入”，或点击“全部加入”；文字追加到已有草稿，不自动回答。点击主窗口“提交回答”才会使用既有演练/在线回答流程；在线模式发送题目文字与有限问答上下文，不发送图片。仅识别截图无需开启会议、麦克风或 ASR 模型。OCR 可能误读代码、公式和细小文字，请先核对。本版不自动截屏，也不读取复制文件路径对应的图片；请复制图片本身。

截图核对弹窗将文字区与底部操作区分开，只有文字区伸缩和滚动；提示自动换行，最小宽度按按钮实际宽度计算，避免缩小窗口或字体缩放时按钮被挤出。Ctrl+Enter 将全部文字加入主窗口问题框，Esc 取消；普通 Enter 仍可在弹窗编辑文字换行，不会自动回答。

拖选左侧字幕，点击“选中字幕加入问题”（字幕区域也支持 Ctrl+Enter），把文字加入底部可编辑的问题框；自动去掉完整行开头的时间，多次加入可以合并多段字幕。加入和编辑不会触发模型调用。可先修正听错的词，再点击“提交回答”（编辑框支持 Ctrl+Enter）。无需开启会议也可直接输入主题；手动提交不依赖疑问词或 Jev 自动判断，例如“React Fiber 原理”也可以生成回答。需选择演练或在线模式；在线模式依然将问题及有限问答上下文发送给已配置的服务。

采集期间编辑框仍可使用，提交到同一个回答会话，不会重新启动或停止音频采集。提交新问题会取消旧回答。当前回答上的“编辑当前问题”、问题记录页的“编辑选中问题”会复制问题到编辑框；修改后提交产生新记录，原问题和原回答保持不变。草稿不保存到配置文件。

开发者从源码启动窗口：`.\.venv\Scripts\pythonw.exe -m echomind.gui`。开发环境继续复用仓库 `.models/large-v3-turbo` 和 `.runtime/cuda12`，不强制联网下载，也不改 CLI 的既有路径。重新打包使用 `scripts/build-desktop.ps1 -OutputDirectory dist/light-v0.1.3`，默认轻量，不要求本地模型/CUDA 构建输入；离线完整包可加 `-IncludeComponents`（需要已有本地组件）。指定未包含旧 EchoMind 包的输出目录，脚本不会覆盖已有包。

## MVP 2: 中文问题与流式回答

窗口跟随：在“设置 ▾ → 窗口跟随”勾选启用，选择左侧/右侧并保存，或用 Ctrl+F 切换。跟随所选腾讯会议应用自动识别的主进程及其子进程窗口，无需手选 PID。无需开始录制；默认关闭。会议移动时约每 250 ms 同步 EchoMind 的位置，会议最小化时同步最小化，恢复后仅恢复由跟随功能最小化的 EchoMind；手动最小化不会被强制还原。不移动/调整会议、不抢输入焦点、不强制置顶。找到会议窗口后自动切换紧凑界面，默认约 420×620（按 DPI 缩放），上下显示字幕与问题记录，隐藏设置、工具栏和回答正文；后台采集与回答继续运行，原有快捷键仍可开始/停止。可调整紧凑窗口大小。侧边不够时尝试另一侧，再限制在目标显示器的工作区域（可能重叠）。点击“退出跟随”关闭本次跟随并恢复原界面、窗口尺寸与位置，不改已保存的跟随配置。麦克风模式恢复完整界面；弹窗操作期间暂停移动；所选窗口消失时恢复完整界面并等待重新出现。应用重启后可刷新自动识别新的主进程。跟随设置下次启动读取。

默认仍是仅字幕模式，不访问回答 API。先不启动会议或 GPU，验证问题检测与展示：

```powershell
.\.venv\Scripts\python.exe -m echomind.cli --answers demo --text-question "Redis为什么这么快？"
```

`demo` 输出明确标记的固定演练文本，不是真实模型回答。会议中演练：

```powershell
.\scripts\run-captions.ps1 -Answers demo
```

真实回答使用 OpenAI 兼容的 Chat Completions 流式接口。以 DeepSeek 为例，在当前 PowerShell 会话中配置地址、模型和密钥；密钥请仅在本机设置，不要贴到聊天或提交到仓库：

```powershell
$env:ECHOMIND_LLM_BASE_URL = 'https://api.deepseek.com'
$env:ECHOMIND_LLM_MODEL = 'deepseek-flash'
$env:ECHOMIND_LLM_THINKING = 'disabled'
# Set ECHOMIND_LLM_API_KEY locally, then run:
.\.venv\Scripts\python.exe -m echomind.cli --answers live --text-question "Redis为什么这么快？"
.\scripts\run-captions.ps1 -Answers live
```

根据 [DeepSeek 官方文档](https://api-docs.deepseek.com/)，当前示例模型为 `deepseek-flash`；[思考模式](https://api-docs.deepseek.com/guides/thinking_mode/)默认开启。这里显式关闭思考以优先验证首字延迟。其他兼容服务可替换地址与模型；地址为 API 根地址（需要时包含 `/v1`），不要包含 `/chat/completions`。其他服务不支持 `thinking` 时，移除 `ECHOMIND_LLM_THINKING` 环境变量。

只有 `live` 模式会发送检测到的问题及最近最多 4 轮成功问答到所配置的服务；不会上传原始音频。会议转写或问题可能含敏感内容，使用前应取得参与者授权并确认服务的数据政策。程序不保存密钥、录音或转写；退出后不保留问答上下文。

默认问题检测使用保守的中文规则：只在确定字幕上触发，支持跨片段拼接、常见转述/自我讲解/问题措辞讨论过滤和重复抑制，但可能误判或漏判。新确认的问题会取消旧回答，回答在后台流式生成，不阻塞字幕。尚无简历/知识库输入，不会据此编造个人经历。

### 轻声/短句识别与语义确认

桌面会议/麦克风采集使用随 faster-whisper 打包的 Silero v6 语音检测，每次会话独立保存检测状态，不上传音频。最短有效语音为 180 ms，保留 210 ms 起始音频。桌面在内存中保留最近 15 秒连续音频作为识别上下文，按词时间戳过滤已提交内容；重复说出的同一句话仍会作为新的字幕保留，问题回答的既有去重策略不变。CLI 默认保留原能量检测方式。轻声音频在进入 Whisper 前最多放大 8 倍。

说话时显示临时字幕，停顿 600 ms 后生成确定候选。以“的、和、一下”等结尾的未完成内容或低可信候选继续暂存，最多等待约 2 秒；接着说话就带原音频重新识别。等待期间复用候选，不反复识别同一段静音。超过等待上限的未完成字幕仍由问题缓冲保留，后续主题到达时再拼接，例如“讲解一下 React 的”不会直接触发回答。连续长句依然有切分上限，时间戳不能保证所有边界都完美，真实长会话需继续验证。

“运行设置 → 语音 → 术语场景”可选择“通用”（不预置热词）、“前端开发”（React Hooks、React Fiber、Vue、JavaScript、TypeScript、ES6、Promise）或自定义热词。切换预设会替换当前热词，保存配置后下次恢复；用户手动修改词表时自动标为自定义。旧版默认大词表会收敛为前端词表，自定义词表和空设置不改动。仅填写逗号分隔的短术语（最多 500 字），不填写答案或整句。实时字幕和预热使用 beam 1，不带热词；确定候选使用 beam 5 并带热词。热词可能造成偏置，不保证每次都识别正确，也不会把 Books 强制替换成 Hooks。

桌面仅对低可信或异常的最终候选进行一次无热词复识别，正常字幕和实时预览不额外解码。检测到循环后停止读取该轮后续片段；失败的复识别不污染原结果的时间戳。连续三个或更多相同词/短语、组合循环（如 Vue、JavaScript 反复出现）、异常连续标点，以及时间戳明显早于本次发言的结果会被隔离。复识别得到的异常替代文本保留原文并要求核对；仍异常时不显示为确定字幕，不发送给 Jev 或回答模型，并清空该段滚动音频。界面显示拦截原因，完整原文保留在“核对并编辑”入口供手动恢复；正常术语列表不因此被拦截。该保护是保守的异常检测，真实重复发言也可能需要手动确认，不能识别所有幻觉。

其余低可信问题仍可进行文字复核并给出修改建议，但文字模型没有听到原始音频，不能以“原文无需修正”覆盖音频的不确定性；无论文字复核或语义判断是否通过，都需用户核对/修改后明确提交才生成回答。可信且明确的问题仍自动回答。置信分数不是准确率保证；正常运行不把音频保存到磁盘。源码 `Transcriber(retry_low_confidence=True)` 与桌面使用相同的选择性复识别。

“设计一个……方案/系统”会额外等待可能接续的要求，最长约 2 秒；其他完整问题仍按 600 毫秒停顿尝试确认。麦克风状态显示实际输入设备、音量过低或削波提示。新增本地合成语料生成和回放脚本，包含中英技术词、长停顿、轻声、噪声和静音；前期结果及 Qwen3-ASR 实验见 [语音回归验收](docs/asr-validation-2026-10-10.md)。Qwen3-ASR-1.7B 后续已作为可选引擎接入应用，使用独立运行环境，未替换默认引擎，权重与 GPU 运行库不打入轻量包。

若连续两次可信实时字幕中的英文术语在最终字幕丢失，会额外识别本句音频进行核对，不从实时文本直接补词；没有可靠结果则提示人工核对。复核前原文保留在字幕中。普通完整且稳定的识别结果不增加这一步。

同一页可选择完整的本地 CTranslate2 模型目录，例如完整版 large-v3，用于手动对比；清空路径恢复默认 large-v3-turbo。不自动下载大模型，不能直接选原始 PyTorch 权重。准备相同的 10–20 道中文/中英混合问题（含数字、技术术语、短停顿），分别测试两种模型，记录错字/漏字、错误触发和耗时后再选择。当前没有真实会议录音基准，自动化测试通过不等于准确率已提升到某个百分比。

本地规则命中明确提问就走快速路径，例如“ES6的Promise有了解吗”。要让未命中的确定字幕交给 Jev 做语义补判，请在本机另行配置 TypeSafe/Jev 密钥（不是 DeepSeek 密钥）：

```powershell
$echoMindJevKey = Read-Host '请输入 TypeSafe API Key' -AsSecureString
$env:ECHOMIND_JEV_API_KEY = [System.Net.NetworkCredential]::new('', $echoMindJevKey).Password
Remove-Variable echoMindJevKey
.\scripts\run-captions.ps1 -Answers live -QuestionGate jev
```

仍需同一窗口内的 DeepSeek 配置。`-QuestionGate rules`（默认）不访问 TypeSafe。Jev 接口依据 [TypeSafe API](https://docs.typesafe.ai/api) 和 [Noul 判断](https://docs.typesafe.ai/primitives/noul)：只判断当前发言是否是完整、期待对方现在回答的真实问题/请求，而非转述、措辞讨论、自问自答或闲聊。工程初始门限为 0.85，并非已校准的准确率保证；可在 Python CLI 用 `--question-threshold` 调整。

Jev 模式会将每段本地规则未命中的确定字幕及最近最多 6 段确定字幕（每段最多 200 字）发送到 TypeSafe，包括普通发言、语气词和不含疑问词的请求；实时字幕修订和空字幕不发送。按顺序判断，后一句普通发言不会取消前一句的补判；明确的新规则问题或手动提交可取消过期补判。判断队列最多保留 100 段，满后可见报错而非静默丢弃。相比旧版候选筛选会增加 API 用量和延迟。

界面显示“语义判断”耗时；低概率输出“忽略”，补判失败/超时不猜测、不自动回答。Jev 接受的内容不再经过疑问词筛选，30 秒内的相同内容不重复回答。判断标准包含具体技术话题的“了解吗、熟悉吗、用过吗”，回答给出知识要点但不编造个人经历。现无说话人角色识别。自动化测试使用模拟概率，验证路由与接口，不证明真实 Jev 判断质量。

终端显示的“模型首字”从回答任务启动到首个回答文本增量计算，不包含静音等待、ASR 或完整问题结束到显示的延迟。真实腾讯会议、DeepSeek 网络链路和 0.5–1.2 秒目标仍需实测；目前自动化测试使用本地模拟接口，不代表云服务性能。

GitHub Copilot 也能通过 [官方 SDK](https://docs.github.com/en/copilot/how-tos/copilot-sdk/setup)嵌入应用，但需要其运行时和认证流程，本版尚未实现 Copilot 适配器。
