# Qwen3-ASR-1.7B 本地部署与测试

日期：2026-10-12。下文先记录独立部署测试；随后已按用户要求接入应用并构建窗口版，见文末“应用接入”。默认引擎未被强制替换，没有发布或上传录音。

## 部署

- 模型：官方 `Qwen/Qwen3-ASR-1.7B-hf`，Transformers 原生适配版，Apache-2.0。
- 修订：`bcd2b5b7f32b480ab5790554cfa8347f246a14f3`。
- 位置：`D:\Code\EchoMind\.models\qwen3-asr-1.7b`。
- 权重：4,076,193,080 字节；SHA-256 `2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1`。下载完成后校验通过；配置及分词器同样按仓库摘要校验。
- 独立环境：`D:\Code\EchoMind\.runtime\qwen17-eval`，没有修改应用的 `.venv` 或原 0.6B 测试环境。
- Python 3.11.9、PyTorch 2.8.0+cu128、Transformers 5.19.0、Accelerate 1.12.0、NumPy 2.4.6、librosa 0.11.0、soundfile 0.14.0。
- GPU：RTX 4060 Laptop，8 GB 显存。CUDA 和 BF16 检查通过，`pip check` 通过。
- 推理：BF16、单条音频、固定中文、贪心解码、最多 160 个输出 token；本地离线加载。

下载期间曾发生连接中断，未使用校验失败的安装包。最终 CUDA PyTorch wheel 从官方地址分段下载，SHA-256 `34c55443aafd31046a7963b63d30bc3b628ee4a704f826796c865fdfd05bb596` 校验通过后安装。模型通过 Hugging Face 镜像取得，与上述固定修订及摘要核对。

## 真人录音：四段全部匹配

复用既有录音的四个预切片段，没有重新打开麦克风。关闭热词，忽略大小写、空格与标点后，64 个参考字符错误 0、技术词 8/8。

| 音频内容 | 音频长度 | 处理耗时 | 结果 |
|---|---:|---:|---|
| 讲解一下 React 的 Hooks | 5 秒 | 1.413 秒，含首次推理预热 | 匹配 |
| 讲解一下 React 的 Hooks | 7 秒 | 0.300 秒 | 匹配 |
| 再说一下 React 和 Vue 的区别 | 8 秒 | 0.380 秒 | 匹配 |
| 然后再讲解一下 ES6 的 Promise | 12 秒 | 0.393 秒 | 匹配 |

模型加载 5.35 秒；PyTorch 峰值已分配显存 3,989.8 MiB（约 3.90 GiB），峰值预留 4,188 MiB（约 4.09 GiB）。这些不是整个系统的显存占用。

**处理耗时从完整音频交给模型开始计时，不含说话时长、停顿等待、问题判断和回答生成，不是实时字幕的端到端延迟。** 四个样本不足以代表真人会议整体准确率。

结果：`.runtime/asr-recordings/qwen17-gpu-no-hotwords.json`。

## 合成语料与背景测试

复用冻结的 Windows TTS 语料：14 条发言，另有静音、噪声、嗡声各 10 秒。参考文本在识别前已存在，没有根据输出修改参考。完整语料 SHA-256：`fbdaab4c71ab5c3844e227cf4561977664feda92038e2bb410d095f67378bdc2`。本次按清单中的独立 WAV 测试，不经过流式分段或 VAD。

另外将本地已有 0.6B 放到同一 GPU 环境、使用同一脚本测试。均为单轮，不是稳定性能基准。

| 指标 | 1.7B 无热词 | 1.7B 有热词 | 0.6B 无热词 |
|---|---:|---:|---:|
| 发言严格归一化匹配 | 9/14 | 13/14 | 11/14 |
| 字符编辑数 / 245 | 16 | 10 | 14 |
| 技术词字面命中 | 15/19 | 19/19 | 17/19 |
| 发言处理耗时中位数，排除首条 | 389 ms | 343 ms | 383 ms |
| 峰值已分配显存 | 3,975 MiB | 3,975 MiB | 1,583 MiB |
| 三段无声背景出现文字 | 3/3 | 3/3 | 3/3 |

评分忽略大小写、空格、标点，不转换中英文数字；技术词统计要求字面命中。因此这些数字不能直接等同于语义准确率。

- 1.7B 无热词出现 `Vue → Vu`，以及 `ES6 → E.S. Six / E S 六`。
- 有热词时，唯一未严格匹配的发言是 `123乘以456` 写成“一百二十三乘以四百五十六”，语义相同；10 个字符编辑均来自该数字写法差异。
- **无热词时，三段无声背景均输出“嗯。”。**
- **有热词时，三段无声背景均输出“React Hooks, React Fiber, Vue, JavaScript, TypeScript, ES6, Promise。”，即复述提示词。**
- 本轮 1.7B 无热词并未比 0.6B 无热词更准确，不能据此宣称参数更多就更适合默认替换。

原始结果位于 `.runtime/asr-corpus-20261010-v2/`：

- `qwen17-gpu-no-hotwords.json`
- `qwen17-gpu-hotwords.json`
- `qwen06-gpu-no-hotwords.json`

旧 Whisper 报告经过 VAD、分段和异常过滤，本次 Qwen 是整段裸模型测试，不能拿两者的背景输出或耗时直接做公平排名。

## 结论与后续接入边界

1. 1.7B 已能在本机 8 GB GPU 上运行，短句预切片推理速度可用。
2. 换成 1.7B **不能消除幻觉**。热词提高术语命中，同时明显增加无语音时的提示词复述风险。
3. 若接入应用，应先经过语音活动检测，只在真实语音片段上识别；可疑输出进入待核对区，不能直接进入字幕、Jev 或回答链路。不要仅按“嗯”等字面黑名单过滤正常发言。
4. 目前只完成独立部署与离线样本测试，未接入 GUI、实时采集、窗口字幕或轻量组件下载器，未做长时间会议、重叠说话或口音测试。

## 本机复测

已安装环境和模型可直接复用，不必重新下载。在项目目录执行：

```powershell
.\.runtime\qwen17-eval\Scripts\python.exe -X utf8 scripts/evaluate-qwen-asr.py .runtime/asr-recordings/qwen-clips/manifest.json --model .models/qwen3-asr-1.7b --device cuda --no-hotwords --output .runtime/asr-recordings/qwen17-human-replay.json

.\.runtime\qwen17-eval\Scripts\python.exe -X utf8 scripts/evaluate-qwen-asr.py .runtime/asr-corpus-20261010-v2/manifest.json --model .models/qwen3-asr-1.7b --device cuda --no-hotwords --output .runtime/asr-corpus-20261010-v2/qwen17-replay.json
```

去掉 `--no-hotwords` 即使用测试脚本内的固定前端术语词表。它不是生产环境默认配置；无声样本的文字输出是故意保留的测试证据，不会写进 EchoMind 字幕。

下载入口已支持 `scripts/prepare-asr-comparison.py --models qwen3-asr-1.7b`；评估脚本报告实际模型名、音频长度及显存占用。下载和评估脚本语法检查通过。

官方资料：[模型卡](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)、[Qwen3-ASR 项目](https://github.com/QwenLM/Qwen3-ASR)。

## 应用接入（同日后续）

已构建：`D:\Code\EchoMind\dist\light-v0.1.3-qwen17\EchoMind\EchoMind.exe`。

使用步骤：打开该新版程序，在“语音 → 识别引擎”选择“Qwen3-ASR-1.7B（本地 GPU）”，模型目录留空，然后开始麦克风测试或会议采集。点击保存配置可保留选择。旧 EXE 不会自动更新。

- 复用项目 `.models/qwen3-asr-1.7b` 与 `.runtime/qwen17-eval`，不重复下载。
- Qwen 在独立隐藏进程中运行；不把 PyTorch 或权重塞进轻量包。停止、取消加载、异常退出均关闭工作进程以释放显存。
- 语音活动检测先过滤无声片段；通常停顿 600 ms 开始确认，未完成的提问最多再等到约 2 秒，保留停顿前后的完整问题。
- 不传热词、不携带上一次发言进入模型，不做逐字预览。循环重复与达到输出长度上限的结果走 `rejected` 通道，不进入字幕正文或自动问答。
- 模型加载在后台进行，复用界面的加载、停止及错误提示。此交互按 UI 技能的加载反馈建议保持原布局和可取消状态。
- 这是本机部署的接入版。独立 Python 环境不是通用便携包，移动到其他电脑需要重新部署；尚未增加 Qwen 的首次启动自动下载安装。

### 接入后的验证

- 全量自动化：646 项通过，另有 10 个子测试通过；包含配置保存、GUI 选项、引擎路由、静音不解码、异常输出隔离、未完成提问续接及进程释放。
- 140.3 秒合成语料按真实速度回放：14 条发言均输出最终字幕，技术词 19/19，字符编辑 10/245（全部来自阿拉伯数字转中文数字）；30 秒背景字幕事件 0。带停顿的请求和设计题均合为单条最终字幕。
- 发言结束至最终字幕事件中位 1.22 秒，范围 0.97–1.73 秒；最终解码中位 596 ms。此次包含实时等待、IPC 与 VAD，不包含 UI 绘制、Jev 或回答生成。它不能保证真实会议的同等准确率。
- 真人完整录音加 60 秒背景的回放：背景字幕事件 0；仍出现 `Vue → View`，以及 `ES6 → E.S.六` 的文字差异，严格字符错误 3/64。整段独立测试的 4/4 并不等于分段采集的效果，不隐藏此差异。
- 使用打包目录内的 worker，并模拟冻结程序路径与 Windows DLL 搜索目录完成加载、识别、关闭检查，返回“讲解一下 React 的 Hooks。”。未重新打开麦克风或发起真实会议；GUI 控件由隐藏窗口自动化测试验证。

接入后结果：`.runtime/asr-corpus-20261010-v2/qwen17-integrated-realtime.json` 和 `.runtime/asr-recordings/qwen17-integrated.json`。

应用分段链路复测：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate-asr-corpus.py .runtime/asr-corpus-20261010-v2/manifest.json --engine qwen3-asr-1.7b --no-hotwords --realtime --output .runtime/asr-corpus-20261010-v2/qwen17-integrated-replay.json
```
