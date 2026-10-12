# 本地 ASR 对照：2026-10-10

结论：本次录音上 SenseVoiceSmall 和流式 Paraformer 更快，但中英技术词错误更多。因此保留 Whisper large-v3-turbo 为默认，仅新增两个可切换的实验引擎，不声称更换模型已解决准确率问题。

## 方法与限制

- 复用已获授权的 60 秒本机麦克风录音，没有重新录音、调用云端识别或上传音频。
- 重放经过生产字幕链路：16 kHz 单声道 PCM、100 ms 输入块、Silero v6 语音检测、现有字幕事件与重复幻觉拦截。
- Whisper 使用 GPU、int8_float16、前端热词和现有轻声增益；SenseVoice 和 Paraformer 使用 CPU 4 线程、int8。后两种当前接口没有接入热词偏置。这是各自可运行配置的比较，不是相同硬件/解码参数下的模型排名。
- Whisper/SenseVoice 保留 15 秒音频上下文并用时间戳过滤旧内容；Paraformer 自带流式缓存，无词时间戳，使用当前发言缓冲区。均在 600 ms 停顿时尝试确认，后续问题合并逻辑不变。
- 参考句按此次录制目标句对齐，尚非人工逐字标注的正式评测集。仅 4 个请求，不可推广为一般准确率；大小写、标点和空格不计入字符错误。
- 追加 20 秒数字静音、20 秒低强度高斯噪声（RMS 40/32768，随机种子 17）、20 秒低强度 50 Hz 嗡声（幅度 100/32768）。只覆盖这三种合成背景，不能保证其他环境不会出现幻觉。
- 运行时详细事件报告保存在忽略目录 `.runtime/asr-recordings/engine-comparison.json`，不纳入 Git。

## 实测结果

| 方案 | 运行设备 | 120 秒音频处理耗时（不含预热） | 参考字符错误 / 64 | 技术词命中 / 8 | 追加背景字幕事件 |
|---|---|---:|---:|---:|---:|
| Whisper large-v3-turbo | GPU | 9.26 秒 | 0 | 8 | 0 |
| SenseVoiceSmall int8 | CPU / 4 线程 | 6.41 秒 | 6 | 5 | 0 |
| 流式 Paraformer bilingual int8 | CPU / 4 线程 | 1.12 秒 | 25 | 1 | 0 |

处理耗时是加速重放的总计算时间，不是用户说完到看到字幕/答案的延迟。预热分别为 6.68、1.20、0.99 秒。最终确认调用的中位耗时分别约 350、212、23 ms；Paraformer 的 23 ms 仅为已做增量识别后的剩余确认工作，不能当作完整发言识别耗时。此实验不测 LLM 首字延迟。

| 参考请求 | Whisper | SenseVoice | Paraformer |
|---|---|---|---|
| 讲解一下 React 的 Hooks（第一次） | 讲解一下 React 的 Hooks | 讲解一下 react books | 讲解一下 react 的 fo |
| 讲解一下 React 的 Hooks（第二次） | 讲解一下 React 的 Hooks | 讲解一下 react books | 讲解一下这类 act 看个 x |
| 再说一下 React 和 Vue 的区别 | 再说一下 React 和 Vue 的区别 | 再说一下 react 和 vi 的区别 | 再说一下喽女爱上课所有的区别 |
| 然后再讲解一下〔停顿〕ES6 的 Promise | 然后再讲解一下 ES6 的 Promise | 然后再讲解一下。ES6 的 promise | 然后再讲解一下 e s 六 promi |

另将四段完整发言直接送入两个候选，不经过 VAD 分句或旧词过滤：SenseVoice 仍把 Hooks 识别为 Books/Mix、Vue 识别为 Will；Paraformer 也仍有类似技术词错误。因此当前差异不能只归因于接入切分。没有加入 Books → Hooks 等无证据替换。

## 使用与复现

窗口中停止采集，进入“运行设置 → 语音 → 识别引擎”，切换实验选项；用相同麦克风、相同句子对比。两个实验引擎固定使用 CPU，不需要下载 CUDA。选择引擎会清空旧的自定义模型路径，避免拿 Whisper 模型加载 ONNX；保存配置后下次恢复。文本/截图回答不依赖语音模型。

模型查找顺序：程序旁 `models/sensevoice` 或 `models/paraformer` → 用户组件缓存对应目录 → 附近项目 `.models/`。本机已准备模型，无需再下载；如果程序移出项目且没有缓存，请复制相应模型文件夹至程序旁的 `models`，或在“本地识别模型目录”指定路径。实验模型不打入轻量主程序，不会触发 Whisper 的首次下载。

开发准备与本次录音重放：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".\services\transcription[dev,desktop-build,asr-onnx]"
.\.venv\Scripts\python.exe .\scripts\prepare-asr-comparison.py
.\.venv\Scripts\python.exe .\scripts\compare-asr.py .\.runtime\asr-recordings\microphone-20261009-000349.wav --output .\.runtime\asr-recordings\engine-comparison.json
```

准备脚本只下载并校验两个量化模型与必要文件，合计约 455 MiB，不下载原始全精度副本；下载失败保留分片，已有匹配文件复用。对其他录音必须先修改并核对比较脚本的参考句与时间范围，否则错误率无意义。

官方模型/API 来源：[SenseVoice](https://k2-fsa.github.io/sherpa/onnx/sense-voice/pretrained.html)、[流式 Paraformer](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-paraformer/paraformer-models.html)、[Windows Python 安装](https://k2-fsa.github.io/sherpa/onnx/python/install.html)。模型、运行库的使用和分发应遵守各自许可证。

下一步若继续追求中文技术词准确率，需要更多真实口音、停顿与噪声样本，再评测其他中文/中英模型；本次结果不支持直接替换默认识别器。
