# 语音回归验收：2026-10-10

## 本次范围

自行生成音频、回放生产语音链路、验证已有真人录音，并在独立环境评测 Qwen3-ASR。全程本地推理，没有上传音频、开启麦克风或调用 Jev/回答 API。

交付保留 Whisper large-v3-turbo 默认引擎，不把“单条合成语音识别对了”当成可替换默认模型的证据。

## 已实现

- 保留 15 秒声学上下文，但用本次语音开始时间和已提交时间共同过滤旧词，避免前一句词尾变成下一句的开头。允许约 800 ms 对齐漂移，以词尾是否越过边界判断，避免截掉开头的“讲”“再”。
- “设计/实现/制定……方案或系统”在短停顿后先作为候选等待，最长约 2 秒，以接住后面的要求；普通完整句仍在 600 ms 静音后尝试确认。
- 仅最终低可信或异常候选额外解码一次，关闭热词偏置；正常最终字幕与实时预览不重复解码。失败的重试不会污染第一遍的时间戳、置信标记或异常原因。
- 若连续两次可信实时字幕均出现的英文术语在最终字幕消失，再单独识别本句的原始音频，不拼接旧字幕。仅接受无异常、包含稳定术语且整体文本相近的复识别结果；否则保留最终原文并要求核对。该复核与前一类复核不叠加执行。
- 对循环异常的替代结果保留原文、标为待核对，不自动当成可信问题回答。普通低可信结果仅在内容相近、分数改善且无异常时接受复识别。分数不是正确率保证。
- 检测到循环后停止继续读取该轮后续片段，避免在已确定异常时继续生成。
- 前导缓存 480 ms；麦克风状态显示实际输入设备、无声、音量偏低或削波提示，不修改用户音频设备设置。

不采用的实验：每次发言完全清空先前音频。它改善部分合成样本，却使真人录音的 Hooks 变成了其他词，因此撤回该改动。未加入 Books → Hooks、Readies → Redis 等硬编码替换。

也试过收紧模型生成长度，但未改善误差，交付版未保留该参数调整，仅保留检测异常后停止后续片段的改动。

## 合成语料及评分口径

固定语料位于 `.runtime/asr-corpus-20261010-v2/`，`corpus.wav` 时长 140.297375 秒、16 kHz 单声道 16-bit PCM。

SHA-256：`fbdaab4c71ab5c3844e227cf4561977664feda92038e2bb410d095f67378bdc2`。

参考文本先写入 `scripts/asr-corpus.json`，再用 Windows 本地 Huihui/Zira 声音生成；没有用识别输出反推参考答案。共 17 个场景：14 段发言，以及各 10 秒的静音、低强度随机噪声、50 Hz 嗡声。发言包含 React Hooks/Fiber、ES6 Promise、Vue、Redis、TCP、算术、普通陈述和感谢，还包含 1.1–1.4 秒停顿、约 -22 dB 轻声和 15 dB 信噪比背景。

第一版合成器切换声音时带入了额外停顿，不作为正式比较。第二版分别合成片段并裁去合成器首尾空白，只保留指定停顿；冻结音频后所有 Whisper 调参均使用同一 SHA-256。语料是开发回归集，不是独立盲测集，更不代表真人会议准确率。

字符评分忽略大小写、空格和标点，中英字符均逐字符计算；技术词按归一化后的文本命中统计。算术参考使用数字形式，中文数字的同义写法仍会在严格字符评分中计错，不能直接据此判定模型听错。脚本内 `questions` 只是本地规则候选，不是完整 Jev/LLM 链路验收。

## 同一合成音频的前后结果

| 指标 | 本轮改动前 | 交付代码 |
|---|---:|---:|
| 字符错误 / 245 | 22（8.98%） | 4（1.63%） |
| 技术词命中 / 19 | 18 | 18 |
| 30 秒背景的字幕事件 | 0 | 0 |
| 带停顿的设计题最终片段数 | 2 | 1 |

交付代码的 14 段发言均产生最终字幕，其中 Redis 一段保守提示核对，不自动回答；普通陈述和感谢未产生本地规则问题候选。剩余错误是 Redis → Readies（2 个字符编辑）、React/Vue 停顿句漏“和”（1）、“放量”→“放亮”（1）。没有宣称准确率达到 100%，也没有用文字替换隐藏这些错误。

真实时间回放以每 100 ms 到达一块音频模拟输入：音频片段结束到最终字幕事件的中位延迟 **1.19 秒**，范围 **0.97–2.01 秒**；最终解码调用中位 **490 ms**，含实际发生的选择性复核。TCP 额外复核后约 1.85 秒，Redis 待核对约 2.01 秒。相较不做复核，少数可疑句会更慢，这是准确性与延迟的明确取舍；这些数字不包含 Jev、回答模型或窗口绘制，不能当作“首批回答已达 0.5–1.2 秒”。

详细记录：改动前 `.runtime/asr-corpus-20261010-v2/whisper-before.json`；交付代码 `.runtime/asr-corpus-20261010-v2/whisper-shipping-realtime.json`。原版是加速回放，交付版为真实时间回放，二者总运行时不能直接作速度比较。

## 真人录音回归

复用原有已授权的 60 秒录音 `microphone-20261009-000349.wav`，追加 60 秒静音/背景。四个目标请求分别为两次 React Hooks、React 与 Vue 的区别、停顿后的 ES6 Promise。

交付代码回放结果：四句归一化参考文本全部匹配，64 个参考字符错误 0，技术词 8/8；追加背景字幕事件 0。最后一次 120 秒音频加速回放用时 11.89 秒，最终解码中位 432 ms；该次同时在运行自动化测试，不据此判断速度提升。详细结果：`.runtime/asr-recordings/whisper-release-final.json`。

原版在这份真人录音上也为 0 错误、8/8，因此本次是保持该样本的效果，不声称真人准确率再次提高。这里只有四个目标请求，并非正式人工逐字标注的大规模测试。

## Qwen3-ASR 实验：尚未替换默认

使用官方 `Qwen/Qwen3-ASR-0.6B-hf`，模型修订 `7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c`。环境为独立的 `.runtime/qwen-eval`，PyTorch 2.8.0 CPU、Transformers 5.19.0，CPU 4 线程、float32。GPU 运行库下载未完成，随后改用 CPU 验证；本次没有 Qwen GPU 延迟或显存实测。

输入为完整、事先切好的音频段，不经过 VAD/流式分段，不能与上面的 Whisper 生产链路做公平性能排名。

- 带相同前端热词的 14 段合成发言中，12 段严格归一化文本匹配；另两段仅为 `ES六`/`ES6` 和中文数字/阿拉伯数字写法差异。Redis、React、Hooks、Fiber 均读出。
- 去掉热词后，出现一处 `Vue` → `Vu`；说明上下文偏置有收益，也有风险。
- **直接送入无声/噪声/嗡声时，带热词版本三段都复述词表；无热词版本三段都输出“嗯”。** 所以换模型不能替代语音检测与异常拦截。
- 真人四个预切片段均匹配参考，CPU 每段约 2.23、2.30、2.62、3.23 秒；这包含片段中的空白，不等于在线首字延迟。首段还含首次推理开销。

结果：`.runtime/asr-corpus-20261010-v2/qwen-cpu.json`、`qwen-cpu-no-hotwords.json`，及 `.runtime/asr-recordings/qwen-human.json`。

Qwen 是值得继续验证的最终句复核候选，但目前没有证据证明它在本机实时 GPU 链路上既更准又更快。因此没有将 PyTorch 或 Qwen 权重放入轻量包，也没有增加默认运行时下载。后续接入必须先通过 GPU、相同流式切分和长时间背景测试，不能仅凭这次完整片段结果自动升级默认引擎。

官方接口与模型依据：[Qwen 模型卡](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf)、[Qwen3-ASR 项目](https://github.com/QwenLM/Qwen3-ASR)。

## 复现

用新目录生成，脚本拒绝覆盖已有标注语料：

```powershell
.\.venv\Scripts\python.exe scripts/generate-asr-corpus.py --output .runtime/asr-corpus-replay
.\.venv\Scripts\python.exe scripts/evaluate-asr-corpus.py .runtime/asr-corpus-replay/manifest.json --retry --realtime --output .runtime/asr-corpus-replay/whisper.json
.\.venv\Scripts\python.exe scripts/compare-asr.py .runtime/asr-recordings/microphone-20261009-000349.wav --engine whisper --output .runtime/asr-recordings/whisper-replay.json
```

Qwen 仅为开发者独立实验（本机模型和环境已准备好，不必再安装）：

```powershell
.\.venv\Scripts\python.exe scripts/prepare-asr-comparison.py --models qwen3-asr-0.6b
.\.venv\Scripts\python.exe -m venv .runtime/qwen-eval
.\.runtime\qwen-eval\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.runtime\qwen-eval\Scripts\python.exe -m pip install transformers==5.19.0 accelerate==1.12.0 librosa soundfile
.\.runtime\qwen-eval\Scripts\python.exe scripts/evaluate-qwen-asr.py .runtime/asr-corpus-20261010-v2/manifest.json --model .models/qwen3-asr-0.6b --device cpu --output .runtime/asr-corpus-20261010-v2/qwen-replay.json
```

测试语料、录音和模型留在 Git 忽略目录。正常运行不保存录音；合成回归脚本才会写入测试 WAV。没有推送 GitHub、发布 Release 或更改用户密钥。

自动化验证：全量 `pytest -q`，555 项通过。新增回归覆盖暂停后续要求、旧词尾过滤、时间戳漂移、异常复识别状态隔离、稳定术语丢失复核、正常拼写修订不误触发、静音/低音量/削波诊断。

已构建窗口版：`dist/light-v0.1.3-asr-validated/EchoMind/EchoMind.exe`。保留整个相邻文件夹运行，不要单独复制 EXE。本机继续复用已有模型与 GPU 组件；源码 ASR/窗口自动化测试通过，未重新录制真人麦克风或在腾讯会议中进行本次人工验收。
