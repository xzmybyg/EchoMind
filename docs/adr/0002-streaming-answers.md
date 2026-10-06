# ADR 0002: Optional streaming answers for MVP 2

Status: accepted for the terminal MVP.

## Decision

Keep the existing capture/ASR path unchanged by default. Only explicit `--answers live` enables network requests. Use one OpenAI-compatible Chat Completions provider and an offline, clearly labeled demo provider. The user proposed DeepSeek or GitHub Copilot; document DeepSeek configuration first because it fits this small streaming interface without an agent runtime. Copilot remains a separate possible adapter, not a supported API alias.

Final Chinese captions feed a bounded heuristic question buffer. Partial captions do not trigger requests. Generate answers in a background asyncio thread; cancel superseded questions, suppress stale output, and close streams on cancellation or shutdown. Keep only the latest four successful question/answer turns in memory. No RAG is added in this milestone.

Following meeting tests, add optional `--question-gate jev` semantic confirmation using the existing HTTP client. It requires a separate TypeSafe key and explicit live mode. Send the candidate and at most six recent final transcript pieces; accept only a Noul probability at or above the initial 0.85 policy threshold. This is not a calibrated quality guarantee. Judgment errors fail closed; a rejected candidate must not cancel the current answer. Report judgment time separately from model time. The local rules mode remains the default and filters observed meta-discussion false positives without network calls.

## Privacy and measurement

Read model credentials exclusively from environment variables. Do not log credentials, response bodies, or remote exception strings. Live mode sends questions and recent answer context, not audio. Require informed participant consent before using meeting content with a remote provider.

Report request-to-first-answer-text and total generation time separately from ASR receive-to-caption time. Neither metric alone proves the full question-end-to-answer target. Tencent Meeting and live cloud latency remain manual acceptance checks.

## Verification

Local tests cover Chinese question fragments, duplicates and reported speech; SSE UTF-8 chunking, role-only frames, errors, truncation and resource closure; nonblocking submission, cancellation and bounded context; and CLI integration without capture or GPU. No live provider call is made without local credentials and explicit live mode.

References: [Chat Completions streaming](https://developers.openai.com/api/docs/guides/streaming-responses), [DeepSeek API](https://api-docs.deepseek.com/), [DeepSeek thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/), [Copilot SDK](https://docs.github.com/en/copilot/how-tos/copilot-sdk/setup).
