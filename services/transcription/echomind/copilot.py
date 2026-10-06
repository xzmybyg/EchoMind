"""Run answer generation outside the synchronous audio/transcription loop."""

import asyncio
import threading
import time
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from echomind.llm import AnswerProvider, AnswerProviderError
from echomind.question import PausedQuestion, QuestionBuffer, _classify, is_filler
from echomind.intent import QuestionGate, QuestionGateError
from echomind.text import normalize_transcript
from echomind.repair import local_review, normalize_question, suspect_question


@dataclass(frozen=True)
class AnswerEvent:
    kind: str
    question_id: int
    question: str
    text: str = ""
    first_token_ms: float | None = None
    elapsed_ms: float | None = None
    decision_ms: float | None = None
    reason: str = ""


class Copilot:
    def __init__(
        self,
        provider: AnswerProvider,
        on_event: Callable[[AnswerEvent], None],
        *,
        question_buffer: QuestionBuffer | None = None,
        max_context_turns: int = 4,
        question_gate: QuestionGate | None = None,
        question_threshold: float = 0.85,
        question_reviser=None,
    ) -> None:
        if max_context_turns < 0:
            raise ValueError("max_context_turns must be nonnegative")
        if not math.isfinite(question_threshold) or not 0 <= question_threshold <= 1:
            raise ValueError("question_threshold must be between zero and one")
        self._provider = provider
        self._on_event = on_event
        self._buffer = question_buffer if question_buffer is not None else QuestionBuffer()
        self._max_context_turns = max_context_turns
        self._question_gate = question_gate
        self._question_threshold = question_threshold
        self._transcripts: deque[str] = deque(maxlen=6)
        self._candidate_id = 0
        self._gate_task: asyncio.Task | None = None
        self._review_task: asyncio.Task | None = None
        self._question_reviser = question_reviser
        self._reviewed: deque[tuple[str, float]] = deque(maxlen=16)
        self._semantic_queue: deque = deque()
        self._semantic_answered: deque[tuple[str, float]] = deque(maxlen=16)
        self._paused_question = PausedQuestion()
        self._context: list[dict[str, str]] = []
        self._loop = asyncio.new_event_loop()
        self._lock = threading.Lock()
        self._pending = 0
        self._closed = False
        self._idle = threading.Event()
        self._idle.set()
        self._ready = threading.Event()
        self._question_id = 0
        self._active: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        self._thread = threading.Thread(target=self._run, name="echomind-answers", daemon=True)
        self._thread.start()
        self._ready.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            try:
                # Async-generator finalizers may outlive the answer task itself.
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            finally:
                self._loop.close()

    def submit_transcript(self, text: str, timestamp_ms: int, final: bool = True, suspect: bool = False) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Copilot is closed")
            self._pending += 1
            self._idle.clear()
            self._loop.call_soon_threadsafe(self._submit, text, timestamp_ms, final, suspect)

    def submit_question(self, text: str) -> None:
        """Explicit user intent does not need automatic question detection."""
        question = normalize_transcript(text)
        if not question:
            raise ValueError("请填写问题内容")
        with self._lock:
            if self._closed:
                raise RuntimeError("Copilot is closed")
            self._pending += 1
            self._idle.clear()
            self._loop.call_soon_threadsafe(self._submit_question, question)

    def _submit_question(self, question: str) -> None:
        try:
            if self._closed:
                return
            # An older pending judgment must not override a manual correction.
            self._paused_question.clear()
            clear_pending = getattr(self._buffer, "clear_pending", None)
            if clear_pending is not None:
                clear_pending()
            self._invalidate_candidate()
            self._start_answer(question)
        finally:
            with self._lock:
                self._pending -= 1
                self._update_idle()

    def _invalidate_candidate(self) -> None:
        self._candidate_id += 1
        self._semantic_queue.clear()
        for task in (self._gate_task, self._review_task):
            if task is not None:
                task.cancel()

    def _submit(self, text: str, timestamp_ms: int, final: bool, suspect: bool = False) -> None:
        try:
            if self._closed:
                return
            recent = list(self._transcripts)
            if final and text.strip():
                self._transcripts.append(text.strip()[-200:])
            normalized = normalize_transcript(text)
            joined = False
            if final and normalized:
                if suspect_question(normalized):
                    self._paused_question.clear()
                normalized, suspect, joined = self._paused_question.push(normalized, timestamp_ms, suspect)
                if joined:
                    clear_pending = getattr(self._buffer, "clear_pending", None)
                    if clear_pending is not None:
                        clear_pending()
            # A malformed task opening plus newly captured constraints needs review,
            # even when a literal question marker was lost by ASR.
            joined_suspect = joined and suspect_question(normalized)
            if self._question_gate is not None and _classify(normalized) != "question" and not joined_suspect:
                if final and normalized:
                    if len(self._semantic_queue) >= 100:
                        self._emit(AnswerEvent("gate_error", 0, normalized, "语义判断积压过多，请停止后重试"))
                        return
                    self._semantic_queue.append((normalized, timestamp_ms, recent, suspect))
                    if self._gate_task is None or self._gate_task.done():
                        task = self._loop.create_task(self._drain_judgments(self._candidate_id))
                        self._gate_task = task
                        self._track(task)
                return
            if final and (suspect_question(normalized) or (suspect and _classify(normalized) == "question")):
                now = time.monotonic()
                while self._reviewed and now - self._reviewed[0][1] > 30:
                    self._reviewed.popleft()
                key = normalized.casefold().strip("，,。.!！?？;； ")
                if any(previous == key for previous, _ in self._reviewed):
                    return
                self._reviewed.append((key, now))
                clear_pending = getattr(self._buffer, "clear_pending", None)
                if clear_pending is not None:
                    clear_pending()
                self._invalidate_candidate()
                task = self._loop.create_task(self._review(self._candidate_id, normalized, timestamp_ms, recent, suspect))
                self._review_task = task
                self._track(task)
                return
            if is_filler(normalized):
                return
            question = self._buffer.push(normalized, timestamp_ms, final=final)
            if question is None:
                if final and text.strip() and self._review_task is None:
                    self._emit(AnswerEvent("untriggered", 0, text,
                                           getattr(self._buffer, "decision", "等待完整提问")))
                return
            self._invalidate_candidate()
            self._accept_question(question.text, recent)
        finally:
            with self._lock:
                self._pending -= 1
                self._update_idle()

    def _accept_question(self, text: str, recent: list[str]) -> None:
        self._start_answer(text)

    async def _drain_judgments(self, candidate_id: int) -> None:
        while self._semantic_queue and not self._closed and candidate_id == self._candidate_id:
            text, timestamp, recent, suspect = self._semantic_queue.popleft()
            await self._judge(candidate_id, text, recent, timestamp, suspect)

    async def _review(self, candidate_id: int, text: str, timestamp_ms: int, recent: list[str], suspect: bool = False,
                      *, already_judged: bool = False, decision_ms: float | None = None) -> None:
        try:
            if self._closed or candidate_id != self._candidate_id:
                return
            self._emit(AnswerEvent("reviewing", candidate_id, text))
            result = local_review(text) if self._question_reviser is None else await asyncio.wait_for(
                self._question_reviser.review(text, recent), timeout=8,
            )
            if self._closed or candidate_id != self._candidate_id:
                return
            local_uncertainty = suspect and self._question_reviser is None
            if result.needs_confirmation or local_uncertainty or normalize_question(result.corrected) != normalize_question(text):
                reason = "语音识别置信度偏低，请核对原话" if local_uncertainty and not result.needs_confirmation else result.reason
                self._emit(AnswerEvent("needs_confirmation", candidate_id, text, result.corrected, reason=reason))
                return
            if already_judged:
                self._start_answer(result.corrected, decision_ms)
                return
            # Reviewing one uncertain piece must not append a prior incomplete request.
            buffer = QuestionBuffer(semantic_candidates=self._question_gate is not None)
            question = buffer.push(result.corrected, timestamp_ms)
            if question is None:
                self._emit(AnswerEvent("untriggered", candidate_id, text, buffer.decision))
            else:
                self._accept_question(question.text, recent)
        except asyncio.CancelledError:
            raise
        except Exception:
            if not self._closed and candidate_id == self._candidate_id:
                self._emit(AnswerEvent("repair_error", candidate_id, text, "问题复核失败或超时，请编辑原文后手动提交"))

    def _track(self, task: asyncio.Task) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._finished)

    def _start_answer(self, question: str, decision_ms: float | None = None) -> None:
        self._question_id += 1
        if self._active is not None:
            self._active.cancel()
        task = self._loop.create_task(self._answer(self._question_id, question, decision_ms))
        self._active = task
        self._track(task)

    async def _judge(self, candidate_id: int, question: str, recent: list[str], timestamp_ms: int = 0, suspect: bool = False) -> None:
        started = time.perf_counter()
        try:
            if self._closed or candidate_id != self._candidate_id:
                return
            self._emit(AnswerEvent("checking", candidate_id, question))
            probability = await asyncio.wait_for(self._question_gate.evaluate(question, recent), timeout=3)
            if self._closed or candidate_id != self._candidate_id:
                return
            if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise QuestionGateError("问题判断返回无效概率")
            elapsed = (time.perf_counter() - started) * 1000
            if probability >= self._question_threshold:
                clear_pending = getattr(self._buffer, "clear_pending", None)
                if clear_pending is not None:
                    clear_pending()
                now = time.monotonic()
                key = normalize_question(question).casefold().strip("，,。.!！?？;； ")
                while self._semantic_answered and now - self._semantic_answered[0][1] > 30:
                    self._semantic_answered.popleft()
                if any(previous == key for previous, _ in self._semantic_answered):
                    self._emit(AnswerEvent("skipped", candidate_id, question, "30 秒内的重复提问，已判断但不重复回答"))
                    return
                self._semantic_answered.append((key, now))
                if suspect or suspect_question(question):
                    await self._review(candidate_id, question, timestamp_ms, recent, suspect,
                                       already_judged=True, decision_ms=elapsed)
                else:
                    self._start_answer(question, elapsed)
            else:
                self._emit(AnswerEvent("skipped", candidate_id, question,
                                      f"已判断“{question[:120]}”：不是明确等待回答的提问（概率 {probability:.2f}）", decision_ms=elapsed))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closed and candidate_id == self._candidate_id:
                message = str(exc) if isinstance(exc, QuestionGateError) else "问题判断失败或超时，未触发回答"
                self._emit(AnswerEvent("gate_error", candidate_id, question, message))

    def _update_idle(self) -> None:
        if self._pending == 0 and not self._tasks:
            self._idle.set()

    def _finished(self, task: asyncio.Task) -> None:
        # Consume failures so asyncio never logs remote exception strings.
        if not task.cancelled():
            task.exception()
        self._tasks.discard(task)
        if task is self._active:
            self._active = None
        if task is self._gate_task:
            self._gate_task = None
        if task is self._review_task:
            self._review_task = None
        with self._lock:
            self._update_idle()

    def _emit(self, event: AnswerEvent) -> None:
        self._on_event(event)

    async def _answer(self, question_id: int, question: str, decision_ms: float | None = None) -> None:
        started = time.perf_counter()
        first_token_ms = None
        pieces: list[str] = []
        stream = None
        try:
            if self._closed or question_id != self._question_id:
                return
            self._emit(AnswerEvent("start", question_id, question, decision_ms=decision_ms))
            stream = self._provider.stream_answer(question, [dict(m) for m in self._context])
            async for delta in stream:
                if self._closed or question_id != self._question_id:
                    return
                if not delta:
                    continue
                elapsed = (time.perf_counter() - started) * 1000
                if first_token_ms is None:
                    first_token_ms = elapsed
                pieces.append(delta)
                self._emit(AnswerEvent("delta", question_id, question, delta, first_token_ms, elapsed))
            if self._closed or question_id != self._question_id:
                return
            answer = "".join(pieces)
            if not answer.strip():
                raise AnswerProviderError("回答模型未返回回答内容，请检查模型设置后重试")
            if self._max_context_turns:
                self._context.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ])
                self._context = self._context[-self._max_context_turns * 2:]
            self._emit(AnswerEvent(
                "done", question_id, question, answer, first_token_ms,
                (time.perf_counter() - started) * 1000,
                decision_ms,
            ))
        except asyncio.CancelledError:
            self._emit(AnswerEvent("cancelled", question_id, question))
            raise
        except Exception as exc:
            if not self._closed and question_id == self._question_id:
                # Remote exception strings may contain URLs or credentials.
                message = str(exc) if isinstance(exc, AnswerProviderError) else f"{type(exc).__name__}: 回答生成失败"
                self._emit(AnswerEvent("error", question_id, question, message))
        finally:
            if stream is not None and hasattr(stream, "aclose"):
                try:
                    await stream.aclose()
                except Exception as exc:
                    if not self._closed and question_id == self._question_id:
                        self._emit(AnswerEvent(
                            "error", question_id, question,
                            f"{type(exc).__name__}: 回答连接关闭失败",
                        ))

    def wait_idle(self, timeout: float) -> bool:
        return self._idle.wait(timeout)

    async def _shutdown(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def close(self) -> None:
        if threading.current_thread() is self._thread:
            raise RuntimeError("close must be called outside the answer callback")
        with self._lock:
            if self._closed:
                return
            self._closed = True
        shutdown = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        try:
            shutdown.result(timeout=5)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
