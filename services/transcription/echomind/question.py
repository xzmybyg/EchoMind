"""Small Chinese question heuristic for final ASR pieces.

Partials are revisions, so only finals enter the buffer. Incomplete question
markers carry across nearby finals; ordinary statements do not. This is not a
semantic classifier: indirect questions and uncommon phrasing may be missed,
and a grammatical fragment can still look like a complete question.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re

from .text import normalize_transcript


@dataclass(frozen=True)
class Question:
    text: str
    timestamp_ms: int


_MARKER = re.compile(r"请(?:你)?(?:介绍|解释|讲解|说说|讲讲)|(?:介绍|解释|讲解)(?:一下)|谈谈|为什么|怎么|如何|什么|多少|能不能|可不可以|是否|是不是")
_REPORTED = re.compile(r"不知道|不清楚|不明白|知道了|明白了|告诉(?:我|你)|(?:他|她|他们|有人|刚才)(?:说|问)|问题是")
_QUOTED = re.compile(r'“[^”]*”|「[^」]*」|『[^』]*』|"[^"]*"|‘[^’]*’')
_BOUNDARY = re.compile(r"[。！？!?；;\n]+")
_META = re.compile(
    r"(?:就是说|也就是说|或者说|比如说|例如).*(?:请问|请说出|怎么问)"
    r"|(?:提问|问题)(?:的)?(?:表达|格式|方式|措辞)"
    r"|(?:识别|判断|检测|当成).*(?:问题|提问)"
)
_SELF_EXPLANATION = re.compile(r"^\s*(?:(?:我来|让我|我会|我们来)(?:解释|介绍|讲)|(?:原因|答案)(?:就是|是)|之所以)")
_SELF_LECTURE = re.compile(r"^\s*(?:我|我们)(?:正在|在|给你|给大家)(?:讲解|解释|介绍)")
_TASK_REQUEST = re.compile(r"^\s*(?:请(?:你)?|帮我)?\s*(?:设计|实现|制定|给出)\s*(?:一个|一种|一套|一下)")
_NUMBER = r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万]+)"
_ARITHMETIC_REQUEST = re.compile(
    rf"^\s*(?:请问\s*)?{_NUMBER}\s*(?:加|减|乘以?|除以?|[+×÷*\-/])\s*{_NUMBER}"
    r"\s*等于\s*几(?:呢)?[.\s]*$"
)


def _key(text: str) -> str:
    return re.sub(r"[\W_]+", "", text).casefold()


def is_filler(text: str) -> bool:
    return _key(text) in ("", "嗯", "啊", "哦", "好", "好的", "ok", "谢谢", "谢谢大家")


def _classify(text: str, *, semantic_candidates: bool = False) -> str:
    """Return question, incomplete, or statement without using punctuation."""
    visible = _QUOTED.sub("", text)
    incomplete = False
    for clause in _BOUNDARY.split(visible):
        direct = re.match(r"\s*(?:请|为什么|怎么|如何|什么|能不能|可不可以)", clause)
        if not semantic_candidates and ((not direct and _META.search(clause)) or _SELF_EXPLANATION.search(clause) or _SELF_LECTURE.search(clause)):
            continue
        if _ARITHMETIC_REQUEST.fullmatch(clause):
            return "question"
        task = _TASK_REQUEST.match(clause)
        if task:
            if _key(clause[task.end():]):
                return "question"
            incomplete = True
        for match in _MARKER.finditer(clause):
            prefix = clause[:match.start()]
            if not semantic_candidates and _REPORTED.search(prefix):
                continue
            suffix = _key(clause[match.end():])
            # Redis 是什么 is complete even though 什么 ends the sentence.
            if suffix or (match.group() == "什么" and re.search(r".+是\s*$", prefix)) or (match.group() == "多少" and _key(prefix)):
                return "question"
            incomplete = True
        # Question particles alone are not enough: discard bare filler such as 啊呢.
        clean = _key(clause)
        if len(clean) >= 4 and clean.endswith("吗") and not _REPORTED.search(clause):
            return "question"
    return "incomplete" if incomplete else "statement"


class PausedQuestion:
    """Join at most three related final pieces within a 12 s capture window."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.text = ""
        self.started_at = 0
        self.parts = 0
        self.suspect = False

    def push(self, text: str, timestamp_ms: int, suspect: bool = False) -> tuple[str, bool, bool]:
        if is_filler(text):
            return text, suspect, False
        continuation = re.match(r"\s*(?:支持|要求|需要|其中|并且|包括|还要|同时)", text)
        incomplete = self.text and _classify(self.text) == "incomplete"
        joined = bool(self.text and 0 <= timestamp_ms - self.started_at <= 12_000
                      and self.parts < 3 and (continuation or incomplete))
        if joined:
            combined = text if text.startswith(self.text) else (
                self.text.rstrip("，,。.!！?？;； ") + ("，" if continuation else "") + text
            )
            if len(combined) <= 200:
                self.text = combined
                self.parts += 1
                self.suspect = self.suspect or suspect
                return combined, self.suspect, True
        self.clear()
        if _classify(text) == "incomplete" or _TASK_REQUEST.match(text) or re.search(r"方案[。.\s]*$", text):
            self.text, self.started_at, self.parts, self.suspect = text, timestamp_ms, 1, suspect
        return text, suspect, False


class QuestionBuffer:
    """Bounded final-text buffer with an 8 s carry and 30 s duplicate window.

    ``timestamp_ms`` is the ending capture time of the final piece. A returned
    question uses that time, including when it completes a preceding fragment.
    """

    def __init__(self, *, semantic_candidates: bool = False) -> None:
        # Semantic mode lets the remote judge, rather than exclusion keywords,
        # distinguish reported speech from genuine questions about that speech.
        self._semantic_candidates = semantic_candidates
        self._pending = ""
        self._pending_at = 0
        self._recent: deque[tuple[str, int]] = deque(maxlen=16)
        self.decision = "等待完整提问"

    def clear_pending(self) -> None:
        """Discard a fragment superseded by a separately reviewed question."""
        self._pending = ""
        self._pending_at = 0

    def push(self, text: str, timestamp_ms: int, *, final: bool = True) -> Question | None:
        if not final:
            return None
        text = normalize_transcript(text).strip()
        if not text:
            return None
        if not 0 <= timestamp_ms - self._pending_at <= 8_000:
            self._pending = ""
        if self._pending:
            # A final can revise/repeat the previous fragment as a whole.
            if not text.startswith(self._pending):
                text = self._pending.rstrip("，,。.!！?？;； ") + text
        self._pending = ""
        text = text[-200:]
        kind = _classify(text, semantic_candidates=self._semantic_candidates)
        if kind == "incomplete":
            self._pending = text
            self._pending_at = timestamp_ms
            self.decision = "问题尚未完整，请在 8 秒内继续说出主题"
            return None
        if kind != "question":
            self.decision = "未触发：当前发言不是明确提问"
            return None
        key = _key(text)
        while self._recent and not 0 <= timestamp_ms - self._recent[0][1] <= 30_000:
            self._recent.popleft()
        if any(previous == key for previous, _ in self._recent):
            self.decision = "未触发：30 秒内的重复提问"
            return None
        self._recent.append((key, timestamp_ms))
        self.decision = "已识别完整提问"
        return Question(text, timestamp_ms)
