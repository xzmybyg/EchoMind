"""Normalize Chinese script without rewriting recognized technical terms."""

from functools import lru_cache

from opencc import OpenCC


@lru_cache(maxsize=1)
def _converter() -> OpenCC:
    return OpenCC("t2s")


def normalize_transcript(text: str) -> str:
    """Use simplified Chinese consistently for captions and question detection."""
    return _converter().convert(text).strip()
