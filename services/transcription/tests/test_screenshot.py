"""OCR validates local files and redacts subprocess errors without cloud calls."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from echomind.screenshot import ScreenshotError, recognize_clipboard, recognize_screenshot


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "面试题.png"
    path.write_bytes(b"test")
    return path


def test_local_ocr_uses_safe_arguments_and_simplified_text(image, monkeypatch):
    def run(args, **kwargs):
        assert args[-1] == str(image.resolve())
        assert "-File" in args and "-NoProfile" in args
        assert not kwargs.get("shell", False)
        assert kwargs["timeout"] == 30
        assert kwargs["creationflags"] == subprocess.CREATE_NO_WINDOW
        return SimpleNamespace(returncode=0, stdout=json.dumps({"text": "設 計 一 個方案\nReact Fiber"}))

    monkeypatch.setattr("echomind.screenshot.subprocess.run", run)
    assert recognize_screenshot(image) == "设计一个方案\nReact Fiber"


@pytest.mark.parametrize("error,notice", [("language", "简体中文 OCR"), ("dimensions", "裁剪"), ("recognition", "识别失败")])
def test_ocr_error_messages_are_actionable(image, monkeypatch, error, notice):
    monkeypatch.setattr("echomind.screenshot.subprocess.run", lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout=json.dumps({"error": error})))
    with pytest.raises(ScreenshotError, match=notice):
        recognize_screenshot(image)


@pytest.mark.parametrize("output", ["", "not json", "[]", '{"text": 3}', '{"text": ""}'])
def test_invalid_or_empty_result_fails_safely(image, monkeypatch, output):
    monkeypatch.setattr("echomind.screenshot.subprocess.run", lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout=output))
    with pytest.raises(ScreenshotError):
        recognize_screenshot(image)


def test_timeout_does_not_expose_arguments(image, monkeypatch):
    def run(*a, **k):
        raise subprocess.TimeoutExpired("private command", 30)
    monkeypatch.setattr("echomind.screenshot.subprocess.run", run)
    with pytest.raises(ScreenshotError, match="超时") as exc:
        recognize_screenshot(image)
    assert "private" not in str(exc.value)


def test_missing_and_unsupported_files_are_rejected(tmp_path):
    with pytest.raises(ScreenshotError, match="不存在"):
        recognize_screenshot(tmp_path / "missing.png")
    with pytest.raises(ScreenshotError, match="请选择"):
        recognize_screenshot(tmp_path / "script.ps1")


def test_clipboard_ocr_uses_sta_and_no_temporary_file(monkeypatch):
    def run(args, **kwargs):
        assert "-STA" in args and args[-1] == "-Clipboard"
        assert "-ImagePath" not in args
        return SimpleNamespace(returncode=0, stdout=json.dumps({"text": "题目"}))
    monkeypatch.setattr("echomind.screenshot.subprocess.run", run)
    assert recognize_clipboard() == "题目"


def test_empty_clipboard_returns_useful_error(monkeypatch):
    monkeypatch.setattr("echomind.screenshot.subprocess.run", lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout='{"error":"clipboard"}'))
    with pytest.raises(ScreenshotError, match="剪贴板中没有图片"):
        recognize_clipboard()
