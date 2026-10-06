"""Local Windows screenshot OCR. Images and OCR text are never uploaded here."""

import json
import ctypes
import os
import re
from pathlib import Path
import subprocess

from .text import normalize_transcript


class ScreenshotError(RuntimeError):
    pass


def recognize_screenshot(path: str | Path) -> str:
    image = Path(path).resolve()
    if image.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
        raise ScreenshotError("请选择 PNG、JPG 或 BMP 截图。")
    try:
        if not image.is_file() or not 0 < image.stat().st_size <= 20 * 1024 * 1024:
            raise ScreenshotError("图片不存在、为空或超过 20 MB，请重新选择。")
    except OSError:
        raise ScreenshotError("无法读取图片，请检查文件权限。") from None
    return _recognize(["-ImagePath", str(image)])


def clipboard_has_image() -> bool:
    """Inspect format metadata only, without reading clipboard text or images."""
    return os.name == "nt" and any(ctypes.windll.user32.IsClipboardFormatAvailable(fmt) for fmt in (2, 8, 17))


def recognize_clipboard() -> str:
    return _recognize(["-Clipboard"])


def _recognize(arguments: list[str]) -> str:
    if os.name != "nt":
        raise ScreenshotError("本地截图识别仅支持 Windows。")
    script = Path(__file__).with_name("windows_ocr.ps1")
    shell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    try:
        result = subprocess.run(
            [str(shell), "-STA", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script), *arguments],
            capture_output=True, encoding="utf-8", timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode != 0:
            raise ValueError
        data = json.loads(result.stdout.lstrip("\ufeff"))
        if data.get("error") == "clipboard":
            raise ScreenshotError("剪贴板中没有图片，请先截图或复制图片，再点击粘贴。")
        if data.get("error") == "language":
            raise ScreenshotError("未安装简体中文 OCR，请在 Windows 语言设置中安装简体中文的文字识别组件后重试。")
        if data.get("error") == "dimensions":
            raise ScreenshotError("图片尺寸超过本机 OCR 上限，请裁剪为单道题后重新导入。")
        text = data.get("text")
        if not isinstance(text, str) or len(text) > 20_000:
            raise ValueError
        text = normalize_transcript(text)
        text = re.sub(r"(?<=[\u4e00-\u9fff]) +(?=[\u4e00-\u9fff])", "", text)
        if not text:
            raise ScreenshotError("未识别到文字，请选择清晰的题目截图。")
        return text
    except ScreenshotError:
        raise
    except subprocess.TimeoutExpired:
        raise ScreenshotError("截图识别超时，请裁剪图片后重试。") from None
    except (OSError, ValueError, TypeError, AttributeError):
        raise ScreenshotError("本地截图识别失败，请检查图片格式及 Windows OCR 组件。") from None
