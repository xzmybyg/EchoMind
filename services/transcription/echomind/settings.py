"""Explicitly saved desktop settings with current-user Windows key protection."""

import base64
from collections.abc import Mapping
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import tempfile
from .shortcuts import DEFAULT_SHORTCUTS, validate_shortcuts


class SettingsError(RuntimeError):
    """A safe settings error suitable for display, without file contents."""


_TEXT_FIELDS = ("api_base_url", "api_model", "answer_mode", "device", "source", *DEFAULT_SHORTCUTS)
_SECRET_FIELDS = ("api_key", "jev_key")


def config_path() -> Path:
    directory = os.environ.get("LOCALAPPDATA")
    if not directory:
        raise SettingsError("无法确定本机配置目录。")
    return Path(directory) / "EchoMind" / "settings.json"


def _dpapi(data: bytes, *, decrypt: bool) -> bytes:
    if os.name != "nt":
        raise SettingsError("密钥保存需要 Windows 用户加密服务。")

    class Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    function = crypt32.CryptUnprotectData if decrypt else crypt32.CryptProtectData
    function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.POINTER(Blob),
                         ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    function.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    buffer = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    output = Blob()
    try:
        # No LOCAL_MACHINE flag: only the current Windows user can decrypt.
        if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(output)):
            raise SettingsError("Windows 无法保护或读取配置密钥，请使用原 Windows 账户。")
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        if output.pbData:
            kernel32.LocalFree(output.pbData)


def _protect_secret(value: str) -> str:
    return base64.b64encode(_dpapi(value.encode("utf-8"), decrypt=False)).decode("ascii")


def _unprotect_secret(value: str) -> str:
    return _dpapi(base64.b64decode(value, validate=True), decrypt=True).decode("utf-8")


def _validate(settings: Mapping[str, str | bool]) -> dict[str, str | bool]:
    allowed = {*_TEXT_FIELDS, *_SECRET_FIELDS, "jev_enabled"}
    if not isinstance(settings, Mapping) or set(settings) - allowed:
        raise ValueError("invalid settings fields")
    result = dict(settings)
    for key, value in result.items():
        if key == "jev_enabled":
            if not isinstance(value, bool):
                raise ValueError("invalid setting type")
        elif not isinstance(value, str):
            raise ValueError("invalid setting type")
    for key, allowed_values in {"answer_mode": ("off", "demo", "live"), "device": ("cuda", "cpu"),
                                "source": ("腾讯会议", "本机麦克风")}.items():
        if key in result and result[key] not in allowed_values:
            raise ValueError("invalid setting value")
    validate_shortcuts(result)
    return result


def load_settings(path: Path | None = None) -> dict[str, str | bool]:
    """Read saved settings; missing files return defaults, corrupt files are preserved."""
    target = Path(path) if path is not None else config_path()
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("invalid settings version")
        version = document.pop("version", None)
        if type(version) is not int or version != 1:
            raise ValueError("invalid settings version")
        result = {}
        for key in _SECRET_FIELDS:
            if key in document:
                raise ValueError("plaintext secret is not allowed")
            encrypted = document.pop(key + "_dpapi", None)
            if encrypted is not None:
                if not isinstance(encrypted, str):
                    raise ValueError("invalid encrypted setting")
                result[key] = _unprotect_secret(encrypted)
        result.update(document)
        return _validate(result)
    except FileNotFoundError:
        return {}
    except Exception:
        raise SettingsError("无法读取配置文件，文件可能损坏或不属于当前 Windows 用户；原文件未改动。") from None


def save_settings(settings: Mapping[str, str | bool], path: Path | None = None) -> None:
    """Save explicitly, encrypting keys before writing any bytes to disk."""
    target = Path(path) if path is not None else config_path()
    temporary = None
    try:
        document = _validate(settings)
        for key in _SECRET_FIELDS:
            value = document.pop(key, "")
            if value:
                document[key + "_dpapi"] = _protect_secret(value)
        document["version"] = 1
        encoded = json.dumps(document, ensure_ascii=False, indent=2)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=".settings-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        raise SettingsError("无法保存配置，请检查本机配置目录权限和 Windows 加密服务。") from None
    finally:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                raise SettingsError("无法清理本机配置临时文件，请检查配置目录权限。") from None
