"""GitHub release checks and validated staging; no writes to the running app."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import httpx

from .version import VERSION

REPOSITORY = "xzmybyg/EchoMind"
RELEASE_API = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
ASSET_NAME = "EchoMind-update.zip"
COMPONENTS = ("EchoMind.exe", "_internal", "capture")
MAX_DOWNLOAD = 2 * 1024**3
MAX_EXTRACTED = 4 * 1024**3


class UpdateError(RuntimeError):
    pass


@dataclass(frozen=True)
class Release:
    version: str
    notes: str
    url: str
    sha256: str
    size: int


def version_tuple(value: str) -> tuple[int, int, int]:
    if not isinstance(value, str) or not re.fullmatch(r"v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", value):
        raise UpdateError("版本号格式无效，需要稳定版本，例如 v0.1.2。")
    return tuple(int(part) for part in value.removeprefix("v").split("."))


def check_release(*, current=VERSION, client=None) -> Release | None:
    if client is None:
        with httpx.Client(timeout=15) as owned:
            return check_release(current=current, client=owned)
    try:
        response = client.get(RELEASE_API, headers={"Accept": "application/vnd.github+json", "User-Agent": f"EchoMind/{VERSION}"})
        if response.status_code == 404:
            raise UpdateError("未找到可公开访问的正式发布版本。请先发布 GitHub Release；私有仓库暂不支持自动更新。")
        if response.status_code == 403:
            raise UpdateError("GitHub 请求受限，请稍后重试。")
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or data.get("draft") or data.get("prerelease"):
            raise UpdateError("未找到有效的正式发布版本。")
        tag = data.get("tag_name")
        if version_tuple(tag) <= version_tuple(current):
            return None
        assets = data.get("assets", [])
        asset = next((item for item in assets if isinstance(item, dict) and item.get("name") == ASSET_NAME), None)
        if asset is None:
            raise UpdateError(f"新版 {tag} 尚未提供 {ASSET_NAME}，暂不能一键更新。")
        url = asset.get("browser_download_url", "")
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.netloc != "github.com" or not parsed.path.startswith(f"/{REPOSITORY}/releases/download/{tag}/"):
            raise UpdateError("更新包下载地址不可信。")
        digest = asset.get("digest", "")
        if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
            raise UpdateError("更新包缺少 SHA-256 校验信息，已拒绝更新。")
        size = asset.get("size")
        if type(size) is not int or not 0 < size <= MAX_DOWNLOAD:
            raise UpdateError("更新包大小无效。")
        return Release(tag.removeprefix("v"), str(data.get("body") or "暂无更新说明")[:8000], url, digest[7:].lower(), size)
    except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
        raise UpdateError("检查更新失败，请检查网络或稍后重试。") from exc


def extract_package(archive: Path, destination: Path, version: str):
    """Validate every Windows path before extracting any file."""
    try:
        with zipfile.ZipFile(archive) as package:
            files = package.infolist()
            if len(files) > 20000 or sum(item.file_size for item in files) > MAX_EXTRACTED:
                raise UpdateError("更新包解压大小超过限制。")
            seen = set()
            for item in files:
                if item.flag_bits & 1:
                    raise UpdateError("更新包不能加密。")
                path = PurePosixPath(item.filename)
                parts = path.parts
                if not parts or path.is_absolute() or "\\" in item.filename or any(
                    part in (".", "..") or ":" in part or re.search(r'[<>"|?*\x00-\x1f]', part) or part.endswith((".", " "))
                    or re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part)
                    for part in parts
                ):
                    raise UpdateError("更新包包含不安全的文件路径。")
                key = str(path).casefold()
                if key in seen or (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise UpdateError("更新包包含重复路径或符号链接。")
                seen.add(key)
                if parts[0] not in (*COMPONENTS, "update.json") or (parts[0] in ("EchoMind.exe", "update.json") and len(parts) != 1):
                    raise UpdateError("更新包包含非程序组件，模型和配置不能被覆盖。")
            manifest = json.loads(package.read("update.json"))
            if manifest != {"version": version, "format": 1}:
                raise UpdateError("更新包版本或格式不匹配。")
            if not {"echomind.exe", "_internal/python311.dll", "capture/echomind.capture.exe"} <= seen:
                raise UpdateError("更新包缺少必要的程序组件。")
            for name in ("EchoMind.exe", "_internal/python311.dll", "capture/EchoMind.Capture.exe"):
                item = package.getinfo(name)
                if item.is_dir() or item.file_size == 0:
                    raise UpdateError("更新包必要组件为空或不是文件。")
            package.extractall(destination)
    except (OSError, zipfile.BadZipFile, KeyError, ValueError, NotImplementedError) as exc:
        raise UpdateError("更新包损坏或无法解压。") from exc


def stage_update(release: Release, *, progress=lambda percent: None, client=None, directory=None) -> Path:
    if client is None:
        with httpx.Client(timeout=httpx.Timeout(30, connect=15), follow_redirects=True) as owned:
            return stage_update(release, progress=progress, client=owned, directory=directory)
    base = Path(directory) if directory else Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "EchoMind" / "updates"
    base.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="update-", dir=base))
    archive = stage / ASSET_NAME
    try:
        digest = hashlib.sha256()
        received = 0
        reported = -1
        deadline = time.monotonic() + 1800
        with client.stream("GET", release.url, headers={"User-Agent": f"EchoMind/{VERSION}"}) as response:
            response.raise_for_status()
            with archive.open("wb") as output:
                for chunk in response.iter_bytes(256 * 1024):
                    received += len(chunk)
                    if received > release.size or received > MAX_DOWNLOAD or time.monotonic() > deadline:
                        raise UpdateError("更新包下载大小异常或超时。")
                    digest.update(chunk)
                    output.write(chunk)
                    percent = min(99, received * 100 // release.size)
                    if percent != reported:
                        progress(percent)
                        reported = percent
        if received != release.size or digest.hexdigest() != release.sha256:
            raise UpdateError("更新包校验失败，未修改当前程序。")
        extract_package(archive, stage / "app", release.version)
        shutil.copyfile(Path(__file__).with_name("windows_update.ps1"), stage / "apply.ps1")
        progress(100)
        return stage
    except (httpx.HTTPError, OSError) as exc:
        raise UpdateError("更新下载失败，请检查网络、磁盘空间或目录权限。") from exc
    finally:
        # Only the exact archive created in our unique staging folder is removed.
        archive.unlink(missing_ok=True)


def launch_installer(stage: Path, *, install_dir=None, pid=None):
    if not getattr(sys, "frozen", False) and install_dir is None:
        raise UpdateError("源码运行不能自动替换，请使用打包后的 EchoMind.exe。")
    target = Path(install_dir or Path(sys.executable).parent).resolve()
    staged = stage.resolve()
    if not (target / "EchoMind.exe").is_file() or not (staged / "apply.ps1").is_file():
        raise UpdateError("找不到安装目录或更新助手。")
    try:
        subprocess.Popen(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                          "-File", str(staged / "apply.ps1"), "-InstallDir", str(target),
                          "-StageDir", str(staged / "app"), "-ProcessId", str(pid or os.getpid())],
                         creationflags=subprocess.CREATE_NO_WINDOW)
    except OSError as exc:
        raise UpdateError("无法启动更新助手，当前程序未修改。") from exc
