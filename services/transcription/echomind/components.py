"""Pinned, resumable first-run components; development remains offline."""

from __future__ import annotations

import hashlib
import os
import shutil
import ssl
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx


class ComponentError(RuntimeError):
    pass


class DownloadCancelled(ComponentError):
    pass


@dataclass(frozen=True)
class Component:
    name: str
    archive: str
    sha256: str
    size: int
    extracted_size: int
    folder: str
    files: tuple[str, ...]

    @property
    def url(self):
        return f"https://github.com/xzmybyg/EchoMind/releases/download/v0.1.1/{self.archive}"


MODEL = Component("中文识别模型", "EchoMind-v0.1.1-model.zip",
                  "a7c5363b556a923b6988b980a33fcd0630470499e640e5c256f854af1163e239",
                  1492333404, 1622000000, "models/large-v3-turbo",
                  ("config.json", "model.bin", "preprocessor_config.json", "tokenizer.json", "vocabulary.json"))
GPU = Component("GPU 组件", "EchoMind-v0.1.1-cuda.zip",
                "e2c365a914b08a5595dcc61b8ea58aa0320f8e5ff05ac746441e751be301ae28",
                962096703, 1423000000, "cuda",
                ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn_adv64_9.dll", "cudnn_cnn64_9.dll",
                 "cudnn_engines_precompiled64_9.dll", "cudnn_engines_runtime_compiled64_9.dll",
                 "cudnn_graph64_9.dll", "cudnn_heuristic64_9.dll", "cudnn_ops64_9.dll", "cudnn64_9.dll"))


def cache_root() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "EchoMind/components"


def ready(folder: Path, component: Component) -> bool:
    try:
        return all((folder / name).is_file() and (folder / name).stat().st_size > 0 for name in component.files)
    except OSError:
        return False


def resolve_paths(app_root: Path, *, directory=None) -> tuple[Path, Path]:
    """Reuse legacy portable installations before the persistent user cache."""
    cache = Path(directory) if directory is not None else cache_root()
    return tuple(app_root / item.folder if ready(app_root / item.folder, item) else cache / item.folder
                 for item in (MODEL, GPU))


def missing(model: Path, cuda: Path, device: str) -> list[Component]:
    return [item for item, path in ((MODEL, model), (GPU, cuda))
            if (item is MODEL or device == "cuda") and not ready(path, item)]


def _check_cancel(cancel):
    if cancel.is_set():
        raise DownloadCancelled("下载已取消；已下载的部分保留，下次可继续。")


def _download(item, archive, client, progress, cancel):
    offset = archive.stat().st_size if archive.exists() else 0
    if offset > item.size:
        archive.unlink()  # Only our fixed, private download fragment.
        offset = 0
    if offset < item.size:
        headers = {"Accept-Encoding": "identity", "User-Agent": "EchoMind-component-download"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with client.stream("GET", item.url, headers=headers) as response:
            response.raise_for_status()
            if response.status_code == 206:
                expected = f"bytes {offset}-{item.size - 1}/{item.size}"
                if response.headers.get("Content-Range") != expected:
                    raise ComponentError("下载续传信息不匹配，请稍后重试。")
            elif response.status_code == 200:
                offset = 0  # The server does not support Range; restart safely.
            else:
                raise ComponentError("组件服务器返回了无效响应。")
            last_report = 0.0
            with archive.open("ab" if offset else "wb") as output:
                for chunk in response.iter_bytes(1024 * 1024):
                    _check_cancel(cancel)
                    offset += len(chunk)
                    if offset > item.size:
                        raise ComponentError("组件下载大小超出预期。")
                    output.write(chunk)
                    now = time.monotonic()
                    if now - last_report >= 0.2:
                        progress(f"正在下载{item.name} · {offset / 1024**2:.0f} / {item.size / 1024**2:.0f} MB", offset * 100 / item.size)
                        last_report = now
    _check_cancel(cancel)
    if archive.stat().st_size < item.size:
        raise ComponentError("组件下载未完成；已下载部分保留，请重试续传。")
    progress(f"正在校验{item.name}…", 100)
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            _check_cancel(cancel)
            digest.update(chunk)
    if archive.stat().st_size != item.size or digest.hexdigest() != item.sha256:
        archive.unlink()
        raise ComponentError("组件 SHA-256 校验失败，已丢弃损坏下载；请重试。")


def _extract(item, archive, destination, progress, cancel):
    prefix = f"EchoMind/{item.folder}/"
    expected = {prefix + name: name for name in item.files}
    with zipfile.ZipFile(archive) as package:
        entries = package.infolist()
        # A strict allowlist rejects traversal, symlinks, duplicate paths and unrelated files.
        if (len(entries) != len(expected) or {entry.filename for entry in entries} != set(expected)
                or sum(entry.file_size for entry in entries) > item.extracted_size
                or any(entry.file_size <= 0 or entry.flag_bits & 1
                       or (entry.external_attr >> 16) & 0o170000 == 0o120000 for entry in entries)):
            raise ComponentError("组件包文件不完整或包含不安全路径。")
        for index, entry in enumerate(entries):
            progress(f"正在安装{item.name} · {index + 1}/{len(entries)}", index * 100 / len(entries))
            with package.open(entry) as source, (destination / expected[entry.filename]).open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    _check_cancel(cancel)
                    output.write(chunk)


def install_components(items, *, directory=None, progress=lambda text, percent: None, cancel=None, client=None):
    """Validate in staging, then install. Never touch the running app or settings."""
    cancel = cancel if cancel is not None else threading.Event()
    directory = Path(directory) if directory is not None else cache_root()
    if client is None:
        with httpx.Client(verify=ssl.create_default_context(), follow_redirects=True,
                          timeout=httpx.Timeout(30, connect=15)) as owned:
            return install_components(items, directory=directory, progress=progress, cancel=cancel, client=owned)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".install.lock").open("a+b") as lock:
            if os.name == "nt":
                import msvcrt
                lock.seek(0)
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise ComponentError("另一 EchoMind 窗口正在准备组件，请稍后重试。") from exc
            for item in items:
                _check_cancel(cancel)
                target = directory / item.folder
                if ready(target, item):
                    continue
                archive = directory / (item.archive + ".part")
                downloaded = min(archive.stat().st_size, item.size) if archive.exists() else 0
                if shutil.disk_usage(directory).free < item.size - downloaded + item.extracted_size + 128 * 1024**2:
                    raise ComponentError(f"缓存盘空间不足，{item.name}需要约 {(item.size - downloaded + item.extracted_size) / 1024**3:.1f} GiB 空闲空间。")
                _download(item, archive, client, progress, cancel)
                with tempfile.TemporaryDirectory(prefix=".staging-", dir=directory) as staging:
                    staged = Path(staging) / "component"
                    staged.mkdir()
                    _extract(item, archive, staged, progress, cancel)
                    _check_cancel(cancel)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        target.rename(target.with_name(target.name + f".backup-{time.time_ns()}"))
                    staged.rename(target)
                archive.unlink()
                progress(f"{item.name}已准备完成", 100)
    except (httpx.HTTPError, OSError, zipfile.BadZipFile, NotImplementedError) as exc:
        raise ComponentError("组件准备失败，请检查网络、磁盘空间或文件权限，然后重试。") from exc
