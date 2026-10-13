"""Pinned Qwen model and application-owned, portable Python/CUDA runtime."""
from dataclasses import replace
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import time
import zipfile

import httpx

from . import components
from .components import (ComponentError, QWEN_MODEL, QWEN_RUNTIME, _check_cancel,
                         _download, ready)


def manifest():
    return json.loads(Path(__file__).with_name("qwen_components.json").read_text(encoding="utf-8"))


def runtime_executable(folder):
    folder = Path(folder)
    return next((p for p in (folder / "Scripts/python.exe", folder / "python.exe") if p.is_file()), None)


def runtime_ready(folder):
    folder = Path(folder)
    required = ("torch/__init__.py", "torch/lib/torch_cuda.dll", "transformers/__init__.py",
                "librosa/__init__.py", "numpy/__init__.py", "soundfile.py")
    python = runtime_executable(folder)
    try:
        return python is not None and python.stat().st_size > 0 and all(
            (folder / "Lib/site-packages" / name).is_file()
            and (folder / "Lib/site-packages" / name).stat().st_size > 0 for name in required)
    except OSError:
        return False


def resolve_runtime(app_root, model_path, *, directory=None):
    cache = Path(directory) if directory is not None else components.cache_root()
    root = Path(app_root).resolve()
    roots = [root, *Path(model_path).resolve().parents[:4], cache]
    candidates = [parent / name / "qwen17-eval" for parent in roots for name in (".runtime", "runtime")]
    return next((p for p in candidates if runtime_ready(p)), cache / QWEN_RUNTIME.folder)


def missing_qwen(model_path, runtime_path):
    return [item for item, complete in ((QWEN_MODEL, ready(Path(model_path), QWEN_MODEL)),
                                       (QWEN_RUNTIME, runtime_ready(runtime_path))) if not complete]


def _fetch(entry, downloads, client, progress, cancel):
    item = replace(QWEN_MODEL, name=entry["filename"], archive=entry["filename"],
                   size=entry["size"], sha256=entry["sha256"], source_url=entry["url"])
    part = downloads / (entry["filename"] + ".part")
    try:
        _download(item, part, client, progress, cancel)
    except httpx.HTTPError:
        # Same fixed revision and SHA-256, not a different model or unverified mirror.
        if not item.url.startswith("https://huggingface.co/"):
            raise
        progress("官方模型源连接失败，尝试同版本镜像（仍校验 SHA-256）…", 0)
        _download(replace(item, source_url=item.url.replace("https://huggingface.co/", "https://hf-mirror.com/", 1)),
                  part, client, progress, cancel)
    return part


def _unpack(archive, destination, cancel, *, wheel=False, budget):
    """Vendor wheel libraries; no pip, global Python, PATH or executable install scripts."""
    with zipfile.ZipFile(archive) as package:
        entries = package.infolist()
        if sum(e.file_size for e in entries) > budget:
            raise ComponentError("Qwen 运行组件解压大小超出预期。")
        seen = set()
        for entry in entries:
            _check_cancel(cancel)
            name = PurePosixPath(entry.filename)
            if (name.is_absolute() or ".." in name.parts or "\\" in entry.filename or ":" in entry.filename
                    or entry.flag_bits & 1 or (entry.external_attr >> 16) & 0o170000 == 0o120000):
                raise ComponentError("Qwen 运行组件包含不安全路径。")
            parts = name.parts
            if not parts:
                continue
            base = destination / "Lib/site-packages" if wheel else destination
            if wheel and parts[0].endswith(".data"):
                if len(parts) < 3 or parts[1] == "scripts":
                    continue  # Worker imports libraries only; entry-point launchers are unnecessary.
                if parts[1] not in ("purelib", "platlib", "data", "headers"):
                    raise ComponentError("Qwen wheel 数据目录无效。")
                base = destination if parts[1] == "data" else base
                parts = parts[2:]
            target = base.joinpath(*parts)
            relative = str(target.relative_to(destination)).casefold()
            if relative in seen:
                raise ComponentError("Qwen 运行组件包含重复路径。")
            seen.add(relative)
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.open(entry) as source, target.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    _check_cancel(cancel)
                    output.write(chunk)


def _verify_runtime(folder, cancel):
    code = ("import torch, librosa, soundfile, accelerate; "
            "from transformers import AutoProcessor, AutoModelForMultimodalLM; "
            "assert torch.__version__ == '2.8.0+cu128'; "
            "assert callable(AutoProcessor.from_pretrained)")
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "_MEIPASS2"):
        env.pop(key, None)
    process = subprocess.Popen([str(folder / "python.exe"), "-I", "-X", "utf8", "-c", code],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    deadline = time.monotonic() + 120
    try:
        while process.poll() is None:
            _check_cancel(cancel)
            if time.monotonic() >= deadline:
                raise ComponentError("Qwen 运行环境校验超时，请重试。")
            cancel.wait(0.1)
        if process.returncode:
            raise ComponentError("Qwen 运行环境校验失败；请检查 Windows 64 位系统及 VC++ 运行库。")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)


def install_qwen(item, directory, client, progress, cancel):
    target = directory / item.folder
    if (ready(target, item) if item == QWEN_MODEL else runtime_ready(target)):
        return
    if item == QWEN_RUNTIME and (os.name != "nt" or os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64") not in ("AMD64", "x86")):
        raise ComponentError("Qwen GPU 组件目前仅支持 Windows x64。")
    data = manifest()
    entries = data["model"] if item == QWEN_MODEL else [data["python"], *data["wheels"]]
    downloads = directory / ".qwen-downloads"
    downloads.mkdir(exist_ok=True)
    pending = sum(max(0, e["size"] - min(e["size"], (downloads / (e["filename"] + ".part")).stat().st_size))
                  if (downloads / (e["filename"] + ".part")).exists() else e["size"] for e in entries)
    if shutil.disk_usage(directory).free < pending + item.extracted_size + 128 * 1024**2:
        raise ComponentError(f"缓存盘空间不足，{item.name}需要约 {(pending + item.extracted_size) / 1024**3:.1f} GiB 空闲空间。")
    with tempfile.TemporaryDirectory(prefix=".qwen-staging-", dir=directory) as staging:
        staged = Path(staging) / "component"
        staged.mkdir()
        for index, entry in enumerate(entries):
            progress(f"{item.name} · {index + 1}/{len(entries)}", index * 100 / len(entries))
            archive = _fetch(entry, downloads, client, progress, cancel)
            _check_cancel(cancel)
            if item == QWEN_MODEL:
                # Copy, rather than move, so cancellation retains verified download fragments.
                with archive.open("rb") as source, (staged / entry["filename"]).open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        _check_cancel(cancel)
                        output.write(chunk)
            else:
                progress(f"正在安装 {entry['filename']}…", index * 100 / len(entries))
                _unpack(archive, staged, cancel, wheel=index > 0, budget=item.extracted_size)
        if item == QWEN_RUNTIME:
            # The embedded distribution is application-managed; no system Python/pip is used.
            (staged / "python311._pth").write_text("python311.zip\n.\nLib/site-packages\nimport site\n", encoding="utf-8")
            progress("正在校验 Qwen 独立 GPU 环境…", 100)
            _verify_runtime(staged, cancel)
            (staged / ".qwen-ready.json").write_text(json.dumps({"revision": data["revision"]}), encoding="utf-8")
        _check_cancel(cancel)
        if not (ready(staged, item) if item == QWEN_MODEL else runtime_ready(staged)):
            raise ComponentError("Qwen 组件安装不完整，请重试。")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.rename(target.with_name(target.name + f".backup-{time.time_ns()}"))
        staged.rename(target)
    for entry in entries:
        (downloads / (entry["filename"] + ".part")).unlink()
    progress(f"{item.name}已准备完成", 100)
