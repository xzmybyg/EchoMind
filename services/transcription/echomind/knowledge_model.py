"""Explicit, pinned CPU retrieval model; never download during a query."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time

from . import components


def manifest():
    return json.loads(Path(__file__).with_name("knowledge_model.json").read_text(encoding="utf-8"))


_data = manifest()
MODEL_ID = _data["model"] + "@" + _data["revision"]
KNOWLEDGE_MODEL = components.Component("知识库语义检索模型（CPU）", "bge-small-zh-v1.5", "",
    sum(f["size"] for f in _data["files"]), 100 * 1024**2,
    "models/bge-small-zh-v1.5", tuple(f["filename"] for f in _data["files"]))


def model_path(directory=None):
    return (Path(directory) if directory else components.cache_root()) / KNOWLEDGE_MODEL.folder


def model_ready(folder=None):
    folder = Path(folder) if folder else model_path()
    return all((folder / f["filename"]).is_file() and (folder / f["filename"]).stat().st_size == f["size"]
               for f in _data["files"])


def install_model(directory, client, progress, cancel):
    target = model_path(directory)
    if model_ready(target):
        return
    downloads = directory / ".knowledge-downloads"
    downloads.mkdir(exist_ok=True)
    if shutil.disk_usage(directory).free < 300 * 1024**2:
        raise components.ComponentError("检索模型需要至少 300 MB 空闲空间。")
    with tempfile.TemporaryDirectory(prefix=".knowledge-staging-", dir=directory) as temporary:
        staging = Path(temporary)
        for index, entry in enumerate(_data["files"]):
            components._check_cancel(cancel)
            url = f"https://huggingface.co/Qdrant/bge-small-zh-v1.5/resolve/{_data['revision']}/{entry['filename']}"
            item = replace(KNOWLEDGE_MODEL, archive=entry["filename"], size=entry["size"],
                           sha256=entry["sha256"], source_url=url)
            part = downloads / (entry["filename"] + ".part")
            components._download(item, part, client, progress, cancel)
            shutil.copyfile(part, staging / entry["filename"])
            progress(f"检索模型 · {index + 1}/{len(_data['files'])}", (index + 1) * 100 / len(_data["files"]))
        components._check_cancel(cancel)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.rename(target.with_name(target.name + f".backup-{time.time_ns()}"))
        staging.rename(target)
    for entry in _data["files"]:
        (downloads / (entry["filename"] + ".part")).unlink()
    progress("知识库检索模型已准备完成", 100)


class LocalEmbedder:
    def __init__(self, folder=None):
        folder = Path(folder) if folder else model_path()
        if not model_ready(folder):
            raise ValueError("语义模型尚未准备；请到组件管理下载。")
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=_data["model"], specific_model_path=str(folder),
                                   local_files_only=True, providers=["CPUExecutionProvider"], threads=2)
        self._lock = threading.Lock()

    def encode(self, texts):
        import numpy as np
        with self._lock:
            return np.asarray(list(self.model.embed(texts, batch_size=16, parallel=None)), dtype=np.float32)
