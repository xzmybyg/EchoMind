import hashlib
import io
import json
import threading
import zipfile
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from echomind import components, qwen_components as qwen
from echomind.components import ComponentError, DownloadCancelled, QWEN_MODEL, QWEN_RUNTIME, install_components


def populate_runtime(folder, *, portable=False):
    names = ["python.exe" if portable else "Scripts/python.exe", "Lib/site-packages/torch/__init__.py",
             "Lib/site-packages/torch/lib/torch_cuda.dll", "Lib/site-packages/transformers/__init__.py",
             "Lib/site-packages/librosa/__init__.py", "Lib/site-packages/numpy/__init__.py",
             "Lib/site-packages/soundfile.py"]
    for name in names:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")


@pytest.mark.parametrize("portable", [False, True])
def test_resolve_reuses_nearby_deployment_and_cache(tmp_path, portable):
    project = tmp_path / "project"
    model = project / ".models/qwen3-asr-1.7b"
    runtime = project / ".runtime/qwen17-eval"
    cache = tmp_path / "cache"
    assert qwen.resolve_runtime(project / "dist/light/EchoMind", model, directory=cache) == cache / QWEN_RUNTIME.folder
    populate_runtime(runtime, portable=portable)
    assert qwen.resolve_runtime(project / "dist/light/EchoMind", model, directory=cache) == runtime
    assert qwen.runtime_executable(runtime).name == "python.exe"
    assert qwen.missing_qwen(model, runtime) == [QWEN_MODEL]
    for name in QWEN_MODEL.files:
        path = model / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model")
    assert qwen.missing_qwen(model, runtime) == []
    (runtime / "Lib/site-packages/torch/lib/torch_cuda.dll").unlink()
    populate_runtime(cache / QWEN_RUNTIME.folder, portable=True)
    assert qwen.resolve_runtime(project / "dist/light/EchoMind", model, directory=cache) == cache / QWEN_RUNTIME.folder


def entry(name, data):
    return dict(filename=name, size=len(data), sha256=hashlib.sha256(data).hexdigest(), url=f"https://example.test/{name}")


def tiny_manifest():
    blobs = {name: name.encode() for name in QWEN_MODEL.files}
    return dict(revision="fixture", model=[entry(n, b) for n, b in blobs.items()]), blobs


def test_model_installs_verified_files_then_skips_network(tmp_path, monkeypatch):
    data, blobs = tiny_manifest()
    monkeypatch.setattr(qwen, "manifest", lambda: data)
    calls = []
    def server(request):
        calls.append(request)
        return httpx.Response(200, content=blobs[request.url.path.lstrip("/")])
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        install_components([QWEN_MODEL], directory=tmp_path, client=client)
        install_components([QWEN_MODEL], directory=tmp_path, client=client)
    assert len(calls) == len(QWEN_MODEL.files)
    assert components.ready(tmp_path / QWEN_MODEL.folder, QWEN_MODEL)
    assert not list((tmp_path / ".qwen-downloads").glob("*.part"))


def test_model_failure_does_not_publish_incomplete_install(tmp_path, monkeypatch):
    data, blobs = tiny_manifest()
    monkeypatch.setattr(qwen, "manifest", lambda: data)
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"corrupt"))) as client:
        with pytest.raises(ComponentError):
            install_components([QWEN_MODEL], directory=tmp_path, client=client)
    assert not (tmp_path / QWEN_MODEL.folder).exists()
    assert not list(tmp_path.glob(".qwen-staging-*"))


def test_model_resume_from_verified_fragment(tmp_path, monkeypatch):
    data, blobs = tiny_manifest()
    monkeypatch.setattr(qwen, "manifest", lambda: data)
    downloads = tmp_path / ".qwen-downloads"
    downloads.mkdir()
    fragment = downloads / "model.safetensors.part"
    full = blobs["model.safetensors"]
    fragment.write_bytes(full[:3])
    def server(request):
        name = request.url.path.lstrip("/")
        if name == "model.safetensors":
            assert request.headers["range"] == "bytes=3-"
            return httpx.Response(206, content=full[3:], headers={"Content-Range": f"bytes 3-{len(full)-1}/{len(full)}"})
        return httpx.Response(200, content=blobs[name])
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        install_components([QWEN_MODEL], directory=tmp_path, client=client)
    assert (tmp_path / QWEN_MODEL.folder / "model.safetensors").read_bytes() == full


def zip_bytes(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def runtime_manifest():
    python = zip_bytes({"python.exe": b"python", "python311._pth": b"python311.zip"})
    wheel = zip_bytes({name.removeprefix("Lib/site-packages/"): b"library" for name in (
        "Lib/site-packages/torch/__init__.py", "Lib/site-packages/torch/lib/torch_cuda.dll",
        "Lib/site-packages/transformers/__init__.py", "Lib/site-packages/librosa/__init__.py",
        "Lib/site-packages/numpy/__init__.py", "Lib/site-packages/soundfile.py")})
    blobs = {"python.zip": python, "fixture.whl": wheel}
    return dict(revision="fixture", python=entry("python.zip", python), wheels=[entry("fixture.whl", wheel)]), blobs


def test_runtime_install_is_portable_verified_and_skips_existing(tmp_path, monkeypatch):
    data, blobs = runtime_manifest()
    monkeypatch.setattr(qwen, "manifest", lambda: data)
    verify = Mock()
    monkeypatch.setattr(qwen, "_verify_runtime", verify)
    calls = []
    def server(request):
        calls.append(request)
        return httpx.Response(200, content=blobs[request.url.path.lstrip("/")])
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        install_components([QWEN_RUNTIME], directory=tmp_path, client=client)
        install_components([QWEN_RUNTIME], directory=tmp_path, client=client)
    target = tmp_path / QWEN_RUNTIME.folder
    assert qwen.runtime_ready(target) and len(calls) == 2
    assert json.loads((target / ".qwen-ready.json").read_text())["revision"] == "fixture"
    assert "Lib/site-packages\nimport site" in (target / "python311._pth").read_text()
    verify.assert_called_once()


def test_failed_runtime_validation_keeps_downloads_and_old_runtime(tmp_path, monkeypatch):
    data, blobs = runtime_manifest()
    monkeypatch.setattr(qwen, "manifest", lambda: data)
    monkeypatch.setattr(qwen, "_verify_runtime", Mock(side_effect=ComponentError("校验失败")))
    target = tmp_path / QWEN_RUNTIME.folder
    target.mkdir(parents=True)
    (target / "old.txt").write_text("keep")
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=blobs[r.url.path.lstrip("/")]))) as client:
        with pytest.raises(ComponentError, match="校验失败"):
            install_components([QWEN_RUNTIME], directory=tmp_path, client=client)
    assert (target / "old.txt").read_text() == "keep"
    assert not qwen.runtime_ready(target)
    assert len(list((tmp_path / ".qwen-downloads").glob("*.part"))) == 2


@pytest.mark.parametrize("name", ["../escape.dll", "/escape.dll", "C:/escape.dll", "..\\escape.dll"])
def test_archive_rejects_unsafe_paths(tmp_path, name):
    archive = tmp_path / "bad.zip"
    archive.write_bytes(zip_bytes({name: b"bad"}))
    with pytest.raises(ComponentError, match="不安全"):
        qwen._unpack(archive, tmp_path / "stage", threading.Event(), budget=100)
    assert not (tmp_path / "escape.dll").exists()


def test_wheel_data_layout_and_no_script_execution(tmp_path):
    archive = tmp_path / "wheel.whl"
    archive.write_bytes(zip_bytes({"pkg.data/purelib/package.py": b"library", "pkg.data/scripts/setup.exe": b"ignored"}))
    qwen._unpack(archive, tmp_path / "stage", threading.Event(), wheel=True, budget=100)
    assert (tmp_path / "stage/Lib/site-packages/package.py").read_bytes() == b"library"
    assert not list((tmp_path / "stage").rglob("*.exe"))


def test_cancel_before_download_never_connects(tmp_path):
    cancel = threading.Event()
    cancel.set()
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("network"))) as client:
        with pytest.raises(DownloadCancelled):
            install_components([QWEN_MODEL], directory=tmp_path, client=client, cancel=cancel)


def test_low_disk_space_no_network(tmp_path, monkeypatch):
    monkeypatch.setattr(qwen.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("network"))) as client:
        with pytest.raises(ComponentError, match="空间不足"):
            install_components([QWEN_MODEL], directory=tmp_path, client=client)


def test_manifest_download_sizes_and_pinned_sources():
    data = qwen.manifest()
    assert sum(e["size"] for e in data["model"]) == QWEN_MODEL.size
    assert data["python"]["size"] + sum(e["size"] for e in data["wheels"]) == QWEN_RUNTIME.size
    for e in [*data["model"], data["python"], *data["wheels"]]:
        assert e["url"].startswith("https://") and len(e["sha256"]) == 64
    assert all(data["revision"] in e["url"] for e in data["model"])
