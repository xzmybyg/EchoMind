import hashlib
import io
import threading
import zipfile
from dataclasses import replace

import httpx
import pytest

from echomind.components import (MODEL, GPU, ComponentError, DownloadCancelled,
                                 install_components, missing, ready, resolve_paths)


def package(names=None):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name in names or MODEL.files:
            archive.writestr(f"EchoMind/{MODEL.folder}/{name}", b"test content")
    data = output.getvalue()
    item = replace(MODEL, size=len(data), sha256=hashlib.sha256(data).hexdigest())
    return item, data


def populate(path, item):
    path.mkdir(parents=True, exist_ok=True)
    for name in item.files:
        (path / name).write_bytes(b"test")


def test_cpu_never_requires_gpu(tmp_path):
    populate(tmp_path / "model", MODEL)
    assert missing(tmp_path / "model", tmp_path / "gpu", "cpu") == []
    assert missing(tmp_path / "model", tmp_path / "gpu", "cuda") == [GPU]


def test_legacy_complete_components_take_priority(tmp_path):
    app, cache = tmp_path / "app", tmp_path / "cache"
    populate(app / MODEL.folder, MODEL)
    populate(cache / GPU.folder, GPU)
    assert resolve_paths(app, directory=cache) == (app / MODEL.folder, cache / GPU.folder)
    populate(app / GPU.folder, GPU)
    assert resolve_paths(app, directory=cache)[1] == app / GPU.folder


def test_incomplete_component_is_not_ready(tmp_path):
    populate(tmp_path, MODEL)
    (tmp_path / "tokenizer.json").write_bytes(b"")
    assert not ready(tmp_path, MODEL)


def test_download_install_and_skip_without_network(tmp_path):
    item, data = package()
    calls, progress = [], []

    def server(request):
        calls.append(request)
        return httpx.Response(200, content=data)

    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        install_components([item], directory=tmp_path, client=client, progress=lambda *args: progress.append(args))
        install_components([item], directory=tmp_path, client=client)
    assert len(calls) == 1
    assert ready(tmp_path / item.folder, item)
    assert not (tmp_path / (item.archive + ".part")).exists()
    assert progress[-1][1] == 100


@pytest.mark.parametrize("range_supported", [True, False])
def test_partial_download_resume_or_safe_restart(tmp_path, range_supported):
    item, data = package()
    partial = tmp_path / (item.archive + ".part")
    partial.write_bytes(data[:42])

    def server(request):
        assert request.headers["range"] == "bytes=42-"
        if range_supported:
            return httpx.Response(206, headers={"Content-Range": f"bytes 42-{len(data)-1}/{len(data)}"}, content=data[42:])
        return httpx.Response(200, content=data)

    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        install_components([item], directory=tmp_path, client=client)
    assert ready(tmp_path / item.folder, item)


def test_incorrect_range_never_appends(tmp_path):
    item, data = package()
    partial = tmp_path / (item.archive + ".part")
    partial.write_bytes(data[:42])
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(206, headers={"Content-Range": "bytes 0-10/11"}, content=b"bad"))) as client:
        with pytest.raises(ComponentError, match="续传"):
            install_components([item], directory=tmp_path, client=client)
    assert partial.read_bytes() == data[:42]


def test_bad_hash_not_installed(tmp_path):
    item, data = package()
    item = replace(item, sha256="0" * 64)
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=data))) as client:
        with pytest.raises(ComponentError, match="SHA-256"):
            install_components([item], directory=tmp_path, client=client)
    assert not ready(tmp_path / item.folder, item)
    assert not (tmp_path / (item.archive + ".part")).exists()


@pytest.mark.parametrize("names", [MODEL.files[:-1], (*MODEL.files, "../../settings.json"), (*MODEL.files[:-1], "model.bin")])
def test_invalid_archive_never_writes_outside_staging(tmp_path, names):
    if len(set(names)) != len(names):
        with pytest.warns(UserWarning, match="Duplicate name"):
            item, data = package(names)
    else:
        item, data = package(names)
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=data))) as client:
        with pytest.raises(ComponentError, match="不完整|不安全"):
            install_components([item], directory=tmp_path, client=client)
    assert not (tmp_path / item.folder).exists()
    assert not list(tmp_path.glob(".staging-*"))


def test_cancel_preserves_fragment_and_never_installs(tmp_path):
    item, data = package()
    fragment = tmp_path / (item.archive + ".part")
    fragment.write_bytes(data[:42])
    cancel = threading.Event()
    cancel.set()
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("must not call network"))) as client:
        with pytest.raises(DownloadCancelled):
            install_components([item], directory=tmp_path, client=client, cancel=cancel)
    assert fragment.read_bytes() == data[:42]


def test_partial_eof_can_resume(tmp_path):
    item, data = package()
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=data[:42]))) as client:
        with pytest.raises(ComponentError, match="未完成"):
            install_components([item], directory=tmp_path, client=client)
    assert (tmp_path / (item.archive + ".part")).read_bytes() == data[:42]


def test_low_disk_space_does_not_download(tmp_path, monkeypatch):
    item, data = package()
    from types import SimpleNamespace
    monkeypatch.setattr("echomind.components.shutil.disk_usage", lambda _: SimpleNamespace(free=0))
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("must not call network"))) as client:
        with pytest.raises(ComponentError, match="空间不足"):
            install_components([item], directory=tmp_path, client=client)


def test_incomplete_old_cache_preserved_on_install(tmp_path):
    item, data = package()
    target = tmp_path / item.folder
    target.mkdir(parents=True)
    (target / "old.txt").write_bytes(b"keep")
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=data))) as client:
        install_components([item], directory=tmp_path, client=client)
    assert ready(target, item)
    assert list(target.parent.glob("large-v3-turbo.backup-*/old.txt"))
