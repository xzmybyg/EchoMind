"""Offline releases and isolated Windows installations; never touch real apps."""

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile

import httpx
import pytest

from echomind.updater import (ASSET_NAME, Release, UpdateError, check_release,
                              extract_package, launch_installer, stage_update, version_tuple)


def asset_payload(**overrides):
    asset = {"name": ASSET_NAME, "size": 100, "digest": "sha256:" + "a" * 64,
             "browser_download_url": "https://github.com/xzmybyg/EchoMind/releases/download/v0.1.2/EchoMind-update.zip"}
    asset.update(overrides)
    return {"tag_name": "v0.1.2", "body": "修复识别问题", "draft": False, "prerelease": False, "assets": [asset]}


def client_for(data=None, status=200):
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(status, json=data or {})))


def archive_bytes(extra=None, version="0.1.2"):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("update.json", json.dumps({"version": version, "format": 1}))
        archive.writestr("EchoMind.exe", b"new-exe")
        archive.writestr("_internal/python311.dll", b"new-runtime")
        archive.writestr("capture/EchoMind.Capture.exe", b"new-capture")
        if extra:
            for name, content in extra.items():
                info = zipfile.ZipInfo("placeholder")
                info.filename = name
                archive.writestr(info, content)
    return stream.getvalue()


def test_semantic_version_comparison_not_lexical():
    assert version_tuple("v0.10.0") > version_tuple("0.9.9")


@pytest.mark.parametrize("invalid", [None, "0.1", "0.1.2-beta", "1.01.0", "v0.1.2/evil"])
def test_invalid_or_preview_versions_rejected(invalid):
    with pytest.raises(UpdateError):
        version_tuple(invalid)


def test_release_check_returns_new_asset_and_notes():
    with client_for(asset_payload()) as client:
        release = check_release(client=client)
    assert release.version == "0.1.2" and release.notes == "修复识别问题"
    assert release.sha256 == "a" * 64


@pytest.mark.parametrize("current", ["0.1.2", "1.0.0"])
def test_no_update_when_equal_or_older(current):
    with client_for(asset_payload()) as client:
        assert check_release(current=current, client=client) is None


@pytest.mark.parametrize("status", [404, 403, 500])
def test_github_failures_not_reported_as_up_to_date(status):
    with client_for(status=status) as client, pytest.raises(UpdateError):
        check_release(client=client)


@pytest.mark.parametrize("overrides", [{"digest": None}, {"size": -1}, {"size": True},
                                     {"name": "source.zip"}, {"browser_download_url": "http://github.com/file"},
                                     {"browser_download_url": "https://github.com/other/repo/releases/download/v0.1.2/file.zip"}])
def test_untrusted_incomplete_assets_rejected(overrides):
    with client_for(asset_payload(**overrides)) as client, pytest.raises(UpdateError):
        check_release(client=client)


def test_timeout_wrapped():
    def fail(request):
        raise httpx.ConnectTimeout("offline")
    with httpx.Client(transport=httpx.MockTransport(fail)) as client, pytest.raises(UpdateError):
        check_release(client=client)


def test_download_stage_checksum_and_progress(tmp_path):
    data = archive_bytes()
    release = Release("0.1.2", "", "https://github.com/package.zip", hashlib.sha256(data).hexdigest(), len(data))
    progress = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=data))) as client:
        stage = stage_update(release, client=client, directory=tmp_path, progress=progress.append)
    assert (stage / "app/EchoMind.exe").read_bytes() == b"new-exe"
    assert (stage / "apply.ps1").is_file()
    assert not (stage / ASSET_NAME).exists()
    assert progress[-1] == 100


@pytest.mark.parametrize("corruption", ["digest", "truncated", "extra", "badzip"])
def test_corrupt_download_never_stages_program(tmp_path, corruption):
    expected = archive_bytes()
    data = {"truncated": expected[:-5], "extra": expected + b"extra", "badzip": b"invalid"}.get(corruption, expected)
    checksum = "f" * 64 if corruption == "digest" else hashlib.sha256(data).hexdigest()
    size = len(data) if corruption == "badzip" else len(expected)
    release = Release("0.1.2", "", "https://github.com/package.zip", checksum, size)
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=data))) as client, pytest.raises(UpdateError):
        stage_update(release, client=client, directory=tmp_path)
    assert not list(tmp_path.rglob("EchoMind.exe"))


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/escape", "_internal/../../escape",
                                "_internal/evil:stream", "_internal/CON.txt", "_internal/evil.",
                                "_internal\\..\\..\\escape", "models/model.bin", "settings.json", "cuda/a.dll",
                                "_internal/PYTHON311.dll", "_internal/evil?.dll"])
def test_unsafe_windows_archive_paths_rejected_before_extraction(tmp_path, name):
    archive = tmp_path / "update.zip"
    archive.write_bytes(archive_bytes({name: b"bad"}))
    with pytest.raises(UpdateError):
        extract_package(archive, tmp_path / "app", "0.1.2")
    assert not (tmp_path / "app").exists()


def test_archive_version_must_match_release(tmp_path):
    archive = tmp_path / "update.zip"
    archive.write_bytes(archive_bytes(version="9.0.0"))
    with pytest.raises(UpdateError):
        extract_package(archive, tmp_path / "app", "0.1.2")


def test_source_mode_cannot_launch_installer(tmp_path):
    with pytest.raises(UpdateError):
        launch_installer(tmp_path)


def make_installation(tmp_path):
    install = tmp_path / "EchoMind"
    stage = tmp_path / "stage/app"
    for root, prefix in ((install, b"old"), (stage, b"new")):
        (root / "_internal").mkdir(parents=True)
        (root / "capture").mkdir()
        (root / "EchoMind.exe").write_bytes(prefix + b"-exe")
        (root / "_internal/python311.dll").write_bytes(prefix + b"-runtime")
        (root / "capture/EchoMind.Capture.exe").write_bytes(prefix + b"-capture")
    (install / "models").mkdir()
    (install / "models/model.bin").write_bytes(b"model")
    (install / "settings.json").write_bytes(b"settings")
    return install, stage


@pytest.mark.skipif(os.name != "nt", reason="Windows installer")
def test_windows_helper_replaces_only_components_and_keeps_backup(tmp_path):
    install, stage = make_installation(tmp_path)
    script = Path(__file__).parents[1] / "echomind/windows_update.ps1"
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script),
                             "-InstallDir", str(install), "-StageDir", str(stage), "-ProcessId", "0", "-NoRestart"], capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    status = json.loads((stage.parent / "result.json").read_text(encoding="utf-8-sig"))
    assert status["status"] == "complete"
    assert (Path(status["backup"]) / "EchoMind.exe").read_bytes() == b"old-exe"
    assert (install / "EchoMind.exe").read_bytes() == b"new-exe"
    assert (install / "models/model.bin").read_bytes() == b"model"
    assert (install / "settings.json").read_bytes() == b"settings"


@pytest.mark.skipif(os.name != "nt", reason="Windows installer")
def test_windows_helper_rolls_back_on_partial_move_failure(tmp_path):
    install, stage = make_installation(tmp_path)
    script = Path(__file__).parents[1] / "echomind/windows_update.ps1"
    # Simulate one locked component during replacement; native moves still run for all others.
    quote = lambda path: "'" + str(path).replace("'", "''") + "'"
    command = "function Move-Item { param($LiteralPath,$Destination) if ((Split-Path -Leaf $LiteralPath) -eq '_internal' -and (Split-Path -Leaf (Split-Path -Parent $LiteralPath)) -like '.update-ready-*') { throw 'Simulated locked component' }; Microsoft.PowerShell.Management\\Move-Item -LiteralPath $LiteralPath -Destination $Destination }; & " + quote(script) + " -InstallDir " + quote(install) + " -StageDir " + quote(stage) + " -ProcessId 0 -NoRestart"
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command], capture_output=True, timeout=20)
    assert result.returncode == 1
    status = json.loads((stage.parent / "result.json").read_text(encoding="utf-8-sig"))
    assert status["status"] == "rolled_back", status
    assert (install / "EchoMind.exe").read_bytes() == b"old-exe"
    assert (install / "_internal/python311.dll").read_bytes() == b"old-runtime"
    assert (install / "capture/EchoMind.Capture.exe").read_bytes() == b"old-capture"


@pytest.mark.skipif(os.name != "nt", reason="Windows installer")
def test_windows_helper_handles_download_and_install_on_different_drives(tmp_path):
    workspace = Path(__file__).resolve().parents[3]
    if workspace.drive == tmp_path.drive:
        pytest.skip("Two drives required")
    with tempfile.TemporaryDirectory(prefix="update-test-", dir=workspace / ".runtime") as separate:
        install, _ = make_installation(tmp_path)
        _, stage = make_installation(Path(separate))
        script = Path(__file__).parents[1] / "echomind/windows_update.ps1"
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script),
                                 "-InstallDir", str(install), "-StageDir", str(stage), "-ProcessId", "0", "-NoRestart"], capture_output=True, timeout=20)
        assert result.returncode == 0, result.stderr
        assert (install / "EchoMind.exe").read_bytes() == b"new-exe"
        assert (install / "models/model.bin").read_bytes() == b"model"
