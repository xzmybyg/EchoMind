import json
import os
from unittest.mock import patch

import pytest

from echomind.settings import SettingsError, config_path, load_settings, save_settings


def test_default_path_uses_local_user_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert config_path() == tmp_path / "EchoMind" / "settings.json"


def test_missing_settings_do_not_create_a_file(tmp_path):
    path = tmp_path / "settings.json"
    assert load_settings(path) == {}
    assert not path.exists()


@pytest.mark.parametrize("threshold", [0, 0.65, 0.85, 1])
def test_question_threshold_roundtrip(tmp_path, threshold):
    path = tmp_path / "settings.json"
    save_settings({"question_threshold": threshold}, path)
    assert load_settings(path)["question_threshold"] == threshold


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf"), True, "0.85", None])
def test_invalid_question_threshold_preserves_saved_config(tmp_path, threshold):
    path = tmp_path / "settings.json"
    save_settings({"question_threshold": 0.85}, path)
    with pytest.raises(SettingsError):
        save_settings({"question_threshold": threshold}, path)
    assert load_settings(path)["question_threshold"] == 0.85


def test_speech_settings_roundtrip_and_hotwords_limit(tmp_path):
    path = tmp_path / "settings.json"
    values = {"asr_hotwords": "React Fiber, Promise", "asr_model_path": "D:/models/large-v3"}
    save_settings(values, path)
    assert load_settings(path) == values
    with pytest.raises(SettingsError):
        save_settings({"asr_hotwords": "词" * 501}, path)
    assert load_settings(path) == values


@pytest.mark.parametrize("engine", ["whisper", "sensevoice", "paraformer", "tencent"])
def test_engine_setting_roundtrip_and_invalid_value_preserves_config(tmp_path, engine):
    path = tmp_path / "settings.json"
    save_settings({"asr_engine": engine}, path)
    assert load_settings(path)["asr_engine"] == engine
    with pytest.raises(SettingsError):
        save_settings({"asr_engine": "unknown"}, path)
    assert load_settings(path)["asr_engine"] == engine


def test_roundtrip_all_fields_and_no_plaintext_keys(tmp_path):
    path = tmp_path / "nested" / "settings.json"
    values = {"api_base_url": "https://api.example.com", "api_model": "model",
              "api_key": "fake-api-secret", "jev_key": "fake-jev-secret",
              "answer_mode": "live", "device": "cpu", "source": "本机麦克风", "jev_enabled": True,
              "follow_enabled": True, "follow_side": "left"}
    with patch("echomind.settings._protect_secret", side_effect=lambda value: "encrypted:" + value[::-1]), \
         patch("echomind.settings._unprotect_secret", side_effect=lambda value: value.split(":", 1)[1][::-1]):
        save_settings(values, path)
        assert load_settings(path) == values
    saved = path.read_text(encoding="utf-8")
    assert "fake-api-secret" not in saved
    assert "fake-jev-secret" not in saved
    assert "api_key" not in json.loads(saved)
    assert not list(path.parent.glob(".settings-*.tmp"))


@pytest.mark.parametrize("contents", ["not-json fake-secret", "[]", '{"version": 2}',
                                     '{"version": 1, "api_key": "fake-secret"}',
                                     '{"version": 1, "jev_enabled": "yes"}',
                                     '{"version": 1, "answer_mode": "invalid"}',
                                     '{"version": 1, "source": "invalid"}',
                                     '{"version": 1, "follow_enabled": "yes"}',
                                     '{"version": 1, "follow_side": "top"}',
                                     '{"version": 1, "unrecognized": "fake-secret"}',
                                     '{"version": 1, "api_key_dpapi": "%%%"}'])
def test_corrupt_file_is_preserved_without_leaking_contents(tmp_path, contents):
    path = tmp_path / "settings.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(SettingsError) as error:
        load_settings(path)
    assert "fake-secret" not in str(error.value)
    assert path.read_text(encoding="utf-8") == contents


def test_encryption_failure_never_changes_existing_settings(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("original", encoding="utf-8")
    with patch("echomind.settings._protect_secret", side_effect=RuntimeError("fake-secret")):
        with pytest.raises(SettingsError) as error:
            save_settings({"api_key": "fake-secret"}, path)
    assert "fake-secret" not in str(error.value)
    assert path.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(".settings-*.tmp"))


def test_decryption_failure_preserves_file_and_hides_internal_error(tmp_path):
    path = tmp_path / "settings.json"
    contents = '{"version": 1, "api_key_dpapi": "c2VjcmV0"}'
    path.write_text(contents, encoding="utf-8")
    with patch("echomind.settings._unprotect_secret", side_effect=RuntimeError("fake-secret")):
        with pytest.raises(SettingsError) as error:
            load_settings(path)
    assert "fake-secret" not in str(error.value)
    assert path.read_text(encoding="utf-8") == contents


def test_failed_atomic_replace_preserves_original_and_cleans_temporary_file(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("original", encoding="utf-8")
    with patch("echomind.settings.os.replace", side_effect=PermissionError("fake-secret")):
        with pytest.raises(SettingsError) as error:
            save_settings({"api_model": "model"}, path)
    assert "fake-secret" not in str(error.value)
    assert path.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(".settings-*.tmp"))


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI required")
def test_real_current_user_dpapi_roundtrip(tmp_path):
    path = tmp_path / "settings.json"
    values = {"api_key": "fake-test-api-secret-中文", "jev_key": "fake-test-jev-secret"}
    save_settings(values, path)
    assert load_settings(path) == values
    assert all(secret not in path.read_text(encoding="utf-8") for secret in values.values())


def test_empty_secrets_are_not_encrypted(tmp_path):
    path = tmp_path / "settings.json"
    with patch("echomind.settings._protect_secret") as protect:
        save_settings({"api_key": "", "jev_key": ""}, path)
    protect.assert_not_called()
    assert load_settings(path) == {}
