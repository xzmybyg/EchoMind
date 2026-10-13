"""Window tests use offline events, no capture, credentials, or external service."""

import tkinter as tk
from types import SimpleNamespace

import pytest

from echomind.copilot import AnswerEvent
from echomind.gui import EchoMindWindow, asset_paths, preferred_pid
from echomind.gui import COLORS


@pytest.fixture(scope="module")
def tk_host():
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def window(tk_host, tmp_path):
    root = tk.Toplevel(tk_host)
    root.withdraw()
    app = EchoMindWindow(root, discover=lambda: [], settings_path=tmp_path / "settings.json")
    yield app
    if not app.destroyed:
        app._destroy()


def test_selects_main_process_not_crash_handler():
    assert preferred_pid([
        {"ProcessId": 1, "ParentProcessId": 3, "Name": "WeMeetCrashHandler.exe"},
        {"ProcessId": 2, "ParentProcessId": 3, "Name": "wemeetapp.exe"},
        {"ProcessId": 3, "ParentProcessId": 99, "Name": "wemeetapp.exe"},
    ]) == 3
    assert preferred_pid([]) is None


def test_qwen_option_selects_gpu_and_persists(window, monkeypatch):
    from echomind.asr_backends import ENGINE_LABELS
    from echomind.settings import load_settings
    window.asr_engine.set(ENGINE_LABELS["qwen3-asr-1.7b"])
    window._asr_engine_changed()
    window.source.set("本机麦克风")
    monkeypatch.setattr("echomind.gui.validate_model", lambda *_: None)
    config = window._config()
    assert config.asr_engine == "qwen3-asr-1.7b" and config.device == "cuda"
    assert "不传热词" in window.asr_engine_hint.get()
    assert window.save_config()
    assert load_settings(window.settings_path)["asr_engine"] == "qwen3-asr-1.7b"


def test_cloud_form_consent_and_model_bypass(window, monkeypatch):
    from echomind.asr_backends import ENGINE_LABELS
    from echomind.tencent_asr import TencentASRError
    window.asr_engine.set(ENGINE_LABELS["tencent"])
    window._asr_engine_changed()
    assert window.cloud_asr_fields.grid_info()
    assert not window.local_asr_fields.grid_info()
    assert "上传音频" in window.caption_source.get()
    window.source.set("本机麦克风")
    monkeypatch.setattr("echomind.gui.validate_model", lambda *_: pytest.fail("cloud must not check local models"))
    monkeypatch.setattr(window, "open_components", lambda: pytest.fail("cloud must not download models"))
    window._first_run_components()
    with pytest.raises(TencentASRError, match="同意"):
        window._config()
    for key, value in (("tencent_app_id", "123"), ("tencent_secret_id", "fake-id"), ("tencent_secret_key", "fake-key")):
        window.tencent_fields[key].set(value)
    window.cloud_consent.set(True)
    config = window._config()
    assert config.tencent.secret_key == "fake-key" and config.tencent.consent
    assert config.device == "cpu"
    assert window.save_config()
    from echomind.settings import load_settings
    saved = load_settings(window.settings_path)
    assert saved["tencent_secret_key"] == "fake-key"
    assert "consent" not in saved
    assert "fake-key" not in window.settings_path.read_text(encoding="utf-8")
    window.asr_engine.set(ENGINE_LABELS["whisper"])
    window._asr_engine_changed()
    assert not window.cloud_consent.get()
    assert window.local_asr_fields.grid_info() and not window.cloud_asr_fields.grid_info()


def test_speech_hotwords_and_model_directory_reach_session(window, tmp_path):
    window.source.set("本机麦克风")
    window.asr_hotwords.set("React Fiber, Promise")
    window.asr_model_path.set(str(tmp_path))
    for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.json"):
        (tmp_path / name).touch()
    config = window._config()
    assert config.model_path == tmp_path
    assert config.asr_hotwords == "React Fiber, Promise"
    window.asr_model_path.set(str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="不完整"):
        window._config()
    window.asr_model_path.set("")
    assert window._config().model_path == window.model_path


@pytest.mark.parametrize("engine", ["sensevoice", "paraformer"])
def test_engine_selection_clears_mismatched_path_and_routes_cpu(window, tmp_path, engine):
    from echomind.asr_backends import ENGINE_LABELS, MODEL_FILES
    window.source.set("本机麦克风")
    window.asr_model_path.set("old-whisper-model")
    window.asr_engine.set(ENGINE_LABELS[engine])
    window._asr_engine_changed()
    assert window.asr_model_path.get() == ""
    assert "CPU" in window.asr_engine_hint.get()
    for name in MODEL_FILES[engine]:
        (tmp_path / name).write_bytes(b"fixture")
    window.asr_model_path.set(str(tmp_path))
    config = window._config()
    assert config.asr_engine == engine and config.device == "cpu"
    assert config.model_path == tmp_path
    assert window.save_config()
    from echomind.settings import load_settings
    assert load_settings(window.settings_path)["asr_engine"] == engine


def test_experimental_first_launch_does_not_download_whisper(window, monkeypatch):
    from echomind.asr_backends import ENGINE_LABELS
    window.asr_engine.set(ENGINE_LABELS["sensevoice"])
    monkeypatch.setattr(window, "open_components", lambda: pytest.fail("must not download Whisper for SenseVoice"))
    window._first_run_components()
    window.asr_model_path.set("D:/missing-sensevoice-model")
    window.mode.set("离线演练")
    assert window._config(text_test=True).asr_engine == "sensevoice"
    window.source.set("本机麦克风")
    with pytest.raises(ValueError, match="不完整"):
        window._config()


def test_audio_revision_displays_original_without_overwriting_subtitle(window):
    window._handle("transcript", SimpleNamespace(kind="final", text="React Fiber 原理", original_text="React Fibr 原理", suspect=False))
    captions = window.caption_text.get("1.0", "end")
    assert "React Fiber 原理" in captions
    assert "音频复识别前：React Fibr 原理" in captions


def test_rejected_asr_is_not_a_confirmed_caption_and_can_be_manually_recovered(window):
    from echomind.asr import TranscriptEvent
    text = 'Hooks, React, Vue, JavaScript, Vue, JavaScript, Vue, JavaScript'
    window._handle('transcript', TranscriptEvent('rejected', text, 1, 1, True,
                                               rejection_reason='识别结果出现循环重复'))
    assert text not in window.caption_text.get('1.0', 'end')
    assert text not in window.compact_caption_text.get('1.0', 'end')
    assert '已拦截' in window.partial.get()
    assert '未发送' in window.repair_notice.get()
    assert window.pending_repair == text
    window.edit_repair()
    assert window.test_question.get() == text


@pytest.mark.parametrize("text", ["2-" * 35 + "2优优独播剧场——YoYo Television Series Exclusive",
                                  "React, Vue, " * 12])
def test_hallucination_pipeline_never_writes_captions_or_routes_question(window, text):
    import struct
    from unittest.mock import Mock
    from echomind.asr import Transcriber
    from echomind.cli import show_events

    stream = Transcriber(retry_low_confidence=True, partial_interval_ms=30000)
    stream._model = Mock()
    stream._model.transcribe.side_effect = [
        ([SimpleNamespace(text=text, avg_logprob=-.6)], None),
        ([SimpleNamespace(text="欢迎订阅明镜", avg_logprob=-.2)], None),
    ]
    before = window.caption_text.get("1.0", "end")
    compact_before = window.compact_caption_text.get("1.0", "end")
    events = stream.push_pcm(struct.pack("<h", 800) * 4800 + bytes(600 * 32))
    copilot = Mock()
    show_events(events, copilot)
    for event in events:
        window._handle("transcript", event)
    assert [event.kind for event in events] == ["rejected"]
    assert window.caption_text.get("1.0", "end") == before
    assert window.compact_caption_text.get("1.0", "end") == compact_before
    assert "欢迎订阅明镜" not in window.partial.get()
    assert "已拦截" in window.partial.get()
    assert not window.question_records
    copilot.submit_transcript.assert_not_called()
    stream._model.transcribe.assert_called_once()


def test_speech_topic_choice_controls_session_hotwords_and_custom_words(window):
    from echomind.asr import DEFAULT_HOTWORDS
    window.asr_topic.set("前端开发")
    window._asr_topic_changed()
    assert window.asr_hotwords.get() == DEFAULT_HOTWORDS
    window.mode.set("离线演练")
    assert window._config(text_test=True).asr_hotwords == DEFAULT_HOTWORDS
    window.asr_hotwords.set("Redis, Kafka")
    assert window.asr_topic.get() == "自定义"
    window.asr_topic.set("通用")
    window._asr_topic_changed()
    assert window.asr_hotwords.get() == ""


def test_hotwords_dialog_saves_multiline_and_comma_terms(window):
    from echomind.settings import load_settings
    window.asr_hotwords.set("React Fiber, Promise")
    window.root.deiconify()
    window.hotwords_button.invoke()
    dialog = window.hotwords_dialog
    assert window.hotwords_editor.get("1.0", "end-1c") == "React Fiber\nPromise"
    window.open_hotwords()
    assert window.hotwords_dialog is dialog
    window.hotwords_editor.delete("1.0", "end")
    window.hotwords_editor.insert("1.0", " Redis， Kafka \nReact Fiber,\n Promise\n")
    assert window.asr_hotwords.get() == "React Fiber, Promise"
    window.hotwords_save_button.invoke()
    expected = "Redis, Kafka, React Fiber, Promise"
    assert window.hotwords_dialog is None
    assert window.asr_hotwords.get() == expected
    assert window.asr_topic.get() == "自定义"
    assert load_settings(window.settings_path)["asr_hotwords"] == expected
    window.mode.set("离线演练")
    assert window._config(text_test=True).asr_hotwords == expected


def test_hotwords_cancel_keeps_original_and_clear_can_be_saved(window):
    from echomind.settings import load_settings
    window.asr_hotwords.set("Redis, Kafka")
    window.root.deiconify()
    window.open_hotwords()
    window.hotwords_clear_button.invoke()
    assert window.hotwords_editor.get("1.0", "end-1c") == ""
    assert window.asr_hotwords.get() == "Redis, Kafka"
    window.hotwords_cancel_button.invoke()
    assert window.asr_hotwords.get() == "Redis, Kafka"
    assert not window.settings_path.exists()
    window.open_hotwords()
    window.hotwords_clear_button.invoke()
    window.hotwords_save_button.invoke()
    assert window.asr_hotwords.get() == "" and window.asr_topic.get() == "通用"
    assert load_settings(window.settings_path)["asr_hotwords"] == ""


def test_hotwords_limit_and_failed_save_preserve_original_and_draft(window, monkeypatch):
    from echomind.settings import SettingsError
    window.asr_hotwords.set("Redis")
    window.root.deiconify()
    window.open_hotwords()
    window.hotwords_editor.delete("1.0", "end")
    window.hotwords_editor.insert("1.0", "词" * 501)
    window.hotwords_save_button.invoke()
    assert "500" in window.hotwords_notice.get()
    assert window.hotwords_dialog is not None
    assert window.asr_hotwords.get() == "Redis" and not window.settings_path.exists()
    window.hotwords_editor.delete("1.0", "end")
    window.hotwords_editor.insert("1.0", "React Hooks")
    def fail(*args):
        raise SettingsError("配置写入失败")
    monkeypatch.setattr("echomind.gui.save_settings", fail)
    window.hotwords_save_button.invoke()
    assert window.hotwords_dialog is not None
    assert window.asr_hotwords.get() == "Redis"
    assert window.hotwords_editor.get("1.0", "end-1c") == "React Hooks"
    assert "配置写入失败" in window.hotwords_notice.get()
    window.hotwords_cancel_button.invoke()


def test_hotwords_dialog_keyboard_save_cancel_and_busy_guard(window):
    window.root.deiconify()
    window._busy(True)
    window.open_hotwords()
    assert window.hotwords_dialog is None
    window._busy(False)
    window.open_hotwords()
    window.hotwords_editor.delete("1.0", "end")
    window.hotwords_editor.insert("1.0", "React Hooks\nVue")
    window.hotwords_editor.focus_force()
    window.root.update()
    window.hotwords_editor.event_generate("<Control-Return>")
    window.root.update()
    assert window.hotwords_dialog is None
    assert window.asr_hotwords.get() == "React Hooks, Vue"
    window.open_hotwords()
    window.hotwords_editor.insert("end", "\nRedis")
    window.hotwords_editor.focus_force()
    window.root.update()
    window.hotwords_editor.event_generate("<Escape>")
    window.root.update()
    assert window.hotwords_dialog is None and window.asr_hotwords.get() == "React Hooks, Vue"


@pytest.mark.parametrize("scaling", [1.0, 1.5, 2.0, 3.0])
def test_hotwords_dialog_controls_fit_and_focus_visible(window, scaling):
    previous = window.root.tk.call("tk", "scaling")
    window.root.tk.call("tk", "scaling", scaling)
    window.root.deiconify()
    try:
        window.open_hotwords()
        dialog = window.hotwords_dialog
        dialog.geometry(f"{dialog.minsize()[0]}x{dialog.minsize()[1]}")
        window.root.update()
        for button in (window.hotwords_clear_button, window.hotwords_cancel_button, window.hotwords_save_button):
            assert button.winfo_rooty() + button.winfo_height() <= dialog.winfo_rooty() + dialog.winfo_height()
            assert button.winfo_rootx() + button.winfo_width() <= dialog.winfo_rootx() + dialog.winfo_width()
        assert window.hotwords_editor.cget("highlightcolor") == COLORS["primary"]
        assert window.hotwords_editor.bind("<Tab>")
        window.hotwords_cancel_button.invoke()
    finally:
        window.root.tk.call("tk", "scaling", previous)


@pytest.mark.parametrize("saved_words", ["", "React Hooks, 自定义术语", "legacy"])
def test_default_hotword_upgrade_preserves_custom_and_disabled_words(window, monkeypatch, saved_words):
    from echomind.asr import DEFAULT_HOTWORDS, PREVIOUS_DEFAULT_HOTWORDS
    value = PREVIOUS_DEFAULT_HOTWORDS if saved_words == "legacy" else saved_words
    monkeypatch.setattr("echomind.gui.load_settings", lambda _: {"asr_hotwords": value})
    root = tk.Toplevel(window.root)
    root.withdraw()
    app = EchoMindWindow(root, discover=lambda: [], settings_path=window.settings_path)
    try:
        assert app.asr_hotwords.get() == (DEFAULT_HOTWORDS if saved_words == "legacy" else value)
    finally:
        app._destroy()


@pytest.mark.parametrize("scaling", [1.0, 2.0, 3.0])
def test_sidebar_collapse_reclaims_space_and_preserves_configuration(window, scaling):
    previous = window.root.tk.call("tk", "scaling")
    try:
        window.root.tk.call("tk", "scaling", scaling)
        window.root.deiconify()
        window.root.update()
        window.test_question.set("已有问题草稿")
        window.jev_key.set("offline-test-value")
        panels = window.caption_text.master.master.master
        width = panels.winfo_width()
        geometry = window.root.geometry()
        window.settings_button.invoke()
        window.sidebar_toggle.invoke()
        window.root.update_idletasks()
        assert window.sidebar_collapsed
        assert not window.sidebar.winfo_ismapped()
        assert not window.settings_expanded
        assert panels.winfo_width() > width + 250
        assert window.sidebar_toggle.cget("text") == "展开"
        assert window.navigation_rail.winfo_ismapped()
        assert window.settings_button.winfo_ismapped()
        assert window.root.geometry() == geometry
        window.sidebar_toggle.invoke()
        window.root.update_idletasks()
        assert window.sidebar.winfo_ismapped()
        assert panels.winfo_width() == width
        assert window.test_question.get() == "已有问题草稿"
        assert window.jev_key.get() == "offline-test-value"
    finally:
        window.root.tk.call("tk", "scaling", previous)


def test_sidebar_toggle_does_not_interrupt_capture_or_follow(window):
    session = SimpleNamespace(is_running=True)
    window.session = session
    window._busy(True)
    window.sidebar_toggle.invoke()
    window._set_compact_mode(True)
    window._set_compact_mode(False)
    assert window.sidebar_collapsed
    assert window.session is session
    assert str(window.source_combo.cget("state")) == "disabled"
    window.sidebar_toggle.invoke()
    assert not window.sidebar_collapsed


def test_unified_source_selection_switches_microphone_and_meeting(window):
    assert not hasattr(window, "process_combo")
    window.source_combo.set("本机麦克风")
    window.source_combo.event_generate("<<ComboboxSelected>>")
    assert window.start_button.cget("text") == "开始测试"
    window.source_combo.set("腾讯会议")
    window.source_combo.event_generate("<<ComboboxSelected>>")
    assert window.start_button.cget("text") == "开始采集"


@pytest.mark.parametrize("geometry", ["1040x740", "1320x820"])
@pytest.mark.parametrize("collapsed", [False, True])
def test_optimized_layout_aligns_controls_and_preserves_content(window, geometry, collapsed):
    window.root.geometry(geometry)
    window.root.deiconify()
    if collapsed:
        window.sidebar_toggle.invoke()
    window.root.update()
    toolbar = window.capture_toolbar
    controls = (window.source_combo, window.refresh_button, window.start_button, window.stop_button)
    centers = [control.winfo_rooty() + control.winfo_height() // 2 for control in controls]
    assert max(centers) - min(centers) <= 2
    for control in controls:
        assert control.winfo_rootx() >= toolbar.winfo_rootx()
        assert control.winfo_rootx() + control.winfo_width() <= toolbar.winfo_rootx() + toolbar.winfo_width()
    assert window.caption_text.winfo_height() >= 100
    assert window.answer_text.winfo_height() >= 100
    assert window.add_caption_button.winfo_ismapped()
    assert window.question_entry.winfo_width() >= 100
    window.source_combo.set("本机麦克风")
    window.source_combo.event_generate("<<ComboboxSelected>>")
    window.root.update_idletasks()
    assert not window.refresh_button.winfo_ismapped()
    assert window.start_button.winfo_ismapped()


@pytest.mark.parametrize("scaling", [1.0, 2.0, 3.0])
def test_compact_follow_layout_restores_full_window(window, scaling):
    window.root.after_cancel(window._follow_timer)
    previous = window.root.tk.call("tk", "scaling")
    try:
        window.root.tk.call("tk", "scaling", scaling)
        window.root.deiconify()
        window.root.update()
        geometry, minimum = window.root.geometry(), window.root.minsize()
        session = SimpleNamespace(is_running=True)
        window.session = session
        window._set_compact_mode(True)
        window.root.update()
        assert not window.full_view.winfo_ismapped()
        assert window.compact_caption_text.winfo_ismapped()
        assert window.compact_question_text.winfo_ismapped()
        assert not window.answer_text.winfo_ismapped()
        assert window.root.winfo_width() < int(geometry.split("x")[0])
        button = window.exit_follow_button
        assert button.winfo_rootx() + button.winfo_width() <= window.root.winfo_rootx() + window.root.winfo_width()
        assert window._set_compact_mode(True) is False
        window._set_compact_mode(False)
        window.root.update()
        assert window.root.geometry() == geometry
        assert window.root.minsize() == minimum
        assert window.full_view.winfo_ismapped()
        assert window.session is session
    finally:
        window.root.tk.call("tk", "scaling", previous)


def test_compact_content_updates_without_switching_session(window):
    window._handle("transcript", SimpleNamespace(kind="final", text="解释一下 React Fiber 原理"))
    window._handle("answer", AnswerEvent("start", 1, "解释一下 React Fiber 原理"))
    window._set_compact_mode(True)
    assert "React Fiber" in window.compact_caption_text.get("1.0", "end")
    assert "React Fiber" in window.compact_question_text.get("1.0", "end")
    window._handle("transcript", SimpleNamespace(kind="partial", text="还有一个问题"))
    assert "还有一个问题" in window.partial.get()
    window._handle("transcript", SimpleNamespace(kind="final", text="Promise 的作用是什么"))
    window._handle("answer", AnswerEvent("start", 2, "Promise 的作用是什么"))
    assert "Promise" in window.compact_caption_text.get("1.0", "end")
    questions = window.compact_question_text.get("1.0", "end")
    assert questions.index("Promise") < questions.index("React Fiber")
    window._set_compact_mode(False)
    assert "Promise" in window.caption_text.get("1.0", "end")


def test_follow_compacts_only_with_target_and_exits(window):
    window.follow_enabled.set(True)
    window.processes = {"meeting": 42}
    window.process.set("meeting")
    calls = []
    window.follower = SimpleNamespace(target=None, tick=lambda *args, **kwargs: calls.append(kwargs) or "跟随")
    window._follow_tick()
    assert not window.compact_mode
    window.follower.target = 123
    window._follow_tick()
    assert window.compact_mode
    window.exit_follow_button.invoke()
    assert not window.follow_enabled.get()
    assert not window.compact_mode
    assert calls[-1]["enabled"] is False


def test_follow_lost_target_and_microphone_restore_full_view(window):
    window.follow_enabled.set(True)
    window.follower = SimpleNamespace(target=123, tick=lambda *args, **kwargs: "跟随")
    window._follow_tick()
    assert window.compact_mode
    window.follower.target = None
    window._follow_tick()
    assert not window.compact_mode
    window.follower.target = 123
    window._follow_tick()
    window.source.set("本机麦克风")
    window._follow_tick()
    assert not window.compact_mode


def test_compact_switch_preserves_manual_minimization(window):
    window.root.after_cancel(window._follow_timer)
    window.root.deiconify()
    window.root.update()
    window.root.iconify()
    window.root.update()
    window._set_compact_mode(True)
    assert window.root.state() == "iconic"
    window._set_compact_mode(False)
    assert window.root.state() == "iconic"


def test_screenshot_cancel_preserves_draft(window, monkeypatch):
    monkeypatch.setattr("echomind.gui.filedialog.askopenfilename", lambda **kwargs: "")
    window.test_question.set("已有草稿")
    window.import_screenshot()
    assert window.test_question.get() == "已有草稿"
    assert not window.importing_screenshot


def test_default_shortcut_bindings_are_scoped(window):
    assert window.question_entry.bind("<Return>")
    assert window.question_entry.bind("<KP_Enter>")
    assert not window.root.bind("<Return>")
    assert window.root.bind("<Control-r>")
    assert window.root.bind("<Control-s>")


def test_keyboard_events_dispatch_submit_start_and_stop(window):
    called = []
    window.test_button.configure(command=lambda: called.append("submit"))
    window.start_button.configure(command=lambda: called.append("start"))
    window.stop_button.configure(command=lambda: called.append("stop"), state="normal")
    window.root.deiconify()
    window.root.update_idletasks()
    window.question_entry.focus_force()
    mapped = tk.BooleanVar(window.root)
    window.root.after(80, lambda: mapped.set(True))
    window.root.wait_variable(mapped)
    window.question_entry.focus_force()
    window.root.update()
    window.question_entry.event_generate("<Return>")
    window.question_entry.event_generate("<Control-r>")
    window.question_entry.event_generate("<Control-s>")
    assert called == ["submit", "start", "stop"]


def test_submit_shortcut_uses_button_and_respects_disabled_state(window):
    called = []
    window.test_button.configure(command=lambda: called.append("submit"))
    assert window._submit_shortcut() == "break"
    assert called == ["submit"]
    window.test_button.configure(state="disabled")
    window._submit_shortcut()
    assert called == ["submit"]


def test_capture_shortcuts_respect_busy_stopping_and_closing(window):
    called = []
    window.start_button.configure(command=lambda: called.append("start"))
    window.stop_button.configure(command=lambda: called.append("stop"), state="normal")
    window._start_shortcut()
    window._stop_shortcut()
    assert called == ["start", "stop"]
    window.start_button.configure(state="disabled")
    window.stop_button.configure(state="disabled")
    window._start_shortcut()
    window._stop_shortcut()
    assert called == ["start", "stop"]
    window.closing = True
    window.test_button.configure(command=lambda: called.append("submit"))
    window._submit_shortcut()
    assert called == ["start", "stop"]


def test_custom_shortcuts_remove_old_bindings_and_save(window, monkeypatch):
    monkeypatch.setattr("echomind.gui.save_settings", lambda values, path: None)
    window.shortcuts["shortcut_start"].set("F8")
    window.shortcuts["shortcut_stop"].set("F9")
    window.shortcuts["shortcut_submit"].set("Alt+Enter")
    window.save_config()
    assert window.root.bind("<F8>") and window.root.bind("<F9>")
    assert not window.root.bind("<Control-r>")
    assert not window.question_entry.bind("<Return>")
    assert window.question_entry.bind("<Alt-Return>")
    assert "F8" in window.shortcut_hint.get()


def test_conflict_does_not_change_existing_bindings_or_save(window, monkeypatch):
    def unexpected(*args):
        raise AssertionError("invalid shortcut settings must not be saved")
    monkeypatch.setattr("echomind.gui.save_settings", unexpected)
    window.shortcuts["shortcut_stop"].set("Ctrl+R")
    window.save_config()
    assert "不能重复" in window.config_notice.get()
    assert window.root.bind("<Control-r>") and window.root.bind("<Control-s>")


@pytest.fixture
def shortcut_dialog(window):
    from tkinter import ttk
    window.root.deiconify()
    window.root.update()
    window.settings_button.invoke()
    window.shortcut_settings_button.invoke()
    dialog = window.settings_dialog
    dialog.update()

    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)

    widgets = list(descendants(dialog))
    entries = [widget for widget in widgets if isinstance(widget, ttk.Entry)]
    buttons = {widget.cget("text"): widget for widget in widgets if isinstance(widget, ttk.Button)}
    yield dialog, entries, buttons
    if dialog.winfo_exists():
        buttons["取消"].invoke()


def test_settings_entry_is_fixed_below_sidebar_scroll(window, shortcut_dialog):
    dialog, entries, buttons = shortcut_dialog
    assert window.settings_button.winfo_ismapped()
    assert window.settings_button.master == window.navigation_rail
    assert window.settings_button.winfo_rooty() > window.navigation_buttons[-1].winfo_rooty()
    assert [entry.get() for entry in entries] == ["Enter", "Ctrl+R", "Ctrl+S", "Ctrl+F"]
    assert dialog.grab_current() == dialog
    assert dialog.bind("<Escape>")
    for button in buttons.values():
        assert button.winfo_ismapped()
        assert button.winfo_rooty() + button.winfo_height() <= dialog.winfo_rooty() + dialog.winfo_height()


def test_settings_cancel_does_not_change_shortcuts(window, shortcut_dialog):
    dialog, entries, buttons = shortcut_dialog
    entries[1].delete(0, "end")
    entries[1].insert(0, "F8")
    buttons["取消"].invoke()
    assert not dialog.winfo_exists()
    assert window.settings_dialog is None
    assert window.shortcuts["shortcut_start"].get() == "Ctrl+R"
    assert window.root.bind("<Control-r>") and not window.root.bind("<F8>")
    assert not window.settings_path.exists()


def test_settings_save_updates_shortcuts_and_persists(window, shortcut_dialog):
    from echomind.settings import load_settings
    dialog, entries, buttons = shortcut_dialog
    entries[1].delete(0, "end")
    entries[1].insert(0, "F8")
    buttons["保存"].invoke()
    assert not dialog.winfo_exists()
    assert window.root.bind("<F8>") and not window.root.bind("<Control-r>")
    assert load_settings(window.settings_path)["shortcut_start"] == "F8"


def test_settings_conflict_stays_open_without_changing_bindings(window, shortcut_dialog):
    dialog, entries, buttons = shortcut_dialog
    entries[2].delete(0, "end")
    entries[2].insert(0, "Ctrl+R")
    buttons["保存"].invoke()
    assert dialog.winfo_exists()
    assert window.shortcuts["shortcut_stop"].get() == "Ctrl+S"
    assert window.root.bind("<Control-s>")
    assert not window.settings_path.exists()


def test_settings_save_failure_preserves_shortcuts(window, shortcut_dialog, monkeypatch):
    from echomind.settings import SettingsError
    def fail(*args):
        raise SettingsError("无法保存配置")
    monkeypatch.setattr("echomind.gui.save_settings", fail)
    dialog, entries, buttons = shortcut_dialog
    entries[1].delete(0, "end")
    entries[1].insert(0, "F8")
    buttons["保存"].invoke()
    assert dialog.winfo_exists()
    assert window.shortcuts["shortcut_start"].get() == "Ctrl+R"
    assert window.root.bind("<Control-r>") and not window.root.bind("<F8>")


def test_settings_cannot_open_during_capture(window):
    window._busy(True)
    window.open_settings()
    assert window.settings_dialog is None


def test_update_menu_checks_in_background_and_displays_release(window, monkeypatch):
    from echomind.updater import Release
    release = Release("0.1.2", "更新说明", "https://github.com/example", "a" * 64, 123)
    monkeypatch.setattr("echomind.gui.check_release", lambda: release)
    monkeypatch.setattr("echomind.gui.threading.Thread", lambda target, **kwargs: SimpleNamespace(start=target))
    window.root.deiconify()
    window.settings_button.invoke()
    assert window.update_dialog is None
    window.maintenance_menu_button.invoke()
    window.update_menu_button.invoke()
    assert window.checking_updates
    assert window.recheck_update_button.instate(["disabled"])
    _, kind, payload = window.events.get_nowait()
    window._handle(kind, payload)
    assert not window.checking_updates
    assert "0.1.2" in window.update_notice.get()
    assert "更新说明" in window.update_notes_text.get("1.0", "end")
    # Source runs can check but may never replace the development environment.
    assert window.install_update_button.instate(["disabled"])


def test_update_network_failure_leaves_app_usable(window, monkeypatch):
    from echomind.updater import UpdateError
    def fail():
        raise UpdateError("网络失败")
    monkeypatch.setattr("echomind.gui.check_release", fail)
    monkeypatch.setattr("echomind.gui.threading.Thread", lambda target, **kwargs: SimpleNamespace(start=target))
    window.check_updates(automatic=True)
    _, kind, payload = window.events.get_nowait()
    window._handle(kind, payload)
    assert window.update_dialog is None
    assert window.update_notice.get() == "网络失败"
    assert not window.checking_updates and not window.destroyed


def test_declining_update_does_not_download(window, monkeypatch):
    monkeypatch.setattr("echomind.gui.sys.frozen", True, raising=False)
    monkeypatch.setattr("echomind.gui.messagebox.askyesno", lambda *args, **kwargs: False)
    window.available_release = SimpleNamespace(version="0.1.2")
    window.install_update()
    assert not window.updating


def test_staged_update_waits_for_capture_to_finish(window, monkeypatch, tmp_path):
    called = []
    window.session = SimpleNamespace(is_running=True, stop=lambda: called.append("stop"))
    monkeypatch.setattr("echomind.gui.launch_installer", lambda stage: called.append(stage))
    window._handle("update_staged", tmp_path)
    assert called == ["stop"]
    assert not window.destroyed
    window._handle("finished", None)
    assert called == ["stop", tmp_path]
    assert window.destroyed


def test_installer_launch_failure_does_not_exit_app(window, monkeypatch, tmp_path):
    from echomind.updater import UpdateError
    def fail(stage):
        raise UpdateError("无法启动更新助手")
    monkeypatch.setattr("echomind.gui.launch_installer", fail)
    window.updating = True
    window._handle("update_staged", tmp_path)
    assert not window.destroyed and not window.closing and not window.updating
    assert window.pending_update is None
    assert "无法启动" in window.update_notice.get()


def test_settings_button_expands_menu_without_opening_shortcuts(window):
    window.root.deiconify()
    window.root.update()
    assert window.settings_button.cget("style") == "Rail.TButton"
    assert not window.settings_menu.winfo_ismapped()
    window.settings_button.invoke()
    window.root.update()
    assert window.settings_expanded
    assert window.settings_menu.winfo_ismapped()
    assert window.shortcut_settings_button.winfo_ismapped()
    assert window.settings_dialog is None
    assert window.settings_button.instate(["selected"])
    window.settings_button.invoke()
    window.root.update()
    assert not window.settings_expanded
    assert not window.settings_menu.winfo_ismapped()
    assert window.settings_button.instate(["!selected"])


@pytest.mark.parametrize("scaling", [1.0, 2.0, 3.0])
def test_dropdown_stays_visible_without_reflowing_sidebar(window, scaling):
    previous = window.root.tk.call("tk", "scaling")
    try:
        window.root.tk.call("tk", "scaling", scaling)
        window.root.deiconify()
        window.root.update()
        previous_y = window.settings_button.winfo_rooty()
        previous_canvas_height = window.settings_canvas.winfo_height()
        window.settings_button.invoke()
        window.root.update()
        for button in (window.settings_button, window.shortcut_settings_button):
            assert button.winfo_ismapped()
            assert button.winfo_rooty() + button.winfo_height() <= window.root.winfo_rooty() + window.root.winfo_height()
        assert window.settings_button.winfo_rooty() == previous_y
        assert window.settings_canvas.winfo_height() == previous_canvas_height
        assert window.settings_popup.winfo_rootx() >= window.root.winfo_rootx()
        assert window.settings_popup.winfo_rootx() + window.settings_popup.winfo_width() <= window.root.winfo_rootx() + window.root.winfo_width()
    finally:
        window.root.tk.call("tk", "scaling", previous)


def test_dropdown_outside_click_and_keyboard_navigation(window):
    window.root.deiconify()
    window.root.update()
    window.root.focus_force()
    window.settings_button.invoke()
    window.root.update()
    window.shortcut_settings_button.event_generate("<Down>")
    window.root.update()
    assert window.root.focus_get() == window.follow_menu_button
    window.follow_menu_button.event_generate("<Up>")
    window.root.update()
    assert window.root.focus_get() == window.shortcut_settings_button
    window.shortcut_settings_button.event_generate("<Escape>")
    window.root.update()
    assert not window.settings_expanded
    assert window.root.focus_get() == window.settings_button
    window.settings_button.invoke()
    window.question_entry.event_generate("<ButtonPress-1>")
    window.root.update()
    assert not window.settings_expanded


def test_dropdown_selection_closes_before_opening_settings(window):
    window.root.deiconify()
    window.root.update()
    window.settings_button.invoke()
    window.shortcut_settings_button.invoke()
    assert not window.settings_expanded
    assert not window.settings_popup.winfo_ismapped()
    assert window.settings_dialog is not None
    window.settings_dialog.event_generate("<Escape>")


def test_dropdown_uniform_rows_and_focus_loss_closes(window):
    window.root.deiconify()
    window.root.update()
    window.root.focus_force()
    window.settings_button.invoke()
    window.root.update()
    assert len({button.winfo_height() for button in window._settings_buttons}) == 1
    assert all(button.cget("style") == "Dropdown.TButton" for button in window._settings_buttons)
    other = tk.Toplevel(window.root)
    try:
        other.update()
        other.focus_force()
        window.root.update()
        assert not window.settings_expanded
    finally:
        other.destroy()


def test_dropdown_enter_opens_only_once_and_returns_focus(window):
    window.root.deiconify()
    window.root.update()
    window.root.focus_force()
    window.settings_button.invoke()
    window.root.update()
    window.shortcut_settings_button.event_generate("<Return>")
    window.root.update()
    assert window.settings_dialog is not None
    assert not window.settings_expanded
    window.settings_dialog.event_generate("<Escape>")
    window.root.update()
    assert window.settings_dialog is None
    assert window.root.focus_get() == window.settings_button


def test_dropdown_busy_state_closes_and_prevents_opening(window):
    window.root.deiconify()
    window.root.update()
    window.settings_button.invoke()
    window._busy(True)
    assert not window.settings_expanded
    window.settings_button.invoke()
    assert not window.settings_expanded


def test_burst_stream_yields_for_paint_before_completion(window):
    window.root.after_cancel(window._poll)
    window.events.put((window.run_id, "answer", AnswerEvent("start", 1, "问题")))
    for index in range(12):
        window.events.put((window.run_id, "answer", AnswerEvent("delta", 1, "问题", str(index), first_token_ms=20)))
    window.events.put((window.run_id, "answer", AnswerEvent("done", 1, "问题", "完整回答", first_token_ms=20, elapsed_ms=50)))
    window._drain()
    window.root.update_idletasks()
    assert window.answer_text.get("1.0", "end-1c")
    assert not window.events.empty()
    assert window.question_records["0:1"].status == "生成中"
    assert "流式" in window.detection.get()
    assert "20 ms" in window.metrics.get()


def test_delayed_sse_displays_first_chunk_before_server_finishes(window):
    import asyncio
    import threading
    import httpx
    from echomind.copilot import Copilot
    from echomind.llm import OpenAICompatibleProvider

    arrived, release = threading.Event(), threading.Event()

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
            while not release.is_set():
                await asyncio.sleep(0.005)
            yield b'data: {"choices":[{"delta":{"content":" second"}}]}\n\ndata: [DONE]\n\n'

    def transport(request):
        import json
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, stream=Stream())

    def on_event(event):
        window.events.put((window.run_id, "answer", event))
        if event.kind == "delta":
            arrived.set()

    provider = OpenAICompatibleProvider("https://example.invalid", "mock", "fake-key", transport=httpx.MockTransport(transport))
    copilot = Copilot(provider, on_event)
    window.root.after_cancel(window._poll)
    try:
        copilot.submit_question("问题")
        assert arrived.wait(2)
        window._drain()
        window.root.update_idletasks()
        assert window.answer_text.get("1.0", "end-1c") == "first"
        assert window.history_text.get("1.0", "end-1c") == "first"
        assert window.question_records["0:1"].status == "生成中"
        assert not copilot.wait_idle(0)
        release.set()
        assert copilot.wait_idle(2)
        window.root.after_cancel(window._poll)
        window._drain()
        assert window.answer_text.get("1.0", "end-1c") == "first second"
        assert window.question_records["0:1"].status == "已完成"
    finally:
        release.set()
        copilot.close()


def test_history_stream_appends_without_replacing_visible_text(window, monkeypatch):
    window._handle("answer", AnswerEvent("start", 1, "问题"))
    replacements = []
    original = window._write
    def write(widget, text, *, replace=False):
        if widget is window.history_text:
            replacements.append(replace)
        original(widget, text, replace=replace)
    monkeypatch.setattr(window, "_write", write)
    window._handle("answer", AnswerEvent("delta", 1, "问题", "首段"))
    window._handle("answer", AnswerEvent("delta", 1, "问题", "后段"))
    assert replacements == [False, False]
    assert window.history_text.get("1.0", "end-1c") == "首段后段"


def test_image_paste_shortcut_uses_clipboard_not_file_picker(window, monkeypatch):
    called = []
    monkeypatch.setattr("echomind.gui.clipboard_has_image", lambda: True)
    monkeypatch.setattr(window, "_start_screenshot", called.append)
    assert window._paste_image_shortcut(SimpleNamespace(widget=window.question_entry)) == "break"
    assert len(called) == 1 and called[0].__name__ == "recognize_clipboard"


def test_text_paste_shortcut_keeps_native_behavior(window, monkeypatch):
    monkeypatch.setattr("echomind.gui.clipboard_has_image", lambda: False)
    assert window._paste_image_shortcut(SimpleNamespace(widget=window.question_entry)) is None
    assert not window.importing_screenshot


def test_paste_into_other_text_controls_does_not_trigger_ocr(window, monkeypatch):
    monkeypatch.setattr("echomind.gui.clipboard_has_image", lambda: True)
    assert window._paste_image_shortcut(SimpleNamespace(widget=window.caption_text)) is None
    assert not window.importing_screenshot


def test_paste_button_without_image_preserves_draft(window, monkeypatch):
    monkeypatch.setattr("echomind.gui.clipboard_has_image", lambda: False)
    window.test_question.set("草稿")
    window.paste_screenshot()
    assert window.test_question.get() == "草稿"
    assert "剪贴板中没有图片" in window.screenshot_notice.get()
    assert window.session is None


def test_repeated_paste_while_recognizing_is_ignored(window, monkeypatch):
    def unexpected():
        raise AssertionError("must not read clipboard while busy")
    monkeypatch.setattr("echomind.gui.clipboard_has_image", unexpected)
    window.importing_screenshot = True
    window.paste_screenshot()


def test_screenshot_confirmation_appends_without_starting_session(window):
    window.test_question.set("已有条件")
    window._append_screenshot_question("设计一个灰度发布方案")
    assert window.test_question.get() == "已有条件 设计一个灰度发布方案"
    assert window.session is None
    assert not window.question_records


@pytest.mark.parametrize("scaling", [1.0, 1.5, 2.0, 3.0])
def test_screenshot_actions_remain_visible_in_small_scaled_dialog(window, scaling):
    from tkinter import ttk
    original_scaling = window.root.tk.call("tk", "scaling")
    dialog = None
    try:
        window.root.tk.call("tk", "scaling", scaling)
        window.root.deiconify()
        window.root.update_idletasks()
        window._preview_screenshot("如何实现 Promise 的 then 方法？\n" * 100)
        dialog = next(child for child in window.root.winfo_children() if isinstance(child, tk.Toplevel))
        dialog.geometry("540x360")
        dialog.update()

        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)

        buttons = [widget for widget in descendants(dialog) if isinstance(widget, ttk.Button)]
        assert {button.cget("text") for button in buttons} == {"选中题目加入", "全部加入", "取消"}
        for button in buttons:
            assert button.winfo_ismapped()
            x = button.winfo_rootx() - dialog.winfo_rootx()
            y = button.winfo_rooty() - dialog.winfo_rooty()
            assert 0 <= x and x + button.winfo_width() <= dialog.winfo_width()
            assert 0 <= y and y + button.winfo_height() <= dialog.winfo_height()
        editor = next(widget for widget in descendants(dialog) if isinstance(widget, tk.Text))
        assert editor.winfo_height() > 20
        assert editor.bind("<Control-Return>") and dialog.bind("<Escape>")
    finally:
        if dialog is not None and dialog.winfo_exists():
            dialog.destroy()
        window.root.tk.call("tk", "scaling", original_scaling)


def test_screenshot_error_restores_button_and_preserves_draft(window):
    window.importing_screenshot = True
    window.screenshot_button.configure(state="disabled")
    window.test_question.set("草稿")
    window._handle("screenshot_error", "未识别到文字，请重新选择。")
    assert not window.importing_screenshot
    assert str(window.screenshot_button.cget("state")) == "normal"
    assert window.test_question.get() == "草稿"
    assert "未识别" in window.screenshot_notice.get()


def test_screenshot_ready_opens_preview_without_auto_submission(window, monkeypatch):
    previews = []
    monkeypatch.setattr(window, "_preview_screenshot", previews.append)
    window._handle("screenshot_ready", "第一道题\n第二道题")
    assert previews == ["第一道题\n第二道题"]
    assert window.test_question.get() == ""
    assert window.session is None


def test_screenshot_preview_selects_one_question_and_preserves_draft(window):
    from tkinter import ttk
    window.root.deiconify()
    window.root.update_idletasks()
    window.test_question.set("背景条件")
    window._preview_screenshot("第一道题\n第二道题")
    dialog = next(child for child in window.root.winfo_children() if isinstance(child, tk.Toplevel))

    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)

    widgets = list(descendants(dialog))
    editor = next(widget for widget in widgets if isinstance(widget, tk.Text))
    editor.tag_add("sel", "2.0", "2.end")
    button = next(widget for widget in widgets if isinstance(widget, ttk.Button) and widget.cget("text") == "选中题目加入")
    button.invoke()
    assert window.test_question.get() == "背景条件 第二道题"
    assert not dialog.winfo_exists()
    assert window.session is None


def test_ant_design_palette_and_accessible_action_states(window):
    from tkinter import ttk
    style = ttk.Style(window.root)
    assert COLORS["primary"] == "#1677FF"
    assert COLORS["background"] == "#F5F5F5"
    assert window.test_button.cget("style") == "Primary.TButton"
    assert style.lookup("TEntry", "bordercolor", ("focus",)) == COLORS["primary"]
    assert style.lookup("Primary.TButton", "foreground", ("disabled",)) == COLORS["disabled"]
    assert style.lookup("Treeview", "background", ("selected",)) == COLORS["tint"]

    def luminance(color):
        channels = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return sum(c * w for c, w in zip(linear, (0.2126, 0.7152, 0.0722)))

    for state in ((), ("active",), ("pressed",)):
        background = style.lookup("Primary.TButton", "background", state)
        assert (1.0 + 0.05) / (luminance(background) + 0.05) >= 4.5


def test_tab_sizes_are_equal_and_do_not_change_on_selection(window):
    window.root.deiconify()
    mapped = tk.BooleanVar(window.root)
    window.root.after(100, lambda: mapped.set(True))
    window.root.wait_variable(mapped)
    for notebook in (window.answer_tabs,):
        measurements = []
        for selected in range(len(notebook.tabs())):
            notebook.select(selected)
            window.root.update_idletasks()
            # Native hit testing gives the real tab rectangles, not style guesses.
            bounds = {}
            page = window.root.nametowidget(notebook.select())
            for y in range(page.winfo_y()):
                for x in range(notebook.winfo_width()):
                    try:
                        index = notebook.index(f"@{x},{y}")
                    except tk.TclError:
                        continue
                    if index not in bounds:
                        bounds[index] = [x, y, x, y]
                    else:
                        bounds[index][2] = max(bounds[index][2], x)
                        bounds[index][3] = max(bounds[index][3], y)
            assert len(bounds) == len(notebook.tabs())
            sizes = [(b[2] - b[0] + 1, b[3] - b[1] + 1) for b in bounds.values()]
            assert len(set(sizes)) == 1
            measurements.append(bounds)
        assert all(bounds == measurements[0] for bounds in measurements)
    window.root.withdraw()


def test_flat_scrollbar_layout_and_scroll_command_remain_functional(window):
    from tkinter import ttk
    style = ttk.Style(window.root)
    layout = str(style.layout("Vertical.TScrollbar"))
    assert "uparrow" not in layout and "downarrow" not in layout
    assert "thumb" in layout
    assert str(style.lookup("Vertical.TScrollbar", "gripcount")) == "0"
    window.root.deiconify()
    mapped = tk.BooleanVar(window.root)
    window.root.after(100, lambda: mapped.set(True))
    window.root.wait_variable(mapped)
    window._write(window.caption_text, "字幕\n" * 200)
    window.root.update_idletasks()
    scroll = next(child for child in window.caption_text.master.winfo_children() if isinstance(child, ttk.Scrollbar))
    window.root.tk.call(scroll.cget("command"), "moveto", 0.5)
    window.root.update_idletasks()
    assert 0.45 <= window.caption_text.yview()[0] <= 0.55
    assert scroll.get() == pytest.approx(window.caption_text.yview())
    window.root.withdraw()


@pytest.mark.parametrize("geometry", ["1040x740", "1320x820"])
def test_desktop_layout_keeps_editor_and_panels_visible(window, geometry):
    window.root.geometry(geometry)
    window.root.deiconify()
    mapped = tk.BooleanVar(window.root)
    window.root.after(100, lambda: mapped.set(True))
    window.root.wait_variable(mapped)
    window.root.update_idletasks()
    assert window.question_entry.winfo_width() >= 100
    assert window.caption_text.winfo_width() >= 180
    assert window.answer_text.winfo_width() >= 180
    right = window.test_button.winfo_rootx() + window.test_button.winfo_width()
    bottom = window.test_button.winfo_rooty() + window.test_button.winfo_height()
    assert right <= window.root.winfo_rootx() + window.root.winfo_width()
    assert bottom <= window.root.winfo_rooty() + window.root.winfo_height()
    upper = (window.import_screenshot_button, window.screenshot_button)
    lower = (window.clear_question_button, window.test_button)
    for first, second in zip(upper, lower):
        assert first.winfo_rootx() == second.winfo_rootx()
        assert first.winfo_width() == second.winfo_width()
        assert first.winfo_height() == second.winfo_height()
    for row in (upper, lower):
        assert row[0].winfo_rooty() == row[1].winfo_rooty()
        assert row[0].winfo_width() == row[1].winfo_width()
        assert row[1].winfo_rootx() - row[0].winfo_rootx() - row[0].winfo_width() == 8
    assert upper[1].winfo_rooty() + upper[1].winfo_height() + 8 == lower[1].winfo_rooty()
    assert window.add_caption_button.winfo_rootx() == window.merge_caption_button.winfo_rootx()
    assert window.add_caption_button.winfo_width() == window.merge_caption_button.winfo_width()
    assert window.add_caption_button.winfo_height() == window.merge_caption_button.winfo_height()
    window.test_question.set("待清空的草稿")
    window.clear_question_button.invoke()
    assert window.test_question.get() == ""
    window.root.withdraw()


def test_settings_tabs_and_mode_explanation(window):
    assert [window.settings_tabs.tab(tab, "text") for tab in window.settings_tabs.tabs()] == ["基本", "语音", "模型", "Jev"]
    window.mode.set("离线演练")
    assert "固定示例" in window.mode_hint.get()
    window.mode.set("DeepSeek 在线回答")
    assert "密钥" in window.mode_hint.get()
    window.mode.set("仅字幕")
    assert "不生成回答" in window.mode_hint.get()
    assert "不检测问题" in window.detection.get()


def test_navigation_rail_switches_pages_without_losing_drafts(window):
    window.root.deiconify()
    window.root.update()
    window.api_key.set("draft-key")
    window.navigation_buttons[2].invoke()
    window.root.update()
    assert window.settings_tabs.index("current") == 2
    assert window.settings_heading.get() == "回答模型"
    assert window.navigation_buttons[2].instate(["selected"])
    window.navigation_buttons[2].invoke()
    window.root.update()
    assert window.sidebar_collapsed and not window.sidebar.winfo_ismapped()
    assert window.navigation_buttons[2].instate(["!selected"])
    window.navigation_buttons[1].invoke()
    window.root.update()
    assert not window.sidebar_collapsed and window.sidebar.winfo_ismapped()
    assert window.settings_tabs.index("current") == 1
    assert window.api_key.get() == "draft-key"
    assert all(button.cget("image") for button in window.navigation_buttons)
    assert len({button.winfo_height() for button in window.navigation_buttons}) == 1


def test_settings_submenu_stays_inside_window_and_supports_keyboard(window):
    window.root.geometry("1040x740")
    window.root.deiconify()
    window.root.update()
    window.root.focus_force()
    window.toggle_sidebar()
    window.settings_button.invoke()
    window.root.update()
    window.maintenance_menu_button.invoke()
    window.root.update()
    assert window.settings_submenu_expanded
    assert window.update_menu_button.winfo_ismapped()
    assert window.root.focus_get() == window.update_menu_button
    window.update_menu_button.event_generate("<Down>")
    window.root.update()
    assert window.root.focus_get() == window.component_menu_button
    popup = window.settings_subpopup
    assert popup.winfo_x() >= 0 and popup.winfo_y() >= 0
    assert popup.winfo_x() + popup.winfo_width() <= window.root.winfo_width()
    assert popup.winfo_y() + popup.winfo_height() <= window.root.winfo_height()
    window.component_menu_button.event_generate("<Escape>")
    window.root.update()
    assert not window.settings_submenu_expanded
    assert window.settings_expanded
    assert window.root.focus_get() == window.maintenance_menu_button
    window.maintenance_menu_button.invoke()
    window.root.event_generate("<ButtonPress-1>", x=900, y=30)
    window.root.update()
    assert not window.settings_expanded and not popup.winfo_ismapped()


def test_question_decision_is_visible_without_overwriting_answer(window):
    window.mode.set("离线演练")
    window._handle("answer", AnswerEvent("start", 1, "解释一下React Fiber原理"))
    window._handle("answer", AnswerEvent("delta", 1, "解释一下React Fiber原理", "已有回答"))
    window._handle("answer", AnswerEvent("untriggered", 0, "嗯", "当前发言不是明确提问"))
    assert "不是明确提问" in window.detection.get()
    assert window.answer_text.get("1.0", "end").strip() == "已有回答"


def test_saved_configuration_loads_on_next_launch(window, tk_host, monkeypatch):
    for name in ("ECHOMIND_LLM_API_KEY", "ECHOMIND_JEV_API_KEY", "ECHOMIND_LLM_BASE_URL", "ECHOMIND_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    window.api_key.set("fake-local-test-key")
    window.jev_key.set("fake-local-jev-key")
    window.source.set("本机麦克风")
    window.mode.set("DeepSeek 在线回答")
    window.save_config()
    assert "已保存" in window.config_notice.get()
    assert "fake-local-test-key" not in window.settings_path.read_text(encoding="utf-8")
    root = tk.Toplevel(tk_host)
    root.withdraw()
    reopened = EchoMindWindow(root, discover=lambda: [], settings_path=window.settings_path)
    try:
        assert reopened.api_key.get() == "fake-local-test-key"
        assert reopened.jev_key.get() == "fake-local-jev-key"
        assert reopened.source.get() == "本机麦克风"
        assert reopened.mode.get() == "DeepSeek 在线回答"
        assert reopened.key_entry.cget("show") == "*"
        assert "已读取" in reopened.config_notice.get()
    finally:
        reopened._destroy()


def test_jev_threshold_is_saved_loaded_and_passed_to_session(window, tk_host):
    assert window.jev_threshold.get() == "85"
    window.jev_threshold.set("67.5")
    assert window.save_config()
    root = tk.Toplevel(tk_host)
    root.withdraw()
    reopened = EchoMindWindow(root, discover=lambda: [], settings_path=window.settings_path)
    try:
        assert reopened.jev_threshold.get() == "67.5"
        reopened.mode.set("离线演练")
        assert reopened._config(text_test=True).question_threshold == 0.675
        reopened._busy(True)
        assert reopened.jev_threshold_input.instate(["disabled"])
    finally:
        reopened._destroy()


@pytest.mark.parametrize("value", ["", "abc", "-1", "101", "nan", "inf"])
def test_invalid_jev_threshold_is_rejected_before_save_and_start(window, value):
    assert window.save_config()
    previous = window.settings_path.read_bytes()
    window.jev_threshold.set(value)
    assert not window.save_config()
    assert "0–100" in window.error.get()
    assert window.settings_path.read_bytes() == previous
    window.mode.set("离线演练")
    with pytest.raises(ValueError, match="阈值"):
        window._config(text_test=True)


def test_save_failure_has_visible_feedback_and_retains_inputs(window, monkeypatch):
    from echomind.settings import SettingsError

    def fail(*args):
        raise SettingsError("配置保存失败，请检查目录权限")

    monkeypatch.setattr("echomind.gui.save_settings", fail)
    window.api_key.set("fake-local-test-key")
    window.save_config()
    assert "保存失败" in window.config_notice.get()
    assert "保存失败" in window.error.get()
    assert window.api_key.get() == "fake-local-test-key"


def test_frozen_assets_reuse_complete_portable_components(monkeypatch, tmp_path):
    from echomind.components import MODEL, GPU
    for item in (MODEL, GPU):
        folder = tmp_path / item.folder
        folder.mkdir(parents=True)
        for name in item.files:
            (folder / name).write_bytes(b"test")
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys.executable", str(tmp_path / "EchoMind.exe"))
    capture, model, cuda = asset_paths()
    assert capture == tmp_path / "capture" / "EchoMind.Capture.exe"
    assert model == tmp_path / "models" / "large-v3-turbo"
    assert cuda == tmp_path / "cuda"


@pytest.mark.parametrize("scaling", [1.0, 1.5, 2.0, 3.0])
def test_component_dialog_actions_visible_and_escape_closes(window, scaling, monkeypatch, tmp_path):
    previous = window.root.tk.call("tk", "scaling")
    window.root.tk.call("tk", "scaling", scaling)
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("echomind.components.cache_root", lambda: tmp_path / "components")
    window.model_path = tmp_path / "missing-model"
    window.cuda_dir = tmp_path / "missing-cuda"
    window.root.deiconify()
    try:
        window.open_components()
        dialog = window.component_dialog
        window.root.update()
        assert "2.29 GiB" in window.component_details.get()
        assert str(window.component_download_button.cget("state")) == "normal"
        for button in (window.component_close_button, window.component_download_button):
            assert button.winfo_rooty() + button.winfo_height() <= dialog.winfo_rooty() + dialog.winfo_height()
        window.device.set("CPU（较慢）")
        window._component_state()
        assert "约 1.39 GiB" in window.component_details.get()
        assert "待下载：中文识别模型，" in window.component_details.get()
        dialog.focus_force()
        dialog.event_generate("<Escape>")
        window.root.update()
        assert window.component_dialog is None
    finally:
        window.root.tk.call("tk", "scaling", previous)


def test_development_component_manager_never_downloads(window, monkeypatch):
    monkeypatch.setattr("echomind.gui.install_components", lambda *args, **kwargs: pytest.fail("must not download in development"))
    window.root.deiconify()
    window.open_components()
    assert "开发环境" in window.component_details.get()
    assert str(window.component_download_button.cget("state")) == "disabled"
    window.download_components()
    assert not window.downloading_components
    window.component_close_button.invoke()


def test_frozen_missing_capture_opens_components_not_session(window, monkeypatch, tmp_path):
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("echomind.gui.asset_paths", lambda: (tmp_path / "capture.exe", tmp_path / "model", tmp_path / "cuda"))
    called = []
    monkeypatch.setattr(window, "open_components", lambda: called.append("components"))
    window.start_capture()
    assert called == ["components"] and window.session is None


def test_cancel_download_and_completion_restores_retry(window, monkeypatch, tmp_path):
    import time
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("echomind.gui.asset_paths", lambda: (tmp_path / "capture.exe", tmp_path / "model", tmp_path / "cuda"))
    from echomind.components import DownloadCancelled
    seen = []

    def download(items, *, cancel, progress):
        seen.extend(items)
        assert cancel.wait(5)
        raise DownloadCancelled("已取消，可重试")

    monkeypatch.setattr("echomind.gui.install_components", download)
    window.model_path, window.cuda_dir = tmp_path / "model", tmp_path / "cuda"
    window.root.deiconify()
    window.open_components()
    window.component_download_button.invoke()
    assert window.downloading_components
    assert str(window.component_device_combo.cget("state")) == "disabled"
    window.component_close_button.invoke()
    deadline = time.monotonic() + 5
    while window.downloading_components and time.monotonic() < deadline:
        window.root.update()
        time.sleep(0.01)
    assert not window.downloading_components
    assert len(seen) == 2
    assert window.component_notice.get() == "已取消，可重试"
    assert str(window.component_download_button.cget("state")) == "normal"
    window.component_close_button.invoke()


def test_first_run_only_prompts_when_components_missing(window, monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(window, "open_components", lambda: called.append(True))
    window.model_path, window.cuda_dir = tmp_path / "model", tmp_path / "cuda"
    window._first_run_components()
    assert called == [True]
    monkeypatch.setattr("echomind.gui.missing", lambda *args: [])
    window._first_run_components()
    assert called == [True]


def test_qwen_component_manager_reuses_existing_deployment(window, monkeypatch, tmp_path):
    from echomind.asr_backends import ENGINE_LABELS
    from echomind.components import QWEN_MODEL
    from test_qwen_components import populate_runtime
    model = tmp_path / "model"
    model.mkdir()
    for name in QWEN_MODEL.files:
        (model / name).write_bytes(b"fixture")
    runtime = tmp_path / "runtime"
    populate_runtime(runtime)
    window.asr_engine.set(ENGINE_LABELS["qwen3-asr-1.7b"])
    window.asr_model_path.set(str(model))
    monkeypatch.setattr(window, "_qwen_runtime_path", lambda: runtime)
    monkeypatch.setattr("echomind.gui.install_components", lambda *a, **k: pytest.fail("must reuse"))
    window.root.deiconify()
    window.open_components()
    assert window._component_items() == []
    assert "已就绪" in window.component_notice.get()
    assert str(window.component_download_button.cget("state")) == "disabled"
    assert str(window.component_device_combo.cget("state")) == "disabled"
    assert "无需另装 Python" in window.component_details.get()
    window.download_components()
    window.component_close_button.invoke()


def test_missing_qwen_prepares_only_qwen_and_supports_manual_development_download(window, monkeypatch, tmp_path):
    from echomind.asr_backends import ENGINE_LABELS
    from echomind.components import QWEN_MODEL, QWEN_RUNTIME
    import time
    window.asr_engine.set(ENGINE_LABELS["qwen3-asr-1.7b"])
    window.asr_model_path.set(str(tmp_path / "missing-model"))
    monkeypatch.setattr(window, "_qwen_runtime_path", lambda: tmp_path / "missing-runtime")
    seen = []
    monkeypatch.setattr("echomind.gui.install_components", lambda items, **kwargs: seen.extend(items))
    window.root.deiconify()
    window.open_components()
    assert "7.17 GiB" in window.component_details.get()
    assert str(window.component_download_button.cget("state")) == "normal"
    window.component_download_button.invoke()
    assert str(window.component_engine_combo.cget("state")) == "disabled"
    deadline = time.monotonic() + 5
    while window.downloading_components and time.monotonic() < deadline:
        window.root.update()
        time.sleep(0.01)
    assert seen == [QWEN_MODEL, QWEN_RUNTIME]
    assert not window.asr_model_path.get()
    assert not window.downloading_components
    window.component_close_button.invoke()


def test_qwen_start_opens_component_manager_if_runtime_missing(window, monkeypatch):
    from echomind.asr_backends import ENGINE_LABELS
    from echomind.components import QWEN_RUNTIME
    window.asr_engine.set(ENGINE_LABELS["qwen3-asr-1.7b"])
    monkeypatch.setattr(window, "_component_items", lambda: [QWEN_RUNTIME])
    called = []
    monkeypatch.setattr(window, "open_components", lambda: called.append(True))
    window.start_capture()
    assert called == [True] and window.session is None


@pytest.mark.parametrize("scaling", [1.0, 1.5, 2.0, 3.0])
def test_qwen_component_controls_visible_at_high_dpi(window, scaling, monkeypatch, tmp_path):
    from echomind.asr_backends import ENGINE_LABELS
    previous = window.root.tk.call("tk", "scaling")
    window.root.tk.call("tk", "scaling", scaling)
    window.asr_engine.set(ENGINE_LABELS["qwen3-asr-1.7b"])
    window.asr_model_path.set(str(tmp_path / "model"))
    monkeypatch.setattr(window, "_qwen_runtime_path", lambda: tmp_path / "runtime")
    window.root.deiconify()
    try:
        window.open_components()
        window.root.update()
        dialog = window.component_dialog
        for widget in (window.component_engine_combo, window.component_download_button, window.component_close_button):
            assert widget.winfo_rooty() + widget.winfo_height() <= dialog.winfo_rooty() + dialog.winfo_height()
            assert widget.winfo_rootx() + widget.winfo_width() <= dialog.winfo_rootx() + dialog.winfo_width()
        window.component_close_button.invoke()
    finally:
        window.root.tk.call("tk", "scaling", previous)


@pytest.mark.parametrize("scaling", [1.0, 1.5, 2.0, 3.0])
def test_follow_settings_cancel_and_save(window, scaling):
    from echomind.settings import load_settings
    previous = window.root.tk.call("tk", "scaling")
    window.root.tk.call("tk", "scaling", scaling)
    window.root.deiconify()
    window.open_follow_settings()
    window.follow_toggle.invoke()
    window.follow_dialog.focus_force()
    window.follow_dialog.event_generate("<Escape>")
    window.root.update()
    assert not window.follow_enabled.get()
    assert window.follow_dialog is None
    window.open_follow_settings()
    window.follow_toggle.invoke()
    window.follow_side_combo.set("左侧")
    window.root.update_idletasks()
    dialog = window.follow_dialog
    assert window.follow_save_button.winfo_rooty() + window.follow_save_button.winfo_height() <= dialog.winfo_rooty() + dialog.winfo_height()
    window.follow_save_button.invoke()
    assert window.follow_dialog is None
    assert window.follow_enabled.get() and window.follow_side.get() == "left"
    saved = load_settings(window.settings_path)
    assert saved["follow_enabled"] and saved["follow_side"] == "left"
    window.root.tk.call("tk", "scaling", previous)


def test_application_selection_automatically_chooses_main_process(window):
    rows = [{"ProcessId": 2, "ParentProcessId": 3, "Name": "wemeetapp.exe"}, {"ProcessId": 3, "Name": "wemeetapp.exe"}]
    window._handle("processes", rows)
    assert window.process.get() == "wemeetapp.exe · PID 3"
    assert window.process_details == rows
    assert window.source.get() == "腾讯会议"
    assert tuple(window.source_combo.cget("values")) == ("腾讯会议", "本机麦克风")
    window._handle("processes", [{"ProcessId": 4, "Name": "wemeetapp.exe"}])
    assert window.processes[window.process.get()] == 4
    window._handle("processes", [])
    assert window.process.get() == ""
    assert "未找到" in window.application_notice.get()


def test_follow_shortcut_toggle_during_capture_and_guard_dialog(window):
    window.session = SimpleNamespace(is_running=True)
    window.processes = {"meeting": 42}
    window.process.set("meeting")
    window.follower = SimpleNamespace(target=123, tick=lambda *args, **kwargs: "跟随")
    window._busy(True)
    window._follow_shortcut()
    assert window.follow_enabled.get() and window.compact_mode
    window._follow_shortcut()
    assert not window.follow_enabled.get() and not window.compact_mode
    assert window.session.is_running
    assert window.root.bind("<Control-f>")
    window.source.set("本机麦克风")
    window._follow_shortcut()
    assert not window.follow_enabled.get()
    assert "腾讯会议" in window.error.get()


def test_custom_follow_shortcut_is_saved_and_rebound(window):
    from echomind.settings import load_settings
    window.shortcuts["shortcut_follow"].set("F10")
    assert window.save_config()
    assert not window.root.bind("<Control-f>")
    assert window.root.bind("<F10>")
    assert load_settings(window.settings_path)["shortcut_follow"] == "F10"


def test_follow_tick_uses_selection_and_pauses_for_microphone(window, monkeypatch):
    calls = []
    window.follower = SimpleNamespace(tick=lambda *args, **kwargs: calls.append((args, kwargs)) or "跟随测试")
    window.follow_enabled.set(True)
    window.processes = {"会议": 22}
    window.process.set("会议")
    window._follow_tick()
    assert calls[-1][0][0] == 22 and calls[-1][1]["enabled"]
    window.source.set("本机麦克风")
    window._follow_tick()
    assert not calls[-1][1]["enabled"]
    assert "麦克风" in window.follow_notice.get()


def test_secrets_are_masked_by_default_and_clear_on_exit(window):
    assert window.key_entry.cget("show") == "*"
    assert window.jev_entry.cget("show") == "*"
    window.api_key.set("fake-local-test-key")
    window.jev_key.set("fake-local-jev-key")
    window._destroy()
    assert window.api_key.get() == ""
    assert window.jev_key.get() == ""


def test_meeting_capture_does_not_require_participant_checkbox(window):
    window.processes = {"腾讯会议": 123}
    window.process.set("腾讯会议")
    assert window._config().pid == 123
    window.mode.set("离线演练")
    assert window._config(text_test=True).answer_mode == "demo"


def test_sidebar_has_no_config_status_privacy_block_or_participant_checkbox(window):
    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)

    for widget in descendants(window.root):
        if "textvariable" in widget.keys():
            assert str(widget.cget("textvariable")) != str(window.config_notice)
        if "text" in widget.keys():
            text = str(widget.cget("text"))
            assert "使用与隐私" not in text
            assert "参与者授权" not in text
            assert "在线回答发送问题及近期问答" not in text


def test_microphone_configuration_bypasses_meeting(window):
    window.source.set("本机麦克风")
    window._source_changed()
    assert window.start_button.cget("text") == "开始测试"
    config = window._config()
    assert config.microphone
    assert config.pid == 0
    window.mode.set("离线演练")
    assert not window._config(text_test=True).microphone


def test_local_source_is_not_overwritten_by_meeting_discovery(window):
    window.source.set("本机麦克风")
    window._source_changed()
    window._handle("processes", [])
    assert window.status.get() == "就绪 · 使用系统默认麦克风"


def test_current_partial_is_not_appended_as_final(window):
    window._handle("transcript", SimpleNamespace(kind="partial", text="Redis为"))
    assert window.caption_text.get("1.0", "end").strip() == ""
    window._handle("transcript", SimpleNamespace(kind="final", text="Redis为什么快"))
    assert window.caption_text.get("1.0", "end").count("Redis为什么快") == 1


def test_streaming_answer_does_not_mix_with_captions_or_stale_answers(window):
    window._handle("answer", AnswerEvent("start", 2, "当前问题"))
    window._handle("answer", AnswerEvent("delta", 1, "旧问题", "过期内容"))
    window._handle("answer", AnswerEvent("delta", 2, "当前问题", "第一点"))
    window._handle("answer", AnswerEvent("delta", 2, "当前问题", "第二点"))
    window._handle("answer", AnswerEvent("done", 2, "当前问题", "第一点第二点", 100, 200, 10))
    assert window.answer_text.get("1.0", "end").strip() == "第一点第二点"
    assert "100 ms" in window.metrics.get()
    assert "10 ms" in window.metrics.get()
    assert window.detection.get() == "回答已完成。"
    assert window.caption_text.get("1.0", "end").strip() == ""


def test_form_is_locked_during_session_and_stop_is_nonblocking(window):
    class Session:
        is_running = True
        stopped = False

        def stop(self):
            self.stopped = True

    session = Session()
    window.session = session
    window._busy(True)
    assert str(window.start_button.cget("state")) == "disabled"
    assert str(window.stop_button.cget("state")) == "normal"
    window.stop()
    assert session.stopped
    window._handle("finished", None)
    assert str(window.start_button.cget("state")) == "normal"
    assert str(window.stop_button.cget("state")) == "disabled"


def test_session_start_uses_ui_configuration_without_environment_writes(window):
    received = []

    class Session:
        is_running = False

        def __init__(self, config, callback):
            received.append(config)
            self.callback = callback

        def start(self, question=None):
            self.is_running = True
            received.append(question)

    window.session_factory = Session
    window.mode.set("离线演练")
    window.test_question.set("解释一下 React Fiber 原理")
    window.start_test()
    assert received[0].answer_mode == "demo"
    assert received[1] == window.test_question.get()
    window._handle("finished", None)


def test_selected_caption_appends_editable_draft_without_timestamps(window):
    window._write(window.caption_text, "01:55:07  解释一下React Fiber\n01:55:11  原理。\n")
    window.caption_text.tag_add("sel", "1.0", "3.0")
    assert window.test_question.get() == ""
    window.add_selected_caption()
    assert window.test_question.get() == "解释一下React Fiber 原理。"
    assert window.session is None
    window.caption_text.tag_remove("sel", "1.0", "end")
    window.caption_text.tag_add("sel", "2.10", "2.end")
    window.add_selected_caption()
    assert window.test_question.get() == "解释一下React Fiber 原理。 原理。"


def test_missing_caption_selection_preserves_draft(window):
    window.test_question.set("已有草稿")
    window.add_selected_caption()
    assert "拖选" in window.error.get()
    assert window.test_question.get() == "已有草稿"


def test_caption_checkboxes_click_to_select_and_merge_without_ctrl(window):
    window.root.geometry("1040x740+30+30")
    window.root.deiconify()
    window._write(window.caption_text, "01:55:07  设计一个灰度发布方案\n01:55:08  好的\n01:55:11  支持逐步放量\n")
    window.root.update_idletasks()
    text = window.caption_text
    for row in (1, 3):
        text.see(f"{row}.0")
        window.root.update()
        x, y, width, height = text.bbox(f"{row}.0")
        text.event_generate("<Button-1>", x=x + width // 2, y=y + height // 2)
        text.event_generate("<ButtonRelease-1>", x=x + width // 2, y=y + height // 2)
        assert text.image_cget(f"{row}.0", "image") == str(window.caption_checkbox_images[1])
    assert "已选 2 条" in window.caption_selection_notice.get()
    text.event_generate("<Button-1>", x=x + width // 2, y=y + height // 2)
    assert "已选 1 条" in window.caption_selection_notice.get()
    assert text.image_cget("3.0", "image") == str(window.caption_checkbox_images[0])
    text.focus_force()
    window.root.update()
    text.event_generate("<Control-space>")
    assert "已选 2 条" in window.caption_selection_notice.get()
    window.merge_caption_button.invoke()
    assert window.test_question.get() == "设计一个灰度发布方案 支持逐步放量"
    assert not text.tag_ranges("caption_pick")
    assert all(text.image_cget(f"{row}.0", "image") == str(window.caption_checkbox_images[0]) for row in (1, 2, 3))
    assert text.cget("state") == "disabled" and window.session is None


def test_caption_checkbox_updates_without_duplicates_and_can_be_cleared(window):
    text = window.caption_text
    window._write(text, "01:55:07  解释一下React Fiber\n  音频复识别前：React Fibr\n")
    window._toggle_caption_selection("1.0")
    window._write(text, "01:55:11  原理。\n")
    assert len(text.image_names()) == 2
    assert "caption_checkbox" not in text.tag_names("2.0")
    assert text.image_cget("1.0", "image") == str(window.caption_checkbox_images[1])
    assert text.image_cget("3.0", "image") == str(window.caption_checkbox_images[0])
    window._clear_caption_selection()
    assert text.image_cget("1.0", "image") == str(window.caption_checkbox_images[0])
    window._write(text, "01:56:00  新的提问\n", replace=True)
    assert len(text.image_names()) == 1
    assert "新的提问" in text.get("1.0", "end") and not text.tag_ranges("caption_pick")


def test_selected_captions_merge_in_display_order_without_unselected_lines(window):
    window._write(window.caption_text, "01:55:07  设计一个灰度发布方案\n01:55:08  好的\n01:55:11  支持按用户ID逐步放量\n")
    window.test_question.set("另一道题的旧草稿")
    window._toggle_caption_selection("3.0")
    window._toggle_caption_selection("1.0")
    assert "已选 2 条" in window.caption_selection_notice.get()
    window.merge_selected_captions()
    assert window.test_question.get() == "设计一个灰度发布方案 支持按用户ID逐步放量"
    assert window.session is None and not window.question_records
    assert not window.caption_text.tag_ranges("caption_pick")


def test_caption_selection_can_be_toggled_and_ignores_diagnostic_notes(window):
    window._write(window.caption_text, "01:55:07  解释一下React Fiber\n  音频复识别前：React Fibr\n01:55:11  原理。\n")
    window._toggle_caption_selection("1.0")
    window._toggle_caption_selection("1.0")
    window._toggle_caption_selection("2.0")
    assert not window.caption_text.tag_ranges("caption_pick")
    window.caption_text.tag_add("sel", "1.0", "4.0")
    window.merge_selected_captions()
    assert window.test_question.get() == "解释一下React Fiber 原理。"
    assert "React Fibr" in window.caption_text.get("1.0", "end")


def test_new_subtitle_does_not_join_existing_manual_selection(window):
    window._write(window.caption_text, "01:55:07  解释一下React Fiber\n")
    window._toggle_caption_selection("1.0")
    window._handle("transcript", SimpleNamespace(kind="final", text="下一道问题"))
    window.merge_selected_captions()
    assert window.test_question.get() == "解释一下React Fiber"


def test_caption_selection_remains_correct_when_old_history_is_trimmed(window):
    window._write(window.caption_text, "无关字幕\n" * 395 + "01:55:07  保留的提问\n")
    window._toggle_caption_selection("396.0")
    window._write(window.caption_text, "新字幕\n" * 20)
    window.merge_selected_captions()
    assert window.test_question.get() == "保留的提问"


def test_missing_merge_selection_preserves_draft(window):
    window.test_question.set("已有草稿")
    window.merge_selected_captions()
    assert "多选" in window.error.get()
    assert window.test_question.get() == "已有草稿"


def test_current_and_history_questions_can_be_edited_without_changing_records(window):
    window.edit_current_question()
    assert "暂无" in window.error.get()
    window._handle("answer", AnswerEvent("start", 1, "React Fibre原因"))
    window._handle("answer", AnswerEvent("done", 1, "React Fibre原因", "原回答", 1, 2))
    window.edit_current_question()
    assert window.test_question.get() == "React Fibre原因"
    window.test_question.set("解释一下 React Fiber 原理")
    assert window.question_records["0:1"].question == "React Fibre原因"
    assert window.question_records["0:1"].answer == "原回答"
    window.edit_history_question()
    assert window.test_question.get() == "React Fibre原因"


def test_manual_submission_uses_running_session_and_draft_stays_editable(window):
    received = []
    window.session = SimpleNamespace(is_running=True, submit_question=received.append)
    window.mode.set("离线演练")
    window._busy(True)
    window.test_question.set("React Fiber 原理")
    assert str(window.question_entry.cget("state")) == "normal"
    assert str(window.test_button.cget("state")) == "normal"
    window.start_test()
    assert received == ["React Fiber 原理"]
    assert window.run_id == 0
    window.mode.set("仅字幕")
    window._busy(True)
    assert str(window.test_button.cget("state")) == "disabled"


def test_repair_requires_explicit_edit_and_submit_preserving_caption_and_answer(window):
    window.mode.set("离线演练")
    window.test_question.set("已有草稿")
    window._write(window.caption_text, "15:21:26  1加1等于进。\n")
    window._handle("answer", AnswerEvent("start", 1, "旧问题"))
    window._handle("answer", AnswerEvent("delta", 1, "旧问题", "旧回答"))
    window._handle("answer", AnswerEvent("reviewing", 2, "1加1等于进。"))
    window._handle("answer", AnswerEvent("needs_confirmation", 2, "1加1等于进。", "1加1等于几？", reason="结尾可能误听"))
    assert "原识别：1加1等于进。" in window.repair_notice.get()
    assert "建议：1加1等于几？" in window.repair_notice.get()
    assert window.repair_panel.winfo_manager() == "pack"
    assert window.test_question.get() == "已有草稿"
    assert window.answer_text.get("1.0", "end").strip() == "旧回答"
    assert len(window.question_records) == 1
    window._handle("answer", AnswerEvent("done", 1, "旧问题", "旧回答", 1, 2))
    assert window.pending_repair == "1加1等于几？"
    assert window.repair_panel.winfo_manager() == "pack"
    assert window.test_question.get() == "已有草稿"
    received = []
    window.session = SimpleNamespace(is_running=True, submit_question=received.append)
    window.edit_repair()
    assert window.test_question.get() == "1加1等于几？"
    assert not received
    window.start_test()
    assert received == ["1加1等于几？"]
    assert not window.pending_repair
    assert not window.repair_panel.winfo_manager()
    assert "1加1等于进。" in window.caption_text.get("1.0", "end")


def test_new_question_clears_old_repair_and_low_confidence_is_visible(window):
    window._handle("answer", AnswerEvent("needs_confirmation", 1, "原话", "建议", reason="不确定"))
    window._handle("answer", AnswerEvent("start", 2, "新问题"))
    assert not window.pending_repair
    window._handle("transcript", SimpleNamespace(kind="final", text="原话", suspect=True))
    assert "低可信" in window.partial.get()
    assert window.caption_text.get("1.0", "end").strip().endswith("原话")


def test_selected_and_edited_caption_generates_new_question_record(window):
    window.mode.set("离线演练")
    window._write(window.caption_text, "01:55:07  React Fibre原因\n")
    window.caption_text.tag_add("sel", "1.0", "2.0")
    window.add_selected_caption()
    window.test_question.set("React Fiber 原理")
    window.start_test()
    assert window.session.wait(3)
    while not window.events.empty():
        run_id, kind, payload = window.events.get_nowait()
        if run_id == window.run_id:
            window._handle(kind, payload)
    assert window.question_records["1:1"].question == "React Fiber 原理"
    assert window.question_records["1:1"].status == "已完成"
    assert "静态演练" in window.question_records["1:1"].answer
    assert "React Fibre原因" in window.caption_text.get("1.0", "end")


def test_process_refresh_does_not_unlock_running_form(window):
    window.session = SimpleNamespace(is_running=True)
    window._busy(True)
    window.status.set("正在生成回答…")
    window._handle("processes", [{"Name": "wemeetapp.exe", "ProcessId": 42}])
    assert str(window.refresh_button.cget("state")) == "disabled"
    assert window.status.get() == "正在生成回答…"


def test_text_history_is_bounded(window):
    window._write(window.caption_text, "文字\n" * 1000)
    assert int(window.caption_text.index("end-1c").split(".")[0]) <= 400


def test_question_history_records_answer_and_completion(window):
    window._handle("answer", AnswerEvent("start", 1, "Redis为什么快"))
    window._handle("answer", AnswerEvent("delta", 1, "Redis为什么快", "内存读写"))
    assert window.question_records["0:1"].answer == "内存读写"
    window._handle("answer", AnswerEvent("done", 1, "Redis为什么快", "内存读写与高效结构", 120, 300))
    record = window.question_records["0:1"]
    assert record.status == "已完成"
    assert "120 ms" in record.metrics
    assert window.history_text.get("1.0", "end").strip() == record.answer


def test_reviewing_old_answer_is_not_interrupted_by_new_question(window):
    window._handle("answer", AnswerEvent("start", 1, "旧问题"))
    window._handle("answer", AnswerEvent("done", 1, "旧问题", "旧回答", 1, 2))
    window.history_tree.selection_set("0:1")
    window._show_history()
    window._handle("answer", AnswerEvent("start", 2, "新问题"))
    window._handle("answer", AnswerEvent("delta", 2, "新问题", "新回答"))
    assert window.history_question.get() == "旧问题"
    assert window.history_text.get("1.0", "end").strip() == "旧回答"
    assert window.answer_text.get("1.0", "end").strip() == "新回答"
    window.history_tree.selection_set("0:2")
    window._show_history()
    assert window.history_text.get("1.0", "end").strip() == "新回答"


def test_question_history_keeps_cancelled_and_failed_answers(window):
    window._handle("answer", AnswerEvent("start", 1, "旧问题"))
    window._handle("answer", AnswerEvent("delta", 1, "旧问题", "部分回答"))
    window._handle("answer", AnswerEvent("start", 2, "新问题"))
    window._handle("answer", AnswerEvent("cancelled", 1, "旧问题"))
    window._handle("answer", AnswerEvent("error", 2, "新问题", "回答请求失败"))
    assert window.question_records["0:1"].status == "已停止"
    assert window.question_records["0:1"].answer == "部分回答"
    assert window.question_records["0:2"].status == "失败"
    assert window.question_records["0:2"].metrics == "回答请求失败"


def test_question_history_is_bounded_and_session_ids_do_not_collide(window):
    for number in range(102):
        window._handle("answer", AnswerEvent("start", number, f"问题{number}"))
    assert len(window.question_records) == len(window.history_tree.get_children()) == 100
    assert "0:0" not in window.question_records
    window.run_id = 1
    window._handle("answer", AnswerEvent("start", 101, "新会话的问题"))
    assert window.question_records["0:101"].question == "问题101"
    assert window.question_records["1:101"].question == "新会话的问题"
    window._handle("answer", AnswerEvent("delta", 101, "新会话的问题", "文" * 25_000))
    assert len(window.question_records["1:101"].answer) == 20_000
    window._destroy()
    assert not window.question_records
