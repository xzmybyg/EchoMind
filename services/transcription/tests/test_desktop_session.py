import asyncio
from dataclasses import replace
import os
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

from echomind.desktop_session import DesktopSession, SessionConfig


class DesktopSessionTests(unittest.TestCase):
    def test_cloud_session_skips_models_vad_and_gpu_and_closes(self):
        from echomind.tencent_asr import TencentConfig
        config = replace(self.config, microphone=True, asr_engine="tencent",
                         tencent=TencentConfig("123", "fake-id", "fake-key", consent=True))
        events = []
        with patch("echomind.desktop_session.TencentTranscriber") as factory, \
             patch("echomind.desktop_session.run_microphone") as capture, \
             patch("echomind.vad.SpeechVAD", side_effect=AssertionError("must not load VAD")), \
             patch.object(Path, "is_file", return_value=False):
            factory.return_value.flush.return_value = []
            session = DesktopSession(config, lambda *event: events.append(event))
            session.start()
            self.assertTrue(session.wait(3))
            self.assertIs(capture.call_args.kwargs["asr_options"]["cloud_transcriber"], factory.return_value)
            factory.return_value.close.assert_called_once()
            self.assertFalse(any(kind == "error" for kind, _ in events))

    def test_cloud_consent_is_required_for_microphone_but_not_text_question(self):
        from echomind.tencent_asr import TencentASRError
        session = DesktopSession(replace(self.config, microphone=True, asr_engine="tencent"), lambda *_: None)
        with self.assertRaisesRegex(TencentASRError, "同意"):
            session.start()
        session.start("React Hooks 原理")
        self.assertTrue(session.wait(3))

    def setUp(self):
        self.config = SessionConfig(Path("missing-capture.exe"), Path("missing-model"), Path("missing-cuda"), answer_mode="demo")

    def test_demo_question_bypasses_audio_and_finishes(self):
        events = []
        with patch("echomind.desktop_session.cli.run") as capture:
            session = DesktopSession(self.config, lambda *event: events.append(event))
            session.start("Redis为什么这么快？")
            self.assertTrue(session.wait(3))
            capture.assert_not_called()
        self.assertEqual(events[-1], ("finished", None))
        self.assertIn("done", [payload.kind for kind, payload in events if kind == "answer"])

    def test_question_threshold_changes_semantic_answer_trigger(self):
        class Gate:
            async def evaluate(self, *_):
                return 0.8

        class Provider:
            async def stream_answer(self, *_):
                yield "回答"

        def capture(model, device, compute, copilot, **kwargs):
            copilot.submit_transcript("缓存一致性展开聊聊", 11)
            self.assertTrue(copilot.wait_idle(2))

        for threshold, should_answer in ((0.65, True), (0.8, True), (0.85, False)):
            with self.subTest(threshold=threshold):
                events = []
                config = replace(self.config, question_threshold=threshold, microphone=True, device="cpu",
                                 answer_mode="live", question_gate="jev", api_key="fake-key", jev_key="fake-jev")
                with patch.object(Path, "is_file", return_value=True), \
                     patch("echomind.desktop_session.run_microphone", side_effect=capture), \
                     patch("echomind.desktop_session.JevQuestionGate", return_value=Gate()), \
                     patch("echomind.desktop_session.OpenAICompatibleProvider", return_value=Provider()):
                    session = DesktopSession(config, lambda *event: events.append(event))
                    session.start()
                    self.assertTrue(session.wait(3))
                self.assertFalse(any(kind == "error" for kind, _ in events))
                self.assertEqual(any(kind == "answer" and event.kind == "done" for kind, event in events), should_answer)

    def test_invalid_question_threshold_is_rejected_before_start(self):
        for value in (-0.01, 1.01, float("nan"), float("inf"), True, "0.85", None):
            with self.subTest(value=value):
                session = DesktopSession(replace(self.config, question_threshold=value), lambda *_: None)
                with self.assertRaisesRegex(ValueError, "阈值"):
                    session.start("问题")
                self.assertFalse(session.is_running)

    def test_audio_session_uses_speech_gate_pause_hotwords_and_retry(self):
        config = replace(self.config, microphone=True, device="cpu", asr_hotwords="React Fiber")
        events = []
        with patch.object(Path, "is_file", return_value=True), patch("echomind.desktop_session.run_microphone") as capture:
            session = DesktopSession(config, lambda *event: events.append(event))
            session.start()
            self.assertTrue(session.wait(3))
        from echomind.vad import SpeechVAD
        options = capture.call_args.kwargs["asr_options"]
        self.assertIsInstance(options["vad"], SpeechVAD)
        self.assertEqual(options["end_silence_ms"], 600)
        self.assertEqual(options["hotwords"], "React Fiber")
        self.assertTrue(options["retry_low_confidence"])
        self.assertEqual(options["context_window_ms"], 15000)
        self.assertFalse(any(kind == "error" for kind, _ in events))

    def test_edited_topic_without_question_marker_generates_answer(self):
        events = []
        session = DesktopSession(self.config, lambda *event: events.append(event))
        session.start("大家好")
        self.assertTrue(session.wait(3))
        self.assertTrue(any(kind == "answer" and payload.kind == "done" and payload.question == "大家好" for kind, payload in events))

    def test_manual_question_uses_current_capture_copilot(self):
        ready = threading.Event()
        events = []

        def run(*args, **kwargs):
            ready.set()
            kwargs["stop_event"].wait(3)

        config = replace(self.config, microphone=True, device="cpu")
        with patch.object(Path, "is_file", return_value=True), patch("echomind.desktop_session.run_microphone", side_effect=run):
            session = DesktopSession(config, lambda *event: events.append(event))
            session.start()
            self.assertTrue(ready.wait(2))
            try:
                session.submit_question("  React Fiber 原理  ")
                self.assertTrue(session._copilot.wait_idle(2))
                self.assertTrue(any(kind == "answer" and payload.kind == "done" and payload.question == "React Fiber 原理" for kind, payload in events))
                with self.assertRaisesRegex(ValueError, "问题内容"):
                    session.submit_question("  ")
            finally:
                session.stop()
                with self.assertRaisesRegex(RuntimeError, "停止"):
                    session.submit_question("主题")
                self.assertTrue(session.wait(3))
            self.assertIsNone(session._copilot)

    def test_manual_question_rejects_stopped_initializing_and_off_states(self):
        session = DesktopSession(self.config, lambda *_: None)
        with self.assertRaisesRegex(RuntimeError, "未运行"):
            session.submit_question("主题")
        session._running = True
        with self.assertRaisesRegex(RuntimeError, "尚未就绪"):
            session.submit_question("主题")
        off = DesktopSession(replace(self.config, answer_mode="off"), lambda *_: None)
        with self.assertRaisesRegex(ValueError, "仅字幕"):
            off.submit_question("主题")

    def test_capture_teardown_does_not_close_copilot_during_submission(self):
        ready = threading.Event()
        finish_capture = threading.Event()
        submitting = threading.Event()
        release_submission = threading.Event()
        closed = threading.Event()
        errors = []

        class FakeCopilot:
            def submit_question(self, text):
                submitting.set()
                release_submission.wait(3)
                if closed.is_set():
                    errors.append("closed during submission")

            def close(self):
                closed.set()

        def capture(*args, **kwargs):
            ready.set()
            finish_capture.wait(3)

        config = replace(self.config, microphone=True, device="cpu")
        with patch.object(Path, "is_file", return_value=True), patch("echomind.desktop_session.Copilot", return_value=FakeCopilot()), patch("echomind.desktop_session.run_microphone", side_effect=capture):
            session = DesktopSession(config, lambda *_: None)
            session.start()
            self.assertTrue(ready.wait(2))
            thread = threading.Thread(target=session.submit_question, args=("主题",))
            thread.start()
            try:
                self.assertTrue(submitting.wait(2))
                finish_capture.set()
                self.assertFalse(closed.wait(0.05))
            finally:
                release_submission.set()
                thread.join(2)
                self.assertTrue(session.wait(3))
            self.assertEqual(errors, [])
            self.assertTrue(closed.is_set())

    def test_invalid_live_configuration_is_synchronous(self):
        session = DesktopSession(replace(self.config, answer_mode="live"), lambda *_: None)
        with self.assertRaisesRegex(ValueError, "API Key"):
            session.start("为什么快？")
        self.assertFalse(session.is_running)

    def test_url_rejects_credentials_without_disclosure(self):
        config = replace(self.config, answer_mode="live", api_key="private", api_base_url="https://user:secret@example.com")
        session = DesktopSession(config, lambda *_: None)
        with self.assertRaises(ValueError) as caught:
            session.start("为什么快？")
        self.assertNotIn("secret", str(caught.exception))
        self.assertNotIn("private", repr(config))

    def test_stop_cancels_running_provider(self):
        began = threading.Event()
        closed = threading.Event()

        class SlowProvider:
            async def stream_answer(self, *_):
                began.set()
                try:
                    await asyncio.sleep(30)
                    yield "late"
                finally:
                    closed.set()

        with patch("echomind.desktop_session.DemoProvider", return_value=SlowProvider()):
            session = DesktopSession(self.config, lambda *_: None)
            session.start("Redis为什么快？")
            self.assertTrue(began.wait(2))
            session.stop()
            self.assertTrue(session.wait(3))
        self.assertTrue(closed.is_set())

    def test_session_rejects_second_start(self):
        ready = threading.Event()

        class SlowProvider:
            async def stream_answer(self, *_):
                ready.set()
                await asyncio.sleep(30)
                yield "late"

        session = DesktopSession(self.config, lambda *_: None)
        with patch("echomind.desktop_session.DemoProvider", return_value=SlowProvider()):
            session.start("Redis为什么快？")
            self.assertTrue(ready.wait(1))
            with self.assertRaisesRegex(ValueError, "尚未停止"):
                session.start("Redis为什么快？")
            session.stop()
            self.assertTrue(session.wait(3))

    def test_audio_routes_callbacks_and_redacts_unexpected_failure(self):
        events = []
        config = replace(self.config, pid=123, device="cpu", answer_mode="off")

        def run(*args, **kwargs):
            kwargs["on_status"]("capturing")
            kwargs["on_transcript"]("transcript")
            raise RuntimeError("fake-private-key")

        with patch.object(Path, "is_file", return_value=True), patch("echomind.desktop_session.cli.run", side_effect=run) as capture:
            session = DesktopSession(config, lambda *event: events.append(event))
            session.start()
            self.assertTrue(session.wait(3))
        self.assertEqual(capture.call_args.args[:1], (123,))
        self.assertIn(("transcript", "transcript"), events)
        self.assertNotIn("fake-private-key", repr(events))
        self.assertEqual(events[-1], ("finished", None))

    def test_microphone_needs_no_pid_or_capture_program(self):
        events = []
        config = replace(self.config, microphone=True, device="cpu", answer_mode="off")
        with patch.object(Path, "is_file", return_value=True), patch("echomind.desktop_session.cli.run") as capture, patch("echomind.desktop_session.run_microphone") as local:
            session = DesktopSession(config, lambda *event: events.append(event))
            session.start()
            self.assertTrue(session.wait(3))
            capture.assert_not_called()
            self.assertEqual(local.call_args.args[0], "missing-model")
            self.assertEqual(local.call_args.args[1], "cpu")
        self.assertEqual(events[-1], ("finished", None))

    def test_microphone_still_requires_local_model(self):
        config = replace(self.config, microphone=True, device="cpu")
        session = DesktopSession(config, lambda *_: None)
        with self.assertRaisesRegex(ValueError, "识别模型"):
            session.start()
        self.assertFalse(session.is_running)

    def test_experimental_engines_run_on_cpu_without_cuda(self):
        for engine in ("sensevoice", "paraformer"):
            config = replace(self.config, microphone=True, device="cuda", asr_engine=engine)
            with patch.object(Path, "is_file", return_value=True), patch("echomind.desktop_session.os.add_dll_directory") as dll, patch("echomind.desktop_session.run_microphone") as capture:
                session = DesktopSession(config, lambda *_: None)
                session.start()
                self.assertTrue(session.wait(3))
            dll.assert_not_called()
            self.assertEqual(capture.call_args.args[1:3], ("cpu", "int8"))
            self.assertEqual(capture.call_args.kwargs["asr_options"]["engine"], engine)

    def test_microphone_errors_are_actionable(self):
        from echomind.microphone import MicrophoneError
        config = replace(self.config, microphone=True, device="cpu", answer_mode="off")
        events = []
        with patch.object(Path, "is_file", return_value=True), patch("echomind.desktop_session.run_microphone", side_effect=MicrophoneError("无法打开麦克风，请检查权限")):
            session = DesktopSession(config, lambda *event: events.append(event))
            session.start()
            self.assertTrue(session.wait(3))
        self.assertIn(("error", "无法打开麦克风，请检查权限"), events)

    def test_provider_uses_memory_key_without_environment_mutation(self):
        events = []
        config = replace(self.config, answer_mode="live", api_key="fake-private-key")
        with patch.dict(os.environ, {"ECHOMIND_LLM_API_KEY": "unchanged"}), patch("echomind.desktop_session.OpenAICompatibleProvider") as provider:
            from echomind.llm import DemoProvider
            provider.return_value = DemoProvider()
            session = DesktopSession(config, lambda *event: events.append(event))
            session.start("Redis为什么快？")
            self.assertTrue(session.wait(3))
            self.assertEqual(os.environ["ECHOMIND_LLM_API_KEY"], "unchanged")
            self.assertEqual(provider.call_args.args[2], "fake-private-key")

    def test_live_reviser_reuses_model_configuration_without_network_on_manual_submit(self):
        from echomind.llm import DemoProvider
        config = replace(self.config, answer_mode="live", api_key="fake-private-key", api_model="custom-model")
        with patch("echomind.desktop_session.OpenAICompatibleProvider", return_value=DemoProvider()), patch("echomind.desktop_session.OpenAICompatibleQuestionReviser") as reviser:
            session = DesktopSession(config, lambda *_: None)
            session.start("Redis为什么快")
            self.assertTrue(session.wait(3))
            reviser.assert_called_once_with(config.api_base_url, "custom-model", "fake-private-key")
            reviser.return_value.review.assert_not_called()

    def test_demo_never_instantiates_remote_reviser(self):
        with patch("echomind.desktop_session.OpenAICompatibleQuestionReviser") as reviser:
            session = DesktopSession(self.config, lambda *_: None)
            session.start("Redis为什么快")
            self.assertTrue(session.wait(3))
            reviser.assert_not_called()


if __name__ == "__main__":
    unittest.main()
