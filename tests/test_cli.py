from __future__ import annotations

import json
import io
import os
import subprocess
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "transcription"))

from echomind import cli


class CliTests(unittest.TestCase):
    @patch("echomind.cli.subprocess.run")
    def test_meeting_processes_accepts_single_json_object(self, run):
        run.return_value = SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"ProcessId": 42, "Name": "WeMeetApp.exe", "ExecutablePath": "C:\\WeMeetApp.exe"}),
            stderr="",
        )
        self.assertEqual(cli.meeting_processes()[0]["ProcessId"], 42)

    @patch("builtins.input", return_value="2")
    def test_choose_pid_by_number(self, _input):
        processes = [{"ProcessId": 42, "Name": "a"}, {"ProcessId": 99, "Name": "b"}]
        self.assertEqual(cli.choose_pid(processes), 99)

    def test_missing_capture_exe_reports_error(self):
        self.assertEqual(cli.main(["--pid", "42", "--capture-exe", "missing.exe"]), 1)

    @patch("echomind.cli.subprocess.Popen")
    def test_text_demo_needs_no_capture(self, popen):
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(cli.main(["--answers", "demo", "--text-question", "Redis为什么这么快？"]), 0)
        self.assertIn("静态演练", output.getvalue())
        self.assertIn("模型首字", output.getvalue())
        popen.assert_not_called()

    def test_live_configuration_errors_before_capture(self):
        with patch.dict(os.environ, {}, clear=True), patch("echomind.cli.subprocess.Popen") as popen:
            self.assertEqual(cli.main(["--answers", "live", "--text-question", "如何使用Redis？"]), 1)
        popen.assert_not_called()

    def test_text_demo_reports_nonquestion(self):
        self.assertEqual(cli.main(["--answers", "demo", "--text-question", "大家好"]), 1)

    def test_jev_mode_requires_its_own_key_before_capture(self):
        from echomind.llm import DemoProvider
        with patch.dict(os.environ, {}, clear=True), patch("echomind.llm.OpenAICompatibleProvider.from_env", return_value=DemoProvider()):
            with patch("sys.stderr", new_callable=io.StringIO) as output:
                self.assertEqual(cli.main(["--answers", "live", "--question-gate", "jev",
                                           "--text-question", "Redis为什么这么快？"]), 1)
        self.assertIn("ECHOMIND_JEV_API_KEY", output.getvalue())

    def test_semantic_rejection_is_successful_no_answer_test(self):
        from echomind.llm import DemoProvider

        class Gate:
            async def evaluate(self, text, recent):
                return 0.3

        with patch("echomind.llm.OpenAICompatibleProvider.from_env", return_value=DemoProvider()), patch("echomind.intent.JevQuestionGate.from_env", return_value=Gate()):
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(cli.main(["--answers", "live", "--question-gate", "jev",
                                           "--text-question", "继续聊聊缓存"]), 0)
        self.assertIn("[忽略]", output.getvalue())
        self.assertNotIn("[回答]", output.getvalue())

    def test_transcripts_are_routed_without_waiting(self):
        copilot = Mock()
        events = [SimpleNamespace(kind=kind, text="为什么", timestamp_ms=10, latency_ms=2, suspect=True)
                  for kind in ("partial", "final")]
        cli.show_events(events, copilot)
        self.assertEqual(copilot.submit_transcript.call_count, 2)
        self.assertEqual(copilot.submit_transcript.call_args_list[0].kwargs, {"final": False, "suspect": True})
        self.assertEqual(copilot.submit_transcript.call_args_list[1].kwargs, {"final": True, "suspect": True})
        copilot.wait_idle.assert_not_called()

    def test_suspect_question_prints_confirmation_instead_of_false_failure(self):
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(cli.main(["--answers", "demo", "--text-question", "1加1等于进。"]), 0)
        self.assertIn("[需确认]", output.getvalue())
        self.assertIn("1加1等于几？", output.getvalue())
        self.assertNotIn("[回答]", output.getvalue())

    def test_text_provider_error_returns_failure(self):
        class FailingProvider:
            async def stream_answer(self, question, context):
                raise RuntimeError("secret")
                yield

        with patch("echomind.llm.OpenAICompatibleProvider.from_env", return_value=FailingProvider()):
            with patch("sys.stderr", new_callable=io.StringIO) as output:
                self.assertEqual(cli.main(["--answers", "live", "--text-question", "Redis为什么这么快？"]), 1)
        self.assertNotIn("secret", output.getvalue())

    def test_live_text_mode_through_local_streaming_http(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, body))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.end_headers()
                for text in ("1. 内存访问。", "2. 高效数据结构。"):
                    data = json.dumps({"choices": [{"delta": {"content": text}}]}, ensure_ascii=False)
                    self.wfile.write(("data: " + data + "\n\n").encode("utf-8"))
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            config = {
                "ECHOMIND_LLM_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                "ECHOMIND_LLM_MODEL": "local-test-model",
                "ECHOMIND_LLM_API_KEY": "fake-local-test-key",
                "ECHOMIND_LLM_THINKING": "disabled",
            }
            with patch.dict(os.environ, config), patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(cli.main(["--answers", "live", "--text-question", "Redis为什么这么快？"]), 0)
            self.assertIn("1. 内存访问。2. 高效数据结构。", output.getvalue())
            self.assertIn("[完成 #1]", output.getvalue())
            self.assertEqual(requests[0][0], "/v1/chat/completions")
            self.assertEqual(requests[0][1]["thinking"], {"type": "disabled"})
            self.assertEqual(requests[0][1]["messages"][-1]["content"], "Redis为什么这么快？")
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)

    @patch("echomind.asr.Transcriber")
    @patch("echomind.cli.subprocess.Popen")
    def test_stop_during_warmup_never_starts_capture(self, popen, transcriber):
        stopping = threading.Event()
        transcriber.return_value.warm_up.side_effect = stopping.set
        self.assertEqual(cli.run(42, Path(__file__), "test", "cpu", "default",
                                 stop_event=stopping, on_status=lambda _: None), 0)
        popen.assert_not_called()

    @patch("echomind.asr.Transcriber")
    @patch("echomind.cli.subprocess.Popen")
    def test_gui_capture_callbacks_and_stop_do_not_print(self, popen, transcriber):
        child = Mock()
        child.stdout = io.BytesIO(b"\x00\x00" * 1600)
        child.stderr = io.BytesIO(b"diagnostic")
        child.poll.return_value = None
        child.wait.return_value = -1
        popen.return_value = child
        event = SimpleNamespace(kind="final", text="Redis为什么快", timestamp_ms=1, latency_ms=1, suspect=False)
        transcriber.return_value.push_pcm.return_value = [event]
        stopping = threading.Event()
        transcripts = []

        def on_transcript(value):
            transcripts.append(value)
            stopping.set()

        with patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(cli.run(42, Path(__file__), "test", "cpu", "default", stop_event=stopping,
                                     on_transcript=on_transcript, on_status=lambda _: None), 0)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(transcripts, [event])
        child.terminate.assert_called_once()
        transcriber.return_value.flush.assert_not_called()

    @patch("echomind.asr.Transcriber")
    @patch("echomind.cli.subprocess.Popen")
    def test_capture_pcm_reaches_asr_and_flushes(self, popen, transcriber):
        child = Mock()
        child.stdout = io.BytesIO(b"\x00\x00" * 1600)
        child.wait.return_value = 0
        child.poll.return_value = 0
        popen.return_value = child
        transcriber.return_value.push_pcm.return_value = []
        transcriber.return_value.flush.return_value = []

        self.assertEqual(cli.run(42, Path(__file__), "test", "cpu", "default"), 0)

        self.assertEqual(transcriber.return_value.push_pcm.call_args.args[0], b"\x00\x00" * 1600)
        transcriber.return_value.flush.assert_called_once()
        popen.assert_called_once_with(
            [str(Path(__file__)), "--pid", "42"],
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=cli.PCM_CHUNK_BYTES * 4,
        )


if __name__ == "__main__":
    unittest.main()
