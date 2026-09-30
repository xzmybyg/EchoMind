from __future__ import annotations

import json
import io
import subprocess
import unittest
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
