import json
import subprocess
from types import SimpleNamespace

import pytest

from echomind.cli import meeting_processes


@pytest.mark.parametrize('payload', ['', json.dumps({'ProcessId':42, 'ParentProcessId':1, 'Name':'wemeetapp.exe'}), json.dumps([
    {'ProcessId':50, 'Name':'wemeetapp.exe'}, {'ProcessId':10, 'Name':'WeMeetCrashHandler.exe'}])])
def test_discovery_handles_empty_single_and_multiple_json(monkeypatch, payload):
    calls = []
    def run(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(returncode=0, stdout='\ufeff' + payload if payload else '', stderr='')
    monkeypatch.setattr(subprocess, 'run', run)
    rows = meeting_processes()
    assert [r['ProcessId'] for r in rows] == sorted(r['ProcessId'] for r in rows)
    assert len(rows) == (0 if not payload else 2 if payload.startswith('[') else 1)
    assert calls[0]['timeout'] == 8


def test_stalled_process_query_is_bounded(monkeypatch):
    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs['timeout'])
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(subprocess.TimeoutExpired):
        meeting_processes()
