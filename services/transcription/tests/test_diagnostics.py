from unittest.mock import patch

from echomind import diagnostics


def test_fault_log_is_local_bounded_and_contains_no_application_state(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, '_fault_file', None)
    path = tmp_path / 'native-crash.log'
    path.write_text('old stack\n' * 20_000)
    with patch.object(diagnostics.faulthandler, 'enable') as enable:
        diagnostics.enable_fault_log(tmp_path)
        diagnostics.enable_fault_log(tmp_path)
        enable.assert_called_once_with(file=diagnostics._fault_file, all_threads=True)
    diagnostics._fault_file.close()
    assert (tmp_path / 'native-crash.previous.log').is_file()
    assert path.read_text().startswith('\nEchoMind start ')
    assert len(path.read_bytes()) < 200


def test_unwritable_log_never_prevents_startup(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, '_fault_file', None)
    blocked = tmp_path / 'file'
    blocked.write_text('not a directory')
    diagnostics.enable_fault_log(blocked / 'logs')
    assert diagnostics._fault_file is None
