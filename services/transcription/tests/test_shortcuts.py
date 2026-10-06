import pytest

from echomind.shortcuts import DEFAULT_SHORTCUTS, shortcut_sequence, validate_shortcuts
from echomind.settings import SettingsError, load_settings, save_settings


def test_default_shortcut_sequences():
    assert validate_shortcuts({}) == {"shortcut_submit": "<Return>", "shortcut_start": "<Control-r>", "shortcut_stop": "<Control-s>"}


@pytest.mark.parametrize("value,expected", [(" Ctrl+Alt+R ", "<Control-Alt-r>"), ("Alt+Shift+F8", "<Alt-Shift-F8>"), ("Ctrl+Enter", "<Control-Return>"), ("F12", "<F12>"), ("Ctrl+Shift+R", "<Control-Shift-R>")])
def test_custom_shortcut_format(value, expected):
    assert shortcut_sequence(value) == expected


@pytest.mark.parametrize("values", [
    {"shortcut_stop": "Ctrl+R"}, {"shortcut_start": "r"}, {"shortcut_start": "Enter"},
    {"shortcut_submit": "Ctrl+V"}, {"shortcut_stop": "Ctrl+Enter"}, {"shortcut_start": ""},
    {"shortcut_start": "Ctrl+Ctrl+R"}, {"shortcut_submit": "Alt+F4"},
])
def test_conflicting_or_unsafe_shortcuts_are_rejected(values):
    with pytest.raises(ValueError):
        validate_shortcuts(values)


def test_shortcuts_roundtrip_and_invalid_settings_preserve_file(tmp_path):
    path = tmp_path / "settings.json"
    values = dict(DEFAULT_SHORTCUTS, shortcut_start="F8", shortcut_stop="F9")
    save_settings(values, path)
    assert load_settings(path) == values
    original = path.read_bytes()
    with pytest.raises(SettingsError):
        save_settings(dict(values, shortcut_stop="F8"), path)
    assert path.read_bytes() == original
