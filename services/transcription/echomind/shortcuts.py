"""Validate a small set of window-local shortcuts, without global hotkeys."""

import re

DEFAULT_SHORTCUTS = {"shortcut_submit": "Enter", "shortcut_start": "Ctrl+R", "shortcut_stop": "Ctrl+S"}


def shortcut_sequence(value: str) -> str:
    parts = value.strip().lower().split("+")
    modifiers, key = parts[:-1], parts[-1]
    if len(set(modifiers)) != len(modifiers) or any(item not in ("ctrl", "alt", "shift") for item in modifiers):
        raise ValueError("快捷键格式如 Enter、Ctrl+R、Alt+S、F8；修饰键不能重复。")
    if key == "enter":
        key = "Return"
    elif re.fullmatch(r"f(?:[1-9]|1[0-2])", key):
        key = key.upper()
    elif not re.fullmatch(r"[a-z0-9]", key) or not set(modifiers) & {"ctrl", "alt"}:
        raise ValueError("字母或数字快捷键需搭配 Ctrl 或 Alt；也可使用 Enter、F1–F12。")
    elif "shift" in modifiers:
        key = key.upper()
    tokens = [name for item, name in (("ctrl", "Control"), ("alt", "Alt"), ("shift", "Shift")) if item in modifiers]
    return "<" + "-".join(tokens + [key]) + ">"


def validate_shortcuts(values) -> dict[str, str]:
    sequences = {key: shortcut_sequence(values.get(key, default)) for key, default in DEFAULT_SHORTCUTS.items()}
    if len(set(sequences.values())) != len(sequences):
        raise ValueError("提交、开始、停止的快捷键不能重复。")
    if set(sequences.values()) & {f"<Control-{key}>" for key in "acvxzy"} or "<Alt-F4>" in sequences.values():
        raise ValueError("请保留复制、粘贴、剪切、全选、撤销和关闭窗口快捷键，选择其他键位。")
    if any(sequences[key] == "<Control-Return>" for key in ("shortcut_start", "shortcut_stop")):
        raise ValueError("Ctrl+Enter 已用于字幕加入问题，请选择其他快捷键。")
    if any(sequences[key] in ("<Return>", "<Shift-Return>") for key in ("shortcut_start", "shortcut_stop")):
        raise ValueError("开始和停止请使用组合键或功能键，避免输入时误触。")
    return sequences
