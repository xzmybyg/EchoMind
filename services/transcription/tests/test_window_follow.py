from echomind.window_follow import Window, WindowFollower, dock_position, process_family
import os
import subprocess
import sys

import pytest


def test_selected_audio_child_includes_ancestor_not_unrelated_meeting():
    rows = [{"ProcessId": 1, "ParentProcessId": 999}, {"ProcessId": 2, "ParentProcessId": 1},
            {"ProcessId": 3, "ParentProcessId": 2}, {"ProcessId": 4, "ParentProcessId": 999}]
    assert process_family(2, rows) == {1, 2, 3}
    assert process_family(1, rows) == {1, 2, 3}
    assert process_family(88, rows) == {88}


def test_process_family_handles_cycle():
    assert process_family(1, [{"ProcessId": 1, "ParentProcessId": 2}, {"ProcessId": 2, "ParentProcessId": 1}]) == {1, 2}


def test_dock_left_right_and_other_side_fallback():
    work = (0, 0, 2560, 1400)
    assert dock_position((800, 100, 1600, 900), (600, 600), work, "left") == (192, 100, False, False)
    assert dock_position((800, 100, 1600, 900), (600, 600), work, "right") == (1608, 100, False, False)
    assert dock_position((1400, 100, 2400, 900), (600, 600), work, "right") == (792, 100, True, False)


def test_dock_clamps_to_negative_monitor_and_taskbar_work_area():
    assert dock_position((-1920, -200, -920, 500), (800, 600), (-1920, 0, 0, 1040), "left") == (-912, 0, True, False)
    assert dock_position((0, 0, 1920, 1080), (1320, 820), (0, 0, 1920, 1040), "right") == (600, 0, False, True)
    assert dock_position((0, 0, 600, 400), (1320, 820), (0, 0, 800, 600), "left") == (0, 0, False, True)


class API:
    def __init__(self):
        self.target = Window(20, 2, (800, 100, 1600, 900))
        self.is_minimized = False
        self.is_maximized = False
        self.calls = []
        self.own_rect = (0, 0, 600, 600)

    def find(self, pid, pids, own, previous):
        self.calls.append(("find", pid, pids, previous))
        return self.target

    def minimized(self, own):
        return self.is_minimized

    def maximized(self, own):
        return self.is_maximized

    def minimize(self, own):
        self.calls.append(("minimize", own))
        self.is_minimized = True

    def restore(self, own):
        self.calls.append(("restore", own))
        self.is_minimized = False

    def rect(self, own):
        return self.own_rect

    def work_area(self, handle):
        return (0, 0, 2560, 1400)

    def move(self, own, x, y):
        self.calls.append(("move", own, x, y))
        self.own_rect = (x, y, x + 600, y + 600)


def tick(follower, **kwargs):
    return follower.tick(kwargs.pop("pid", 2), {2}, "right", enabled=kwargs.pop("enabled", True), **kwargs)


def test_follow_move_and_no_redundant_moves():
    api = API()
    follower = WindowFollower(api, 10)
    tick(follower)
    tick(follower)
    assert [c for c in api.calls if c[0] == "move"] == [("move", 10, 1608, 100)]
    api.target = Window(20, 2, (700, 200, 1500, 900))
    tick(follower)
    assert api.calls[-1] == ("move", 10, 1508, 200)
    assert api.calls[-2][-1] == 20


def test_minimize_once_restore_once_without_activating_target():
    api = API()
    follower = WindowFollower(api, 10)
    api.target = Window(20, 2, (800, 100, 1600, 900), True)
    tick(follower)
    tick(follower)
    assert [c for c in api.calls if c[0] == "minimize"] == [("minimize", 10)]
    api.target = Window(20, 2, (800, 100, 1600, 900))
    tick(follower)
    tick(follower)
    assert [c for c in api.calls if c[0] == "restore"] == [("restore", 10)]


def test_manual_minimize_never_auto_restores():
    api = API()
    follower = WindowFollower(api, 10)
    api.is_minimized = True
    assert "手动最小化" in tick(follower)
    api.target = Window(20, 2, api.target.rect, True)
    tick(follower)
    api.target = Window(20, 2, api.target.rect)
    tick(follower)
    tick(follower, enabled=False)
    assert not any(c[0] in ("restore", "move") for c in api.calls)


def test_missing_target_and_disable_restore_only_follow_minimization():
    api = API()
    follower = WindowFollower(api, 10)
    api.target = Window(20, 2, api.target.rect, True)
    tick(follower)
    api.target = None
    assert "等待" in tick(follower)
    assert api.calls[-1] == ("restore", 10)
    assert follower.target is None
    tick(follower, enabled=False)
    assert not api.is_minimized


def test_paused_dialog_and_maximized_app_never_move():
    api = API()
    follower = WindowFollower(api, 10)
    tick(follower, paused=True)
    api.is_maximized = True
    assert "还原" in tick(follower)
    assert not any(c[0] == "move" for c in api.calls)


def test_switching_process_does_not_reuse_old_window():
    api = API()
    follower = WindowFollower(api, 10)
    tick(follower)
    tick(follower, pid=3)
    assert api.calls[-1][0] == "find"
    assert api.calls[-1][-1] is None


@pytest.mark.skipif(os.name != "nt", reason="Windows native window integration")
def test_native_window_backend_in_isolated_process():
    # Operate only two synthetic windows owned by this isolated test process.
    script = '''
import os
import tkinter as tk
from echomind.window_follow import WindowsAPI
target = tk.Tk()
own = tk.Tk()
try:
    target.title("EchoMind test target")
    target.geometry("500x300+100+100")
    own.title("EchoMind test follower")
    own.geometry("400x250+700+100")
    target.update()
    own.update()
    api = WindowsAPI()
    target_handle = api.root_handle(target.winfo_id())
    own_handle = api.root_handle(own.winfo_id())
    found = api.find(os.getpid(), {os.getpid()}, own_handle)
    assert found.handle == target_handle
    assert api.find(999999999, {999999999}, own_handle) is None
    assert len(api.work_area(target_handle)) == 4
    api.move(own_handle, 600, 150)
    assert api.rect(own_handle)[:2] == (600, 150)
    api.minimize(own_handle)
    assert api.minimized(own_handle)
    api.restore(own_handle)
    assert not api.minimized(own_handle)
finally:
    own.destroy()
    target.destroy()
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            timeout=20, creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode == 0, result.stderr
