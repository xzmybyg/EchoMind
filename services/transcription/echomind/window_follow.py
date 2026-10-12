"""Follow a selected process's window without changing it or stealing focus."""

import ctypes
from ctypes import wintypes as w
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Window:
    handle: int
    pid: int
    rect: tuple[int, int, int, int]
    minimized: bool = False


def process_family(pid, processes):
    """Only the selected branch and its ancestors, never unrelated meetings."""
    parents = {int(p["ProcessId"]): int(p.get("ParentProcessId") or 0) for p in processes}
    family = {pid}
    # Audio may belong to a child while the window belongs to its parent.
    parent = parents.get(pid)
    while parent in parents and parent not in family:
        family.add(parent)
        parent = parents.get(parent)
    descendants = {pid}
    while True:
        added = {child for child, parent in parents.items() if parent in descendants} - descendants
        if not added:
            break
        descendants.update(added)
    return family | descendants


def dock_position(target, size, work, side, gap=8):
    left, top, right, bottom = work
    width, height = size
    preferred = target[2] + gap if side == "right" else target[0] - width - gap
    alternate = target[0] - width - gap if side == "right" else target[2] + gap
    fits = lambda x: left <= x and x + width <= right
    flipped = not fits(preferred) and fits(alternate)
    x = alternate if flipped else preferred
    clipped = not fits(x)
    return (max(left, min(x, max(left, right - width))),
            max(top, min(target[1], max(top, bottom - height))), flipped, clipped)


class WindowsAPI:
    def __init__(self):
        if os.name != "nt":
            raise OSError("窗口跟随仅支持 Windows。")
        self.dll = ctypes.WinDLL("user32", use_last_error=True)
        self.callback = ctypes.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
        signatures = {
            "EnumWindows": ([self.callback, w.LPARAM], w.BOOL),
            "GetWindowThreadProcessId": ([w.HWND, ctypes.POINTER(w.DWORD)], w.DWORD),
            "IsWindowVisible": ([w.HWND], w.BOOL), "IsIconic": ([w.HWND], w.BOOL),
            "IsZoomed": ([w.HWND], w.BOOL), "GetWindow": ([w.HWND, w.UINT], w.HWND),
            "GetWindowTextLengthW": ([w.HWND], ctypes.c_int),
            "GetWindowLongW": ([w.HWND, ctypes.c_int], w.LONG),
            "GetWindowRect": ([w.HWND, ctypes.POINTER(w.RECT)], w.BOOL),
            "GetAncestor": ([w.HWND, w.UINT], w.HWND),
            "MonitorFromWindow": ([w.HWND, w.DWORD], w.HANDLE),
            "GetMonitorInfoW": ([w.HANDLE, ctypes.c_void_p], w.BOOL),
            "ShowWindow": ([w.HWND, ctypes.c_int], w.BOOL),
            "SetWindowPos": ([w.HWND, w.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, w.UINT], w.BOOL),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.dll, name)
            function.argtypes, function.restype = arguments, result

    def root_handle(self, handle):
        return self.dll.GetAncestor(handle, 2)

    def rect(self, handle):
        rect = w.RECT()
        if not self.dll.GetWindowRect(handle, ctypes.byref(rect)):
            raise OSError("无法读取窗口位置。")
        return rect.left, rect.top, rect.right, rect.bottom

    def snapshot(self, handle, pids):
        pid = w.DWORD()
        self.dll.GetWindowThreadProcessId(handle, ctypes.byref(pid))
        if pid.value not in pids or not self.dll.IsWindowVisible(handle):
            return None
        return Window(handle, pid.value, self.rect(handle), bool(self.dll.IsIconic(handle)))

    def find(self, pid, pids, own, previous=None):
        if previous:
            existing = self.snapshot(previous, pids)
            if existing:
                return existing
        candidates = []

        def inspect(handle, _):
            try:
                if handle == own or self.dll.GetWindow(handle, 4) or self.dll.GetWindowLongW(handle, -20) & 0x80:
                    return True
                if not self.dll.GetWindowTextLengthW(handle):
                    return True
                window = self.snapshot(handle, pids)
                if window:
                    rect = window.rect
                    if window.minimized or (rect[2] - rect[0] >= 200 and rect[3] - rect[1] >= 120):
                        candidates.append(window)
            except OSError:
                pass  # Window disappeared during enumeration.
            return True

        if not self.dll.EnumWindows(self.callback(inspect), 0):
            raise OSError("无法查找会议窗口。")
        return max(candidates, key=lambda item: (item.pid == pid,
                   (item.rect[2] - item.rect[0]) * (item.rect[3] - item.rect[1])), default=None)

    def work_area(self, handle):
        class MonitorInfo(ctypes.Structure):
            _fields_ = [("size", w.DWORD), ("monitor", w.RECT), ("work", w.RECT), ("flags", w.DWORD)]
        info = MonitorInfo()
        info.size = ctypes.sizeof(info)
        monitor = self.dll.MonitorFromWindow(handle, 2)
        if not self.dll.GetMonitorInfoW(monitor, ctypes.byref(info)):
            raise OSError("无法读取显示器工作区域。")
        return info.work.left, info.work.top, info.work.right, info.work.bottom

    def minimized(self, handle):
        return bool(self.dll.IsIconic(handle))

    def maximized(self, handle):
        return bool(self.dll.IsZoomed(handle))

    def minimize(self, handle):
        self.dll.ShowWindow(handle, 7)  # SW_SHOWMINNOACTIVE

    def restore(self, handle):
        self.dll.ShowWindow(handle, 4)  # SW_SHOWNOACTIVATE

    def move(self, handle, x, y):
        if not self.dll.SetWindowPos(handle, None, x, y, 0, 0, 0x15):
            raise OSError("无法移动 EchoMind 窗口。")


class WindowFollower:
    def __init__(self, api, own):
        self.api, self.own = api, own
        self.pid = None
        self.target = None
        self.minimized_by_follow = False

    def detach(self):
        if self.minimized_by_follow:
            self.api.restore(self.own)
        self.minimized_by_follow = False
        self.pid = self.target = None

    def tick(self, pid, pids, side, *, enabled, paused=False):
        if not enabled or not pid:
            self.detach()
            return "窗口跟随已关闭" if not enabled else "窗口跟随：请选择会议进程"
        if self.pid != pid:
            self.detach()
            self.pid = pid
        target = self.api.find(pid, pids, self.own, self.target)
        if target is None:
            self.detach()
            return "窗口跟随：等待所选进程的会议窗口"
        self.target = target.handle
        if paused:
            return "窗口跟随：弹窗操作期间暂停"
        if target.minimized:
            if not self.api.minimized(self.own):
                self.api.minimize(self.own)
                self.minimized_by_follow = True
            return "窗口跟随：会议已最小化"
        if self.minimized_by_follow:
            self.api.restore(self.own)
            self.minimized_by_follow = False
        if self.api.minimized(self.own):
            return "窗口跟随：EchoMind 已手动最小化"
        if self.api.maximized(self.own):
            return "窗口跟随：请还原 EchoMind 窗口后停靠"
        rect = self.api.rect(self.own)
        x, y, flipped, clipped = dock_position(target.rect, (rect[2] - rect[0], rect[3] - rect[1]),
                                              self.api.work_area(target.handle), side)
        if (x, y) != rect[:2]:
            self.api.move(self.own, x, y)
        label = "左侧" if (side == "left") != flipped else "右侧"
        return f"窗口跟随：{label}" + ("（空间不足，可能重叠）" if clipped else "")
