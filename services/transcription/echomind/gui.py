"""Windowed EchoMind MVP. Widgets stay on the Tk thread."""

from __future__ import annotations

import os
import math
import queue
import re
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

from .cli import meeting_processes
from .asr import DEFAULT_HOTWORDS, PREVIOUS_DEFAULT_HOTWORDS, LEGACY_DEFAULT_HOTWORDS
from .asr_backends import ENGINE_LABELS, experimental_model_path, validate_model
from .desktop_session import DesktopSession, SessionConfig
from .tencent_asr import TencentConfig
from .components import (ComponentError, install_components,
                         missing, resolve_paths)
from .settings import SettingsError, config_path, load_settings, save_settings
from .screenshot import ScreenshotError, clipboard_has_image, recognize_clipboard, recognize_screenshot
from .shortcuts import DEFAULT_SHORTCUTS, validate_shortcuts
from .updater import UpdateError, check_release, launch_installer, stage_update
from .version import VERSION
from .window_follow import WindowsAPI, WindowFollower, process_family
from .gui_icons import navigation_icon


COLORS = {"background": "#F5F5F5", "surface": "#FFFFFF", "text": "#141414",
          "muted": "#595959", "primary": "#1677FF", "border": "#D9D9D9",
          "tint": "#E6F4FF", "active": "#0958D9", "error": "#B42318",
          "control_border": "#8C8C8C", "disabled": "#737373", "table_header": "#FAFAFA",
          "pressed": "#003EB3", "rail": "#F3F0EB", "rail_hover": "#E9E5DF",
          "rail_selected": "#E0DCD5"}
ANSWER_MODES = {"仅字幕": "off", "离线演练": "demo", "DeepSeek 在线回答": "live"}
DEVICE_MODES = {"GPU（推荐）": "cuda", "CPU（较慢）": "cpu"}
MODE_HINTS = {"仅字幕": "仅显示字幕，不生成回答。是否上传音频取决于“语音”页选择的识别引擎。",
              "离线演练": "检测到问题后展示固定示例，不是真实 AI 回答。",
              "DeepSeek 在线回答": "检测到问题后调用在线模型，请先在“模型”页填写密钥。"}


@dataclass
class QuestionRecord:
    question: str
    timestamp: str
    answer: str = ""
    status: str = "生成中"
    metrics: str = "正在生成回答…"


def asset_paths() -> tuple[Path, Path, Path]:
    if getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent
        model, cuda = resolve_paths(root)
        return root / "capture" / "EchoMind.Capture.exe", model, cuda
    root = Path(__file__).resolve().parents[3]
    return (root / "native/audio-capture/bin/Release/net9.0/EchoMind.Capture.exe",
            root / ".models/large-v3-turbo", root / ".runtime/cuda12")


def preferred_pid(processes: list[dict]) -> int | None:
    ids = {int(item["ProcessId"]) for item in processes}
    mains = [item for item in processes if str(item["Name"]).lower() == "wemeetapp.exe"]
    roots = [item for item in mains if item.get("ParentProcessId") not in ids]
    selected = (roots or mains)
    return int(selected[0]["ProcessId"]) if selected else None


class EchoMindWindow:
    def __init__(self, root: tk.Tk, *, session_factory=DesktopSession, discover=meeting_processes, settings_path=None):
        self.root = root
        self.session_factory = session_factory
        self.discover = discover
        self.events: queue.Queue = queue.Queue()
        self.session = None
        self.run_id = 0
        self.closing = False
        self.destroyed = False
        self.refreshing = False
        self.importing_screenshot = False
        self.screenshot_notice = tk.StringVar(value="截图后 Ctrl+V 粘贴；本机识别，核对后提交。")
        self.processes: dict[str, int] = {}
        self.process_details = []
        self.application_notice = tk.StringVar(value="自动选择主进程 · 请启动腾讯会议后刷新")
        self.follow_enabled = tk.BooleanVar(value=False)
        self.follow_side = tk.StringVar(value="right")
        self.follow_notice = tk.StringVar(value="窗口跟随已关闭")
        self.follower = None
        self.follow_dialog = None
        self.compact_mode = False
        self.full_window_geometry = None
        self.answer_id = None
        self.question_records: dict[str, QuestionRecord] = {}
        self.history_question = tk.StringVar(value="暂无问题记录，检测到问题并触发回答后会显示在这里。")
        self.history_status = tk.StringVar(value="仅保留本次运行最近 100 条；退出后清空，不写入磁盘。")
        self.form_controls = []
        self.settings_path = Path(settings_path) if settings_path is not None else config_path()
        settings_error = ""
        try:
            saved = load_settings(self.settings_path)
            settings_notice = "已读取本地配置" if saved else "首次填写后点击保存，下次启动自动读取。"
        except SettingsError as exc:
            saved = {}
            settings_notice = str(exc)
            settings_error = str(exc)
        self.config_notice = tk.StringVar(value=settings_notice)
        self.follow_enabled.set(saved.get("follow_enabled", False))
        self.follow_side.set(saved.get("follow_side", "right"))
        self.capture_exe, self.model_path, self.cuda_dir = asset_paths()
        self.status = tk.StringVar(value="就绪 · 请选择会议进程，或先测试回答")
        self.error = tk.StringVar(value=settings_error)
        self.partial = tk.StringVar(value="等待会议声音；当前不直接采集你的麦克风。")
        self.question = tk.StringVar(value="等待一个完整问题")
        self.metrics = tk.StringVar(value="模型首字耗时不包含语音识别和停顿等待")
        self.detection = tk.StringVar()
        self.repair_notice = tk.StringVar()
        self.pending_repair = ""
        self.process = tk.StringVar()
        self.source = tk.StringVar(value=saved.get("source", "腾讯会议"))
        self.mode = tk.StringVar(value=next((label for label, value in ANSWER_MODES.items() if value == saved.get("answer_mode")), "仅字幕"))
        self.mode_hint = tk.StringVar(value=MODE_HINTS[self.mode.get()])
        self.mode.trace_add("write", lambda *_: self._mode_changed())
        self.device = tk.StringVar(value=next((label for label, value in DEVICE_MODES.items() if value == saved.get("device")), "GPU（推荐）"))
        hotwords = saved.get("asr_hotwords", DEFAULT_HOTWORDS)
        if hotwords in (PREVIOUS_DEFAULT_HOTWORDS, LEGACY_DEFAULT_HOTWORDS):
            hotwords = DEFAULT_HOTWORDS
        self.asr_hotwords = tk.StringVar(value=hotwords)
        self.asr_topic = tk.StringVar(value=self._asr_topic_for(hotwords))
        self.asr_hotwords.trace_add("write", lambda *_: self.asr_topic.set(self._asr_topic_for(self.asr_hotwords.get())))
        self.asr_model_path = tk.StringVar(value=saved.get("asr_model_path", ""))
        self.asr_engine = tk.StringVar(value=ENGINE_LABELS[saved.get("asr_engine", "whisper")])
        self.asr_engine_hint = tk.StringVar(value=self._engine_hint())
        self.tencent_fields = {key: tk.StringVar(value=saved.get(key, default)) for key, default in (
            ("tencent_app_id", ""), ("tencent_secret_id", ""), ("tencent_secret_key", ""), ("tencent_engine", "16k_zh_en"))}
        self.cloud_consent = tk.BooleanVar(value=False)
        self.caption_source = tk.StringVar(value="实时转写 / 腾讯云识别（上传音频）" if self._engine() == "tencent" else "实时转写 / 本机识别")
        self.api_url = tk.StringVar(value=os.environ.get("ECHOMIND_LLM_BASE_URL", saved.get("api_base_url", "https://api.deepseek.com")))
        self.api_model = tk.StringVar(value=os.environ.get("ECHOMIND_LLM_MODEL", saved.get("api_model", "deepseek-flash")))
        self.api_key = tk.StringVar(value=os.environ.get("ECHOMIND_LLM_API_KEY", saved.get("api_key", "")))
        self.jev_key = tk.StringVar(value=os.environ.get("ECHOMIND_JEV_API_KEY", saved.get("jev_key", "")))
        self.jev = tk.BooleanVar(value=saved.get("jev_enabled", False))
        self.jev_threshold = tk.StringVar(value=f"{saved.get('question_threshold', 0.85) * 100:g}")
        self.show_keys = tk.BooleanVar(value=False)
        self.test_question = tk.StringVar()
        self.shortcuts = {key: tk.StringVar(value=saved.get(key, default)) for key, default in DEFAULT_SHORTCUTS.items()}
        self.shortcut_hint = tk.StringVar()
        self._shortcut_bindings = []
        self.settings_dialog = None
        self.settings_expanded = False
        self._settings_focus_timer = None
        self.component_dialog = None
        self.downloading_components = False
        self.component_cancel = threading.Event()
        self.component_notice = tk.StringVar()
        self.component_details = tk.StringVar()
        self.component_percent = tk.DoubleVar(value=0)
        self.update_dialog = None
        self.checking_updates = False
        self.updating = False
        self.available_release = None
        self.pending_update = None
        self.update_notice = tk.StringVar(value=f"当前版本 {VERSION} · 点击检查更新")
        self.update_notes = tk.StringVar(value="更新会保留本机配置、语音模型与 CUDA。")
        self._build()
        self.root.bind("<Control-v>", self._paste_image_shortcut)
        self._apply_shortcuts()
        self._mode_changed()
        if self.source.get() == "本机麦克风":
            self._source_changed()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._poll = self.root.after(60, self._drain)
        self._refresh = self.root.after(150, lambda: self.refresh() if self.source.get() == "腾讯会议" else None)
        self._update_timer = self.root.after(2500, lambda: self.check_updates(automatic=True)) if getattr(sys, "frozen", False) else None
        self._component_timer = self.root.after(500, self._first_run_components) if getattr(sys, "frozen", False) else None
        self._follow_timer = self.root.after(250, self._follow_tick)

    def _build(self):
        self.root.title("EchoMind · 中文会议助手")
        self.root.geometry("1320x820")
        self.root.minsize(1040, 740)
        self.root.configure(bg=COLORS["background"])
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei UI", 10), foreground=COLORS["text"])
        style.configure("TFrame", background=COLORS["background"])
        style.configure("Card.TFrame", background=COLORS["surface"])
        style.configure("Panel.TFrame", background=COLORS["surface"], bordercolor=COLORS["border"],
                        lightcolor=COLORS["border"], darkcolor=COLORS["border"], borderwidth=1, relief="solid")
        style.configure("TLabel", background=COLORS["background"])
        style.configure("Card.TLabel", background=COLORS["surface"])
        style.configure("Muted.TLabel", foreground=COLORS["muted"])
        style.configure("Error.TLabel", foreground=COLORS["error"])
        style.configure("Badge.TLabel", background=COLORS["tint"], foreground=COLORS["active"], padding=(12, 6))
        style.configure("Hint.TLabel", background=COLORS["tint"], foreground=COLORS["muted"], padding=12)
        style.configure("TButton", padding=(12, 6), background=COLORS["surface"], relief="flat",
                        borderwidth=1, bordercolor=COLORS["control_border"],
                        lightcolor=COLORS["surface"], darkcolor=COLORS["surface"], focuscolor=COLORS["primary"])
        style.map("TButton", background=[("disabled", COLORS["background"]), ("pressed", COLORS["tint"]), ("active", COLORS["surface"])],
                  foreground=[("disabled", COLORS["disabled"]), ("active", COLORS["active"])],
                  bordercolor=[("focus", COLORS["primary"]), ("active", COLORS["primary"])])
        # Darker Ant blue keeps normal-sized white action text readable.
        style.configure("Primary.TButton", background=COLORS["active"], foreground="white", bordercolor=COLORS["active"])
        style.map("Primary.TButton", background=[("disabled", COLORS["background"]), ("pressed", COLORS["pressed"]), ("active", COLORS["pressed"])],
                  foreground=[("disabled", COLORS["disabled"]), ("!disabled", "white")],
                  bordercolor=[("disabled", COLORS["border"]), ("!disabled", COLORS["primary"])])
        style.layout("Dropdown.TButton", [("Button.padding", {"sticky": "nswe", "children": [
            ("Button.focus", {"sticky": "nswe", "children": [("Button.label", {"sticky": "nswe"})]})]})])
        style.configure("Dropdown.TButton", background=COLORS["surface"], foreground=COLORS["text"],
                        padding=(12, 8), anchor="w", focuscolor=COLORS["primary"])
        style.map("Dropdown.TButton", background=[("pressed", COLORS["tint"]), ("focus", COLORS["tint"]), ("active", COLORS["background"])],
                  foreground=[("disabled", COLORS["disabled"]), ("!disabled", COLORS["text"])])
        style.configure("Rail.TFrame", background=COLORS["rail"])
        style.configure("Rail.TButton", padding=(8, 8), width=4, anchor="center",
                        background=COLORS["rail"], borderwidth=0, font=("Microsoft YaHei UI", 9))
        style.map("Rail.TButton", background=[("selected", COLORS["rail_selected"]), ("pressed", COLORS["rail_selected"]),
                  ("active", COLORS["rail_hover"])], foreground=[("!disabled", COLORS["text"])],
                  bordercolor=[("focus", COLORS["primary"])])
        style.configure("TEntry", padding=(8, 6), borderwidth=1, bordercolor=COLORS["control_border"],
                        lightcolor=COLORS["surface"], darkcolor=COLORS["surface"], fieldbackground=COLORS["surface"])
        style.map("TEntry", bordercolor=[("focus", COLORS["primary"])],
                  foreground=[("disabled", COLORS["disabled"])])
        style.configure("TCombobox", padding=(8, 6), borderwidth=1, bordercolor=COLORS["control_border"],
                        lightcolor=COLORS["surface"], darkcolor=COLORS["surface"], arrowcolor=COLORS["muted"])
        style.map("TCombobox", fieldbackground=[("readonly", COLORS["surface"])],
                  background=[("readonly", COLORS["surface"])], bordercolor=[("focus", COLORS["primary"])],
                  foreground=[("disabled", COLORS["disabled"]), ("readonly", COLORS["text"])])
        style.configure("TCheckbutton", background=COLORS["surface"], padding=(0, 6))
        style.configure("TNotebook", background=COLORS["surface"], borderwidth=0, tabmargins=(0, 0, 0, 1))
        # Omit the theme's raised tab border and selected-only padding.
        style.layout("TNotebook.Tab", [("Notebook.padding", {"sticky": "nswe", "children": [
            ("Notebook.focus", {"sticky": "nswe", "children": [
                ("Notebook.label", {"sticky": "nswe"})]})]})])
        style.configure("TNotebook.Tab", padding=(12, 10), width=8, anchor="center",
                        background=COLORS["surface"], focuscolor=COLORS["primary"])
        style.layout("Settings.TNotebook.Tab", [])
        style.configure("Settings.TNotebook", bordercolor=COLORS["surface"],
                        lightcolor=COLORS["surface"], darkcolor=COLORS["surface"])
        style.map("TNotebook.Tab", padding=[], background=[("selected", COLORS["tint"]), ("active", COLORS["background"])],
                  foreground=[("selected", COLORS["active"]), ("active", COLORS["active"])])
        style.layout("Vertical.TScrollbar", [("Vertical.Scrollbar.trough", {"sticky": "ns", "children": [
            ("Vertical.Scrollbar.thumb", {"sticky": "nswe"})]})])
        style.configure("Vertical.TScrollbar", background=COLORS["control_border"], troughcolor=COLORS["surface"],
                        bordercolor=COLORS["surface"], lightcolor=COLORS["surface"], darkcolor=COLORS["surface"],
                        borderwidth=0, relief="flat", arrowsize=10, gripcount=0, sliderlength=24)
        style.map("Vertical.TScrollbar", background=[("pressed", COLORS["muted"]), ("active", COLORS["muted"])])
        style.configure("TPanedwindow", background=COLORS["background"], sashwidth=12)
        style.configure("Treeview", rowheight=tkfont.Font(font=("Microsoft YaHei UI", 10)).metrics("linespace") + 10,
                        background=COLORS["surface"], fieldbackground=COLORS["surface"], foreground=COLORS["text"])
        style.configure("Treeview.Heading", background=COLORS["table_header"], foreground=COLORS["muted"],
                        relief="flat", padding=(8, 6), borderwidth=1)
        style.map("Treeview", background=[("selected", COLORS["tint"])], foreground=[("selected", COLORS["active"])])
        self.full_view = ttk.Frame(self.root)
        self.full_view.pack(fill="both", expand=True)
        self.navigation_rail = ttk.Frame(self.full_view, style="Rail.TFrame", padding=(6, 12))
        self.navigation_rail.pack(side="left", fill="y")
        icon_size = max(20, round(tkfont.Font(font=("Microsoft YaHei UI", 10)).metrics("linespace") * 1.2))
        self.navigation_icons = {name: navigation_icon(self.root, name, icon_size, COLORS["muted"])
                                 for name in ("panel", "basic", "speech", "model", "jev", "settings", "shortcut", "follow", "update", "component")}
        self.sidebar_collapsed = False
        self.sidebar_toggle = ttk.Button(self.navigation_rail, text="收起", image=self.navigation_icons["panel"],
                                         compound="top", style="Rail.TButton", command=self.toggle_sidebar)
        self.sidebar_toggle.pack(fill="x", pady=(0, 20))
        self.navigation_buttons = []
        for index, (name, label) in enumerate(zip(("basic", "speech", "model", "jev"), ("基本", "语音", "模型", "Jev"))):
            button = ttk.Button(self.navigation_rail, text=label, image=self.navigation_icons[name], compound="top",
                                style="Rail.TButton", command=lambda i=index: self._select_settings_page(i))
            button.pack(fill="x", pady=4)
            button.bind("<Return>", self._settings_select_key)
            self.navigation_buttons.append(button)
        outer = ttk.Frame(self.full_view, padding=16)
        outer.pack(side="left", fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 12))
        tk.Label(header, text="E", bg=COLORS["active"], fg="white", font=("Segoe UI", 16, "bold"),
                 padx=12, pady=4).pack(side="left", padx=(0, 12))
        brand = ttk.Frame(header)
        brand.pack(side="left")
        ttk.Label(brand, text="EchoMind", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(brand, text="听清问题 · 梳理思路", style="Muted.TLabel").pack(anchor="w")
        ttk.Label(header, textvariable=self.status, style="Badge.TLabel", wraplength=420).pack(side="right")
        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        sidebar = ttk.Frame(body, style="Panel.TFrame", width=280)
        self.sidebar = sidebar
        sidebar.grid(row=0, column=0, sticky="ns", padx=(0, 16))
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(0, weight=1)
        self.settings_canvas = tk.Canvas(sidebar, bg=COLORS["surface"], highlightthickness=0, width=270)
        settings_scroll = ttk.Scrollbar(sidebar, orient="vertical", command=self.settings_canvas.yview)
        settings_scroll.grid(row=0, column=1, sticky="ns")
        self.settings_canvas.grid(row=0, column=0, sticky="nsew")
        self.settings_canvas.configure(yscrollcommand=settings_scroll.set)
        settings = ttk.Frame(self.settings_canvas, padding=16, style="Card.TFrame")
        self.settings_window = self.settings_canvas.create_window(0, 0, window=settings, anchor="nw")
        settings.bind("<Configure>", lambda event: self.settings_canvas.configure(scrollregion=self.settings_canvas.bbox("all")))
        self.settings_canvas.bind("<Configure>", lambda event: self.settings_canvas.itemconfigure(self.settings_window, width=event.width))
        settings.columnconfigure(0, weight=1)
        self.settings_heading = tk.StringVar(value="基本设置")
        ttk.Label(settings, textvariable=self.settings_heading, font=("Microsoft YaHei UI", 12, "bold"), style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 12))
        self.settings_tabs = ttk.Notebook(settings, style="Settings.TNotebook")
        self.settings_tabs.grid(row=1, column=0, sticky="ew")
        basic, speech, model, semantic = [ttk.Frame(self.settings_tabs, style="Card.TFrame", padding=(0, 12)) for _ in range(4)]
        for page, title in zip((basic, speech, model, semantic), ("基本", "语音", "模型", "Jev")):
            page.columnconfigure(0, weight=1)
            self.settings_tabs.add(page, text=title)
        self.settings_tabs.bind("<<NotebookTabChanged>>", self._sync_navigation)
        self._sync_navigation()
        self._combo(basic, 0, "回答方式", self.mode, list(ANSWER_MODES))
        ttk.Label(basic, textvariable=self.mode_hint, style="Hint.TLabel", wraplength=225).grid(row=2, column=0, sticky="ew", pady=12)
        self._combo(basic, 3, "识别设备", self.device, list(DEVICE_MODES))
        ttk.Label(basic, text="切换模式前请先停止采集。\n默认自动选择腾讯会议主进程。", style="Card.TLabel",
                  foreground=COLORS["muted"], wraplength=240).grid(row=5, column=0, sticky="w", pady=(16, 0))
        engine_combo = self._combo(speech, 0, "识别引擎", self.asr_engine, list(ENGINE_LABELS.values()))
        engine_combo.bind("<<ComboboxSelected>>", self._asr_engine_changed)
        topic_combo = self._combo(speech, 2, "术语场景", self.asr_topic, ["通用", "前端开发", "自定义"])
        topic_combo.bind("<<ComboboxSelected>>", self._asr_topic_changed)
        self._entry(speech, 4, "技术热词（逗号分隔）", self.asr_hotwords)
        ttk.Label(speech, text="仅填写短术语，不填写整句或答案。留空不传热词。腾讯云热词参与流式识别；实验引擎不支持热词。", style="Card.TLabel",
                  foreground=COLORS["muted"], wraplength=225).grid(row=6, column=0, sticky="ew", pady=(0, 12))
        self.local_asr_fields = ttk.Frame(speech, style="Card.TFrame")
        self.local_asr_fields.columnconfigure(0, weight=1)
        self.local_asr_fields.grid(row=7, column=0, sticky="ew")
        self._entry(self.local_asr_fields, 0, "本地识别模型目录（可选）", self.asr_model_path)
        browse_model = ttk.Button(self.local_asr_fields, text="选择模型目录", command=self._choose_asr_model)
        browse_model.grid(row=2, column=0, sticky="ew", pady=8)
        self.form_controls.append((browse_model, "normal"))
        self.cloud_asr_fields = ttk.Frame(speech, style="Card.TFrame")
        self.cloud_asr_fields.columnconfigure(0, weight=1)
        self.cloud_asr_fields.grid(row=8, column=0, sticky="ew")
        for row, (key, label) in enumerate((("tencent_app_id", "腾讯云 AppID"), ("tencent_secret_id", "腾讯云 SecretId"), ("tencent_secret_key", "腾讯云 SecretKey"))):
            self._entry(self.cloud_asr_fields, row * 2, label, self.tencent_fields[key], secret=key != "tencent_app_id")
        self._combo(self.cloud_asr_fields, 6, "云端模型（中英 / 中文通用）", self.tencent_fields["tencent_engine"], ["16k_zh_en", "16k_zh"])
        cloud_consent = ttk.Checkbutton(self.cloud_asr_fields, text="同意上传音频到腾讯云\n（可能计费，仅本次启动有效）", variable=self.cloud_consent)
        cloud_consent.grid(row=8, column=0, sticky="w", pady=8)
        self.form_controls.append((cloud_consent, "normal"))
        self._sync_asr_fields()
        ttk.Label(speech, textvariable=self.asr_engine_hint,
                  style="Hint.TLabel", wraplength=225).grid(row=10, column=0, sticky="ew", pady=8)
        self._entry(model, 0, "回答 API 地址", self.api_url)
        self._entry(model, 2, "回答模型", self.api_model)
        self.key_entry = self._entry(model, 4, "回答模型密钥", self.api_key, secret=True)
        reveal = ttk.Checkbutton(model, text="显示密钥", variable=self.show_keys, command=self._reveal_keys)
        reveal.grid(row=6, column=0, sticky="w", pady=8)
        self.form_controls.append((reveal, "normal"))
        ttk.Label(model, text="保存后密钥以 Windows 加密形式写入配置，仅当前用户可读取。", style="Card.TLabel", foreground=COLORS["muted"],
                  wraplength=240).grid(row=7, column=0, sticky="w")
        self.jev_toggle = ttk.Checkbutton(semantic, text="启用 Jev 语义补判", variable=self.jev)
        self.jev_toggle.grid(row=0, column=0, sticky="w")
        self.form_controls.append((self.jev_toggle, "normal"))
        self.jev_entry = self._entry(semantic, 1, "TypeSafe 密钥（可选）", self.jev_key, secret=True)
        ttk.Label(semantic, text="触发阈值（%）", style="Card.TLabel").grid(row=3, column=0, sticky="w", pady=(12, 4))
        self.jev_threshold_input = ttk.Spinbox(semantic, from_=0, to=100, increment=1, textvariable=self.jev_threshold, width=8)
        self.jev_threshold_input.grid(row=4, column=0, sticky="ew")
        self.form_controls.append((self.jev_threshold_input, "normal"))
        ttk.Label(semantic, text="0–100%，默认 85%。越低越易触发，越高越严格。仅影响 Jev 补判。",
                  style="Card.TLabel", foreground=COLORS["muted"], wraplength=225).grid(row=5, column=0, sticky="ew", pady=(4, 8))
        ttk.Label(semantic, text="需在线回答模式及独立 TypeSafe 密钥。本地规则未命中的确定字幕交给 Jev 判断，增加调用量和延迟。",
                  style="Hint.TLabel", wraplength=225).grid(row=6, column=0, sticky="ew", pady=12)
        self.save_button = ttk.Button(settings, text="保存配置（含加密密钥）", command=self.save_config)
        self.save_button.grid(row=2, column=0, sticky="ew", pady=(12, 8))
        self.form_controls.append((self.save_button, "normal"))
        self.settings_popup = tk.Canvas(self.root, bg=COLORS["background"], highlightthickness=0)
        self.settings_menu = ttk.Frame(self.settings_popup, style="Card.TFrame", padding=4)
        self._settings_menu_window = self.settings_popup.create_window(8, 8, anchor="nw", window=self.settings_menu)
        ttk.Label(self.settings_menu, text="EchoMind", style="Card.TLabel", padding=(12, 8), font=("Segoe UI", 11, "bold")).pack(fill="x")
        ttk.Separator(self.settings_menu).pack(fill="x", padx=8, pady=(0, 4))
        self.shortcut_settings_button = ttk.Button(self.settings_menu, text="  快捷键设置", image=self.navigation_icons["shortcut"], compound="left", style="Dropdown.TButton", command=lambda: self._settings_action(self.open_settings))
        self.shortcut_settings_button.pack(fill="x")
        self.form_controls.append((self.shortcut_settings_button, "normal"))
        self.settings_subpopup = tk.Canvas(self.root, bg=COLORS["background"], highlightthickness=0)
        self.settings_submenu = ttk.Frame(self.settings_subpopup, style="Card.TFrame", padding=4)
        self._settings_submenu_window = self.settings_subpopup.create_window(8, 8, anchor="nw", window=self.settings_submenu)
        self.settings_submenu_expanded = False
        self.update_menu_button = ttk.Button(self.settings_submenu, text="  检查更新", image=self.navigation_icons["update"], compound="left", style="Dropdown.TButton", command=lambda: self._settings_action(self.check_updates))
        self.update_menu_button.pack(fill="x")
        self.form_controls.append((self.update_menu_button, "normal"))
        self.component_menu_button = ttk.Button(self.settings_submenu, text="  组件管理", image=self.navigation_icons["component"], compound="left", style="Dropdown.TButton", command=lambda: self._settings_action(self.open_components))
        self.component_menu_button.pack(fill="x")
        self.form_controls.append((self.component_menu_button, "normal"))
        self.follow_menu_button = ttk.Button(self.settings_menu, text="  窗口跟随", image=self.navigation_icons["follow"], compound="left", style="Dropdown.TButton", command=lambda: self._settings_action(self.open_follow_settings))
        self.follow_menu_button.pack(fill="x")
        self.form_controls.append((self.follow_menu_button, "normal"))
        ttk.Separator(self.settings_menu).pack(fill="x", padx=8, pady=4)
        self.maintenance_menu_button = ttk.Button(self.settings_menu, text="  应用维护    ›", image=self.navigation_icons["component"], compound="left", style="Dropdown.TButton", command=self._open_settings_submenu)
        self.maintenance_menu_button.pack(fill="x")
        self.maintenance_menu_button.bind("<Right>", lambda _: self._open_settings_submenu() or "break")
        self.settings_button = ttk.Button(self.navigation_rail, text="设置", image=self.navigation_icons["settings"], compound="top", style="Rail.TButton", command=self.toggle_settings)
        self.settings_button.pack(side="bottom", fill="x")
        self.form_controls.append((self.settings_button, "normal"))
        self._settings_buttons = (self.shortcut_settings_button, self.follow_menu_button, self.maintenance_menu_button)
        self._settings_subbuttons = (self.update_menu_button, self.component_menu_button)
        for index, button in enumerate(self._settings_subbuttons):
            for sequence, offset in (("<Down>", 1), ("<Up>", -1), ("<Tab>", 1), ("<Shift-Tab>", -1)):
                button.bind(sequence, lambda event, i=index, step=offset: self._settings_subnavigate(i + step))
            button.bind("<Left>", lambda _: self._close_settings_submenu() or "break")
            button.bind("<Escape>", lambda _: self._close_settings_submenu() or "break")
            button.bind("<Return>", self._settings_select_key)
        for index, button in enumerate(self._settings_buttons):
            for sequence, offset in (("<Down>", 1), ("<Up>", -1), ("<Tab>", 1), ("<Shift-Tab>", -1)):
                button.bind(sequence, lambda event, i=index, step=offset: self._settings_navigate(i + step))
            button.bind("<Escape>", lambda _: self._hide_settings_menu() or "break")
            button.bind("<Return>", self._settings_select_key)
        self.settings_button.bind("<Down>", lambda _: self._settings_open_key())
        self.root.bind("<ButtonPress-1>", self._settings_outside_click, add="+")
        self.root.bind("<FocusOut>", self._settings_focus_out, add="+")
        self.root.bind("<Configure>", self._settings_reposition, add="+")
        self.root.bind("<Unmap>", lambda event: self._hide_settings_menu(focus=False) if event.widget == self.root else None, add="+")
        main = ttk.Frame(body)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)
        toolbar = ttk.Frame(main, style="Panel.TFrame", padding=12)
        self.capture_toolbar = toolbar
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        source_row = ttk.Frame(toolbar, style="Card.TFrame")
        source_row.pack(fill="x", pady=(0, 8))
        ttk.Label(source_row, text="输入来源", style="Card.TLabel", font=("Microsoft YaHei UI", 11, "bold")).pack(side="left", padx=(0, 12))
        source_combo = ttk.Combobox(source_row, textvariable=self.source, values=("腾讯会议", "本机麦克风"), state="readonly", width=12)
        self.source_combo = source_combo
        source_combo.pack(side="left")
        source_combo.bind("<<ComboboxSelected>>", lambda _: self._source_changed())
        self.form_controls.append((source_combo, "readonly"))
        row = ttk.Frame(source_row, style="Card.TFrame")
        row.pack(side="right")
        self.refresh_button = ttk.Button(row, text="刷新", command=self.refresh)
        self.refresh_button.pack(side="left", padx=(0, 8))
        self.start_button = ttk.Button(row, text="开始采集", style="Primary.TButton", command=self.start_capture)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(row, text="停止", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        details = ttk.Frame(toolbar, style="Card.TFrame")
        details.pack(fill="x")
        shortcut_label = ttk.Label(details, textvariable=self.shortcut_hint, style="Card.TLabel", justify="left",
                                   foreground=COLORS["muted"])
        shortcut_label.pack(side="left", fill="x", expand=True)
        follow_label = ttk.Label(details, textvariable=self.follow_notice, style="Card.TLabel", justify="right",
                                foreground=COLORS["muted"])
        follow_label.pack(side="right", padx=(16, 0))
        details.bind("<Configure>", lambda e: [label.configure(wraplength=max(120, (e.width - 16) // 2)) for label in (shortcut_label, follow_label)])
        ttk.Label(main, textvariable=self.error, style="Error.TLabel", wraplength=660).grid(row=1, column=0, sticky="ew", pady=(0, 8))
        panels = ttk.Panedwindow(main, orient="horizontal")
        panels.grid(row=2, column=0, sticky="nsew")
        captions = ttk.Frame(panels, style="Panel.TFrame", padding=12)
        self.answer_tabs = ttk.Notebook(panels)
        answers = ttk.Frame(self.answer_tabs, style="Card.TFrame", padding=12)
        history = ttk.Frame(self.answer_tabs, style="Card.TFrame", padding=12)
        self.answer_tabs.add(answers, text="当前回答")
        self.answer_tabs.add(history, text="问题记录")
        panels.add(captions, weight=2)
        panels.add(self.answer_tabs, weight=3)
        caption_header = ttk.Frame(captions, style="Card.TFrame")
        caption_header.pack(fill="x")
        ttk.Label(caption_header, text="会议字幕", style="Card.TLabel", font=("Microsoft YaHei UI", 12, "bold")).pack(side="left")
        ttk.Label(captions, textvariable=self.caption_source, style="Card.TLabel", foreground=COLORS["muted"]).pack(anchor="w", pady=(4, 8))
        partial_label = ttk.Label(captions, textvariable=self.partial, style="Hint.TLabel", wraplength=320)
        partial_label.pack(side="bottom", fill="x", pady=(8, 0))
        self.add_caption_button = ttk.Button(caption_header, text="加入问题", command=self.add_selected_caption)
        self.add_caption_button.pack(side="right")
        self.caption_text = self._text(captions)
        self.caption_text.configure(exportselection=False)
        self.caption_text.bind("<Control-Return>", lambda _: self.add_selected_caption() or "break")
        self.caption_text.tag_configure("caption_pick", background=COLORS["tint"], foreground=COLORS["active"])
        self.caption_text.bind("<Control-Button-1>", self._pick_caption_click)
        self.caption_text.bind("<Control-space>", lambda _: self._toggle_caption_selection("insert"))
        self.caption_text.bind("<Button-1>", lambda _: self._clear_caption_selection())
        self.caption_text.bind("<Escape>", lambda _: self._clear_caption_selection() or "break")
        self.caption_selection_notice = tk.StringVar(value="Ctrl＋单击多选字幕；也可拖选连续文字")
        selection_bar = ttk.Frame(captions, style="Card.TFrame")
        selection_bar.pack(fill="x", before=self.caption_text.master, pady=(0, 8))
        self.merge_caption_button = ttk.Button(selection_bar, text="合并为问题", command=self.merge_selected_captions)
        self.merge_caption_button.pack(side="right")
        selection_hint = ttk.Label(selection_bar, textvariable=self.caption_selection_notice, style="Card.TLabel",
                                   foreground=COLORS["muted"], wraplength=220)
        selection_hint.pack(side="left", fill="x", expand=True, padx=(0, 8))
        selection_bar.bind("<Configure>", lambda e: selection_hint.configure(
            wraplength=max(60, e.width - self.merge_caption_button.winfo_reqwidth() - 16)))
        answer_header = ttk.Frame(answers, style="Card.TFrame")
        answer_header.pack(fill="x")
        ttk.Label(answer_header, text="回答要点", style="Card.TLabel", font=("Microsoft YaHei UI", 12, "bold")).pack(side="left")
        ttk.Button(answer_header, text="编辑当前问题", command=self.edit_current_question).pack(side="right")
        detection_label = ttk.Label(answers, textvariable=self.detection, style="Card.TLabel", foreground=COLORS["muted"], wraplength=330)
        detection_label.pack(fill="x", pady=(8, 0))
        question_label = ttk.Label(answers, textvariable=self.question, style="Hint.TLabel", wraplength=330)
        question_label.pack(fill="x", pady=8)
        metrics_label = ttk.Label(answers, textvariable=self.metrics, style="Card.TLabel", foreground=COLORS["muted"], wraplength=330)
        metrics_label.pack(side="bottom", fill="x", pady=(8, 0))
        captions.bind("<Configure>", lambda e: partial_label.configure(wraplength=max(120, e.width - 56)))
        answers.bind("<Configure>", lambda e: [label.configure(wraplength=max(120, e.width - 56)) for label in (question_label, metrics_label, detection_label)])
        self.answer_text = self._text(answers)
        self.repair_panel = ttk.Frame(answers, style="Card.TFrame")
        repair_label = ttk.Label(self.repair_panel, textvariable=self.repair_notice,
                                 style="Hint.TLabel", wraplength=330)
        repair_label.pack(fill="x")
        ttk.Button(self.repair_panel, text="核对并编辑", command=self.edit_repair).pack(anchor="w", pady=(6, 0))
        answers.bind("<Configure>", lambda e: repair_label.configure(wraplength=max(120, e.width - 56)), add="+")
        ttk.Button(history, text="编辑选中问题", command=self.edit_history_question).pack(anchor="w", pady=(0, 8))
        history_list = ttk.Frame(history, style="Card.TFrame")
        history_list.pack(fill="x")
        self.history_tree = ttk.Treeview(history_list, columns=("time", "state"), height=4, selectmode="browse")
        self.history_tree.heading("#0", text="问题")
        self.history_tree.heading("time", text="时间")
        self.history_tree.heading("state", text="状态")
        self.history_tree.column("#0", width=180, minwidth=80)
        self.history_tree.column("time", width=70, minwidth=70, stretch=False)
        self.history_tree.column("state", width=65, minwidth=65, stretch=False)
        history_scroll = ttk.Scrollbar(history_list, command=self.history_tree.yview)
        history_scroll.pack(side="right", fill="y")
        self.history_tree.pack(fill="x", expand=True)
        self.history_tree.configure(yscrollcommand=history_scroll.set)
        self.history_tree.bind("<<TreeviewSelect>>", lambda _: self._show_history())
        history_question = ttk.Label(history, textvariable=self.history_question, style="Hint.TLabel", wraplength=330)
        history_question.pack(fill="x", pady=(12, 0))
        history_status = ttk.Label(history, textvariable=self.history_status, style="Card.TLabel", foreground=COLORS["muted"], wraplength=330)
        history_status.pack(side="bottom", fill="x", pady=(8, 0))
        self.history_text = self._text(history)
        history.bind("<Configure>", lambda e: [label.configure(wraplength=max(120, e.width - 56)) for label in (history_question, history_status)])
        test = ttk.Frame(main, style="Panel.TFrame", padding=12)
        test.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        edit_header = ttk.Frame(test, style="Card.TFrame")
        edit_header.pack(fill="x", pady=(0, 8))
        ttk.Label(edit_header, text="问题编辑 · 核对后提交", style="Card.TLabel").pack(side="left")
        self.screenshot_button = ttk.Button(edit_header, text="粘贴题目截图", command=self.paste_screenshot)
        self.screenshot_button.pack(side="right")
        self.import_screenshot_button = ttk.Button(edit_header, text="选择图片", command=self.import_screenshot)
        self.import_screenshot_button.pack(side="right", padx=(8, 8))
        test_row = ttk.Frame(test, style="Card.TFrame")
        test_row.pack(fill="x")
        self.question_entry = ttk.Entry(test_row, textvariable=self.test_question)
        self.question_entry.pack(side="left", fill="x", expand=True)
        self.question_entry.bind("<Control-v>", self._paste_image_shortcut)
        ttk.Button(test_row, text="清空", command=lambda: self.test_question.set("")).pack(side="left", padx=(8, 0))
        self.test_button = ttk.Button(test_row, text="提交回答 (Enter)", style="Primary.TButton", command=self.start_test)
        self.test_button.pack(side="left", padx=(8, 0))
        screenshot_hint = ttk.Label(test, textvariable=self.screenshot_notice, style="Card.TLabel", foreground=COLORS["muted"], justify="left")
        screenshot_hint.pack(fill="x", pady=(8, 0))
        test.bind("<Configure>", lambda e: screenshot_hint.configure(wraplength=max(120, e.width - 32)))
        for widget, _ in self.form_controls:
            if str(widget).startswith(str(settings)):
                widget.bind("<FocusIn>", self._reveal_field)
        self._build_compact_view()

    def _build_compact_view(self):
        self.compact_view = ttk.Frame(self.root, padding=12)
        self.compact_view.columnconfigure(0, weight=1)
        self.compact_view.rowconfigure(1, weight=2)
        self.compact_view.rowconfigure(2, weight=1)
        header = ttk.Frame(self.compact_view)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(header, text="EchoMind · 跟随", font=("Microsoft YaHei UI", 11, "bold")).pack(side="left")
        self.exit_follow_button = ttk.Button(header, text="退出跟随", command=self._exit_follow)
        self.exit_follow_button.pack(side="right")
        captions = ttk.Frame(self.compact_view, style="Card.TFrame", padding=12)
        captions.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        ttk.Label(captions, text="字幕", style="Card.TLabel", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w")
        partial = ttk.Label(captions, textvariable=self.partial, style="Hint.TLabel", justify="left")
        partial.pack(side="bottom", fill="x", pady=(8, 0))
        self.compact_caption_text = self._text(captions)
        questions = ttk.Frame(self.compact_view, style="Card.TFrame", padding=12)
        questions.grid(row=2, column=0, sticky="nsew")
        ttk.Label(questions, text="问题记录", style="Card.TLabel", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w")
        self.compact_question_text = self._text(questions)
        self._write(self.compact_question_text, "等待识别问题…", replace=True)
        ttk.Label(self.compact_view, textvariable=self.error, style="Error.TLabel", wraplength=320).grid(row=3, column=0, sticky="ew")
        captions.bind("<Configure>", lambda event: partial.configure(wraplength=max(100, event.width - 24)))

    def _set_compact_mode(self, enabled):
        if enabled == self.compact_mode:
            return False
        self._hide_settings_menu(focus=False)
        if enabled:
            self.root.update_idletasks()
            self.full_window_geometry = self.root.geometry()
            self.full_window_minimum = self.root.minsize()
            self.full_window_state = self.root.state()
            if self.full_window_state == "zoomed":
                self.root.state("normal")
            self.full_view.pack_forget()
            self.compact_view.pack(fill="both", expand=True)
            scale = float(self.root.tk.call("tk", "scaling")) / (96 / 72)
            width = min(round(420 * scale), self.root.winfo_screenwidth())
            height = min(round(620 * scale), self.root.winfo_screenheight() - 80)
            self.root.minsize(min(width, round(320 * scale)), min(height, round(380 * scale)))
            self.root.geometry(f"{width}x{height}")
        else:
            self.compact_view.pack_forget()
            self.full_view.pack(fill="both", expand=True)
            self.root.minsize(*self.full_window_minimum)
            self.root.geometry(self.full_window_geometry)
            if self.full_window_state == "zoomed":
                self.root.state("zoomed")
        self.compact_mode = enabled
        self.root.update_idletasks()
        return True

    def _exit_follow(self):
        self.follow_enabled.set(False)
        self._follow_tick()
        (self.sidebar_toggle if self.sidebar_collapsed else self.settings_button).focus_set()

    def toggle_sidebar(self):
        self._hide_settings_menu(focus=False)
        self.sidebar_collapsed = not self.sidebar_collapsed
        if self.sidebar_collapsed:
            self.sidebar.grid_remove()
        else:
            self.sidebar.grid()
        self.sidebar_toggle.configure(text="展开" if self.sidebar_collapsed else "收起")
        self._sync_navigation()
        self.sidebar_toggle.focus_set()

    def _select_settings_page(self, index):
        if self.sidebar_collapsed or self.settings_tabs.index("current") == index:
            self.toggle_sidebar()
        self.settings_tabs.select(index)
        self.settings_canvas.yview_moveto(0)
        self._sync_navigation()
        self.navigation_buttons[index].focus_set()

    def _sync_navigation(self, event=None):
        index = self.settings_tabs.index("current")
        self.settings_heading.set(("基本设置", "语音识别", "回答模型", "Jev 语义判断")[index])
        for i, button in enumerate(self.navigation_buttons):
            button.state(["selected" if i == index and not self.sidebar_collapsed else "!selected"])
        self.root.after_idle(self._fit_settings_page)

    def _fit_settings_page(self):
        if not self.destroyed:
            page = self.root.nametowidget(self.settings_tabs.select())
            self.settings_tabs.configure(height=page.winfo_reqheight())

    def _mode_changed(self):
        self.mode_hint.set(MODE_HINTS.get(self.mode.get(), ""))
        self.detection.set("仅字幕模式：不检测问题、不生成回答。" if ANSWER_MODES.get(self.mode.get()) == "off" else "问题检测已启用：请说出完整问题。")

    def import_screenshot(self):
        if self.importing_screenshot or self.closing or self.destroyed:
            return
        path = filedialog.askopenfilename(parent=self.root, title="选择面试题截图",
                                         filetypes=[("题目截图", "*.png *.jpg *.jpeg *.bmp")])
        if not path:
            return
        self._start_screenshot(lambda: recognize_screenshot(path))

    def _paste_image_shortcut(self, event=None):
        # Do not take over normal text paste, or paste into model/key settings.
        if event is not None and isinstance(event.widget, (ttk.Entry, tk.Text)) and event.widget is not self.question_entry:
            return None
        if clipboard_has_image():
            self.paste_screenshot()
            return "break"
        return None

    def paste_screenshot(self):
        if self.importing_screenshot or self.closing or self.destroyed:
            return
        if not clipboard_has_image():
            self.screenshot_notice.set("剪贴板中没有图片，请先截图或复制图片，再粘贴。")
            return
        self._start_screenshot(recognize_clipboard)

    def _start_screenshot(self, recognizer):
        self.importing_screenshot = True
        self.screenshot_button.configure(state="disabled")
        self.import_screenshot_button.configure(state="disabled")
        self.screenshot_notice.set("正在本机识别截图… 可继续使用其他功能。")

        def recognize():
            try:
                text = recognizer()
                self.events.put((None, "screenshot_ready", text))
            except Exception as exc:
                message = str(exc) if isinstance(exc, ScreenshotError) else "截图识别失败，请重新选择图片。"
                self.events.put((None, "screenshot_error", message))

        threading.Thread(target=recognize, name="echomind-screenshot", daemon=True).start()

    def _preview_screenshot(self, text):
        dialog = tk.Toplevel(self.root)
        dialog.title("核对截图中的面试题")
        dialog.geometry("760x480")
        dialog.minsize(540, 360)
        dialog.transient(self.root)
        dialog.configure(bg=COLORS["surface"])
        frame = ttk.Frame(dialog, padding=16, style="Card.TFrame")
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        heading = ttk.Label(frame, text="可编辑识别结果；多道题请选中需要的一题。", style="Card.TLabel", wraplength=480)
        heading.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        editor_frame = ttk.Frame(frame, style="Card.TFrame")
        editor_frame.grid(row=1, column=0, sticky="nsew")
        editor = tk.Text(editor_frame, wrap="word", font=("Microsoft YaHei UI", 11), undo=True, width=1, height=1,
                         foreground=COLORS["text"], background=COLORS["surface"])
        scrollbar = ttk.Scrollbar(editor_frame, command=editor.yview)
        scrollbar.pack(side="right", fill="y")
        editor.configure(yscrollcommand=scrollbar.set)
        editor.pack(fill="both", expand=True)
        editor.insert("1.0", text)
        notice = tk.StringVar(value="不会自动回答或上传图片。加入后请在主窗口提交回答。")
        notice_label = ttk.Label(frame, textvariable=notice, style="Card.TLabel", foreground=COLORS["muted"], wraplength=480)
        notice_label.grid(row=2, column=0, sticky="ew", pady=8)
        shortcut_label = ttk.Label(frame, text="Ctrl+Enter 全部加入 · Esc 取消", style="Card.TLabel", foreground=COLORS["muted"], wraplength=480)
        shortcut_label.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        actions = ttk.Frame(frame, style="Card.TFrame")
        actions.grid(row=4, column=0, sticky="ew")
        frame.bind("<Configure>", lambda event: [label.configure(wraplength=max(120, event.width - 32)) for label in (heading, notice_label, shortcut_label)])

        def add(selected=False):
            try:
                content = editor.get("sel.first", "sel.last") if selected else editor.get("1.0", "end-1c")
            except tk.TclError:
                notice.set("请先选中一道题，或点击“全部加入”。")
                return
            content = " ".join(content.split())
            if not content:
                notice.set("请填写或选中题目文字。")
                return
            self._append_screenshot_question(content)
            dialog.destroy()
            self.question_entry.focus_set()

        ttk.Button(actions, text="取消", command=dialog.destroy).pack(side="right")
        ttk.Button(actions, text="全部加入", command=add).pack(side="right", padx=8)
        ttk.Button(actions, text="选中题目加入", style="Primary.TButton", command=lambda: add(True)).pack(side="right")
        dialog.bind("<Escape>", lambda _: dialog.destroy() or "break")
        editor.bind("<Control-Return>", lambda _: add() or "break")
        dialog.bind("<Control-Return>", lambda _: add() or "break")
        dialog.update_idletasks()
        minimum_width = max(540, actions.winfo_reqwidth() + 32)
        dialog.minsize(minimum_width, 360)
        dialog.geometry(f"{max(760, minimum_width)}x480")
        editor.focus_set()
        dialog.wait_visibility()
        dialog.grab_set()

    def _append_screenshot_question(self, text):
        draft = self.test_question.get().strip()
        self._edit_question((draft + " " + text).strip())
        self.screenshot_notice.set("题目已加入编辑框，请核对后点击“提交回答”。")

    def _edit_question(self, text):
        self.test_question.set(text)
        self.question_entry.focus_set()
        self.question_entry.selection_range(0, "end")
        self.error.set("")

    def edit_repair(self):
        if self.pending_repair:
            self._edit_question(self.pending_repair)
            self.detection.set("请核对编辑框中的问题，再点击“提交回答”。")

    def _clear_repair(self):
        self.pending_repair = ""
        self.repair_notice.set("")
        self.repair_panel.pack_forget()

    def _pick_caption_click(self, event):
        self.caption_text.focus_set()
        return self._toggle_caption_selection(f"@{event.x},{event.y}")

    def _toggle_caption_selection(self, index):
        start = self.caption_text.index(f"{index} linestart")
        end = self.caption_text.index(f"{start} lineend +1c")
        if re.match(r"^\d{2}:\d{2}:\d{2}\s+\S", self.caption_text.get(start, end)):
            self.caption_text.tag_remove("sel", "1.0", "end")
            if "caption_pick" in self.caption_text.tag_names(start):
                self.caption_text.tag_remove("caption_pick", start, end)
            else:
                self.caption_text.tag_add("caption_pick", start, end)
            self._caption_selection_changed()
        return "break"

    def _clear_caption_selection(self):
        self.caption_text.tag_remove("caption_pick", "1.0", "end")
        self.caption_text.tag_remove("sel", "1.0", "end")
        self._caption_selection_changed()

    def _caption_selection_changed(self):
        ranges = self.caption_text.tag_ranges("caption_pick")
        count = sum(len(self.caption_text.get(start, end).splitlines()) for start, end in zip(ranges[::2], ranges[1::2]))
        self.caption_selection_notice.set(f"已选 {count} 条 · Ctrl＋单击取消，Esc 清除" if count else "Ctrl＋单击多选字幕；也可拖选连续文字")

    def _selected_caption_text(self):
        ranges = self.caption_text.tag_ranges("caption_pick") or self.caption_text.tag_ranges("sel")
        if not ranges:
            self.error.set("请先拖选字幕文字，或按住 Ctrl 单击多选字幕。")
            return ""
        parts = []
        for start, end in zip(ranges[::2], ranges[1::2]):
            for line in self.caption_text.get(start, end).splitlines():
                line = re.sub(r"^\s*\d{2}:\d{2}:\d{2}[ \t]+", "", line).strip()
                if line and not line.startswith("音频复识别前："):
                    parts.append(line)
        selected = " ".join(" ".join(parts).split())
        if not selected:
            self.error.set("选中的字幕没有可用文字。")
        return selected

    def add_selected_caption(self):
        selected = self._selected_caption_text()
        if not selected:
            return
        draft = self.test_question.get().strip()
        self._edit_question((draft + " " + selected).strip())

    def merge_selected_captions(self):
        selected = self._selected_caption_text()
        if selected:
            self._edit_question(selected)
            self._clear_caption_selection()

    def edit_current_question(self):
        if self.answer_id is None:
            self.error.set("暂无已识别的问题，请先选中字幕加入问题。")
            return
        self._edit_question(self.question.get())

    def edit_history_question(self):
        selected = self.history_tree.selection()
        if not selected or selected[0] not in self.question_records:
            self.error.set("请先选择一条问题记录。")
            return
        self._edit_question(self.question_records[selected[0]].question)

    def _first_run_components(self):
        if self.destroyed or self.closing:
            return
        # A custom model is managed by the user, never download Turbo in its place.
        if self.asr_model_path.get().strip() or self._engine() != "whisper":
            return
        if missing(self.model_path, self.cuda_dir, DEVICE_MODES[self.device.get()]):
            if self.root.grab_current() is not None:
                self._component_timer = self.root.after(500, self._first_run_components)
            else:
                self.open_components()

    def _component_state(self):
        items = missing(self.model_path, self.cuda_dir, DEVICE_MODES[self.device.get()])
        details = ("CPU 只需模型；GPU 还需 CUDA 组件及支持 CUDA 的 NVIDIA 显卡和驱动。\n"
                   "下载来自 EchoMind GitHub Release，校验 SHA-256 后安装；不上传音频、字幕或密钥。\n"
                   f"模型：{self.model_path}\nGPU：{self.cuda_dir}")
        if not getattr(sys, "frozen", False):
            details += "\n开发环境复用 .models 与 .runtime/cuda12，不自动联网下载。"
        elif items:
            size = sum(item.size for item in items) / 1024**3
            details += f"\n待下载：{'、'.join(item.name for item in items)}，约 {size:.2f} GiB；缓存升级后保留。"
        self.component_details.set(details)
        if not self.downloading_components:
            self.component_notice.set("需要准备语音组件；可稍后下载，先使用文字/截图回答。" if items else "语音组件已就绪，可关闭窗口开始采集。")
        if self.component_dialog is not None:
            self.component_download_button.configure(state="normal" if items and not self.downloading_components and getattr(sys, "frozen", False) else "disabled")
            self.component_device_combo.configure(state="disabled" if self.downloading_components else "readonly")
            self.component_close_button.configure(text="取消下载" if self.downloading_components else "关闭")

    def open_components(self):
        if self.component_dialog is not None and self.component_dialog.winfo_exists():
            self.component_dialog.lift()
            return
        if self.root.grab_current() is not None or self.updating or (self.session and self.session.is_running):
            return
        dialog = tk.Toplevel(self.root)
        self.component_dialog = dialog
        dialog.title("设置 · 语音组件 · EchoMind")
        dialog.transient(self.root)
        frame = ttk.Frame(dialog, padding=24, style="Card.TFrame")
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(3, weight=1)
        ttk.Label(frame, text="准备语音识别组件", style="Card.TLabel", font=("Microsoft YaHei UI", 12, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 16))
        row = ttk.Frame(frame, style="Card.TFrame")
        row.grid(row=1, column=0, sticky="ew", pady=(0, 16))
        ttk.Label(row, text="识别设备", style="Card.TLabel").pack(side="left", padx=(0, 12))
        self.component_device_combo = ttk.Combobox(row, textvariable=self.device, values=list(DEVICE_MODES), state="readonly", width=16)
        self.component_device_combo.pack(side="left")
        self.component_device_combo.bind("<<ComboboxSelected>>", lambda _: self._component_state())
        notice = ttk.Label(frame, textvariable=self.component_notice, style="Card.TLabel", wraplength=540)
        notice.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        details = ttk.Label(frame, textvariable=self.component_details, style="Card.TLabel", foreground=COLORS["muted"], wraplength=540, justify="left")
        details.grid(row=3, column=0, sticky="new", pady=(0, 16))
        ttk.Progressbar(frame, variable=self.component_percent, maximum=100).grid(row=4, column=0, sticky="ew", pady=(0, 16))
        actions = ttk.Frame(frame, style="Card.TFrame")
        actions.grid(row=5, column=0, sticky="ew")
        self.component_download_button = ttk.Button(actions, text="下载并准备", style="Primary.TButton", command=self.download_components)
        self.component_download_button.pack(side="right")

        def cancel():
            if self.downloading_components:
                self.component_cancel.set()
                self.component_notice.set("正在取消，请稍候；已下载部分保留。")
                return
            self.component_dialog = None
            dialog.destroy()
            self.settings_button.focus_set()

        self.component_close_button = ttk.Button(actions, text="稍后", command=cancel)
        self.component_close_button.pack(side="right", padx=(0, 8))
        dialog.protocol("WM_DELETE_WINDOW", cancel)
        dialog.bind("<Escape>", lambda _: cancel() or "break")
        frame.bind("<Configure>", lambda e: [label.configure(wraplength=max(240, e.width - 48)) for label in (notice, details)])
        self._component_state()
        dialog.update_idletasks()
        dialog.minsize(max(540, actions.winfo_reqwidth() + 48), frame.winfo_reqheight())
        dialog.geometry(f"640x{max(420, frame.winfo_reqheight())}")
        dialog.wait_visibility()
        dialog.grab_set()
        self.component_download_button.focus_set()

    def download_components(self):
        if self.downloading_components or self.updating or not getattr(sys, "frozen", False):
            return
        items = missing(self.model_path, self.cuda_dir, DEVICE_MODES[self.device.get()])
        if not items:
            return
        self.downloading_components = True
        self.component_cancel = threading.Event()
        cancel = self.component_cancel
        self.component_percent.set(0)
        self.component_notice.set("正在准备下载…")
        self._component_state()

        def worker():
            try:
                install_components(items, cancel=cancel, progress=lambda text, percent: self.events.put((None, "component_progress", (text, percent))))
            except ComponentError as exc:
                self.events.put((None, "component_finished", str(exc)))
            except Exception:
                self.events.put((None, "component_finished", "组件准备失败，请稍后重试。"))
            else:
                self.events.put((None, "component_finished", "语音组件已准备完成，可开始采集。"))
        threading.Thread(target=worker, name="echomind-components", daemon=True).start()

    def _open_update_dialog(self):
        if self.component_dialog is not None:
            return
        if self.update_dialog is not None and self.update_dialog.winfo_exists():
            self.update_dialog.lift()
            return
        dialog = tk.Toplevel(self.root)
        self.update_dialog = dialog
        dialog.title("设置 · 检查更新 · EchoMind")
        dialog.transient(self.root)
        frame = ttk.Frame(dialog, padding=24, style="Card.TFrame")
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text=f"EchoMind {VERSION}", font=("Microsoft YaHei UI", 12, "bold"), style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 16))
        ttk.Label(frame, textvariable=self.update_notice, style="Card.TLabel", wraplength=420).grid(row=1, column=0, sticky="ew", pady=(0, 12))
        frame.rowconfigure(2, weight=1)
        notes_frame = ttk.Frame(frame, style="Card.TFrame")
        notes_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 16))
        self.update_notes_text = tk.Text(notes_frame, width=1, height=1, wrap="word", bg=COLORS["surface"], fg=COLORS["muted"], font=("Microsoft YaHei UI", 10))
        notes_scroll = ttk.Scrollbar(notes_frame, command=self.update_notes_text.yview)
        notes_scroll.pack(side="right", fill="y")
        self.update_notes_text.pack(side="left", fill="both", expand=True)
        self.update_notes_text.configure(yscrollcommand=notes_scroll.set)
        self._write(self.update_notes_text, self.update_notes.get(), replace=True)
        actions = ttk.Frame(frame, style="Card.TFrame")
        actions.grid(row=3, column=0, sticky="ew")
        self.install_update_button = ttk.Button(actions, text="立即更新", style="Primary.TButton", command=self.install_update, state="disabled")
        self.install_update_button.pack(side="right")
        self.recheck_update_button = ttk.Button(actions, text="检查更新", command=self.check_updates)
        self.recheck_update_button.pack(side="right", padx=(0, 8))

        def cancel():
            if self.updating:
                self.update_notice.set("正在下载或准备安装，请等待完成。")
                return
            self.update_dialog = None
            dialog.destroy()
            self.settings_button.focus_set()

        ttk.Button(actions, text="关闭", command=cancel).pack(side="right", padx=(0, 8))
        dialog.protocol("WM_DELETE_WINDOW", cancel)
        dialog.bind("<Escape>", lambda event: cancel() or "break")
        dialog.update_idletasks()
        minimum_width = max(540, actions.winfo_reqwidth() + 48)
        dialog.minsize(minimum_width, 360)
        dialog.geometry(f"{max(620, minimum_width)}x450")
        dialog.wait_visibility()
        dialog.grab_set()
        self._update_buttons()

    def _update_buttons(self):
        if self.update_dialog is not None and self.update_dialog.winfo_exists():
            self._write(self.update_notes_text, self.update_notes.get(), replace=True)
            self.recheck_update_button.configure(state="disabled" if self.checking_updates or self.updating else "normal")
            self.install_update_button.configure(state="normal" if self.available_release and not self.checking_updates and not self.updating and getattr(sys, "frozen", False) else "disabled")

    def check_updates(self, automatic=False):
        if self.destroyed or self.closing or self.updating or self.downloading_components:
            return
        if not automatic:
            self._open_update_dialog()
        if self.checking_updates:
            return
        self.checking_updates = True
        self.available_release = None
        self.update_notice.set(f"当前版本 {VERSION} · 正在检查更新…")
        self._update_buttons()

        def worker():
            try:
                release = check_release()
            except UpdateError as exc:
                self.events.put((None, "update_error", str(exc)))
            else:
                self.events.put((None, "update_checked", (release, automatic)))
        threading.Thread(target=worker, daemon=True).start()

    def install_update(self):
        if not self.available_release or self.updating or self.checking_updates or self.downloading_components:
            return
        if not getattr(sys, "frozen", False):
            self.update_notice.set("源码运行不支持替换，请使用打包后的 EXE。")
            return
        if not messagebox.askyesno("更新 EchoMind", f"下载并安装 {self.available_release.version}？\n下载完成后会停止当前会话、退出并重新启动。模型、配置与密钥保留。", parent=self.update_dialog or self.root):
            return
        self.updating = True
        self._update_buttons()
        self.update_notice.set("正在下载更新包… 0%")
        release = self.available_release

        def worker():
            try:
                stage = stage_update(release, progress=lambda percent: self.events.put((None, "update_progress", percent)))
            except UpdateError as exc:
                self.events.put((None, "update_error", str(exc)))
            else:
                self.events.put((None, "update_staged", stage))
        threading.Thread(target=worker, daemon=True).start()

    def _finish_update(self):
        try:
            launch_installer(self.pending_update)
        except UpdateError as exc:
            self.pending_update = None
            self.updating = False
            self.closing = False
            self._busy(False)
            self.update_notice.set(str(exc))
            self._update_buttons()
        else:
            self._destroy()

    def toggle_settings(self):
        if self.settings_button.instate(["disabled"]):
            return
        if self.settings_expanded:
            self._hide_settings_menu()
        else:
            self.settings_expanded = True
            self.settings_button.state(["selected"])
            self._position_settings_menu()
            self.shortcut_settings_button.focus_set()

    def _position_settings_menu(self):
        self.root.update_idletasks()
        width = max(self.settings_button.winfo_width(), self.settings_menu.winfo_reqwidth()) + 16
        height = self.settings_menu.winfo_reqheight() + 16
        x = self.settings_button.winfo_rootx() - self.root.winfo_rootx() - 8
        button_y = self.settings_button.winfo_rooty() - self.root.winfo_rooty()
        y = button_y + self.settings_button.winfo_height() + 4
        if y + height > self.root.winfo_height():
            y = button_y - height - 4
        x = max(0, min(x, max(0, self.root.winfo_width() - width)))
        y = max(0, min(y, max(0, self.root.winfo_height() - height)))
        self._place_settings_panel(self.settings_popup, self._settings_menu_window, x, y, width, height)
        if self.settings_submenu_expanded:
            self._position_settings_submenu()

    def _place_settings_panel(self, popup, window, x, y, width, height):
        popup.configure(width=width, height=height)
        popup.itemconfigure(window, width=width - 16)
        popup.delete("panel")
        for left, top, right, bottom, fill in ((4, 4, width - 2, height - 1, COLORS["border"]),
                                             (2, 2, width - 4, height - 4, COLORS["surface"])):
            radius = 8
            points = (left + radius, top, right - radius, top, right, top, right, top + radius,
                      right, bottom - radius, right, bottom, right - radius, bottom, left + radius, bottom,
                      left, bottom, left, bottom - radius, left, top + radius, left, top)
            popup.create_polygon(points, smooth=True, fill=fill, outline="", tags="panel")
        popup.tag_lower("panel")
        popup.place(x=x, y=y)
        tk.Misc.lift(popup)

    def _open_settings_submenu(self):
        self.settings_submenu_expanded = True
        self.maintenance_menu_button.state(["selected"])
        self._position_settings_submenu()
        self.update_menu_button.focus_set()

    def _position_settings_submenu(self):
        self.root.update_idletasks()
        width = self.settings_submenu.winfo_reqwidth() + 16
        height = self.settings_submenu.winfo_reqheight() + 16
        x = self.settings_popup.winfo_x() + self.settings_popup.winfo_width() - 4
        if x + width > self.root.winfo_width():
            x = self.settings_popup.winfo_x() - width + 4
        y = self.maintenance_menu_button.winfo_rooty() - self.root.winfo_rooty() - 8
        x = max(0, min(x, self.root.winfo_width() - width))
        y = max(0, min(y, self.root.winfo_height() - height))
        self._place_settings_panel(self.settings_subpopup, self._settings_submenu_window, x, y, width, height)

    def _close_settings_submenu(self, focus=True):
        self.settings_submenu_expanded = False
        self.settings_subpopup.place_forget()
        self.maintenance_menu_button.state(["!selected"])
        if focus:
            self.maintenance_menu_button.focus_set()

    def _settings_subnavigate(self, index):
        self._settings_subbuttons[index % len(self._settings_subbuttons)].focus_set()
        return "break"

    def _hide_settings_menu(self, *, focus=True):
        if self._settings_focus_timer is not None:
            self.root.after_cancel(self._settings_focus_timer)
            self._settings_focus_timer = None
        if not self.settings_expanded:
            return
        self.settings_expanded = False
        self._close_settings_submenu(focus=False)
        self.settings_popup.place_forget()
        self.settings_button.state(["!selected"])
        if focus:
            self.settings_button.focus_set()

    def _settings_action(self, action):
        self._hide_settings_menu(focus=False)
        action()

    def _settings_navigate(self, index):
        self._close_settings_submenu(focus=False)
        self._settings_buttons[index % len(self._settings_buttons)].focus_set()
        return "break"

    def _settings_select_key(self, event):
        event.widget.invoke()
        return "break"

    def _settings_open_key(self):
        if not self.settings_expanded:
            self.toggle_settings()
        return "break"

    def _settings_outside_click(self, event):
        if self.settings_expanded and event.widget != self.settings_button and not any(
                str(event.widget).startswith(str(popup)) for popup in (self.settings_popup, self.settings_subpopup)):
            self._hide_settings_menu(focus=False)

    def _settings_focus_out(self, event):
        if self.settings_expanded and self._settings_focus_timer is None:
            self._settings_focus_timer = self.root.after_idle(self._settings_check_focus)

    def _settings_check_focus(self):
        self._settings_focus_timer = None
        focus = None if self.destroyed else self.root.focus_displayof()
        if not self.destroyed and (focus is None or focus.winfo_toplevel() != self.root):
            self._hide_settings_menu(focus=False)

    def _settings_reposition(self, event):
        if event.widget == self.root and self.settings_expanded:
            self._position_settings_menu()

    def _follow_tick(self):
        if self.destroyed:
            return
        self.root.after_cancel(self._follow_timer)
        try:
            enabled = self.follow_enabled.get() and self.source.get() == "腾讯会议" and not self.closing
            if enabled and self.follower is None:
                api = WindowsAPI()
                self.follower = WindowFollower(api, api.root_handle(self.root.winfo_id()))
            if self.follower is not None:
                pid = self.processes.get(self.process.get(), 0)
                notice = self.follower.tick(pid, process_family(pid, self.process_details), self.follow_side.get(),
                                            enabled=enabled, paused=self.root.grab_current() is not None)
            else:
                notice = "窗口跟随已关闭"
            if self.follow_enabled.get() and self.source.get() != "腾讯会议":
                notice = "窗口跟随：麦克风模式下暂停"
            self.follow_notice.set(notice)
            changed = self._set_compact_mode(bool(enabled and self.follower and getattr(self.follower, "target", None)))
            if changed and self.compact_mode:
                self.follower.tick(pid, process_family(pid, self.process_details), self.follow_side.get(),
                                   enabled=enabled, paused=self.root.grab_current() is not None)
        except OSError as exc:
            self._set_compact_mode(False)
            self.follow_notice.set(f"窗口跟随暂不可用：{exc}")
        finally:
            if not self.destroyed:
                self._follow_timer = self.root.after(250, self._follow_tick)

    def open_follow_settings(self):
        if self.follow_dialog is not None and self.follow_dialog.winfo_exists():
            self.follow_dialog.lift()
            return
        if self.root.grab_current() is not None:
            return
        dialog = tk.Toplevel(self.root)
        self.follow_dialog = dialog
        dialog.title("设置 · 窗口跟随 · EchoMind")
        dialog.transient(self.root)
        frame = ttk.Frame(dialog, padding=24, style="Card.TFrame")
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        enabled = tk.BooleanVar(dialog, value=self.follow_enabled.get())
        side = tk.StringVar(dialog, value="左侧" if self.follow_side.get() == "left" else "右侧")
        ttk.Label(frame, text="跟随所选会议窗口", style="Card.TLabel", font=("Microsoft YaHei UI", 12, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 16))
        self.follow_toggle = ttk.Checkbutton(frame, text="启用窗口跟随及同步最小化", variable=enabled)
        self.follow_toggle.grid(row=1, column=0, sticky="w", pady=(0, 12))
        row = ttk.Frame(frame, style="Card.TFrame")
        row.grid(row=2, column=0, sticky="ew", pady=(0, 16))
        ttk.Label(row, text="停靠位置", style="Card.TLabel").pack(side="left", padx=(0, 12))
        self.follow_side_combo = ttk.Combobox(row, textvariable=side, values=("左侧", "右侧"), state="readonly", width=12)
        self.follow_side_combo.pack(side="left")
        hint = ttk.Label(frame, text="跟随主窗口上方所选进程的会议窗口，不要求先开始录制。\n会议移动、最小化和恢复时同步跟随，不抢焦点；手动最小化 EchoMind 不会被强制恢复。\n优先停靠指定侧，空间不足时尝试另一侧或限制在显示器工作区域，可能与会议重叠。\n麦克风模式与弹窗操作期间暂停跟随；取消勾选可恢复自由拖动。", style="Card.TLabel", foreground=COLORS["muted"], wraplength=480, justify="left")
        hint.grid(row=3, column=0, sticky="ew", pady=(0, 16))
        notice = tk.StringVar(dialog)
        ttk.Label(frame, textvariable=notice, style="Error.TLabel", wraplength=480).grid(row=4, column=0, sticky="ew", pady=(0, 8))
        actions = ttk.Frame(frame, style="Card.TFrame")
        actions.grid(row=5, column=0, sticky="ew")

        def cancel():
            self.follow_dialog = None
            dialog.destroy()
            self.settings_button.focus_set()

        def save():
            old = self.follow_enabled.get(), self.follow_side.get()
            self.follow_enabled.set(enabled.get())
            self.follow_side.set("left" if side.get() == "左侧" else "right")
            if self.save_config():
                cancel()
            else:
                self.follow_enabled.set(old[0])
                self.follow_side.set(old[1])
                notice.set(self.config_notice.get())

        self.follow_save_button = ttk.Button(actions, text="保存并应用", style="Primary.TButton", command=save)
        self.follow_save_button.pack(side="right")
        ttk.Button(actions, text="取消", command=cancel).pack(side="right", padx=(0, 8))
        dialog.protocol("WM_DELETE_WINDOW", cancel)
        dialog.bind("<Escape>", lambda _: cancel() or "break")
        dialog.update_idletasks()
        dialog.minsize(max(540, dialog.winfo_reqwidth()), dialog.winfo_reqheight())
        self.follow_toggle.focus_set()
        dialog.wait_visibility()
        dialog.grab_set()

    def open_settings(self):
        if self.settings_dialog is not None and self.settings_dialog.winfo_exists():
            self.settings_dialog.lift()
            self.settings_dialog.focus_set()
            return
        if self.settings_button.instate(["disabled"]):
            return
        dialog = tk.Toplevel(self.root)
        self.settings_dialog = dialog
        dialog.title("设置 · 快捷键 · EchoMind")
        dialog.transient(self.root)
        frame = ttk.Frame(dialog, padding=24, style="Card.TFrame")
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text="快捷键", style="Card.TLabel", font=("Microsoft YaHei UI", 12, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 12))
        draft = {key: tk.StringVar(dialog, value=variable.get()) for key, variable in self.shortcuts.items()}
        entries = []
        for row, (key, label) in enumerate(zip(DEFAULT_SHORTCUTS, ("提交回答", "开始采集", "停止采集", "切换窗口跟随"))):
            ttk.Label(frame, text=label, style="Card.TLabel").grid(row=row * 2 + 1, column=0, sticky="w", pady=(8, 4))
            entry = ttk.Entry(frame, textvariable=draft[key])
            entry.grid(row=row * 2 + 2, column=0, sticky="ew")
            entries.append(entry)
        ttk.Label(frame, text="默认：Enter 提交 · Ctrl+R 开始 · Ctrl+S 停止 · Ctrl+F 跟随。\n支持 Ctrl/Alt/Shift 组合键和 F1–F12，仅在主窗口有效。保存后下次启动自动读取。",
                  style="Card.TLabel", foreground=COLORS["muted"], wraplength=420).grid(row=9, column=0, sticky="ew", pady=(16, 8))
        notice = tk.StringVar(dialog)
        ttk.Label(frame, textvariable=notice, style="Card.TLabel", foreground=COLORS["error"], wraplength=420).grid(row=10, column=0, sticky="ew", pady=(0, 8))
        actions = ttk.Frame(frame, style="Card.TFrame")
        actions.grid(row=11, column=0, sticky="ew")

        def cancel():
            self.settings_dialog = None
            dialog.destroy()
            self.settings_button.focus_set()

        def save():
            values = {key: variable.get().strip() for key, variable in draft.items()}
            try:
                validate_shortcuts(values)
            except ValueError as exc:
                notice.set(str(exc))
                return
            previous = {key: variable.get() for key, variable in self.shortcuts.items()}
            for key, value in values.items():
                self.shortcuts[key].set(value)
            if self.save_config():
                cancel()
            else:
                for key, value in previous.items():
                    self.shortcuts[key].set(value)
                notice.set(self.config_notice.get())

        ttk.Button(actions, text="保存", style="Primary.TButton", command=save).pack(side="right")
        ttk.Button(actions, text="取消", command=cancel).pack(side="right", padx=(0, 8))
        dialog.protocol("WM_DELETE_WINDOW", cancel)
        dialog.bind("<Escape>", lambda event: cancel() or "break")
        dialog.update_idletasks()
        dialog.minsize(dialog.winfo_reqwidth(), dialog.winfo_reqheight())
        entries[0].focus_set()
        dialog.wait_visibility()
        dialog.grab_set()

    def save_config(self):
        values = {"api_base_url": self.api_url.get().strip(), "api_model": self.api_model.get().strip(),
                  "api_key": self.api_key.get().strip(), "jev_key": self.jev_key.get().strip(),
                  "answer_mode": ANSWER_MODES[self.mode.get()], "device": DEVICE_MODES[self.device.get()],
                  "source": self.source.get(), "jev_enabled": self.jev.get(),
                  "follow_enabled": self.follow_enabled.get(), "follow_side": self.follow_side.get(),
                  "asr_hotwords": self.asr_hotwords.get().strip(), "asr_model_path": self.asr_model_path.get().strip()}
        values["asr_engine"] = self._engine()
        values.update({key: variable.get().strip() for key, variable in self.tencent_fields.items()})
        values.update({key: variable.get().strip() for key, variable in self.shortcuts.items()})
        self.config_notice.set("正在保存配置…")
        try:
            values["question_threshold"] = self._question_threshold()
            validate_shortcuts(values)
            save_settings(values, self.settings_path)
        except (SettingsError, ValueError) as exc:
            self.config_notice.set(str(exc))
            self.error.set(str(exc))
            return False
        else:
            self._apply_shortcuts()
            self.config_notice.set("配置已保存，下次启动自动读取。密钥已加密。")
            return True

    def _question_threshold(self):
        try:
            percent = float(self.jev_threshold.get().strip())
        except ValueError:
            raise ValueError("Jev 触发阈值请输入 0–100 之间的百分比数字。") from None
        if not math.isfinite(percent) or not 0 <= percent <= 100:
            raise ValueError("Jev 触发阈值请输入 0–100 之间的百分比数字。")
        return percent / 100

    def _apply_shortcuts(self):
        values = {key: variable.get().strip() for key, variable in self.shortcuts.items()}
        sequences = validate_shortcuts(values)
        for widget, sequence in self._shortcut_bindings:
            widget.unbind(sequence)
        self._shortcut_bindings.clear()
        actions = {"shortcut_submit": self._submit_shortcut, "shortcut_start": self._start_shortcut, "shortcut_stop": self._stop_shortcut, "shortcut_follow": self._follow_shortcut}
        for key, sequence in sequences.items():
            widget = self.question_entry if key == "shortcut_submit" else self.root
            widget.bind(sequence, actions[key])
            self._shortcut_bindings.append((widget, sequence))
        if sequences["shortcut_submit"] == "<Return>":
            self.question_entry.bind("<KP_Enter>", self._submit_shortcut)
            self._shortcut_bindings.append((self.question_entry, "<KP_Enter>"))
        if sequences["shortcut_submit"] != "<Control-Return>":
            self.question_entry.bind("<Control-Return>", self._submit_shortcut)
            self._shortcut_bindings.append((self.question_entry, "<Control-Return>"))
        self.test_button.configure(text=f"提交回答 ({values['shortcut_submit']})")
        self.shortcut_hint.set(f"{values['shortcut_start']} 开始 · {values['shortcut_stop']} 停止 · {values['shortcut_follow']} 跟随")
        self.exit_follow_button.configure(text=f"退出跟随 ({values['shortcut_follow']})")

    def _source_changed(self):
        local = self.source.get() == "本机麦克风"
        self.refresh_button.pack_forget()
        if not local:
            self.refresh_button.pack(side="left", before=self.start_button, padx=(0, 8))
        self.start_button.configure(text="开始测试" if local else "开始采集")
        self.partial.set("点击开始测试后，请对着本机麦克风说话。" if local else "等待会议声音；当前不直接采集你的麦克风。")
        self.status.set("就绪 · 使用系统默认麦克风" if local else "就绪 · 请刷新会议进程")

    def _record_answer(self, event):
        key = f"{self.run_id}:{event.question_id}"
        if event.kind == "start":
            self.question_records[key] = QuestionRecord(event.question, time.strftime("%H:%M:%S"))
            self.history_tree.insert("", 0, iid=key, text=event.question,
                                     values=(self.question_records[key].timestamp, "生成中"))
            while len(self.question_records) > 100:
                oldest = next(iter(self.question_records))
                del self.question_records[oldest]
                self.history_tree.delete(oldest)
            if not self.history_tree.selection():
                self.history_tree.selection_set(key)
            self._write(self.compact_question_text, "\n\n".join(
                f"{item.timestamp}  {item.question}" for item in reversed(list(self.question_records.values()))
            ), replace=True)
            self.compact_question_text.yview_moveto(0)
        record = self.question_records.get(key)
        if record is None:
            return
        if event.kind == "delta":
            previous_length = len(record.answer)
            record.answer = (record.answer + event.text)[:20_000]
        elif event.kind == "done":
            record.answer = event.text[:20_000]
            record.status = "已完成"
            first = f"{event.first_token_ms:.0f} ms" if event.first_token_ms is not None else "无内容"
            record.metrics = f"模型首字 {first} · 总计 {event.elapsed_ms:.0f} ms"
        elif event.kind == "cancelled":
            record.status, record.metrics = "已停止", "回答已停止，保留已生成内容。"
        elif event.kind == "error":
            record.status, record.metrics = "失败", event.text
        self.history_tree.item(key, values=(record.timestamp, record.status))
        if key in self.history_tree.selection():
            if event.kind == "delta":
                self._write(self.history_text, record.answer[previous_length:])
            else:
                self._show_history(preserve_scroll=event.kind != "start")

    def _show_history(self, *, preserve_scroll=False):
        selected = self.history_tree.selection()
        if not selected or selected[0] not in self.question_records:
            self.history_question.set("请选择一条问题记录")
            self.history_status.set("仅保留本次运行最近 100 条；退出后清空，不写入磁盘。")
            self._write(self.history_text, "", replace=True)
            return
        record = self.question_records[selected[0]]
        self.history_question.set(record.question)
        self.history_status.set(f"{record.timestamp} · {record.status}\n{record.metrics}")
        view = self.history_text.yview()
        self._write(self.history_text, record.answer, replace=True)
        if preserve_scroll and view[1] < 0.98:
            self.history_text.yview_moveto(view[0])

    def _reveal_field(self, event):
        self.root.update_idletasks()
        canvas = self.settings_canvas
        total = canvas.bbox("all")
        if not total:
            return
        y = event.widget.winfo_rooty() - canvas.winfo_rooty() + canvas.canvasy(0)
        visible_top = canvas.canvasy(0)
        visible_bottom = visible_top + canvas.winfo_height()
        if y < visible_top:
            canvas.yview_moveto(max(0, y - 12) / max(1, total[3]))
        elif y + event.widget.winfo_height() > visible_bottom:
            canvas.yview_moveto((y + event.widget.winfo_height() + 12 - canvas.winfo_height()) / max(1, total[3]))

    def _entry(self, parent, row, label, variable, secret=False):
        ttk.Label(parent, text=label, style="Card.TLabel").grid(row=row, column=0, sticky="w", pady=(8, 4))
        entry = ttk.Entry(parent, textvariable=variable, show="*" if secret else "")
        entry.grid(row=row + 1, column=0, sticky="ew")
        self.form_controls.append((entry, "normal"))
        return entry

    def _combo(self, parent, row, label, variable, values):
        ttk.Label(parent, text=label, style="Card.TLabel").grid(row=row, column=0, sticky="w", pady=(8, 4))
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        combo.grid(row=row + 1, column=0, sticky="ew")
        self.form_controls.append((combo, "readonly"))
        return combo

    def _text(self, parent):
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.pack(fill="both", expand=True, pady=(12, 0))
        text = tk.Text(frame, wrap="word", font=("Microsoft YaHei UI", 11), width=20,
                       bg=COLORS["surface"], fg=COLORS["text"], relief="flat", padx=4,
                       spacing1=4, spacing3=8, height=8, state="disabled", takefocus=True,
                       selectbackground=COLORS["tint"], selectforeground=COLORS["text"],
                       highlightthickness=1, highlightbackground=COLORS["surface"], highlightcolor=COLORS["primary"])
        scroll = ttk.Scrollbar(frame, command=text.yview)
        scroll.pack(side="right", fill="y")
        text.pack(fill="both", expand=True)
        text.configure(yscrollcommand=scroll.set)
        return text

    def _write(self, widget, text, *, replace=False):
        at_bottom = widget.yview()[1] >= 0.98
        widget.configure(state="normal")
        if replace:
            widget.delete("1.0", "end")
        widget.insert("end", text)
        lines = int(widget.index("end-1c").split(".")[0])
        if lines > 400:
            widget.delete("1.0", f"{lines - 400 + 1}.0")
        widget.configure(state="disabled")
        if widget is self.caption_text:
            self._caption_selection_changed()
        if at_bottom or replace:
            widget.see("end")

    def _reveal_keys(self):
        mask = "" if self.show_keys.get() else "*"
        self.key_entry.configure(show=mask)
        self.jev_entry.configure(show=mask)

    def refresh(self):
        if self.refreshing or (self.session and self.session.is_running):
            return
        self.refreshing = True
        self.refresh_button.configure(state="disabled")
        self.status.set("正在查找腾讯会议…")

        def discover():
            try:
                self.events.put((None, "processes", self.discover()))
            except Exception:
                self.events.put((None, "discovery_error", "无法查找会议进程，请启动腾讯会议后刷新。"))

        threading.Thread(target=discover, name="echomind-discovery", daemon=True).start()

    @staticmethod
    def _asr_topic_for(words):
        return "通用" if not words.strip() else "前端开发" if words == DEFAULT_HOTWORDS else "自定义"

    def _asr_topic_changed(self, event=None):
        if self.asr_topic.get() == "通用":
            self.asr_hotwords.set("")
        elif self.asr_topic.get() == "前端开发":
            self.asr_hotwords.set(DEFAULT_HOTWORDS)

    def _choose_asr_model(self):
        path = filedialog.askdirectory(parent=self.root, title=f"选择 {self._engine()} 语音模型目录")
        if path:
            self.asr_model_path.set(path)

    def _selected_model_path(self):
        if self._engine() == "tencent":
            return self.model_path  # Unused by cloud ASR; do not resolve or require local models.
        custom = self.asr_model_path.get().strip()
        if custom:
            return Path(custom)
        if self._engine() == "whisper":
            return self.model_path
        root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[3]
        return experimental_model_path(root, self._engine())

    def _engine(self):
        return next(key for key, label in ENGINE_LABELS.items() if label == self.asr_engine.get())

    def _engine_hint(self):
        if self._engine() == "qwen3-asr-1.7b":
            return "本地 GPU 识别，约需 4 GB 显存。自动复用已部署模型和独立运行环境；不自动下载。\n\n停顿后显示确认字幕，不传热词、不做逐字预览。语音检测和异常拦截仍启用。"
        if self._engine() == "tencent":
            return "通过腾讯云流式识别，不需要本地模型或 GPU。请先开通语音识别服务；中英模型 16k_zh_en 与通用模型计费不同。凭证随“保存配置”加密保存。断网会停止，不会自动切换引擎。"
        if self._engine() == "whisper":
            return "留空使用默认 Turbo。可选完整的 CTranslate2 large-v3 本地目录，不自动下载。\n\n保留最近 15 秒音频，仅提交本次发言；通常停顿 600 毫秒确认，未说完或任务要求最多等约 2 秒。可疑结果额外无热词复核，仍不确定时需手动核对。"
        return ("实验引擎固定使用 CPU，当前接入不支持技术热词。中英技术词未必比 Whisper 准确。\n\n"
                "留空复用本机对应模型；缺少时请先准备模型或选择目录，不自动下载。切换引擎会清空自定义模型路径。")

    def _asr_engine_changed(self, event=None):
        self.asr_model_path.set("")
        self.asr_engine_hint.set(self._engine_hint())
        self.cloud_consent.set(False)
        self._sync_asr_fields()

    def _sync_asr_fields(self):
        cloud = self._engine() == "tencent"
        self.local_asr_fields.grid_remove() if cloud else self.local_asr_fields.grid()
        self.cloud_asr_fields.grid() if cloud else self.cloud_asr_fields.grid_remove()
        self.caption_source.set("实时转写 / 腾讯云识别（上传音频）" if cloud else "实时转写 / 本机识别")
        self.root.after_idle(self._fit_settings_page)

    def _config(self, *, text_test=False):
        local = not text_test and self.source.get() == "本机麦克风"
        mode = ANSWER_MODES[self.mode.get()]
        if text_test and mode == "off":
            raise ValueError("提交问题前，请把回答方式改为“离线演练”或“DeepSeek 在线回答”。")
        pid = 0 if text_test or local else self.processes.get(self.process.get(), 0)
        if not text_test and not local and pid <= 0:
            raise ValueError("请启动腾讯会议，刷新并选择一个会议进程。")
        if not text_test and self._engine() != "tencent" and (self.asr_model_path.get().strip() or self._engine() != "whisper"):
            selected = self._selected_model_path()
            if not selected.is_absolute():
                raise ValueError("模型目录需要使用绝对路径。")
            validate_model(selected, self._engine())
        tencent = TencentConfig(**{key.removeprefix("tencent_"): variable.get().strip() for key, variable in self.tencent_fields.items()}, consent=self.cloud_consent.get())
        if not text_test and self._engine() == "tencent":
            tencent.validate()
        return SessionConfig(pid=pid, capture_exe=self.capture_exe, model_path=self._selected_model_path(),
                             cuda_dir=self.cuda_dir, device="cuda" if self._engine() == "qwen3-asr-1.7b" else DEVICE_MODES[self.device.get()] if self._engine() == "whisper" else "cpu",
                             answer_mode=mode, question_gate="jev" if self.jev.get() else "rules",
                             question_threshold=self._question_threshold(),
                             api_base_url=self.api_url.get().strip(), api_model=self.api_model.get().strip(),
                             api_key=self.api_key.get().strip(), jev_key=self.jev_key.get().strip(),
                             microphone=local, asr_hotwords=self.asr_hotwords.get().strip(), asr_engine=self._engine(), tencent=tencent)

    def _start(self, question=None):
        if self.session and self.session.is_running:
            return
        if self.downloading_components:
            self.error.set("请等待组件准备完成，或取消下载后再运行。")
            return
        if question is None and getattr(sys, "frozen", False):
            self.capture_exe, self.model_path, self.cuda_dir = asset_paths()
            if self._engine() == "whisper" and not self.asr_model_path.get().strip() and missing(self.model_path, self.cuda_dir, DEVICE_MODES[self.device.get()]):
                self.open_components()
                return
        self.error.set("")
        try:
            if question is not None and not question.strip():
                raise ValueError("请输入要测试的完整问题。")
            config = self._config(text_test=question is not None)
            self.run_id += 1
            run_id = self.run_id
            self.session = self.session_factory(config, lambda kind, payload: self.events.put((run_id, kind, payload)))
            self.session.start(question=question)
        except (ValueError, RuntimeError) as exc:
            self.error.set(str(exc))
            return
        self._busy(True)
        self._clear_repair()
        self.status.set("正在启动…")

    def start_capture(self):
        self._start()

    def _submit_shortcut(self, event=None):
        if not self.closing and not self.destroyed:
            self.test_button.invoke()
        return "break"

    def _start_shortcut(self, event=None):
        if not self.closing and not self.destroyed:
            self.start_button.invoke()
        return "break"

    def _stop_shortcut(self, event=None):
        if not self.closing and not self.destroyed:
            self.stop_button.invoke()
        return "break"

    def _follow_shortcut(self, event=None):
        if self.closing or self.destroyed or self.root.grab_current() is not None:
            return "break"
        if self.source.get() != "腾讯会议":
            self.error.set("窗口跟随仅支持腾讯会议，请先切换输入来源。")
            return "break"
        self.follow_enabled.set(not self.follow_enabled.get())
        if self.follow_enabled.get() and not self.processes:
            self.refresh()
        self._follow_tick()
        return "break"

    def start_test(self):
        question = self.test_question.get().strip()
        if self.session and self.session.is_running:
            self.error.set("")
            try:
                self.session.submit_question(question)
            except (ValueError, RuntimeError) as exc:
                self.error.set(str(exc))
            else:
                self._clear_repair()
            return
        self._start(question)

    def _busy(self, busy):
        if busy:
            self._hide_settings_menu(focus=False)
        for widget, idle_state in self.form_controls:
            widget.configure(state="disabled" if busy else idle_state)
        for widget in (self.start_button, self.refresh_button):
            widget.configure(state="disabled" if busy else "normal")
        self.test_button.configure(state="disabled" if busy and ANSWER_MODES[self.mode.get()] == "off" else "normal")
        self.stop_button.configure(state="normal" if busy else "disabled")

    def stop(self):
        if self.session:
            self.session.stop()
            self.status.set("正在停止并释放资源…")
            self.stop_button.configure(state="disabled")

    def _handle(self, kind, payload):
        if kind.startswith("component_"):
            if self.destroyed:
                return
            if kind == "component_progress":
                self.component_notice.set(payload[0])
                self.component_percent.set(payload[1])
            elif kind == "component_finished":
                self.downloading_components = False
                self.capture_exe, self.model_path, self.cuda_dir = asset_paths()
                self._component_state()
                self.component_notice.set(payload)
                if self.closing:
                    self._destroy()
            return
        if kind.startswith("update_"):
            if self.destroyed:
                return
            if kind == "update_checked":
                self.checking_updates = False
                release, automatic = payload
                self.available_release = release
                self.update_notice.set(f"发现新版 {release.version} · 当前 {VERSION}" if release else f"当前 {VERSION} 已是最新版本。")
                self.update_notes.set((release.notes[:1200] + "\n\n更新会保留模型、CUDA 和本机配置。") if release else "无需更新。")
                if release and automatic and self.root.grab_current() is None and self.settings_dialog is None and self.update_dialog is None and not self.closing:
                    self._open_update_dialog()
            elif kind == "update_error":
                self.checking_updates = False
                self.updating = False
                self.update_notice.set(payload)
            elif kind == "update_progress":
                self.update_notice.set(f"正在下载并校验更新包… {payload}%")
            elif kind == "update_staged":
                self.pending_update = payload
                self.closing = True
                self.update_notice.set("校验完成，正在停止会话并安装…")
                self._busy(True)
                if self.session and self.session.is_running:
                    self.stop()
                else:
                    self._finish_update()
            self._update_buttons()
        elif kind in ("screenshot_ready", "screenshot_error"):
            if self.closing or self.destroyed:
                return
            self.importing_screenshot = False
            self.screenshot_button.configure(state="normal")
            self.import_screenshot_button.configure(state="normal")
            if kind == "screenshot_error":
                self.screenshot_notice.set(payload)
            else:
                self.screenshot_notice.set("识别完成，请在弹窗中核对或选择题目。")
                self._preview_screenshot(payload)
            return
        if kind in ("processes", "discovery_error"):
            self.refreshing = False
            running = bool(self.session and self.session.is_running)
            self.refresh_button.configure(state="disabled" if running else "normal")
            if kind == "discovery_error":
                if self.source.get() == "本机麦克风":
                    return
                self.error.set(payload)
                self.status.set("未找到会议")
                return
            self.process_details = payload
            self.processes = {f"{item['Name']} · PID {item['ProcessId']}": int(item["ProcessId"]) for item in payload}
            chosen = preferred_pid(payload)
            self.process.set(next((label for label, pid in self.processes.items() if pid == chosen), ""))
            self.application_notice.set("已连接腾讯会议 · 主进程已自动选择" if chosen else "未找到腾讯会议 · 启动应用后点击刷新")
            if not running and self.source.get() == "腾讯会议":
                self.status.set(self.application_notice.get())
        elif kind == "status":
            self.status.set(payload)
        elif kind == "error":
            self.error.set(payload)
        elif kind == "transcript":
            if payload.kind == "rejected":
                self.partial.set("已拦截疑似识别异常：" + payload.rejection_reason + "；请重说或手动核对。")
                self.pending_repair = payload.text
                self.repair_notice.set(f"疑似识别异常：{payload.rejection_reason}\n未确认原文：{payload.text[:180]}\n未发送给 Jev 或回答模型。可点击核对按钮编辑完整原文后提交，或重新说一遍。")
                self.repair_panel.pack(fill="x", before=self.answer_text.master, pady=(4, 0))
            elif payload.kind == "partial":
                self.partial.set("实时：" + payload.text)
            else:
                prefix = "确定（低可信，待复核）：" if getattr(payload, "suspect", False) else "确定："
                self.partial.set(prefix + payload.text)
                self._write(self.caption_text, f"{time.strftime('%H:%M:%S')}  {payload.text}\n")
                self._write(self.compact_caption_text, f"{time.strftime('%H:%M:%S')}  {payload.text}\n")
                if getattr(payload, "original_text", ""):
                    note = f"  音频复识别前：{payload.original_text}\n"
                    self._write(self.caption_text, note)
                    self._write(self.compact_caption_text, note)
        elif kind == "answer":
            event = payload
            self._record_answer(event)
            if event.kind == "start":
                self._clear_repair()
                self.detection.set("已识别提问，正在生成回答。")
                self.answer_id = event.question_id
                self.question.set(event.question)
                self.metrics.set("正在生成回答…")
                self._write(self.answer_text, "", replace=True)
            elif event.kind == "delta" and event.question_id == self.answer_id:
                self._write(self.answer_text, event.text)
                self.detection.set("正在流式生成回答…")
                if event.first_token_ms is not None:
                    self.metrics.set(f"模型首字 {event.first_token_ms:.0f} ms · 正在流式输出…")
            elif event.kind == "done" and event.question_id == self.answer_id:
                self.detection.set("回答已完成。")
                first = f"{event.first_token_ms:.0f} ms" if event.first_token_ms is not None else "无内容"
                decision = f" · 语义判断 {event.decision_ms:.0f} ms" if event.decision_ms is not None else ""
                self.metrics.set(f"模型首字 {first} · 总计 {event.elapsed_ms:.0f} ms{decision}")
            elif event.kind == "checking":
                self.detection.set("正在语义确认：" + event.question)
                self.status.set("正在判断是否需要回答…")
            elif event.kind == "reviewing":
                self._clear_repair()
                self.detection.set("疑似识别有误，正在复核问题；尚未生成回答。")
            elif event.kind == "needs_confirmation":
                self.pending_repair = event.text or event.question
                self.repair_notice.set(f"原识别：{event.question}\n建议：{self.pending_repair}\n{event.reason}\n请核对后提交；原字幕保留。")
                self.repair_panel.pack(fill="x", before=self.answer_text.master, pady=(4, 0))
                self.detection.set("问题需要确认，未自动回答。")
            elif event.kind == "repair_error":
                self.detection.set("问题复核失败，未自动回答；可选中字幕编辑后提交。")
                self.error.set(event.text)
            elif event.kind == "untriggered":
                self.detection.set(event.text)
            elif event.kind == "skipped":
                self.detection.set(event.text)
                self.status.set("已忽略非明确提问，继续监听")
            elif event.kind in ("error", "gate_error"):
                self.detection.set("问题判断失败，未触发回答。" if event.kind == "gate_error" else "已识别问题，但回答生成失败。")
                self.error.set(event.text)
            elif event.kind == "cancelled" and event.question_id == self.answer_id:
                self.detection.set("回答已停止。")
                self.metrics.set("回答已停止")
        elif kind == "finished":
            self.session = None
            self._busy(False)
            self.status.set("已停止" if not self.error.get() else "已停止 · 请检查提示")
            if self.closing:
                if self.pending_update is not None:
                    self._finish_update()
                else:
                    self._destroy()

    def _drain(self):
        started = time.perf_counter()
        deltas = 0
        for _ in range(250):
            try:
                run_id, kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if run_id is None or run_id == self.run_id:
                self._handle(kind, payload)
                if kind == "answer" and payload.kind == "delta":
                    deltas += 1
            if self.destroyed:
                return
            # Yield to Tk painting/input before a burst reaches its final event.
            if deltas >= 4 or time.perf_counter() - started >= 0.008:
                break
        self._poll = self.root.after(8 if not self.events.empty() else 16, self._drain)

    def close(self):
        if self.updating:
            messagebox.showinfo("正在更新", "请等待下载和安装完成。", parent=self.root)
            return
        if self.closing:
            return
        if self.downloading_components:
            self.closing = True
            self.component_cancel.set()
            self.component_notice.set("正在取消下载并退出…")
            return
        if self.session and self.session.is_running:
            if not messagebox.askyesno("退出 EchoMind", "当前正在运行。停止采集并退出？", parent=self.root):
                return
            self.closing = True
            self.stop()
        else:
            self._destroy()

    def _destroy(self):
        self._hide_settings_menu(focus=False)
        self.component_cancel.set()
        self.destroyed = True
        self.root.after_cancel(self._poll)
        self.root.after_cancel(self._refresh)
        self.root.after_cancel(self._follow_timer)
        if self._update_timer is not None:
            self.root.after_cancel(self._update_timer)
        if self._component_timer is not None:
            self.root.after_cancel(self._component_timer)
        self.api_key.set("")
        self.jev_key.set("")
        self.question_records.clear()
        self.session = None
        self.root.destroy()


def main():
    if os.name == "nt":
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    EchoMindWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
