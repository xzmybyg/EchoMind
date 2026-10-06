"""Windowed EchoMind MVP. Widgets stay on the Tk thread."""

from __future__ import annotations

import os
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
from .desktop_session import DesktopSession, SessionConfig
from .settings import SettingsError, config_path, load_settings, save_settings
from .screenshot import ScreenshotError, clipboard_has_image, recognize_clipboard, recognize_screenshot
from .shortcuts import DEFAULT_SHORTCUTS, validate_shortcuts
from .updater import UpdateError, check_release, launch_installer, stage_update
from .version import VERSION


COLORS = {"background": "#F5F5F5", "surface": "#FFFFFF", "text": "#141414",
          "muted": "#595959", "primary": "#1677FF", "border": "#D9D9D9",
          "tint": "#E6F4FF", "active": "#0958D9", "error": "#B42318",
          "control_border": "#8C8C8C", "disabled": "#737373", "table_header": "#FAFAFA",
          "pressed": "#003EB3"}
ANSWER_MODES = {"仅字幕（本地）": "off", "离线演练": "demo", "DeepSeek 在线回答": "live"}
DEVICE_MODES = {"GPU（推荐）": "cuda", "CPU（较慢）": "cpu"}
MODE_HINTS = {"仅字幕（本地）": "仅显示字幕，不生成回答，也不访问云服务。",
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
        return root / "capture" / "EchoMind.Capture.exe", root / "models" / "large-v3-turbo", root / "cuda"
    root = Path(__file__).resolve().parents[3]
    return (root / "native/audio-capture/bin/Release/net9.0/EchoMind.Capture.exe",
            root / ".models/large-v3-turbo", root / ".runtime/cuda12")


def preferred_pid(processes: list[dict]) -> int | None:
    ids = {int(item["ProcessId"]) for item in processes}
    mains = [item for item in processes if str(item["Name"]).lower() == "wemeetapp.exe"]
    roots = [item for item in mains if item.get("ParentProcessId") not in ids]
    selected = (roots or mains or processes)
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
        self.answer_id = None
        self.question_records: dict[str, QuestionRecord] = {}
        self.history_question = tk.StringVar(value="暂无问题记录，检测到问题并触发回答后会显示在这里。")
        self.history_status = tk.StringVar(value="仅保留本次运行最近 100 条；退出后清空，不写入磁盘。")
        self.form_controls = []
        self.settings_path = Path(settings_path) if settings_path is not None else config_path()
        try:
            saved = load_settings(self.settings_path)
            settings_notice = "已读取本地配置" if saved else "首次填写后点击保存，下次启动自动读取。"
        except SettingsError as exc:
            saved = {}
            settings_notice = str(exc)
        self.config_notice = tk.StringVar(value=settings_notice)
        self.capture_exe, self.model_path, self.cuda_dir = asset_paths()
        self.status = tk.StringVar(value="就绪 · 请选择会议进程，或先测试回答")
        self.error = tk.StringVar()
        self.partial = tk.StringVar(value="等待会议声音；当前不直接采集你的麦克风。")
        self.question = tk.StringVar(value="等待一个完整问题")
        self.metrics = tk.StringVar(value="模型首字耗时不包含语音识别和停顿等待")
        self.detection = tk.StringVar()
        self.repair_notice = tk.StringVar()
        self.pending_repair = ""
        self.process = tk.StringVar()
        self.source = tk.StringVar(value=saved.get("source", "腾讯会议"))
        self.mode = tk.StringVar(value=next((label for label, value in ANSWER_MODES.items() if value == saved.get("answer_mode")), "仅字幕（本地）"))
        self.mode_hint = tk.StringVar(value=MODE_HINTS[self.mode.get()])
        self.mode.trace_add("write", lambda *_: self._mode_changed())
        self.device = tk.StringVar(value=next((label for label, value in DEVICE_MODES.items() if value == saved.get("device")), "GPU（推荐）"))
        self.api_url = tk.StringVar(value=os.environ.get("ECHOMIND_LLM_BASE_URL", saved.get("api_base_url", "https://api.deepseek.com")))
        self.api_model = tk.StringVar(value=os.environ.get("ECHOMIND_LLM_MODEL", saved.get("api_model", "deepseek-flash")))
        self.api_key = tk.StringVar(value=os.environ.get("ECHOMIND_LLM_API_KEY", saved.get("api_key", "")))
        self.jev_key = tk.StringVar(value=os.environ.get("ECHOMIND_JEV_API_KEY", saved.get("jev_key", "")))
        self.jev = tk.BooleanVar(value=saved.get("jev_enabled", False))
        self.consent = tk.BooleanVar(value=False)
        self.show_keys = tk.BooleanVar(value=False)
        self.test_question = tk.StringVar()
        self.shortcuts = {key: tk.StringVar(value=saved.get(key, default)) for key, default in DEFAULT_SHORTCUTS.items()}
        self.shortcut_hint = tk.StringVar()
        self._shortcut_bindings = []
        self.settings_dialog = None
        self.settings_expanded = False
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
        style.configure("Settings.TNotebook.Tab", width=4)
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
        outer = ttk.Frame(self.root, padding=24)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 16))
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
        ttk.Label(settings, text="运行设置", font=("Microsoft YaHei UI", 12, "bold"), style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 12))
        self.settings_tabs = ttk.Notebook(settings, style="Settings.TNotebook")
        self.settings_tabs.grid(row=1, column=0, sticky="ew")
        basic, model, semantic = [ttk.Frame(self.settings_tabs, style="Card.TFrame", padding=(0, 12)) for _ in range(3)]
        for page, title in zip((basic, model, semantic), ("基本", "模型", "Jev")):
            page.columnconfigure(0, weight=1)
            self.settings_tabs.add(page, text=title)
        self._combo(basic, 0, "回答方式", self.mode, list(ANSWER_MODES))
        ttk.Label(basic, textvariable=self.mode_hint, style="Hint.TLabel", wraplength=225).grid(row=2, column=0, sticky="ew", pady=12)
        self._combo(basic, 3, "识别设备", self.device, list(DEVICE_MODES))
        ttk.Label(basic, text="切换模式前请先停止采集。\n默认自动选择腾讯会议主进程。", style="Card.TLabel",
                  foreground=COLORS["muted"], wraplength=240).grid(row=5, column=0, sticky="w", pady=(16, 0))
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
        ttk.Label(semantic, text="需在线回答模式及独立 TypeSafe 密钥。本地规则未命中的确定字幕交给 Jev 判断，增加调用量和延迟。",
                  style="Hint.TLabel", wraplength=225).grid(row=3, column=0, sticky="ew", pady=12)
        self.save_button = ttk.Button(settings, text="保存配置（含加密密钥）", command=self.save_config)
        self.save_button.grid(row=2, column=0, sticky="ew", pady=(12, 8))
        self.form_controls.append((self.save_button, "normal"))
        ttk.Label(settings, textvariable=self.config_notice, style="Card.TLabel", foreground=COLORS["muted"], wraplength=240).grid(row=3, column=0, sticky="ew")
        ttk.Separator(settings).grid(row=4, column=0, sticky="ew", pady=16)
        ttk.Label(settings, text="使用与隐私", style="Card.TLabel", font=("Microsoft YaHei UI", 11, "bold")).grid(row=5, column=0, sticky="w")
        consent_footer = ttk.Frame(sidebar, style="Card.TFrame", padding=(16, 12))
        consent_footer.grid(row=1, column=0, columnspan=2, sticky="ew")
        consent = tk.Checkbutton(consent_footer, text="已取得参与者授权，\n允许处理会议内容", variable=self.consent,
                                 bg=COLORS["surface"], fg=COLORS["text"], activebackground=COLORS["tint"],
                                 justify="left", anchor="w", font=("Microsoft YaHei UI", 10), wraplength=235)
        consent.pack(fill="x")
        self.form_controls.append((consent, "normal"))
        self.settings_menu = ttk.Frame(consent_footer, style="Panel.TFrame", padding=8)
        self.shortcut_settings_button = ttk.Button(self.settings_menu, text="快捷键设置", command=self.open_settings)
        self.shortcut_settings_button.pack(fill="x")
        self.shortcut_settings_button.bind("<Escape>", lambda event: self.toggle_settings() or "break")
        self.form_controls.append((self.shortcut_settings_button, "normal"))
        self.update_menu_button = ttk.Button(self.settings_menu, text="检查更新", command=self.check_updates)
        self.update_menu_button.pack(fill="x", pady=(8, 0))
        self.form_controls.append((self.update_menu_button, "normal"))
        self.settings_button = ttk.Button(consent_footer, text="设置  +", style="Primary.TButton", command=self.toggle_settings)
        self.settings_button.pack(fill="x", pady=(12, 0))
        self.form_controls.append((self.settings_button, "normal"))
        ttk.Label(settings, text="在线回答发送问题及近期问答；疑似误听复核另发送少量近期字幕。\nJev 另发送少量近期转写，不上传音频。",
                  style="Card.TLabel", foreground=COLORS["muted"], wraplength=240).grid(row=6, column=0, sticky="w", pady=(8, 0))
        main = ttk.Frame(body)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)
        toolbar = ttk.Frame(main, style="Panel.TFrame", padding=16)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        source_row = ttk.Frame(toolbar, style="Card.TFrame")
        source_row.pack(fill="x", pady=(0, 8))
        ttk.Label(source_row, text="输入来源", style="Card.TLabel", font=("Microsoft YaHei UI", 11, "bold")).pack(side="left", padx=(0, 12))
        source_combo = ttk.Combobox(source_row, textvariable=self.source, values=("腾讯会议", "本机麦克风"), state="readonly", width=12)
        source_combo.pack(side="left")
        source_combo.bind("<<ComboboxSelected>>", lambda _: self._source_changed())
        self.form_controls.append((source_combo, "readonly"))
        row = ttk.Frame(toolbar, style="Card.TFrame")
        row.pack(fill="x")
        inputs = ttk.Frame(row, style="Card.TFrame")
        inputs.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.meeting_row = ttk.Frame(inputs, style="Card.TFrame")
        self.meeting_row.pack(fill="x")
        self.microphone_row = ttk.Frame(inputs, style="Card.TFrame")
        ttk.Label(self.microphone_row, text="系统默认输入设备 · 无需打开会议", style="Card.TLabel", wraplength=350).pack(anchor="w", pady=8)
        self.process_combo = ttk.Combobox(self.meeting_row, textvariable=self.process, state="readonly")
        self.process_combo.pack(side="left", fill="x", expand=True)
        self.refresh_button = ttk.Button(self.meeting_row, text="刷新", command=self.refresh)
        self.refresh_button.pack(side="left", padx=8)
        self.start_button = ttk.Button(row, text="开始采集", style="Primary.TButton", command=self.start_capture)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(row, text="停止", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        ttk.Label(toolbar, textvariable=self.shortcut_hint, style="Card.TLabel",
                  foreground=COLORS["muted"]).pack(anchor="w", pady=(8, 0))
        ttk.Label(main, textvariable=self.error, style="Error.TLabel", wraplength=660).grid(row=1, column=0, sticky="ew", pady=(0, 8))
        panels = ttk.Panedwindow(main, orient="horizontal")
        panels.grid(row=2, column=0, sticky="nsew")
        captions = ttk.Frame(panels, style="Panel.TFrame", padding=16)
        self.answer_tabs = ttk.Notebook(panels)
        answers = ttk.Frame(self.answer_tabs, style="Card.TFrame", padding=16)
        history = ttk.Frame(self.answer_tabs, style="Card.TFrame", padding=16)
        self.answer_tabs.add(answers, text="当前回答")
        self.answer_tabs.add(history, text="问题记录")
        panels.add(captions, weight=1)
        panels.add(self.answer_tabs, weight=1)
        ttk.Label(captions, text="会议字幕", style="Card.TLabel", font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w")
        ttk.Label(captions, text="实时转写 / 本机识别", style="Card.TLabel", foreground=COLORS["muted"]).pack(anchor="w", pady=(4, 8))
        partial_label = ttk.Label(captions, textvariable=self.partial, style="Hint.TLabel", wraplength=320)
        partial_label.pack(side="bottom", fill="x", pady=(8, 0))
        self.add_caption_button = ttk.Button(captions, text="选中字幕加入问题", command=self.add_selected_caption)
        self.add_caption_button.pack(anchor="w", pady=(4, 0))
        self.caption_text = self._text(captions)
        self.caption_text.configure(exportselection=False)
        self.caption_text.bind("<Control-Return>", lambda _: self.add_selected_caption() or "break")
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
        ttk.Button(self.repair_panel, text="使用建议编辑", command=self.edit_repair).pack(anchor="w", pady=(6, 0))
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
        test = ttk.Frame(main, style="Panel.TFrame", padding=16)
        test.grid(row=3, column=0, sticky="ew", pady=(16, 0))
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
        ttk.Label(test, textvariable=self.screenshot_notice, style="Card.TLabel", foreground=COLORS["muted"]).pack(anchor="w", pady=(8, 0))
        for widget, _ in self.form_controls:
            if str(widget).startswith(str(settings)):
                widget.bind("<FocusIn>", self._reveal_field)

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

    def add_selected_caption(self):
        try:
            selected = self.caption_text.get("sel.first", "sel.last")
        except tk.TclError:
            self.error.set("请先在字幕中拖选文字，再点击“选中字幕加入问题”。")
            return
        selected = re.sub(r"(?m)^\s*\d{2}:\d{2}:\d{2}\s+", "", selected)
        selected = " ".join(selected.split())
        if not selected:
            self.error.set("选中的字幕没有可用文字。")
            return
        draft = self.test_question.get().strip()
        self._edit_question((draft + " " + selected).strip())

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

    def _open_update_dialog(self):
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
        if self.destroyed or self.closing or self.updating:
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
        if not self.available_release or self.updating or self.checking_updates:
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
        self.settings_expanded = not self.settings_expanded
        if self.settings_expanded:
            self.settings_menu.pack(fill="x", pady=(12, 0), before=self.settings_button)
            self.settings_button.configure(text="设置  −")
            self.shortcut_settings_button.focus_set()
        else:
            self.settings_menu.pack_forget()
            self.settings_button.configure(text="设置  +")
            self.settings_button.focus_set()

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
        for row, (key, label) in enumerate(zip(DEFAULT_SHORTCUTS, ("提交回答", "开始采集", "停止采集"))):
            ttk.Label(frame, text=label, style="Card.TLabel").grid(row=row * 2 + 1, column=0, sticky="w", pady=(8, 4))
            entry = ttk.Entry(frame, textvariable=draft[key])
            entry.grid(row=row * 2 + 2, column=0, sticky="ew")
            entries.append(entry)
        ttk.Label(frame, text="默认：Enter 提交 · Ctrl+R 开始 · Ctrl+S 停止。\n支持 Ctrl/Alt/Shift 组合键和 F1–F12，仅在主窗口有效。保存后下次启动自动读取。",
                  style="Card.TLabel", foreground=COLORS["muted"], wraplength=420).grid(row=7, column=0, sticky="ew", pady=(16, 8))
        notice = tk.StringVar(dialog)
        ttk.Label(frame, textvariable=notice, style="Card.TLabel", foreground=COLORS["error"], wraplength=420).grid(row=8, column=0, sticky="ew", pady=(0, 8))
        actions = ttk.Frame(frame, style="Card.TFrame")
        actions.grid(row=9, column=0, sticky="ew")

        def cancel():
            self.settings_dialog = None
            dialog.destroy()
            self.shortcut_settings_button.focus_set()

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
                  "source": self.source.get(), "jev_enabled": self.jev.get()}
        values.update({key: variable.get().strip() for key, variable in self.shortcuts.items()})
        self.config_notice.set("正在保存配置…")
        try:
            validate_shortcuts(values)
            save_settings(values, self.settings_path)
        except (SettingsError, ValueError) as exc:
            self.config_notice.set(str(exc))
            return False
        else:
            self._apply_shortcuts()
            self.config_notice.set("配置已保存，下次启动自动读取。密钥已加密。")
            return True

    def _apply_shortcuts(self):
        values = {key: variable.get().strip() for key, variable in self.shortcuts.items()}
        sequences = validate_shortcuts(values)
        for widget, sequence in self._shortcut_bindings:
            widget.unbind(sequence)
        self._shortcut_bindings.clear()
        actions = {"shortcut_submit": self._submit_shortcut, "shortcut_start": self._start_shortcut, "shortcut_stop": self._stop_shortcut}
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
        self.shortcut_hint.set(f"{values['shortcut_start']} 开始 · {values['shortcut_stop']} 停止")

    def _source_changed(self):
        local = self.source.get() == "本机麦克风"
        self.meeting_row.pack_forget()
        self.microphone_row.pack_forget()
        (self.microphone_row if local else self.meeting_row).pack(fill="x")
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

    def _config(self, *, text_test=False):
        local = not text_test and self.source.get() == "本机麦克风"
        if not text_test and not local and not self.consent.get():
            raise ValueError("请先勾选参与者授权确认。")
        mode = ANSWER_MODES[self.mode.get()]
        if text_test and mode == "off":
            raise ValueError("提交问题前，请把回答方式改为“离线演练”或“DeepSeek 在线回答”。")
        pid = 0 if text_test or local else self.processes.get(self.process.get(), 0)
        if not text_test and not local and pid <= 0:
            raise ValueError("请启动腾讯会议，刷新并选择一个会议进程。")
        return SessionConfig(pid=pid, capture_exe=self.capture_exe, model_path=self.model_path,
                             cuda_dir=self.cuda_dir, device=DEVICE_MODES[self.device.get()],
                             answer_mode=mode, question_gate="jev" if self.jev.get() else "rules",
                             api_base_url=self.api_url.get().strip(), api_model=self.api_model.get().strip(),
                             api_key=self.api_key.get().strip(), jev_key=self.jev_key.get().strip(),
                             microphone=local)

    def _start(self, question=None):
        if self.session and self.session.is_running:
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
        for widget, idle_state in self.form_controls:
            widget.configure(state="disabled" if busy else idle_state)
        for widget in (self.start_button, self.refresh_button):
            widget.configure(state="disabled" if busy else "normal")
        self.test_button.configure(state="disabled" if busy and ANSWER_MODES[self.mode.get()] == "off" else "normal")
        self.process_combo.configure(state="disabled" if busy else "readonly")
        self.stop_button.configure(state="normal" if busy else "disabled")

    def stop(self):
        if self.session:
            self.session.stop()
            self.status.set("正在停止并释放资源…")
            self.stop_button.configure(state="disabled")

    def _handle(self, kind, payload):
        if kind.startswith("update_"):
            if self.destroyed:
                return
            if kind == "update_checked":
                self.checking_updates = False
                release, automatic = payload
                self.available_release = release
                self.update_notice.set(f"发现新版 {release.version} · 当前 {VERSION}" if release else f"当前 {VERSION} 已是最新版本。")
                self.update_notes.set((release.notes[:1200] + "\n\n更新会保留模型、CUDA 和本机配置。") if release else "无需更新。")
                if release and automatic and self.settings_dialog is None and self.update_dialog is None and not self.closing:
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
            self.processes = {f"{item['Name']} · PID {item['ProcessId']}": int(item["ProcessId"]) for item in payload}
            self.process_combo.configure(values=list(self.processes))
            chosen = preferred_pid(payload)
            self.process.set(next((label for label, pid in self.processes.items() if pid == chosen), ""))
            if not running and self.source.get() == "腾讯会议":
                self.status.set("就绪" if chosen else "请启动腾讯会议后刷新")
        elif kind == "status":
            self.status.set(payload)
        elif kind == "error":
            self.error.set(payload)
        elif kind == "transcript":
            if payload.kind == "partial":
                self.partial.set("实时：" + payload.text)
            else:
                prefix = "确定（低可信，待复核）：" if getattr(payload, "suspect", False) else "确定："
                self.partial.set(prefix + payload.text)
                self._write(self.caption_text, f"{time.strftime('%H:%M:%S')}  {payload.text}\n")
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
        if self.session and self.session.is_running:
            if not messagebox.askyesno("退出 EchoMind", "当前正在运行。停止采集并退出？", parent=self.root):
                return
            self.closing = True
            self.stop()
        else:
            self._destroy()

    def _destroy(self):
        self.destroyed = True
        self.root.after_cancel(self._poll)
        self.root.after_cancel(self._refresh)
        if self._update_timer is not None:
            self.root.after_cancel(self._update_timer)
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
