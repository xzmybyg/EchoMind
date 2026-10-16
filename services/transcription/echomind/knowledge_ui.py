"""Offline knowledge management. Workers communicate only through a local queue."""
import hashlib
import json
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .knowledge import KnowledgeError, KnowledgeService
from .knowledge_import import DocumentPreview, Section, preview_document


class KnowledgeWindow:
    def __init__(self, host, directory, colors):
        self.host, self.colors = host, colors
        self.root = tk.Toplevel(host.root)
        self.root.withdraw()
        self.root.title("知识库 · EchoMind")
        self.root.transient(host.root)
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.busy = False
        self.closed = False
        self.service = None
        self.rows = {}
        self.offset = 0
        self.notice = tk.StringVar(self.root, value="正在读取本地知识库…")
        self.search = tk.StringVar(self.root)
        frame = ttk.Frame(self.root, padding=20, style="Card.TFrame")
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(4, weight=1)
        heading = ttk.Frame(frame, style="Card.TFrame")
        heading.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(heading, text="面试题知识库", style="Card.TLabel", font=("Microsoft YaHei UI", 14, "bold")).pack(side="left")
        ttk.Checkbutton(heading, text="启用知识库增强回答", variable=host.knowledge_enabled,
                        command=self._enable).pack(side="right")
        ttk.Label(frame, text="资料和检索留在本机；启用后，仅将命中的片段发送给在线回答接口。重新开始采集后生效。",
                  style="Card.TLabel", foreground=colors['muted'], wraplength=800).grid(row=1, column=0, sticky="ew", pady=(0, 12))
        actions = ttk.Frame(frame, style="Card.TFrame")
        actions.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        self.buttons = []
        for title, action in (("导入文档", self.import_file), ("新增题目", self.new_question),
                              ("编辑", self.edit), ("删除", self.delete), ("重建语义索引", self.reindex),
                              ("检索模型", self.components)):
            button = ttk.Button(actions, text=title, command=action)
            button.pack(side="left", padx=(0, 8))
            self.buttons.append(button)
        row = ttk.Frame(frame, style="Card.TFrame")
        row.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        entry = ttk.Entry(row, textvariable=self.search)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        entry.bind("<Return>", lambda _: self.find() or "break")
        ttk.Button(row, text="搜索", command=self.find).pack(side="left", padx=(0, 8))
        ttk.Button(row, text="全部资料", command=self.refresh).pack(side="left")
        panes = ttk.Panedwindow(frame, orient="horizontal")
        panes.grid(row=4, column=0, sticky="nsew")
        listing = ttk.Frame(panes, style="Card.TFrame")
        self.tree = ttk.Treeview(listing, columns=("kind",), selectmode="browse", height=12)
        self.tree.heading("#0", text="题目 / 文档")
        self.tree.heading("kind", text="类型")
        self.tree.column("#0", width=260, minwidth=140)
        self.tree.column("kind", width=75, minwidth=65, stretch=False)
        scroll = ttk.Scrollbar(listing, command=self.tree.yview)
        scroll.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.bind("<<TreeviewSelect>>", self.selected)
        detail = ttk.Frame(panes, style="Card.TFrame")
        self.detail = tk.Text(detail, width=35, height=10, wrap="word", state="disabled",
                              font=("Microsoft YaHei UI", 10), background=colors['surface'], foreground=colors['text'], relief="flat")
        detail_scroll = ttk.Scrollbar(detail, command=self.detail.yview)
        detail_scroll.pack(side="right", fill="y")
        self.detail.pack(fill="both", expand=True)
        self.detail.configure(yscrollcommand=detail_scroll.set)
        panes.add(listing, weight=2)
        panes.add(detail, weight=3)
        ttk.Label(frame, textvariable=self.notice, style="Card.TLabel", foreground=colors['muted'], wraplength=800).grid(row=5, column=0, sticky="ew", pady=12)
        self.progress = ttk.Progressbar(frame, mode='indeterminate')
        self.progress.grid(row=6, column=0, sticky='ew', pady=(0, 12))
        self.progress.grid_remove()
        footer = ttk.Frame(frame, style="Card.TFrame")
        footer.grid(row=7, column=0, sticky="ew")
        ttk.Button(footer, text="取消任务", command=self.cancel.set).pack(side="left")
        ttk.Button(footer, text="上一页", command=lambda: self.page(-1)).pack(side="left", padx=(16, 8))
        ttk.Button(footer, text="下一页", command=lambda: self.page(1)).pack(side="left")
        ttk.Button(footer, text="关闭", command=self.close).pack(side="right")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda _: self.close() or "break")
        self.root.bind("<Destroy>", self._destroyed, add="+")
        self.root.update_idletasks()
        self.root.minsize(max(860, actions.winfo_reqwidth() + 40), max(560, frame.winfo_reqheight()))
        host._center_dialog(self.root, (980, 640))
        self.root.wait_visibility()
        self.root.grab_set()
        self._timer = self.root.after(60, self._poll)
        self._work(lambda: KnowledgeService(directory), self._loaded)

    def _enable(self):
        if not self.host.save_config():
            self.host.knowledge_enabled.set(not self.host.knowledge_enabled.get())
            self.notice.set(self.host.config_notice.get())

    def _loaded(self, service):
        self.service = service
        self.refresh()

    def _work(self, action, done):
        if self.busy or self.closed:
            return
        self.busy = True
        self.progress.grid()
        self.progress.start(20)
        self.cancel = threading.Event()
        for button in self.buttons:
            button.state(["disabled"])
        self.notice.set("正在本地处理… 可取消，不影响字幕。")
        def worker():
            try:
                result = action()
                self.events.put((done, result, ""))
            except Exception as exc:
                self.events.put((None, None, str(exc) if isinstance(exc, KnowledgeError) else "本地操作失败，请检查文件、磁盘空间或稍后重试。"))
        threading.Thread(target=worker, name="echomind-knowledge-ui", daemon=True).start()

    def _poll(self):
        try:
            while True:
                done, value, error = self.events.get_nowait()
                self.busy = False
                self.progress.stop()
                self.progress.grid_remove()
                for button in self.buttons:
                    button.state(["!disabled"])
                if error:
                    self.notice.set(error)
                elif done:
                    done(value)
        except queue.Empty:
            pass
        if not self.closed:
            self._timer = self.root.after(60, self._poll)

    def _destroyed(self, event):
        if event.widget == self.root:
            self.closed = True
            self.cancel.set()
            self.root.after_cancel(self._timer)

    def close(self):
        if self.busy:
            self.cancel.set()
            self.notice.set("正在取消任务，请稍候再关闭。")
            return
        self.root.destroy()

    def refresh(self):
        if self.service is not None:
            self._work(lambda: self.service.list_documents(offset=self.offset), self._show_documents)

    def page(self, direction):
        if not self.busy:
            self.offset = max(0, self.offset + direction * 200)
            self.refresh()

    def _show_documents(self, rows):
        self.tree.delete(*self.tree.get_children())
        self.rows = {r['id']: r for r in rows}
        for key, row in self.rows.items():
            self.tree.insert("", "end", iid=key, text=row['title'], values=("问答" if row['kind'] == "manual" else "文档",))
        self.notice.set(f"第 {self.offset // 200 + 1} 页 · {len(rows)} 份资料 · {self.service.notice}")

    def find(self):
        if self.service is None or not self.search.get().strip():
            return
        query = self.search.get()
        def search():
            self.service.prepare(self.cancel)
            return self.service.retrieve(query, cancel=self.cancel)
        def show(hits):
            self.tree.delete(*self.tree.get_children())
            self.rows = {}
            for hit in hits:
                key = "hit:" + str(hit.chunk_id)
                self.rows[key] = {'id':hit.document_id, 'title':hit.title, 'hit':hit}
                self.tree.insert("", "end", iid=key, text=hit.title, values=("相关片段",))
            self.notice.set(f"找到 {len(hits)} 个相关片段 · {self.service.notice}")
        self._work(search, show)

    def selected(self, event=None):
        picked = self.tree.selection()
        if self.busy or not picked or picked[0] not in self.rows:
            return
        row = self.rows[picked[0]]
        if 'hit' in row:
            hit = row['hit']
            text = f"{hit.title}\n{hit.source} · {hit.location}\n\n{hit.text}"
            self._show_text(self.detail, text)
        else:
            def show(pieces):
                self._show_text(self.detail, "\n\n".join(p['location'] + "\n" + p['text'] for p in pieces)[:100_000])
                self.notice.set(self.service.notice)
            self._work(lambda: self.service.contents(row['id']), show)

    @staticmethod
    def _show_text(widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")
        widget.see("1.0")

    def components(self):
        self.root.grab_release()
        self.host.open_components(knowledge=True)
        dialog = self.host.component_dialog
        if dialog is not None:
            dialog.bind('<Destroy>', lambda event: self.root.grab_set()
                        if event.widget == dialog and not self.closed else None, add='+')
        elif not self.closed:
            self.root.grab_set()

    def reindex(self):
        if self.service is not None:
            self._work(lambda: self.service.prepare(self.cancel), lambda _: self.refresh())

    def import_file(self):
        path = filedialog.askopenfilename(parent=self.root, title="导入面试题文档", filetypes=[("文字资料", "*.md *.txt *.docx *.pdf")])
        if path:
            self._work(lambda: preview_document(path, self.cancel), self._preview)

    def _preview(self, preview):
        dialog = tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title("核对导入资料 · EchoMind")
        dialog.transient(self.root)
        frame = ttk.Frame(dialog, padding=20, style="Card.TFrame")
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=f"{preview.filename} · {len(preview.sections)} 段；确认后才入库。", style="Card.TLabel").pack(anchor="w", pady=(0, 12))
        actions = ttk.Frame(frame, style="Card.TFrame")
        actions.pack(side="bottom", fill="x")
        text = self.host._text(frame)
        content = "\n\n".join(s.location + "\n" + s.text for s in preview.sections)
        self._show_text(text, content[:100_000] + ("\n\n（预览已截取前 10 万字；确认后导入完整文字。）" if len(content) > 100_000 else ""))
        def close():
            dialog.destroy()
            self.root.grab_set()
        def commit():
            close()
            self._work(lambda: self.service.import_document(preview, cancel=self.cancel), lambda _: self.refresh())
        ttk.Button(actions, text="确认导入", style="Primary.TButton", command=commit).pack(side="right")
        ttk.Button(actions, text="取消", command=close).pack(side="right", padx=8)
        dialog.protocol("WM_DELETE_WINDOW", close)
        dialog.bind("<Escape>", lambda _: close() or "break")
        dialog.minsize(620, 400)
        self.host._center_dialog(dialog, (760, 520))
        dialog.wait_visibility()
        dialog.grab_set()

    def new_question(self):
        if self.service is not None and not self.busy:
            self._editor()

    def edit(self):
        picked = self.tree.selection()
        if picked and not self.busy:
            identifier = self.rows[picked[0]]['id']
            self._work(lambda: self.service.document(identifier), self._editor)

    def _editor(self, row=None):
        dialog = tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title("编辑知识条目 · EchoMind")
        dialog.transient(self.root)
        frame = ttk.Frame(dialog, padding=20, style="Card.TFrame")
        frame.pack(fill="both", expand=True)
        values = [tk.StringVar(dialog, value=(row or {}).get(k, '')) for k in ('question', 'aliases', 'tags')]
        manual = not row or row['kind'] == 'manual'
        if not manual:
            values[0].set(row['title'])
        for variable, label in zip(values, ("问题 / 标题", "同义问法（逗号分隔）", "标签（逗号分隔）")):
            ttk.Label(frame, text=label, style="Card.TLabel").pack(anchor="w", pady=(0, 4))
            ttk.Entry(frame, textvariable=variable).pack(fill="x", pady=(0, 8))
        ttk.Label(frame, text="参考答案 / 资料正文", style="Card.TLabel").pack(anchor="w")
        actions = ttk.Frame(frame, style="Card.TFrame")
        actions.pack(side="bottom", fill="x", pady=(12, 0))
        editor = self.host._text(frame)
        editor.configure(state="normal", undo=True)
        content = row['answer'] if row and manual else "\n\n".join(s['text'] for s in json.loads(row['sections'])) if row else ""
        editor.insert('1.0', content)
        def close():
            dialog.destroy()
            self.root.grab_set()
        def save():
            question, aliases, tags = [v.get().strip() for v in values]
            answer = editor.get('1.0', 'end-1c').strip()
            if not question or not answer:
                messagebox.showinfo("请补充内容", "请填写问题/标题及参考答案。", parent=dialog)
                return
            if manual:
                action = lambda: self.service.save_question(question, answer, aliases, tags, document_id=row['id'] if row else None, cancel=self.cancel)
            else:
                preview = DocumentPreview(question, row['filename'], row['origin'], hashlib.sha256(answer.encode()).hexdigest(),
                                          (Section(question, "手动修订正文", answer),))
                action = lambda: self.service._save(preview, document_id=row['id'], aliases=aliases, tags=tags, cancel=self.cancel)
            close()
            self._work(action, lambda _: self.refresh())
        ttk.Button(actions, text="保存", style="Primary.TButton", command=save).pack(side="right")
        ttk.Button(actions, text="取消", command=close).pack(side="right", padx=8)
        dialog.protocol("WM_DELETE_WINDOW", close)
        dialog.bind("<Escape>", lambda _: close() or "break")
        dialog.minsize(620, 500)
        self.host._center_dialog(dialog, (740, 600))
        dialog.wait_visibility()
        dialog.grab_set()

    def delete(self):
        picked = self.tree.selection()
        if picked and not self.busy and messagebox.askyesno("删除资料", "从知识库删除该资料及索引？不会删除原文件。", parent=self.root):
            identifier = self.rows[picked[0]]['id']
            self._work(lambda: self.service.delete(identifier), lambda _: self.refresh())
