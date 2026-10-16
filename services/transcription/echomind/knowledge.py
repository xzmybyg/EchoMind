"""Local SQLite/NumPy hybrid retrieval. No cloud requests or implicit downloads."""
from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid

import numpy as np

from .knowledge_import import (DocumentPreview, KnowledgeError, Section, check_cancel, chunk_sections)
from .knowledge_model import LocalEmbedder, MODEL_ID, model_ready


def knowledge_path():
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "EchoMind/knowledge"


@dataclass(frozen=True)
class KnowledgeHit:
    chunk_id: int
    document_id: str
    title: str
    source: str
    location: str
    text: str
    score: float


def _key(text):
    return re.sub(r"[^a-z0-9\u3400-\u9fff]", "", text.casefold())


def _query_text(text):
    return re.sub(r"^(?:请|帮我|给我|讲解|解释|介绍|说明|一下|什么是)+", "", text.strip()).strip("？?。 ")


class KnowledgeService:
    def __init__(self, directory=None, *, embedder=None, threshold=0.44):
        self.directory = Path(directory) if directory else knowledge_path()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / "knowledge.db"
        self._embedder = embedder
        self.threshold = threshold
        self.notice = "语义模型未准备，仅关键词检索"
        self._lock = threading.RLock()
        self._preparing = False
        self._revision = None
        self._rows = []
        self._matrix = np.empty((0, 512), dtype=np.float32)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, filename TEXT NOT NULL,
                    origin TEXT NOT NULL, digest TEXT NOT NULL, kind TEXT NOT NULL,
                    question TEXT NOT NULL DEFAULT '', answer TEXT NOT NULL DEFAULT '',
                    aliases TEXT NOT NULL DEFAULT '', tags TEXT NOT NULL DEFAULT '', sections TEXT NOT NULL, updated REAL NOT NULL);
                CREATE UNIQUE INDEX IF NOT EXISTS document_origin ON documents(origin);
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    title TEXT NOT NULL, location TEXT NOT NULL, text TEXT NOT NULL,
                    embedding BLOB, model TEXT NOT NULL DEFAULT '');
                CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(title, aliases, tags, text, tokenize=trigram);
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
                INSERT OR IGNORE INTO meta VALUES ('revision', 0);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=0.2)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def list_documents(self, limit=200, offset=0):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT id,title,filename,kind,tags,updated FROM documents ORDER BY updated DESC LIMIT ? OFFSET ?", (limit, offset))]

    def document(self, document_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
            return dict(row) if row else None

    def contents(self, document_id):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT title,location,text FROM chunks WHERE document_id=? ORDER BY id", (document_id,))]

    def prepare(self, cancel=None, progress=lambda text, percent: None):
        with self._lock:
            if self._preparing:
                return
            self._preparing = True
        try:
            check_cancel(cancel)
            if self._embedder is None:
                if not model_ready():
                    return
                self.notice = "语义索引准备中，暂用关键词检索"
                with self._lock:
                    if self._embedder is None:
                        self._embedder = LocalEmbedder()
            with self.connect() as db:
                pending = db.execute("SELECT id,title,text FROM chunks WHERE embedding IS NULL OR model!=?", (MODEL_ID,)).fetchall()
            for start in range(0, len(pending), 16):
                check_cancel(cancel)
                batch = pending[start:start + 16]
                vectors = self._embedder.encode([r['title'] + "\n" + r['text'] for r in batch])
                with self.connect() as db:
                    check_cancel(cancel)
                    for row, vector in zip(batch, vectors):
                        db.execute("UPDATE chunks SET embedding=?,model=? WHERE id=?", (self._vector(vector), MODEL_ID, row['id']))
                    db.execute("UPDATE meta SET value=value+1 WHERE key='revision'")
                progress("正在建立语义索引", min(100, (start + len(batch)) * 100 / max(1, len(pending))))
            self.notice = "本地混合检索已就绪"
        finally:
            self._preparing = False

    def prepare_async(self):
        def worker():
            try:
                self.prepare()
            except Exception:
                self.notice = "语义索引未就绪，仅关键词检索；可在知识库重建索引。"
        threading.Thread(target=worker, name="echomind-knowledge-index", daemon=True).start()

    @staticmethod
    def _vector(vector):
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (512,) or not np.isfinite(vector).all():
            raise KnowledgeError("检索模型返回了无效向量。")
        return (vector / max(float(np.linalg.norm(vector)), 1e-12)).tobytes()

    def save_question(self, question, answer, aliases="", tags="", *, document_id=None, cancel=None):
        question, answer = question.strip(), answer.strip()
        if not question or not answer:
            raise KnowledgeError("请填写问题和参考答案。")
        if len(question) > 500 or len(answer) > 100_000 or len(aliases) > 2000 or len(tags) > 500:
            raise KnowledgeError("题目内容过长，请拆分后保存。")
        digest = hashlib.sha256((question + answer + aliases + tags).encode()).hexdigest()
        identifier = document_id or uuid.uuid4().hex
        preview = DocumentPreview(question, "手动题目", "manual:" + identifier, digest,
                                  (Section(question[:120], "参考答案", answer),))
        return self._save(preview, document_id=identifier, kind="manual", question=question,
                          answer=answer, aliases=aliases, tags=tags, cancel=cancel)

    def import_document(self, preview, *, cancel=None):
        return self._save(preview, cancel=cancel)

    def _save(self, preview, *, document_id=None, kind="document", question="", answer="", aliases="", tags="", cancel=None):
        check_cancel(cancel)
        pieces = chunk_sections(preview.sections, cancel)
        if not pieces:
            raise KnowledgeError("资料没有可检索的文字。")
        vectors = []
        if self._embedder is None and model_ready():
            with self._lock:
                if self._embedder is None:
                    self._embedder = LocalEmbedder()
        if self._embedder is not None:
            for start in range(0, len(pieces), 16):
                check_cancel(cancel)
                vectors.extend(self._embedder.encode([p.title + "\n" + p.text for p in pieces[start:start + 16]]))
        try:
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                duplicate = db.execute("SELECT id FROM documents WHERE digest=? AND kind=?", (preview.digest, kind)).fetchone()
                if duplicate and duplicate['id'] != document_id:
                    return duplicate['id']
                old = db.execute("SELECT id FROM documents WHERE origin=?", (preview.origin,)).fetchone()
                identifier = old['id'] if old else document_id or uuid.uuid4().hex
                self._delete(db, identifier)
                check_cancel(cancel)
                db.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (identifier, preview.title, preview.filename,
                    preview.origin, preview.digest, kind, question, answer, aliases.strip(), tags.strip(),
                    json.dumps([{'title':s.title,'location':s.location,'text':s.text} for s in preview.sections], ensure_ascii=False), time.time()))
                for index, piece in enumerate(pieces):
                    check_cancel(cancel)
                    rowid = db.execute("INSERT INTO chunks(document_id,title,location,text,embedding,model) VALUES(?,?,?,?,?,?)",
                        (identifier, piece.title, piece.location, piece.text,
                         self._vector(vectors[index]) if vectors else None, MODEL_ID if vectors else "")).lastrowid
                    db.execute("INSERT INTO search(rowid,title,aliases,tags,text) VALUES(?,?,?,?,?)",
                               (rowid, piece.title, aliases, tags, piece.text))
                db.execute("UPDATE meta SET value=value+1 WHERE key='revision'")
                check_cancel(cancel)
            return identifier
        except sqlite3.Error:
            raise KnowledgeError("知识库正在被其他操作使用或写入失败，请稍后重试。") from None

    @staticmethod
    def _delete(db, document_id):
        db.execute("DELETE FROM search WHERE rowid IN (SELECT id FROM chunks WHERE document_id=?)", (document_id,))
        db.execute("DELETE FROM documents WHERE id=?", (document_id,))

    def delete(self, document_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._delete(db, document_id)
            db.execute("UPDATE meta SET value=value+1 WHERE key='revision'")

    def retrieve(self, query, limit=3, *, cancel=None):
        check_cancel(cancel)
        if limit <= 0:
            return []
        query = _query_text(query)[:1000]
        key = _key(query)
        if not key:
            return []
        with self.connect() as db:
            db.execute('BEGIN')
            revision = db.execute("SELECT value FROM meta WHERE key='revision'").fetchone()[0]
            with self._lock:
                if revision != self._revision:
                    self._rows = [dict(r) for r in db.execute("SELECT c.*,d.filename,d.aliases,d.tags FROM chunks c JOIN documents d ON c.document_id=d.id ORDER BY c.id")]
                    self._revision = revision
                    valid = [r for r in self._rows if r['model'] == MODEL_ID and r['embedding'] and len(r['embedding']) == 2048]
                    self._vector_rows = valid
                    self._matrix = np.asarray([np.frombuffer(r['embedding'], dtype=np.float32) for r in valid], dtype=np.float32).reshape(-1, 512)
                rows, vector_rows, matrix = self._rows, self._vector_rows, self._matrix
            terms = re.findall(r"[a-zA-Z][a-zA-Z0-9_+.#/-]*|[\u3400-\u9fff]{2,}", query)
            tokens = []
            for term in terms:
                tokens.extend(term[i:i + 3] for i in range(len(term) - 2))
            expression = " OR ".join('"' + t.replace('"', '""') + '"' for t in dict.fromkeys(tokens[:64]))
            lexical = [r[0] for r in db.execute("SELECT rowid FROM search WHERE search MATCH ? ORDER BY bm25(search,5,4,3,1) LIMIT 20", (expression,))] if expression else []
        semantic = []
        similarities = {}
        if self._embedder is not None and len(matrix):
            check_cancel(cancel)
            vector = np.frombuffer(self._vector(self._embedder.encode([query])[0]), dtype=np.float32)
            scores = matrix @ vector
            for index in np.argsort(-scores)[:20]:
                row = vector_rows[int(index)]
                similarities[row['id']] = float(scores[index])
                # A real lexical hit corroborates a weaker semantic match.
                if scores[index] >= self.threshold - (0.05 if row['id'] in lexical else 0):
                    semantic.append(row['id'])
        exact = []
        for row in rows:
            names = [row['title'], *re.split(r"[,，\n]", row['aliases']), *re.split(r"[,，\n]", row['tags'])]
            if (any(len(name_key := _key(name)) >= 2 and (name_key in key or key in name_key) for name in names)
                    or (row['id'] in lexical and len(key) >= 3 and key in _key(row['text']))):
                exact.append(row['id'])
        rankings = {}
        for ranking in (lexical, semantic):
            for rank, identifier in enumerate(ranking, 1):
                rankings[identifier] = rankings.get(identifier, 0) + 1 / (60 + rank)
        for identifier in exact:
            rankings[identifier] = rankings.get(identifier, 0) + 1
        by_id = {r['id']: r for r in rows}
        selected, seen = [], set()
        for identifier in sorted(rankings, key=lambda k: (-rankings[k], k)):
            check_cancel(cancel)
            if identifier not in exact and identifier not in semantic:
                continue  # Weak keyword overlaps alone must not inject unrelated material.
            row = by_id[identifier]
            duplicate = _key(row['text'])
            if duplicate in seen:
                continue
            seen.add(duplicate)
            selected.append(KnowledgeHit(identifier, row['document_id'], row['title'], row['filename'],
                                         row['location'], row['text'], similarities.get(identifier, rankings[identifier])))
            if len(selected) >= min(limit, 3):
                break
        return selected
