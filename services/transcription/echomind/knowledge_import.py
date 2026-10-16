"""Local-only document preview and tokenizer-aware, source-preserving chunks."""
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import re

from tokenizers import Tokenizer


class KnowledgeError(RuntimeError):
    pass


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise KnowledgeError("操作已取消，原有资料保留。")


@dataclass(frozen=True)
class Section:
    title: str
    location: str
    text: str


@dataclass(frozen=True)
class DocumentPreview:
    title: str
    filename: str
    origin: str
    digest: str
    sections: tuple[Section, ...]


def preview_document(path, cancel=None):
    path = Path(path)
    if path.suffix.lower() not in (".md", ".txt", ".docx", ".pdf"):
        raise KnowledgeError("支持 Markdown、TXT、DOCX 和文字型 PDF；旧版 DOC 请先转换。")
    try:
        if path.stat().st_size > 50 * 1024**2:
            raise KnowledgeError("单个文档请控制在 50 MB 以内。")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        sections = []
        check_cancel(cancel)
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise KnowledgeError("请先解密 PDF 再导入。")
            if len(reader.pages) > 500:
                raise KnowledgeError("PDF 请拆分为不超过 500 页的文档。")
            for number, page in enumerate(reader.pages, 1):
                check_cancel(cancel)
                text = page.extract_text() or ""
                if text.strip():
                    sections.append(Section(path.stem, f"第 {number} 页", text.strip()))
        elif path.suffix.lower() == ".docx":
            from docx import Document
            title = path.stem
            for number, block in enumerate(Document(io.BytesIO(data)).iter_inner_content(), 1):
                check_cancel(cancel)
                text = block.text if hasattr(block, "text") else "\n".join(" | ".join(cell.text for cell in row.cells) for row in block.rows)
                if hasattr(block, "style") and block.style.name.startswith("Heading"):
                    title = text or title
                if text.strip():
                    sections.append(Section(title, f"段落/表格 {number}", text.strip()))
        else:
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = data.decode("gb18030")
            title = path.stem
            paragraph, first_line = [], 1
            for number, line in enumerate(text.splitlines() + [""], 1):
                check_cancel(cancel)
                heading = re.match(r"^#{1,6}\s+(.+)", line)
                if paragraph and (not line.strip() or heading):
                    sections.append(Section(title, f"第 {first_line} 行", "\n".join(paragraph).strip()))
                    paragraph = []
                if heading:
                    title = heading.group(1).strip()
                if line.strip():
                    if not paragraph:
                        first_line = number
                    paragraph.append(line)
        if not sections:
            raise KnowledgeError("没有提取到文字；扫描件请先 OCR 或转换为文字文档。")
        if sum(len(s.text) for s in sections) > 2_000_000:
            raise KnowledgeError("文档文字过多，请拆分后导入。")
        return DocumentPreview(path.stem, path.name, str(path.resolve()), digest, tuple(sections))
    except KnowledgeError:
        raise
    except Exception:
        raise KnowledgeError("无法读取文档，请检查格式、文件权限或是否损坏。") from None


def chunk_sections(sections, cancel=None):
    tokenizer = Tokenizer.from_file(str(Path(__file__).with_name("knowledge_tokenizer.json")))
    tokenizer.no_truncation()
    chunks = []
    for section in sections:
        check_cancel(cancel)
        heading_encoded = tokenizer.encode(section.title, add_special_tokens=False)
        title = section.title[:heading_encoded.offsets[59][1]] if len(heading_encoded.ids) > 60 else section.title
        budget = 384 - min(60, len(heading_encoded.ids)) - 4
        paragraphs = re.split(r"\n\s*\n|\n(?=\s*(?:(?:问题|题目|问)\s*[:：]|Q\s*\d*\s*[:：]))", section.text)
        for paragraph in paragraphs:
            check_cancel(cancel)
            encoded = tokenizer.encode(paragraph, add_special_tokens=False)
            offsets = encoded.offsets
            for start in range(0, len(offsets), budget - 64):
                end = min(start + budget, len(offsets))
                text = paragraph[offsets[start][0]:offsets[end - 1][1]].strip()
                if text:
                    chunks.append(Section(title, section.location, text))
                if end == len(offsets):
                    break
    return chunks
