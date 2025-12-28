from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import re

from docx import Document
from pypdf import PdfReader

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

@dataclass
class Sentence:
    text: str
    page: Optional[int]  # 1-based for PDF pages, None for txt/docx

def _clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_sentences(text: str, page: Optional[int]) -> list[Sentence]:
    text = _clean_text(text)
    if not text:
        return []
    parts = _SENTENCE_SPLIT.split(text)
    out: list[Sentence] = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(Sentence(text=p, page=page))
    return out

def parse_txt(path: Path) -> list[Sentence]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return split_sentences(raw, None)

def parse_docx(path: Path) -> list[Sentence]:
    doc = Document(str(path))
    chunks = []
    for para in doc.paragraphs:
        t = para.text.strip()
        if t:
            chunks.append(t)
    raw = "\n".join(chunks)
    return split_sentences(raw, None)

def parse_pdf(path: Path) -> tuple[list[Sentence], int]:
    reader = PdfReader(str(path))
    all_sentences: list[Sentence] = []
    for i, page in enumerate(reader.pages):
        page_num = i + 1
        txt = page.extract_text() or ""
        all_sentences.extend(split_sentences(txt, page_num))
        print(f"Parsed PDF page {page_num}/{len(reader.pages)}")
    return all_sentences, len(reader.pages)
