# utils/chunking.py

from dataclasses import dataclass
import tiktoken

from .parsers import Sentence

ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    chunk_number: int
    page_number: str      # "n" or "start-end" or "3,5,6"
    embedded_text: str    # filename + content
    raw_text: str         # content only


def _tok_len(s: str) -> int:
    return len(ENC.encode(s))


def _page_range(pages: list[int]) -> str:
    pages = sorted(set(pages))
    if not pages:
        return ""
    if len(pages) == 1:
        return str(pages[0])

    consecutive = all(pages[i] + 1 == pages[i + 1] for i in range(len(pages) - 1))
    if consecutive:
        return f"{pages[0]}-{pages[-1]}"
    return ",".join(str(p) for p in pages)


def _split_text_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Split arbitrary text into token-bounded pieces, optionally with overlap.
    Used when a single sentence is longer than the target chunk size.
    """
    toks = ENC.encode(text)
    if len(toks) <= max_tokens:
        return [text]

    pieces: list[str] = []
    step = max_tokens - max(0, overlap_tokens)
    if step <= 0:
        # overlap >= max_tokens would otherwise never progress
        step = max_tokens

    start = 0
    while start < len(toks):
        end = min(start + max_tokens, len(toks))
        piece = ENC.decode(toks[start:end]).strip()
        if piece:
            pieces.append(piece)
        if end == len(toks):
            break
        start += step

    return pieces


def chunk_sentences(
    sentences: list[Sentence],
    *,
    file_name: str,
    target_tokens: int = 2000,
    overlap_tokens: int = 200
) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur: list[Sentence] = []
    cur_toks = 0
    chunk_no = 1

    def finalize(current: list[Sentence], n: int) -> Chunk:
        raw = " ".join(s.text for s in current).strip()

        pages = [s.page for s in current if s.page is not None]
        page_str = _page_range([p for p in pages if p is not None])

        # txt/docx requirement: use chunk_number as page_number
        if not page_str:
            page_str = str(n)

        embedded = f"{file_name}\n\n{raw}"
        return Chunk(
            chunk_number=n,
            page_number=page_str,
            embedded_text=embedded,
            raw_text=raw,
        )

    i = 0
    while i < len(sentences):
        s = sentences[i]
        t = (s.text or "").strip()

        if not t:
            i += 1
            continue

        s_tokens = _tok_len(t)

        # If one sentence is bigger than the target, split it mid-sentence by tokens.
        if s_tokens > target_tokens:
            # flush current chunk first
            if cur:
                chunks.append(finalize(cur, chunk_no))
                chunk_no += 1
                cur = []
                cur_toks = 0

            parts = _split_text_by_tokens(t, target_tokens, overlap_tokens)
            for part in parts:
                chunks.append(finalize([Sentence(text=part, page=s.page)], chunk_no))
                chunk_no += 1

            i += 1
            continue

        # Normal "fits?" logic
        if cur and (cur_toks + s_tokens > target_tokens):
            # finalize current chunk
            prev_cur = cur[:]  # keep for overlap comparison
            prev_len = len(prev_cur)
            prev_toks = cur_toks

            chunks.append(finalize(prev_cur, chunk_no))

            # build overlap by last sentences totaling ~overlap_tokens
            overlap: list[Sentence] = []
            overlap_toks = 0
            for ss in reversed(prev_cur):
                tt = _tok_len(ss.text)
                # keep at least one sentence if overlap is empty
                if overlap and overlap_toks + tt > overlap_tokens:
                    break
                overlap.append(ss)
                overlap_toks += tt
            overlap.reverse()

            # IMPORTANT: prevent infinite loop when overlap == entire chunk
            # (e.g., small chunk + next sentence doesn't fit -> would repeat forever)
            if len(overlap) == prev_len and overlap_toks == prev_toks:
                overlap = []
                overlap_toks = 0

            chunk_no += 1
            cur = overlap
            cur_toks = overlap_toks
            continue  # re-try adding same sentence s

        # add sentence
        cur.append(s)
        cur_toks += s_tokens
        i += 1

    if cur:
        chunks.append(finalize(cur, chunk_no))

    return chunks
