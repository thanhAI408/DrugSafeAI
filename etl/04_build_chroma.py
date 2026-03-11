# etl/04_build_chroma.py
# .venv\Scripts\activate
# python etl/04_build_chroma.py

import os
import json
import shutil
import warnings
from pathlib import Path
from typing import Dict, Any, List, Iterable, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings


warnings.filterwarnings("ignore")
os.environ["ANONYMIZED_TELEMETRY"] = "False"

ROOT = Path(__file__).resolve().parents[1]

INPUT_JSONL_STOCKLEY = ROOT / "data" / "processed" / "pdf_pages_stockley.jsonl"
INPUT_JSONL_DUOC_THU = ROOT / "data" / "processed" / "pdf_pages_duoc_thu.jsonl"

PERSIST_STOCKLEY = ROOT / "chroma_db" / "stockley"
PERSIST_DUOC_THU = ROOT / "chroma_db" / "duoc_thu"

# Chunk: Stockley nên vừa phải; Dược thư có thể dài hơn (dùng classification)
CHUNK_SIZE_STOCKLEY = 1200
CHUNK_OVERLAP_STOCKLEY = 180

CHUNK_SIZE_DUOC_THU = 2200
CHUNK_OVERLAP_DUOC_THU = 250

RESET_DB = True


def norm_space(s: str) -> str:
    return " ".join((s or "").split())


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[WARN] JSONL lỗi {path.name}:{i} -> {e}")


def detect_page_no(obj: Dict[str, Any]) -> Optional[int]:
    md = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    for x in [obj.get("page"), obj.get("page_no"), md.get("page")]:
        try:
            if x is None:
                continue
            return int(x)
        except Exception:
            pass
    return None


def detect_text(obj: Dict[str, Any]) -> str:
    for k in ["text", "content", "page_text", "chunk_text"]:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v
    md = obj.get("metadata")
    if isinstance(md, dict):
        v = md.get("text")
        if isinstance(v, str) and v.strip():
            return v
    return ""


def split_text_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = norm_space(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        ch = text[start:end].strip()
        if ch:
            chunks.append(ch)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def jsonl_to_documents(path: Path, default_source: str, chunk_size: int, overlap: int) -> List[Document]:
    docs: List[Document] = []
    for obj in read_jsonl(path):
        page_no = detect_page_no(obj)
        if page_no is None:
            continue

        text = norm_space(detect_text(obj))
        if not text:
            continue

        md_in = obj.get("metadata", {})
        if not isinstance(md_in, dict):
            md_in = {}

        source = md_in.get("source") or default_source
        column = md_in.get("column")
        generic = md_in.get("generic")  # duoc_thu có thể có ""

        chunks = split_text_with_overlap(text, chunk_size=chunk_size, overlap=overlap)
        for i, ch in enumerate(chunks):
            meta = {
                "source": source,
                "page": int(page_no),
                "chunk_index": i,
                "chunk_count": len(chunks),
                "origin": "pdf_jsonl",
            }
            if column:
                meta["column"] = column
            if generic is not None:
                meta["generic"] = generic

            for k in ["url", "title"]:
                if k in md_in and k not in meta:
                    meta[k] = md_in[k]

            docs.append(Document(page_content=ch, metadata=meta))
    return docs


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        encode_kwargs={"normalize_embeddings": True},
    )


def reset_dir(p: Path):
    if RESET_DB and p.exists():
        print(f"[INFO] Xoá DB cũ: {p}")
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)


def build_db(docs: List[Document], persist_dir: Path, collection_name: str):
    reset_dir(persist_dir)
    emb = get_embeddings()
    print(f"[INFO] Building Chroma: {persist_dir} | collection={collection_name} | docs={len(docs)}")

    db = Chroma.from_documents(
        documents=docs,
        embedding=emb,
        persist_directory=str(persist_dir),
        collection_name=collection_name,
    )
    try:
        db.persist()
    except Exception:
        pass
    print(f"[OK] Done: {persist_dir}")


def main():
    if not INPUT_JSONL_STOCKLEY.exists():
        raise RuntimeError(f"Thiếu file: {INPUT_JSONL_STOCKLEY}")
    if not INPUT_JSONL_DUOC_THU.exists():
        raise RuntimeError(f"Thiếu file: {INPUT_JSONL_DUOC_THU}")

    print("=== BUILD 2 CHROMA DBs ===")
    print(f"- Stockley JSONL : {INPUT_JSONL_STOCKLEY}")
    print(f"- Dược thư JSONL : {INPUT_JSONL_DUOC_THU}")
    print(f"- OUT Stockley DB: {PERSIST_STOCKLEY}")
    print(f"- OUT Dược thư DB: {PERSIST_DUOC_THU}")
    print()

    docs_stockley = jsonl_to_documents(
        INPUT_JSONL_STOCKLEY,
        default_source="stockley_9e",
        chunk_size=CHUNK_SIZE_STOCKLEY,
        overlap=CHUNK_OVERLAP_STOCKLEY,
    )
    docs_duoc_thu = jsonl_to_documents(
        INPUT_JSONL_DUOC_THU,
        default_source="duoc_thu_qg_2018",
        chunk_size=CHUNK_SIZE_DUOC_THU,
        overlap=CHUNK_OVERLAP_DUOC_THU,
    )

    print(f"[STAT] stockley docs: {len(docs_stockley)}")
    print(f"[STAT] duoc_thu docs: {len(docs_duoc_thu)}")
    print()

    if not docs_stockley:
        raise RuntimeError("Không có docs Stockley để build.")
    if not docs_duoc_thu:
        raise RuntimeError("Không có docs Dược thư để build.")

    build_db(docs_stockley, PERSIST_STOCKLEY, collection_name="stockley")
    build_db(docs_duoc_thu, PERSIST_DUOC_THU, collection_name="duoc_thu")


if __name__ == "__main__":
    main()