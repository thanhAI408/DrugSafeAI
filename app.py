# .venv\Scripts\activate
# streamlit run app.py

import os
import re
import json
import hashlib
import warnings
import itertools
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from urllib import request, error

import streamlit as st
from unidecode import unidecode
import fitz  # PyMuPDF

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


# ============================================================
# CONFIG
# ============================================================
warnings.filterwarnings("ignore")
os.environ["ANONYMIZED_TELEMETRY"] = "False"

PERSIST_STOCKLEY = "./chroma_db/stockley"
STOCKLEY_COLLECTION = "stockley"

ALIAS_MAP_PATH = "./data/processed/alias_map_duoc_thu.json"
ALIAS_SOURCES_PATH = "./data/processed/alias_map_duoc_thu_sources.jsonl"

RAW_PDFS = {
    "stockley_9e": "./data/raw/stockley.pdf",
}

SOURCE_LABELS = {
    "duoc_thu_qg_2018": "Dược thư QG VN 2018",
    "stockley_9e": "Stockley’s Drug Interactions (9e)",
}

SOURCE_REMOTE_URL = {
    "duoc_thu_qg_2018": "https://trungtamthuoc.com/upload/pdf/duoc-thu-quoc-gia-viet-nam-2018.pdf",
    "stockley_9e": "https://eprints.poltekkesadisutjipto.ac.id/id/eprint/2137/1/Stockley%27s%20Drug%20Interactions%2C%209th%20Edition.pdf",
}

CACHE_DIR = "./cache"
CACHE_PDF_DIR = os.path.join(CACHE_DIR, "pdf_highlight")
CACHE_IMG_DIR = os.path.join(CACHE_DIR, "preview_img")
os.makedirs(CACHE_PDF_DIR, exist_ok=True)
os.makedirs(CACHE_IMG_DIR, exist_ok=True)

MAX_ENTITIES = 10
MAX_STOCKLEY_CANDIDATES = 18
MAX_INTERNAL_PAGES = 70

MAX_RAG_DOCS = 8
MAX_RAG_CONTEXT_CHARS = 5000

STOCKLEY_START_PAGE = 22
STOCKLEY_LAST_CONTENT_PAGE = 1587

COLOR_PAIR_BOX = (0.10, 0.10, 0.10)
COLOR_PAIR_HL = (1.0, 0.92, 0.20)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# model dịch cho pair mode
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "translategemma:12b")

# model RAG cho free-chat mode
OLLAMA_RAG_MODEL = os.environ.get("OLLAMA_RAG_MODEL", "llama3.1:8b")

OLLAMA_TIMEOUT_SEC = int(os.environ.get("OLLAMA_TIMEOUT_SEC", "90"))

MAX_SUMMARY_LOOKAHEAD_PAGES = 2

FEATURE_PAIRWISE = "Kiểm tra tương tác theo cặp"
FEATURE_RAG = "Hỏi đáp RAG về tương tác thuốc"


# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class Entity:
    raw: str
    canonical: str
    drug_name_if_any: Optional[str] = None
    duoc_url: Optional[str] = None


@dataclass
class EvidenceItem:
    page: int
    pair_text: str
    snippet_en: str
    interaction_summary_en: Optional[str] = None
    interaction_summary_vi: Optional[str] = None
    open_url: Optional[str] = None
    cache_pdf_path: Optional[str] = None
    preview_image_paths: Optional[List[str]] = None
    header_rect: Optional[Tuple[float, float, float, float]] = None
    confidence: str = "medium"


@dataclass
class AgentLog:
    agent: str
    status: str
    detail: str


@dataclass
class PairTask:
    a: Entity
    b: Entity
    queries: List[str]


@dataclass
class RetrievalResult:
    pages: List[int]
    logs: List[AgentLog] = field(default_factory=list)


@dataclass
class HeaderCandidate:
    page: int
    text: str
    rect: Tuple[float, float, float, float]
    uid: str
    score: int
    raw_block: Dict[str, Any]


@dataclass
class PairResult:
    a: Entity
    b: Entity
    evidence: Optional[EvidenceItem]
    logs: List[AgentLog] = field(default_factory=list)


@dataclass
class RAGChunk:
    page: int
    text: str
    open_url: Optional[str] = None


@dataclass
class GeneralRAGResult:
    prompt: str
    normalized_prompt: str
    entities: List[Entity]
    answer_vi: Optional[str]
    answer_raw: Optional[str]
    chunks: List[RAGChunk]
    logs: List[AgentLog] = field(default_factory=list)


# ============================================================
# BASIC HELPERS
# ============================================================
def norm_text(s: str) -> str:
    s = unidecode((s or "").strip().lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_inline_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def uniq_keep_order(seq: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for x in seq:
        if isinstance(x, (dict, list, tuple)):
            k = json.dumps(x, ensure_ascii=False, sort_keys=isinstance(x, dict))
        else:
            k = str(x)
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def sha1_of_dict(d: Dict[str, Any]) -> str:
    raw = json.dumps(d, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def build_remote_pdf_url(source: str, page_no: int) -> str:
    base = SOURCE_REMOTE_URL.get(source, "")
    return f"{base}#page={page_no}" if base else ""


def is_valid_stockley_page(p: int) -> bool:
    return STOCKLEY_START_PAGE <= p <= STOCKLEY_LAST_CONTENT_PAGE


def make_block_uid(page_no: int, blk: Dict[str, Any]) -> str:
    return (
        f"{page_no}|{round(blk['x0'],1)}|{round(blk['y0'],1)}|"
        f"{round(blk['x1'],1)}|{round(blk['y1'],1)}|{blk['text'][:120]}"
    )


def clean_chunk_text(s: str, max_len: int = 900) -> str:
    s = clean_inline_text(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len]


def regex_term_pattern(term_norm: str) -> re.Pattern:
    return re.compile(rf"(?<![a-z0-9]){re.escape(term_norm)}(?![a-z0-9])", flags=re.I)


# ============================================================
# OLLAMA
# ============================================================
def _ollama_post_json(endpoint: str, payload: Dict[str, Any], timeout: int = OLLAMA_TIMEOUT_SEC) -> Dict[str, Any]:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
    return json.loads(body)


@st.cache_data(show_spinner=False)
def check_ollama_model_available(model_name: str) -> Tuple[bool, str]:
    try:
        payload = {
            "model": model_name,
            "prompt": "Reply with exactly: OK",
            "stream": False,
            "options": {"temperature": 0},
        }
        obj = _ollama_post_json("/api/generate", payload, timeout=20)
        txt = clean_inline_text(obj.get("response", ""))
        if txt:
            return True, ""
        return False, f"Ollama phản hồi rỗng cho model `{model_name}`."
    except error.URLError as e:
        return False, f"Không kết nối được Ollama tại {OLLAMA_BASE_URL}: {e}"
    except Exception as e:
        return False, f"Ollama chưa sẵn sàng hoặc model `{model_name}` chưa load được: {e}"


def cleanup_ollama_translation_output(text: str) -> str:
    t = (text or "").strip()
    leak_markers = [
        "translate the english medical text below",
        "rules:",
        "text:",
        "do not explain",
        "do not repeat the english text",
    ]
    t_norm = norm_text(t)

    cut_positions = []
    for marker in leak_markers:
        pos = t_norm.find(marker)
        if pos > 0:
            cut_positions.append(pos)

    if cut_positions:
        t = t[:min(cut_positions)].strip()

    t = t.strip().strip('"').strip("'").strip()
    t = re.sub(r"^(ban dich|translation)\s*:\s*", "", t, flags=re.I).strip()
    return clean_inline_text(t)


@st.cache_data(show_spinner=False)
def translate_text_vi_ollama(text: str, model_name: str = OLLAMA_MODEL) -> Tuple[Optional[str], Optional[str]]:
    text = clean_inline_text(text)
    if not text:
        return None, "Không có nội dung để dịch."

    prompt = f"""Translate the English medical text below into Vietnamese.

Rules:
- Output only the Vietnamese translation.
- Do not repeat the English text.
- Do not explain.
- Do not add notes.
- Do not add bullet points.
- Keep drug names accurate.
- Keep the translation concise and natural.

TEXT:
{text}
"""

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 300,
        },
    }

    try:
        obj = _ollama_post_json("/api/generate", payload, timeout=OLLAMA_TIMEOUT_SEC)
        translated = cleanup_ollama_translation_output(obj.get("response", ""))
        if not translated:
            return None, f"Ollama trả về rỗng với model `{model_name}`."
        return translated, None
    except error.URLError as e:
        return None, f"Không kết nối được Ollama tại {OLLAMA_BASE_URL}: {e}"
    except Exception as e:
        return None, f"Lỗi dịch qua Ollama model `{model_name}`: {e}"


def generate_general_rag_answer(user_prompt: str, chunks: List[RAGChunk]) -> Tuple[Optional[str], Optional[str]]:
    if not chunks:
        return None, "Không có context để tổng hợp câu trả lời."

    context_parts = []
    total_chars = 0
    for i, ch in enumerate(chunks, start=1):
        piece = f"[Chunk {i} | page {ch.page}] {ch.text}"
        if total_chars + len(piece) > MAX_RAG_CONTEXT_CHARS:
            break
        context_parts.append(piece)
        total_chars += len(piece)

    context_text = "\n\n".join(context_parts).strip()
    if not context_text:
        return None, "Context rỗng sau khi gom chunk."

    prompt = f"""You are a medical evidence assistant.

Task:
Answer in Vietnamese, based ONLY on the provided context excerpts from Stockley's Drug Interactions.

User question:
{user_prompt}

Rules:
- Only use the context below.
- Do not invent facts not supported by the context.
- If the context is insufficient, say clearly that the retrieved excerpts are insufficient.
- Write a concise answer in Vietnamese (3-6 sentences).
- Mention that the answer is based on retrieved excerpts from Stockley.
- Prefer short paragraph form, not bullet-heavy.

CONTEXT:
{context_text}

FINAL ANSWER IN VIETNAMESE:
"""

    payload = {
        "model": OLLAMA_RAG_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 400,
        },
    }

    try:
        obj = _ollama_post_json("/api/generate", payload, timeout=OLLAMA_TIMEOUT_SEC)
        txt = clean_inline_text(obj.get("response", ""))
        if not txt:
            return None, f"Ollama trả về rỗng với model `{OLLAMA_RAG_MODEL}`."
        return txt, None
    except error.URLError as e:
        return None, f"Không kết nối được Ollama tại {OLLAMA_BASE_URL}: {e}"
    except Exception as e:
        return None, f"Lỗi sinh câu trả lời RAG qua Ollama model `{OLLAMA_RAG_MODEL}`: {e}"


# ============================================================
# LOADERS
# ============================================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource
def load_stockley_db():
    if not os.path.exists(PERSIST_STOCKLEY) or not os.listdir(PERSIST_STOCKLEY):
        return None
    emb = get_embeddings()
    return Chroma(
        persist_directory=PERSIST_STOCKLEY,
        embedding_function=emb,
        collection_name=STOCKLEY_COLLECTION,
    )


@st.cache_data(show_spinner=False)
def load_alias_map() -> Dict[str, str]:
    if not os.path.exists(ALIAS_MAP_PATH):
        return {}
    try:
        with open(ALIAS_MAP_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_alias_sources_index() -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(ALIAS_SOURCES_PATH):
        return idx

    with open(ALIAS_SOURCES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue

            alias_norm = (r.get("alias_norm") or "").strip()
            if not alias_norm:
                continue

            try:
                page = int(r.get("page"))
            except Exception:
                page = 10**9

            if alias_norm not in idx:
                idx[alias_norm] = r
            else:
                try:
                    old_page = int(idx[alias_norm].get("page"))
                except Exception:
                    old_page = 10**9
                if page < old_page:
                    idx[alias_norm] = r

    return idx


@st.cache_data(show_spinner=False)
def build_alias_catalog(alias_map: Dict[str, str], alias_sources: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    seen = set()

    # alias / tên thương mại
    for alias, canonical in alias_map.items():
        alias_norm = norm_text(alias)
        canonical_norm = norm_text(canonical)
        if not alias_norm:
            continue

        src = alias_sources.get(alias_norm) or alias_sources.get(alias) or {}
        duoc_url = src.get("url")
        if not duoc_url:
            try:
                duoc_url = build_remote_pdf_url("duoc_thu_qg_2018", int(src.get("page", 1) or 1))
            except Exception:
                duoc_url = None

        key = ("alias", alias_norm, canonical_norm)
        if key not in seen:
            seen.add(key)
            items.append(
                {
                    "term_norm": alias_norm,
                    "canonical": canonical_norm if canonical_norm else alias_norm,
                    "kind": "alias",
                    "display": alias,
                    "duoc_url": duoc_url,
                    "priority": 2,
                }
            )

    # canonical terms
    for canonical in set(alias_map.values()):
        canonical_norm = norm_text(canonical)
        if not canonical_norm:
            continue

        key = ("canonical", canonical_norm, canonical_norm)
        if key not in seen:
            seen.add(key)
            items.append(
                {
                    "term_norm": canonical_norm,
                    "canonical": canonical_norm,
                    "kind": "canonical",
                    "display": canonical,
                    "duoc_url": None,
                    "priority": 1,
                }
            )

    # fallback
    for alias, canonical in ALIAS_FALLBACK.items():
        alias_norm = norm_text(alias)
        canonical_norm = norm_text(canonical)

        key = ("fallback", alias_norm, canonical_norm)
        if key not in seen:
            seen.add(key)
            items.append(
                {
                    "term_norm": alias_norm,
                    "canonical": canonical_norm if canonical_norm else alias_norm,
                    "kind": "fallback",
                    "display": alias,
                    "duoc_url": None,
                    "priority": 3,
                }
            )

    items.sort(key=lambda x: (-len(x["term_norm"]), -x["priority"], x["term_norm"]))
    return items


# ============================================================
# INPUT PARSING
# ============================================================
QUESTION_NOISE_PATTERNS = [
    r"\bcó tương tác không\b", r"\bco tuong tac khong\b",
    r"\bdùng chung được không\b", r"\bdung chung duoc khong\b",
    r"\bcó uống rượu được không\b", r"\bco uong ruou duoc khong\b",
    r"\bcó sao không\b", r"\bco sao khong\b",
    r"\bđược không\b", r"\bduoc khong\b",
    r"\bcó tương tác gì\b", r"\bco tuong tac gi\b",
    r"\bcó tương tác được với các chất nào hay không\b",
    r"\bco tuong tac duoc voi cac chat nao hay khong\b",
    r"\bcó tương tác với thuốc nào\b",
    r"\bco tuong tac voi thuoc nao\b",
    r"\bnhững tương tác quan trọng nào\b",
    r"\binteractions?\b",
    r"\binteraction\b",
    r"\buống\b", r"\bdùng\b",
    r"\bhoi\b", r"\bhỏi\b",
    r"\btoi\b", r"\btôi\b",
]

ALIAS_FALLBACK = {
    "ruou": "alcohol",
    "rượu": "alcohol",
    "bia": "alcohol",
    "beer": "alcohol",
    "wine": "alcohol",
    "alcohol": "alcohol",
    "ethanol": "alcohol",
    "coffee": "caffeine",
    "coffe": "caffeine",
    "cafe": "caffeine",
    "cà phê": "caffeine",
    "caffeine": "caffeine",
}


def clean_query_for_split(s: str) -> str:
    t = s.strip()
    t = re.sub(r"[()]", " ", t)
    for p in QUESTION_NOISE_PATTERNS:
        t = re.sub(p, " ", t, flags=re.I)
    t = re.sub(r"\b(va|và|and|with)\b", " | ", t, flags=re.I)
    for ch in ["+", "/", ",", ";"]:
        t = t.replace(ch, " | ")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s*\|\s*", "|", t)
    return t.strip(" |")


def split_to_entities(user_text: str) -> List[str]:
    t = clean_query_for_split(user_text)
    parts = [p.strip() for p in t.split("|") if p.strip()]

    nt = norm_text(user_text)
    if len(parts) == 1 and any(x in nt for x in ["ruou", "rượu", "alcohol", "bia", "beer", "wine"]):
        parts.append("alcohol")

    if not parts:
        parts = [user_text.strip()]

    out = []
    seen = set()
    for p in parts:
        k = norm_text(p)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(p)

    return out[:MAX_ENTITIES]


# ============================================================
# ENTITY MAPPING
# ============================================================
def map_to_entity(raw_text: str, alias_map: Dict[str, str], alias_sources: Dict[str, Dict[str, Any]]) -> Entity:
    x = norm_text(raw_text)
    x = re.sub(r"^\s*(thuoc|thuốc)\s+", "", x).strip()
    x = x.strip(" ?.,;:!-/|+")

    if x in alias_map:
        canonical = alias_map[x]
        src = alias_sources.get(x) or {}
        duoc_url = src.get("url")
        if not duoc_url:
            try:
                duoc_url = build_remote_pdf_url("duoc_thu_qg_2018", int(src.get("page", 1) or 1))
            except Exception:
                duoc_url = None
        return Entity(
            raw=raw_text.strip(),
            canonical=canonical,
            drug_name_if_any=raw_text.strip(),
            duoc_url=duoc_url,
        )

    if x in ALIAS_FALLBACK:
        return Entity(raw=raw_text.strip(), canonical=ALIAS_FALLBACK[x], drug_name_if_any=None, duoc_url=None)

    return Entity(raw=raw_text.strip(), canonical=x, drug_name_if_any=None, duoc_url=None)


def query_parser_agent_extract_mentions(
    prompt: str,
    alias_map: Dict[str, str],
    alias_sources: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], List[AgentLog]]:
    logs: List[AgentLog] = []
    prompt_norm = norm_text(prompt)
    catalog = build_alias_catalog(alias_map, alias_sources)

    matches: List[Tuple[int, int, Dict[str, Any], str]] = []
    occupied: List[Tuple[int, int]] = []

    for item in catalog:
        pat = regex_term_pattern(item["term_norm"])
        for m in pat.finditer(prompt_norm):
            s, e = m.span()

            overlap = False
            for os_, oe_ in occupied:
                if not (e <= os_ or s >= oe_):
                    overlap = True
                    break
            if overlap:
                continue

            occupied.append((s, e))
            matches.append((s, e, item, prompt[s:e]))

    matches.sort(key=lambda x: x[0])

    raw_mentions: List[str] = []
    seen = set()
    for _, _, item, raw_span in matches:
        k = (item["term_norm"], item["canonical"])
        if k in seen:
            continue
        seen.add(k)
        raw_mentions.append(raw_span.strip())

    if raw_mentions:
        logs.append(AgentLog("query_parser_agent", "ok", f"Nhận diện được {len(raw_mentions)} tên/thực thể trong câu hỏi tự nhiên."))
        return raw_mentions[:MAX_ENTITIES], logs

    fallback = split_to_entities(prompt)
    logs.append(AgentLog("query_parser_agent", "warn", f"Không detect được entity rõ ràng bằng alias scan, fallback sang tách chuỗi thường: {len(fallback)} mục."))
    return fallback[:MAX_ENTITIES], logs


def entity_linker_agent_link_mentions(
    raw_mentions: List[str],
    alias_map: Dict[str, str],
    alias_sources: Dict[str, Dict[str, Any]],
) -> Tuple[List[Entity], List[AgentLog]]:
    logs: List[AgentLog] = []
    entities = [map_to_entity(x, alias_map, alias_sources) for x in raw_mentions]

    uniq_entities: List[Entity] = []
    seen = set()
    for e in entities:
        if not e.canonical or e.canonical in seen:
            continue
        seen.add(e.canonical)
        uniq_entities.append(e)

    logs.append(AgentLog("entity_linker_agent", "ok", f"Chuẩn hoá được {len(uniq_entities)} thực thể canonical duy nhất."))
    return uniq_entities, logs


def build_rag_focus_prompt(original_prompt: str, entities: List[Entity]) -> str:
    if not entities:
        return original_prompt

    canonical_terms = [e.canonical for e in entities]
    joined = " + ".join(canonical_terms)

    q_norm = norm_text(original_prompt)

    if any(x in q_norm for x in ["tuong tac", "interaction", "interact", "luu y", "can than", "có gì cần lưu ý"]):
        return f"Tương tác thuốc liên quan đến {joined}"

    if len(canonical_terms) == 1:
        return canonical_terms[0]

    return joined


def normalize_prompt_for_rag(
    prompt: str,
    alias_map: Dict[str, str],
    alias_sources: Dict[str, Dict[str, Any]],
) -> Tuple[str, List[Entity], List[AgentLog]]:
    parser_mentions, parser_logs = query_parser_agent_extract_mentions(prompt, alias_map, alias_sources)
    entities, linker_logs = entity_linker_agent_link_mentions(parser_mentions, alias_map, alias_sources)

    logs = parser_logs + linker_logs

    prompt_norm = build_rag_focus_prompt(prompt, entities)
    logs.append(AgentLog("entity_linker_agent", "ok", f"Prompt RAG sau chuẩn hoá: `{prompt_norm}`"))
    return prompt_norm, entities, logs


# ============================================================
# HEADER MATCH HELPERS
# ============================================================
def side_contains_entity(side_text: str, entity: str) -> bool:
    side_norm = norm_text(side_text)
    ent_norm = norm_text(entity)

    if not side_norm or not ent_norm:
        return False

    pattern = re.compile(rf"(?<![a-z0-9]){re.escape(ent_norm)}(?![a-z0-9])", flags=re.I)
    return bool(pattern.search(side_norm))


def header_matches_pair(header_text: str, a: str, b: str) -> bool:
    txt = clean_inline_text(header_text)
    if "+" not in txt:
        return False

    parts = txt.split("+", 1)
    if len(parts) != 2:
        return False

    left = parts[0].strip()
    right = parts[1].strip()

    direct = side_contains_entity(left, a) and side_contains_entity(right, b)
    reverse = side_contains_entity(left, b) and side_contains_entity(right, a)
    return direct or reverse


# ============================================================
# STOCKLEY PAGE BLOCKS / READING ORDER
# ============================================================
@st.cache_data(show_spinner=False)
def get_raw_stockley_page_blocks(page_no_1idx: int) -> List[Dict[str, Any]]:
    if not is_valid_stockley_page(page_no_1idx):
        return []

    path = RAW_PDFS["stockley_9e"]
    if not os.path.exists(path):
        return []

    with fitz.open(path) as doc:
        if page_no_1idx > len(doc):
            return []

        page = doc[page_no_1idx - 1]
        blocks = page.get_text("blocks")
        w = page.rect.width
        mid_x = w / 2.0

        out = []
        for b in blocks:
            x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
            txt = clean_inline_text(txt)
            if not txt:
                continue

            width = x1 - x0
            cx = (x0 + x1) / 2.0
            item = {
                "page": page_no_1idx,
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
                "width": float(width),
                "cx": float(cx),
                "text": txt,
                "text_norm": norm_text(txt),
                "is_full_width": bool(width > w * 0.80),
                "column": "left" if cx < mid_x else "right",
            }
            out.append(item)

        return out


def _order_blocks_two_columns_with_fullwidth(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not blocks:
        return []

    full_blocks = sorted([b for b in blocks if b["is_full_width"]], key=lambda z: (z["y0"], z["x0"]))
    col_blocks = [b for b in blocks if not b["is_full_width"]]

    ordered: List[Dict[str, Any]] = []
    prev_y = -10**9

    for fb in full_blocks:
        segment = [b for b in col_blocks if prev_y <= b["y0"] < fb["y0"]]
        left = sorted([b for b in segment if b["column"] == "left"], key=lambda z: (z["y0"], z["x0"]))
        right = sorted([b for b in segment if b["column"] == "right"], key=lambda z: (z["y0"], z["x0"]))
        ordered.extend(left)
        ordered.extend(right)
        ordered.append(fb)
        prev_y = fb["y1"]

    tail = [b for b in col_blocks if b["y0"] >= prev_y]
    left = sorted([b for b in tail if b["column"] == "left"], key=lambda z: (z["y0"], z["x0"]))
    right = sorted([b for b in tail if b["column"] == "right"], key=lambda z: (z["y0"], z["x0"]))
    ordered.extend(left)
    ordered.extend(right)

    final = []
    for b in ordered:
        b2 = dict(b)
        b2["uid"] = make_block_uid(b2["page"], b2)
        final.append(b2)
    return final


@st.cache_data(show_spinner=False)
def get_stockley_page_blocks(page_no_1idx: int) -> List[Dict[str, Any]]:
    raw = get_raw_stockley_page_blocks(page_no_1idx)
    return _order_blocks_two_columns_with_fullwidth(raw)


def get_stockley_blocks_across_pages(start_page: int, max_pages: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in range(start_page, min(STOCKLEY_LAST_CONTENT_PAGE, start_page + max_pages - 1) + 1):
        out.extend(get_stockley_page_blocks(p))
    return out


# ============================================================
# CORE RULES
# ============================================================
def retrieve_candidates_stockley(db: Chroma, query: str, k: int = 10) -> List[Document]:
    if db is None:
        return []
    try:
        docs = db.similarity_search(query, k=k, filter={"source": "stockley_9e"})
    except Exception:
        docs = db.similarity_search(query, k=max(2 * k, 12))
        docs = [d for d in docs if (d.metadata or {}).get("source") == "stockley_9e"][:k]
    return docs


def pages_from_docs(docs: List[Document]) -> List[int]:
    out = []
    for d in docs:
        p = (d.metadata or {}).get("page")
        try:
            p = int(p)
        except Exception:
            continue
        if is_valid_stockley_page(p):
            out.append(p)
    return uniq_keep_order(out)


def expand_candidate_pages(pages: List[int], radius: int = 1, max_out: int = MAX_INTERNAL_PAGES) -> List[int]:
    out = []
    for p in pages:
        for d in range(-radius, radius + 1):
            q = p + d
            if is_valid_stockley_page(q):
                out.append(q)
    return uniq_keep_order(out)[:max_out]


def is_reference_line(line: str) -> bool:
    ln = norm_text(line)
    return bool(re.match(r"^\d+\.", ln))


def is_clinical_evidence_heading(line: str) -> bool:
    ln = norm_text(line)
    return "clinical evidence" in ln


def is_probable_pair_header(line: str) -> bool:
    ln = clean_inline_text(line)
    if not ln:
        return False
    if "+" not in ln:
        return False
    if len(ln) > 100:
        return False
    if ln.endswith("."):
        return False
    if re.match(r"^\d+\.", ln):
        return False
    if len(ln.split()) > 16:
        return False
    return True


def score_header_block(block: Dict[str, Any], a: str, b: str) -> int:
    text = block["text"]
    score = 0
    if is_probable_pair_header(text):
        score += 5
    if block.get("is_full_width"):
        score += 4
    if len(text) <= 70:
        score += 2
    if ";" in text or "(" in text or ")" in text or " or " in text.lower():
        score += 1
    if header_matches_pair(text, a, b):
        score += 5
    return score


# ============================================================
# MULTI-AGENT LAYER
# ============================================================
def planner_agent_build_tasks(
    prompt: str,
    alias_map: Dict[str, str],
    alias_sources: Dict[str, Dict[str, Any]],
) -> Tuple[List[PairTask], List[Entity], List[AgentLog]]:
    logs: List[AgentLog] = []
    raw_items = split_to_entities(prompt)
    logs.append(AgentLog("planner_agent", "ok", f"Tách được {len(raw_items)} thực thể đầu vào."))

    entities = [map_to_entity(x, alias_map, alias_sources) for x in raw_items]
    uniq_entities = []
    seen = set()
    for e in entities:
        if not e.canonical or e.canonical in seen:
            continue
        seen.add(e.canonical)
        uniq_entities.append(e)

    logs.append(AgentLog("planner_agent", "ok", f"Sau chuẩn hoá còn {len(uniq_entities)} thực thể duy nhất."))

    tasks: List[PairTask] = []
    for a, b in itertools.combinations(uniq_entities[:MAX_ENTITIES], 2):
        queries = [
            f"{a.canonical} + {b.canonical}",
            f"{a.canonical} {b.canonical}",
            f"{a.canonical}",
            f"{b.canonical}",
        ]
        tasks.append(PairTask(a=a, b=b, queries=queries))

    logs.append(AgentLog("planner_agent", "ok", f"Lập kế hoạch cho {len(tasks)} cặp cần kiểm tra."))
    return tasks, uniq_entities, logs


def retriever_agent_stockley(stockley_db: Chroma, task: PairTask) -> RetrievalResult:
    logs: List[AgentLog] = []

    docs: List[Document] = []
    for q in task.queries:
        part = retrieve_candidates_stockley(stockley_db, q, k=MAX_STOCKLEY_CANDIDATES)
        docs.extend(part)
        logs.append(AgentLog("retriever_agent", "ok", f"Query `{q}` trả về {len(part)} candidate docs."))

    pages = pages_from_docs(docs)
    pages = expand_candidate_pages(pages, radius=1, max_out=MAX_INTERNAL_PAGES)

    for p in [22, 51, 59, 66, 68, 80, 86, 95, 165, 217, 295, 302, 539, 560]:
        if is_valid_stockley_page(p):
            pages.append(p)
    pages = uniq_keep_order(pages)

    logs.append(AgentLog("retriever_agent", "ok", f"Rút gọn còn {len(pages)} trang candidate."))
    return RetrievalResult(pages=pages, logs=logs)


def retrieve_general_rag_chunks(
    stockley_db: Chroma,
    prompt: str,
    entities: List[Entity],
) -> Tuple[List[RAGChunk], List[AgentLog]]:
    logs: List[AgentLog] = []

    canonical_terms = [e.canonical for e in entities]
    queries: List[str] = []

    if prompt:
        queries.append(prompt)
        queries.append(norm_text(prompt))

    if canonical_terms:
        queries.append(" + ".join(canonical_terms))
        for term in canonical_terms:
            queries.append(term)
            queries.append(f"drug interaction {term}")
            queries.append(f"{term} interaction")

        if len(canonical_terms) >= 2:
            for a, b in itertools.combinations(canonical_terms[:4], 2):
                queries.append(f"{a} + {b}")

    queries = uniq_keep_order([q for q in queries if clean_inline_text(q)])

    docs: List[Document] = []
    for q in queries:
        part = retrieve_candidates_stockley(stockley_db, q, k=MAX_STOCKLEY_CANDIDATES)
        docs.extend(part)
        logs.append(AgentLog("stockley_retriever_agent", "ok", f"RAG query `{q}` trả về {len(part)} candidate docs."))

    chunks: List[RAGChunk] = []
    seen = set()

    for d in docs:
        meta = d.metadata or {}
        try:
            page = int(meta.get("page"))
        except Exception:
            continue

        if not is_valid_stockley_page(page):
            continue

        txt = clean_chunk_text(getattr(d, "page_content", "") or "", max_len=850)
        if not txt:
            continue

        key = (page, txt[:200])
        if key in seen:
            continue
        seen.add(key)

        chunks.append(
            RAGChunk(
                page=page,
                text=txt,
                open_url=build_remote_pdf_url("stockley_9e", page),
            )
        )

        if len(chunks) >= MAX_RAG_DOCS:
            break

    logs.append(AgentLog("stockley_retriever_agent", "ok", f"Giữ lại {len(chunks)} chunks sau dedupe."))
    return chunks, logs


def header_agent_find_candidates(page_no_1idx: int, a: str, b: str) -> List[HeaderCandidate]:
    blocks = get_stockley_page_blocks(page_no_1idx)
    if not blocks:
        return []

    candidates: List[HeaderCandidate] = []
    for blk in blocks:
        txt = blk["text"]
        if not is_probable_pair_header(txt):
            continue
        if not header_matches_pair(txt, a, b):
            continue

        score = score_header_block(blk, a, b)
        candidates.append(
            HeaderCandidate(
                page=page_no_1idx,
                text=txt,
                rect=(blk["x0"], blk["y0"], blk["x1"], blk["y1"]),
                uid=blk["uid"],
                score=score,
                raw_block=blk,
            )
        )

    candidates.sort(key=lambda z: (-z.score, z.page, z.raw_block["y0"], z.raw_block["x0"]))
    return candidates


def summary_agent_extract_after_header(start_page: int, header_candidate: HeaderCandidate) -> Optional[str]:
    blocks = get_stockley_blocks_across_pages(start_page, MAX_SUMMARY_LOOKAHEAD_PAGES)
    if not blocks:
        return None

    try:
        start_idx = next(i for i, b in enumerate(blocks) if b["uid"] == header_candidate.uid)
    except StopIteration:
        return None

    collected: List[str] = []
    found_clinical = False

    for blk in blocks[start_idx + 1:]:
        txt = clean_inline_text(blk["text"])
        if not txt:
            continue

        if is_clinical_evidence_heading(txt):
            found_clinical = True
            break

        if is_probable_pair_header(txt):
            return None

        if is_reference_line(txt):
            continue

        collected.append(txt)

    if not found_clinical:
        return None

    summary = clean_inline_text(" ".join(collected))
    if len(summary) < 25:
        return None

    if "+" in summary and len(summary.split()) <= 12:
        return None

    return summary[:1500]


def verifier_agent_build_evidence(task: PairTask, retrieval: RetrievalResult) -> PairResult:
    logs: List[AgentLog] = []
    a = task.a
    b = task.b

    for p in retrieval.pages:
        headers = header_agent_find_candidates(p, a.canonical, b.canonical)
        logs.append(AgentLog("header_agent", "ok", f"Trang {p}: tìm được {len(headers)} header candidates."))

        if not headers:
            continue

        for hc in headers:
            summary = summary_agent_extract_after_header(p, hc)
            if not summary:
                logs.append(AgentLog("summary_agent", "skip", f"Trang {p}: header `{hc.text}` không trích được summary hợp lệ."))
                continue

            evidence = EvidenceItem(
                page=p,
                pair_text=clean_inline_text(hc.text),
                snippet_en=summary,
                interaction_summary_en=summary,
                open_url=build_remote_pdf_url("stockley_9e", p),
                header_rect=hc.rect,
                confidence="high" if hc.score >= 12 else "medium",
            )
            logs.append(AgentLog("verifier_agent", "ok", f"Trang {p}: candidate pass đủ 2 tầng, confidence={evidence.confidence}."))
            return PairResult(a=a, b=b, evidence=evidence, logs=retrieval.logs + logs)

    logs.append(AgentLog("verifier_agent", "miss", "Không tìm được candidate nào pass đủ 2 tầng."))
    return PairResult(a=a, b=b, evidence=None, logs=retrieval.logs + logs)


def translator_agent_translate_evidence(ev: EvidenceItem) -> Tuple[EvidenceItem, Optional[str]]:
    if ev is None or not ev.interaction_summary_en:
        return ev, None

    translated, err = translate_text_vi_ollama(ev.interaction_summary_en, OLLAMA_MODEL)
    if translated:
        ev.interaction_summary_vi = translated
        return ev, None
    return ev, err


def coordinator_agent_run_pair(stockley_db: Chroma, task: PairTask, use_translation: bool) -> PairResult:
    retrieval = retriever_agent_stockley(stockley_db, task)
    result = verifier_agent_build_evidence(task, retrieval)

    if result.evidence is not None and use_translation:
        ev, err = translator_agent_translate_evidence(result.evidence)
        result.evidence = ev
        if err:
            result.logs.append(AgentLog("translator_agent", "warn", err))
        else:
            result.logs.append(AgentLog("translator_agent", "ok", "Đã dịch summary sang tiếng Việt."))

    return result


def coordinator_agent_run_general_rag(
    stockley_db: Chroma,
    user_prompt: str,
    alias_map: Dict[str, str],
    alias_sources: Dict[str, Dict[str, Any]],
    use_rag_model: bool,
) -> GeneralRAGResult:
    prompt_norm, entities, norm_logs = normalize_prompt_for_rag(
        user_prompt,
        alias_map,
        alias_sources,
    )

    chunks, logs = retrieve_general_rag_chunks(stockley_db, prompt_norm, entities)
    logs = norm_logs + logs

    if not chunks:
        logs.append(AgentLog("answer_synthesizer_agent", "miss", "Không có chunk phù hợp để sinh câu trả lời."))
        return GeneralRAGResult(
            prompt=user_prompt,
            normalized_prompt=prompt_norm,
            entities=entities,
            answer_vi=None,
            answer_raw=None,
            chunks=[],
            logs=logs,
        )

    if not use_rag_model:
        logs.append(AgentLog("answer_synthesizer_agent", "warn", f"Model RAG `{OLLAMA_RAG_MODEL}` chưa sẵn sàng, chỉ hiển thị retrieved chunks."))
        return GeneralRAGResult(
            prompt=user_prompt,
            normalized_prompt=prompt_norm,
            entities=entities,
            answer_vi=None,
            answer_raw=None,
            chunks=chunks,
            logs=logs,
        )

    answer_vi, err = generate_general_rag_answer(prompt_norm, chunks)
    if err:
        logs.append(AgentLog("answer_synthesizer_agent", "warn", err))
    else:
        logs.append(AgentLog("answer_synthesizer_agent", "ok", f"Đã tạo câu trả lời RAG bằng model `{OLLAMA_RAG_MODEL}`."))

    return GeneralRAGResult(
        prompt=user_prompt,
        normalized_prompt=prompt_norm,
        entities=entities,
        answer_vi=answer_vi,
        answer_raw=answer_vi,
        chunks=chunks,
        logs=logs,
    )


# ============================================================
# HIGHLIGHT
# ============================================================
def _find_exact_pair_rects(page: fitz.Page, a: str, b: str) -> List[fitz.Rect]:
    patterns = [
        f"{a} + {b}",
        f"{b} + {a}",
        f"{a}+{b}",
        f"{b}+{a}",
    ]

    rects: List[fitz.Rect] = []
    seen = set()

    for pat in patterns:
        try:
            hits = page.search_for(pat, quads=False)
        except Exception:
            hits = []

        for r in hits:
            key = (round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1))
            if key in seen:
                continue
            seen.add(key)
            rects.append(r)

    return rects


def make_stockley_highlight_preview(ev: EvidenceItem, a: Entity, b: Entity) -> EvidenceItem:
    src_path = RAW_PDFS["stockley_9e"]
    if not os.path.exists(src_path):
        return ev

    key_data = {
        "src": "stockley",
        "page": ev.page,
        "a": a.canonical,
        "b": b.canonical,
        "pair_text": ev.pair_text,
    }
    cache_key = sha1_of_dict(key_data)
    cache_pdf = os.path.join(CACHE_PDF_DIR, f"stockley_{cache_key}.pdf")
    img_path = os.path.join(CACHE_IMG_DIR, f"stockley_{cache_key}_p1.png")

    if os.path.exists(cache_pdf) and os.path.exists(img_path):
        ev.cache_pdf_path = cache_pdf
        ev.preview_image_paths = [img_path]
        return ev

    try:
        with fitz.open(src_path) as src:
            out = fitz.open()
            out.insert_pdf(src, from_page=ev.page - 1, to_page=ev.page - 1)
            page = out[-1]

            rects: List[fitz.Rect] = []
            if ev.header_rect:
                x0, y0, x1, y1 = ev.header_rect
                rects = [fitz.Rect(x0, y0, x1, y1)]
            else:
                rects = _find_exact_pair_rects(page, a.canonical, b.canonical)

            for r in rects[:3]:
                try:
                    ann = page.add_highlight_annot(r)
                    ann.set_colors(stroke=COLOR_PAIR_HL)
                    ann.update()
                except Exception:
                    pass

                try:
                    shape = page.new_shape()
                    shape.draw_rect(r)
                    shape.finish(color=COLOR_PAIR_BOX, fill=None, width=1.5)
                    shape.commit()
                except Exception:
                    pass

            out.save(cache_pdf, deflate=True, garbage=4)
            out.close()

        with fitz.open(cache_pdf) as hdoc:
            pix = hdoc[0].get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)
            pix.save(img_path)

        ev.cache_pdf_path = cache_pdf
        ev.preview_image_paths = [img_path]
        return ev

    except Exception:
        return ev


# ============================================================
# UI HELPERS
# ============================================================
def render_agent_trace(logs: List[AgentLog], title: str):
    with st.expander(title, expanded=False):
        st.markdown("**Agent trace** là nhật ký để xem từng agent đã làm gì, pass/fail ở đâu, rất hữu ích để debug và tinh chỉnh hệ thống.")
        for lg in logs:
            icon = "✅" if lg.status == "ok" else ("⚠️" if lg.status == "warn" else "⏭️" if lg.status == "skip" else "❌")
            st.markdown(f"- {icon} **{lg.agent}**: {lg.detail}")


def render_pair_block(result: PairResult, pair_idx: int):
    a = result.a
    b = result.b
    ev = result.evidence

    st.markdown(f"## Cặp {pair_idx}: `{a.canonical}` + `{b.canonical}`")

    st.markdown("### Phân tích dược chất")
    st.markdown(
        f"- **Dược chất A:** `{a.canonical}`"
        + (f" (đầu vào: **{a.raw}**)" if a.raw and norm_text(a.raw) != norm_text(a.canonical) else "")
        + (f" (tên thuốc: **{a.drug_name_if_any}**)" if a.drug_name_if_any else "")
    )
    if a.drug_name_if_any and a.duoc_url:
        try:
            st.link_button(f"📌 Nguồn Dược thư cho A ({a.drug_name_if_any})", a.duoc_url, use_container_width=False)
        except TypeError:
            st.markdown(f"[Nguồn Dược thư cho A]({a.duoc_url})")

    st.markdown(
        f"- **Dược chất B:** `{b.canonical}`"
        + (f" (đầu vào: **{b.raw}**)" if b.raw and norm_text(b.raw) != norm_text(b.canonical) else "")
        + (f" (tên thuốc: **{b.drug_name_if_any}**)" if b.drug_name_if_any else "")
    )
    if b.drug_name_if_any and b.duoc_url:
        try:
            st.link_button(f"📌 Nguồn Dược thư cho B ({b.drug_name_if_any})", b.duoc_url, use_container_width=False)
        except TypeError:
            st.markdown(f"[Nguồn Dược thư cho B]({b.duoc_url})")

    if ev is None:
        st.markdown("**Chưa có thông tin về tương tác giữa 2 hoạt chất.**")
    else:
        st.markdown("**Có thông tin về tương tác giữa 2 hoạt chất.**")

        st.markdown("### Trích dẫn Stockley")
        st.markdown(f"- **Nguồn:** {SOURCE_LABELS['stockley_9e']}")

        if ev.interaction_summary_en:
            st.markdown("### Mô tả tương tác ngắn")
            st.markdown(ev.interaction_summary_en)

        if ev.interaction_summary_vi:
            st.markdown("### Bản dịch tiếng Việt")
            st.markdown(ev.interaction_summary_vi)

        if ev.open_url:
            try:
                st.link_button("🔗 Mở PDF Stockley đúng trang", ev.open_url, use_container_width=False)
            except TypeError:
                st.markdown(f"[Mở PDF Stockley đúng trang]({ev.open_url})")

        if ev.preview_image_paths:
            with st.expander("👁️ Bản xem trước highlight (Stockley)", expanded=True):
                for p in ev.preview_image_paths:
                    if os.path.exists(p):
                        st.image(
                            p,
                            use_container_width=True,
                            caption=f"Stockley — trang {ev.page} (highlight đúng header match)",
                        )

        if ev.cache_pdf_path and os.path.exists(ev.cache_pdf_path):
            with open(ev.cache_pdf_path, "rb") as f:
                st.download_button(
                    "⬇️ Tải PDF highlight (Stockley)",
                    data=f.read(),
                    file_name=os.path.basename(ev.cache_pdf_path),
                    mime="application/pdf",
                    key=f"dl_pair_{pair_idx}_{ev.page}_{a.canonical}_{b.canonical}",
                    use_container_width=False,
                )

    render_agent_trace(result.logs, f"🧠 Agent trace cho cặp {pair_idx}")
    st.markdown("---")


def render_general_rag_block(result: GeneralRAGResult):
    st.markdown("## RAG mode")

    st.markdown("### Câu hỏi gốc")
    st.markdown(result.prompt)

    st.markdown("### Chuẩn hoá thực thể")
    if result.entities:
        for e in result.entities:
            line = f"- **{e.raw}** → `{e.canonical}`"
            if e.drug_name_if_any and e.duoc_url:
                st.markdown(line)
                try:
                    st.link_button(
                        f"📌 Nguồn Dược thư ({e.drug_name_if_any})",
                        e.duoc_url,
                        use_container_width=False,
                        key=f"duoc_rag_{e.raw}_{e.canonical}",
                    )
                except TypeError:
                    st.markdown(f"[Nguồn Dược thư]({e.duoc_url})")
            else:
                st.markdown(line)
    else:
        st.markdown("- Không chuẩn hoá được thực thể nào, dùng câu hỏi gốc để retrieve.")

    st.markdown("### Prompt sau chuẩn hoá để retrieve")
    st.code(result.normalized_prompt)

    st.markdown("### Câu trả lời tổng hợp")
    if result.answer_vi:
        st.markdown(result.answer_vi)
    else:
        st.warning("Chưa tạo được câu trả lời tổng hợp bằng model RAG. Bên dưới là các đoạn context đã retrieve.")

    st.markdown("### Retrieved context từ Stockley")
    for i, ch in enumerate(result.chunks, start=1):
        st.markdown(f"**Chunk {i} — trang {ch.page}**")
        st.markdown(ch.text)
        if ch.open_url:
            try:
                st.link_button(
                    f"🔗 Mở PDF trang {ch.page}",
                    ch.open_url,
                    use_container_width=False,
                    key=f"open_general_{i}_{ch.page}"
                )
            except TypeError:
                st.markdown(f"[Mở PDF trang {ch.page}]({ch.open_url})")
        st.markdown("---")

    render_agent_trace(result.logs, "🧠 Agent trace cho RAG mode")


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_full_pipeline(
    prompt: str,
    stockley_db: Chroma,
    alias_map: Dict[str, str],
    alias_sources: Dict[str, Dict[str, Any]],
    translate_ok: bool,
    rag_ok: bool,
    feature_mode: str,
):
    with st.chat_message("assistant"):
        try:
            with st.spinner("Coordinator agent đang điều phối các subagents..."):

                # ====================================================
                # FEATURE 2: RAG mode
                # ====================================================
                if feature_mode == FEATURE_RAG:
                    result = coordinator_agent_run_general_rag(
                        stockley_db=stockley_db,
                        user_prompt=prompt,
                        alias_map=alias_map,
                        alias_sources=alias_sources,
                        use_rag_model=rag_ok,
                    )

                    render_general_rag_block(result)

                    short_answer = result.answer_vi or "Đã retrieve các context liên quan nhưng chưa tổng hợp được câu trả lời tự động."
                    st.session_state.history.append(
                        {
                            "role": "assistant",
                            "content": f"Kết quả RAG:\n{short_answer}",
                        }
                    )
                    return

                # ====================================================
                # FEATURE 1: Pairwise mode
                # ====================================================
                tasks, uniq_entities, planner_logs = planner_agent_build_tasks(prompt, alias_map, alias_sources)

                if len(uniq_entities) == 0:
                    answer = "Không nhận diện được dược chất hợp lệ để tra cứu."
                    st.markdown(answer)
                    st.session_state.history.append({"role": "assistant", "content": answer})
                    return

                if len(uniq_entities) < 2:
                    answer = (
                        "Chế độ hiện tại là **Kiểm tra tương tác theo cặp**, "
                        "nên bạn hãy nhập ít nhất 2 thuốc hoặc dược chất.\n\n"
                        "Ví dụ: `paracetamol + caffeine`"
                    )
                    st.markdown(answer)
                    st.session_state.history.append({"role": "assistant", "content": answer})
                    return

                if len(tasks) > 10:
                    st.info(f"Có {len(tasks)} cặp cần kiểm tra.")

                summary_lines = []

                for i, task in enumerate(tasks, start=1):
                    pair_result = coordinator_agent_run_pair(
                        stockley_db=stockley_db,
                        task=task,
                        use_translation=translate_ok,
                    )

                    if pair_result.evidence is not None:
                        pair_result.evidence = make_stockley_highlight_preview(
                            pair_result.evidence, task.a, task.b
                        )

                    if i == 1:
                        pair_result.logs = planner_logs + pair_result.logs

                    render_pair_block(pair_result, i)

                    ev = pair_result.evidence
                    if ev is None:
                        summary_lines.append(
                            f"[{i}] {task.a.canonical} + {task.b.canonical}: chưa có match đủ 2 tầng"
                        )
                    else:
                        if ev.interaction_summary_vi:
                            summary_lines.append(
                                f"[{i}] {task.a.canonical} + {task.b.canonical}: có header match ở trang {ev.page}, có summary hợp lệ, đã dịch VI"
                            )
                        else:
                            summary_lines.append(
                                f"[{i}] {task.a.canonical} + {task.b.canonical}: có header match ở trang {ev.page}, có summary hợp lệ"
                            )

                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": "Kết quả theo cặp:\n" + "\n".join(f"- {x}" for x in summary_lines),
                    }
                )

        except Exception as e:
            err = f"Đã xảy ra lỗi khi xử lý: {e}"
            st.error(err)
            st.session_state.history.append({"role": "assistant", "content": err})


# ============================================================
# MAIN
# ============================================================
st.set_page_config(page_title="DrugSafe AI", page_icon="💊", layout="wide")
st.title("💊 DrugSafe AI - tra cứu tương tác thuốc")
st.warning("⚠️ Công cụ hỗ trợ tham khảo, không thay thế tư vấn bác sĩ/dược sĩ.")

feature_mode = st.selectbox(
    "Chọn chế độ",
    [FEATURE_PAIRWISE, FEATURE_RAG],
    index=0,
    help="Bạn có thể đổi chế độ bất cứ lúc nào. Chọn xong rồi hỏi tiếp là được.",
)

if feature_mode == FEATURE_PAIRWISE:
    st.caption("Chế độ hiện tại: kiểm tra tương tác theo từng cặp, có verify header và highlight PDF.")
else:
    st.caption("Chế độ hiện tại: RAG multi-agent. Hệ thống sẽ tách entity từ câu hỏi, chuẩn hoá tên thương mại sang dược chất, retrieve từ Stockley, rồi mới tổng hợp câu trả lời.")

stockley_db = load_stockley_db()
if stockley_db is None:
    st.error("❌ Chưa có Chroma DB Stockley. Hãy build trước: `python etl/04_build_chroma.py`")
    st.stop()

if not os.path.exists(RAW_PDFS["stockley_9e"]):
    st.error("❌ Thiếu file `data/raw/stockley.pdf` để tạo bản xem trước highlight.")
    st.stop()

alias_map = load_alias_map()
alias_sources = load_alias_sources_index()

translate_ok, translate_msg = check_ollama_model_available(OLLAMA_MODEL)
rag_ok, rag_msg = check_ollama_model_available(OLLAMA_RAG_MODEL)

if translate_ok:
    st.success(f"✅ Model dịch sẵn sàng: `{OLLAMA_MODEL}`")
else:
    st.info(f"ℹ️ Model dịch chưa sẵn sàng: {translate_msg}")

if rag_ok:
    st.success(f"✅ Model RAG sẵn sàng: `{OLLAMA_RAG_MODEL}`")
else:
    st.info(f"ℹ️ Model RAG chưa sẵn sàng: {rag_msg}")

if "history" not in st.session_state:
    st.session_state.history = [
        {
            "role": "assistant",
            "content": (
                "Xin chào tôi là chatbot tra cứu tương tác thuốc.\n\n"
                "Bạn có thể đổi chế độ ở phía trên bất cứ lúc nào.\n\n"
                "Ví dụ:\n"
                "- Pairwise mode: `paracetamol + caffeine`\n"
                "- Pairwise mode: `Disopyramide + Rifampicin + paracetamol`\n"
                "- RAG mode: `warfarin có những tương tác quan trọng nào?`\n"
                "- RAG mode: `Tatanol và rượu có gì cần lưu ý?`\n"
                "- RAG mode: `Pancidol có tương tác với các chất nào?`"
            ),
        }
    ]

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Nhập câu hỏi hoặc tên thuốc để tra cứu")

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    run_full_pipeline(
        prompt,
        stockley_db,
        alias_map,
        alias_sources,
        translate_ok,
        rag_ok,
        feature_mode,
    )