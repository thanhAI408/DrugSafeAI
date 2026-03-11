# etl/02_parse_pdf_to_jsonl.py
import json
import pathlib
import re
from typing import Dict, List, Optional, Tuple

import fitz  # pymupdf

try:
    from unidecode import unidecode
except Exception:
    def unidecode(x: str) -> str:
        return x


# =========================
# CONFIG
# =========================
PAGE_RANGES = {
    "duoc_thu_qg_2018": (100, 1496),
    "stockley_9e": (22, 1587),
}

RAW_PDFS = {
    "duoc_thu_qg_2018": "data/raw/duoc_thu_2018.pdf",
    "stockley_9e": "data/raw/stockley.pdf",
}

BASE_URL = {
    "duoc_thu_qg_2018": "https://trungtamthuoc.com/upload/pdf/duoc-thu-quoc-gia-viet-nam-2018.pdf",
    "stockley_9e": "https://eprints.poltekkesadisutjipto.ac.id/id/eprint/2137/1/Stockley%27s%20Drug%20Interactions%2C%209th%20Edition.pdf",
}

OUT_JSONL = {
    "duoc_thu_qg_2018": "data/processed/pdf_pages_duoc_thu.jsonl",
    "stockley_9e": "data/processed/pdf_pages_stockley.jsonl",
}

OUT_ALIAS_MAP = "data/processed/alias_map_duoc_thu.json"
OUT_ALIAS_SOURCES = "data/processed/alias_map_duoc_thu_sources.jsonl"

# Dược thư header hay nằm sát trên cùng. Chỉ bỏ block ngắn trong vùng top.
TOP_HEADER_Y_MAX = 70.0
TOP_HEADER_MAX_LEN = 40

# Fix: mục "Tên thương mại" đôi khi cực dài -> tăng limit (tránh rụng cuối danh sách)
TRADE_SECTION_MAX_CHARS = 6000


# =========================
# HELPERS
# =========================
def is_valid_page(source: str, page_no: int) -> bool:
    lo, hi = PAGE_RANGES[source]
    return lo <= page_no <= hi


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def norm_key(s: str) -> str:
    """key chuẩn: lower + bỏ dấu + gom space"""
    t = unidecode((s or "").strip().lower())
    t = re.sub(r"\s+", " ", t).strip()
    return t


def safe_write_jsonl_line(f, obj: Dict):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def looks_like_top_header_noise(y1: float, txt: str) -> bool:
    """
    Bỏ DTQGVN 2 / "Acitretin 145" ...
    Điều kiện: nằm sát top, ngắn, có dtqgvn hoặc kết thúc bằng số trang
    """
    if y1 > TOP_HEADER_Y_MAX:
        return False
    t = norm_space(txt)
    if not t:
        return True
    if len(t) > TOP_HEADER_MAX_LEN:
        return False
    nt = norm_key(t)
    if "dtqgvn" in nt:
        return True
    if re.search(r"\b\d{1,4}\b$", t):
        return True
    return False


# =========================
# BLOCK SPLIT / ORDER
# =========================
def split_blocks_by_column_duoc_thu(blocks, page_width: float):
    mid_x = page_width / 2.0
    left, right, full = [], [], []

    for b in blocks:
        x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
        if not txt or not str(txt).strip():
            continue

        if looks_like_top_header_noise(y1, str(txt)):
            continue

        width = x1 - x0
        cx = (x0 + x1) / 2.0

        if width > page_width * 0.80:
            full.append(b)
        elif cx < mid_x:
            left.append(b)
        else:
            right.append(b)

    left_sorted = sorted(left, key=lambda x: (x[1], x[0]))
    right_sorted = sorted(right, key=lambda x: (x[1], x[0]))
    full_sorted = sorted(full, key=lambda x: (x[1], x[0]))
    return {"left": left_sorted, "right": right_sorted, "full": full_sorted}


def split_blocks_by_column_generic(blocks, page_width: float):
    """dùng cho Stockley (không lọc header)"""
    mid_x = page_width / 2.0
    left, right, full = [], [], []
    for b in blocks:
        x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
        if not txt or not str(txt).strip():
            continue
        width = x1 - x0
        cx = (x0 + x1) / 2.0
        if width > page_width * 0.80:
            full.append(b)
        elif cx < mid_x:
            left.append(b)
        else:
            right.append(b)

    left_sorted = sorted(left, key=lambda x: (x[1], x[0]))
    right_sorted = sorted(right, key=lambda x: (x[1], x[0]))
    full_sorted = sorted(full, key=lambda x: (x[1], x[0]))
    return {"left": left_sorted, "right": right_sorted, "full": full_sorted}


def sort_blocks_two_column_reading(blocks, page_width: float):
    parts = split_blocks_by_column_generic(blocks, page_width)
    full_sorted = parts["full"]
    left_sorted = parts["left"]
    right_sorted = parts["right"]
    return full_sorted[:1] + left_sorted + right_sorted + full_sorted[1:]


def blocks_to_text(blocks) -> str:
    texts = []
    for b in blocks:
        txt = (b[4] or "").strip()
        if txt:
            texts.append(txt)
    text = "\n".join(texts)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def extract_duoc_thu_page_text_by_column(page: fitz.Page) -> Dict[str, str]:
    blocks = page.get_text("blocks")
    parts = split_blocks_by_column_duoc_thu(blocks, page.rect.width)
    return {
        "left": blocks_to_text(parts["left"]),
        "right": blocks_to_text(parts["right"]),
        "full": blocks_to_text(parts["full"] + parts["left"] + parts["right"]),
    }


def extract_stockley_page_text_ordered(page: fitz.Page) -> str:
    blocks = page.get_text("blocks")
    ordered = sort_blocks_two_column_reading(blocks, page.rect.width)
    return blocks_to_text(ordered)


# =========================
# DƯỢC THƯ: ENTRY SPLIT
# =========================
def is_heading_all_caps(line: str) -> bool:
    t = (line or "").strip()
    if len(t) < 3 or len(t) > 45:
        return False
    if not re.search(r"[A-Za-zÀ-Ỹ]", t):
        return False
    return (t == t.upper()) and (not re.search(r"[a-zà-ỹ]", t))


def clean_generic(raw: str) -> str:
    """
    Fix: 'Aciclovir (Acyclovir).' -> 'Aciclovir'
    """
    s = norm_space(raw)
    # cắt phần trong ngoặc/ dấu ';' / dấu '.' ',' đầu tiên
    s = re.split(r"[\(\);]", s)[0].strip()
    s = re.split(r"[.;,]", s)[0].strip()
    s = s.strip(" :|-–—")
    return s


def find_generic_markers(lines: List[str]) -> List[Tuple[int, str]]:
    out = []
    n = len(lines)
    for i, ln in enumerate(lines):
        nt = norm_key(ln)
        if "ten chung quoc te" in nt:
            after = ""
            if ":" in ln:
                after = ln.split(":", 1)[1].strip()
            if (not after) and (i + 1 < n):
                after = lines[i + 1].strip()
            g = clean_generic(after)
            if g:
                out.append((i, g))
    return out


def find_prev_allcaps_heading(lines: List[str], marker_idx: int, lookback: int = 12) -> Optional[int]:
    for j in range(marker_idx - 1, max(-1, marker_idx - lookback) - 1, -1):
        if is_heading_all_caps(lines[j]):
            return j
    return None


def slice_entry(lines: List[str], start_line: int, end_line: int) -> str:
    s = start_line
    for j in range(max(0, start_line - 6), start_line):
        if is_heading_all_caps(lines[j]):
            s = j
            break
    return "\n".join(lines[s:end_line]).strip()


def split_column_into_segments(lines: List[str], current_generic: str) -> Tuple[List[Tuple[str, str]], str]:
    segments: List[Tuple[str, str]] = []
    markers = find_generic_markers(lines)

    def add_segment(g: str, text: str):
        text = (text or "").strip()
        if len(text) < 120:
            return
        segments.append((g or "", text))

    cur = current_generic or ""

    if not markers:
        add_segment(cur, "\n".join(lines))
        return segments, cur

    first_idx = markers[0][0]
    if first_idx > 0:
        add_segment(cur, "\n".join(lines[:first_idx]))

    for mi, (idx, graw) in enumerate(markers):
        next_marker = markers[mi + 1][0] if mi + 1 < len(markers) else len(lines)
        end_idx = next_marker
        if mi + 1 < len(markers):
            hd = find_prev_allcaps_heading(lines, next_marker, lookback=12)
            if hd is not None and hd > idx:
                end_idx = hd  # cắt trước heading thuốc mới (ACICLOVIR, ...)
        entry_text = slice_entry(lines, idx, end_idx)

        gnorm = norm_key(graw)
        if gnorm:
            cur = gnorm

        add_segment(cur, entry_text)

    return segments, cur


# =========================
# TRÍCH "TÊN THƯƠNG MẠI"
# =========================
TRADE_HEAD_RE = re.compile(r"(Tên\s*thương\s*mại|Ten\s*thuong\s*mai)", flags=re.I)

STOP_HEADINGS = [
    "Tên chung quốc tế", "Ten chung quoc te",
    "Mã ATC", "Ma ATC",
    "Loại thuốc", "Loai thuoc",
    "Dạng thuốc", "Dang thuoc",
    "Dược lý", "Duoc ly",
    "Chỉ định", "Chi dinh",
    "Chống chỉ định", "Chong chi dinh",
    "Thận trọng", "Than trong",
    "Tương tác thuốc", "Tuong tac thuoc",
    "Tác dụng không mong muốn", "Tac dung khong mong muon",
    "Liều lượng", "Lieu luong",
    "Quá liều", "Qua lieu",
    "Bảo quản", "Bao quan",
    "Thông tin", "Thong tin",
]
STOP_RE = re.compile(r"(?im)^\s*(?:" + "|".join(re.escape(x) for x in STOP_HEADINGS) + r")\b")


def extract_trade_names_from_segment(seg_text: str) -> List[str]:
    if not seg_text:
        return []

    m = TRADE_HEAD_RE.search(seg_text)
    if not m:
        return []

    tail = seg_text[m.end():]
    tail = re.sub(r"^\s*[:\-–—]?\s*", "", tail)

    # stop nếu gặp heading ALL-CAPS (thuốc mới)
    tail_lines = tail.splitlines()
    cut_pos = None
    pos = 0
    for ln in tail_lines:
        if is_heading_all_caps(ln):
            cut_pos = pos
            break
        pos += len(ln) + 1
    if cut_pos is not None:
        tail = tail[:cut_pos]

    # stop theo heading section
    mstop = STOP_RE.search(tail)
    sec = tail[: mstop.start()] if mstop else tail

    sec = sec.strip()[:TRADE_SECTION_MAX_CHARS]
    sec = re.sub(r"\s+", " ", sec).strip()
    if not sec:
        return []

    parts = re.split(r"[;,/•\u2022]\s*", sec)
    out = []
    for p in parts:
        p = (p or "").strip()
        if not p:
            continue
        p = re.sub(r"\(.*?\)", " ", p)
        p = re.sub(r"\b\d+(\.\d+)?\s*(mg|g|mcg|µg|ml|iu|%)\b", " ", p, flags=re.I)
        p = norm_space(p).strip(" .:-–—")
        if len(p) < 2 or len(p) > 80:
            continue
        if not re.search(r"[A-Za-zÀ-ỹ]", p):
            continue
        out.append(p)

    seen = set()
    uniq = []
    for x in out:
        k = norm_key(x)
        if not k or k in seen:
            continue
        seen.add(k)
        uniq.append(x)
    return uniq


# =========================
# JSONL WRITERS
# =========================
def write_record_jsonl(f, text: str, source: str, page_no: int, column: Optional[str], generic_norm: str):
    md = {
        "source": source,
        "page": int(page_no),
        "url": f"{BASE_URL[source]}#page={page_no}",
        "column": column or "",
        "generic": generic_norm or "",
    }
    md["title"] = f"{source} - page {page_no}" + (f" - {column}" if column else "") + (f" - {generic_norm}" if generic_norm else "")
    safe_write_jsonl_line(f, {"text": text, "metadata": md})


def write_alias_source_line(f, alias_raw: str, alias_norm: str, generic_norm: str, page_no: int, column: str):
    safe_write_jsonl_line(
        f,
        {
            "alias_raw": alias_raw,
            "alias_norm": alias_norm,
            "generic": generic_norm,
            "page": int(page_no),
            "column": column,
            "url": f"{BASE_URL['duoc_thu_qg_2018']}#page={page_no}",
            "source": "duoc_thu_qg_2018",
        },
    )


# =========================
# PARSERS
# =========================
def parse_stockley():
    source = "stockley_9e"
    pdf_path = pathlib.Path(RAW_PDFS[source])
    out_path = pathlib.Path(OUT_JSONL[source])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    kept = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(doc.page_count):
            page_no = i + 1
            if not is_valid_page(source, page_no):
                continue
            page = doc.load_page(i)
            text = extract_stockley_page_text_ordered(page).strip()
            if not text:
                continue
            write_record_jsonl(f, text, source, page_no, column=None, generic_norm="")
            kept += 1

    print(f"[OK] stockley kept {kept} records -> {out_path}")


def parse_duoc_thu():
    source = "duoc_thu_qg_2018"
    pdf_path = pathlib.Path(RAW_PDFS[source])
    out_path = pathlib.Path(OUT_JSONL[source])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    alias_map: Dict[str, str] = {}
    conflicts = 0
    alias_lines = 0
    kept = 0

    alias_sources_path = pathlib.Path(OUT_ALIAS_SOURCES)
    alias_sources_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)

    current_generic = ""

    with out_path.open("w", encoding="utf-8") as f_out, alias_sources_path.open("w", encoding="utf-8") as f_src:
        for i in range(doc.page_count):
            page_no = i + 1
            if not is_valid_page(source, page_no):
                continue

            page = doc.load_page(i)
            cols = extract_duoc_thu_page_text_by_column(page)

            for col_name in ["left", "right"]:
                col_text = (cols.get(col_name) or "").strip()
                if len(col_text) < 120:
                    continue

                lines = [ln.strip() for ln in col_text.splitlines() if ln.strip()]
                segments, current_generic = split_column_into_segments(lines, current_generic)

                if not segments:
                    write_record_jsonl(f_out, col_text, source, page_no, column=col_name, generic_norm=current_generic)
                    kept += 1
                    continue

                for gnorm, seg_text in segments:
                    write_record_jsonl(f_out, seg_text, source, page_no, column=col_name, generic_norm=gnorm)
                    kept += 1

                    if gnorm:
                        trade_names = extract_trade_names_from_segment(seg_text)
                        for tn in trade_names:
                            tn_norm = norm_key(tn)
                            if not tn_norm:
                                continue
                            if tn_norm in alias_map and alias_map[tn_norm] != gnorm:
                                conflicts += 1
                            else:
                                alias_map[tn_norm] = gnorm
                            write_alias_source_line(f_src, tn, tn_norm, gnorm, page_no, col_name)
                            alias_lines += 1

    alias_path = pathlib.Path(OUT_ALIAS_MAP)
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    with alias_path.open("w", encoding="utf-8") as wf:
        json.dump(alias_map, wf, ensure_ascii=False, indent=2)

    print(f"[OK] duoc_thu kept {kept} records -> {out_path}")
    print(f"[OK] alias_map -> {alias_path} (size={len(alias_map)}, conflicts={conflicts}, alias_lines={alias_lines})")
    print(f"[OK] alias_sources -> {alias_sources_path}")


if __name__ == "__main__":
    parse_stockley()
    parse_duoc_thu()
    print("Done.")