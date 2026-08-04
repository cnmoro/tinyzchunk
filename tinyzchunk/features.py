"""Vectorized per-character feature extraction for the tiny chunker.

All features are computed purely from the raw character string, with numpy only,
so the same code powers both training and inference (no GPU, no tokenizer).

Two design rules keep the library robust across wildly different inputs:

1. `normalize_text` is **length preserving**.  Every substitution maps one
   character to exactly one character, so character offsets computed on the
   normalized text index the original text unchanged, and chunks can be sliced
   straight out of the caller's string.
2. Features prefer *relative structure* (does this line look like its
   neighbours? how long is it compared with the document average?) over
   hard-coded markup, so unseen formats still produce usable signals.
"""
import hashlib
import re

import numpy as np

# ---------------------------------------------------------------- normalization

# every mapping below is exactly one character -> one character
_CHAR_MAP = {
    # spaces of all widths
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", " ": " ", " ": " ", " ": " ", "　": " ",
    # zero-width / invisible
    "​": " ", "‌": " ", "‍": " ", "﻿": " ", "⁠": " ",
    # quotes
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    # dashes
    "–": "-", "—": "-", "―": "-", "−": "-", "­": "-",
    # bullets
    "‣": "•", "●": "•", "▪": "•", "◦": "•",
    "⁃": "•", "·": "•", "∙": "•",
    # misc
    "…": ".",   # ellipsis -> sentence terminator
    "\x0c": "\n",    # form feed (PDF page break) -> line break
    "\x0b": "\n",
    " ": "\n", " ": "\n",
}
_TRANS = str.maketrans(_CHAR_MAP)


def normalize_with_map(text: str):
    """Canonicalise unicode; return (normalized, index_map).

    Every substitution maps one character to one character, so offsets are
    preserved -- except for the `\\r` of a CRLF pair, which is deleted outright
    so that a Windows file scores *identically* to a unix one rather than
    merely similarly.  When characters are dropped, `index_map[i]` gives the
    offset of `normalized[i]` in the original string; it is None when nothing
    was dropped and offsets already line up.
    """
    if "\r" not in text:
        return text.translate(_TRANS), None
    arr = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    is_cr = arr == 0x0D
    next_lf = np.zeros(len(arr), dtype=bool)
    next_lf[:-1] = arr[1:] == 0x0A
    keep = ~(is_cr & next_lf)          # drop only the CR of a CRLF pair
    index_map = np.flatnonzero(keep)
    stripped = text.replace("\r\n", "\n").replace("\r", "\n")
    return stripped.translate(_TRANS), index_map


def normalize_text(text: str) -> str:
    """Canonicalised text only (see `normalize_with_map` for offset tracking)."""
    return normalize_with_map(text)[0]


# ---------------------------------------------------------------------- regexes

ENUM_RE = re.compile(r"^\s*\(?\d+\s*[.)\]]\s")
BULLET_RE = re.compile(r"^\s*[-*•▸‣]+\s")
HASH_RE = re.compile(r"^\s*#{1,6}\s")
CAPSWORD_RE = re.compile(r"^\s*([A-ZÀ-Ý][a-zA-Zà-ÿ]+)(\s|[:.])")
DASH_RE = re.compile(r"^\s*[-–—]{1,2}\s")
TITLE_RE = re.compile(r"^\s*[A-ZÀ-Ý][a-zà-ÿ]+")
CAPS_TOKEN_RE = re.compile(r"[A-ZÀ-Ý][A-ZÀ-Ý0-9&\-.']*")
ENDS_COLON_RE = re.compile(r":\s*$")
ENDS_Q_RE = re.compile(r"[?¿]\s*$|\?\*{1,2}\s*$")
ENDS_SENT_RE = re.compile(r"[.!?;:]\s*$")
FIELD_RE = re.compile(r"^[A-ZÀ-Ý][\wÀ-ÿ ]{2,40}:\s")
BOLD_FIELD_RE = re.compile(r"^[-*•]?\s*\*{1,2}[\wÀ-ÿ ]{1,40}:\*{0,2}")

# EN + PT-BR interrogative openers
QWORD_RE = re.compile(
    r"^(qual|quais|quem|quando|onde|como|por que|por quê|porque|quanto|quantos"
    r"|quantas|o que|que|existe|existem|tem|há|pode|posso|poderia|deve|devo"
    r"|é possível|para que|será que|você sabe|preciso|precisa"
    r"|what|which|who|whom|whose|when|where|how|why|can|could|does|do|did|is|are"
    r"|was|were|will|would|should|may|might|am|have|has)\b",
    re.I)
DATE_RE = re.compile(r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}")
MONTH_RE = re.compile(
    r"\b(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez|feb|apr|may|aug|sep|oct|dec"
    r"|janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro"
    r"|novembro|dezembro|january|february|march|april|june|july|august|september"
    r"|october|november|december)[a-zç]*\.?\s+\d{1,2}", re.I)
YEAR_CTX_RE = re.compile(
    r"(de|em|dia|data|sessão|aula|palestra|evento|workshop|session|day|date)\b", re.I)
TIME_RE = re.compile(r"\b\d{1,2}\s?[:h]\s?\d{2}\b|\b([01]?\d|2[0-3])\s?[hH]\b|\b\d{1,2}\s?(am|pm)\b", re.I)

# --- structural / format-agnostic ---
FENCE_RE = re.compile(r"^\s*(```|~~~)")
HR_RE = re.compile(r"^\s*([-*_=])\1{2,}\s*$")
SETEXT_RE = re.compile(r"^\s*(=|-){3,}\s*$")
QUOTE_LINE_RE = re.compile(r"^\s*>")
PAGE_NUM_RE = re.compile(
    r"^\s*(p[aá]g(ina)?\.?\s*)?[-\[(]?\s*(\d{1,4}|[ivxlcdm]{1,7})\s*[-\])]?"
    r"(\s*(/|de|of)\s*\d{1,4})?\s*$", re.I)
URL_RE = re.compile(r"(https?://|www\.|\b[\w.+-]+@[\w-]+\.[\w.]{2,})", re.I)
SPEAKER_RE = re.compile(
    r"^\s*(\[\s*\d{1,2}[:h]\d{2}(:\d{2})?\s*\]\s*|\(\d{1,2}[:h]\d{2}\)\s*|\d{1,2}[:h]\d{2}\s+)?"
    r"([A-ZÀ-Ý][\wÀ-ÿ.'-]*(\s+[A-ZÀ-Ý][\wÀ-ÿ.'-]*){0,3})\s*[:>–-]\s")
LEGAL_RE = re.compile(
    r"^\s*(art(igo)?\.?\s*\d+|§\s*\d+|par[aá]grafo\s|inciso\s|se[cç][aã]o\s"
    r"|cap[ií]tulo\s|t[ií]tulo\s[IVXLC]|anexo\s|cl[aá]usula\s"
    r"|section\s+\d+|article\s+\d+|chapter\s+[\dIVXLC]|§{1,2}\s*\d)", re.I)
ROMAN_ENUM_RE = re.compile(r"^\s*\(?([ivxlcdm]{1,7}|[a-z])\s*[.)\]]\s+", re.I)
REF_ENTRY_RE = re.compile(
    r"^\s*[A-ZÀ-Ý][\wÀ-ÿ'-]+,\s*[A-ZÀ-Ý]\.")
TABLE_RE = re.compile(r"\|.*\|")
TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]*-{2,}[\s:|-]*\|?\s*$")
TWO_COL_RE = re.compile(r"\S(\t|\s{3,})\S")
KV_RE = re.compile(r"^[^\s:][^:]{0,38}:\s+\S")

ACCENTS = set("áàâãéêíóôõú"
              "üçÁÀÂÃÉÊÍÓÔ"
              "ÕÚÜÇñÑ")


def char_class(text: str) -> dict:
    """Per-position character classes as boolean numpy arrays.

    Classification runs over the *distinct* code points in the document (a few
    dozen) and is then broadcast back with a single fancy-index, instead of
    calling str.isalpha/isdigit/isupper once per character.
    """
    cp = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    uniq, inv = np.unique(cp, return_inverse=True)
    chars = [chr(u) for u in uniq]

    def cls(fn):
        return np.array([fn(c) for c in chars], dtype=bool)[inv]

    L = cls(str.isalpha)
    D = cls(str.isdigit)
    SP = cls(lambda c: c in " \t")
    NL = cls(lambda c: c == "\n")
    PUNC = ~(L | D | SP | NL)
    UPP = cls(str.isupper) & L
    TERM = cls(lambda c: c in ".!?;:")
    EOSC = cls(lambda c: c in ".!?")
    QUOTE = cls(lambda c: c in "\"'")
    LBRACK = cls(lambda c: c in "([{«")
    RBRACK = cls(lambda c: c in ")]}»")
    ACC = cls(lambda c: c in ACCENTS)
    return dict(L=L, D=D, SP=SP, NL=NL, PUNC=PUNC, UPP=UPP, TERM=TERM,
                EOSC=EOSC, QUOTE=QUOTE, LBRACK=LBRACK, RBRACK=RBRACK, ACC=ACC)


def _shift(arr, k, fill=False):
    if k == 0:
        return arr
    out = np.full_like(arr, fill)
    if k > 0:
        out[k:] = arr[:-k]
    else:
        out[:k] = arr[-k:]
    return out


def _shape(s, k=10):
    """Coarse layout signature of a line: `- **Data:**` -> `-__**Aaaa:`."""
    out = []
    for ch in s[:k]:
        if ch.isupper():
            out.append("A")
        elif ch.islower():
            out.append("a")
        elif ch.isdigit():
            out.append("9")
        elif ch in " \t":
            out.append("_")
        else:
            out.append(ch)
    return "".join(out)


def _first_word(s):
    m = re.match(r"\s*(\S+)", s)
    return m.group(1) if m else ""


def _frac(num, den):
    return num / den if den else 0.0


def extract_features(text: str, normalize=True) -> np.ndarray:
    """Return (n, F) float32 feature matrix, one row per character."""
    if normalize:
        text = normalize_text(text)
    n = len(text)
    cols = []
    names = []

    def add(name, arr):
        cols.append(np.asarray(arr, dtype=np.float32))
        names.append(name)

    C = char_class(text)
    L, D, SP, NL = C["L"], C["D"], C["SP"], C["NL"]
    PUNC, UPP = C["PUNC"], C["UPP"]
    TERM, EOSC, QUOTE = C["TERM"], C["EOSC"], C["QUOTE"]
    LBRACK, RBRACK, ACC = C["LBRACK"], C["RBRACK"], C["ACC"]

    # ---- raw char classes (local) ----
    add("L", L); add("D", D); add("SP", SP); add("NL", NL)
    add("PUNC", PUNC); add("UPP", UPP); add("TERM", TERM)
    add("EOSC", EOSC); add("QUOTE", QUOTE); add("LBRACK", LBRACK)
    add("RBRACK", RBRACK); add("ACC", ACC)
    add("prev_L", _shift(L, 1)); add("next_L", _shift(L, -1))
    add("prev_SP", _shift(SP, 1)); add("next_SP", _shift(SP, -1))
    add("prev_NL", _shift(NL, 1)); add("next_NL", _shift(NL, -1))
    add("prev_TERM", _shift(TERM, 1)); add("next_TERM", _shift(TERM, -1))
    add("prev_UPP", _shift(UPP, 1)); add("next_UPP", _shift(UPP, -1))
    add("prev_D", _shift(D, 1)); add("next_D", _shift(D, -1))

    # ---- end-of-sentence detection ----
    next_ns = ~(SP | NL)
    nxt = np.zeros(n, dtype=bool)
    for step in range(1, 6):
        nxt |= _shift(next_ns & _shift(UPP, -step), step)
    eos_here = EOSC & (nxt | _shift(SP, -1) | _shift(NL, -1) |
                       _shift(RBRACK, -1) | _shift(QUOTE, -1))
    add("eos_here", eos_here)
    add("prev_eos", _shift(eos_here, 1))
    add("next_eos", _shift(eos_here, -1))
    add("eos_peak", eos_here & ~_shift(EOSC, -1))

    # ---- newline / paragraph structure ----
    add("nl_end", NL & ~_shift(NL, -1))
    para_break = NL & _shift(NL, -1)
    add("para_start", para_break)
    add("para_end", _shift(para_break, 1))

    # ---- per-line analysis ----
    line_id = np.cumsum(NL)
    line_starts = np.concatenate([[0], np.flatnonzero(NL) + 1])
    n_lines = len(line_starts)
    line_end_idx = np.concatenate([np.flatnonzero(NL), [n]])
    line_lens = (line_end_idx - line_starts).astype(np.float64)

    segs = [text[s:e] for s, e in zip(line_starts, line_end_idx)]
    strips = [s.strip() for s in segs]
    # length features use the STRIPPED length so that trailing whitespace --
    # which CRLF normalization introduces -- cannot change any score
    strip_lens = np.array([len(s) for s in strips], dtype=np.float64)
    shapes = [_shape(s) for s in segs]
    firstw = [_first_word(s) for s in segs]
    blank = np.array([not s for s in strips], dtype=bool)

    B = lambda: np.zeros(n_lines, dtype=bool)      # noqa: E731
    Fl = lambda: np.zeros(n_lines, dtype=np.float32)  # noqa: E731

    (ls_enum, ls_bullet, ls_hash, ls_dash, ls_upper_word, ls_allcaps, ls_title,
     ls_indent, ls_ends_colon, ls_ends_q, ls_has_q, ls_ends_sent, ls_qword,
     ls_date, ls_time, ls_field, ls_short_caps, ls_bold_field) = (B() for _ in range(18))
    (ls_starts_lower, ls_ends_hyphen, ls_ends_comma, ls_table, ls_fence, ls_in_code,
     ls_bquote, ls_setext, ls_hr, ls_page, ls_url, ls_speaker, ls_legal,
     ls_roman, ls_ref, ls_two_col, ls_kv, ls_all_lower, ls_no_alpha) = (B() for _ in range(19))
    (ls_indent_depth, ls_digit_frac, ls_upper_frac, ls_punct_frac,
     ls_word_count) = (Fl() for _ in range(5))

    in_code = False
    for i in range(n_lines):
        seg, st = segs[i], strips[i]
        if not st:
            ls_in_code[i] = in_code
            continue
        if FENCE_RE.match(seg):
            ls_fence[i] = True
            ls_in_code[i] = True
            in_code = not in_code
            continue
        ls_in_code[i] = in_code

        ls_enum[i] = bool(ENUM_RE.match(seg))
        ls_bullet[i] = bool(BULLET_RE.match(seg))
        ls_hash[i] = bool(HASH_RE.match(seg))
        ls_dash[i] = bool(DASH_RE.match(seg))
        ls_upper_word[i] = bool(CAPSWORD_RE.match(seg))
        ls_title[i] = bool(TITLE_RE.match(seg))
        ls_indent[i] = seg[:1] in " \t"
        ls_ends_colon[i] = bool(ENDS_COLON_RE.search(st))
        ls_ends_q[i] = bool(ENDS_Q_RE.search(st))
        ls_has_q[i] = "?" in st or "¿" in st
        ls_ends_sent[i] = bool(ENDS_SENT_RE.search(st)) or ls_ends_q[i]
        ls_qword[i] = bool(QWORD_RE.match(st))
        ls_field[i] = bool(FIELD_RE.match(st)) and len(st) <= 70
        ls_bold_field[i] = bool(BOLD_FIELD_RE.match(st)) and ":" in st[:44]

        caps = CAPS_TOKEN_RE.findall(seg)
        ls_allcaps[i] = len(caps) >= 2 and _frac(sum(map(len, caps)), len(st)) > 0.55
        words = st.split()
        nw = len(words)
        ls_word_count[i] = min(nw / 40.0, 1.0)
        ls_short_caps[i] = (2 <= len(st) <= 60) and not re.search(r"[.!?;:,]$", st) and \
            _frac(sum(1 for w in words if w[:1].isupper()), nw) >= 0.5

        ls_date[i] = bool(DATE_RE.search(st)) or bool(MONTH_RE.search(st)) or \
            bool(re.search(r"\b(19|20)\d{2}\b", st) and YEAR_CTX_RE.search(st))
        ls_time[i] = bool(TIME_RE.search(st))

        # --- structural, markup-agnostic ---
        ls_starts_lower[i] = st[0].islower()
        ls_ends_hyphen[i] = st.endswith("-")
        ls_ends_comma[i] = st.endswith(",")
        ls_table[i] = st.count("|") >= 2 or bool(TABLE_SEP_RE.match(st))
        ls_bquote[i] = bool(QUOTE_LINE_RE.match(seg))
        ls_setext[i] = bool(SETEXT_RE.match(seg))
        ls_hr[i] = bool(HR_RE.match(seg))
        ls_page[i] = bool(PAGE_NUM_RE.match(st)) and len(st) <= 24
        ls_url[i] = bool(URL_RE.search(st))
        ls_speaker[i] = bool(SPEAKER_RE.match(seg)) and len(st) > 4
        ls_legal[i] = bool(LEGAL_RE.match(st))
        ls_roman[i] = bool(ROMAN_ENUM_RE.match(seg))
        ls_ref[i] = bool(REF_ENTRY_RE.match(st))
        ls_two_col[i] = bool(TWO_COL_RE.search(st))
        ls_kv[i] = bool(KV_RE.match(st))
        ls_indent_depth[i] = min((len(seg) - len(seg.lstrip(" \t"))) / 8.0, 1.0)

    # per-line character-class fractions, summed with reduceat over line ranges
    if n_lines and n:
        edges = np.clip(line_starts, 0, max(n - 1, 0))
        def _line_sum(mask):
            tot = np.add.reduceat(mask.astype(np.int32), edges)
            return tot.astype(np.float64)
        n_alpha = _line_sum(L)
        n_upper = _line_sum(UPP)
        n_digit = _line_sum(D)
        n_punct = _line_sum(PUNC)
        denom = np.maximum(strip_lens, 1.0)
        ls_digit_frac = (n_digit / denom).astype(np.float32)
        ls_upper_frac = (n_upper / np.maximum(n_alpha, 1.0)).astype(np.float32)
        ls_punct_frac = (n_punct / denom).astype(np.float32)
        ls_all_lower = (n_alpha > 0) & (n_upper == 0)
        ls_no_alpha = n_alpha == 0

    # --- neighbour relations over NON-BLANK lines ---
    nonblank = np.flatnonzero(~blank)
    prev_nb = np.full(n_lines, -1, dtype=np.int64)
    next_nb = np.full(n_lines, -1, dtype=np.int64)
    last = -1
    for i in range(n_lines):
        prev_nb[i] = last
        if not blank[i]:
            last = i
    last = -1
    for i in range(n_lines - 1, -1, -1):
        next_nb[i] = last
        if not blank[i]:
            last = i

    ls_prev_ends_sent = B(); ls_prev_ends_open = B(); ls_next_starts_lower = B()
    ls_rep_prev = B(); ls_rep_next = B(); ls_same_fw_prev = B()
    ls_has_setext_under = B()
    ls_len_delta_prev = Fl(); ls_len_delta_next = Fl()
    ls_blank_run = Fl()
    for i in range(n_lines):
        p, q = prev_nb[i], next_nb[i]
        if p >= 0:
            ls_prev_ends_sent[i] = ls_ends_sent[p]
            tail = strips[p][-1:] if strips[p] else ""
            ls_prev_ends_open[i] = bool(tail) and (tail.islower() or tail in ",-;")
            ls_rep_prev[i] = shapes[i][:6] == shapes[p][:6] and bool(strips[i])
            ls_same_fw_prev[i] = bool(firstw[i]) and firstw[i] == firstw[p]
            ls_len_delta_prev[i] = np.clip((strip_lens[i] - strip_lens[p]) / 80.0, -1, 1)
        if q >= 0:
            ls_next_starts_lower[i] = ls_starts_lower[q]
            ls_rep_next[i] = shapes[i][:6] == shapes[q][:6] and bool(strips[i])
            ls_len_delta_next[i] = np.clip((strip_lens[i] - strip_lens[q]) / 80.0, -1, 1)
            ls_has_setext_under[i] = ls_setext[q] and q == i + 1 and not blank[i]
        run = 0
        j = i - 1
        while j >= 0 and blank[j]:
            run += 1
            j -= 1
        ls_blank_run[i] = min(run / 3.0, 1.0)

    ls_after_blank = np.zeros(n_lines, dtype=bool)
    ls_after_blank[1:] = blank[:-1]

    # --- document-level calibration ---
    nb_lens = strip_lens[nonblank] if len(nonblank) else np.array([0.0])
    mean_len = float(nb_lens.mean()) if len(nb_lens) else 0.0
    ls_len_ratio = np.clip(strip_lens / max(mean_len, 1.0) / 2.0, 0, 1).astype(np.float32)
    ls_idx_frac = (np.arange(n_lines) / max(n_lines - 1, 1)).astype(np.float32)
    doc_n_lines = min(np.log1p(n_lines) / 8.0, 1.0)
    doc_blank_frac = _frac(int(blank.sum()), n_lines)
    doc_mean_len = min(mean_len / 120.0, 1.0)

    li = np.minimum(line_id, n_lines - 1)
    for nm, a in [
        ("line_after_blank", ls_after_blank), ("line_blank", blank),
        ("line_enum", ls_enum), ("line_bullet", ls_bullet), ("line_hash", ls_hash),
        ("line_dash", ls_dash), ("line_upper_word", ls_upper_word),
        ("line_allcaps", ls_allcaps), ("line_title", ls_title),
        ("line_indent", ls_indent), ("line_colon", ls_ends_colon),
        ("line_ends_q", ls_ends_q), ("line_has_q", ls_has_q),
        ("line_qword", ls_qword), ("line_ends_sent", ls_ends_sent),
        ("prev_line_ends_sent", ls_prev_ends_sent), ("line_date", ls_date),
        ("line_time", ls_time), ("line_field", ls_field),
        ("line_short_caps", ls_short_caps), ("line_bold_field", ls_bold_field),
        ("line_starts_lower", ls_starts_lower), ("line_ends_hyphen", ls_ends_hyphen),
        ("line_ends_comma", ls_ends_comma), ("prev_line_ends_open", ls_prev_ends_open),
        ("next_line_starts_lower", ls_next_starts_lower),
        ("line_table", ls_table), ("line_fence", ls_fence), ("line_in_code", ls_in_code),
        ("line_bquote", ls_bquote), ("line_setext", ls_setext),
        ("line_has_setext_under", ls_has_setext_under), ("line_hr", ls_hr),
        ("line_page_num", ls_page), ("line_url", ls_url), ("line_speaker", ls_speaker),
        ("line_legal", ls_legal), ("line_roman_enum", ls_roman), ("line_ref", ls_ref),
        ("line_two_col", ls_two_col), ("line_kv", ls_kv),
        ("line_all_lower", ls_all_lower), ("line_no_alpha", ls_no_alpha),
        ("line_repeat_prev", ls_rep_prev), ("line_repeat_next", ls_rep_next),
        ("line_same_firstword_prev", ls_same_fw_prev),
        ("line_indent_depth", ls_indent_depth), ("line_digit_frac", ls_digit_frac),
        ("line_upper_frac", ls_upper_frac), ("line_punct_frac", ls_punct_frac),
        ("line_word_count", ls_word_count), ("line_len_ratio", ls_len_ratio),
        ("line_len_delta_prev", ls_len_delta_prev),
        ("line_len_delta_next", ls_len_delta_next),
        ("line_blank_run", ls_blank_run), ("line_idx_frac", ls_idx_frac),
    ]:
        add(nm, a[li])

    # ---- position within line ----
    chars_since_nl = np.arange(n) - np.maximum.accumulate(np.where(NL, np.arange(n), 0))
    add("chars_since_nl", np.clip(chars_since_nl / 200.0, 0, 1))
    add("line_len", np.clip(strip_lens[li] / 300.0, 0, 1))
    add("is_line_end", NL | np.concatenate([NL[1:], [True]]) if n else NL)

    # ---- sentence-level structure ----
    is_ws = SP | NL
    fwd_nws = np.where(is_ws, n, np.arange(n))
    next_nws = np.minimum.accumulate(fwd_nws[::-1])[::-1]
    after_break = np.zeros(n, dtype=bool)
    if n:
        after_break[0] = True
    term_pos = np.flatnonzero(eos_here)
    term_pos = term_pos[term_pos + 1 < n]
    if len(term_pos):
        nx = next_nws[term_pos + 1]
        valid = (nx > term_pos) & (nx < n)
        after_break[nx[valid]] = True
    sent_start_idx = np.flatnonzero(after_break)
    sent_id = np.zeros(n, dtype=np.int64)
    sent_id[after_break] = np.arange(after_break.sum())
    sent_id = np.maximum.accumulate(sent_id)
    sent_starts = np.concatenate([sent_start_idx, [n]])
    sent_len = np.diff(sent_starts)
    chars_in_sent = np.arange(n) - sent_starts[np.minimum(sent_id, max(len(sent_starts) - 2, 0))]
    add("chars_since_sent", np.clip(chars_in_sent / 300.0, 0, 1))
    add("sent_len", np.clip(sent_len[np.minimum(sent_id, len(sent_len) - 1)] / 400.0, 0, 1)
        if len(sent_len) else np.zeros(n))
    add("at_sent_start", after_break)

    fwd_terms = np.where(eos_here, np.arange(n), n)
    next_term = np.minimum.accumulate(fwd_terms[::-1])[::-1]
    add("dist_to_term", np.clip((next_term - np.arange(n)) / 200.0, 0, 1))

    # ---- word-level ----
    word_break = SP | NL
    wb = np.where(word_break, np.arange(n), n)
    prev_wb = np.maximum.accumulate(np.where(word_break, np.arange(n), -1))
    next_wb = np.minimum.accumulate(wb[::-1])[::-1]
    add("word_len_cur", np.clip((next_wb - prev_wb) / 30.0, 0, 1))
    add("in_word", ~word_break)

    add("open_paren", (np.cumsum(LBRACK.astype(int)) - np.cumsum(RBRACK.astype(int))) > 0)

    # ---- document position / shape ----
    add("doc_pos", np.arange(n) / max(n, 1))
    add("near_end", np.arange(n) > n - 8)
    add("doc_n_lines", np.full(n, doc_n_lines))
    add("doc_blank_frac", np.full(n, doc_blank_frac))
    add("doc_mean_line_len", np.full(n, doc_mean_len))

    global FEATURE_NAMES
    if FEATURE_NAMES is None:
        FEATURE_NAMES = tuple(names)
    return np.stack(cols, axis=1)


FEATURE_NAMES = None


def feature_schema_hash() -> str:
    """Short digest of the feature list; stored in the weights for a version check."""
    if FEATURE_NAMES is None:
        extract_features("a\nb")
    return hashlib.sha1("|".join(FEATURE_NAMES).encode()).hexdigest()[:12]


def n_features() -> int:
    if FEATURE_NAMES is None:
        extract_features("a\nb")
    return len(FEATURE_NAMES)


if __name__ == "__main__":
    t = "Hello world. This is a test.\n\nSecond paragraph here.\n\n1. List item one.\n2. Item two."
    X = extract_features(t)
    print("shape:", X.shape, "schema:", feature_schema_hash())
