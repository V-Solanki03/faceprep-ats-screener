"""
FacePrep Campus - ATS Resume Screener
Upload a spreadsheet of Google Drive resume links, paste a job description,
and get every candidate ranked by an ATS match score. 100% free, runs locally.
"""

import os
import re
import json
import time
import tempfile
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st
import gdown
import pdfplumber
import openpyxl
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

RESUME_CACHE_DIR = os.path.join(os.getcwd(), "resume_cache")
os.makedirs(RESUME_CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Spreadsheet loading — reads REAL hyperlinks, not just visible cell text
# ---------------------------------------------------------------------------

def load_spreadsheet(uploaded_file):
    """
    Returns a DataFrame plus a note on whether hyperlinks were resolved.
    xlsx: reads the actual hyperlink target behind each cell (many sheets show
          a filename as text while the real Drive link is hidden underneath —
          plain pandas.read_excel misses this entirely).
    csv:  hyperlinks can't exist in CSV, so cell text is used as-is. If a CSV
          only has filenames with no URLs, resumes can't be located.

    Accepts either a Streamlit UploadedFile OR a plain path string (both
    pandas and openpyxl accept path strings directly) — the latter lets this
    same function load a persisted reference list straight off disk.
    """
    name = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file), False

    wb = openpyxl.load_workbook(uploaded_file)
    ws = wb.active
    raw_header = [c.value for c in ws[1]]

    # A blank header cell becomes a column literally named None, which
    # breaks the frontend table preview (it can't serialize a NaN/None
    # column name to JSON — this was the "Unexpected token 'N'" crash).
    # Duplicate header text also silently collides when building records
    # into a DataFrame (later columns overwrite earlier ones with the same
    # key). Both are fixed here, once, before anything downstream sees them.
    seen_names = {}
    header = []
    for i, h in enumerate(raw_header):
        name = str(h).strip() if h is not None and str(h).strip() != "" else f"Column_{i + 1}"
        if name in seen_names:
            seen_names[name] += 1
            name = f"{name}_{seen_names[name]}"
        else:
            seen_names[name] = 0
        header.append(name)

    data = []
    for row in ws.iter_rows(min_row=2):
        record = {}
        for col_idx, cell in enumerate(row):
            col_name = header[col_idx] if col_idx < len(header) else f"Column_{col_idx + 1}"
            # Prefer the real hyperlink target if the cell has one; otherwise
            # fall back to whatever text/value is in the cell.
            if cell.hyperlink and cell.hyperlink.target:
                record[col_name] = cell.hyperlink.target
            else:
                record[col_name] = cell.value
        if any(v is not None for v in record.values()):
            data.append(record)

    return pd.DataFrame(data), True


def normalize_name(name) -> str:
    """
    Lowercase, collapse whitespace, strip punctuation — a reasonable match
    key for comparing a name across two independently-maintained
    spreadsheets. This is EXACT matching on the normalized string, not fuzzy
    matching: "Sivaraj P" and "Sivaraj Perumal" won't match, and a typo in
    either sheet won't match. Good enough for two lists that were both
    typed/copied from similar sources; if match rates look low, it's usually
    a real formatting mismatch worth checking (middle names, initials,
    extra spaces) rather than a bug.
    """
    if name is None:
        return ""
    s = str(name).strip().lower()
    s = re.sub(r"[.\-_]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def load_reference_name_set(uploaded_file, name_col: str) -> set:
    """Load a reference spreadsheet (placed candidates, top profiles, etc.)
    and return the set of normalized names it contains."""
    df, _ = load_spreadsheet(uploaded_file)
    if name_col not in df.columns:
        return set()
    return {normalize_name(n) for n in df[name_col] if normalize_name(n)}


# ---------------------------------------------------------------------------
# Persisted reference lists — uploaded once, reused automatically on every
# future run without re-uploading. Stored as actual files on disk (not just
# in Streamlit session state, which clears on restart) plus a small JSON
# sidecar remembering which column holds the name in each file.
# ---------------------------------------------------------------------------

REFERENCE_DIR = os.path.join(os.getcwd(), "reference_lists")
os.makedirs(REFERENCE_DIR, exist_ok=True)
REFERENCE_CONFIG_PATH = os.path.join(REFERENCE_DIR, "config.json")


def _load_reference_config() -> dict:
    if os.path.exists(REFERENCE_CONFIG_PATH):
        try:
            with open(REFERENCE_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_reference_config(cfg: dict):
    with open(REFERENCE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f)


def save_reference_list(kind: str, uploaded_file, name_col: str):
    """Persist an uploaded reference spreadsheet to disk and remember its
    name column, so future runs use it automatically with no re-upload."""
    ext = os.path.splitext(uploaded_file.name)[1] or ".xlsx"
    dest_path = os.path.join(REFERENCE_DIR, f"{kind}{ext}")
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    cfg = _load_reference_config()
    cfg[kind] = {"path": dest_path, "name_col": name_col, "original_filename": uploaded_file.name}
    _save_reference_config(cfg)


def remove_reference_list(kind: str):
    cfg = _load_reference_config()
    entry = cfg.pop(kind, None)
    if entry and os.path.exists(entry["path"]):
        os.remove(entry["path"])
    _save_reference_config(cfg)


def get_persisted_reference(kind: str):
    """Returns (path, name_col, original_filename) or None if nothing saved."""
    cfg = _load_reference_config()
    entry = cfg.get(kind)
    if not entry or not os.path.exists(entry["path"]):
        return None
    return entry["path"], entry["name_col"], entry["original_filename"]


# ---------------------------------------------------------------------------

def classify_link(url):
    """Returns (kind, file_id). kind in: file, gdoc, folder, none, bare_filename, external_pdf, other."""
    if not isinstance(url, str) or not url.strip():
        return "none", None
    if "/folders/" in url:
        m = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
        return "folder", (m.group(1) if m else None)
    if "docs.google.com/document" in url:
        m = re.search(r"/document/d/([a-zA-Z0-9_-]+)", url)
        return "gdoc", (m.group(1) if m else None)
    for p in [r"/file/d/([a-zA-Z0-9_-]+)", r"[?&]id=([a-zA-Z0-9_-]+)", r"/d/([a-zA-Z0-9_-]+)"]:
        m = re.search(p, url)
        if m:
            return "file", m.group(1)
    is_url = url.lower().startswith(("http://", "https://"))
    if url.lower().endswith((".pdf", ".doc", ".docx")):
        if is_url:
            return "external_pdf", url
        # Looks like a plain filename with no link at all — a strong sign the
        # real hyperlink existed in the original file but didn't survive
        # (classic case: a Google Sheet with hyperlinked filenames gets
        # exported to CSV, which can't carry hyperlinks at all).
        return "bare_filename", url
    return "other", None


def summarize_download_error(e) -> str:
    """
    gdown's exceptions are often multi-line FAQ dumps (permission errors,
    quota errors) that are technically informative but make the results
    table unreadable — a single failed row can be 10+ lines of text. This
    recognizes the common cases and returns one clean sentence instead;
    anything unrecognized falls back to just the first line, truncated.
    """
    msg = str(e)
    lower = msg.lower()
    if "permission" in lower and "anyone with the link" in lower:
        return "File is private — not shared as 'Anyone with the link can view'"
    if "too many users have viewed or downloaded" in lower or "quota" in lower:
        return "Google Drive download quota exceeded for this file — try again later"
    if "timed out" in lower or "timeout" in lower:
        return "Network timeout while downloading — often transient, may succeed on retry"
    if "cannot retrieve the public link" in lower:
        return "Couldn't access this file — check it's shared as 'Anyone with the link can view'"
    first_line = msg.strip().splitlines()[0] if msg.strip() else "Unknown error"
    return first_line[:150]


def download_pdf(kind: str, file_id: str, out_dir: str):
    """Download/export a single resume as PDF text-extractable bytes on disk.
    out_dir is a PERSISTENT cache directory (not a temp dir) — if a valid
    PDF for this file_id is already there from a previous run, this returns
    it immediately with no network call at all. Massive speedup on re-runs
    of the same spreadsheet, which is the common case (tweak JD, re-score)."""
    if kind == "file":
        out_path = os.path.join(out_dir, f"{file_id}.pdf")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            with open(out_path, "rb") as f:
                head = f.read(20)
            if head.startswith(b"%PDF"):
                return out_path, None  # cache hit, no download needed
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            gdown.download(url=url, output=out_path, quiet=True, use_cookies=False)
        except Exception as e:  # noqa: BLE001
            return None, f"Download error: {summarize_download_error(e)}"
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            # Guard against Google returning an HTML block/rate-limit page
            # instead of the actual PDF (same size range, wrong content).
            with open(out_path, "rb") as f:
                head = f.read(20)
            if head.startswith(b"%PDF"):
                return out_path, None
            return None, "Download blocked or rate-limited by Google Drive — will retry"
        return None, "Download failed — file may be private or not shared as 'Anyone with the link'"

    if kind == "gdoc":
        out_path = os.path.join(out_dir, f"{file_id}.pdf")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
            return out_path, None  # cache hit
        url = f"https://docs.google.com/document/d/{file_id}/export?format=pdf"
        try:
            gdown.download(url=url, output=out_path, quiet=True, use_cookies=False)
        except Exception as e:  # noqa: BLE001
            return None, f"Download error: {summarize_download_error(e)}"
        if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
            return out_path, None
        return None, "Could not export Google Doc — check sharing settings"

    if kind == "external_pdf":
        out_path = os.path.join(out_dir, re.sub(r"[^a-zA-Z0-9_-]", "_", file_id)[:80] + ".pdf")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            with open(out_path, "rb") as f:
                head = f.read(20)
            if head.startswith(b"%PDF"):
                return out_path, None  # cache hit
        try:
            resp = requests.get(file_id, timeout=30)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(resp.content)
        except Exception as e:  # noqa: BLE001
            return None, f"Download error: {summarize_download_error(e)}"
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            with open(out_path, "rb") as f:
                head = f.read(20)
            if head.startswith(b"%PDF"):
                return out_path, None
            return None, "That link didn't return an actual PDF file"
        return None, "Download failed — link may not be a direct, publicly-accessible file URL"

    if kind == "bare_filename":
        return None, ("This cell only has a filename, no actual link (e.g. \"resume.pdf\") — "
                       "the real Drive link likely existed in the original file but was lost. "
                       "Common cause: a Google Sheet with hyperlinked filenames exported to CSV "
                       "(CSV can't store hyperlinks at all). Re-upload the original .xlsx instead.")

    if kind == "folder":
        return None, "Link points to a Drive FOLDER, not a single file — rejected as noise"

    if kind == "none":
        return None, "No resume link found in this cell"

    return None, "Unrecognized link format"


def extract_jd_from_file(uploaded_file) -> str:
    """Extract text from an uploaded JD file — PDF or DOCX."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".pdf"):
            text = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
            return text
        if name.endswith(".docx"):
            document = docx.Document(uploaded_file)
            return "\n".join(p.text for p in document.paragraphs)
    except Exception as e:  # noqa: BLE001
        st.error(f"Couldn't read the JD file: {e}")
        return ""
    return ""


def _group_words_into_lines(words, y_tolerance=3):
    """Group words sharing roughly the same vertical position into text lines,
    left-to-right within each line — reconstructs readable line order instead
    of pdfplumber's raw word stream order."""
    if not words:
        return ""
    lines, current_line, current_top = [], [], None
    for w in sorted(words, key=lambda w: w["top"]):
        if current_top is None or abs(w["top"] - current_top) <= y_tolerance:
            current_line.append(w)
            current_top = w["top"] if current_top is None else current_top
        else:
            lines.append(current_line)
            current_line = [w]
            current_top = w["top"]
    if current_line:
        lines.append(current_line)
    return "\n".join(" ".join(w["text"] for w in sorted(line, key=lambda w: w["x0"]))
                      for line in lines)


def _detect_column_split(words, page_width):
    """
    Look for a genuine vertical gutter (empty band with nothing crossing it)
    splitting the page into two real columns — common in sidebar-style resume
    templates (skills column + main content column). Returns the split x
    position, or None if this looks like a normal single-column resume.
    """
    for frac in (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65):
        split = page_width * frac
        if any(w["x0"] < split < w["x1"] for w in words):
            continue  # something straddles this line — not a real gutter here
        left = sum(1 for w in words if w["x1"] <= split)
        right = sum(1 for w in words if w["x0"] >= split)
        total = left + right
        if total and min(left, right) / total >= 0.15:
            return split
    return None


def extract_text_via_ocr(pdf_path: str) -> str:
    """
    Fallback for scanned/image-based PDFs that have no selectable text layer
    at all — renders each page to an image and runs OCR on it. Slower than
    normal text extraction, only used when normal extraction comes back
    empty. Requires the pytesseract + pdf2image packages AND the underlying
    Tesseract OCR engine + poppler installed on the machine (see README) —
    if either isn't set up, this silently returns "" and the resume falls
    back to being marked as needing manual review, same as before.
    """
    if not OCR_AVAILABLE:
        return ""
    try:
        images = convert_from_path(pdf_path, dpi=200)
    except Exception:  # noqa: BLE001
        return ""  # poppler likely not installed/on PATH
    text_parts = []
    for img in images:
        try:
            t = pytesseract.image_to_string(img)
        except Exception:  # noqa: BLE001
            return ""  # tesseract binary likely not installed/on PATH
        if t and t.strip():
            text_parts.append(t)
    return "\n\n".join(text_parts)


def extract_text(pdf_path: str) -> str:
    """
    Extract resume text with column-awareness. Many resume templates use a
    sidebar (skills/contact) next to a main content column — naively reading
    left-to-right across the whole page interleaves the two, scrambling both
    the readable output AND the words used for scoring (e.g. two column
    headers on the same line get glued together with no space). When a real
    column gutter is detected, each column is extracted separately, top to
    bottom, then concatenated — producing coherent, correctly-ordered text.

    If normal extraction comes back with little to no text (typically a
    scanned/photographed resume with no real text layer, just an image),
    falls back to OCR rather than giving up — this is what previously
    accounted for a large share of "Failed" resumes that never got scored.
    """
    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                words = page.extract_words(keep_blank_chars=False, use_text_flow=False)
                if not words:
                    continue
                split = _detect_column_split(words, page.width)
                if split:
                    left_words = [w for w in words if w["x1"] <= split]
                    right_words = [w for w in words if w["x0"] >= split]
                    text_parts.append(_group_words_into_lines(left_words))
                    text_parts.append(_group_words_into_lines(right_words))
                else:
                    # layout=True preserves inter-word spacing better than the
                    # default, reducing words getting glued together
                    t = page.extract_text(layout=True) or page.extract_text() or ""
                    text_parts.append(t)
    except Exception:  # noqa: BLE001
        text_parts = []

    text = "\n\n".join(tp for tp in text_parts if tp)
    if len(text.strip()) < 40:
        ocr_text = extract_text_via_ocr(pdf_path)
        if len(ocr_text.strip()) > len(text.strip()):
            return ocr_text
    return text


def detect_keyword_stuffing(resume_text: str, skills: list):
    """
    Heuristic check for keyword stuffing — a resume that just lists/repeats
    skill words without real surrounding content. Not proof of dishonesty,
    just a signal that this candidate's high score deserves a closer human
    look rather than being trusted at face value.

    Flags on:
    1. Any single required skill term appearing unusually often relative to
       the resume's total length (repetition well beyond normal mention).
    2. Unusually low overall vocabulary diversity in a longer document —
       a sign of the same handful of terms being repeated throughout rather
       than genuine varied content (projects, experience, context).
    """
    words = re.findall(r"\b\w+\b", resume_text.lower())
    total = len(words)
    if total < 20:
        return False, []

    reasons = []
    for skill in skills:
        skill = skill.strip()
        if not skill:
            continue
        count = len(re.findall(r"\b" + re.escape(skill.lower()) + r"\b", resume_text.lower()))
        # more than ~1 mention per 40 words of the whole resume is unusually dense
        if count >= 6 and count / total > 0.025:
            reasons.append(f'"{skill}" repeated {count}x')

    if total > 150:
        unique_ratio = len(set(words)) / total
        if unique_ratio < 0.30:
            reasons.append(f"low vocabulary diversity ({unique_ratio:.0%} unique words)")

    return (len(reasons) > 0), reasons


def keyword_score(resume_text: str, skills: list):
    resume_lower = resume_text.lower()
    matched, missing = [], []
    for skill in skills:
        skill = skill.strip()
        if not skill:
            continue
        if re.search(r"\b" + re.escape(skill.lower()) + r"\b", resume_lower):
            matched.append(skill)
        else:
            missing.append(skill)
    pct = (len(matched) / len(skills) * 100) if skills else 0
    return pct, matched, missing


def compute_similarity_scores(jd_text: str, resume_texts: list):
    """
    TF-IDF + cosine similarity between the JD and every resume, converted to
    a 0-100 score with a FIXED smooth curve — not rescaled relative to this
    batch at all.

    Earlier versions rescaled scores relative to whoever else was in the same
    batch (min-max against the single top score, then against the 90th
    percentile). Both approaches have the same root problem: the "100" mark
    is defined by other candidates in the batch, not by the resume itself.
    With a small batch, several resumes can end up above that relative
    ceiling and all get hard-capped at the identical value 100.0 — which
    looks like a tie in quality, but is actually just several different raw
    similarity scores getting flattened by the same clip.

    This version applies score = 100 * (1 - e^(-k * similarity)) directly to
    each resume's own raw similarity — a smooth curve that asymptotically
    approaches 100 without ever hard-clipping, so two resumes only land on
    the same score if their actual content-similarity to the JD is genuinely
    almost identical, not because of where the batch's cutoff happened to
    fall. k is calibrated so a moderate match (~0.15 similarity) scores
    around 50 and a strong match (~0.35) scores around 80 — adjust K_CURVE
    below to make scores generally stricter or more generous.

    Caveat worth knowing: the TF-IDF vocabulary is rebuilt fresh from each
    batch's own text, so raw similarity magnitudes aren't perfectly
    comparable run-to-run either — treat scores as a within-run relative
    ranking + rough quality signal, not an exact, portable percentage.

    sublinear_tf dampens the reward for simply repeating a word many times
    (uses 1 + log(count) instead of raw count) — this alone meaningfully
    reduces the payoff from crude keyword-stuffing, though it can't fully
    eliminate it. See detect_keyword_stuffing() for an explicit flag on top.
    """
    K_CURVE = 4.6  # tune: higher = stricter (harder to reach high scores)

    corpus = [jd_text] + resume_texts
    vectorizer = TfidfVectorizer(stop_words="english", max_features=8000, sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return [0.0] * len(resume_texts)
    jd_vec = matrix[0:1]
    resume_vecs = matrix[1:]
    sims = cosine_similarity(jd_vec, resume_vecs)[0]

    scaled = []
    for s, t in zip(sims, resume_texts):
        if not t.strip():
            scaled.append(0.0)
        else:
            score = 100 * (1 - pow(2.718281828, -K_CURVE * max(0.0, s)))
            scaled.append(round(score, 1))
    return scaled


def process_one_resume(idx: int, name, link, skills: list, tmp_dir: str):
    """Download, extract, and keyword-score a single resume. Runs in a worker thread."""
    kind, file_id = classify_link(str(link) if link is not None else "")
    pdf_path, err = download_pdf(kind, file_id, tmp_dir)

    if err:
        return idx, dict(Name=name, text="", kw_pct=0, matched=[], missing=skills,
                          Status=f"Failed — {err}", Link=link, file_id=file_id, kind=kind,
                          stuffing_flag=False, stuffing_reasons=[])

    text = extract_text(pdf_path)
    try:
        os.remove(pdf_path)
    except OSError:
        pass

    if not text.strip():
        return idx, dict(Name=name, text="", kw_pct=0, matched=[], missing=skills,
                          Status="Failed — no extractable text (likely a scanned image PDF)",
                          Link=link, file_id=file_id, kind=kind,
                          stuffing_flag=False, stuffing_reasons=[])

    kw_pct, matched, missing = keyword_score(text, skills)
    stuffing_flag, stuffing_reasons = detect_keyword_stuffing(text, skills)
    return idx, dict(Name=name, text=text, kw_pct=kw_pct, matched=matched,
                      missing=missing, Status="OK", Link=link, file_id=file_id, kind=kind,
                      stuffing_flag=stuffing_flag, stuffing_reasons=stuffing_reasons)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="FacePrep ATS", layout="wide")
st.title("📋 FacePrep Campus — ATS Resume Screener")
st.caption(
    "Upload your spreadsheet of Google Drive resume links, paste a job description, "
    "and click once. It downloads and reads every resume, then scores and ranks "
    "everyone — best to worst. Download the ranked results as CSV straight from "
    "the app when it's done."
)

SKILL_STOPWORDS = {
    "and", "or", "the", "a", "an", "with", "of", "to", "in", "for", "on",
    "strong", "good", "excellent", "solid", "basic", "ability", "abilities",
    "experience", "knowledge", "understanding", "skills", "skill",
    "proficiency", "familiarity", "working", "hands", "hands-on",
}

# Fragments that START with one of these words are almost always a leftover
# piece of a longer natural-language sentence that got shredded by comma-
# splitting (e.g. "Excellent verbal, written communication skills" splits
# into "Excellent verbal" + "written communication skills" — the first
# piece is meaningless on its own). Dropping fragments led by these words
# removes most of that noise without needing real NLP.
FRAGMENT_LEAD_FILTER = {
    "or", "and", "similar", "such", "etc", "excellent", "strong", "good",
    "solid", "basic", "ability", "willingness", "preferred", "required",
}


def auto_extract_skills(jd_text: str, max_skills: int = 15) -> list:
    """
    Heuristically pull a required-skills list straight out of the JD text,
    so the user doesn't have to retype what's already written.

    Strategy:
    1. Find a "Requirements / Skills / Qualifications" section if one exists
       and extract short bullet-style lines from it.
    2. Within those lines, split on commas/slashes to get individual skills.
    3. Fall back to scanning the whole JD for short, capitalized/acronym-like
       tokens (AWS, SQL, React) if no clear section is found.

    Caveat: this is a heuristic over natural-language prose, not real parsing.
    JD bullets that pack multiple ideas into one comma-separated sentence
    (common in casually-written JDs) will still produce some imperfect
    fragments — the filters below catch the most common failure patterns,
    not all of them. Always spot-check the result before running a batch.
    """
    section_headers = r"(requirements?|required skills?|technical skills?|qualifications?|must[\s-]have|tools)"
    lines = jd_text.splitlines()
    in_section = False
    candidate_lines = []

    for line in lines:
        stripped = line.strip()
        if re.match(rf"^\W*{section_headers}\W*$", stripped, re.IGNORECASE):
            in_section = True
            continue
        if in_section:
            if not stripped:
                continue
            # A short capitalized line is almost always a (sub-)heading, not
            # a requirement itself — e.g. "Preferred Qualifications",
            # "Technical Skills". Either way it should never be appended as
            # a candidate skill; the only question is whether to keep
            # collecting afterward (it's a same-topic heading) or stop (it's
            # an unrelated section starting, e.g. "About the Company").
            if re.match(r"^[A-Z][A-Za-z /&]{2,30}$", stripped) and len(stripped.split()) <= 4 \
                    and not re.match(r"^[•\-\*]", stripped):
                if not re.search(section_headers, stripped, re.IGNORECASE):
                    in_section = False
                continue  # never treat a header line itself as a skill
            candidate_lines.append(stripped.lstrip("•-* \t"))

    skills = []
    if candidate_lines:
        for line in candidate_lines:
            parts = re.split(r",|/| and | or | & ", line)
            for p in parts:
                p = p.strip(" .;:")
                word_count = len(p.split())
                if not p or not (1 <= word_count <= 4):
                    continue
                if p.lower() in SKILL_STOPWORDS:
                    continue
                first_word = p.split()[0].lower()
                if first_word in FRAGMENT_LEAD_FILTER:
                    continue
                skills.append(p)
    else:
        # fallback: acronyms and CamelCase/Title-Case tech-looking tokens anywhere in the JD
        tokens = re.findall(r"\b[A-Z][A-Za-z0-9+.#]{1,15}\b", jd_text)
        seen = set()
        for t in tokens:
            if t.lower() in SKILL_STOPWORDS or t in seen:
                continue
            seen.add(t)
            skills.append(t)

    # de-duplicate while preserving order, cap the list
    final, seen = [], set()
    for s in skills:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            final.append(s)
    return final[:max_skills]


with st.sidebar:
    st.header("1. Spreadsheet")
    file = st.file_uploader("Excel or CSV with names + Drive links", type=["xlsx", "csv"])

    st.header("2. Job Description")
    jd_file = st.file_uploader("Upload JD (PDF or DOCX)", type=["pdf", "docx"], key="jd_file")

    if "jd_text" not in st.session_state:
        st.session_state.jd_text = ""
    if jd_file is not None and st.session_state.get("_last_jd_file") != jd_file.name:
        st.session_state.jd_text = extract_jd_from_file(jd_file)
        st.session_state._last_jd_file = jd_file.name

    jd_text = st.text_area(
        "Job description text (auto-filled if you upload above — edit freely either way)",
        key="jd_text", height=180,
    )
    st.header("3. Speed")
    max_workers = st.slider(
        "Parallel downloads", 2, 8, 4,
        help="Google Drive rate-limits anonymous link downloads — going much "
             "above 4-6 tends to cause a burst of 'Failed' results rather than "
             "actually finishing faster. Failed downloads are retried "
             "automatically with lower concurrency.",
    )
    cached_count = len([f for f in os.listdir(RESUME_CACHE_DIR) if f.endswith(".pdf")]) \
        if os.path.isdir(RESUME_CACHE_DIR) else 0
    st.caption(
        f"📦 {cached_count} resume(s) cached locally from previous runs — "
        f"re-running the same spreadsheet skips re-downloading those, so "
        f"only new/changed resumes hit the network."
    )
    if cached_count and st.button("🗑️ Clear resume cache"):
        for f in os.listdir(RESUME_CACHE_DIR):
            if f.endswith(".pdf"):
                os.remove(os.path.join(RESUME_CACHE_DIR, f))
        st.success("Cache cleared. Next run re-downloads everything.")

    st.header("4. Reference Lists (optional)")
    st.caption(
        "Upload once — saved locally and reused automatically on every future "
        "run, no re-uploading. Matching is by name (case/spacing-insensitive, "
        "not fuzzy) — a real spelling difference between sheets won't match."
    )

    def _render_reference_slot(kind: str, label: str):
        """One upload-once-then-remember slot for a reference list. Returns
        (path_or_none, name_col_or_none) to use for this run's scoring."""
        persisted = get_persisted_reference(kind)
        if persisted:
            path, name_col, original_filename = persisted
            try:
                _df, _ = load_spreadsheet(path)
                row_count = len(_df)
            except Exception:  # noqa: BLE001
                row_count = "?"
            st.markdown(f"**{label}**")
            st.caption(f"✅ Using saved list: **{original_filename}** "
                       f"({name_col} column, {row_count} names)")
            col_a, col_b = st.columns([2, 1])
            with col_a:
                replacement = st.file_uploader(
                    f"Replace {label.lower()}", type=["xlsx", "csv"],
                    key=f"{kind}_replace", label_visibility="collapsed",
                )
            with col_b:
                if st.button(f"🗑️ Remove", key=f"{kind}_remove", use_container_width=True):
                    remove_reference_list(kind)
                    st.rerun()
            if replacement is not None:
                _new_df, _ = load_spreadsheet(replacement)
                new_name_col = st.selectbox(
                    "Name column in the replacement file", _new_df.columns, key=f"{kind}_replace_namecol",
                    index=next((i for i, c in enumerate(_new_df.columns) if "name" in str(c).lower()), 0),
                )
                if st.button(f"💾 Save replacement", key=f"{kind}_save_replace"):
                    save_reference_list(kind, replacement, new_name_col)
                    st.rerun()
            return path, name_col
        else:
            upload = st.file_uploader(label, type=["xlsx", "csv"], key=f"{kind}_upload")
            if upload is not None:
                _df, _ = load_spreadsheet(upload)
                name_col = st.selectbox(
                    "Name column in that sheet", _df.columns, key=f"{kind}_namecol",
                    index=next((i for i, c in enumerate(_df.columns) if "name" in str(c).lower()), 0),
                )
                if st.button(f"💾 Save (used automatically from now on)", key=f"{kind}_save"):
                    save_reference_list(kind, upload, name_col)
                    st.rerun()
            return None, None

    placed_path, placed_name_col = _render_reference_slot("placed", "Already-placed candidates")
    st.divider()
    top_profiles_path, top_profiles_name_col = _render_reference_slot("top_profiles", "Top-profile candidates")

    run_button = st.button("🚀 Run ATS Screening", type="primary", use_container_width=True)

if file is not None:
    df, hyperlinks_resolved = load_spreadsheet(file)
    if hyperlinks_resolved:
        st.caption("✅ Reading actual hyperlinks behind each cell (not just visible text).")
    else:
        st.caption("⚠️ CSV uploaded — hyperlinks aren't possible in CSV. "
                   "Make sure the Drive URL is the literal cell text, or upload the original .xlsx instead.")

    st.subheader("Preview")
    st.dataframe(df.head(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        name_col = st.selectbox("Column with candidate names", df.columns)
    with col2:
        # try to default to a column that actually looks like it has links
        link_guess = next((c for c in df.columns if "resume" in str(c).lower()
                            or "cv" in str(c).lower() or "link" in str(c).lower()), df.columns[0])
        link_col = st.selectbox("Column with Drive links", df.columns,
                                 index=list(df.columns).index(link_guess))

    if run_button:
        if not jd_text.strip():
            st.error("Please paste a job description first.")
        else:
            # Skills are extracted automatically from the JD purely to show
            # Matched/Missing Skills as context on each candidate — they do
            # NOT factor into the ATS Score itself, which is pure JD-to-resume
            # content similarity. No manual skill list, no weight slider.
            skills = auto_extract_skills(jd_text)

            total = len(df)
            progress = st.progress(0)
            status = st.empty()

            rows = [None] * total
            cache_dir = RESUME_CACHE_DIR
            tasks = [(i, row[name_col], row[link_col]) for i, row in df.iterrows()]
            completed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_one_resume, i, name, link, skills, cache_dir): i
                    for i, name, link in tasks
                }
                for future in as_completed(futures):
                    idx, result = future.result()
                    rows[idx] = result
                    completed += 1
                    status.text(f"Processed {completed}/{total}: {result['Name']}")
                    progress.progress(completed / total)

            # Retry pass: download/network failures are often transient
            # (rate limits, momentary connection issues) — retry once, with
            # LIMITED concurrency (not full speed, not fully sequential) so
            # a large batch of failures doesn't take forever but also
            # doesn't immediately re-trigger the same rate limit. Skip cases
            # that can never succeed no matter how many times we retry:
            # folder links, missing links, unrecognized formats.
            retry_idxs = [i for i, r in enumerate(rows)
                          if r["Status"].startswith("Failed")
                          and r.get("kind") in ("file", "gdoc", "external_pdf")]
            if retry_idxs:
                status.text(f"Retrying {len(retry_idxs)} failed download(s)...")
                time.sleep(3)
                retry_workers = min(3, max_workers, len(retry_idxs))
                retry_done = 0
                with ThreadPoolExecutor(max_workers=retry_workers) as executor:
                    futures = {
                        executor.submit(process_one_resume, i, df.iloc[i][name_col],
                                         df.iloc[i][link_col], skills, cache_dir): i
                        for i in retry_idxs
                    }
                    for future in as_completed(futures):
                        idx, result = future.result()
                        rows[idx] = result
                        retry_done += 1
                        status.text(f"Retrying failed downloads: {retry_done}/{len(retry_idxs)}...")

            status.text("Scoring content similarity...")
            texts_for_sim = [r["text"] if r["text"] else " " for r in rows]
            sim_scores = compute_similarity_scores(jd_text, texts_for_sim)
            for r, sim in zip(rows, sim_scores):
                r["sim_score"] = sim if r["Status"] == "OK" else 0

            placed_names = (load_reference_name_set(placed_path, placed_name_col)
                             if placed_path is not None and placed_name_col else set())
            top_profile_names = (load_reference_name_set(top_profiles_path, top_profiles_name_col)
                                  if top_profiles_path is not None and top_profiles_name_col else set())

            results = []
            for r in rows:
                review_flag = ("⚠️ " + "; ".join(r["stuffing_reasons"])) if r.get("stuffing_flag") else ""
                norm = normalize_name(r["Name"])
                results.append({
                    "Name": r["Name"],
                    "ATS Score": round(r["sim_score"], 1),
                    "Already Placed": "⚠️ Yes" if norm in placed_names else "",
                    "Top Profile": "⭐ Yes" if norm in top_profile_names else "",
                    "Review Flag": review_flag,
                    "Matched Skills": ", ".join(r["matched"]),
                    "Missing Skills": ", ".join(r["missing"]),
                    "Status": r["Status"],
                    "Drive Link": r["Link"],
                    "_file_id": r["file_id"],
                    "_text": r["text"],
                })

            status.text("Done!")
            results_df = pd.DataFrame(results).sort_values("ATS Score", ascending=False).reset_index(drop=True)

            st.session_state["ats_results_df"] = results_df
            st.session_state["ats_placed_matches"] = sum(1 for r in results if r["Already Placed"])
            st.session_state["ats_top_matches"] = sum(1 for r in results if r["Top Profile"])
            st.session_state["ats_total"] = total

    # -----------------------------------------------------------------------
    # Render results from session_state (NOT gated on run_button) so the
    # ranked table, downloads, and candidate preview stay visible across any
    # later interaction on the page -- clicking a download button, expanding
    # a candidate, etc. all trigger a Streamlit rerun, and run_button is only
    # True on the exact click that ran the pipeline.
    # -----------------------------------------------------------------------
    if "ats_results_df" in st.session_state:
        results_df = st.session_state["ats_results_df"]
        total = st.session_state["ats_total"]
        placed_matches = st.session_state.get("ats_placed_matches", 0)
        top_matches = st.session_state.get("ats_top_matches", 0)

        display_cols = ["Name", "ATS Score", "Already Placed", "Top Profile", "Review Flag",
                         "Matched Skills", "Missing Skills", "Status", "Drive Link"]

        flagged_count = (results_df["Review Flag"] != "").sum()
        if flagged_count:
            st.caption(
                f"⚠️ {flagged_count} candidate(s) flagged for possible keyword stuffing — "
                "unusually dense/repetitive skill mentions relative to resume length. "
                "A high score there deserves a manual look before trusting it at face value."
            )
        if placed_matches or top_matches:
            st.caption(
                f"🔎 Cross-checked against reference lists: {placed_matches} already-placed "
                f"match(es), {top_matches} top-profile match(es) found in this batch."
            )

        st.subheader("📊 Ranked Results")
        ok_count = (results_df["Status"] == "OK").sum()
        st.caption(f"{ok_count}/{total} resumes read successfully.")
        def _highlight_flags(row):
            if row.get("Already Placed"):
                return ["background-color: rgba(0, 200, 0, 0.22)"] * len(row)
            if row.get("Top Profile"):
                return ["background-color: rgba(255, 215, 0, 0.22)"] * len(row)
            return [""] * len(row)

        styled = results_df[display_cols].style.apply(_highlight_flags, axis=1)
        st.dataframe(styled, use_container_width=True, height=500)

        csv = results_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download results as CSV", csv, "ats_results.csv", "text/csv",
                            use_container_width=True)

        # -----------------------------------------------------------------
        # Preview candidates: embedded PDF + extracted skill summary
        # -----------------------------------------------------------------
        st.divider()
        st.subheader("🔍 Preview Candidates")
        previewable = results_df[results_df["Status"] == "OK"].reset_index(drop=True)
        if previewable.empty:
            st.warning("No successfully-read resumes to preview.")
        else:
            st.caption(f"Showing all {len(previewable)} successfully-scored candidates, ranked highest first.")
            for idx in range(len(previewable)):
                r = previewable.iloc[idx]
                with st.expander(f"#{idx + 1} — {r['Name']}  ·  ATS Score: {r['ATS Score']}"):
                    left, right = st.columns([1, 1.4])
                    with left:
                        st.markdown(f"**ATS Score:** {r['ATS Score']}%")
                        st.markdown(f"**Matched skills:** {r['Matched Skills'] or '—'}")
                        st.markdown(f"**Missing skills:** {r['Missing Skills'] or '—'}")
                        st.markdown(f"[Open in Google Drive]({r['Drive Link']})")
                        st.text_area(
                            "Extracted resume text (raw)",
                            value=r["_text"][:3000],
                            height=250,
                            key=f"text_{idx}",
                        )
                    with right:
                        if r["_file_id"]:
                            preview_url = f"https://drive.google.com/file/d/{r['_file_id']}/preview"
                            st.components.v1.iframe(preview_url, height=500)
                        else:
                            st.info("No preview available for this file.")

        # -----------------------------------------------------------------
        # Rejected / failed breakdown — so it's obvious WHY, not just a count
        # -----------------------------------------------------------------
        failed_df = results_df[results_df["Status"] != "OK"]
        if not failed_df.empty:
            st.divider()
            st.subheader(f"⚠️ {len(failed_df)} Not Scored — Reasons")
            st.dataframe(failed_df[["Name", "Status", "Drive Link"]],
                         use_container_width=True, height=min(400, 40 + 35 * len(failed_df)))
else:
    st.info("👈 Upload your spreadsheet in the sidebar to get started.")
