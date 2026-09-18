# FacePrep Campus — ATS Resume Screener

A Streamlit app that reads resumes from Google Drive links in a spreadsheet,
scores each one against a job description using content-similarity matching,
and ranks candidates best to worst. Free, no API keys needed.

---

## 1. One-time setup

You need Python installed (3.9 or newer). Check if you already have it:

```
python3 --version
```

If not installed, download it from https://www.python.org/downloads/ (tick
"Add Python to PATH" during install on Windows).

Then, in a terminal / command prompt, navigate to this folder and run:

```
pip install -r requirements.txt
```

## 2. Running the app

From this folder, run:

```
streamlit run ats_app.py
```

A browser tab opens automatically (usually at `http://localhost:8501`).

## 3. Before you upload your spreadsheet

**Drive link sharing:** every resume's sharing setting must be "Anyone with
the link can view." A restricted file shows up as "Failed" in results.

**Spreadsheet format:** any Excel (.xlsx) or CSV, with one column for
candidate names and one for the Drive link to their resume. Hidden
hyperlinks behind a cell (not just visible text) are read automatically for
.xlsx files — CSV can't carry real hyperlinks, so the Drive URL must be the
literal cell text there.

## 4. Using the app

1. **Spreadsheet** — upload it in the sidebar, then pick the name column and
   the resume-link column.
2. **Job description** — paste it as text, or upload a PDF/DOCX and it's
   extracted automatically (editable either way).
3. **Speed** — parallel-download slider (default 4; Google Drive rate-limits
   anonymous downloads, so going much higher tends to cause failures instead
   of finishing faster). Previously-downloaded resumes are cached locally, so
   re-running the same spreadsheet only fetches new/changed resumes.
4. **Reference Lists (optional)** — upload a spreadsheet of already-placed
   candidates and/or one of top-profile candidates, each with its name
   column picked once. These are saved locally and reused automatically on
   every future run — no re-uploading. Matching is by name
   (case/spacing-insensitive, not fuzzy), and flags rather than filters:
   matched rows are never removed, just marked and color-highlighted
   (green = already placed, yellow = top profile) in the results table.
5. Click **Run ATS Screening**.
6. Results show every candidate ranked highest-to-lowest by ATS score, with
   any reference-list flags. Download as CSV to share with the team.

## How the scoring works

- **Content Similarity** — a TF-IDF + cosine similarity comparison between
  the full job description and each resume's extracted text, converted to a
  0–100 score via a fixed saturation curve (not rescaled relative to other
  resumes in the batch, so scores are comparable run to run and never
  produce artificial ties or misleading cliffs).
- A **keyword-stuffing flag** calls out resumes whose vocabulary is unusually
  repetitive relative to length — a sign of scores inflated by repeating
  buzzwords rather than real content.

This is a lexical/statistical matching approach (no paid AI API), so treat
scores as a strong first-pass filter, not a final hiring decision — always
have a human review the top candidates.

## Known limitations

- Scanned/image-only PDF resumes (no selectable text) can't be read — OCR
  fallback (pytesseract + Poppler) covers most of these, but ask a student to
  re-upload a text-based PDF if theirs still shows "Failed" and they rank
  important.
- Very large batches (200+) can occasionally hit Google Drive's rate limits.
  If you see a burst of failures, wait 10–15 minutes and re-run — cached
  resumes won't be re-downloaded.
- All processing happens locally in memory / on this machine's disk (the
  resume cache and reference lists) — nothing is uploaded anywhere else.

## Possible upgrades later

- Swap the TF-IDF similarity for real semantic embeddings
  (`sentence-transformers`) for smarter matching.
- Deploy to Streamlit Community Cloud for a shared URL instead of
  local-only — note this resets the local resume cache and reference lists
  on every redeploy, since Streamlit Cloud's filesystem isn't persistent.
