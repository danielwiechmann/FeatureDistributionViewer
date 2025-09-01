# AI & CEFR Explorer — Writing Analytics for Classrooms

Interactive Streamlit app to explore **sentence-structure** and **vocabulary** metrics, compare **Students vs AI**, and profile cohorts against **CEFR (A1–C2)** reference means. All computation happens locally while the app runs.

## What it does
- **Tutorial**: Two curated example texts with inline **highlights**; per-metric cards with **CEFR chips** (hover explains levels).
- **Students vs AI**: Filter Students by **COURSE**, pick a metric, and view **boxplots with jittered points** (hover shows **TID**, value, course).
- **CEFR Profiling**: Cohort distributions with **nearest CEFR mean** guides, **quartile band**, and an **individual student** readout.
- **Unlimited metric selection**: default shows 6 to avoid clutter; users can select any number of metrics.

## Data you can load
Use the **sidebar** to upload a **demo ZIP** (recommended) or mirror its structure with your own CSVs.

**1) Main data (required)**
- Columns: `TID`, `Group` (`AI`/`Human` or `1`/`0`)  
- Optional: `COURSE`, `text`  
- Plus numeric feature columns (e.g., `MLS`, `MLC`, `cPC`, `N_fulltext`, …)

**2) CEFR reference (optional)**
- Columns (case-insensitive): `level` (A1–C2), `feature`, `mean`, `sd`

**3) Example texts (optional, Tutorial)**
- Columns: `TID`, `text` (may also include metric columns)

**4) Feature descriptions (optional)**
- Required: `feature`, `display`, `definition`  
- Optional: `why`, `example_high`, `example_low`, `subcategory`, `family`

## Configuration (in code)
- Whitelists: `SENTENCE_STRUCTURE_SUBCATS`, `VOCABULARY_SUBCATS`
- Rename labels (UI only): `CUSTOM_RENAMES = {"N_fulltext": "Number of Words Tokens (instances)", ...}`
- Remove metrics everywhere: `PRUNE_FEATURES = {"NDW", "cTTR", ...}`
- Meta columns never shown: `ALWAYS_EXCLUDE = {"TID","Group","COURSE","wid","text"}`

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
