# Feature Distribution Viewer

Interactive Streamlit app for visualizing **feature distributions** by group (Humans vs AI).  
Upload your own CSVs, filter humans by **COURSE**, pick a **feature**, and explore **boxplots with jittered points** (hover shows **TID** and value). No data is stored.

## What it does
- Upload **Measurements CSV** (features + `TID`, `Group`)
- Upload **Meta CSV** (at least `TID`, `COURSE`)
- Filter: Humans (all or by COURSE) and optionally include AI
- Choose **All features** or a curated **NM selection**
- Plot **original-scale** distributions with hoverable **TID**
- Optional aggregation to **one row per (Group, TID)** (mean)

## Expected CSV columns
- **Measurements CSV:** `TID`, `Group` (values `AI`/`Human` or `1`/`0`), plus numeric feature columns
- **Meta CSV:** `TID`, `COURSE`
- Any extra columns are ignored

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
