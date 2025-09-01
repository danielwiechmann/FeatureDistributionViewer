# ------------------------------------------------------------
# AI & CEFR Level Analyzer  — Global feature config + helpers
# ------------------------------------------------------------

import math
import html
import os
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="AI_CEFR_Explorer", layout="centered")



# ---------- Defaults for feature docs if we don't override ----------
DEFAULT_FEATURE_DOCS = {
    "MLS": {
        "display": "Mean Length of Sentence (MLS)",
        "definition": "Average number of words per sentence.",
        "why": "Short sentences can feel list-like; very long ones can strain readability.",
        "example_high": "",
        "example_low": "",
        "subcategory": "Length of Production Unit",
        "family": "Sentence Structure",
    },
    "MLC": {
        "display": "Mean Length of Clause (MLC)",
        "definition": "Average number of words per clause.",
        "why": "Captures clause-level elaboration within sentences.",
        "example_high": "",
        "example_low": "",
        "subcategory": "Length of Production Unit",
        "family": "Sentence Structure",
    },
    "LD": {
        "display": "Lexical Density (LD)",
        "definition": "Proportion of content words.",
        "why": "Higher density often means more information; too high can reduce readability.",
        "example_high": "",
        "example_low": "",
        "subcategory": "Lexical Density",
        "family": "Vocabulary",
    },
}

# ========================= STRICT FEATURE MAPPING (WHITELIST) =========================
# Edit these dicts to control which metrics are available anywhere in the app.
# Keys = internal column codes in your data; values = default display labels.
SENTENCE_STRUCTURE_SUBCATS = {
    "Length of Production Unit": {
        "MLS": "Mean Length of Sentence (MLS)",
        "MLC": "Mean Length of Clause (MLC)",
        "MLT": "Mean Length of T-unit (MLT)",
        "n_sentences": "Number of Sentences",
        "total_words": "Number of Words",
        "total_T_units": "Number of T-units",
        "total_clauses": "Number of Clauses",
    },
    "Sentence Complexity": {
        "CS": "Complex T-Unit Ratio (CS)",

    },
    "Subordination": {
        "CT": "Dependent Clauses per T-unit (CT)",
        "cTT": "T-Unit w/ Subordination (cTT)",
        "dCC": "Dependent Clauses per Clause (dCC)",
        "dCT": "Dependent Clauses per T-unit (dCT)",
        "total_dependent_clauses": "Number of Dependent Clauses",
        "percentage_sent_with_depClause": "% Sentences with Dependent Clause",
    },
    "Coordination": {
        "cPC": "Coordinate Phrases per Clause (cPC)",
        "cPT": "Coordinate Phrases per T-unit (cPT)",
        "TS": "T/S Ratio (TS)",
        "sum_TS_cPT": "Sum TS + cPT",
        "pct_sentences_with_coordP": "% Sentences with Coordination",
    },
    "Noun Phrase Complexity": {
        "cNC": "Complex Nominals per Clause (cNC)",
        "cNT": "Complex Nominals per T-unit (cNT)",
        "cNS": "Complex Nominals per Sentence (cNS)",
        "NPpre": "NP Pre-modification (NPpre)",
        "NPpost": "NP Post-modification (NPpost)",
        "n_nominals_spacy": "Number of Noun Phrases (spaCy)",
        "total_complex_nominals": "Number of Complex Nominals",
        "percentage_nominals_complex": "% Complex Nominals",
    },
}

VOCABULARY_SUBCATS = {
    "Lexical Density": {
        "LD": "Lexical Density (LD)",
        "percentage_CW": "Content Words (%)",
        "content_words_percent": "Content Words (%)",
    },
    "Lexical Diversity": {
        "NDW": "Number of Different Words (NDW)",
        "cTTR": "Corrected TTR (cTTR)",
        "rTTR": "Root TTR (rTTR)",
        "lwVAR": "Lexical Word Variation",
        "lwVAR_fulltext": "Lexical Word Variation (Full Text)",
        "N_fulltext": "N (Full Text)",
        "T_fulltext": "Tokens (Full Text)",
        "NDW_fulltext": "NDW (Full Text)",
        "TTR_fulltext": "TTR (Full Text)",
        "cTTR_fulltext": "cTTR (Full Text)",
        "rTTR_fulltext": "rTTR (Full Text)",
        "bTTR_fulltext": "bi-gram TTR (Full Text)",
        "N_lex_fulltext": "N (Lex Only, Full Text)",
        "T_lex_fulltext": "Tokens (Lex Only, Full Text)",
    },
    "Lexical Sophistication": {
        "MLWc": "Mean Word Length (chars)",
        "MLWs": "Mean Word Length (syllables)",
        "B2KBANC": "Beyond 2000 Words (BANC)",
        "B2KBBNC": "Beyond 2000 Words (BNC)",
        "percentage_B2KBANC": "% Beyond 2000 (BANC)",
        "percentage_B2KBBNC": "% Beyond 2000 (BNC)",
        "T10KCOCAw": "Top 10K COCA Written Coverage",
        "T10KCOCAs": "Top 10K COCA Spoken Coverage",
        "top2k_BNC_percent": "Top 2k BNC (%)",
        "top2k_ANC_percent": "Top 2k ANC (%)",
        "JDCOCAw": "Julliand’s D COCA Written",
        "JDCOCAw_weighted_sum": "JDCOCAw Weighted Sum",
        "JDCOCAw_typeweighted_norm": "JDCOCAw Type-weighted Norm",
        "JDCOCAw_score_0_100": "JDCOCAw Dispersion Score (0–100)",
        "tri10k_cov__COCA_acad": "Tri10K Cov COCA Academic",
        "tri10k_cov__COCA_fiction": "Tri10K Cov COCA Fiction",
        "tri10k_cov__COCA_mag": "Tri10K Cov COCA Magazine",
        "tri10k_cov__COCA_news": "Tri10K Cov COCA News",
        "tri10k_cov__COCA_spok": "Tri10K Cov COCA Spoken",
        "3GNLFa": "3GNLF Academic",
        "3GNLFf": "3GNLF Fiction",
        "3GNLFs": "3GNLF Spoken",
        "3GNLFtv": "3GNLF TV/Media",
        "3GNLFw": "3GNLF Web",
    },
}

# Order of subcategories inside each family (for selectors)
SUBCAT_ORDER = {
    "Sentence Structure": [
        "Length of Production Unit",
        "Sentence Complexity",
        "Subordination",
        "Coordination",
        "Noun Phrase Complexity",
    ],
    "Vocabulary": [
        "Lexical Density",
        "Lexical Diversity",
        "Lexical Sophistication",
    ],
}

# Optional: nicer display names for subcategories in the UI
SUBCAT_LABEL_OVERRIDES = {
    # "Subordination": "Clause subordination",
}

# Optional: rename metrics globally (DISPLAY ONLY; keys/codes stay the same)
CUSTOM_RENAMES = {
    "n_sentences": "Number of Sentences",
    "N_fulltext": "Number of Words Tokens (instances)",
    "T_fulltext": "Number of Word Types (different words)",
    "NDW_fulltext": "NDW (Full Text)",
    "TTR_fulltext": "Type/Token Ratio (TTR)",
    "cTTR_fulltext": "Corrected TTR (cTTR)",
    "rTTR_fulltext": "rTTR",
    "bTTR_fulltext": "bi-logarithmic TTR (Full Text)",
    "N_lex_fulltext": "Number of Lexical Words",
    "T_lex_fulltext": "Number of Lexical Word Tokens (instances)",
    "JDCOCAw_score_0_100": "JDCOCAw Dispersion Score (0–100)",
    # "LD": "Lexical Density (content-word share)",
}

# === Option B: global prune list (fully removes metrics from the app) ===
# Put metric CODES here to remove them everywhere (selectors, counts, plots).
PRUNE_FEATURES = {
    # Examples (uncomment to prune):
    "sum_TS_cPT", "NDW", "cTTR", "rTTR", "lwVAR",
    "bTTR_fulltext", "rTTR_fulltext", "NDW_fulltext",
    "JDCOCAw","JDCOCAw_weighted_sum","JDCOCAw_typeweighted_norm"
}

# Always exclude/meta columns (never selectable)
ALWAYS_EXCLUDE = {"TID", "Group", "COURSE", "wid", "text"}

def _flatten_mapping():
    """Return (all_features, display_overrides, feature->subcategory, family->dict(subcat->mapping))."""
    all_feats = set()
    display = {}
    feat_to_sub = {}
    fam_to_map = {
        "Sentence Structure": SENTENCE_STRUCTURE_SUBCATS,
        "Vocabulary": VOCABULARY_SUBCATS,
    }
    for fam, submap in fam_to_map.items():
        for sub, feats in submap.items():
            for code, disp in feats.items():
                if code in PRUNE_FEATURES:
                    continue  # fully removed
                all_feats.add(code)
                display[code] = disp
                feat_to_sub[code] = sub
    return all_feats, display, feat_to_sub, fam_to_map

ALL_MAPPED_FEATURES, DISPLAY_OVERRIDES, FEAT_TO_SUBCATEGORY, FAMILY_TO_SUBCATS = _flatten_mapping()

# Apply custom renames *after* building from the mapping (display-only)
DISPLAY_OVERRIDES.update(CUSTOM_RENAMES)

def _is_excluded(name: str) -> bool:
    """
    Exclude meta columns, pruned features, and stray CE.* columns from any
    ad-hoc feature discovery (safety net). Primary pruning happens in the mapping.
    """
    if name in ALWAYS_EXCLUDE or name in PRUNE_FEATURES:
        return True
    if re.match(r"(?i)^ce[._]", str(name or "")):
        return True
    return False

_is_hidden = _is_excluded

# ---------------- Helpers used across the app ----------------

# Salutation / closing lines to *ignore* in highlights
GREET_RE  = re.compile(r"^\s*(dear|hi|hello)\s+[^,\n]{1,60},\s*", re.IGNORECASE)
CLOSING_RE = re.compile(r"\n\s*(sincerely|best|yours|kind regards)[^.\n]*\n", re.IGNORECASE)


def is_dark_theme() -> bool:
    """Return True if Streamlit theme base is dark."""
    try:
        return str(st.get_option("theme.base")).lower() == "dark"
    except Exception:
        return False

def quartile_band_html(percentile: float, *, height: int = 22) -> str:
    """
    Render an adaptive Q1–Q4 band with a pointer at `percentile` (0–100).
    Colors switch for dark/light theme automatically.
    """
    dark = is_dark_theme()
    # segment fills
    col_low = "#fecaca" if not dark else "#f87171"   # red-ish
    col_mid = "#bfdbfe" if not dark else "#93c5fd"   # blue-ish
    col_top = "#a7f3d0" if not dark else "#34d399"   # green-ish
    # card/lines/text
    tri = "#111827" if not dark else "#f3f4f6"
    bar_bg = "#ffffff" if not dark else "#0f172a"
    border = "#e5e7eb" if not dark else "rgba(255,255,255,0.20)"
    txt = "#111827" if not dark else "#e5e7eb"

    left = max(0.0, min(100.0, float(percentile)))

    return f"""
<div style="width:100%; color:{txt}">
  <div style="position:relative; height:{height}px; border:1px solid {border};
              border-radius:999px; overflow:hidden; background:{bar_bg}">
    <div style="position:absolute; left:0;     top:0; bottom:0; width:25%; background:{col_low};"></div>
    <div style="position:absolute; left:25%;  top:0; bottom:0; width:50%; background:{col_mid};"></div>
    <div style="position:absolute; left:75%;  top:0; bottom:0; width:25%; background:{col_top};"></div>

    <!-- pointer -->
    <div style="position:absolute; top:-6px; left:calc({left}% - 6px);
                width:0; height:0; border-left:6px solid transparent; border-right:6px solid transparent;
                border-top:10px solid {tri};"></div>
  </div>
  <div style="display:flex; justify-content:space-between; margin-top:6px;">
    <span>Q1</span><span>Q2</span><span>Q3</span><span>Q4</span>
  </div>
</div>
"""


def subcategories_for_family(df: pd.DataFrame, family: str):
    """
    Return a list of subcategory keys for a family,
    filtered to subcats that have at least one present, non-excluded metric in df.
    """
    submap = FAMILY_TO_SUBCATS.get(family, {})
    if not submap:
        return []

    ordered = SUBCAT_ORDER.get(family, list(submap.keys()))
    result = []
    for sub in ordered:
        feats = [f for f in submap.get(sub, {}).keys() if f not in PRUNE_FEATURES]
        feats = [f for f in feats if f in df.columns and not _is_excluded(f)]
        if feats:
            result.append(sub)
    return result

def features_in_subcategory(df: pd.DataFrame, family: str, subcat: str):
    """
    Return feature codes for a given (family, subcat),
    filtered to columns that exist in df and are not excluded.
    """
    feats_map = FAMILY_TO_SUBCATS.get(family, {}).get(subcat, {})
    feats = [f for f in feats_map.keys() if f not in PRUNE_FEATURES]
    feats = [f for f in feats if f in df.columns and not _is_excluded(f)]
    # numeric only (robustness)
    feats = [f for f in feats if pd.api.types.is_numeric_dtype(df[f])]
    return feats

def _display_name_for(feat: str) -> str:
    """
    Resolve display label for a metric:
    1) st.session_state.feat_docs[feat]['display'] if present,
    2) DISPLAY_OVERRIDES (mapping + CUSTOM_RENAMES),
    3) fall back to the raw code.
    """
    fd = (st.session_state.get("feat_docs") or {}).get(feat, {})
    return (fd.get("display")
            or DISPLAY_OVERRIDES.get(feat)
            or feat)

def _display_subcat_name(subcat: str) -> str:
    return SUBCAT_LABEL_OVERRIDES.get(subcat, subcat)

def feature_selectbox(label: str, options: list[str], key: str):
    """Selectbox that shows pretty names but returns the code."""
    return st.selectbox(label, options=options, key=key, format_func=_display_name_for)

def feature_multiselect(label: str, options: list[str], key: str, default: list[str] | None = None):
    """Multiselect that shows pretty names but returns the codes."""
    return st.multiselect(label, options=options, default=default or [], key=key, format_func=_display_name_for)


# ========================= Copy / text =========================


PROMPT_TITLE = "Task: FutureMe Letter"
PROMPT_MD = """
**Writing Task.** Write a letter to your future self, imagining it’s the year **2050** – 25 years from now.  
**Length.** At least **350 words** (more is fine).

**Some points to consider**
- What goals and dreams do you want to have achieved by 2050?
- Your ideal future in key areas (career, finances, family, friendships, health, hobbies)?
- What might the world be like in 2050 (technology, society, environment), and how could this affect your life?
"""

DEFAULT_ABOUT_MD = """
### 1) Motivation
We compare writing across conditions to understand how structural and vocabulary features vary in authentic texts.

### 2) Data & Measures
Data are aggregated **per TID**. Each row contains numeric text-analytic metrics and optional metadata (`Group`, `COURSE`, `text`).

### 3) Interpretation Caveats
Metrics are indicators, not absolutes. Compare like with like (same prompt/course). Outliers and typos can shift scores; consider multiple features jointly.
"""

TAB_PANELS = {
    "tutorial": {
        "about_title": "About this tutorial",
        "about_md": """
        
### How to use this app 

This tutorial is a guided walk-through that will give you a clear, practical feel for what each **feature subcategory** measures. As you read, the color highlights act as signposts that connect visible **text cues** to the **numbers** on the metric cards, so you can see exactly how wording and structure relate to the scores. You can switch between a **more complex** and a **less complex** example to notice which choices—like longer sentences, denser noun phrases, or greater word variety—tend to push scores up or down.

### How does it work?
1. **Pick a family and subcategory** at the top (e.g., *Sentence Structure → Subordination*).  
2. **Choose an example text** (More complex / Less complex). The text is automatically **highlighted** to show the selected feature in context.  
3. Scan the **metric cards** to see the example’s scores for all metrics in this subcategory, with simple explanations.

#### What will I see?
- **Two example texts** (curated - not actual student work) to build intuition.  
- **Color highlights** that make the feature patterns visible in the text.  
- **Metric cards** that summarize the example’s values across the subcategory.

        """,
        "prompt_title": "Background & Writing Prompt",
        "prompt_expanded": True,
        "prompt_md": """

**Introduction: Generative AI and its Significance in Education**

**Generative Artificial Intelligence (AI)** comprises technologies that can **autonomously create new content** based on large-scale datasets— including text, images, video, and code. Particularly powerful are **Large Language Models (LLMs)** such as the **Generative Pre-trained Transformer (GPT)**, which transform natural-language prompts into human-like text and can carry out complex language tasks like translation, editing, or argumentative writing. With the release of **ChatGPT** (OpenAI, 2022), a conversation-optimized variant of the GPT family, public debate about the implications of such tools for teaching and learning has intensified markedly. 

**Didactic Potential and Challenges of AI-Supported Writing**

Research highlights the pedagogical potential of these systems, for example through immediate, adaptive feedback that can improve both text quality and learner motivation (Sailer et al., 2023; Jansen et al., 2024; Meyer et al., 2024). At the same time, concerns are growing that **excessive reliance on AI for writing** may erode writing’s function as a central instrument of thinking and knowledge construction. As a recent editorial in *[Nature Reviews Bioengineering](https://www.nature.com/articles/s44222-025-00323-4)* underscores, **writing** is not merely stringing words together; it is **a foundational cognitive practice** essential for structuring ideas, building argumentation skills, and acquiring communicative competence (Hutson, 2022). If this process is outsourced to AI, learners lose opportunities to engage deeply with subject matter and to transform research findings into a coherent, traceable, and convincing form.

A recent MIT study offers instructive evidence: participants were assigned to one of three conditions—writing an essay from their own knowledge (“brain-only”), using a search engine, or using ChatGPT. EEG measures of brain connectivity (Dynamic Directed Transfer Function, dDTF), combined with linguistic analyses and interviews, revealed a clear pattern: connectivity was strongest without tools, weaker with a search engine, and lowest with ChatGPT. LLM-assisted texts were more homogeneous, less well remembered, and more weakly associated with a sense of authorship. Particularly striking was a form of “**cognitive debt**”: after **repeated ChatGPT use**, those who then had to write without AI showed **under-activation and deficits in memory retrieval**. Conversely, switching from own writing to ChatGPT reactivated broader neural networks. These findings suggest that while **ChatGPT**—and comparable LLMs—may yield short-term efficiency gains, **they can impair learning capacity, memory formation, and cognitive independence over the long term** (Kosmyna et al., 2024).

**Writing in Foreign Language Education: Importance and Challenges**

In **foreign language learning**, **writing** plays a **pivotal role**: it is widely considered the most cognitively demanding skill and a central benchmark in language education. Unlike reading or listening, writing requires learners not only to retrieve linguistic forms but to actively construct, organize, and revise their ideas. **Writing** is both a key assessment target—e.g., in the German Abitur—and a **core competence in academic and professional contexts** (applications, formal correspondence, reports). Successful writing presupposes complex sentence structures and a sophisticated lexicon, as well as the ability to navigate different registers and task types. Moreover, **writing fosters critical thinking by prompting planning, structuring, and revision processes**—which is why it is often deemed more demanding than speaking, listening, or reading.

**Teachers and the Detection of AI-Generated Texts**

With the increasing classroom use of ChatGPT, the question arises to what extent teachers can reliably distinguish AI-generated texts from student writing—central both to fair assessment and to safeguarding authentic learning and developmental processes. The North Rhine-Westphalia Ministry of Education assumes that teachers can recognize AI-generated texts based on their experience (Ministerium für Schule und Bildung des Landes Nordrhein-Westfalen, 2023b, p. 8). Empirical research paints a different picture: hit rates are typically around 60%—barely above chance (Köbis & Mossiness, 2021; Gunser et al., 2022). Gao et al. (2023) likewise report limited detection: reviewers correctly identified only about two-thirds of ChatGPT-generated abstracts, while 14% of human texts were falsely flagged as AI-generated. Fleckenstein et al. (2024) confirm this in school settings: in two studies, neither prospective nor experienced teachers could reliably distinguish student essays from ChatGPT texts. Notably, teachers often made highly confident judgments that were nonetheless frequently incorrect.

**Research Questions and Study Design Underpinning This Work**

Against this backdrop, the Master’s thesis of **Nina Menger**, supervised by **PD Dr. Elma Kerz** (CEO of **[Exaia Technologies](https://exaia-tech.com/)** and Privatdozentin at **[RWTH Aachen](https://www.rwth-aachen.de/cms/~a/root/?lidx=1)**), investigates whether quantifiable measures of linguistic complexity and sophistication—established in second-language research—can be used to reliably differentiate student texts from ChatGPT texts. The dataset is a parallel corpus of **148 personal “letters to the future” essays** based on the prompt “Write a letter to your future self, imagining it’s the year 2050.” Of these, **74 texts** were produced by **10th-grade students** at Gymnasien in North Rhine-Westphalia learning English as a foreign language. The students had **45 minutes** to write in an authentic classroom setting. The comparison set consists of **74 texts generated by ChatGPT-4**.

For analysis, the study employed the **next-generation text mining and analytics tool CYMO**—developed by **Exaia Technologies**. CYMO offers a broad portfolio of scientifically grounded, expert-curated metrics that capture multiple dimensions of **language use**, enabling a high-resolution profile of linguistic competence. For the present study, **33 metrics** were selected as key indicators of **syntactic complexity and diversity** (i.e., expansion and variation of sentence structure) and of **lexical richness, sophistication, and variety** (i.e., breadth and quality of vocabulary). These metrics allow for a precise and differentiated quantification of learners’ linguistic performance in comparison to AI-generated texts. In combining **scalable NLP processing** with **research-based metrics**, CYMO goes far beyond traditional text-analysis approaches and opens new avenues for **empirical educational research**, **second-language acquisition studies**, and the development of robust strategies for **AI literacy**.

Additionally, students completed two questionnaires: the **Language Experience and Proficiency Questionnaire (LEAP-Q)** to capture biographical and language backgrounds, and the **Big Five Inventory (BFI-44)** to measure individual personality traits. 

The study addresses the following **central research questions**:

* **Syntactic complexity**: Can student and ChatGPT texts be distinguished using 33 selected metrics of sentence-structure complexity (length of production units, degree of coordination and subordination, structural variety, structural affinity)?
* **Lexical complexity**: Do differences emerge in lexical density, diversity, and sophistication (including register use)?
* **Individual differences**: To what extent do language-learning experiences and personality traits (Big Five) influence learners’ linguistic productivity?

---

*How this informs the app:* Building on these insights, this app is designed to **support English teachers in Germany** by providing **transparent, research-based indicators of writing proficiency** (e.g., syntactic and lexical profiles) that complement human judgment. Rather than replacing writing or teacher expertise, it aims to **strengthen fair assessment and AI literacy** while keeping the cognitive value of student writing at the center of classroom practice.


        """,
    },
    "ai": {
        "about_title": "About - Comparison: AI vs Human",
        "about_md": """
**Goal — Compare Student vs. AI Distributions:** The app visualizes distributions of objective metrics for sentence-structure complexity and vocabulary sophistication. 

- Explore how the objective writing metrics differ between student-written and AI-generated texts. Spot overlaps, gaps, and systematic shifts in sentence-structure complexity and vocabulary sophistication.
- Use the COURSE filter to include/exclude specific cohorts from the study.
- **How to read the visualization:** Each black dot represents a single text (student or AI). Hover to see the exact metric value and source.

        """,
        "prompt_title": "Writing Prompt",
        "prompt_md": """

**Writing Task.** Write a letter to your future self, imagining it’s the year **2050** – 25 years from now.  
**Length.** At least **350 words** (more is fine).

**Some points to consider**
- What goals and dreams do you want to have achieved by 2050?
- Your ideal future in key areas (career, finances, family, friendships, health, hobbies)?
- What might the world be like in 2050 (technology, society, environment), and how could this affect your life?

        """,
    },
    "cefr": {
        "about_title": "About - Writing Proficiency Profiling",
        "about_md": """

**Goal — See where each student stands, at a glance**

The app visualizes distributions of **objective metrics** for **sentence-structure complexity** and **vocabulary sophistication**.

- **Group insight:** See your selected group’s performance and how scores align with **mean CEFR (A1–C2) benchmarks** from the [EFCAMDAT](https://ef-lab.mmll.cam.ac.uk/EFCAMDAT.html) L2 reference database.
- **Individual insight:** Locate each student within the distribution to identify strengths, gaps, and outliers for targeted feedback.

        """,
        "prompt_title": "Writing Prompt",
        "prompt_md": """
**Writing Task.** Write a letter to your future self, imagining it’s the year **2050** – 25 years from now.  
**Length.** At least **350 words** (more is fine).

**Some points to consider**
- What goals and dreams do you want to have achieved by 2050?
- Your ideal future in key areas (career, finances, family, friendships, health, hobbies)?
- What might the world be like in 2050 (technology, society, environment), and how could this affect your life?
        """,
    },
}

# ========================= Session init =========================
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df_agg" not in st.session_state:
    st.session_state.df_agg = None
if "cefr" not in st.session_state:
    st.session_state.cefr = None
if "examples" not in st.session_state:
    st.session_state.examples = None
if "feat_docs" not in st.session_state:
    st.session_state.feat_docs = DEFAULT_FEATURE_DOCS.copy()

# ========================= Core helpers =========================
def _load(upload):
    return pd.read_csv(upload) if upload is not None else None

def normalize_group_column(df):
    if "Group" in df.columns and df["Group"].dtype == object:
        df["Group"] = df["Group"].map({"AI": 1, "Human": 0, "Students": 0}).fillna(df["Group"])
    if "Group" in df.columns:
        df["Group"] = pd.to_numeric(df["Group"], errors="coerce")
    return df

def aggregate_to_tid(df_in):
    """No-op when already one row per (Group, TID)."""
    df = df_in.drop(columns=[c for c in ["wid"] if c in df_in.columns]).copy()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    num_cols = [c for c in num_cols if c != "Group"]
    agg = {c: "mean" for c in num_cols}
    if "COURSE" in df.columns:
        agg["COURSE"] = "first"
    if "text" in df.columns:
        agg["text"] = "first"
    return df.groupby(["Group", "TID"], dropna=False, as_index=False).agg(agg)

def _pretty_label(code: str) -> str:
    return code.replace("__", " ").replace("_", " ").strip().title()

def feature_family(name: str):
    if name in ALL_MAPPED_FEATURES:
        for sub in SENTENCE_STRUCTURE_SUBCATS:
            if name in SENTENCE_STRUCTURE_SUBCATS[sub]:
                return "Sentence Structure"
        return "Vocabulary"
    return None

def feature_subcategory(name: str):
    return FEAT_TO_SUBCATEGORY.get(name)

def subcategories_for_family(df_in: pd.DataFrame | None, family: str):
    mapping = SENTENCE_STRUCTURE_SUBCATS if family == "Sentence Structure" else VOCABULARY_SUBCATS
    order = SUBCAT_ORDER[family]
    if df_in is None:
        return []
    present = []
    for sub in order:
        feats = mapping[sub]
        if any((f in df_in.columns) and not _is_hidden(f) for f in feats.keys()):
            present.append(sub)
    return present

def features_in_subcategory(df_in: pd.DataFrame | None, family: str, subcategory: str):
    if df_in is None:
        return []
    mapping = SENTENCE_STRUCTURE_SUBCATS if family == "Sentence Structure" else VOCABULARY_SUBCATS
    feats = mapping.get(subcategory, {})
    return sorted([f for f in feats.keys() if (f in df_in.columns) and not _is_hidden(f)])

# Build a reverse lookup: subcategory -> family (used for docs)
SUBCAT_TO_FAMILY = {}
for fam, submap in FAMILY_TO_SUBCATS.items():
    for sub in submap.keys():
        SUBCAT_TO_FAMILY[sub] = fam

def ensure_feature_docs(df: pd.DataFrame):
    """
    Ensure st.session_state.feat_docs has entries for every *visible* metric:
    - Metric must be in our strict whitelist mapping (ALL_MAPPED_FEATURES)
    - Not in PRUNE_FEATURES / excluded
    - Present in df
    """
    if "feat_docs" not in st.session_state or not isinstance(st.session_state.feat_docs, dict):
        st.session_state.feat_docs = {}

    cols = set(df.columns)

    # Only mapped + present + not excluded
    showable = {c for c in cols if (c in ALL_MAPPED_FEATURES) and not _is_excluded(c)}

    for code in sorted(showable):
        # seed from provided defaults if available
        seed = DEFAULT_FEATURE_DOCS.get(code, {})
        subcat = FEAT_TO_SUBCATEGORY.get(code, seed.get("subcategory", ""))
        family = SUBCAT_TO_FAMILY.get(subcat, seed.get("family", ""))

        # prefer explicit docs from session, else build a sensible default
        doc = st.session_state.feat_docs.get(code, {})
        if not doc:
            st.session_state.feat_docs[code] = {
                "display": seed.get("display", DISPLAY_OVERRIDES.get(code, code)),
                "definition": seed.get("definition", ""),
                "why": seed.get("why", ""),
                "example_high": seed.get("example_high", ""),
                "example_low": seed.get("example_low", ""),
                "subcategory": subcat,
                "family": family,
            }
        else:
            # make sure display name reflects current overrides if not set
            doc.setdefault("display", DISPLAY_OVERRIDES.get(code, code))
            doc.setdefault("subcategory", subcat)
            doc.setdefault("family", family)


def load_feat_desc(df_desc):
    if df_desc is None or df_desc.empty:
        return DEFAULT_FEATURE_DOCS, None
    lower = {c.lower(): c for c in df_desc.columns}
    need = {"feature", "display", "definition"}
    if not need.issubset(lower.keys()):
        return DEFAULT_FEATURE_DOCS, None
    df = df_desc.rename(
        columns={
            lower["feature"]: "feature",
            lower["display"]: "display",
            lower["definition"]: "definition",
            lower.get("why", "why"): "why",
            lower.get("example_high", "example_high"): "example_high",
            lower.get("example_low", "example_low"): "example_low",
            lower.get("subcategory", "subcategory"): "subcategory",
            lower.get("family", "family"): "family",
        }
    )
    docs = {}
    for _, r in df.iterrows():
        feat = str(r["feature"])
        docs[feat] = {
            "display": str(r.get("display", feat)),
            "definition": str(r.get("definition", "")),
            "why": str(r.get("why", "")),
            "example_high": str(r.get("example_high", "")),
            "example_low": str(r.get("example_low", "")),
            "subcategory": (str(r.get("subcategory", "")).strip() or None),
            "family": (str(r.get("family", "")).strip() or None),
        }
    merged = {**DEFAULT_FEATURE_DOCS}
    merged.update(docs)
    listed = set(str(x) for x in df["feature"].astype(str).tolist())
    return merged, listed

def process_loaded_files(
    df_data,
    df_cefr: pd.DataFrame | None = None,
    df_examples: pd.DataFrame | None = None,
    df_featdesc: pd.DataFrame | None = None,
    source_label: str = "manual",
) -> bool:
    if df_data is None:
        st.sidebar.error("No Data CSV found.")
        return False
    for c in ("TID", "Group"):
        if c not in df_data.columns:
            st.sidebar.error(f"Data CSV must contain '{c}'.")
            return False

    df_data = normalize_group_column(df_data)
    if "COURSE" not in df_data.columns:
        df_data["COURSE"] = np.nan
    if "text" not in df_data.columns:
        df_data["text"] = np.nan

    st.session_state.df_raw = df_data.copy()
    st.session_state.df_agg = aggregate_to_tid(df_data)
    st.session_state.cefr = df_cefr

    if df_examples is not None:
        df_examples = normalize_group_column(df_examples)
        if "wid" in df_examples.columns:
            if "Group" not in df_examples.columns:
                df_examples["Group"] = 0
            if "COURSE" not in df_examples.columns:
                df_examples["COURSE"] = np.nan
            df_examples = aggregate_to_tid(df_examples)
        st.session_state.examples = df_examples
    else:
        st.session_state.examples = None

    if df_featdesc is not None and not df_featdesc.empty:
        docs, _ = load_feat_desc(df_featdesc)
        st.session_state.feat_docs.update(docs)

    ensure_feature_docs(st.session_state.df_agg)

    flags = [
        "CEFR ✓" if df_cefr is not None else "CEFR —",
        "Examples ✓" if df_examples is not None else "Examples —",
        "FeatDesc ✓" if (df_featdesc is not None and not df_featdesc.empty) else "FeatDesc —",
    ]
    st.sidebar.success(f"Loaded ({source_label}): {df_data.shape[0]} rows. " + " | ".join(flags))
    return True

# ---------- Quick-load from folder ----------
def _find_first(folder: str, patterns: list[str]):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(folder, pat)))
        if hits:
            return hits[0]
    return None

def load_all_from_folder(folder: str) -> bool:
    if not folder or not os.path.isdir(folder):
        st.sidebar.error(f"Folder not found: {folder!r}")
        return False
    cand = {
        "data": ["1_*.csv", "1-*.csv", "1.csv", "data.csv", "1_text_feature*.csv", "1_text*with_meta*.csv"],
        "cefr": ["2_*.csv", "2-*.csv", "2.csv", "cefr.csv"],
        "examples": ["3_*.csv", "3-*.csv", "3.csv", "examples*.csv", "example_texts.csv"],
        "featdesc": ["4_*.csv", "4-*.csv", "4.csv", "feature_descriptions.csv", "features.csv", "featdesc.csv"],
    }
    paths = {k: _find_first(folder, v) for k, v in cand.items()}
    if not paths["data"]:
        st.sidebar.error("Required file missing: something like **1_data.csv** (or data.csv) in that folder.")
        return False

    try:
        df_data = pd.read_csv(paths["data"])
        df_cefr = pd.read_csv(paths["cefr"]) if paths["cefr"] else None
        df_examples = pd.read_csv(paths["examples"]) if paths["examples"] else None
        df_featdesc = pd.read_csv(paths["featdesc"]) if paths["featdesc"] else None
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV(s): {e}")
        return False

    try:
        ok = process_loaded_files(
            df_data,
            df_cefr=df_cefr,
            df_examples=df_examples,
            df_featdesc=df_featdesc,
            source_label=f"folder: {os.path.basename(folder)}",
        )
        return bool(ok)
    except Exception as e:
        st.sidebar.error(f"Failed to process loaded files: {e}")
        return False

def _display_name_for(feat: str) -> str:
    fd = st.session_state.feat_docs.get(feat, {})
    return fd.get("display") or DISPLAY_OVERRIDES.get(feat, feat)

# ---------- Minimal base CSS (only for equal-width chip buttons) ----------
def _inject_base_css_once():
    if st.session_state.get("_base_css_done"):
        return
    st.markdown(
        """
<style>
/* Feature chip buttons */
div[data-testid="stButton"] > button {
  border-radius: 999px !important;
  padding: 8px 12px !important;
  font-size: 0.95rem !important;
  border: 1px solid #e5e7eb !important;
  width: 100% !important; min-height: 40px !important;
  white-space: nowrap !important; overflow:hidden; text-overflow: ellipsis;
}
div[data-testid="stButton"] > button[kind="secondary"] {
  background: #f3f4f6 !important; color:#111827 !important; border-color:#e5e7eb !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
  background: #d1d5db !important; color:#111827 !important; border-color:#9ca3af !important;
}
div[data-testid="stButton"] > button:hover { filter: brightness(0.96); }
</style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["_base_css_done"] = True

_inject_base_css_once()

# ---------- Feature chip picker (supports unlimited selection) ----------
def render_feature_chip_picker(
    feats: list[str],
    state_key: str,
    max_sel: int | None = None,          # None = no cap
    caption: str | None = None           # optional custom caption
) -> list[str]:
    feats = feats or []
    # Seed selection if missing (caller usually seeds to first 6; this is a fallback)
    sel = st.session_state.get(state_key)
    if sel is None:
        sel = feats[:6]
        st.session_state[state_key] = sel

    # Caption (reflect the cap only if one is set)
    if caption is None:
        if max_sel is None:
            st.caption("Click metrics to (de)select for the overview.")
        else:
            st.caption(f"Click metrics to (de)select for the overview (max {max_sel}).")
    else:
        st.caption(caption)

    # Render chips in 3 columns (or fewer if small)
    ncols = 3 if len(feats) >= 3 else max(1, len(feats))
    rows = [feats[i:i + ncols] for i in range(0, len(feats), ncols)]

    for row in rows:
        cols = st.columns(len(row))
        for col, feat in zip(cols, row):
            with col:
                is_selected = feat in sel
                if st.button(
                    _display_name_for(feat),
                    key=f"chip::{state_key}::{feat}",
                    type=("primary" if is_selected else "secondary"),
                    use_container_width=True,
                ):
                    if is_selected:
                        sel = [x for x in sel if x != feat]
                    else:
                        # Enforce only if a cap is provided
                        if (max_sel is not None) and (len(sel) >= max_sel):
                            st.warning(f"Limit reached ({max_sel}). Deselect another metric first.")
                            st.stop()
                        sel = sel + [feat]

                    st.session_state[state_key] = sel
                    st.rerun()

    return sel

# ####################################################################
# ==================== SIDEBAR (Load demo ZIP only) ====================

st.sidebar.header("Load the demo package")

st.sidebar.markdown("""
**How to load the demo**

Upload the **demo ZIP** we shared. We’ll extract it to a temporary folder and load everything from there.

**What’s inside?**
- Student and AI-generated texts from Nina Menger MA thesis 
- [CYMO](https://exaia-tech.com/cymo) feature analyses (per text)  
- CEFR reference data from **EFCAMDAT** — a large learner corpus of English essays with CEFR-related metadata curated by the University of Cambridge ([learn more](https://ef-lab.mmll.cam.ac.uk/EFCAMDAT.html))
""")

import tempfile, zipfile, shutil
from pathlib import Path

DEMO_EXTRACT_KEY = "demo_extract_root"

# ZIP uploader (the only control)
zip_file = st.sidebar.file_uploader("Upload demo ZIP", type=["zip"])

# Load button
if st.sidebar.button("Load demo ZIP", type="primary", use_container_width=True):
    if zip_file is None:
        st.sidebar.warning("Please upload a ZIP first.")
    else:
        # Clean up any previous extraction
        old_root = st.session_state.get(DEMO_EXTRACT_KEY)
        if old_root and Path(old_root).exists():
            try:
                shutil.rmtree(old_root)
            except Exception:
                pass

        # Extract to a fresh temp directory
        try:
            extract_root = tempfile.mkdtemp(prefix="ai_cefr_demo_")
            with zipfile.ZipFile(zip_file) as zf:
                zf.extractall(extract_root)

            # If the ZIP has a single top-level folder, use that folder as the root
            entries = [p for p in Path(extract_root).iterdir()]
            if len(entries) == 1 and entries[0].is_dir():
                demo_root = str(entries[0])
            else:
                demo_root = extract_root

            st.session_state[DEMO_EXTRACT_KEY] = demo_root
            st.sidebar.success("ZIP extracted. Loading data…")

            # Your existing loader
            load_all_from_folder(demo_root)

        except zipfile.BadZipFile:
            st.sidebar.error("That file doesn't look like a valid ZIP.")
        except Exception as e:
            st.sidebar.error(f"Couldn’t load the ZIP: {e}")







# --- (Temporarily disabled) individual file uploaders ---
# st.sidebar.markdown("---")
# st.sidebar.caption("Advanced: manual file loading (disabled for this tutorial)")
# data_upload = st.sidebar.file_uploader(
#     "Data CSV (rows=TIDs; required: TID, Group; optional: COURSE, text)", type=["csv"], key="u_data"
# )
# cefr_upload = st.sidebar.file_uploader(
#     "CEFR reference CSV (level, feature, mean, sd) [optional]", type=["csv"], key="u_cefr"
# )
# examples_upload = st.sidebar.file_uploader(
#     "Example texts CSV (independent exemplars) [optional]", type=["csv"], key="u_examples"
# )
# featdesc_upload = st.sidebar.file_uploader(
#     "Feature descriptions CSV [optional]", type=["csv"], key="u_featdesc"
# )
# load_btn = st.sidebar.button("Load / Reload", use_container_width=True)
# if load_btn:
#     try:
#         process_loaded_files(
#             _load(data_upload),
#             _load(cefr_upload),
#             _load(examples_upload),
#             _load(featdesc_upload),
#             source_label="manual",
#         )
#     except Exception as e:
#         st.sidebar.error(f"Failed to load: {e}")


# ========================= UI helpers =========================
def format_num(x, dec=1):
    if pd.isna(x):
        return "—"
    try:
        return f"{float(x):.{dec}f}"
    except Exception:
        return str(x)

def format_pct(x, dec=1):
    if pd.isna(x):
        return "—"
    try:
        return f"{float(x):.{dec}f}%"
    except Exception:
        return str(x)

def render_feature_pills(feats):
    chip_style = (
        "display:inline-block;padding:6px 10px;border-radius:999px;"
        "border:1px solid #e5e7eb;background:#f9fafb;color:#111;font-size:.85rem;margin:6px 6px 0 0;"
    )
    chunks = [
        f'<span style="{chip_style}">{html.escape(st.session_state.feat_docs.get(f, {}).get("display", DISPLAY_OVERRIDES.get(f, _pretty_label(f))))}</span>'
        for f in (feats or [])
    ]
    return '<div style="margin:6px 0 12px 0; display:flex; flex-wrap:wrap; gap:6px;">' + "".join(chunks) + "</div>"

def render_background_panel(tab_key: str):
    cfg = TAB_PANELS.get(tab_key, {})
    title = cfg.get("about_title", "About the study")
    body = cfg.get("about_md", DEFAULT_ABOUT_MD)
    with st.expander(f"{title} — click to expand", expanded=bool(cfg.get("about_expanded", False))):
        st.markdown(body)

def render_prompt_panel(tab_key: str):
    cfg = TAB_PANELS.get(tab_key, {})
    title = cfg.get("prompt_title", "Text prompt")
    body = cfg.get("prompt_md", f"### {PROMPT_TITLE}\n\n{PROMPT_MD}")
    with st.expander(f"{title} — click to expand", expanded=bool(cfg.get("prompt_expanded", False))):
        st.markdown(body)

def _is_percentish(feat: str) -> bool:
    f = (feat or "").lower()
    return ("percent" in f) or ("percentage" in f) or ("pct" in f)

def _fmt_feat_value(feat: str, v: float) -> str:
    return format_pct(v, 1) if _is_percentish(feat) else format_num(v, 1)

def _normalize_cefr_table(cefr_ref: pd.DataFrame | None) -> pd.DataFrame | None:
    if cefr_ref is None or cefr_ref.empty:
        return None
    cols = {c.lower(): c for c in cefr_ref.columns}
    need = {"level", "feature", "mean", "sd"}
    if not need.issubset(cols.keys()):
        return None
    return cefr_ref.rename(
        columns={
            cols["level"]: "level",
            cols["feature"]: "feature",
            cols["mean"]: "mean",
            cols["sd"]: "sd",
        }
    )

def render_shared_controls(df_base, prefix, show_pills: bool = True):
    if df_base is None or df_base.empty:
        st.warning("No data loaded yet. Use the sidebar to load your CSVs.")
        st.stop()

    c1, c2 = st.columns(2)
    family = c1.selectbox(
        "Select feature family",
        options=["Sentence Structure", "Vocabulary"],
        key=f"{prefix}_family",
    )
    subs = subcategories_for_family(df_base, family)
    if not subs:
        st.warning("No subcategories available for the selected family with current data/mapping.")
        st.stop()
    subcat = c2.selectbox(
        "Select feature subcategory",
        options=subs,
        key=f"{prefix}_subcat",
    )
    feats_in_sub = features_in_subcategory(df_base, family, subcat)

    # Show pills when available; stay silent otherwise.
    if feats_in_sub and show_pills:
        st.markdown("**Features in this subcategory**")
        st.markdown(render_feature_pills(feats_in_sub), unsafe_allow_html=True)

    return family, subcat, feats_in_sub
    

# --------------------- Shared: quartile band --------------------
def quartile_band_html(percentile_float: float) -> str:
    p = 0.0 if pd.isna(percentile_float) else float(percentile_float)
    p = max(0.0, min(100.0, p))
    return (
        '<div style="margin-top:8px;margin-bottom:6px">'
        '<div style="position:relative;height:14px;border-radius:7px;'
        'background:linear-gradient(to right,'
        '#d73027 0%, #d73027 25%,'
        '#fc8d59 25%, #fc8d59 50%,'
        '#91cf60 50%, #91cf60 75%,'
        '#1a9850 75%, #1a9850 100%);">'
        f'<div style="position:absolute;left:{p}%;top:-7px;transform:translateX(-50%);'
        'font-size:18px;color:#111;">▼</div>'
        '</div>'
        '<div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#666;">'
        '<span>Q1</span><span>Q2</span><span>Q3</span><span>Q4</span>'
        '</div>'
        '</div>'
    )

# --------------------- CEFR overview card renderer --------------------
def render_cefr_subcategory_overview(
    df_base: pd.DataFrame,
    feats_in_sub: list[str],
    cefr_ref: pd.DataFrame | None,
    courses_filter: list[str] | None,
):
    st.markdown(
    """
<style>
/* --- CEFR overview cards --- */
.ce-card{border:1px solid #e5e7eb;border-radius:12px;background:#fff;padding:10px 12px;margin-bottom:12px;}
/* Head becomes responsive: title on one line; stats below on small screens, inline on wide screens */
.ce-head{
  display:flex; flex-wrap:wrap; align-items:flex-end; gap:8px;
}
.ce-title{
  font-weight:800; font-size:1.0rem;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  flex: 1 1 100%;             /* full-width line by default (small screens) */
  line-height:1.2;
}
.ce-meta{
  color:#6b7280; font-size:.82rem; white-space:nowrap;
  flex: 1 1 100%;             /* stats go to the next line by default */
}

/* On wider screens, put title + stats on the same row again */
@media (min-width: 1000px){
  .ce-title{ flex: 0 1 auto; }
  .ce-meta { flex: 0 0 auto; margin-left:auto; }
}

/* Small cosmetic helpers */
.ce-desc{color:#374151;font-size:.86rem;line-height:1.35;margin:4px 0 8px;}
.ce-bwrap{position:relative;height:32px;margin:2px 0 4px;}
.ce-track{position:absolute;left:0;right:0;top:18px;height:5px;background:#eef2f7;border-radius:999px;}
.ce-band{position:absolute;top:15px;height:11px;background:rgba(2,132,199,.08);border-radius:6px;border:1px dashed rgba(2,132,199,.35);}
.ce-dot-stu{position:absolute;top:14px;width:10px;height:10px;background:#111;border-radius:999px;transform:translateX(-50%);}
.ce-dot-cefr{position:absolute;top:15px;width:10px;height:10px;background:#fff;border:2px solid #9ca3af;border-radius:999px;transform:translateX(-50%);}
.ce-lbl{position:absolute;top:-2px;transform:translateX(-50%);font-size:.70rem;color:#374151;background:#fff;padding:0 4px;border-radius:4px;}
.pills{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px;}
.pill{font-size:.75rem;padding:2px 8px;border-radius:999px;border:1px solid #e5e7eb;background:#f9fafb;color:#374151;}
.pill-up{background:#ecfdf5;border-color:#10b981;color:#065f46;}
.pill-down{background:#fee2e2;border-color:#ef4444;color:#7f1d1d;}
.pill-close{background:#fef3c7;border-color:#f59e0b;color:#78350f;}
.badge{font-size:.72rem;padding:2px 6px;border-radius:999px;background:#eef2f7;color:#374151;border:1px solid #e5e7eb;}

.lvl-chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px;}
/* neutral chip */
.chip{
  display:inline-block; padding:2px 8px; border-radius:999px;
  border:1px solid #e5e7eb; background:#f3f4f6; color:#374151; font-size:.78rem;
  line-height:1.6;
}
/* nearest level (closest mean) */
.chip-near{
  background:#dcfce7; border-color:#10b981; color:#065f46;
}
/* levels lower than Students’ mean (helpful “strengths”) */
.chip-lower{
  background:#ecfdf5; border-color:#a7f3d0; color:#065f46;
}
/* hover */
.chip:hover{filter:brightness(.97)}

</style>
    """,
    unsafe_allow_html=True,
)


    def _pos(v, lo, hi):
        if np.isnan(v) or hi <= lo: return 0.0
        p = 100.0 * (v - lo) / (hi - lo)
        return 0.0 if p < 0 else (100.0 if p > 100 else p)

    def _closest_level(stu_mean, levels):
        if np.isnan(stu_mean) or not levels: return None
        lvl, _ = min(((L, abs(stu_mean - mu)) for (L, mu, _) in levels), key=lambda x: x[1])
        return lvl

    # Filter Students by COURSE if provided
    students_all = df_base[df_base["Group"] == 0].copy()
    students = (
        students_all[students_all["COURSE"].astype(str).isin(courses_filter)]
        if courses_filter else students_all
    )

    cefr_tbl = _normalize_cefr_table(cefr_ref)
    feats_ordered = [f for f in feats_in_sub if f in students.columns]
    level_order = ["A1", "A2", "B1", "B2", "C1", "C2"]

    # Force 2-up layout (last row may have 1 card)
    for i in range(0, len(feats_ordered), 2):
        n_remaining = min(2, len(feats_ordered) - i)
        cols = st.columns(n_remaining)
        for col, feat in zip(cols, feats_ordered[i:i + n_remaining]):
            s = pd.to_numeric(students[feat], errors="coerce").dropna()
            if s.empty:
                with col: st.info(f"No data for {DISPLAY_OVERRIDES.get(feat, feat)} in current cohort filter.")
                continue

            n = int(s.size)
            mean_stu = float(s.mean())
            sd_stu = float(s.std(ddof=0))
            mn, mx = float(s.min()), float(s.max())

            rows_f = cefr_tbl[cefr_tbl["feature"] == feat].copy() if cefr_tbl is not None else pd.DataFrame()
            levels = []
            if not rows_f.empty:
                for L in level_order:
                    r = rows_f[rows_f["level"] == L]
                    if r.empty: continue
                    mu = float(r["mean"].iloc[0])
                    sd_lvl = float(r["sd"].iloc[0]) if "sd" in r.columns and not pd.isna(r["sd"].iloc[0]) else np.nan
                    levels.append((L, mu, sd_lvl))

            lo, hi = (mn, mx)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo = (mean_stu - 0.5) if np.isfinite(mean_stu) else 0.0
                hi = (mean_stu + 0.5) if np.isfinite(mean_stu) else 1.0

            disp = st.session_state.feat_docs.get(feat, {}).get("display", DISPLAY_OVERRIDES.get(feat, feat))
            defin = (st.session_state.feat_docs.get(feat, {}).get("definition", "") or "").strip()
            closest = _closest_level(mean_stu, levels)

            # Top-right pill (hover simplified language)
            badge_html = (
                f'<span class="badge" title="Nearest CEFR level to your Students’ average on this metric.">{closest}</span>'
                if closest else ""
            )

            with col:
                st.markdown('<div class="ce-card">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="ce-head"><div class="ce-title">{html.escape(disp)}</div>'
                    f'<div class="ce-meta">n {n} &nbsp;|&nbsp; Mean {format_num(mean_stu,1)} &nbsp;|&nbsp; '
                    f'SD {format_num(sd_stu,1)} &nbsp;|&nbsp; Min {format_num(mn,1)} &nbsp;|&nbsp; Max {format_num(mx,1)} &nbsp; {badge_html}</div></div>',
                    unsafe_allow_html=True
                )
                if defin:
                    st.markdown(f'<div class="ce-desc">{html.escape(defin)}</div>', unsafe_allow_html=True)

                stu_pos = _pos(mean_stu, lo, hi)

                # Dashed band with hover text (simple language)
                band_html = ""
                if levels and closest:
                    mu_cl = [mu for (L, mu, _) in levels if L == closest][0]
                    sd_cl = [sd for (L, _, sd) in levels if L == closest][0]
                    if not (np.isnan(sd_cl) or sd_cl == 0):
                        left = _pos(mu_cl - sd_cl, lo, hi)
                        right = _pos(mu_cl + sd_cl, lo, hi)
                        width = max(0.0, right - left)
                        if width > 0:
                            band_html = (
                                f'<div class="ce-band" style="left:{left}%;width:{width}%"; '
                                f'title="Blue dashed band: ±1 SD around the nearest CEFR level’s mean."></div>'
                            )

                dots_html, lbls_html = [], []
                for lvl, mu, _sd in levels:
                    pos = _pos(mu, lo, hi)
                    dots_html.append(f'<div class="ce-dot-cefr" style="left:{pos}%;" title="{lvl} mean: {format_num(mu,1)}"></div>')
                    lbls_html.append(f'<div class="ce-lbl" style="left:{pos}%;" title="{lvl} mean">{lvl}</div>')

                st.markdown(
                    f'<div class="ce-bwrap">'
                    f'  <div class="ce-track"></div>'
                    f'  {band_html}'
                    f'  <div class="ce-dot-stu" style="left:{stu_pos}%;" title="Students mean: {format_num(mean_stu,1)}"></div>'
                    f'  {"".join(dots_html)}{"".join(lbls_html)}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Simplified colored CEFR chips (no Δ): nearest = green, lower-than-students = light-green, higher = grey
                chip_elems = []
                for lvl, mu, _sd in levels:
                    cls = "chip"
                    title = f"{lvl} mean: {format_num(mu,1)}"
                    if lvl == closest:
                        cls = "chip chip-near"
                        title = f"{lvl} (nearest to Students’ mean): {format_num(mu,1)}"
                    elif mu < mean_stu:
                        cls = "chip chip-lower"
                        title = f"{lvl} (lower than Students’ mean): {format_num(mu,1)}"
                    chip_elems.append(f'<span class="{cls}" title="{html.escape(title)}">{lvl}: {format_num(mu,1)}</span>')

                if chip_elems:
                    st.markdown(f'<div class="lvl-chips">{" ".join(chip_elems)}</div>', unsafe_allow_html=True)


                st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# ======================= LAYOUT (TABS) ======================
# ============================================================
st.title("Writing Analytics for Classrooms")

df_raw = st.session_state.df_raw
df_agg = st.session_state.df_agg
cefr_ref_global = st.session_state.cefr
examples_df_global = st.session_state.examples

if df_raw is None or df_agg is None:
    st.info("Upload the CSVs (or quick-load a folder) on the left and click **Load / Reload**.")
    st.stop()

ensure_feature_docs(df_agg)

tab_tutorial, tab_ai, tab_cefr = st.tabs(["Tutorial", "Comparison: Students vs AI", "Profiling of Writing Proficiency"])


# --------------------- TAB 1: TUTORIAL --------------------

# Call this before rendering the tutorial/narrative/cards section
def inject_tutorial_dark_css():
    if not is_dark_theme():
        return
    st.markdown("""
<style>
/* Tutorial / metric cards */
.tut-card, .metric-card {
  background:#111827 !important;                 /* slate-900 */
  color:#e5e7eb !important;                       /* gray-200 */
  border:1px solid rgba(255,255,255,0.15) !important;
}

/* Titles inside cards */
.tut-card h4, .metric-card h4, .metric-title {
  color:#e5e7eb !important;
  opacity:1 !important;                           /* was too faint */
}

/* Body copy inside cards */
.tut-card p, .metric-card p, .metric-desc, .metric-note {
  color:#e5e7eb !important;
  opacity:.95 !important;
}

/* “muted” helper text (but keep readable) */
.tut-card .muted, .metric-card .muted {
  color:#cbd5e1 !important;                       /* slate-300 */
  opacity:.9 !important;
}

/* CEFR level pills / chips */
.level-pill, .level-chip, .cefr-pill {
  background:rgba(255,255,255,.08) !important;
  color:#e5e7eb !important;
  border:1px solid rgba(255,255,255,.20) !important;
}

/* Active/nearest pill highlight */
.level-pill.active, .level-chip.active, .cefr-pill.active {
  background:rgba(16,185,129,.18) !important;     /* emerald */
  border-color:rgba(16,185,129,.45) !important;
  color:#34d399 !important;
}

/* Small badges with numbers */
.badge, .mini-badge {
  background:rgba(255,255,255,.10) !important;
  color:#e5e7eb !important;
  border-color:rgba(255,255,255,.18) !important;
}

/* If you dimmed titles globally via opacity, undo that in dark mode */
.tut-dim, .metric-dim { opacity:1 !important; }
</style>
    """, unsafe_allow_html=True)

# use it:
inject_tutorial_dark_css()


with tab_tutorial:
    render_background_panel("tutorial")
    render_prompt_panel("tutorial")

    st.subheader("Tutorial Texts — Building Intuition for the Metrics")
    st.markdown(
        """
- Use the selectors below to **choose a feature *family* and *subcategory***.
- We provide two example texts for illustration. **Pick an example text** to view its profile.
        """
    )

    examples_df = st.session_state.examples
    if examples_df is None or examples_df.empty:
        st.error("No Example texts CSV loaded. Please add it in the sidebar (independent exemplars).")
        st.stop()

    # ---- Styling for highlights (tutorial only) ----
    st.markdown(
        """
<style>
.hl-sub{ background:#DCFCE7; padding:0 2px; border-radius:4px; }
.hl-coord{ background:#DBEAFE; padding:0 2px; border-radius:4px; }
.hl-punc{ background:#E5E7EB; padding:0 2px; border-radius:3px; }
.hl-long{ background:#FEF3C7; padding:0 2px; border-radius:4px; box-shadow:inset 0 -1px 0 #F59E0B; }
.hl-short{ background:#FEE2E2; padding:0 2px; border-radius:4px; box-shadow:inset 0 -1px 0 #EF4444; }
.hl-np{ background:#F3E8FF; padding:0 2px; border-radius:4px; }
.hl-of{ background:#EDE9FE; padding:0 2px; border-radius:4px; }
.hl-content{ background:#E0F2FE; padding:0 2px; border-radius:4px; }
.hl-type{ text-decoration: underline; text-decoration-thickness: 3px; text-underline-offset: 3px; text-decoration-color:#A7F3D0; }
.hl-rare{ background:#FDE68A; padding:0 2px; border-radius:4px; }
.ex-card{ border:1px solid #e5e7eb; border-radius:14px; padding:12px 14px; background:#fff; }
.ex-title{ font-weight:700; font-size:0.95rem; margin-bottom:6px; }
.ex-meta{ color:#6B7280; font-size:.85rem; margin-bottom:4px; }
.ex-body{ white-space:pre-wrap; line-height:1.55; font-size:0.95rem; color:#111827; background:#fafafa; padding:10px 12px; border-radius:10px; border:1px solid #e5e7eb;}
.count-chip{ display:inline-block; font-size:.8rem; background:#f5f5f5; color:#374151; padding:4px 8px; border-radius:999px; margin:6px 6px 0 0; border:1px solid #e5e7eb;}
</style>
        """,
        unsafe_allow_html=True,
    )

    import html as _html

    # ---- Subcategory helpers (unchanged logic) ----
    SUBCAT_ALIASES = {
        "length of production unit": {"length of production unit", "length"},
        "coordination": {"coordination"},
        "subordination": {"subordination"},
        "sentence complexity": {"sentence complexity"},
        "noun phrase complexity": {"noun phrase complexity", "nominals", "noun phrase"},
        "lexical density": {"lexical density", "density"},
        "lexical diversity": {"lexical diversity", "diversity"},
        "lexical sophistication": {"lexical sophistication", "sophistication", "frequency bands", "coca frequency", "dispersion", "register signatures"},
    }
    def canon_subcat(lbl: str) -> str:
        s = (lbl or "").strip().lower()
        for k, al in SUBCAT_ALIASES.items():
            if s in al: return k
        return s

    def sentence_bounds(t: str):
        spans, start = [], 0
        for m in re.finditer(r"[.!?]+[\"'’”)\]]*\s*|\Z", t, flags=re.MULTILINE):
            end = m.end()
            if end > start: spans.append((start, end))
            start = end
        return spans

    STOP = set("""
        a an the and or but if while as of for with to in on by at from into over after before
        is am are was were be been being do does did have has had will would can could should may might must
        i you he she it we they me him her us them my your his its our their this that these those not no
    """.split())
    COORDINATORS = ["and","but","or","nor","so","yet"]
    SUBORDINATORS = [
        "although","though","because","since","while","whereas","unless",
        "if","when","whenever","after","before","once","until",
        "which","who","whom","whose","where","why"
    ]

    def token_spans(text: str, tokens: list[str]):
        spans = []
        for tok in tokens:
            pat = re.compile(rf'(?<![A-Za-z0-9’\'-]){re.escape(tok)}(?![A-Za-z0-9’\'-])', re.IGNORECASE)
            for m in pat.finditer(text):
                spans.append((m.start(), m.end()))
        return spans

    def merge_intervals(ints: list[tuple[int,int]]):
        if not ints: return []
        ints = sorted(ints, key=lambda x: (x[0], x[1]))
        merged = [list(ints[0])]
        for s,e in ints[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s,e])
        return [(a,b) for a,b in merged]

    def build_html_with_spans(raw_text: str, spans: list[dict]):
        spans = sorted([s for s in spans if 0 <= s["start"] < s["end"] <= len(raw_text)],
                       key=lambda s: (s["start"], s["end"]))
        out, last = [], 0
        for s in spans:
            if s["start"] > last: out.append(_html.escape(raw_text[last:s["start"]]))
            elif s["start"] < last: continue
            out.append(
                f'<span class="{s["cls"]}" title="{_html.escape(s.get("title",""))}">{_html.escape(raw_text[s["start"]:s["end"]])}</span>'
            )
            last = s["end"]
        out.append(_html.escape(raw_text[last:]))
        return "".join(out)

    def highlight_for_subcategory(subcat_label: str, raw_text: str, emphasize_high: bool):
        spans = []; chips = {}
        s = canon_subcat(subcat_label)

        # Detect greeting / closing once; we’ll avoid highlighting inside them
        greet_m = GREET_RE.match(raw_text or "")
        greet_span = (greet_m.start(), greet_m.end()) if greet_m else None

        close_m = CLOSING_RE.search(raw_text or "")
        closing_span = (close_m.start(), close_m.end()) if close_m else None

        def _overlaps_excluded(a: int, b: int) -> bool:
            if greet_span and not (b <= greet_span[0] or a >= greet_span[1]): return True
            if closing_span and not (b <= closing_span[0] or a >= closing_span[1]): return True
            return False

        # ---- LENGTH OF PRODUCTION UNIT (long/short sentences) ----
        if s == "length of production unit":
            s_spans = sentence_bounds(raw_text)
            if s_spans:
                # word counts per sentence
                lens = [(i, len(re.findall(r"[A-Za-z0-9’'-]+", raw_text[a:b]))) for i,(a,b) in enumerate(s_spans)]
                # Simple, robust thresholds
                wc_vals = [w for _, w in lens]
                long_thr  = max(18, np.percentile(wc_vals, 75))   # “long”
                short_thr = min(7,  np.percentile(wc_vals, 25))   # “short”

                def _clip_salutation(a, b):
                    # If the sentence starts at the very beginning and overlaps "Dear ...,"
                    if greet_span and a <= greet_span[0] and b > greet_span[1]:
                        return (greet_span[1], b)  # start *after* the greeting
                    return (a, b)

                MAX_HL = 3  # don’t overpaint; keeps the tutorial readable

                if emphasize_high:
                    # pick the longest sentences first
                    cand = [i for i, w in sorted(lens, key=lambda t: -t[1]) if w >= long_thr]
                else:
                    # pick the shortest sentences first
                    cand = [i for i, w in sorted(lens, key=lambda t: t[1]) if w <= short_thr]

                picked = 0
                for i in cand:
                    a, b = s_spans[i]
                    a, b = _clip_salutation(a, b)
                    # guard: need a meaningful chunk and not inside excluded regions
                    if b - a < 12:        # avoid tiny snippets like “Dear …,”
                        continue
                    if _overlaps_excluded(a, b):
                        continue
                    spans.append({"start": a, "end": b, "cls": ("hl-long" if emphasize_high else "hl-short"),
                                "title": ("Long sentence" if emphasize_high else "Short sentence")})
                    picked += 1
                    if picked >= MAX_HL:
                        break
                chips["Long sentences" if emphasize_high else "Short sentences"] = picked

        # ---- COORDINATION ----
        elif s == "coordination":
            COORDINATORS = ["and","but","or","nor","so","yet"]
            # highlight only true tokens, not parts of hyphenated compounds
            w_sp = token_spans(raw_text, COORDINATORS)
            w_sp = [(a,b) for a,b in w_sp if not _overlaps_excluded(a,b)]
            # commas often flank coordination but don’t overpaint all commas
            spans += [{"start": a, "end": b, "cls": "hl-coord", "title": "Coordinator"} for (a, b) in w_sp]
            chips["Coordinators"] = len(w_sp)

        # ---- SUBORDINATION ----
        elif s == "subordination":
            SUBORDINATORS = [
                "although","though","because","since","while","whereas","unless",
                "if","when","whenever","after","before","once","until",
                "which","who","whom","whose","where","why"
            ]
            w_sp = token_spans(raw_text, SUBORDINATORS)
            w_sp = [(a,b) for a,b in w_sp if not _overlaps_excluded(a,b)]
            spans += [{"start": a, "end": b, "cls": "hl-sub", "title": "Subordinator/relativizer"} for (a, b) in w_sp]
            chips["Subordinators/relativizers"] = len(w_sp)

        # ---- SENTENCE COMPLEXITY (a little of both, plus strong punctuation) ----
        elif s == "sentence complexity":
            SUBORDINATORS = [
                "although","though","because","since","while","whereas","unless",
                "if","when","whenever","after","before","once","until",
                "which","who","whom","whose","where","why"
            ]
            COORDINATORS = ["and","but","or","nor","so","yet"]
            sub_sp = [(a,b) for (a,b) in token_spans(raw_text, SUBORDINATORS) if not _overlaps_excluded(a,b)]
            coo_sp = [(a,b) for (a,b) in token_spans(raw_text, COORDINATORS)    if not _overlaps_excluded(a,b)]
            p_sp   = [(m.start(), m.end()) for m in re.finditer(r"[;:]", raw_text)]
            p_sp   = [(a,b) for (a,b) in p_sp if not _overlaps_excluded(a,b)]
            spans += [{"start": a, "end": b, "cls": "hl-sub",   "title": "Subordinator"} for (a, b) in sub_sp]
            spans += [{"start": a, "end": b, "cls": "hl-coord", "title": "Coordinator"} for (a, b) in coo_sp]
            spans += [{"start": a, "end": b, "cls": "hl-punc",  "title": "Clause-linking punctuation"} for (a, b) in p_sp]
            chips["Subordinators"] = len(sub_sp); chips["Coordinators"] = len(coo_sp); chips["Punct (;:)"] = len(p_sp)

        # ---- NOUN PHRASE COMPLEXITY (simple heuristics; keep light) ----
        elif s == "noun phrase complexity":
            # “of” + noun and simple pre-modified NPs; skip greeting/closing lines
            of_sp = [(m.start(), m.end()) for m in re.finditer(r"(?i)\bof\b\s+\b[\w’'-]+\b", raw_text)]
            of_sp = [(a,b) for (a,b) in of_sp if not _overlaps_excluded(a,b)]
            np_sp = [(m.start(), m.end()) for m in re.finditer(
                r"(?i)\b(the|a|an|this|that|these|those|my|your|his|her|its|our|their)\s+(?:\w+ly\s+)?[\w’'-]+\s+[\w’'-]+\b",
                raw_text)]
            np_sp = [(a,b) for (a,b) in np_sp if not _overlaps_excluded(a,b)]
            spans += [{"start": a, "end": b, "cls": "hl-of",  "title": "of-phrase (post-mod)"} for (a, b) in of_sp]
            spans += [{"start": a, "end": b, "cls": "hl-np",  "title": "NP pre-mod"} for (a, b) in np_sp]
            chips["of-phrases"] = len(of_sp); chips["NP pre-mods (~)"] = len(np_sp)

        # ---- LEXICAL DENSITY (content words) ----
        elif s == "lexical density":
            tokens = list(re.finditer(r"\b[\w’'-]+\b", raw_text))
            total = len(tokens); content_ct = 0
            for m in tokens:
                if m.group(0).lower() not in STOP and not _overlaps_excluded(m.start(), m.end()):
                    spans.append({"start": m.start(), "end": m.end(), "cls": "hl-content", "title": "Content word"})
                    content_ct += 1
            pct = (content_ct / total * 100.0) if total else 0.0
            chips["Content words"] = f"{content_ct}/{total} ({pct:.1f}%)"

        # ---- LEXICAL DIVERSITY (types / hapax underlines) ----
        elif s == "lexical diversity":
            toks = list(re.finditer(r"\b[\w’'-]+\b", raw_text))
            N = len(toks); low = [m.group(0).lower() for m in toks]
            by_type = {}
            for i, w in enumerate(low): by_type.setdefault(w, []).append(i)
            U = len(by_type)
            hapax_idx = [idxs[0] for idxs in by_type.values() if len(idxs) == 1]
            # Show only a handful to avoid overpainting
            for i in hapax_idx[: max(1, N // 12)]:
                m = toks[i]
                if not _overlaps_excluded(m.start(), m.end()):
                    spans.append({"start": m.start(), "end": m.end(), "cls": "hl-type", "title": "Unique token (type)"})
            chips["Types/Tokens (TTR)"] = f"{U}/{N} ({(U / N * 100.0 if N else 0):.1f}%)"

        # ---- LEXICAL SOPHISTICATION (long/morphologically complex) ----
        elif s == "lexical sophistication":
            long_i  = [(m.start(), m.end()) for m in re.finditer(r"\b[\w’'-]{10,}\b", raw_text)]
            morph_i = [(m.start(), m.end()) for m in re.finditer(
                r"(?i)\b[\w’'-]+(tion|sion|ment|ness|ity|ance|ence|ship|ability|ism|ology)\b", raw_text)]
            merged = merge_intervals([p for p in (long_i + morph_i) if not _overlaps_excluded(*p)])
            for a, b in merged:
                spans.append({"start": a, "end": b, "cls": "hl-rare", "title": "Long/complex word"})
            chips["Sophisticated tokens (≈)"] = len(merged)

        html_text = build_html_with_spans(raw_text, spans)
        return html_text, chips


    # ---- Feature selectors (now placed after explainer) ----
    base_tut = df_agg
    family_tut, subcat_tut, feats_in_sub_tut = render_shared_controls(base_tut, prefix="tut")
    examples_df = st.session_state.examples
    feats_here = [f for f in feats_in_sub_tut if f in examples_df.columns]

    # ---- Determine the two curated examples and which is "more complex" ----
    ex_ids = examples_df["TID"].astype(str).tolist()
    if len(ex_ids) < 2:
        st.error("Need at least two curated example texts in the examples CSV.")
        st.stop()

    exA = examples_df.loc[examples_df["TID"].astype(str) == ex_ids[0]].iloc[0]
    exB = examples_df.loc[examples_df["TID"].astype(str) == (ex_ids[1] if len(ex_ids) > 1 else ex_ids[0])].iloc[0]
    textA = str(exA.get("text", "") or "")
    textB = str(exB.get("text", "") or "")

    def mean_over_feats(row, feats):
        vals = [float(row.get(f)) for f in feats if pd.notna(row.get(f))]
        return np.nan if not vals else float(np.mean(vals))

    exA_mean = mean_over_feats(exA, feats_here)
    exB_mean = mean_over_feats(exB, feats_here)
    A_is_higher = bool(pd.notna(exA_mean) and pd.notna(exB_mean) and exA_mean >= exB_mean)

    # Decide which example is "More complex" vs "Less complex" based on the subcategory metrics
    if A_is_higher:
        choices = {
            "Example 1: More complex": ("A", exA, textA, True),
            "Example 2: Less complex": ("B", exB, textB, False),
        }
    else:
        choices = {
            "Example 1: More complex": ("B", exB, textB, True),
            "Example 2: Less complex": ("A", exA, textA, False),
        }

    sel_label = st.radio(
        "Choose an example",
        options=list(choices.keys()),
        horizontal=True,
        key="tut_example_pick",
    )
    sel_tag, sel_row, sel_text, emphasize_high = choices[sel_label]

    # Build highlights for the selected example
    html_sel, chips_sel = highlight_for_subcategory(subcat_tut, sel_text, emphasize_high)

    # --- Make the selection available to later blocks (narrative + metric cards) ---
    ex_row = sel_row      # use ex_row below when reading metric values
    text_sel = sel_text   # if you need the raw text later

    # --- Render the single selected example card ---
    st.markdown(
        f"""
    <div class="ex-card">
    <div class="ex-title">{_html.escape(sel_label)}</div>
    <div class="ex-meta">TID: <code>{_html.escape(str(ex_row.get("TID","")))}</code></div>
    <div class="ex-body">{html_sel}</div>
    {"".join(
        f'<span class="count-chip">{_html.escape(str(k))}: {_html.escape(str(v))}</span>'
        for k, v in (chips_sel or {}).items() if v not in (None, "")
    )}
    </div>
        """,
        unsafe_allow_html=True
    )

    # === Overview of Metrics Scores of this Subcategory (reference-based) ===
    st.subheader("Overview of Metrics Scores of this Subcategory")
    st.caption("Each card shows the selected example’s score for a metric in this subcategory, plus reference means if available.")

    # 1) Reference table (if provided)
    cefr_tbl = _normalize_cefr_table(st.session_state.cefr)
    level_order = ["A1", "A2", "B1", "B2", "C1", "C2"]

    # 2) Narrative, teacher-friendly summary (no explicit CEFR labels; uses nearest reference mean per metric)
    
    # 2) Narrative, teacher-friendly summary (compare selected vs other; no CEFR)
    # Identify the "other" example
    other_row = exB if sel_tag == "A" else exA

    def disp_name(f):
        fd = st.session_state.feat_docs.get(f, {})
        return fd.get("display", DISPLAY_OVERRIDES.get(f, _pretty_label(f)))

    def _vals(row, feats):
        out = {}
        for f in feats:
            v = pd.to_numeric(row.get(f), errors="coerce")
            if pd.notna(v):
                out[f] = float(v)
        return out

    sel_vals = _vals(ex_row, feats_here)
    oth_vals = _vals(other_row, feats_here)

    diffs = []
    for f in set(sel_vals.keys()) & set(oth_vals.keys()):
        vs, vo = sel_vals[f], oth_vals[f]
        scale = max(abs(vs), abs(vo), 1.0)  # avoids divide-by-zero, normalizes across units
        delta = vs - vo
        rel = delta / scale
        if abs(rel) >= 0.10:   # ≥10% difference = meaningfully different
            tag = "higher" if delta > 0 else "lower"
        else:
            tag = "similar"
        diffs.append({"feat": f, "vs": vs, "vo": vo, "delta": delta, "abs_rel": abs(rel), "tag": tag})

    n = len(diffs)
    n_hi = sum(d["tag"] == "higher" for d in diffs)
    n_lo = sum(d["tag"] == "lower" for d in diffs)
    n_md = sum(d["tag"] == "similar" for d in diffs)

    # Top movers (by relative size) to keep text concise
    top_hi = sorted([d for d in diffs if d["tag"] == "higher"], key=lambda d: -d["abs_rel"])[:3]
    top_lo = sorted([d for d in diffs if d["tag"] == "lower"], key=lambda d: -d["abs_rel"])[:3]
    hi_list = ", ".join(disp_name(d["feat"]) for d in top_hi) if top_hi else "—"
    lo_list = ", ".join(disp_name(d["feat"]) for d in top_lo) if top_lo else "—"

    # Simple, subcategory-aware hint
    sub = (subcat_tut or "").strip().lower()
    hints = {
        "length of production unit": {
            "higher": "It tends toward longer sentences or clauses.",
            "lower": "It uses shorter, more concise sentences.",
            "intermediate": "It keeps sentence length moderate."
        },
        "sentence complexity": {
            "higher": "It links ideas with more clause-level structure.",
            "lower": "It keeps clause linking minimal.",
            "intermediate": "It uses moderate clause linking."
        },
        "subordination": {
            "higher": "It relies more on dependent clauses to qualify ideas.",
            "lower": "It relies less on dependent clauses.",
            "intermediate": "It uses some but not extensive subordination."
        },
        "coordination": {
            "higher": "It chains ideas with coordinators more frequently.",
            "lower": "It prefers simpler, unchained statements.",
            "intermediate": "It coordinates ideas occasionally."
        },
        "noun phrase complexity": {
            "higher": "It packs more detail into noun phrases (pre/post-modification).",
            "lower": "It keeps noun phrases lean.",
            "intermediate": "It adds detail selectively to noun phrases."
        },
        "lexical density": {
            "higher": "It leans on content words (nouns/verbs/adj/adv).",
            "lower": "It uses more function words relative to content words.",
            "intermediate": "It balances content and function words."
        },
        "lexical diversity": {
            "higher": "It uses a wider range of word types with less repetition.",
            "lower": "It repeats words more often.",
            "intermediate": "It shows moderate variety without extremes."
        },
        "lexical sophistication": {
            "higher": "It draws on longer and/or morphologically complex words.",
            "lower": "It prefers more common, shorter forms.",
            "intermediate": "It mixes common and more complex forms."
        },
    }
    is_more_complex = ("More complex" in sel_label)
    band = "higher" if is_more_complex else ("lower" if n_lo > n_hi else "intermediate")
    hint = hints.get(sub, {}).get(band, "")

    label_phrase = "a more complex example" if is_more_complex else "a less complex example"

    st.markdown(
        f"**Profile of the Text.** Across **{n}** metrics in **{subcat_tut}**, this text scores "
        f"**higher on {n_hi}**, **similar on {n_md}**, and **lower on {n_lo}** than the other curated text. "
        f"Taken together, this makes it **{label_phrase}** for this category. {hint} "
        f"**Largest advantages:** {hi_list}. "
        f"{'**Largest gaps:** ' + lo_list if lo_list != '—' else ''}"
    )

    # --- Legend + hover styles for tutorial CEFR chips/cards ---
    _dark = is_dark_theme()
    TIP_BG  = "rgba(17,24,39,.96)" if _dark else "rgba(17,24,39,.96)"
    TIP_TX  = "#e5e7eb" if _dark else "#f9fafb"
    TIP_BD  = "rgba(255,255,255,.28)" if _dark else "rgba(0,0,0,.12)"

    st.markdown(f"""
    <style>
    /* Card hover */
    .tut-metric-card {{
    border:1px solid #e5e7eb; border-radius:12px; padding:10px 12px;
    background:#fff; box-shadow:0 1px 2px rgba(0,0,0,.04);
    transition: box-shadow .08s ease, transform .08s ease;
    }}
    .tut-metric-card:hover {{ box-shadow:0 3px 10px rgba(0,0,0,.10); transform: translateY(-1px); }}

    /* CEFR chips with tooltips */
    .cefr-chip {{
    display:inline-block; margin:2px 6px 0 0; padding:2px 8px; border-radius:999px;
    border:1px solid #e5e7eb; font-size:.78rem; position:relative;
    transition: transform .06s ease, box-shadow .06s ease;
    }}
    .cefr-chip:hover {{ transform: translateY(-1px); box-shadow:0 1px 6px rgba(0,0,0,.12); }}

    .cefr-chip.near   {{ background:#dcfce7; border-color:#10b981; }}
    .cefr-chip.lower  {{ background:#ecfdf5; border-color:#a7f3d0; }}
    .cefr-chip.higher {{ background:#f3f4f6; border-color:#e5e7eb; }}

    /* Tooltip bubble */
    .cefr-chip::after {{
    content: attr(data-tip);
    position:absolute; left:50%; bottom:110%;
    transform: translateX(-50%) scale(.98);
    background:{TIP_BG}; color:{TIP_TX}; border:1px solid {TIP_BD};
    padding:6px 8px; font-size:.78rem; white-space:nowrap; border-radius:6px;
    opacity:0; pointer-events:none; transition: opacity .08s ease, transform .08s ease;
    z-index:5;
    }}
    .cefr-chip:hover::after {{ opacity:1; transform: translateX(-50%) scale(1); }}

    /* Legend */
    .cefr-legend {{ display:flex; flex-wrap:wrap; gap:18px; align-items:center; margin:6px 0 12px; }}
    .legend-item {{ display:inline-flex; align-items:center; gap:8px; color:#6b7280; font-size:.9rem; }}
    .swatch {{ width:14px; height:14px; border-radius:4px; border:1px solid #e5e7eb; }}
    .swatch.near   {{ background:#dcfce7; border-color:#10b981; }}
    .swatch.lower  {{ background:#ecfdf5; border-color:#a7f3d0; }}
    .swatch.higher {{ background:#f3f4f6; border-color:#e5e7eb; }}
    </style>

    <div class="cefr-legend">
    <div class="legend-item"><span class="swatch near"></span>
        <span><strong>Nearest</strong> CEFR mean to the selected value</span></div>
    <div class="legend-item"><span class="swatch lower"></span>
        <span>Lower CEFR levels than the nearest</span></div>
    <div class="legend-item"><span class="swatch higher"></span>
        <span>Higher CEFR levels than the nearest</span></div>
    </div>
    """, unsafe_allow_html=True)


    # 3) Metric cards (show selected text value + reference means for all levels)
    cols_cards = st.columns(3)
    for i, feat in enumerate(feats_here):
        fd = st.session_state.feat_docs.get(feat, {})
        display = fd.get("display", _display_name_for(feat))
        definition = (fd.get("definition") or "").strip()

        # Selected example's value for this metric
        v = pd.to_numeric(ex_row.get(feat), errors="coerce")

        # Gather reference means (if available) and find nearest
        means = []
        nearest_lvl = None
        if cefr_tbl is not None and not pd.isna(v):
            rows_f = cefr_tbl[cefr_tbl["feature"] == feat]
            if not rows_f.empty:
                for L in level_order:
                    r = rows_f[rows_f["level"] == L]
                    if not r.empty and pd.notna(r["mean"].iloc[0]):
                        means.append((L, float(r["mean"].iloc[0])))
                if means:
                    nearest_lvl = min(means, key=lambda t: abs(t[1] - float(v)))[0]


        with cols_cards[i % 3]:
            # Selected example's value for this metric
            v = pd.to_numeric(ex_row.get(feat), errors="coerce")

            # Reference means + nearest
            means = []
            nearest_lvl = None
            if cefr_tbl is not None and not pd.isna(v):
                rows_f = cefr_tbl[cefr_tbl["feature"] == feat]
                if not rows_f.empty:
                    for L in level_order:
                        r = rows_f[rows_f["level"] == L]
                        if not r.empty and pd.notna(r["mean"].iloc[0]):
                            means.append((L, float(r["mean"].iloc[0])))
                    if means:
                        nearest_lvl = min(means, key=lambda t: abs(t[1] - float(v)))[0]

            # Relation text for tooltips (vs selected value)
            def _rel(mu: float, vv: float) -> str:
                if not (pd.notna(mu) and pd.notna(vv)): return ""
                diff = mu - float(vv)
                rel = diff / max(abs(float(vv)), 1e-9)
                if abs(rel) <= 0.05: return "≈ similar to selected"
                return "higher than selected" if diff > 0 else "lower than selected"

            # Chips with classes + data-tip (hover bubble)
            if means:
                chips_html = ""
                for L, mu in means:
                    if nearest_lvl == L:
                        cls = "cefr-chip near"
                    elif nearest_lvl and level_order.index(L) < level_order.index(nearest_lvl):
                        cls = "cefr-chip lower"
                    else:
                        cls = "cefr-chip higher"
                    tip = f"{L} mean: {format_num(mu,1)} — {_rel(mu, v)}"
                    chips_html += f'<span class="{cls}" data-tip="{html.escape(tip)}">{L}: {format_num(mu,1)}</span>'
            else:
                chips_html = '<span style="font-size:.85rem;color:#6b7280;">No reference available for this metric</span>'

            # Pretty name + definition
            fd = st.session_state.feat_docs.get(feat, {})
            display = fd.get("display", _display_name_for(feat))
            definition = (fd.get("definition") or "").strip()

            # Card (now with hover class)
            st.markdown(
                f"""
        <div class="tut-metric-card">
        <div style="font-weight:600;">{html.escape(display)}</div>
        <div style="color:#374151;font-size:0.9rem;margin:4px 0 6px 0;">{html.escape(definition) if definition else "&nbsp;"}</div>
        <div style="font-size:0.9rem; margin-bottom:6px;">
            <strong>Selected text:</strong> {format_num(v, 1)}
        </div>
        <div>{chips_html}</div>
        </div>
                """,
                unsafe_allow_html=True,
            )


# --------------------- TAB 2: Students vs AI --------------------

with tab_ai:
    render_background_panel("ai")
    render_prompt_panel("ai")

    base_ai = df_agg

    # ---- Filter by COURSE (students only) ----
    courses_all = sorted(base_ai.loc[base_ai["Group"] == 0, "COURSE"].dropna().astype(str).unique())
    selected_courses_ai = st.multiselect(
        "Filter Students by COURSE",
        options=courses_all,
        default=courses_all,
        key="ai_courses",
    )

    # ---- Family / Subcategory controls and available metrics in subcat ----
    family_ai, subcat_ai, feats_in_sub_ai = render_shared_controls(base_ai, prefix="ai")

    st.markdown(
        '<div style="margin-top:18px;margin-bottom:6px;font-size:1.25rem;font-weight:800;">Students vs AI</div>',
        unsafe_allow_html=True,
    )

    if feats_in_sub_ai:
        # Sort codes by their pretty label, then show pretty names in the selector.
        feats_in_sub_ai = sorted(feats_in_sub_ai, key=lambda f: _display_name_for(f).lower())
        sel_metric_ai = feature_selectbox(
            "Choose a metric in this subcategory",
            feats_in_sub_ai,
            key="ai_metric",
        )  # returns the CODE; UI shows the pretty label

        # Pretty name + optional definition
        disp = _display_name_for(sel_metric_ai)
        fd = (st.session_state.get("feat_docs") or {}).get(sel_metric_ai, {})
        definition = (fd.get("definition") or "").strip()

        st.markdown(
            f"""
        <div style="border:1px solid #e5e7eb;border-radius:12px;padding:10px 12px;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.04);margin-bottom:8px;">
        <div style="font-weight:700;">{html.escape(disp)}</div>
        <div style="color:#374151;font-size:0.92rem;margin-top:4px;">{html.escape(definition) if definition else ""}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


        st.markdown("---")

        # ---- Build data for both groups ----
        students_all = base_ai[base_ai["Group"] == 0].copy()
        if selected_courses_ai:
            students = students_all[students_all["COURSE"].astype(str).isin(selected_courses_ai)]
        else:
            students = students_all
        ai_ref = base_ai[base_ai["Group"] == 1].copy()

        stu_plot = (
            students[["TID", "COURSE", sel_metric_ai]]
            .dropna(subset=[sel_metric_ai])
            .assign(Group="Students")
        )
        ai_plot = (
            ai_ref[["TID", "COURSE", sel_metric_ai]]
            .dropna(subset=[sel_metric_ai])
            .assign(Group="AI")
        )

        fig = go.Figure()

        def box_with_points(x_label: str, df_in: pd.DataFrame):
            if df_in.empty:
                return
            custom = np.stack(
                [
                    df_in["TID"].astype(str).values,
                    df_in.get("COURSE", pd.Series([""] * len(df_in))).astype(str).values,
                ],
                axis=1,
            )
            fig.add_trace(
                go.Box(
                    x=[x_label] * len(df_in),
                    y=df_in[sel_metric_ai],
                    name=x_label,
                    boxpoints="all",
                    jitter=0.30,
                    pointpos=0.0,
                    marker=dict(size=6, color="black", line=dict(color="white", width=0.6)),
                    line_color="black",
                    fillcolor="white",
                    customdata=custom,
                    hovertemplate=(
                        f"<b>{disp}</b><br>"
                        f"Group: {x_label}<br>"
                        "TID: %{customdata[0]}<br>"
                        "COURSE: %{customdata[1]}<br>"
                        "Value: %{y:.2f}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

        box_with_points("Students", stu_plot)
        box_with_points("AI", ai_plot)

        if stu_plot.empty:
            st.warning("No data available for **Students** with the current COURSE filter/metric.")
        if ai_plot.empty:
            st.warning("No data available for the **AI** reference group for this metric.")

        fig.update_layout(
            template="plotly_white",
            height=460,
            margin=dict(l=50, r=20, t=20, b=40),
            xaxis=dict(tickmode="array", tickvals=["Students", "AI"], ticktext=["Students", "AI"]),
            yaxis_title=disp,
        )
        st.plotly_chart(fig, use_container_width=True)


# --------------------- TAB 3: CEFR Profiling -------------------

with tab_cefr:
    render_background_panel("cefr")
    render_prompt_panel("cefr")

    base_cefr = df_agg  # TID-level
    cefr_ref = cefr_ref_global

    # --- Theme-aware colors ---
    _dark = is_dark_theme()
    STICKY_BG     = "#0f172a" if _dark else "#ffffff"
    STICKY_BORDER = "rgba(255,255,255,0.20)" if _dark else "#e5e7eb"
    CHIP_DOT_BORD = "rgba(255,255,255,0.15)" if _dark else "#e5e7eb"
    PLOT_TEMPLATE = "plotly_dark" if _dark else "plotly_white"
    CEFR_LINE     = "rgba(210,210,210,0.85)" if _dark else "rgba(90,90,90,0.85)"
    CEFR_BADGE_BG = "rgba(17,24,39,0.80)" if _dark else "rgba(255,255,255,0.80)"
    CEFR_BADGE_TX = "#e5e7eb" if _dark else "#374151"
    CEFR_BADGE_BD = "rgba(255,255,255,0.25)" if _dark else "rgba(0,0,0,0.12)"
    BOX_LINE      = "rgba(229,231,235,0.95)" if _dark else "rgba(17,24,39,0.85)"

    # Sticky "Cohort" strip (theme-aware)
    st.markdown(
        f"""
<style>
.sticky-strip{{position:sticky; top:0; z-index:20; background:{STICKY_BG};
  border-bottom:1px solid {STICKY_BORDER}; padding:8px 6px; margin:-10px 0 8px 0;}}
.strip-row{{display:flex; gap:10px; flex-wrap:wrap; align-items:center; font-size:.92rem;}}
.strip-badge{{display:inline-block;padding:4px 8px;border-radius:999px;background:rgba(127,127,127,0.08);
  border:1px solid {STICKY_BORDER};}}
.chips-legend{{display:flex; gap:16px; align-items:center; margin:6px 0 0 0;}}
.chip-dot{{display:inline-block; width:10px; height:10px; border-radius:50%; border:1px solid {CHIP_DOT_BORD};}}
</style>
        """,
        unsafe_allow_html=True,
    )

    # Subgroup (COURSE) picker for Students + feature selectors
    courses_all = sorted(base_cefr.loc[base_cefr["Group"] == 0, "COURSE"].dropna().astype(str).unique())
    c1, c2 = st.columns([2, 2])
    with c1:
        selected_courses = st.multiselect(
            "Students subgroup (COURSE)", options=courses_all, default=courses_all, key="cefr_courses"
        )
    with c2:
        family_cefr, subcat_cefr, feats_in_sub_cefr = render_shared_controls(base_cefr, prefix="cefr", show_pills=False)

    # Sticky strip summary line (remove “up to 6” wording)
    cohort_label = "Students"
    if selected_courses and set(selected_courses) != set(courses_all):
        cohort_label = f"Students COURSE: {', '.join(selected_courses)}"
    subline = f"{family_cefr} → {subcat_cefr} (click chips to add/remove metrics)"
    st.markdown(
        f'<div class="sticky-strip"><div class="strip-row">'
        f'<span class="strip-badge">{html.escape(cohort_label)}</span>'
        f'<span>{html.escape(subline)}</span>'
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # --- Feature chip picker (default 6 shown, unlimited selectable) ---
    chip_key = f"cefr_chip_sel::{family_cefr}::{subcat_cefr}"

    visible_feats = features_in_subcategory(base_cefr, family_cefr, subcat_cefr)

    # Seed selection (first 6) only if nothing is saved yet
    saved = st.session_state.get(chip_key, [])
    saved_clean = [f for f in saved if f in visible_feats]
    if not saved_clean:
        saved_clean = visible_feats[:6]
    st.session_state[chip_key] = saved_clean

    # Render chips with no limit
    selected_feats = render_feature_chip_picker(
        visible_feats,
        chip_key,
        max_sel=None,   # unlimited
        caption="Click metrics to (de)select for the overview."
    )
    selected_feats = [f for f in (selected_feats or []) if f in visible_feats]
    st.session_state[chip_key] = selected_feats

    if selected_feats:
        st.markdown("### Results Overview (all selected metrics)")
        st.caption(
            "Stats for **Students** cohort. Bars bounded by Students’ min–max. "
            "● = Students mean; ◯ = CEFR means; shaded band = ±1 SD around closest CEFR mean; "
            "dashed tick = cohort median."
        )
        render_cefr_subcategory_overview(base_cefr, selected_feats, cefr_ref, selected_courses)
        st.markdown("---")
    else:
        st.info("Select one or more metrics to show in the overview.")
        st.stop()

    # ===== Inspect Feature (Group-Level) =====
    st.markdown(
        '<div style="font-weight:800;font-size:1.25rem;margin:8px 0;">Inspect Feature (Group-Level)</div>',
        unsafe_allow_html=True,
    )

    # Three selectors: family → subcategory → metric
    icol1, icol2, icol3 = st.columns(3)
    family_ins = icol1.selectbox(
        "Feature family",
        options=["Sentence Structure", "Vocabulary"],
        key="inspect_family",
    )
    subs_ins = subcategories_for_family(base_cefr, family_ins)
    subcat_ins = icol2.selectbox(
        "Feature subcategory",
        options=subs_ins if subs_ins else ["—"],
        key="inspect_subcat",
    )
    feats_ins = features_in_subcategory(base_cefr, family_ins, subcat_ins)
    if not feats_ins:
        icol3.selectbox("Metric", options=["—"], key="inspect_metric", disabled=True)
        st.info("No numeric features available for this subcategory with current data and mapping.")
        st.stop()

    # Show renamed labels in dropdown, return code
    feat_cefr = feature_selectbox("Metric", feats_ins, key="inspect_metric")

    # Prepare cohort values (Students, with COURSE filter)
    cohort_all = base_cefr[base_cefr["Group"] == 0].copy()
    cohort = (
        cohort_all[cohort_all["COURSE"].astype(str).isin(selected_courses)]
        if selected_courses else cohort_all
    )

    series = pd.to_numeric(cohort[feat_cefr], errors="coerce").dropna()
    if series.empty:
        st.info("No values available for the selected metric in the human cohort.")
        st.stop()

    # Build plotting frame
    df_plot = pd.DataFrame({
        "_y": series.values,
        "TID": cohort.loc[series.index, "TID"].astype(str).values
    })
    df_plot["_pct"] = df_plot["_y"].rank(pct=True) * 100.0
    df_plot["_q"] = pd.cut(
        df_plot["_pct"], bins=[0, 25, 50, 75, 100],
        labels=[1, 2, 3, 4], include_lowest=True
    ).astype(int)

    # CEFR means table for guide lines
    cefr = cefr_ref_global
    if cefr is not None and not cefr.empty:
        ccols = {c.lower(): c for c in cefr.columns}
        if {"level", "feature", "mean", "sd"}.issubset(ccols.keys()):
            cefr = cefr.rename(columns={
                ccols["level"]: "level",
                ccols["feature"]: "feature",
                ccols["mean"]: "mean",
                ccols["sd"]: "sd",
            })
        else:
            cefr = None

    level_order = ["A1", "A2", "B1", "B2", "C1", "C2"]
    level_means = []
    if cefr is not None:
        rows_feat = cefr[cefr["feature"] == feat_cefr]
        for L in level_order:
            r = rows_feat[rows_feat["level"] == L]
            if not r.empty and pd.notna(r["mean"].iloc[0]):
                muL = float(r["mean"].iloc[0])
                sdL = float(r["sd"].iloc[0]) if "sd" in r.columns and not pd.isna(r["sd"].iloc[0]) else float("nan")
                level_means.append((L, muL, sdL))

    # Nearest CEFR per point (for hover & student card)
    if level_means:
        means_only = np.array([m for _, m, _ in level_means], dtype=float)
        lvls_only = [L for L, _, _ in level_means]
        nearest_idx = np.abs(df_plot["_y"].values.reshape(-1, 1) - means_only.reshape(1, -1)).argmin(axis=1)
        df_plot["_nearest_lvl"] = [lvls_only[i] for i in nearest_idx]
        df_plot["_nearest_mu"] = means_only[nearest_idx]
        df_plot["_delta_mu"]  = df_plot["_y"] - df_plot["_nearest_mu"]
    else:
        df_plot["_nearest_lvl"] = "—"
        df_plot["_nearest_mu"] = np.nan
        df_plot["_delta_mu"]   = np.nan

    # Colors
    COL_TOP = "#20A387"   # green
    COL_MID = "#3C8DFF"   # blue
    COL_LOW = "#E64B35"   # red

    # Legend chips
    st.markdown(
        f'''
<div class="chips-legend">
  <span><span class="chip-dot" style="background:{COL_TOP};"></span> Top 25%</span>
  <span><span class="chip-dot" style="background:{COL_MID};"></span> Middle 50%</span>
  <span><span class="chip-dot" style="background:{COL_LOW};"></span> Bottom 25%</span>
</div>
        ''',
        unsafe_allow_html=True
    )

    st.markdown("#### Distribution vs CEFR Means")

    # Quartiles, axis extents
    yvals = df_plot["_y"].astype(float).values
    q1 = float(np.percentile(yvals, 25))
    q2 = float(np.percentile(yvals, 50))
    q3 = float(np.percentile(yvals, 75))
    ymin = float(np.min(yvals))
    ymax = float(np.max(yvals))

    # Tight jitter
    rng = np.random.default_rng(42)
    df_plot["_x"] = (rng.random(len(df_plot)) - 0.5) * 0.04  # ±0.02 width

    # ---- Figure ----
    fig2 = go.Figure()

    # Background bands
    fig2.add_shape(type="rect", xref="paper", x0=0, x1=1, yref="y", y0=ymin, y1=q1,
                   fillcolor="rgba(230,75,53,0.22)", line=dict(width=0))
    fig2.add_shape(type="rect", xref="paper", x0=0, x1=1, yref="y", y0=q1, y1=q3,
                   fillcolor="rgba(60,141,255,0.18)", line=dict(width=0))
    fig2.add_shape(type="rect", xref="paper", x0=0, x1=1, yref="y", y0=q3, y1=ymax,
                   fillcolor="rgba(32,163,135,0.18)", line=dict(width=0))

    # Box (blue; no min/max caps)
    fig2.add_trace(go.Box(
        x=[0] * len(df_plot),
        y=df_plot["_y"],
        name="Current cohort (humans)",
        boxpoints=False,
        fillcolor="rgba(59,130,246,0.10)",
        line_color=BOX_LINE,
        whiskerwidth=0,           # remove caps
        showlegend=False
    ))

    # Colored whiskers (overlayed lines)
    fig2.add_trace(go.Scatter(x=[0, 0], y=[ymin, q1], mode="lines",
                              line=dict(color=COL_LOW, width=3),
                              hoverinfo="skip", showlegend=False))
    fig2.add_trace(go.Scatter(x=[0, 0], y=[q3, ymax], mode="lines",
                              line=dict(color=COL_TOP, width=3),
                              hoverinfo="skip", showlegend=False))

    # Points by quartile — hover WITHOUT Δ (fixed hovertemplate)
    fd2 = st.session_state.feat_docs.get(feat_cefr, {})
    disp = fd2.get("display", DISPLAY_OVERRIDES.get(feat_cefr, feat_cefr))

    def add_points(mask, color):
        sub = df_plot.loc[mask, ["TID", "_pct", "_q", "_nearest_lvl", "_nearest_mu", "_x", "_y"]].copy()
        sub["_pct"] = pd.to_numeric(sub["_pct"], errors="coerce")
        sub["_nearest_mu"] = pd.to_numeric(sub["_nearest_mu"], errors="coerce")

        custom = np.column_stack([
            sub["TID"].values,
            sub["_pct"].values,                 # 1
            sub["_q"].astype(int).values,       # 2
            sub["_nearest_lvl"].astype(str).values,  # 3
            sub["_nearest_mu"].values,          # 4
        ])

        fig2.add_trace(go.Scatter(
            x=sub["_x"], y=sub["_y"],
            mode="markers",
            marker=dict(size=7, color=color, line=dict(color="white", width=0.6)),
            customdata=custom,
            hovertemplate=(
                f"<b>%{{customdata[0]}}</b><br>"
                f"{html.escape(disp)}: %{{y:.2f}}<br>"
                "Percentile: %{customdata[1]:.1f}% (Q%{customdata[2]})<br>"
                "Nearest ref: %{customdata[3]} · %{customdata[4]:.2f}"
                "<extra></extra>"
            ),
            showlegend=False,
            selected=dict(marker=dict(size=10, opacity=1)),
            unselected=dict(marker=dict(opacity=0.25)),
        ))

    add_points(df_plot["_q"] == 4, COL_TOP)
    add_points(df_plot["_q"].isin([2, 3]), COL_MID)
    add_points(df_plot["_q"] == 1, COL_LOW)

    # CEFR dashed means — shorter and clearly labeled
    if level_means:
        for L, mu_L, _sd_L in level_means:
            fig2.add_shape(
                type="line", xref="paper", x0=0.05, x1=0.95, yref="y",
                y0=mu_L, y1=mu_L,
                line=dict(dash="dash", width=1.6, color=CEFR_LINE)
            )
            fig2.add_annotation(
                x=0.955, xref="paper", y=mu_L, yref="y",
                text=f"CEFR Level: {L}", showarrow=False,
                xanchor="left", yanchor="middle",
                bgcolor=CEFR_BADGE_BG, bordercolor=CEFR_BADGE_BD,
                borderwidth=1, font=dict(size=11, color=CEFR_BADGE_TX)
            )

    # Axis + layout (narrower plot; theme-aware template)
    fig2.update_layout(
        xaxis=dict(tickmode="array", tickvals=[0], ticktext=["Current cohort (humans)"], range=[-0.5, 0.5]),
        yaxis_title=disp,
        template=PLOT_TEMPLATE,
        height=520,
        width=700,  # ~50% narrower
        margin=dict(l=70, r=60, t=30, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=False)

    # ===== Inspect Feature (Individual Student) =====
    st.markdown(
        '<div style="font-weight:800;font-size:1.25rem;margin:12px 0 4px;">Inspect Feature (Individual Student)</div>',
        unsafe_allow_html=True,
    )

    show_tid = st.selectbox(
        "Choose TID",
        options=df_plot["TID"].tolist(),
        index=0,
        key="cefr_report_tid",
    )

    row = df_plot.loc[df_plot["TID"] == show_tid].iloc[0]
    p_cohort = float(row["_pct"])
    cohort_n = len(df_plot)
    q_int = int(row["_q"])
    quartile_label = {1: "Lower", 2: "Lower-middle", 3: "Upper-middle", 4: "Higher"}[q_int]
    quartile_phrase = {
        1: "the lower range of the cohort",
        2: "the lower-middle range of the cohort",
        3: "the upper-middle range of the cohort",
        4: "the higher range of the cohort",
    }[q_int]

    # CEFR comparison text for the card
    if level_means:
        delta_single = float(row["_delta_mu"])
        nearest_lvl = str(row["_nearest_lvl"])
        nearest_mu = float(row["_nearest_mu"])
        if np.isnan(nearest_mu):
            cefr_text = "No CEFR means found for this metric."
        else:
            sign = "+" if delta_single > 0 else ""
            cefr_text = (
                f"Closest CEFR level: {nearest_lvl}. "
                f"Difference from its mean: Δ = {sign}{format_num(delta_single,1)} (vs {format_num(nearest_mu,1)})."
            )
    else:
        cefr_text = "No CEFR reference table loaded, so no comparison is shown."

    # --- Theme-adaptive student card + percentile band ---
    CARD_BG     = "#0f172a" if _dark else "#ffffff"
    CARD_TEXT   = "#e5e7eb" if _dark else "#111827"
    HR_COLOR    = "rgba(255,255,255,0.15)" if _dark else "#eef2f7"
    CARD_BORDER = "rgba(255,255,255,0.20)" if _dark else "#e5e7eb"

    band_html = quartile_band_html(p_cohort)  # uses theme internally

    st.markdown(
        f"""
<div style="border:1px solid {CARD_BORDER}; border-radius:14px; padding:16px 18px; background:{CARD_BG}; color:{CARD_TEXT};">
  <div style="display:flex; gap:10px; align-items:baseline; flex-wrap:wrap;">
    <div style="font-weight:700;">Student/TID:</div>
    <div style="font-family:ui-monospace, SFMono-Regular, Menlo, monospace;">{show_tid}</div>
    <div style="flex:1"></div>
    <div style="font-weight:700;">Feature:</div><div>{html.escape(disp)}</div>
  </div>
  <hr style="border:none; border-top:1px solid {HR_COLOR}; margin:12px 0;"/>
  <div style="margin-top:8px;">
    <div><strong>Cohort standing:</strong> {quartile_label} quartile — this text falls in {quartile_phrase} (n={cohort_n}).</div>
    <div style="margin-top:4px;"><strong>Percentile (cohort):</strong> {format_pct(p_cohort, 1)}</div>
  </div>
  <div style="margin-top:10px;">{band_html}</div>
  <div style="margin-top:12px;">
    <strong>Compared with CEFR means:</strong> {html.escape(cefr_text)}
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    # ===== End Inspect Feature (Group-Level) =====