# Project Requirements — Python Data Scientist: Continuous Time-Series Cross-Correlation & Statistics (7-Day Project)

> Complete, nothing-omitted breakdown of the Upwork job posting **and** the attached project brief (*CrossCorrelnAnalysis_Project Brief.docx*).

---

## 1. Job Posting at a Glance

| Item | Detail |
|---|---|
| **Title** | Python Data Scientist — Continuous Time-Series Cross-Correlation & Statistics (7-day project) |
| **Posted** | 2 days ago |
| **Location eligibility** | Worldwide |
| **Budget** | **$500.00 Fixed-price** |
| **Experience level** | Expert — client states: *"I am willing to pay higher rates for the most experienced freelancers"* |
| **Project type** | One-time project |
| **Duration** | Strict, accelerated **7-day** contract (two milestones) |
| **Connects to apply** | 18 Connects |
| **Attachment** | CrossCorrelnAnalysis_Project Brief.docx (20 KB) |
| **Proposal screening question** | "Please refer to 'FREELANCER REQUIREMENTS / PROFILE' items in description." |

---

## 2. Project Overview

- The project is a **time-series analysis** on a specialised dataset collected for a **Human-Computer Interaction (HCI) study**.
- The study compares **continuous tracking ratings** logged by **Targets** (subjects initiating a baseline tracking event) against **evaluation ratings** logged by **Observers**.
- The goal is to map **observer tracking alignment** (how closely observers follow targets) and to calculate each observer's **personalised temporal cognitive lag** (how far ahead/behind in time their tracking best aligns).
- The entire scope of work must be completed within a **strict, accelerated one-week period**.

---

## 3. Dataset Specifications

| Property | Value |
|---|---|
| **Observers (total)** | 28 |
| **— Test group** | 15 participants |
| **— Control group** | 13 participants |
| **Target trials** | 6 unique video trials, each featuring a **different target individual** |
| **Trial length** | Varying, **1–2 minutes** each |
| **Sampling rate** | Continuous, **one data row every 1000 ms (1 second)** |
| **Dyadic analyses required** | **168 total** (28 observers × 6 trials) |

---

## 4. Key Numbers — Quick Reference

| Quantity | Value | How it's derived |
|---|---|---|
| Dyadic (observer–target) sessions | **168** | 28 observers × 6 trials |
| Lag window | **−5 s to +10 s** | Uniform/standardised for all sessions |
| Lag step size | **1000 ms (1 s)** | Matches the data sampling interval |
| Correlations per session | **Exactly 16** | 5 negative shifts (−5…−1) + 1 zero-lag (0) + 10 positive shifts (+1…+10) |
| Total cross-correlation computations | **2,688** | 168 sessions × 16 lag steps |
| Baseline (zero-lag) Pearson values | **168** | One per session |
| Master Metrics Sheet rows | **Exactly 168** | One row per observer–target pairing |

---

## 5. Scope of Work & Technical Specifications

The analysis must be implemented **strictly in Python**, using libraries such as **pandas, numpy, and scipy or statsmodels**. The pipeline has three parts:

### 5.1 Tier 1 — Baseline Alignment (Zero-Lag Pearson)

- Compute the **instantaneous, moment-by-moment Pearson correlation coefficient (r)** at a **fixed zero-lag baseline (lag = 0)**.
- One baseline value for **each of the 168 observer–target pairings**.

### 5.2 Tier 2 — Temporal Dynamics (Time-Lagged Cross-Correlation)

- Compute **time-lagged cross-correlation coefficients** across a **standardised search window of −5 to +10 seconds** for every session.
- **Granularity requirement:** Because data rows are sampled at 1000 ms increments, cross-correlations must be computed **step-by-step at every single 1000 ms increment shift** within the bounding window.
  - This yields **exactly 16 separate correlation coefficients per session**:
    - **5 negative shifts:** −5, −4, −3, −2, −1 seconds
    - **1 zero-lag shift:** 0 seconds
    - **10 positive shifts:** +1, +2, +3, +4, +5, +6, +7, +8, +9, +10 seconds
  - Across all 168 sessions: 168 × 16 = **2,688 individual correlations**.
- **Peak extraction:** From the 16 coefficients per session, extract:
  - the **personalised ceiling alignment** — the **Peak r** (the point of **maximum absolute correlation**), and
  - the **exact 1000 ms delay point (lag, in seconds)** at which that peak occurs — the observer's **Personalised Latency Delay**.

#### Missing Data Handling (mandatory rule)

- Shifting the series step-by-step creates **empty boundary cells** at the edges of each series.
- The script must **dynamically omit these boundary rows during calculation at each specific 1000 ms shift**.
- Boundary cells must **NOT be filled with zeros** — zero-filling would cause **artificial correlation attenuation**.

### 5.3 Tier 3 — Group-Level Comparative Inferential Analysis

- Perform a group-level comparative analysis — e.g., a **Linear Mixed-Effects Model (LMM)** or **Mixed-Design ANOVA**.
- Compare the **Test group vs. the Control group** on average, to determine differences in:
  - **Tracking alignment strength** (correlation magnitude), and
  - **Latency** (lag/delay).
- The model must evaluate **main effects and interaction indices**.
- Outputs must include **F-statistics, degrees of freedom, and exact p-values**.

---

## 6. Milestones & Timeline (Strict 7-Day Schedule)

### Milestone 1 — Dynamic Time-Series Analysis & Statistical Modeling (Days 1–4)

**Objective:** Execute end-to-end calculations across all baseline alignment, iterative cross-correlation shifts, and group-level inferential metrics.

**Tasks:**
1. Ingest the raw tracking files into a **repeatable Python environment**.
2. Process the **zero-lag Pearson r (lag = 0)** baseline values for **all 168 profiles**.
3. Build the **cross-correlation loop** executing a calculation at **every 1000 ms increment step** within the uniform **−5 to +10 second** lag window, with **dynamic boundary-edge handling** (omit, never zero-fill).
4. Extract the **personalised ceiling alignment (Peak r)** out of the 16 calculated steps and note the **exact 1000 ms delay point (Peak lag)** where it occurs.
5. Build and execute the **group-level comparative model (LMM or Mixed-Design ANOVA)** evaluating **main effects and interaction indices**.

**Milestone 1 Deliverables:**
1. The completed **168-row Master Metrics Sheet** with columns for: Observer ID, Group (Test/Control), Target ID, Trial Length, Baseline Pearson, Peak r, and Peak lag.
2. **Statistical model summary readouts**, including **F-statistics, degrees of freedom, and exact p-values**.

### Milestone 2 — Interactive Code Handoff, Environment Documentation & Technical Verification (Days 5–7)

**Objective:** Finalise and package a fully documented, cell-by-cell reproducible notebook and system mirror script.

**Tasks:**
1. **Clean, comment, and insert inline documentation** throughout a step-by-step **Jupyter Notebook**.
2. **Automatically generate** an exported **mirror production copy** as a standard raw **Python script (.py)**.
3. Extract all environment library dependencies into a standardised Python **requirements.txt** manifest.
4. **Support user reproduction runs** to confirm **identical technical verification outputs across all data frames** (the client will re-run the code and results must match exactly).

**Milestone 2 Deliverables:**
- The optimised, fully commented open-source **Jupyter Notebook (.ipynb)**
- The matching **Python script (.py)**
- The **requirements.txt** environment manifest
- **Publication-ready data summary tables**

---

## 7. Complete Deliverables Checklist

### 7.1 Source Code
- [ ] Fully documented, **step-by-step Jupyter Notebook (.ipynb)** showing **code cells alongside execution readouts** (outputs must be visible in the notebook).
- [ ] A **mirror production copy automatically exported** as a standard **Python script (.py)**.
- [ ] **requirements.txt** listing all library dependencies.

### 7.2 Master Metrics Sheet — exactly 168 rows

A clean spreadsheet / data frame, one row per observer–target session, with these columns:

| Column | Description |
|---|---|
| **Observer ID** | Identifier for each of the 28 observers |
| **Group** | Test or Control |
| **Target ID** | Which of the 6 target trials the row refers to |
| **Trial Length** | Duration of that trial (1–2 min) |
| **Baseline Pearson r** | Pearson correlation at lag = 0 |
| **Peak Cross-Correlation (Peak r)** | Chosen from the 16 progressive 1000 ms shifts (maximum **absolute** correlation) |
| **Personalised Latency Delay (Peak lag)** | In **seconds** — the specific 1000 ms shift where maximum absolute correlation is achieved |

### 7.3 Statistical Outputs
- [ ] Summary tables and **significance testing outputs** from the group-level model (LMM or Mixed-Design ANOVA).
- [ ] Must include **F-statistics, degrees of freedom, and exact p-values**.
- [ ] Publication-ready formatting of data summary tables.

---

## 8. Application Requirements — Proposal MUST Explicitly Address All 3

> ⚠️ Client's exact warning: *"Applicants who do not explicitly address the 3 candidate requirements above will not be considered or contacted for this project."*

1. **Time-Series & Cross-Correlation Portfolio Evidence (REQUIRED / essential)**
   - Provide **short code snippets or case studies** proving you have successfully implemented **cross-correlation coefficients** and **step-by-step lag modelling in Python**.
   - *"Applications without explicit proof of working with time-shifted cross-correlations will not be reviewed."*

2. **Advanced Academic Statistics Background (REQUIRED / essential)**
   - Provide evidence of prior Python work with **nested trial designs**, **signal synchronisation**, or **mixed-effects statistical data analysis**.

3. **APA Style Familiarity (PREFERRED)**
   - State whether you have experience **formatting statistical tables and data summaries in APA style**.
   - Candidates with strong APA skills will be **prioritised for a potential follow-up project**: assisting with the **write-up of the final results section for journal submission**.

---

## 9. Skills & Qualifications

**Mandatory skills (as listed on the posting):**
- Python
- Statistics
- APA Formatting
- Data Science

**Preferred qualifications:**
- English level: **Fluent**

**Expected tech stack:** Python with **pandas**, **numpy**, and **scipy** and/or **statsmodels**; Jupyter Notebook.

**Profile sought:** An advanced Python data scientist / statistical engineer with deep expertise in **time-series data** and **academic statistics**.

---

## 10. Client Information

| Item | Detail |
|---|---|
| Country / City | Australia — Brisbane (local time shown: 5:03 AM) |
| Payment method | Verified ✔ |
| Rating | 5.0 / 5.0 (from 4 reviews) |
| Jobs posted | 7 (100% hire rate, 1 open job) |
| Total spent | $19K |
| Hires | 9 total, 3 active |
| Avg hourly rate paid | $29.46/hr across 569 hours |
| Member since | Feb 20, 2017 |

---

## 11. Job Activity

| Metric | Value |
|---|---|
| Proposals | 5 to 10 |
| Last viewed by client | Yesterday |
| Interviewing | 1 |
| Invites sent | 0 |
| Unanswered invites | 0 |

---

## 12. Critical Constraints & Fine Print (Do Not Miss)

1. **Python only** — the analysis must be implemented strictly in Python (pandas / numpy / scipy / statsmodels).
2. **Exactly 16 lags per session** — every 1000 ms step within −5 s to +10 s; no coarser or finer stepping.
3. **No zero-filling** — boundary rows created by shifting must be **dynamically omitted** at each lag; zero-filling artificially attenuates correlations and is explicitly forbidden.
4. **Peak = maximum absolute correlation** — record both the Peak r and the exact lag (in seconds) where it occurs, per session.
5. **Exactly 168 rows** in the Master Metrics Sheet — one per observer–target pairing.
6. **Group-level model is required** — LMM or Mixed-Design ANOVA, Test vs. Control, with main effects and interactions, reporting F, df, and exact p-values.
7. **Reproducibility is verified** — the client will run reproduction checks; the notebook, mirror .py script, and requirements.txt must produce **identical outputs across all data frames**.
8. **Notebook must show execution readouts** — code cells with visible outputs, cleaned, commented, with inline documentation.
9. **The .py file must be an auto-exported mirror** of the notebook (not a separately written script).
10. **Strict two-milestone split**: all computation/statistics by **Day 4**; all packaging, documentation, and verification by **Day 7**.
11. **Proposal gate**: explicitly answer all 3 candidate-profile items (with proof for items 1 and 2), or the application will not be reviewed or contacted.
12. **Follow-up opportunity**: strong APA formatting skills may lead to a second contract for the journal results-section write-up.
