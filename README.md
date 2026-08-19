# Workforce Attrition Patterns and Risk Hotspot Analysis at Palo Alto Networks

An end-to-end HR analytics project: dataset inspection, cleaning, feature
engineering, exploratory data analysis, KPI calculation, risk hotspot
identification, and an interactive Streamlit dashboard.

## Project Folder Structure

```text
workforce-attrition-project/
│
├── data/
│   └── attrition.csv                  # Source dataset (1,470 employees, 31 columns)
│
├── notebooks/
│   └── workforce_attrition_eda.ipynb  # Full step-by-step EDA notebook
│
├── data_utils.py                      # Shared cleaning & feature-engineering functions
├── run_full_analysis.py               # Script that prints every KPI/stat used in the report
├── app.py                             # Streamlit dashboard
├── requirements.txt
├── README.md
│
├── outputs/
│   ├── figures/                       # Saved chart PNGs
│   └── reports/                       # Full analysis text output
│
└── research_paper/
    └── attrition_analysis_report.docx # Full research paper + executive summary
```

## Dataset Summary

- **Rows:** 1,470 employees
- **Columns:** 31
- **Missing values:** 0
- **Duplicate rows:** 0
- **Overall attrition rate:** 16.12% (237 of 1,470 employees left)

## How to Run Locally

1. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis script** (prints every KPI and hotspot statistic):
   ```bash
   python run_full_analysis.py
   ```

4. **Launch the Streamlit dashboard:**
   ```bash
   streamlit run app.py
   ```
   This opens the dashboard at `http://localhost:8501`.

5. **Open the EDA notebook:**
   ```bash
   jupyter notebook notebooks/workforce_attrition_eda.ipynb
   ```

## How to Run in Google Colab

1. Upload `data_utils.py`, `data/attrition.csv`, and
   `notebooks/workforce_attrition_eda.ipynb` to your Colab session
   (or mount Google Drive and place them in the same relative layout).
2. In the first cell, install any missing packages:
   ```python
   !pip install seaborn -q
   ```
3. Run all cells top to bottom.
4. Colab cannot run Streamlit apps directly — to preview the dashboard from
   Colab, install `streamlit` and use a tunneling tool such as `localtunnel`
   or `ngrok`:
   ```python
   !pip install streamlit -q
   !npm install -g localtunnel
   !streamlit run app.py &>/content/logs.txt & npx localtunnel --port 8501
   ```

## How to Deploy the Streamlit Dashboard Online

The easiest option is [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this project folder to a public (or private) GitHub repository.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Click **New app**, select the repository, branch, and set the main file
   path to `app.py`.
4. Deploy — Streamlit Cloud installs `requirements.txt` automatically.

## Key Files

| File | Purpose |
|---|---|
| `data_utils.py` | Loading, cleaning, and feature-engineering logic shared by the notebook and the dashboard |
| `run_full_analysis.py` | Prints every statistic used in the KPIs, hotspot table, and research paper |
| `app.py` | Interactive Streamlit dashboard (4 modules: Overview, Department/Role Heatmaps, Demographic Explorer, Tenure & Workload) |
| `notebooks/workforce_attrition_eda.ipynb` | Full step-by-step exploratory analysis with charts |
| `research_paper/attrition_analysis_report.docx` | Complete research paper, executive summary, and conclusion |

## Notes on Methodology

- No missing values or duplicates were found in the source data, so no
  imputation was necessary.
- `Attrition` was already encoded as 0/1; a readable `AttritionLabel`
  (Yes/No) column was added for display purposes.
- Outliers (e.g. very long-tenured employees, very high incomes) were
  **retained**, not removed — they are genuine data points, and removing
  them would bias attrition rates for exactly the groups being studied.
- All relationships described (satisfaction, compensation, workload, etc.)
  are reported as **associations**, not causal claims.
- Small-sample segments (fewer than 30 employees) in the risk hotspot table
  are explicitly flagged so they are not mistaken for statistically robust
  findings.
