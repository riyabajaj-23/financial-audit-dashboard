# AI-Powered Financial Audit & Expense Intelligence Platform

A quick-start portfolio MVP that screens expense claims for potential audit review using:
- Exact-match duplicate candidate detection
- Configurable expense-category policy limits
- Isolation Forest anomaly screening on transaction amounts
- Transparent risk scores and review reasons
- Streamlit dashboard, charts, CSV upload and flagged-report export

## Run locally

### 1. Create and activate a virtual environment

Windows PowerShell:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start the app
```bash
streamlit run app.py
```
It should open in your browser. If not, open the local URL printed in the terminal.

## CSV format

Required columns:
`expense_id, employee_id, expense_date, merchant, category, amount, currency, description`

Use the included `sample_expenses.csv` as the example. Column names are normalised to lowercase. Use only synthetic, public, or explicitly authorised data. Never upload confidential employer/client data.

## How the screening works

1. Duplicate candidate: same employee, date, merchant and amount.
2. Policy check: amount exceeds the configurable category limit.
3. Anomaly screening: Isolation Forest identifies unusual amounts when there are enough rows.
4. Risk score: adds rule-based points (duplicate 40, policy 30, anomaly 20, data-quality issue 25), capped at 100.

These are screening heuristics, not proof of fraud or a calibrated probability. Thresholds are illustrative and should be adapted to an actual organisation's policies.

## Before publishing to GitHub

- Replace sample data with a clearly labelled synthetic/public dataset.
- Add screenshots of the dashboard.
- Add tests for duplicate detection, policy limits and invalid data.
- Add a short architecture/methodology section and known limitations.
- Do not claim business savings, accuracy, or fraud detection performance unless you measured it.

## Suggested next improvements

- Split train/demo data and evaluate anomaly flags with labelled synthetic scenarios.
- Add pytest tests and CI.
- Add SQL storage or FastAPI if time permits.
- Add receipt/document parsing only after the core screening workflow is reliable.
