# 🏥 Hospital Readmission Risk Predictor — Streamlit App

Augmented AI platform for clinical teams. Upload patient data, train three ML models,
surface SHAP explanations, audit fairness, and export PDF clinical memos.

---

## 🚀 Deployment (Streamlit Community Cloud — Free)

### Step 1 — Push to GitHub

1. Create a **new GitHub repository** (e.g. `hospital-readmission-app`)
2. Upload these two files to the repo root:
   - `app.py`
   - `requirements.txt`
3. Optionally add `hospital_readmission_risk.csv` as a sample dataset

### Step 2 — Deploy on Streamlit Community Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub
2. Click **"New app"**
3. Select your repository, branch (`main`), and set **Main file path** to `app.py`
4. Click **"Deploy"** — it will be live in ~2 minutes

Your app will get a permanent URL like:
```
https://your-username-hospital-readmission-app-app-xxxxxx.streamlit.app
```

Share this URL with your professor — **no Colab, no code, no setup required**.

---

## 💻 Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## 📋 How to Use the App

| Step | Tab | Action |
|------|-----|--------|
| 1 | Sidebar | Upload your CSV file |
| 2 | Sidebar | Select target & fairness columns |
| 3 | Sidebar | Click **Train All 3 Models** |
| 4 | ROC Curves | View model performance |
| 5 | Explainability | Compute SHAP feature importance |
| 6 | Risk Scoring | Score individual patients |
| 7 | Calibration & Fairness | Audit bias across subgroups |
| 8 | Threshold Policy | Optimise decision threshold |
| 9 | Clinical Memo | Export PDF report |

---

## 📦 File Structure

```
hospital-readmission-app/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── hospital_readmission_risk.csv   # Sample dataset (optional)
└── README.md                       # This file
```

---

## ⚠️ Notes

- All data processing happens in-browser (server-side) — no data is stored permanently
- Works with any binary-class CSV (not just the sample dataset)
- PDF export uses fpdf2 and downloads directly to the user's device
- SHAP computation may take ~30 seconds for large datasets
