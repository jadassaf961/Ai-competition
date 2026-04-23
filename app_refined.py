"""
Hospital Readmission Risk Predictor
Augmented AI Platform for Clinical Teams — Streamlit Edition

Two views:
  • Clinical Staff  (default) — clean, deployment-ready workflow for medical staff
  • Technical / Admin        — full ML backend: metrics, ROC, SHAP, fairness, calibration
"""

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
import json
import os
import re
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix,
    brier_score_loss, f1_score, accuracy_score, balanced_accuracy_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
import shap
from fpdf import FPDF, XPos, YPos

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CareInsight · Readmission Risk Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — light-medical theme + clinical product styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
}

/* ── Technical-view header (original) ───────────────────────────────────── */
.app-hdr {
  background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #f0fdfa 100%);
  border: 1px solid #bae6fd;
  border-radius: 14px;
  padding: 28px 36px;
  margin-bottom: 20px;
  box-shadow: 0 2px 12px rgba(14,165,233,0.08);
}
.app-hdr h1 { color: #0c4a6e; font-size: 26px; font-weight: 700; margin: 0 0 6px; }
.app-hdr p  { color: #475569; font-size: 14px; margin: 0; }

.badge {
  display: inline-block;
  background: rgba(14,165,233,.12);
  border: 1px solid rgba(14,165,233,.35);
  color: #0369a1;
  padding: 3px 12px;
  border-radius: 100px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
  margin-bottom: 10px;
}

.kpi-grid   { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 20px; }
.kpi-grid-5 { display: grid; grid-template-columns: repeat(5,1fr); gap: 12px; margin-bottom: 20px; }
.kpi {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 16px 14px;
  text-align: center;
  box-shadow: 0 1px 4px rgba(0,0,0,.05);
}
.kpi-v { font-size: 26px; font-weight: 700; color: #0c4a6e; }
.kpi-l { font-size: 11px; color: #64748b; font-weight: 500; margin-top: 4px; text-transform: uppercase; letter-spacing: .5px; }

.section-head { font-size: 16px; font-weight: 700; color: #0c4a6e; margin: 20px 0 4px; }
.section-sub  { font-size: 13px; color: #64748b; margin: 0 0 14px; }

.risk-hi { background:#fef2f2;border:2px solid #fca5a5;color:#dc2626;padding:7px 20px;border-radius:100px;font-weight:700;font-size:15px;display:inline-block; }
.risk-md { background:#fffbeb;border:2px solid #fcd34d;color:#d97706;padding:7px 20px;border-radius:100px;font-weight:700;font-size:15px;display:inline-block; }
.risk-lo { background:#f0fdf4;border:2px solid #86efac;color:#16a34a;padding:7px 20px;border-radius:100px;font-weight:700;font-size:15px;display:inline-block; }

.chk-item {
  background: #f8fafc;
  border-left: 4px solid #0ea5e9;
  padding: 10px 14px;
  margin-bottom: 6px;
  border-radius: 0 8px 8px 0;
  font-size: 13px;
  color: #334155;
}

.info-box    { background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;padding:10px 14px;color:#0369a1;font-size:13px;margin-bottom:14px; }
.warn-box    { background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:10px 14px;color:#92400e;font-size:13px;margin-bottom:14px; }
.success-box { background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:10px 14px;color:#166534;font-size:13px;margin-bottom:14px; }
.signal-warn { background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:12px 16px;color:#9a3412;font-size:13px;margin:10px 0 14px; }

table.dfp { border-collapse:collapse;font-size:13px;width:100%; }
table.dfp th { background:#f1f5f9;color:#334155;padding:8px 12px;border:1px solid #e2e8f0;font-weight:600;text-align:left; }
table.dfp td { padding:7px 12px;border:1px solid #e2e8f0;color:#475569; }
table.dfp tr:hover td { background:#f8fafc; }

/* Streamlit overrides */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  background: #f8fafc;
  border-radius: 10px;
  padding: 4px;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 8px;
  padding: 8px 16px;
  font-weight: 500;
  color: #64748b;
}
.stTabs [aria-selected="true"] {
  background: #0284c7 !important;
  color: white !important;
}
.stButton > button {
  background: #0284c7;
  color: white;
  font-weight: 600;
  border: none;
  border-radius: 8px;
  padding: 8px 20px;
  transition: background .2s;
}
.stButton > button:hover { background: #0369a1; }
.stProgress > div > div { background: #0284c7; }

/* ── Clinical-view product styling ──────────────────────────────────────── */
.clinical-hero {
  background: linear-gradient(120deg, #ffffff 0%, #f0f9ff 100%);
  border: 1px solid #e0f2fe;
  border-radius: 16px;
  padding: 28px 36px;
  margin-bottom: 24px;
  box-shadow: 0 2px 14px rgba(14,165,233,0.06);
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.clinical-hero .brand-left { display: flex; align-items: center; gap: 18px; }
.clinical-hero .brand-mark {
  width: 52px; height: 52px; border-radius: 14px;
  background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 100%);
  display: flex; align-items: center; justify-content: center;
  color: white; font-size: 26px;
  box-shadow: 0 4px 14px rgba(14,165,233,0.28);
}
.clinical-hero h1 {
  color: #0c4a6e; font-size: 24px; font-weight: 700; margin: 0; letter-spacing: -0.3px;
}
.clinical-hero .tagline {
  color: #64748b; font-size: 13px; margin: 2px 0 0; font-weight: 400;
}
.clinical-hero .brand-right { text-align: right; }
.clinical-hero .version {
  font-size: 11px; color: #94a3b8; font-weight: 500;
  letter-spacing: .6px; text-transform: uppercase;
}
.clinical-hero .org {
  font-size: 12px; color: #475569; font-weight: 600; margin-top: 4px;
}

.upload-wrap {
  background: white;
  border: 2px dashed #bae6fd;
  border-radius: 16px;
  padding: 48px 36px;
  text-align: center;
  margin: 24px 0;
  transition: all .2s;
}
.upload-wrap:hover { border-color: #38bdf8; background: #f0f9ff; }
.upload-icon {
  font-size: 46px; margin-bottom: 12px;
}
.upload-title {
  font-size: 20px; font-weight: 700; color: #0c4a6e; margin-bottom: 6px;
}
.upload-desc {
  font-size: 14px; color: #64748b; margin-bottom: 4px;
}
.upload-hint {
  font-size: 12px; color: #94a3b8; margin-top: 14px;
}

.stat-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 18px 0 22px; }
.stat-card {
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 20px 22px;
  box-shadow: 0 1px 3px rgba(0,0,0,.04);
  position: relative;
  overflow: hidden;
}
.stat-card::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
  background: #0ea5e9;
}
.stat-card.hi::before { background: #dc2626; }
.stat-card.md::before { background: #d97706; }
.stat-card.lo::before { background: #16a34a; }
.stat-label { font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 600; letter-spacing: .6px; }
.stat-value { font-size: 30px; font-weight: 700; color: #0f172a; margin-top: 6px; }
.stat-sub   { font-size: 12px; color: #94a3b8; margin-top: 2px; }
.stat-value.hi { color: #dc2626; }
.stat-value.md { color: #d97706; }
.stat-value.lo { color: #16a34a; }

.panel {
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 22px 26px;
  margin-bottom: 18px;
  box-shadow: 0 1px 3px rgba(0,0,0,.03);
}
.panel-title { font-size: 15px; font-weight: 700; color: #0c4a6e; margin: 0 0 4px; }
.panel-sub   { font-size: 13px; color: #64748b; margin: 0 0 14px; }

.drill-header {
  background: linear-gradient(120deg, #0c4a6e 0%, #0284c7 100%);
  color: white;
  border-radius: 14px;
  padding: 22px 28px;
  margin-bottom: 18px;
  box-shadow: 0 3px 14px rgba(12,74,110,.15);
}
.drill-header .patient-id {
  font-size: 12px; color: #bae6fd; letter-spacing: .8px;
  text-transform: uppercase; font-weight: 600;
}
.drill-header .patient-name {
  font-size: 22px; font-weight: 700; margin-top: 2px;
}
.drill-header .patient-meta {
  font-size: 13px; color: #e0f2fe; margin-top: 4px;
}

.driver-chip {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 14px; margin-bottom: 8px; border-radius: 10px;
  background: #f8fafc; border-left: 4px solid #cbd5e1;
  font-size: 14px; color: #334155;
}
.driver-chip.up   { border-left-color: #dc2626; background: #fef2f2; color: #991b1b; }
.driver-chip.down { border-left-color: #16a34a; background: #f0fdf4; color: #14532d; }
.driver-chip .arrow { font-weight: 700; font-size: 16px; min-width: 14px; }
.driver-chip .label { font-weight: 500; flex: 1; }
.driver-chip .detail { font-size: 12px; color: #64748b; }

.footer-compliance {
  margin-top: 24px; padding: 14px 18px;
  background: #f8fafc; border: 1px solid #e2e8f0;
  border-radius: 10px; font-size: 12px; color: #64748b; text-align: center;
}
.footer-compliance strong { color: #334155; }

.view-chip {
  display: inline-block;
  background: #ecfeff; border: 1px solid #a5f3fc;
  color: #0e7490; font-size: 11px; font-weight: 600;
  padding: 3px 10px; border-radius: 100px;
  letter-spacing: .5px; text-transform: uppercase;
}

/* Larger primary action button for clinical view */
.big-cta .stButton > button {
  padding: 14px 28px;
  font-size: 15px;
  border-radius: 12px;
  box-shadow: 0 4px 14px rgba(14,165,233,.22);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {'XGBoost': '#0284c7', 'Random Forest': '#0d9488', 'Logistic Regression': '#6366f1'}
HIGH_COLOR, MED_COLOR, LOW_COLOR = '#dc2626', '#d97706', '#16a34a'

BASE_CHECKLIST = [
    'Schedule follow-up appointment within 7 days of discharge',
    'Confirm patient has correct medications and understands dosing',
    "Provide written discharge summary in patient's preferred language",
    'Screen for social determinants: housing, food access, transport',
    'Ensure patient has a primary care provider and contact information',
]

FACTOR_ACTIONS = {
    'diag':      'Review all primary/secondary diagnoses for care gaps',
    'med':       'Perform full medication reconciliation before discharge',
    'age':       'Assess fall risk; arrange geriatric support if indicated',
    'los':       'Coordinate complex transition plan (extended LOS detected)',
    'lab':       'Follow up on abnormal labs before discharge',
    'visit':     'Connect high-utiliser to case management and social work',
    'diabetes':  'Confirm HbA1c plan and enrol in diabetes self-management education',
    'heart':     'Arrange cardiology follow-up within 7 days post-discharge',
    'renal':     'Nephrology referral; monitor BUN/creatinine after discharge',
    'depress':   'Screen for depression; provide mental health referral',
    'insurance': 'Connect with social worker for benefits navigation',
    'race':      'Ensure interpreter services and culturally competent care',
    'ethnicity': 'Ensure interpreter services and culturally competent care',
    'creatinine':'Nephrology referral; monitor renal function post-discharge',
    'hemoglobin':'Check for anaemia; haematology referral if Hb < 10 g/dL',
    'glucose':   'Review glycaemic control; endocrinology referral if indicated',
    'bmi':       'Nutrition assessment and dietitian referral',
    'smoking':   'Smoking cessation counselling and NRT prescription',
    'alcohol':   'Alcohol dependency screening (AUDIT); addiction services referral',
    'mental':    'Mental health screening; psychiatry referral if indicated',
    'followup':  'Reinforce follow-up importance; consider community health worker',
    'social':    'Social work referral for support network assessment',
    'physical':  'Physiotherapy assessment; activity prescription on discharge',
    'admission': 'Emergency admission - ensure root cause addressed before discharge',
    'previous':  'High utiliser - case management referral and care coordination plan',
    'chronic':   'Chronic disease management plan; specialist follow-up arranged',
    'procedure': 'Post-procedure care instructions provided; wound care follow-up',
}

# Plain-English feature label mappings for clinical view
FEATURE_PLAIN_LANGUAGE = {
    'age':                      'Patient age',
    'gender':                   'Patient demographics (gender)',
    'sex':                      'Patient demographics (sex)',
    'race':                     'Patient demographics (race)',
    'ethnicity':                'Patient demographics (ethnicity)',
    'time_in_hospital':         'Length of hospital stay',
    'length_of_stay':           'Length of hospital stay',
    'los':                      'Length of hospital stay',
    'num_medications':          'Number of medications',
    'n_medications':            'Number of medications',
    'num_lab_procedures':       'Lab procedures performed',
    'num_procedures':           'Procedures performed',
    'number_diagnoses':         'Number of diagnoses on record',
    'number_inpatient':         'Prior inpatient admissions',
    'number_emergency':         'Recent emergency visits',
    'number_outpatient':        'Recent outpatient visits',
    'previous_admissions':      'Prior hospital admissions',
    'readmissions':             'Prior readmission history',
    'diabetesmed':              'On diabetes medication',
    'diabetes':                 'Diabetes history',
    'a1cresult':                'HbA1c lab result',
    'hba1c':                    'HbA1c lab result',
    'max_glu_serum':            'Serum glucose result',
    'glucose':                  'Glucose level',
    'insulin':                  'Insulin usage',
    'creatinine':               'Kidney function (creatinine)',
    'bun':                      'Kidney function (BUN)',
    'hemoglobin':               'Hemoglobin level',
    'bmi':                      'Body mass index (BMI)',
    'weight':                   'Patient weight',
    'smoking':                  'Smoking status',
    'alcohol':                  'Alcohol use',
    'heart':                    'Cardiac history',
    'cardiac':                  'Cardiac history',
    'renal':                    'Renal history',
    'kidney':                   'Renal history',
    'depress':                  'Depression / mental health',
    'mental':                   'Mental health history',
    'insurance':                'Insurance coverage',
    'payer':                    'Insurance / payer',
    'change':                   'Recent medication changes',
    'discharge_disposition':    'Discharge destination',
    'admission_type':           'Admission type',
    'admission_source':         'Admission source',
    'physician':                'Attending care team',
    'ward':                     'Ward / unit',
    'diag_1':                   'Primary diagnosis',
    'diag_2':                   'Secondary diagnosis',
    'diag_3':                   'Tertiary diagnosis',
    'diagnosis':                'Diagnosis',
}

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
_defaults = dict(
    df=None,
    X_train=None, X_test=None, y_train=None, y_test=None,
    feature_names=[], target_col=None,
    sensitive_cols=[], cat_cols=[], num_cols=[], le_dict={},
    pos_label=None,
    models={}, metrics={}, probs={},
    best_name=None, chosen_name=None,
    shap_vals=None, explainer=None,
    threshold=0.5,
    X_test_raw=None,
    last_prob=None, last_cat=None, last_idx=None,
    last_top_factors=[], last_checklist=[],
    trained=False,
    shap_computed=False,
    # ─── Clinical view state ───
    view_mode='clinical',               # 'clinical' (default) | 'technical'
    all_probs=None,                     # per-row probability for full dataset
    all_shap=None,                      # SHAP values for full dataset (test-set slice used)
    clinical_selected_idx=None,         # currently-selected patient row in clinical roster
    clinical_filter='All',              # 'All' | 'High' | 'Medium' | 'Low'
    clinical_search='',                 # search string for patient lookup
    clinical_id_col=None,               # column used as patient ID in roster
    analysis_done=False,                # True once models + all-row probs + SHAP done
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

S = st.session_state   # convenience alias

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers — shared
# ─────────────────────────────────────────────────────────────────────────────
def df_to_html(df, max_rows=5):
    rows = ''
    for _, r in df.head(max_rows).iterrows():
        rows += '<tr>' + ''.join(f'<td>{v}</td>' for v in r.values) + '</tr>'
    headers = ''.join(f'<th>{c}</th>' for c in df.columns)
    return f"<table class='dfp'><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>"

def risk_label(p, thr):
    hi = thr + (1 - thr) * 0.4
    lo = thr * 0.4
    if p >= hi:  return 'HIGH RISK',   'risk-hi', HIGH_COLOR
    if p >= lo:  return 'MEDIUM RISK', 'risk-md', MED_COLOR
    return           'LOW RISK',    'risk-lo', LOW_COLOR

def risk_tier(p, thr):
    """Short tier label: 'High' | 'Medium' | 'Low'."""
    hi = thr + (1 - thr) * 0.4
    lo = thr * 0.4
    if p >= hi: return 'High'
    if p >= lo: return 'Medium'
    return 'Low'

def _is_id_like(series):
    if series.dtype != object:
        return False
    return series.nunique() / max(len(series), 1) > 0.8

def preprocess(df, target_col):
    df = df.copy()
    thresh = int(0.4 * len(df))
    df = df.dropna(axis=1, thresh=thresh)

    cat_cols, num_cols = [], []
    for c in df.columns:
        if c == target_col:
            continue
        if _is_id_like(df[c]):
            continue
        if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]) or df[c].nunique() < 15:
            cat_cols.append(c)
        else:
            num_cols.append(c)

    target_series = df[target_col].astype(str).fillna('Unknown')
    val_counts = target_series.value_counts()
    if len(val_counts) != 2:
        raise ValueError(f"Target column '{target_col}' must have exactly 2 unique values, found: {list(val_counts.index)}")

    positive_keywords = ['high','yes','true','1','positive','readmit','risk']
    pos_label = None
    for kw in positive_keywords:
        for v in val_counts.index:
            if kw in str(v).lower():
                pos_label = v
                break
        if pos_label:
            break
    if pos_label is None:
        pos_label = val_counts.index[-1]
    df['__target__'] = (target_series == pos_label).astype(int)

    le_dict = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str).fillna('Missing'))
        le_dict[c] = le
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].fillna(df[c].median())

    return df, cat_cols + num_cols, cat_cols, num_cols, le_dict, pos_label

def train_models(X_tr, y_tr, X_te, y_te):
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    scale_pos = max(n_neg / max(n_pos, 1), 1.0)

    models_cfg = {
        'Logistic Regression': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc',  StandardScaler()),
            ('clf', LogisticRegression(max_iter=5000, class_weight='balanced',
                                       C=1.0, solver='lbfgs', random_state=42)),
        ]),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            class_weight='balanced_subsample', random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos,
            eval_metric='logloss', use_label_encoder=False,
            random_state=42, n_jobs=-1, verbosity=0),
    }

    results = {}
    for name, model in models_cfg.items():
        try:
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_te)[:, 1]
            preds = (probs >= 0.5).astype(int)
            results[name] = dict(
                model=model, probs=probs, preds=preds,
                auc=roc_auc_score(y_te, probs),
                ap=average_precision_score(y_te, probs),
                f1=f1_score(y_te, preds, zero_division=0),
                acc=accuracy_score(y_te, preds),
                brier=brier_score_loss(y_te, probs),
                bal=balanced_accuracy_score(y_te, preds),
            )
        except Exception as e:
            st.warning(f'Warning: {name} failed – {e}')
    return results

def compute_shap(model_name, model, X_te, feature_names):
    try:
        clf = model
        X_shap = X_te

        if isinstance(clf, Pipeline):
            pre = Pipeline(clf.steps[:-1])
            X_shap = pre.transform(X_te)
            clf = clf.steps[-1][1]

        if hasattr(clf, 'calibrated_classifiers_'):
            clf = clf.calibrated_classifiers_[0].estimator

        if model_name in ('XGBoost', 'Random Forest'):
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_shap)
            if isinstance(sv, list):
                sv = sv[1]
            elif sv.ndim == 3:
                sv = sv[:, :, 1]
        else:
            inner_lr = clf
            if hasattr(inner_lr, 'calibrated_classifiers_'):
                inner_lr = inner_lr.calibrated_classifiers_[0].estimator
            explainer = shap.LinearExplainer(inner_lr, X_shap,
                                             feature_perturbation='interventional')
            sv = explainer.shap_values(X_shap)
            if isinstance(sv, list):
                sv = sv[1]
            elif sv.ndim == 3:
                sv = sv[:, :, 1]

        return explainer, np.array(sv)
    except Exception as e:
        st.error(f'SHAP error: {e}')
        return None, None

def compute_fairness(raw, y_te, probs, sensitive_cols, threshold):
    rows = []
    for col in sensitive_cols:
        if col not in raw.columns:
            continue
        for g in raw[col].unique():
            mask = (raw[col] == g).values
            if mask.sum() < 10:
                continue
            gp, gy = probs[mask], y_te[mask]
            gplbl = (gp >= threshold).astype(int)
            try:
                auc = roc_auc_score(gy, gp) if len(np.unique(gy)) > 1 else float('nan')
            except Exception:
                auc = float('nan')
            tpr = float(gplbl[gy == 1].mean()) if gy.sum() > 0        else float('nan')
            fpr = float(gplbl[gy == 0].mean()) if (1 - gy).sum() > 0  else float('nan')
            rows.append({
                'Attribute': col, 'Group': str(g), 'N': int(mask.sum()),
                'Prevalence': f'{gy.mean():.1%}',
                'Pred Pos Rate': f'{gplbl.mean():.1%}',
                'TPR (Sensitivity)': f'{tpr:.1%}' if not np.isnan(tpr) else 'N/A',
                'FPR': f'{fpr:.1%}' if not np.isnan(fpr) else 'N/A',
                'AUC-ROC': f'{auc:.3f}' if not np.isnan(auc) else 'N/A',
            })
    return pd.DataFrame(rows)

def threshold_metrics(y_te, probs, thr):
    p  = (probs >= thr).astype(int)
    cm = confusion_matrix(y_te, p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv  = tp / max(tp + fp, 1)
    npv  = tn / max(tn + fn, 1)
    f1   = 2 * tp / max(2 * tp + fp + fn, 1)
    return {'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
            'Sensitivity': sens, 'Specificity': spec,
            'PPV': ppv, 'NPV': npv, 'F1': f1}

def generate_checklist(top_factors, risk_cat):
    extra = []
    for feat, _ in top_factors[:8]:
        fl = feat.lower()
        for key, action in FACTOR_ACTIONS.items():
            if key in fl and action not in extra:
                extra.append(action)
                break
    if risk_cat == 'HIGH RISK':
        extra.insert(0, 'HIGH RISK - Flag for immediate care coordination team review')
        extra.insert(1, 'Schedule 48-hour post-discharge phone check-in')
    return BASE_CHECKLIST + extra[:5]

def generate_pdf(patient_info, risk_score, risk_cat, top_factors,
                 checklist, model_name, threshold):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    eff_w = pdf.w - pdf.l_margin - pdf.r_margin

    pdf.set_fill_color(12, 74, 110)
    pdf.rect(0, 0, 210, 38, 'F')
    pdf.set_font('Helvetica', 'B', 17)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 8)
    pdf.cell(0, 8, 'Hospital Readmission Risk Report',
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(186, 230, 253)
    pdf.set_xy(10, 20)
    ts = datetime.now().strftime('%B %d, %Y  %H:%M')
    pdf.cell(0, 6,
             f'Generated: {ts}   |   Model: {model_name}   |   Threshold: {threshold:.2f}',
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(12)
    pdf.set_text_color(0, 0, 0)

    def section(title):
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_fill_color(240, 249, 255)
        pdf.cell(eff_w, 8, f'  {title}',
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True, border='B')
        pdf.ln(3)

    section('Patient Information')
    pdf.set_font('Helvetica', '', 11)
    for k, v in patient_info.items():
        if v and v != 'N/A':
            pdf.cell(65, 7, f'{k}:', new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.set_font('Helvetica', 'B', 11)
            pdf.cell(0, 7, str(v), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font('Helvetica', '', 11)
    pdf.ln(4)

    section('Readmission Risk Assessment')
    pct = int(risk_score * 100)
    if risk_cat == 'HIGH RISK':     pdf.set_fill_color(220, 38, 38)
    elif risk_cat == 'MEDIUM RISK': pdf.set_fill_color(217, 119, 6)
    else:                           pdf.set_fill_color(22, 163, 74)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 13)
    pdf.cell(70, 12, f'{pct}%  -  {risk_cat}',
             new_x=XPos.RIGHT, new_y=YPos.TOP, align='C', fill=True, border=1)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 12, '   Predicted probability of 30-day readmission',
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    section('Top Clinical Risk Factors  (SHAP Analysis)')
    pdf.set_font('Helvetica', '', 10)
    for i, (feat, val) in enumerate(top_factors[:8], 1):
        direction = 'Increases' if val > 0 else 'Decreases'
        r, g, b = (220, 38, 38) if val > 0 else (22, 163, 74)
        pdf.set_text_color(r, g, b)
        pdf.cell(8, 7, f'{i}.', new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(80, 7, str(feat)[:40], new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_text_color(r, g, b)
        pdf.cell(0, 7, f'{direction} risk  ({abs(val):.4f})',
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    section('Preventive Action Checklist')
    pdf.set_font('Helvetica', '', 10)
    for item in checklist:
        pdf.set_fill_color(224, 242, 254)
        x_start = pdf.l_margin
        y_start = pdf.get_y()
        pdf.set_xy(x_start, y_start)
        pdf.cell(5, 6, '', fill=True, border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(4, 6, '', new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_x(x_start + 9)
        pdf.multi_cell(eff_w - 9, 6, str(item))
    pdf.ln(4)

    section('Clinical Notes & Recommendations')
    pdf.set_font('Helvetica', '', 10)
    note = (
        f'Based on {model_name}, this patient has a {risk_cat.lower()} profile '
        f'with a predicted readmission probability of {pct}%. '
        'Clinical teams should review the top risk factors and implement the preventive '
        'actions listed above. All predictions are decision-support tools and must be '
        'reviewed by qualified clinical staff before any action is taken.\n\n'
        'This report was generated automatically. '
        'Contact your analytics team with any questions.'
    )
    pdf.set_fill_color(240, 249, 255)
    pdf.multi_cell(eff_w, 6, note, border=1, fill=True)

    pdf.set_y(-18)
    pdf.set_font('Helvetica', 'I', 7.5)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5,
             'CONFIDENTIAL - Contains Protected Health Information (PHI). Handle per HIPAA.',
             align='C')

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers — clinical view
# ─────────────────────────────────────────────────────────────────────────────
def humanize_feature(feat_name: str) -> str:
    """Turn a raw feature column name into a plain-English clinical label."""
    if feat_name is None:
        return 'Clinical factor'
    fl = str(feat_name).lower()
    for key, label in FEATURE_PLAIN_LANGUAGE.items():
        if key in fl:
            return label
    # Fallback: title-case underscored name
    return str(feat_name).replace('_', ' ').strip().title()

def detect_patient_id_column(df: pd.DataFrame):
    """Find a column that looks like a patient identifier (MRN, ID, etc.)."""
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ['patient_id', 'patientid', 'mrn', 'encounter', 'record_id', 'subject_id']):
            return c
    # Fallback: first id-like column
    for c in df.columns:
        cl = c.lower()
        if cl.endswith('id') or cl == 'id':
            return c
    return None

def score_all_rows(df_processed: pd.DataFrame, feats, model):
    """Score every row in the processed dataframe using the chosen model."""
    X_all = df_processed[feats].values
    return model.predict_proba(X_all)[:, 1]

def get_key_factor_for_patient(patient_shap, feature_names, value_row=None):
    """Return a short plain-English summary of the single strongest risk driver."""
    if patient_shap is None:
        return '—'
    idx = int(np.argmax(np.abs(patient_shap)))
    feat = feature_names[idx]
    direction = '▲' if patient_shap[idx] > 0 else '▼'
    return f"{direction} {humanize_feature(feat)}"

def compute_analysis(df_raw, target_col, split_ratio=0.8):
    """End-to-end: preprocess, train, pick best, compute SHAP, score full dataset."""
    df_p, feats, cat_c, num_c, le_d, pos_lbl = preprocess(df_raw, target_col)

    X = df_p[feats].values
    y = df_p['__target__'].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=round(1 - split_ratio, 2),
        stratify=y, random_state=42)

    all_idx = np.arange(len(df_raw))
    _, test_idx = train_test_split(
        all_idx, test_size=round(1 - split_ratio, 2),
        stratify=y, random_state=42)
    X_test_raw = df_raw.iloc[test_idx].reset_index(drop=True)

    results = train_models(X_tr, y_tr, X_te, y_te)
    best = max(results, key=lambda k: results[k]['auc'])

    # SHAP on test set using the best model
    exp, sv = compute_shap(best, results[best]['model'], X_te, feats)

    # Score every row in the uploaded dataset (for the clinical roster)
    all_probs = results[best]['model'].predict_proba(X).astype(float)[:, 1]

    return {
        'df_processed': df_p,
        'feats': feats,
        'cat_cols': cat_c, 'num_cols': num_c,
        'le_dict': le_d, 'pos_label': pos_lbl,
        'X_tr': X_tr, 'X_te': X_te, 'y_tr': y_tr, 'y_te': y_te,
        'X_test_raw': X_test_raw,
        'results': results,
        'best': best,
        'explainer': exp,
        'shap_vals': sv,
        'all_probs': all_probs,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — view toggle (always visible) + mode-specific controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👤 View")
    view_choice = st.radio(
        "Select view",
        ["🩺  Clinical Staff", "🔧  Technical / Admin"],
        index=0 if S.view_mode == 'clinical' else 1,
        label_visibility="collapsed",
    )
    new_mode = 'clinical' if view_choice.startswith("🩺") else 'technical'
    if new_mode != S.view_mode:
        S.view_mode = new_mode
        S.clinical_selected_idx = None
        st.rerun()

    st.markdown("---")

    if S.view_mode == 'technical':
        # ───────────────── Technical sidebar (original, preserved) ──────────
        st.markdown("### 📂 Data Upload")
        uploaded_file = st.file_uploader("Upload patient CSV", type=["csv"],
                                         help="Any CSV with numeric/categorical columns and a binary outcome column.",
                                         key="tech_uploader")

        if uploaded_file is not None:
            try:
                df_raw = pd.read_csv(uploaded_file)
                S.df = df_raw
                S.analysis_done = False
                st.success(f"✅ Loaded **{uploaded_file.name}**  \n{len(df_raw):,} rows × {len(df_raw.columns)} columns")
            except Exception as e:
                st.error(f"Error reading file: {e}")

        if S.df is not None:
            cols = list(S.df.columns)
            st.markdown("---")
            st.markdown("### ⚙️ Configuration")

            # Auto-detect target
            auto_target = cols[-1]
            for c in cols:
                if any(k in c.lower() for k in ['readmit', 'target', 'label', 'outcome', 'risk']):
                    auto_target = c
                    break

            target_col = st.selectbox("Target column", cols,
                                       index=cols.index(auto_target),
                                       help="Binary outcome column (e.g. High/Low, 1/0, Yes/No)")

            auto_sensitive = [c for c in cols if any(k in c.lower()
                              for k in ['gender', 'sex', 'race', 'ethnicity', 'age', 'insurance', 'payer'])]
            sensitive_cols = st.multiselect("Fairness columns",
                                             options=cols,
                                             default=auto_sensitive[:4],
                                             help="Columns for fairness audit (demographic attributes)")

            split_ratio = st.slider("Train/Test split", 50, 90, 80, 5,
                                     format="%d%%",
                                     help="Proportion of data used for training") / 100

            st.markdown("---")
            train_btn = st.button("🚀 Train All 3 Models", use_container_width=True)

            if train_btn:
                with st.spinner("Training Logistic Regression, Random Forest, and XGBoost… (~1 min)"):
                    try:
                        df_p, feats, cat_c, num_c, le_d, pos_lbl = preprocess(S.df, target_col)
                        S.cat_cols = cat_c
                        S.num_cols = num_c
                        S.le_dict = le_d
                        S.target_col = target_col
                        S.sensitive_cols = sensitive_cols
                        S.feature_names = feats
                        S.pos_label = pos_lbl

                        X = df_p[feats].values
                        y = df_p['__target__'].values
                        X_tr, X_te, y_tr, y_te = train_test_split(
                            X, y, test_size=round(1 - split_ratio, 2),
                            stratify=y, random_state=42)
                        S.X_train = X_tr
                        S.X_test  = X_te
                        S.y_train = y_tr
                        S.y_test  = y_te

                        all_idx = np.arange(len(S.df))
                        _, test_idx = train_test_split(
                            all_idx, test_size=round(1 - split_ratio, 2),
                            stratify=y, random_state=42)
                        S.X_test_raw = S.df.iloc[test_idx].reset_index(drop=True)

                        results = train_models(X_tr, y_tr, X_te, y_te)
                        S.models  = {k: v['model'] for k, v in results.items()}
                        S.probs   = {k: v['probs'] for k, v in results.items()}
                        S.metrics = {k: {m: v[m] for m in ['auc','ap','f1','acc','brier','bal']}
                                     for k, v in results.items()}

                        best = max(results, key=lambda k: results[k]['auc'])
                        S.best_name = best
                        S.chosen_name = best
                        S.trained = True
                        S.shap_computed = False
                        S.shap_vals = None

                        st.success(f"✅ Training complete!\nBest model: **{best}**")
                    except Exception as e:
                        import traceback
                        st.error(f"Training error: {e}\n{traceback.format_exc()}")

            if S.trained:
                st.markdown("---")
                st.markdown("### 🎚️ Threshold")
                threshold = st.slider("Decision threshold", 0.01, 0.99,
                                       float(S.threshold), 0.01, format="%.2f")
                S.threshold = threshold

                st.markdown("---")
                st.markdown("### 📊 Status")
                st.markdown(f"- **Models trained:** {'✅' if S.trained else '❌'}")
                st.markdown(f"- **SHAP computed:** {'✅' if S.shap_computed else '❌'}")
                st.markdown(f"- **Best model:** {S.best_name or '–'}")
                st.markdown(f"- **Active model:** {S.chosen_name or '–'}")
                st.markdown(f"- **Threshold:** {S.threshold:.2f}")
    else:
        # ───────────────── Clinical sidebar (compact, professional) ─────────
        st.markdown("### 🏥 Platform Status")
        if S.df is None:
            st.markdown("<div class='info-box'>No dataset loaded yet.</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='success-box'>Dataset loaded<br><strong>{len(S.df):,}</strong> patient records</div>",
                        unsafe_allow_html=True)
            if S.analysis_done:
                st.markdown("<div class='success-box'>Risk analysis complete</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<div class='warn-box'>Analysis pending<br>Click <em>Analyze Dataset</em> in the main panel.</div>",
                            unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            "<div style='font-size:12.5px;color:#475569;line-height:1.55;'>"
            "CareInsight is a clinical decision-support tool for 30-day "
            "readmission risk. Predictions must be reviewed by qualified "
            "clinical staff before any action."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("### 🛡️ Compliance")
        st.markdown(
            "<div style='font-size:11.5px;color:#64748b;line-height:1.55;'>"
            "• PHI handled per HIPAA<br>"
            "• Data processed in-session only<br>"
            "• Audit log available on request"
            "</div>",
            unsafe_allow_html=True,
        )

# ═════════════════════════════════════════════════════════════════════════════
# ░░  TECHNICAL VIEW  ░░  — full existing backend (tabs preserved)
# ═════════════════════════════════════════════════════════════════════════════
if S.view_mode == 'technical':
    # Header
    st.markdown("""
    <div class='app-hdr'>
      <div class='badge'>Healthcare AI &middot; Binary Classification</div>
      <h1>🏥 Hospital Readmission Risk Predictor</h1>
      <p>
        Augmented AI platform for clinical teams — upload patient data,
        train three ML models, surface SHAP explanations,
        audit fairness across subgroups, and export professional PDF clinical memos.
      </p>
    </div>
    """, unsafe_allow_html=True)

    tab_data, tab_roc, tab_shap, tab_score, tab_calib, tab_thr, tab_memo = st.tabs([
        "📁 Data Preview",
        "📈 ROC Curves",
        "🔬 Explainability",
        "🎯 Risk Scoring",
        "⚖️ Calibration & Fairness",
        "🎚️ Threshold Policy",
        "📄 Clinical Memo",
    ])

    # ── Tab 0: Data Preview ───────────────────────────────────────────────────
    with tab_data:
        st.markdown("<p class='section-head'>Dataset Overview</p>", unsafe_allow_html=True)
        if S.df is None:
            st.markdown("""
            <div class='info-box'>
              👈 Upload a CSV file using the sidebar to get started.<br>
              Any CSV with numeric and/or categorical columns is supported.
              Missing values are auto-imputed. ID-like columns are excluded automatically.
            </div>
            """, unsafe_allow_html=True)
        else:
            df = S.df
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{len(df):,}")
            c2.metric("Columns", f"{len(df.columns)}")
            c3.metric("Numeric cols", f"{len(df.select_dtypes(include='number').columns)}")
            c4.metric("Categorical cols", f"{len(df.select_dtypes(include='object').columns)}")

            st.markdown("<p class='section-head'>Data Preview (first 5 rows)</p>", unsafe_allow_html=True)
            st.dataframe(df.head(5), use_container_width=True)

            st.markdown("<p class='section-head'>Descriptive Statistics</p>", unsafe_allow_html=True)
            st.dataframe(df.describe(include='all').round(3), use_container_width=True)

            if S.trained:
                st.markdown(f"""
                <div class='success-box'>
                  Models trained! Best model: <strong>{S.best_name}</strong>
                  (Positive class = <code>{S.pos_label}</code>)
                </div>
                """, unsafe_allow_html=True)

                # Metrics table
                rows_html = ''
                for nm, m in S.metrics.items():
                    star  = ' ★' if nm == S.best_name else ''
                    color = '#0369a1' if nm == S.best_name else 'inherit'
                    rows_html += (f'<tr style="color:{color}">'
                        f'<td><strong>{nm}{star}</strong></td>'
                        f'<td>{m["auc"]:.4f}</td><td>{m["ap"]:.4f}</td>'
                        f'<td>{m["f1"]:.4f}</td><td>{m["acc"]:.4f}</td>'
                        f'<td>{m["bal"]:.4f}</td><td>{m["brier"]:.4f}</td></tr>')

                if S.metrics[S.best_name]['auc'] < 0.60:
                    st.markdown(f"""
                    <div class='signal-warn'>
                      <strong>⚠️ Low Predictive Signal Detected</strong><br>
                      The best AUC-ROC is {S.metrics[S.best_name]['auc']:.3f} (near 0.50 = random).
                      This typically means the dataset labels do not correlate with features —
                      common in purely synthetic/randomly-generated datasets.
                      With a real clinical dataset, AUC values of 0.75–0.90 are typical.
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <table class='dfp'>
                  <thead><tr>
                    <th>Model</th><th>AUC-ROC</th><th>AUC-PR</th>
                    <th>F1</th><th>Accuracy</th><th>Balanced Acc</th><th>Brier Score</th>
                  </tr></thead>
                  <tbody>{rows_html}</tbody>
                </table>
                """, unsafe_allow_html=True)

    # ── Tab 1: ROC Curves ─────────────────────────────────────────────────────
    with tab_roc:
        st.markdown("<p class='section-head'>Model Performance Curves</p>", unsafe_allow_html=True)
        st.markdown("<p class='section-sub'>ROC and Precision-Recall curves for all three models.</p>",
                    unsafe_allow_html=True)

        if not S.trained:
            st.markdown("<div class='warn-box'>Train models first using the sidebar.</div>",
                        unsafe_allow_html=True)
        else:
            y_te = S.y_test
            fig = make_subplots(1, 2,
                                subplot_titles=['ROC Curves (AUC-ROC)', 'Precision-Recall Curves (AP)'])
            for nm, prbs in S.probs.items():
                c = COLORS.get(nm, '#888')
                fpr_, tpr_, _ = roc_curve(y_te, prbs)
                prec, rec,  _ = precision_recall_curve(y_te, prbs)
                auc = S.metrics[nm]['auc']
                ap  = S.metrics[nm]['ap']
                fig.add_trace(go.Scatter(x=fpr_, y=tpr_, name=f'{nm}  (AUC={auc:.3f})',
                                         line=dict(color=c, width=2.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=rec, y=prec, name=f'{nm}  (AP={ap:.3f})',
                                         showlegend=False,
                                         line=dict(color=c, width=2.5, dash='dot')), row=1, col=2)
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                     line=dict(color='#94a3b8', dash='dash', width=1),
                                     name='Random', showlegend=False), row=1, col=1)
            fig.update_xaxes(title_text='False Positive Rate', row=1, col=1)
            fig.update_yaxes(title_text='True Positive Rate',  row=1, col=1)
            fig.update_xaxes(title_text='Recall',              row=1, col=2)
            fig.update_yaxes(title_text='Precision',           row=1, col=2)
            fig.update_layout(height=460, title_text='Model Performance Curves',
                              plot_bgcolor='#f8fafc', paper_bgcolor='#ffffff',
                              legend=dict(x=0.01, y=0.01))
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Explainability ─────────────────────────────────────────────────
    with tab_shap:
        st.markdown("<p class='section-head'>Global Explainability — SHAP Feature Importance</p>",
                    unsafe_allow_html=True)
        st.markdown("<p class='section-sub'>Which features drive readmission risk across the entire patient population?</p>",
                    unsafe_allow_html=True)

        if not S.trained:
            st.markdown("<div class='warn-box'>Train models first using the sidebar.</div>",
                        unsafe_allow_html=True)
        else:
            model_choice = st.selectbox("Select model for SHAP", list(S.models.keys()),
                                         index=list(S.models.keys()).index(S.best_name))
            S.chosen_name = model_choice

            if st.button("🔬 Compute SHAP Values"):
                with st.spinner(f"Computing SHAP for {model_choice}… (~30 s)"):
                    exp, sv = compute_shap(model_choice, S.models[model_choice],
                                           S.X_test, S.feature_names)
                    S.explainer   = exp
                    S.shap_vals   = sv
                    S.shap_computed = sv is not None

            if S.shap_computed and S.shap_vals is not None:
                mean_abs = np.abs(S.shap_vals).mean(axis=0)
                top_idx  = np.argsort(mean_abs)[::-1][:15]
                feats    = [S.feature_names[i] for i in top_idx]
                vals     = mean_abs[top_idx]

                fig = go.Figure(go.Bar(
                    x=vals[::-1], y=feats[::-1], orientation='h',
                    marker_color=[COLORS.get(S.chosen_name, '#0284c7')] * len(feats),
                    text=[f'{v:.4f}' for v in vals[::-1]], textposition='outside'))
                fig.update_layout(
                    title=f'Global Feature Importance — {S.chosen_name}  (Mean |SHAP|)',
                    xaxis_title='Mean |SHAP Value|', height=520,
                    plot_bgcolor='#f8fafc', paper_bgcolor='#ffffff',
                    margin=dict(l=200, r=90, t=60, b=40))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("<div class='success-box'>SHAP complete. Go to Risk Scoring for patient-level explanations.</div>",
                            unsafe_allow_html=True)

    # ── Tab 3: Risk Scoring ───────────────────────────────────────────────────
    with tab_score:
        st.markdown("<p class='section-head'>Patient Risk Scoring</p>", unsafe_allow_html=True)
        st.markdown("""<p class='section-sub'>
          Select a patient row from the test set to score their 30-day readmission risk.
          The gauge shows the predicted probability and risk tier.
        </p>""", unsafe_allow_html=True)

        if not S.trained:
            st.markdown("<div class='warn-box'>Train models first using the sidebar.</div>",
                        unsafe_allow_html=True)
        else:
            n_test = len(S.X_test)
            col_a, col_b = st.columns([3, 1])
            patient_idx = col_a.slider("Patient row # (test set)", 0, n_test - 1, 0)
            score_btn   = col_b.button("🎯 Score Patient", use_container_width=True)

            if score_btn or S.last_idx is not None:
                idx  = patient_idx
                nm   = S.chosen_name or S.best_name
                x    = S.X_test[idx:idx + 1]
                prob = float(S.models[nm].predict_proba(x)[0, 1])
                cat, cls, color = risk_label(prob, S.threshold)
                S.last_prob = prob
                S.last_cat  = cat
                S.last_idx  = idx

                # Gauge
                thr = S.threshold
                fig = go.Figure(go.Indicator(
                    mode='gauge+number',
                    value=round(prob * 100, 1),
                    title={'text': (f'Readmission Risk Score<br>'
                                    f'<span style="font-size:.75em;color:#64748b">'
                                    f'Patient #{idx} | Model: {nm} | Threshold: {thr:.2f}</span>')},
                    number={'suffix': '%', 'font': {'size': 42, 'color': color}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar':  {'color': color, 'thickness': 0.28},
                        'bgcolor': 'white',
                        'steps': [
                            {'range': [0,         thr * 40],    'color': '#dcfce7'},
                            {'range': [thr * 40,  thr * 100],   'color': '#fef9c3'},
                            {'range': [thr * 100, 100],         'color': '#fee2e2'},
                        ],
                        'threshold': {'line': {'color': '#1e293b', 'width': 3},
                                      'thickness': 0.75, 'value': thr * 100},
                    }))
                fig.update_layout(height=320, paper_bgcolor='#ffffff',
                                  margin=dict(t=70, b=20, l=30, r=30))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"""
                <div style='text-align:center;margin:12px 0 20px;'>
                  <span class='{cls}'>{cat}</span>&nbsp;&nbsp;
                  <span style='color:#64748b;font-size:14px'>
                    Probability: <strong>{prob:.1%}</strong>
                  </span>
                </div>
                """, unsafe_allow_html=True)

                # Local SHAP
                st.markdown("---")
                st.markdown("<p class='section-head'>Local SHAP Explanation</p>", unsafe_allow_html=True)
                if S.shap_vals is None:
                    st.markdown("<div class='info-box'>Compute SHAP in the Explainability tab first to see patient-level explanations.</div>",
                                unsafe_allow_html=True)
                else:
                    sv    = S.shap_vals[idx]
                    feats = S.feature_names
                    top_i = np.argsort(np.abs(sv))[::-1][:12]
                    t_f   = [feats[i] for i in top_i]
                    t_v   = [sv[i]    for i in top_i]
                    colors = ['#dc2626' if v > 0 else '#16a34a' for v in t_v]
                    fig = go.Figure(go.Bar(
                        x=t_v[::-1], y=t_f[::-1], orientation='h',
                        marker_color=colors[::-1],
                        text=[f'{v:+.4f}' for v in t_v[::-1]], textposition='outside'))
                    fig.update_layout(
                        title=f'Local Explanation — Patient #{idx}  (Red = increases risk, Green = decreases)',
                        xaxis_title='SHAP Value', height=440,
                        plot_bgcolor='#f8fafc', paper_bgcolor='#ffffff',
                        margin=dict(l=210, r=90, t=60, b=40))
                    st.plotly_chart(fig, use_container_width=True)
                    S.last_top_factors = list(zip(t_f, t_v))

                # Checklist
                st.markdown("---")
                tf  = S.last_top_factors or []
                cat_for_cl = S.last_cat or 'MEDIUM RISK'
                cl  = generate_checklist(tf, cat_for_cl)
                S.last_checklist = cl
                st.markdown("<p class='section-head'>Preventive Action Checklist</p>", unsafe_allow_html=True)
                st.markdown("<p class='section-sub'>Evidence-based actions tailored to this patient's risk factors</p>",
                            unsafe_allow_html=True)
                items_html = ''.join(f"<div class='chk-item'>&#9744;&nbsp;&nbsp;{item}</div>" for item in cl)
                st.markdown(items_html, unsafe_allow_html=True)

    # ── Tab 4: Calibration & Fairness ─────────────────────────────────────────
    with tab_calib:
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("<p class='section-head'>Calibration Curves</p>", unsafe_allow_html=True)
            st.markdown("<p class='section-sub'>A well-calibrated model follows the diagonal. Brier Score (lower = better).</p>",
                        unsafe_allow_html=True)

            if not S.trained:
                st.markdown("<div class='warn-box'>Train models first.</div>", unsafe_allow_html=True)
            else:
                if st.button("📊 Plot Calibration"):
                    fig, ax = plt.subplots(figsize=(7, 4.5))
                    ax.plot([0,1],[0,1],'k--',lw=1.5,alpha=0.45,label='Perfect calibration')
                    for nm, prbs in S.probs.items():
                        try:
                            frac_pos, mean_pred = calibration_curve(S.y_test, prbs, n_bins=10)
                            brier = S.metrics[nm]['brier']
                            ax.plot(mean_pred, frac_pos, 's-',
                                    color=COLORS.get(nm,'#888'), lw=2,
                                    label=f'{nm}  (Brier={brier:.3f})')
                        except Exception:
                            pass
                    ax.set_xlabel('Mean Predicted Probability')
                    ax.set_ylabel('Fraction of Positives')
                    ax.set_title('Calibration Curves')
                    ax.legend(fontsize=9)
                    ax.set_facecolor('#f8fafc')
                    fig.patch.set_facecolor('#ffffff')
                    st.pyplot(fig)
                    plt.close(fig)

        with col_right:
            st.markdown("<p class='section-head'>Fairness Audit</p>", unsafe_allow_html=True)
            st.markdown("<p class='section-sub'>Detect performance disparities across demographic subgroups.</p>",
                        unsafe_allow_html=True)

            if not S.trained:
                st.markdown("<div class='warn-box'>Train models first.</div>", unsafe_allow_html=True)
            else:
                if st.button("⚖️ Run Fairness Audit"):
                    nm = S.chosen_name or S.best_name
                    if not S.sensitive_cols or S.X_test_raw is None:
                        st.markdown("<div class='warn-box'>Select fairness columns in the sidebar first.</div>",
                                    unsafe_allow_html=True)
                    else:
                        raw   = S.X_test_raw
                        avail = [c for c in S.sensitive_cols if c in raw.columns]
                        fair_df = compute_fairness(raw, S.y_test, S.probs[nm], avail, S.threshold)
                        if fair_df.empty:
                            st.markdown("<div class='warn-box'>Not enough samples per subgroup (min 10 required).</div>",
                                        unsafe_allow_html=True)
                        else:
                            st.dataframe(fair_df, use_container_width=True)
                            st.markdown("""
                            <div class='info-box'>
                              <strong>Equalized Odds:</strong> TPR and FPR should be similar across groups.<br>
                              <strong>Demographic Parity:</strong> Predicted Positive Rate should be similar across groups.
                            </div>
                            """, unsafe_allow_html=True)

    # ── Tab 5: Threshold Policy ────────────────────────────────────────────────
    with tab_thr:
        st.markdown("<p class='section-head'>Decision Threshold Policy</p>", unsafe_allow_html=True)
        st.markdown("""<p class='section-sub'>
          Adjust the classification threshold to balance sensitivity and specificity.
          Lower threshold = catches more at-risk patients (higher sensitivity) but more false alarms.
        </p>""", unsafe_allow_html=True)

        if not S.trained:
            st.markdown("<div class='warn-box'>Train models first using the sidebar.</div>",
                        unsafe_allow_html=True)
        else:
            col_opt, col_btn = st.columns([3, 1])
            opt_target = col_opt.selectbox("Auto-optimise for",
                                            ['Sensitivity', 'Specificity', 'F1', 'Balanced'])
            if col_btn.button("⚡ Auto-Optimise"):
                nm    = S.chosen_name or S.best_name
                probs = S.probs[nm]
                y_te  = S.y_test
                best_thr, best_score = 0.5, -1.0
                for thr_val in np.linspace(0.01, 0.99, 197):
                    m = threshold_metrics(y_te, probs, thr_val)
                    score = {
                        'Sensitivity': m['Sensitivity'], 'Specificity': m['Specificity'],
                        'F1': m['F1'], 'Balanced': (m['Sensitivity'] + m['Specificity']) / 2,
                    }[opt_target]
                    if score > best_score:
                        best_score, best_thr = score, thr_val
                S.threshold = round(best_thr, 2)
                st.success(f"Optimal threshold for {opt_target}: **{S.threshold:.2f}** (score = {best_score:.3f})")

            # Show live metrics
            nm  = S.chosen_name or S.best_name
            thr = S.threshold
            m   = threshold_metrics(S.y_test, S.probs[nm], thr)

            st.markdown(f"<p class='section-head'>At threshold {thr:.2f} — {nm}</p>", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sensitivity (TPR)", f"{m['Sensitivity']:.1%}")
            c2.metric("Specificity (TNR)", f"{m['Specificity']:.1%}")
            c3.metric("Precision (PPV)",   f"{m['PPV']:.1%}")
            c4.metric("Neg. Pred. Value",  f"{m['NPV']:.1%}")

            c5, c6, c7, c8, c9 = st.columns(5)
            c5.metric("F1 Score",        f"{m['F1']:.3f}")
            c6.metric("True Positives",  f"{m['TP']}", delta_color="off")
            c7.metric("True Negatives",  f"{m['TN']}", delta_color="off")
            c8.metric("False Positives", f"{m['FP']}", delta_color="off")
            c9.metric("False Negatives", f"{m['FN']}", delta_color="off")

    # ── Tab 6: Clinical Memo PDF ───────────────────────────────────────────────
    with tab_memo:
        st.markdown("<p class='section-head'>Clinical Memo Export</p>", unsafe_allow_html=True)
        st.markdown("""<p class='section-sub'>
          Fill in patient details and export a PDF clinical memo with the risk score,
          top SHAP factors, preventive action checklist, and clinical recommendations.
        </p>""", unsafe_allow_html=True)

        if not S.trained:
            st.markdown("<div class='warn-box'>Train models first using the sidebar.</div>",
                        unsafe_allow_html=True)
        elif S.last_prob is None:
            st.markdown("<div class='info-box'>Score a patient in the <strong>Risk Scoring</strong> tab first, then return here to export.</div>",
                        unsafe_allow_html=True)
        else:
            col_l, col_r = st.columns(2)
            pt_name = col_l.text_input("Patient Name",      placeholder="e.g. Jane Smith")
            pt_id   = col_r.text_input("Patient ID",        placeholder="e.g. MRN-00123")
            pt_dob  = col_l.text_input("Date of Birth",     placeholder="YYYY-MM-DD")
            pt_ward = col_r.text_input("Ward / Unit",       placeholder="e.g. Cardiology")
            pt_phys = col_l.text_input("Attending Physician", placeholder="e.g. Dr. A. Hassan")

            st.markdown("")
            if st.button("📄 Export Clinical Memo PDF", use_container_width=False):
                patient_info = {
                    'Patient Name':        pt_name or 'N/A',
                    'Patient ID':          pt_id   or 'N/A',
                    'Date of Birth':       pt_dob  or 'N/A',
                    'Ward / Unit':         pt_ward or 'N/A',
                    'Attending Physician': pt_phys or 'N/A',
                    'Report Date':         datetime.now().strftime('%Y-%m-%d'),
                }
                with st.spinner("Generating PDF…"):
                    try:
                        pdf_bytes = generate_pdf(
                            patient_info,
                            S.last_prob,
                            S.last_cat,
                            S.last_top_factors or [],
                            S.last_checklist or BASE_CHECKLIST,
                            S.chosen_name or 'Model',
                            S.threshold,
                        )
                        pid = (pt_id or 'patient').replace(' ', '_')
                        st.download_button(
                            label="⬇ Download Clinical Memo PDF",
                            data=pdf_bytes,
                            file_name=f"readmission_memo_{pid}.pdf",
                            mime="application/pdf",
                            use_container_width=False,
                        )
                        st.markdown("<div class='success-box'>Clinical memo generated successfully!</div>",
                                    unsafe_allow_html=True)
                    except Exception as e:
                        import traceback
                        st.error(f"PDF error: {e}\n{traceback.format_exc()}")

            st.markdown("""
            <div class='info-box' style='margin-top:16px;'>
              The memo includes: patient info, risk score gauge, top SHAP risk factors,
              preventive action checklist, and clinical recommendations.
              Marked <strong>CONFIDENTIAL / PHI</strong> per HIPAA.
            </div>
            """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# ░░  CLINICAL STAFF VIEW  ░░  — deployment-ready, non-technical UI
# ═════════════════════════════════════════════════════════════════════════════
else:
    # ── Branded hero header ──────────────────────────────────────────────────
    st.markdown(f"""
    <div class='clinical-hero'>
      <div class='brand-left'>
        <div class='brand-mark'>🏥</div>
        <div>
          <h1>CareInsight · Readmission Risk Platform</h1>
          <p class='tagline'>Clinical decision support for discharge planning &middot; 30-day readmission risk</p>
        </div>
      </div>
      <div class='brand-right'>
        <div class='version'>v1.0 · Clinical Release</div>
        <div class='org'>For authorised clinical staff</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────── STATE 1: No dataset uploaded ────────────────────────────────
    if S.df is None:
        st.markdown("""
        <div class='upload-wrap'>
          <div class='upload-icon'>📋</div>
          <div class='upload-title'>Upload your patient dataset</div>
          <div class='upload-desc'>Select a CSV file containing patient records to begin risk analysis.</div>
          <div class='upload-hint'>Data remains in your session and is not transmitted externally.</div>
        </div>
        """, unsafe_allow_html=True)

        up = st.file_uploader(
            "Upload patient CSV",
            type=["csv"],
            label_visibility="collapsed",
            key="clin_uploader",
        )
        if up is not None:
            try:
                S.df = pd.read_csv(up)
                S.analysis_done = False
                S.all_probs = None
                S.all_shap = None
                S.clinical_selected_idx = None
                st.rerun()
            except Exception as e:
                st.error(f"We couldn't read that file: {e}")

        st.markdown("""
        <div class='footer-compliance'>
          <strong>CONFIDENTIAL &middot; Protected Health Information (PHI)</strong><br>
          Handle all data in accordance with HIPAA and your institution's privacy policy.
          Risk scores are decision-support tools and must be reviewed by qualified clinical staff before any action.
        </div>
        """, unsafe_allow_html=True)

    # ─────────── STATE 2: Dataset uploaded, awaiting analysis ─────────────────
    elif not S.analysis_done:
        # Determine default target + patient ID
        cols = list(S.df.columns)
        auto_target = cols[-1]
        for c in cols:
            if any(k in c.lower() for k in ['readmit', 'target', 'label', 'outcome', 'risk']):
                auto_target = c
                break
        auto_id = detect_patient_id_column(S.df)

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-title'>Dataset received</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='panel-sub'>"
            f"{len(S.df):,} patient records &middot; {len(S.df.columns)} data columns detected. "
            f"Review the settings below and start the analysis when ready."
            f"</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(S.df.head(5), use_container_width=True, height=220)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("⚙️  Data settings  (optional — sensible defaults detected)", expanded=False):
            sel_target = st.selectbox(
                "Which column is the readmission / outcome label?",
                cols, index=cols.index(auto_target),
                help="Binary column indicating the outcome (e.g. readmitted Yes/No).",
            )
            id_options = ["(none — show row numbers)"] + cols
            default_id_idx = id_options.index(auto_id) if (auto_id in cols) else 0
            sel_id = st.selectbox(
                "Patient identifier column",
                id_options, index=default_id_idx,
                help="Optional. Used to label patients in the roster.",
            )
            S.clinical_id_col = None if sel_id.startswith("(none") else sel_id
        if S.clinical_id_col is None:
            S.clinical_id_col = auto_id  # may still be None

        # Locked-in values (from expander if expanded, else defaults)
        final_target = sel_target if 'sel_target' in dir() else auto_target

        st.markdown("<div class='big-cta'>", unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            go_btn = st.button(
                "🧠  Analyze Dataset",
                use_container_width=True,
                help="Build patient risk scores and identify top clinical drivers.",
            )
        st.markdown("</div>", unsafe_allow_html=True)

        if go_btn:
            try:
                progress = st.progress(0, text="Preparing patient data…")
                out = None
                with st.spinner("Analyzing patient cohort… this typically takes under a minute."):
                    progress.progress(15, text="Preprocessing features…")
                    out = compute_analysis(S.df, final_target)
                    progress.progress(75, text="Identifying key clinical drivers…")
                # Persist results into shared session state
                S.target_col    = final_target
                S.cat_cols      = out['cat_cols']
                S.num_cols      = out['num_cols']
                S.le_dict       = out['le_dict']
                S.pos_label     = out['pos_label']
                S.feature_names = out['feats']
                S.X_train       = out['X_tr']
                S.X_test        = out['X_te']
                S.y_train       = out['y_tr']
                S.y_test        = out['y_te']
                S.X_test_raw    = out['X_test_raw']
                S.models        = {k: v['model'] for k, v in out['results'].items()}
                S.probs         = {k: v['probs'] for k, v in out['results'].items()}
                S.metrics       = {k: {m: v[m] for m in ['auc','ap','f1','acc','brier','bal']}
                                   for k, v in out['results'].items()}
                S.best_name     = out['best']
                S.chosen_name   = out['best']
                S.explainer     = out['explainer']
                S.shap_vals     = out['shap_vals']
                S.shap_computed = out['shap_vals'] is not None
                S.all_probs     = out['all_probs']
                S.trained       = True
                S.analysis_done = True
                progress.progress(100, text="Analysis complete.")
                st.rerun()
            except Exception as e:
                import traceback
                st.error(f"Analysis could not be completed: {e}")
                st.code(traceback.format_exc())

        st.markdown("""
        <div class='footer-compliance'>
          <strong>CONFIDENTIAL &middot; Protected Health Information (PHI)</strong> &middot;
          Risk scores are decision-support and must be reviewed by qualified clinical staff.
        </div>
        """, unsafe_allow_html=True)

    # ─────────── STATE 3: Analysis complete — roster / drill-in ───────────────
    else:
        df = S.df
        thr = S.threshold
        n_total = len(df)
        probs = np.asarray(S.all_probs)
        tiers = np.array([risk_tier(p, thr) for p in probs])

        n_high = int((tiers == 'High').sum())
        n_med  = int((tiers == 'Medium').sum())
        n_low  = int((tiers == 'Low').sum())

        # ── If a patient is selected, render the drill-in view ───────────────
        if S.clinical_selected_idx is not None and 0 <= S.clinical_selected_idx < n_total:
            idx = int(S.clinical_selected_idx)

            # Identify patient
            pid_col = S.clinical_id_col
            pid = str(df.iloc[idx][pid_col]) if (pid_col and pid_col in df.columns) else f"Row {idx}"

            # Back button
            c_back, _ = st.columns([1, 5])
            if c_back.button("← Back to Patient Roster", use_container_width=True):
                S.clinical_selected_idx = None
                st.rerun()

            # Build prediction for this patient using the raw df (re-encode on the fly)
            # We reuse S.all_probs which was computed on the processed dataset in-order.
            prob = float(S.all_probs[idx])
            cat, cls, color = risk_label(prob, thr)
            tier = risk_tier(prob, thr)

            # Drill header
            age_val = ''
            for c in df.columns:
                if c.lower() == 'age':
                    age_val = f"Age {df.iloc[idx][c]}"
                    break
            gender_val = ''
            for c in df.columns:
                if c.lower() in ('gender', 'sex'):
                    gender_val = str(df.iloc[idx][c])
                    break
            meta_bits = [b for b in [age_val, gender_val] if b]
            meta_str = ' &middot; '.join(meta_bits) if meta_bits else 'Patient details from record'

            st.markdown(f"""
            <div class='drill-header'>
              <div class='patient-id'>Patient Record</div>
              <div class='patient-name'>{pid}</div>
              <div class='patient-meta'>{meta_str}</div>
            </div>
            """, unsafe_allow_html=True)

            # Two-column layout: gauge + top drivers
            col_gauge, col_drivers = st.columns([1, 1.1])

            with col_gauge:
                st.markdown("<div class='panel'>", unsafe_allow_html=True)
                st.markdown("<div class='panel-title'>30-day readmission risk</div>", unsafe_allow_html=True)
                st.markdown("<div class='panel-sub'>Predicted probability and clinical tier.</div>",
                            unsafe_allow_html=True)

                fig = go.Figure(go.Indicator(
                    mode='gauge+number',
                    value=round(prob * 100, 1),
                    number={'suffix': '%', 'font': {'size': 44, 'color': color}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#94a3b8'},
                        'bar':  {'color': color, 'thickness': 0.28},
                        'bgcolor': 'white',
                        'borderwidth': 1, 'bordercolor': '#e2e8f0',
                        'steps': [
                            {'range': [0,         thr * 40],    'color': '#dcfce7'},
                            {'range': [thr * 40,  thr * 100],   'color': '#fef9c3'},
                            {'range': [thr * 100, 100],         'color': '#fee2e2'},
                        ],
                        'threshold': {'line': {'color': '#1e293b', 'width': 3},
                                      'thickness': 0.75, 'value': thr * 100},
                    }))
                fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)',
                                  margin=dict(t=10, b=10, l=30, r=30))
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    f"<div style='text-align:center;margin-top:2px;'>"
                    f"<span class='{cls}'>{cat}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with col_drivers:
                st.markdown("<div class='panel'>", unsafe_allow_html=True)
                st.markdown("<div class='panel-title'>Key risk drivers for this patient</div>",
                            unsafe_allow_html=True)
                st.markdown("<div class='panel-sub'>Factors that most influenced this score, translated from the model.</div>",
                            unsafe_allow_html=True)

                # SHAP was computed only on the test split (to keep runtime bounded).
                # If this patient is not in the test split, fall back to global importance.
                top_factors_for_pdf = []
                patient_in_test_split = False
                if S.X_test_raw is not None:
                    # Try to find this row index inside X_test_raw by matching row values
                    try:
                        # We preserved df.iloc[test_idx] → X_test_raw; find index in test_idx
                        # Use a simple equality check on the raw row (string-safe)
                        row_key = tuple(df.iloc[idx].astype(str).tolist())
                        test_rows = [tuple(S.X_test_raw.iloc[i].astype(str).tolist())
                                     for i in range(len(S.X_test_raw))]
                        if row_key in test_rows:
                            test_i = test_rows.index(row_key)
                            sv = S.shap_vals[test_i]
                            patient_in_test_split = True
                    except Exception:
                        patient_in_test_split = False

                if patient_in_test_split and S.shap_vals is not None:
                    ranked = sorted(
                        enumerate(sv),
                        key=lambda kv: abs(kv[1]),
                        reverse=True,
                    )
                    top = ranked[:6]
                    for i, v in top:
                        feat = S.feature_names[i]
                        label = humanize_feature(feat)
                        direction_cls = 'up' if v > 0 else 'down'
                        arrow = '▲' if v > 0 else '▼'
                        effect = 'Increases risk' if v > 0 else 'Protective'
                        # Try to include the patient's value for this feature if it's in df
                        val_str = ''
                        if feat in df.columns:
                            try:
                                val_str = f" &middot; value: {df.iloc[idx][feat]}"
                            except Exception:
                                pass
                        st.markdown(
                            f"<div class='driver-chip {direction_cls}'>"
                            f"<span class='arrow'>{arrow}</span>"
                            f"<span class='label'>{label}</span>"
                            f"<span class='detail'>{effect}{val_str}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        top_factors_for_pdf.append((feat, float(v)))
                else:
                    # Fallback: show global top features from SHAP
                    if S.shap_vals is not None:
                        mean_abs = np.abs(S.shap_vals).mean(axis=0)
                        order = np.argsort(mean_abs)[::-1][:6]
                        for i in order:
                            feat = S.feature_names[i]
                            label = humanize_feature(feat)
                            st.markdown(
                                f"<div class='driver-chip'>"
                                f"<span class='arrow'>•</span>"
                                f"<span class='label'>{label}</span>"
                                f"<span class='detail'>Important across population</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            top_factors_for_pdf.append((feat, float(mean_abs[i])))
                        st.markdown(
                            "<div style='font-size:11.5px;color:#94a3b8;margin-top:6px;'>"
                            "Note: this patient was used during model training, so a "
                            "population-level importance view is shown."
                            "</div>",
                            unsafe_allow_html=True,
                        )
                st.markdown("</div>", unsafe_allow_html=True)

            # Checklist
            checklist = generate_checklist(top_factors_for_pdf, cat)

            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("<div class='panel-title'>Recommended preventive actions</div>",
                        unsafe_allow_html=True)
            st.markdown("<div class='panel-sub'>Evidence-based steps for this patient's discharge plan.</div>",
                        unsafe_allow_html=True)
            for item in checklist:
                st.markdown(f"<div class='chk-item'>&#9744;&nbsp;&nbsp;{item}</div>",
                            unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # PDF export
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("<div class='panel-title'>Export clinical report</div>",
                        unsafe_allow_html=True)
            st.markdown("<div class='panel-sub'>Generate a printable discharge memo for this patient.</div>",
                        unsafe_allow_html=True)

            col_l, col_r = st.columns(2)
            pt_name = col_l.text_input("Patient name (optional)", placeholder="e.g. Jane Smith",
                                       key=f"pdf_name_{idx}")
            pt_dob  = col_r.text_input("Date of birth (optional)", placeholder="YYYY-MM-DD",
                                       key=f"pdf_dob_{idx}")
            pt_ward = col_l.text_input("Ward / unit (optional)", placeholder="e.g. Cardiology",
                                       key=f"pdf_ward_{idx}")
            pt_phys = col_r.text_input("Attending physician (optional)", placeholder="e.g. Dr. A. Hassan",
                                       key=f"pdf_phys_{idx}")

            if st.button("📄  Generate clinical report (PDF)", use_container_width=False,
                         key=f"pdf_btn_{idx}"):
                patient_info = {
                    'Patient Name':        pt_name or 'N/A',
                    'Patient ID':          pid or 'N/A',
                    'Date of Birth':       pt_dob or 'N/A',
                    'Ward / Unit':         pt_ward or 'N/A',
                    'Attending Physician': pt_phys or 'N/A',
                    'Report Date':         datetime.now().strftime('%Y-%m-%d'),
                }
                # Stash current scoring context for PDF generator
                S.last_prob = prob
                S.last_cat  = cat
                S.last_idx  = idx
                S.last_top_factors = top_factors_for_pdf
                S.last_checklist = checklist
                with st.spinner("Generating PDF…"):
                    try:
                        pdf_bytes = generate_pdf(
                            patient_info, prob, cat,
                            top_factors_for_pdf, checklist,
                            S.chosen_name or S.best_name or 'Model',
                            thr,
                        )
                        safe_pid = re.sub(r'[^A-Za-z0-9_-]+', '_', pid)
                        st.download_button(
                            label="⬇  Download PDF",
                            data=pdf_bytes,
                            file_name=f"readmission_report_{safe_pid}.pdf",
                            mime="application/pdf",
                        )
                        st.markdown("<div class='success-box'>Report ready to download.</div>",
                                    unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Could not generate the PDF: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("""
            <div class='footer-compliance'>
              <strong>CONFIDENTIAL &middot; PHI</strong> &middot;
              Risk estimates are decision-support only. Final clinical judgement rests with the attending care team.
            </div>
            """, unsafe_allow_html=True)

        # ── Otherwise, render the roster dashboard ──────────────────────────
        else:
            # KPI row
            st.markdown(f"""
            <div class='stat-row'>
              <div class='stat-card'>
                <div class='stat-label'>Patients analyzed</div>
                <div class='stat-value'>{n_total:,}</div>
                <div class='stat-sub'>Full cohort scored</div>
              </div>
              <div class='stat-card hi'>
                <div class='stat-label'>High risk</div>
                <div class='stat-value hi'>{n_high:,}</div>
                <div class='stat-sub'>{(n_high/max(n_total,1)):.0%} of cohort</div>
              </div>
              <div class='stat-card md'>
                <div class='stat-label'>Medium risk</div>
                <div class='stat-value md'>{n_med:,}</div>
                <div class='stat-sub'>{(n_med/max(n_total,1)):.0%} of cohort</div>
              </div>
              <div class='stat-card lo'>
                <div class='stat-label'>Low risk</div>
                <div class='stat-value lo'>{n_low:,}</div>
                <div class='stat-sub'>{(n_low/max(n_total,1)):.0%} of cohort</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("<div class='panel-title'>Patient roster</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='panel-sub'>Filter by risk tier, search by patient ID, "
                "then select a row to open the patient's detail view.</div>",
                unsafe_allow_html=True,
            )

            # Filter + search
            fc, sc = st.columns([1.2, 2])
            with fc:
                tier_filter = st.radio(
                    "Risk tier",
                    ["All", "High", "Medium", "Low"],
                    horizontal=True,
                    index=["All","High","Medium","Low"].index(S.clinical_filter),
                    key="clin_filter",
                )
                S.clinical_filter = tier_filter
            with sc:
                search = st.text_input(
                    "Search patient ID",
                    value=S.clinical_search,
                    placeholder="Type an MRN or patient ID…",
                    key="clin_search",
                )
                S.clinical_search = search

            # Build roster table
            pid_col = S.clinical_id_col
            pid_series = (df[pid_col].astype(str) if (pid_col and pid_col in df.columns)
                          else pd.Series([f"Row {i}" for i in range(n_total)]))
            age_col = next((c for c in df.columns if c.lower() == 'age'), None)
            age_series = df[age_col].astype(str) if age_col else pd.Series(['—'] * n_total)

            # Key factor per row: use global importance label for all rows (fast);
            # we mark with up-arrow to indicate a contributing factor.
            if S.shap_vals is not None:
                mean_abs = np.abs(S.shap_vals).mean(axis=0)
                global_top = S.feature_names[int(np.argmax(mean_abs))]
                key_factor_label = '▲ ' + humanize_feature(global_top)
            else:
                key_factor_label = '—'

            roster_df = pd.DataFrame({
                'Patient':     pid_series.values,
                'Age':         age_series.values,
                'Key factor':  [key_factor_label] * n_total,
                'Risk %':      np.round(probs * 100, 1),
                'Risk tier':   tiers,
                '_idx':        np.arange(n_total),
            })

            # Apply filters
            if S.clinical_filter != 'All':
                roster_df = roster_df[roster_df['Risk tier'] == S.clinical_filter]
            if S.clinical_search.strip():
                q = S.clinical_search.strip().lower()
                roster_df = roster_df[roster_df['Patient'].str.lower().str.contains(q, na=False)]

            # Sort: highest risk first
            roster_df = roster_df.sort_values(['Risk %'], ascending=False).reset_index(drop=True)

            # Cap display to avoid overwhelming the browser
            display_cap = 500
            if len(roster_df) > display_cap:
                roster_df_show = roster_df.head(display_cap).copy()
                st.markdown(
                    f"<div class='info-box'>Showing top <strong>{display_cap}</strong> of "
                    f"<strong>{len(roster_df):,}</strong> matches, sorted by risk. Use search or filters to narrow down.</div>",
                    unsafe_allow_html=True,
                )
            else:
                roster_df_show = roster_df.copy()

            # Render dataframe with row selection
            display_cols = ['Patient', 'Age', 'Key factor', 'Risk %', 'Risk tier']
            event = st.dataframe(
                roster_df_show[display_cols],
                use_container_width=True,
                hide_index=True,
                height=420,
                on_select="rerun",
                selection_mode="single-row",
                key="clin_roster",
                column_config={
                    'Risk %': st.column_config.ProgressColumn(
                        'Risk %',
                        help='Predicted 30-day readmission probability',
                        min_value=0, max_value=100, format='%.1f%%',
                    ),
                    'Risk tier': st.column_config.TextColumn(
                        'Tier',
                        help='Clinical risk tier at the current decision threshold',
                    ),
                },
            )

            # Handle selection → open drill-in
            try:
                selected_rows = event.selection.rows if hasattr(event, 'selection') else []
            except Exception:
                selected_rows = []
            if selected_rows:
                sel_pos = selected_rows[0]
                if 0 <= sel_pos < len(roster_df_show):
                    S.clinical_selected_idx = int(roster_df_show.iloc[sel_pos]['_idx'])
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)  # close panel

            # Quick actions
            col_l, col_r = st.columns([1, 1])
            with col_l:
                if st.button("🔄  Load a different dataset", use_container_width=True):
                    S.df = None
                    S.trained = False
                    S.analysis_done = False
                    S.all_probs = None
                    S.clinical_selected_idx = None
                    st.rerun()
            with col_r:
                # Download roster as CSV for case coordinators
                roster_out = roster_df[['Patient', 'Age', 'Risk %', 'Risk tier']].copy()
                csv_bytes = roster_out.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇  Download roster (CSV)",
                    data=csv_bytes,
                    file_name=f"patient_roster_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.markdown("""
            <div class='footer-compliance'>
              <strong>CONFIDENTIAL &middot; Protected Health Information (PHI)</strong> &middot;
              CareInsight is a decision-support tool. All clinical decisions must be reviewed and
              authorised by qualified medical staff.
            </div>
            """, unsafe_allow_html=True)
