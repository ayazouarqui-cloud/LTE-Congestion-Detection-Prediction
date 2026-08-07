
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# CONFIGURATION PAGE

st.set_page_config(
    page_title="Congestion BTS — Djezzy",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS GLOBAL
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

  :root {
    --blue-dark: #1a2d6b;
    --blue-main: #2563eb;
    --blue-light: #eff6ff;
    --blue-mid: #dbeafe;
    --orange: #f97316;
    --orange-light: #fff7ed;
    --green: #16a34a;
    --green-light: #f0fdf4;
    --red: #dc2626;
    --red-light: #fef2f2;
    --gray-50: #f8fafc;
    --gray-100: #f1f5f9;
    --gray-200: #e2e8f0;
    --gray-400: #94a3b8;
    --gray-600: #475569;
    --gray-800: #1e293b;
    --radius: 12px;
    --radius-sm: 8px;
  }

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--gray-800);
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #fff;
    border-right: 1px solid var(--gray-200);
    padding-top: 0;
  }
  [data-testid="stSidebar"] > div:first-child { padding-top: 0; }

  .sidebar-brand {
    background: var(--blue-dark);
    padding: 24px 20px 20px;
    margin: -1px -1px 0;
    color: #fff;
  }
  .sidebar-brand h2 { font-size: 15px; font-weight: 700; color: #fff; margin: 0 0 4px; }
  .sidebar-brand p { font-size: 11px; color: rgba(255,255,255,0.6); margin: 0; letter-spacing: .06em; }
  .sidebar-brand .djezzy-badge {
    display: inline-block; margin-top: 10px;
    background: var(--red); color: #fff;
    padding: 3px 10px; border-radius: 50px;
    font-size: 11px; font-weight: 700; letter-spacing: .08em;
  }

  /* ── Nav items ── */
  .nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 11px 20px; margin: 2px 8px;
    border-radius: var(--radius-sm); cursor: pointer;
    font-size: 13px; font-weight: 500; color: var(--gray-600);
    transition: all .15s; border: none; background: none; width: calc(100% - 16px);
    text-align: left;
  }
  .nav-item:hover { background: var(--gray-100); color: var(--blue-main); }
  .nav-item.active { background: var(--blue-light); color: var(--blue-main); font-weight: 600; }
  .nav-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; opacity: 0; transition: opacity .2s; }
  .nav-item.active .nav-dot { opacity: 1; }

  /* ── Main ── */
  .main .block-container { padding: 32px 40px 48px; max-width: 1200px; }
  [data-testid="stAppViewContainer"] > .main { background: var(--gray-50); }

  /* ── Page header ── */
  .page-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 36px; padding-bottom: 20px;
    border-bottom: 1px solid var(--gray-200);
  }
  .page-title-group { display: flex; align-items: center; gap: 14px; }
  .accent-bar { width: 4px; height: 36px; background: var(--blue-main); border-radius: 2px; flex-shrink: 0; }
  .page-icon-box {
    width: 38px; height: 38px; background: var(--blue-light);
    border-radius: 10px; display: flex; align-items: center;
    justify-content: center; font-size: 18px;
  }
  .page-title-text h2 { font-size: 21px; font-weight: 700; color: var(--blue-dark); margin: 0 0 2px; }
  .page-title-text p { font-size: 12px; color: var(--gray-400); margin: 0; }
  .page-tag {
    font-size: 11px; font-weight: 600; letter-spacing: .08em;
    color: var(--blue-main); background: var(--blue-light);
    padding: 5px 14px; border-radius: 20px;
  }

  /* ── Cards ── */
  .card {
    background: #fff; border: 1px solid var(--gray-200);
    border-radius: var(--radius); padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
    margin-bottom: 20px;
  }
  .card-title {
    font-size: 13px; font-weight: 600; color: var(--gray-800);
    margin-bottom: 16px; display: flex; align-items: center; gap: 8px;
  }

  /* ── KPI cards ── */
  .kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; margin-bottom: 24px; }
  .kpi-card {
    background: #fff; border: 1px solid var(--gray-200);
    border-radius: var(--radius); padding: 24px 20px; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
  }
  .kpi-value { font-size: 30px; font-weight: 800; color: var(--blue-main); line-height: 1; }
  .kpi-label { font-size: 11px; font-weight: 600; letter-spacing: .08em; color: var(--gray-400); margin-top: 8px; text-transform: uppercase; }

  /* ── Landing hero ── */
  .landing {
    text-align: center;
    background: linear-gradient(165deg, #f8fafc 0%, #eff6ff 100%);
    border-radius: var(--radius); padding: 64px 48px;
    border: 1px solid var(--gray-200);
    margin-bottom: 24px;
  }
  .landing-tag { font-size: 12px; font-weight: 600; letter-spacing: .12em; color: var(--blue-main); text-transform: uppercase; margin-bottom: 18px; }
  .landing-title { font-size: 42px; font-weight: 800; color: var(--blue-dark); line-height: 1.15; max-width: 720px; margin: 0 auto 16px; }
  .landing-title span { color: var(--blue-main); }
  .landing-sub { font-size: 15px; color: var(--gray-600); max-width: 580px; margin: 0 auto 36px; line-height: 1.7; }
  .landing-footer { display: flex; gap: 80px; justify-content: center; margin-top: 40px; padding-top: 32px; border-top: 1px solid var(--gray-200); }
  .lf-col { text-align: center; }
  .lf-label { font-size: 11px; font-weight: 600; letter-spacing: .1em; color: var(--blue-main); text-transform: uppercase; margin-bottom: 6px; }
  .lf-value { font-size: 14px; color: var(--gray-700); }

  /* ── Badges ── */
  .badge-normal { background: var(--green-light); color: var(--green); border: 1.5px solid #bbf7d0; padding: 8px 18px; border-radius: 50px; font-size: 13px; font-weight: 700; display: inline-block; }
  .badge-modere { background: var(--orange-light); color: var(--orange); border: 1.5px solid #fed7aa; padding: 8px 18px; border-radius: 50px; font-size: 13px; font-weight: 700; display: inline-block; }
  .badge-critique { background: var(--red-light); color: var(--red); border: 1.5px solid #fecaca; padding: 8px 18px; border-radius: 50px; font-size: 13px; font-weight: 700; display: inline-block; }

  /* ── Detection header ── */
  .detection-header {
    background: #fff; border: 1px solid var(--gray-200);
    border-radius: var(--radius); padding: 24px 28px;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,.06);
  }
  .cell-id-label { font-size: 11px; font-weight: 600; color: var(--gray-400); letter-spacing: .08em; text-transform: uppercase; margin-bottom: 4px; }
  .cell-id-value { font-size: 20px; font-weight: 700; color: var(--gray-800); }

  /* ── Metric rows ── */
  .metric-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid var(--gray-100); }
  .metric-row:last-child { border-bottom: none; }
  .metric-name { font-size: 13px; color: var(--gray-600); }
  .metric-val { font-size: 14px; font-weight: 700; color: var(--blue-dark); font-family: 'DM Mono', monospace; }
  .metric-val.orange { color: var(--orange); }

  /* ── SHAP bars ── */
  .shap-row { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
  .shap-name { font-size: 12px; color: var(--gray-700); min-width: 180px; font-family: 'DM Mono', monospace; }
  .shap-bar-bg { flex: 1; height: 20px; background: var(--gray-100); border-radius: 4px; overflow: hidden; }
  .shap-bar-pos { height: 100%; background: #fca5a5; border-radius: 4px; }
  .shap-bar-neg { height: 100%; background: #86efac; border-radius: 4px; }
  .shap-val-pos { font-size: 13px; font-weight: 700; color: var(--red); min-width: 50px; font-family: 'DM Mono', monospace; }
  .shap-val-neg { font-size: 13px; font-weight: 700; color: var(--green); min-width: 50px; font-family: 'DM Mono', monospace; }

  /* ── Horizon cards ── */
  .horizon-card {
    background: #fff; border: 1px solid var(--gray-200);
    border-radius: var(--radius); padding: 28px 24px;
    text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,.06);
    position: relative; overflow: hidden;
  }
  .horizon-card::before { content:''; position:absolute; top:0; left:0; right:0; height:4px; }
  .horizon-card.blue::before { background: var(--blue-main); }
  .horizon-card.orange::before { background: var(--orange); }
  .horizon-card.green::before { background: var(--green); }
  .horizon-label { font-size: 11px; font-weight: 600; letter-spacing: .1em; color: var(--gray-400); text-transform: uppercase; margin-bottom: 14px; }
  .horizon-value { font-size: 40px; font-weight: 800; color: var(--gray-800); line-height: 1; margin-bottom: 14px; }
  .horizon-bar-bg { height: 6px; background: var(--gray-100); border-radius: 50px; margin-bottom: 14px; overflow: hidden; }
  .horizon-bar-fill { height: 100%; border-radius: 50px; }
  .horizon-bar-fill.blue { background: var(--blue-main); }
  .horizon-bar-fill.orange { background: var(--orange); }
  .horizon-bar-fill.green { background: var(--green); }

  /* ── Pipeline ── */
  .pipeline-steps { display: flex; align-items: flex-start; gap: 0; margin: 40px 0; position: relative; }
  .pipeline-step { flex: 1; text-align: center; position: relative; }
  .pipeline-step::after {
    content: ''; position: absolute; top: 28px; left: 50%; right: -50%;
    height: 3px; background: var(--gray-200); z-index: 0;
  }
  .pipeline-step:last-child::after { display: none; }
  .pipeline-step.done::after { background: var(--green); }
  .pipeline-step.active::after { background: linear-gradient(90deg, var(--green), var(--gray-200)); }
  .step-circle {
    width: 56px; height: 56px; border-radius: 50%;
    border: 3px solid var(--gray-200); background: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; margin: 0 auto 12px; position: relative; z-index: 1;
  }
  .pipeline-step.done .step-circle { border-color: var(--green); background: var(--green-light); }
  .pipeline-step.active .step-circle { border-color: var(--blue-main); background: var(--blue-light); animation: pulse 1.5s infinite; }
  .step-label { font-size: 13px; font-weight: 600; color: var(--gray-800); margin-bottom: 4px; }
  .step-desc { font-size: 11px; color: var(--gray-400); line-height: 1.5; }
  .pipeline-step.done .step-label { color: var(--green); }
  .pipeline-step.active .step-label { color: var(--blue-main); }
  @keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(37,99,235,.3)} 50%{box-shadow:0 0 0 8px rgba(37,99,235,0)} }
  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner { width:14px; height:14px; border:2px solid var(--blue-mid); border-top-color:var(--blue-main); border-radius:50%; animation:spin .8s linear infinite; display:inline-block; }

  /* ── Upload zone ── */
  .upload-zone {
    border: 2px dashed var(--gray-200); border-radius: var(--radius);
    padding: 48px; text-align: center; background: var(--gray-50);
  }

  /* ── Synth box ── */
  .synth-box {
    background: #fff; border-left: 4px solid var(--green);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 20px 24px;
    border: 1px solid var(--gray-200); border-left: 4px solid var(--green);
    margin-bottom: 16px;
  }
  .synth-quote { font-size: 13px; color: var(--gray-700); line-height: 1.8; font-style: italic; }

  /* ── Progress ── */
  .progress-status {
    border: 1px solid var(--gray-200); border-radius: var(--radius);
    padding: 14px 20px; display: flex; align-items: center; justify-content: space-between;
    margin-top: 20px;
  }
  .status-text { font-size: 13px; color: var(--blue-main); font-weight: 500; }
  .status-pct { font-size: 14px; font-weight: 700; color: var(--blue-dark); }

  /* ── Table ── */
  .data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .data-table th { background: var(--gray-50); padding: 10px 14px; text-align: left; font-size: 11px; font-weight: 600; letter-spacing: .06em; color: var(--gray-400); text-transform: uppercase; border-bottom: 1px solid var(--gray-200); }
  .data-table td { padding: 11px 14px; border-bottom: 1px solid var(--gray-100); color: var(--gray-700); }
  .data-table tr:hover td { background: var(--blue-light); }

  /* ── Streamlit overrides ── */
  .stButton > button {
    background: var(--blue-main); color: #fff;
    border: none; border-radius: 50px;
    padding: 10px 24px; font-size: 14px; font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    cursor: pointer; transition: all .2s;
  }
  .stButton > button:hover { background: #1d4ed8; transform: translateY(-1px); }
  .stButton > button[kind="secondary"] {
    background: #fff; color: var(--gray-600);
    border: 1.5px solid var(--gray-200);
  }
  .stButton > button[kind="secondary"]:hover { border-color: var(--blue-main); color: var(--blue-main); background: var(--blue-light); }

  div[data-testid="stForm"] { border: none; padding: 0; }
  .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div { border-radius: var(--radius-sm) !important; border: 1.5px solid var(--gray-200) !important; font-family: 'DM Sans', sans-serif !important; }
  .stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus { border-color: var(--blue-main) !important; }

   /* hide default streamlit top bar but keep sidebar toggle accessible */
  #MainMenu, footer { visibility: hidden; }
  [data-testid="stHeader"] { background-color: transparent; }

  /* ── Trend box ── */
  .trend-box {
    background: #fff; border: 1px solid var(--gray-200);
    border-radius: var(--radius); padding: 20px 24px; margin-top: 20px;
  }
  .trend-title { font-size: 13px; font-weight: 600; color: var(--gray-800); margin-bottom: 8px; }
  .trend-text { font-size: 13px; color: var(--gray-600); line-height: 1.7; }

  /* Merci page */
  .merci-page { text-align: center; padding: 80px 40px; }
  .merci-title { font-size: 52px; font-weight: 800; color: var(--blue-dark); margin-bottom: 16px; }
  .merci-sub { font-size: 18px; color: var(--gray-600); margin-bottom: 48px; }
  .merci-divider { width: 200px; height: 1px; background: var(--gray-200); margin: 0 auto 40px; }
  .merci-row { display: flex; gap: 120px; justify-content: center; }
  .merci-col { text-align: center; }
  .merci-label { font-size: 11px; font-weight: 600; letter-spacing: .1em; color: var(--blue-main); text-transform: uppercase; margin-bottom: 8px; }
  .merci-val { font-size: 15px; color: var(--gray-700); }
</style>
""", unsafe_allow_html=True)

# PIPELINE DE NETTOYAGE & FEATURES 

FEATURES_DETECTION = [
    'DL_PRB_Usage_Rate', 'LTE_Setup_Success_Rate', 'Avaibility',
    'DL_Average_Throughput', 'UL_Average_Throughput',
    'Cell_Traffic_Volume_DL', 'Cell_Traffic_Volume_UL', 'Avg_User_NB',
    'PRB_per_User', 'PRB_Z_Score', 'Gradient_PRB', 'Rolling_PRB_3h',
    'Spectral_Eff', 'HOUR',
]

FEATURES_PREDICTION = [
    'lte_setup_success_rate', 'cell_traffic_volume_ul', 'dl_average_throughput',
    'ul_average_throughput', 'dl_prb_usage_rate', 'avg_user_nb', 'avaibility',
    'is_weekend', 'spectral_eff', 'is_peak_hour', 'rolling_trafic_3h',
    'rolling_prb_3h', 'hourly_trend', 'rolling_mean_volatility', 'gradient_prb',
    'hour_sin', 'hour_cos', 'lag_prb_1h', 'lag_prb_2h', 'lag_prb_3h'
]

LABELS = {0: 'Normal', 1: 'Modéré', 2: 'Congestionné'}
ICONS  = {0: '✅', 1: '⚠️', 2: '🔴'}
BADGE_CLASS = {0: 'badge-normal', 1: 'badge-modere', 2: 'badge-critique'}
BADGE_LABEL = {0: '✅ CLASSE 0 : NORMAL', 1: '⚠️ CLASSE 1 : MODÉRÉ', 2: '🔴 CLASSE 2 : CRITIQUE'}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def nettoyer_dataframe(df, df_reference=None):
   
    df = df.copy()

    # ── 1. Conversion DATE_ 
    if 'DATE_' in df.columns:
        df['DATE_'] = pd.to_datetime(df['DATE_'], errors='coerce')

    # ── 2. Valeurs impossibles 
    if 'LTE_Setup_Success_Rate' in df.columns:
        df.loc[df['LTE_Setup_Success_Rate'] > 100, 'LTE_Setup_Success_Rate'] = 100

    if 'Avaibility' in df.columns:
        df.loc[df['Avaibility'] > 100, 'Avaibility'] = 100

    # ── 3. Famille 2 — Cellules CRASHÉES (Avaibility = 0) 
    mask_crash = df['Avaibility'] == 0
    if mask_crash.any():
        df.loc[mask_crash, 'DL_PRB_Usage_Rate']     = df.loc[mask_crash, 'DL_PRB_Usage_Rate'].fillna(100)
        df.loc[mask_crash, 'LTE_Setup_Success_Rate'] = df.loc[mask_crash, 'LTE_Setup_Success_Rate'].fillna(0)
        df.loc[mask_crash, 'DL_Average_Throughput']  = df.loc[mask_crash, 'DL_Average_Throughput'].fillna(0)
        df.loc[mask_crash, 'UL_Average_Throughput']  = df.loc[mask_crash, 'UL_Average_Throughput'].fillna(0)

    # ── 4. Famille 3 — Cellules IDLE 
    mask_idle = (
        (df['Avaibility'] == 100) &
        (df['Cell_Traffic_Volume_DL'] == 0) &
        (df['LTE_Setup_Success_Rate'].isna())
    )
    if mask_idle.any():
        df.loc[mask_idle, 'LTE_Setup_Success_Rate'] = df.loc[mask_idle, 'LTE_Setup_Success_Rate'].fillna(0)
        df.loc[mask_idle, 'DL_Average_Throughput']  = df.loc[mask_idle, 'DL_Average_Throughput'].fillna(0)
        df.loc[mask_idle, 'UL_Average_Throughput']  = df.loc[mask_idle, 'UL_Average_Throughput'].fillna(0)

    # ── 5. NaN restants → médianes par cellule ou globales ───
    colonnes_median = [
        'LTE_Setup_Success_Rate', 'DL_Average_Throughput',
        'UL_Average_Throughput',  'DL_PRB_Usage_Rate'
    ] 
    defaults = {
        'LTE_Setup_Success_Rate': 99.5,
        'DL_Average_Throughput' : 25000.0,
        'UL_Average_Throughput' : 2000.0,
        'DL_PRB_Usage_Rate'     : 30.0,
    }

    for col in colonnes_median:
        if col not in df.columns:
            continue
        if df[col].isna().sum() == 0:
            continue
        if df_reference is not None and col in df_reference.columns and 'CELLNAME_ID' in df.columns:
            # Imputation par médiane historique par cellule (identique à test_detection.py)
            mediane_ref = df_reference.groupby('CELLNAME_ID')[col].median()
            for idx in df[df[col].isna()].index:
                cell = df.loc[idx, 'CELLNAME_ID']
                if cell in mediane_ref.index and not pd.isna(mediane_ref[cell]):
                    df.loc[idx, col] = mediane_ref[cell]
                else:
                    df.loc[idx, col] = df_reference[col].median()
        else:
            mediane_globale = df[col].median()
            if pd.isna(mediane_globale):
                mediane_globale = defaults.get(col, 0.0)
            df[col] = df[col].fillna(mediane_globale)

    return df


def calculer_features(df):

    df = df.copy()

    # ── Conversion et tri chronologique 
    if 'DATE_' in df.columns:
        df['DATE_DT'] = pd.to_datetime(df['DATE_'], errors='coerce')
        if 'CELLNAME_ID' in df.columns:
            df = df.sort_values(['CELLNAME_ID', 'DATE_DT']).reset_index(drop=True)
        df['HOUR'] = df['DATE_DT'].dt.hour.fillna(0).astype(int)
    else:
        df['DATE_DT'] = pd.NaT
        df['HOUR'] = 0

    # ── Groupe 1 : Features temporelles 
    if 'DATE_DT' in df.columns and df['DATE_DT'].notna().any():
        df['is_weekend']   = df['DATE_DT'].dt.dayofweek.isin([5, 6]).astype(int)
        df['Is_Peak_Hour'] = df['HOUR'].isin([20, 21, 22, 23]).astype(int)
    else:
        df['is_weekend']   = 0
        df['Is_Peak_Hour'] = 0

    # ── Groupe 2 : Features d'efficacité 
    df['PRB_per_User'] = df['DL_PRB_Usage_Rate'] / (df['Avg_User_NB'] + 1e-6)
    df['Spectral_Eff'] = df['Cell_Traffic_Volume_DL'] / (df['DL_PRB_Usage_Rate'] + 1e-6)

    # ── Groupe 3 : Features glissantes 
    if 'CELLNAME_ID' in df.columns and len(df) > 1:
        
        df['Rolling_PRB_3h'] = df.groupby('CELLNAME_ID')['DL_PRB_Usage_Rate'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean())
        
        df['PRB_Z_Score'] = df.groupby(['CELLNAME_ID', 'HOUR'])['DL_PRB_Usage_Rate'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6))
        df['Gradient_PRB'] = df.groupby('CELLNAME_ID')['DL_PRB_Usage_Rate'].diff().fillna(0)
    else:
        # Saisie manuelle ou ligne unique : approximations raisonnable
        df['Rolling_PRB_3h']          = df['DL_PRB_Usage_Rate']
        df['Rolling_Mean_Volatility'] = 0.0
        df['PRB_Z_Score']             = 0.0
        df['Gradient_PRB']            = 0.0

    # ── Remplir les NaN numériques restants 
    cols_num = df.select_dtypes(include=[np.number]).columns
    df[cols_num] = df[cols_num].fillna(0)

    # ── Supprimer colonne temporaire 
    if 'DATE_DT' in df.columns:
        df = df.drop(columns=['DATE_DT'])

    return df


def calculer_features_prediction(df):
    df = df.copy()
    if 'DATE_' in df.columns:
        df['date_dt'] = pd.to_datetime(df['DATE_'], errors='coerce')
    else:
        df['date_dt'] = pd.NaT
    df['hour'] = df['date_dt'].dt.hour.fillna(12).astype(int) if df['date_dt'].notna().any() else 12
    df['is_weekend'] = (df['date_dt'].dt.dayofweek.isin([5, 6]).astype(int)
                        if df['date_dt'].notna().any() else 0)
    df['is_peak_hour'] = df['hour'].isin([20, 21, 22, 23]).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['spectral_eff'] = df['Cell_Traffic_Volume_DL'] / (df['DL_PRB_Usage_Rate'] + 1e-6)
    # Lowercase rename
    rename_map = {
        'LTE_Setup_Success_Rate':   'lte_setup_success_rate',
        'Cell_Traffic_Volume_UL':   'cell_traffic_volume_ul',
        'Cell_Traffic_Volume_DL':   'cell_traffic_volume_dl',
        'DL_Average_Throughput':    'dl_average_throughput',
        'UL_Average_Throughput':    'ul_average_throughput',
        'DL_PRB_Usage_Rate':        'dl_prb_usage_rate',
        'Avg_User_NB':              'avg_user_nb',
        'Avaibility':               'avaibility',
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    prb = df['dl_prb_usage_rate'].values if 'dl_prb_usage_rate' in df.columns else df.get('DL_PRB_Usage_Rate', 30)
    if 'CELLNAME_ID' in df.columns and len(df) > 1:
        df['rolling_prb_3h'] = df.groupby('CELLNAME_ID')['dl_prb_usage_rate'].transform(
            lambda x: x.rolling(3, min_periods=1).mean())
        df['rolling_mean_volatility'] = df.groupby('CELLNAME_ID')['cell_traffic_volume_dl'].transform(
            lambda x: x.rolling(3, min_periods=1).std().fillna(0))
        df['gradient_prb'] = df.groupby('CELLNAME_ID')['dl_prb_usage_rate'].diff().fillna(0)
        df['lag_prb_1h'] = df.groupby('CELLNAME_ID')['dl_prb_usage_rate'].shift(1).fillna(df['dl_prb_usage_rate'])
        df['lag_prb_2h'] = df.groupby('CELLNAME_ID')['dl_prb_usage_rate'].shift(2).fillna(df['dl_prb_usage_rate'])
        df['lag_prb_3h'] = df.groupby('CELLNAME_ID')['dl_prb_usage_rate'].shift(3).fillna(df['dl_prb_usage_rate'])
    else:
        prb_val = float(df['dl_prb_usage_rate'].iloc[0]) if 'dl_prb_usage_rate' in df.columns else 30.0
        df['rolling_prb_3h'] = prb_val
        df['hourly_trend'] = 0.0
        df['rolling_mean_volatility'] = 0.0
        df['gradient_prb'] = 0.0
        df['lag_prb_1h'] = prb_val
        df['lag_prb_2h'] = prb_val
        df['lag_prb_3h'] = prb_val
    cols_num = df.select_dtypes(include=[np.number]).columns
    df[cols_num] = df[cols_num].fillna(0)
    return df


@st.cache_resource
def load_models():
    model_det = joblib.load(os.path.join(BASE_DIR, 'model_lightgbm_final.pkl'))
    model_1h  = joblib.load(os.path.join(BASE_DIR, 'lgbm_target_1h.pkl'))
    model_3h  = joblib.load(os.path.join(BASE_DIR, 'lgbm_target_3h.pkl'))
    model_6h  = joblib.load(os.path.join(BASE_DIR, 'lgbm_target_6h.pkl'))
    return model_det, model_1h, model_3h, model_6h
#____________________________________________________________
#rani lah9a lhna 
#______________________________________
@st.cache_data
def charger_reference():
 
    candidats = [
        'dataset_bts_nettoye_final.csv',
        'df_avec_score_kmeans.csv',
        'dataset_avec_features2.csv',
        'Datasetbtscongest.csv',
    ]
    for nom in candidats:
        chemin = os.path.join(BASE_DIR, nom)
        if os.path.exists(chemin):
            try:
                df_ref = pd.read_csv(chemin, nrows=50000)
                if 'Unnamed: 0' in df_ref.columns:
                    df_ref.drop(columns=['Unnamed: 0'], inplace=True)
                return df_ref
            except Exception:
                continue
    return None



# SESSION STATE

if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'input_data' not in st.session_state:
    st.session_state.input_data = None
if 'input_source' not in st.session_state:
    st.session_state.input_source = 'manual'
if 'detection_result' not in st.session_state:
    st.session_state.detection_result = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'pipeline_done' not in st.session_state:
    st.session_state.pipeline_done = False


def go_to(page):
    st.session_state.page = page
    st.rerun()



# SIDEBAR NAVIGATION

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <h2>🎓 USTHB | M2 Big Data</h2>
      <p>PROJET DE FIN D'ÉTUDES</p>
      <div class="djezzy-badge">📡 DJEZZY Radio</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    pages = [
        ('home',       '🏠', 'Accueil',        'Page de garde'),
        ('stats',      '📊', 'Dataset & EDA',   'Statistiques & graphiques'),
        ('input',      '⌨️', 'Acquisition',     'Saisie manuelle / CSV'),
        ('pipeline',   '⚙️', 'Pipeline',        'Animation du traitement'),
        ('detection',  '🔍', 'Détection',       'Résultats LightGBM'),
        ('prediction', '📈', 'Prédiction',      'Horizons +1h / +3h / +6h'),
        ('shap',       '🧬', 'Causes & Rapport',  'Explicabilité modèles'),
        ('merci',      '🙏', 'Fin',             'Remerciements')
    ]

    for pid, icon, label, desc in pages:
        active = "active" if st.session_state.page == pid else ""
        if st.button(f"{icon}  {label}", key=f"nav_{pid}",
                     help=desc, use_container_width=True):
            go_to(pid)

    st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:16px 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding:0 8px'>
      <p style='font-size:11px;color:#94a3b8;font-weight:600;letter-spacing:.06em;text-transform:uppercase;margin-bottom:8px'>Modèles</p>
      <p style='font-size:12px;color:#475569;margin:4px 0'>🔍 Détection — LightGBM<br>
      <span style='font-size:11px;color:#94a3b8'>Accuracy 99.9% | F1 99.94%</span></p>
      <p style='font-size:12px;color:#475569;margin:4px 0'>📈 Prédiction — 3 horizons<br>
      <span style='font-size:11px;color:#94a3b8'>1h: 96.2% | 3h: 93.6% | 6h: 92.1%</span></p>
    </div>
    """, unsafe_allow_html=True)


# HELPER: PAGE HEADER

def page_header(icon, title, subtitle, tag):
    st.markdown(f"""
    <div class="page-header">
      <div class="page-title-group">
        <div class="accent-bar"></div>
        <div class="page-icon-box">{icon}</div>
        <div class="page-title-text">
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
      </div>
      <span class="page-tag">{tag}</span>
    </div>
    """, unsafe_allow_html=True)



# PAGE 0 — ACCUEIL

if st.session_state.page == 'home':
    st.markdown("""
    <div class="landing">
      <p class="landing-tag">Projet de Fin d'Études (PFE) — 2025/2026</p>
      <h1 class="landing-title">
        Détection et Prédiction de la Congestion<br>des Sites <span>BTS Radio</span><br>via l'Intelligence Artificielle
      </h1>
      <p class="landing-sub">
        Pipeline end-to-end : nettoyage → feature engineering → détection multiclasse (LightGBM + Optuna)
        → prédiction multi-horizons (1h · 3h · 6h) + explicabilité SHAP.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    kpis = [
        ("7.78M+", "Enregistrements bruts"),
        ("72 851", "Cellules BTS analysées"),
        ("144h", "Période historique"),
        ("99.94%", "F1-macro détection"),
    ]
    for col, (val, lbl) in zip([col1,col2,col3,col4], kpis):
        col.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-value">{val}</div>
          <div class="kpi-label">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1.2, 1.2, 1])
    with col_a:
        st.markdown("""
        <div class="card">
          <div class="card-title">📋 Architecture du Pipeline</div>
          <div class="metric-row"><span class="metric-name">① Nettoyage & profiling</span><span class="metric-val">7.78M → 7.78M lignes</span></div>
          <div class="metric-row"><span class="metric-name">② Features Engineering</span><span class="metric-val">10 → 23 features</span></div>
          <div class="metric-row"><span class="metric-name">③ Détection LightGBM</span><span class="metric-val">3 classes (0/1/2)</span></div>
          <div class="metric-row"><span class="metric-name">④ Prédiction multi-horizons</span><span class="metric-val">+1h / +3h / +6h</span></div>
          <div class="metric-row"><span class="metric-name">⑤ Explicabilité SHAP</span><span class="metric-val">TreeExplainer</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="card">
          <div class="card-title">🏆 Performances Modèles</div>
          <div class="metric-row"><span class="metric-name">Détection — Accuracy</span><span class="metric-val">99.92%</span></div>
          <div class="metric-row"><span class="metric-name">Détection — F1-macro</span><span class="metric-val">99.94%</span></div>
          <div class="metric-row"><span class="metric-name">Prédiction 1h — F1-macro</span><span class="metric-val">96.16%</span></div>
          <div class="metric-row"><span class="metric-name">Prédiction 3h — F1-macro</span><span class="metric-val">93.58%</span></div>
          <div class="metric-row"><span class="metric-name">Prédiction 6h — F1-macro</span><span class="metric-val">92.12%</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown("""
        <div class="card">
          <div class="card-title">👥 Équipe & Encadrement</div>
          <div class="metric-row"><span class="metric-name">Réalisé par</span></div>
          <div style='padding:8px 0 4px'><span class="metric-val">Zouarqui Aya & Khettab Wissam </span></div>
          <div class="metric-row" style='margin-top:8px'><span class="metric-name">Établissement</span></div>
          <div style='padding:8px 0 4px'><span class="metric-val">USTHB — Alger</span></div>
          <div class="metric-row" style='margin-top:8px'><span class="metric-name">Partenaire industriel</span></div>
          <div style='padding:8px 0 4px'><span class="metric-val" style='color:var(--red)'>📡 Djezzy</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("📊 Explorer le Dataset →", key="home_stats", use_container_width=True):
            go_to('stats')
    with c2:
        if st.button("🚀 Lancer l'Application →", key="home_launch", use_container_width=True):
            go_to('input')



# PAGE 1 — DATASET & EDA

elif st.session_state.page == 'stats':
    page_header("📊", "Dataset & Exploration des Données",
                "Statistiques initiales, distributions et corrélations", "PAGE 1 – EDA")

    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("72 851", "Cellules Radio BTS"),
        ("144h", "Période d'historique"),
        ("7.78M+", "Enregistrements bruts"),
        ("23", "Features générées"),
    ]
    for col, (val, lbl) in zip([c1,c2,c3,c4], kpis):
        col.markdown(f"""<div class="kpi-card"><div class="kpi-value">{val}</div><div class="kpi-label">{lbl}</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Sous-KPIs nettoyage
    st.markdown("""
    <div class="card">
      <div class="card-title">🔧 Résultats du Nettoyage (cleaning.py)</div>
    """, unsafe_allow_html=True)
    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        st.metric("Lignes initiales", "7 783 424", delta=None)
    with cc2:
        st.metric("Après dédoublonnage", "7 783 150", delta="-274")
    with cc3:
        st.metric("Valeurs impossibles corrigées", "537", delta=None)
    with cc4:
        st.metric("NaN restants (UL_Thp)", "~18 811", delta="170 cellules")
    st.markdown("</div>", unsafe_allow_html=True)

    # Nettoyage détail
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.markdown("""
        <div class="card">
          <div class="card-title">📋 Distribution des Variables Brutes</div>
          <div class="metric-row"><span class="metric-name">LTE_Setup_Success_Rate — Moyenne</span><span class="metric-val">99.42%</span></div>
          <div class="metric-row"><span class="metric-name">DL_PRB_Usage_Rate — Moyenne / Médiane</span><span class="metric-val">41.6% / 35.2%</span></div>
          <div class="metric-row"><span class="metric-name">Avg_User_NB — Moyenne / Max</span><span class="metric-val">21 / 607</span></div>
          <div class="metric-row"><span class="metric-name">Availability — Médiane</span><span class="metric-val">95.32%</span></div>
          <div class="metric-row"><span class="metric-name">DL_Average_Throughput — Médiane</span><span class="metric-val orange">15 974 kbps</span></div>
          <div class="metric-row"><span class="metric-name">Cell_Traffic_Volume_DL — Médiane</span><span class="metric-val orange">3.35 GB</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col_stat2:
        st.markdown("""
       <div class="card">
            <div class="card-title">🏷️ Profils Détectés dans le Dataset (5 Familles)</div>
            <div class="metric-row"><span class="metric-name">Cellules normales</span><span class="metric-val">94.20%</span></div>
            <div class="metric-row"><span class="metric-name">Cellules crashées</span><span class="metric-val" style='color:var(--red)'>4.60%</span></div>
            <div class="metric-row"><span class="metric-name">Cellules en veille</span><span class="metric-val">0.43%</span></div>
            <div class="metric-row"><span class="metric-name">Anomalies de mesure</span><span class="metric-val">0.18%</span></div>
            <div class="metric-row"><span class="metric-name">Cas ambigus</span><span class="metric-val">0.15%</span></div>
        </div>
        """, unsafe_allow_html=True)

    # Durée dataset Plotly
    st.markdown("""<div class="card"><div class="card-title">⏱️ Durée du Dataset — Répartition Temporelle</div>""", unsafe_allow_html=True)
    heures = list(range(24))
    nb_cellules = 72851
    nb_jours = 6
    lignes_par_heure = [nb_cellules * nb_jours]   # ~436K / heure
    fig_dur = go.Figure()
    fig_dur.add_trace(go.Bar(x=heures, y=[v/1000 for v in lignes_par_heure],
                              marker_color='#2563eb', opacity=0.75,
                              name='Enregistrements (k)'))
    fig_dur.update_layout(
        title=f" 7.7M enregistrements sur 144h × {nb_cellules:,} cellules",
        xaxis_title="Heure", yaxis_title="Enregistrements (milliers)",
        height=260, margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='DM Sans', size=12),
        showlegend=False
    )
    fig_dur.update_xaxes(tickvals=list(range(0,24,2)))
    st.plotly_chart(fig_dur)
    st.markdown("</div>", unsafe_allow_html=True)

    # Distribution temporelle image
    st.markdown("""<div class="card"><div class="card-title">📈 Distribution Temporelle — EDA (graphiques réels)</div>""", unsafe_allow_html=True)
    img_path = os.path.join(BASE_DIR, 'assets', 'B3_distribution_temporelle.png')
    if os.path.exists(img_path):
        st.image(img_path)
    st.markdown("</div>", unsafe_allow_html=True)

    # Corrélation image
    st.markdown("""<div class="card"><div class="card-title">🔗 Matrice de Corrélation des KPI LTE</div>""", unsafe_allow_html=True)
    img_path2 = os.path.join(BASE_DIR, 'assets', 'B4_correlations.png')
    if os.path.exists(img_path2):
        st.image(img_path2)
    st.markdown("</div>", unsafe_allow_html=True)

    # Features engineering résumé
    st.markdown("""
    <div class="card">
      <div class="card-title">✨ Features Engineering — 10 colonnes → 23 features</div>
      <table class="data-table">
        <thead><tr><th>#</th><th>Colonne brute</th><th>Type</th><th>Description</th><th>Features générées</th></tr></thead>
        <tbody>
          <tr><td>1</td><td>DATE_</td><td>datetime</td><td>Horodatage horaire</td><td>hour, Is_Weekend ,s_Peak_Hour,hour_sin, hour_cos</td></tr>
          <tr><td>2</td><td>CELLNAME_ID</td><td>int</td><td>Identifiant cellule BTS</td><td>group by pour rolling/lag</td></tr>
          <tr><td>3</td><td>LTE_Setup_Success_Rate</td><td>float</td><td>Taux établissement LTE</td><td>feature directe</td></tr>
          <tr><td>4</td><td>DL_PRB_Usage_Rate</td><td>float</td><td>🎯 Taux ressources radio DL</td><td>rolling_prb_3h, gradient_prb, Rolling_Volatility_3H,prb_z_score, lag_prb_1/2/3h</td></tr>
          <tr><td>5</td><td>Avg_User_NB</td><td>float</td><td>Utilisateurs actifs moyens</td><td>prb_per_user</td></tr>
          <tr><td>6</td><td>Cell_Traffic_Volume_DL</td><td>float</td><td>Volume trafic descendant (GB)</td><td> spectral_eff , Rolling_Trafic_3H</td></tr>
          <tr><td>7</td><td>DL_Average_Throughput</td><td>float</td><td>Débit moyen DL (kbps)</td><td>feature directe</td></tr>
          <tr><td>8</td><td>Avaibility</td><td>float</td><td>Disponibilité cellule (%)</td><td>feature directe (top SHAP)</td></tr>
        </tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    # Distribution des classes
    labels_cls = ['Classe 0 – Normal', 'Classe 1 – Modéré', 'Classe 2 – Congestionné']
    vals_cls = [4332674, 2672633, 341362]
    colors_cls = ['#16a34a', '#f97316', '#dc2626']
    fig_cls = go.Figure(data=[go.Pie(
        labels=labels_cls, values=vals_cls,
        marker_colors=colors_cls,
        hole=0.4,
        textinfo='label+percent'
    )])
    fig_cls.update_layout(
        title="Distribution des Classes (7.35M lignes)",
        height=360, margin=dict(l=0, r=0, t=50, b=0),
        font=dict(family='DM Sans', size=12),
        paper_bgcolor='white'
    )
    st.markdown("""<div class="card"><div class="card-title">🥧 Distribution des Classes de Congestion</div>""", unsafe_allow_html=True)
    st.plotly_chart(fig_cls, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# PAGE 2 — ACQUISITION

elif st.session_state.page == 'input':
    page_header("⌨️", "Module de Saisie et Acquisition de Données",
                "Saisie manuelle ou upload CSV — données brutes acceptées", "PAGE 2 – ACQUISITION")

    tab_manual, tab_csv = st.tabs(["✏️  Onglet A — Saisie Manuelle", "📄  Onglet B — Upload Fichier CSV"])

    with tab_manual:
        st.markdown("""<div class="card">
        <p style='font-size:13px;color:var(--gray-400);margin-bottom:20px'>
          Simulation de trafic en temps réel — 10 variables critiques (valeurs par défaut préchargées) :
        </p>""", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            cell_id = st.text_input("CELLNAME_ID", value="Cell_1234", key="f_cell")
        with col2:
            date_val = st.text_input("DATE_ (ex: 2026-05-22 17:00)", value="2026-05-22 17:00", key="f_date")
        with col3:
            lte_ssr = st.number_input("LTE_Setup_SR (%)", value=98.5, min_value=0.0, max_value=101.72, step=0.1, key="f_lte")
        with col4:
            prb = st.number_input("DL_PRB_Usage_Rate (%)", value=72.3, min_value=0.0, max_value=100.0, step=0.1, key="f_prb")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            users = st.number_input("Avg_User_NB", value=45.0, min_value=0.0, step=1.0, key="f_users")
        with col6:
            thp_dl = st.number_input("DL_Average_Throughput (kbps)", value=12400.0, min_value=0.0, step=100.0, key="f_tp")
        with col7:
            vol_dl = st.number_input("Cell_Traffic_Volume_DL (GB)", value=4.2, min_value=0.0, step=0.1, key="f_vol")
        with col8:
            avail = st.number_input("Availability (%)", value=100.0, min_value=0.0, max_value=100.0, step=0.1, key="f_avail")

        col9, col10, _, __ = st.columns(4)
        with col9:
            vol_ul = st.number_input("Cell_Traffic_Volume_UL (GB)", value=0.8, min_value=0.0, step=0.1, key="f_volul")
        with col10:
            thp_ul = st.number_input("UL_Average_Throughput (kbps)", value=3800.0, min_value=0.0, step=100.0, key="f_ultp")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        col_hint, col_btn = st.columns([3, 1])
        with col_hint:
            st.markdown("<p style='font-size:12px;color:var(--gray-400);padding-top:10px'>ℹ️ Données exploitables immédiatement pour l'inférence LightGBM</p>", unsafe_allow_html=True)
        with col_btn:
            if st.button("🔍  Analyser la Cellule →", key="btn_manual", use_container_width=True):
                st.session_state.input_data = pd.DataFrame([{
                    'CELLNAME_ID': cell_id, 'DATE_': date_val,
                    'LTE_Setup_Success_Rate': lte_ssr,
                    'Cell_Traffic_Volume_DL': vol_dl,
                    'Cell_Traffic_Volume_UL': vol_ul,
                    'DL_Average_Throughput': thp_dl,
                    'UL_Average_Throughput': thp_ul,
                    'DL_PRB_Usage_Rate': prb,
                    'Avg_User_NB': users,
                    'Avaibility': avail
                }])
                st.session_state.input_source = 'manual'
                st.session_state.pipeline_done = False
                go_to('pipeline')

    with tab_csv:
        st.markdown("""<div class="card">""", unsafe_allow_html=True)
        st.markdown("""
        <div class="upload-zone">
          <p style='font-size:36px;margin-bottom:8px'>📂</p>
          <p style='font-size:14px;font-weight:600;color:var(--gray-600)'>Glisser-déposer votre fichier CSV ici, ou utiliser le bouton ci-dessous</p>
          <p style='font-size:12px;color:var(--gray-400);margin-top:6px'>Format : CSV avec les 10 colonnes du dataset Djezzy</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Choisir un fichier CSV", type=['csv'], key='csv_upload', label_visibility='collapsed')

        if uploaded is not None:
            try:
                df_upload = pd.read_csv(uploaded)
                if 'Unnamed: 0' in df_upload.columns:
                    df_upload.drop(columns=['Unnamed: 0'], inplace=True)
                st.success(f"✅ Fichier chargé : **{len(df_upload):,}** lignes, **{df_upload.shape[1]}** colonnes")
                st.markdown("<p style='font-size:13px;font-weight:600;color:var(--gray-800);margin:12px 0 8px'>Aperçu du fichier chargé :</p>", unsafe_allow_html=True)
                st.dataframe(df_upload.head(6), use_container_width=True)

                c_hint, c_btn_csv = st.columns([3, 1])
                with c_btn_csv:
                    if st.button("🔍  Analyser le Fichier Complet →", key="btn_csv", use_container_width=True):
                        st.session_state.input_data = df_upload
                        st.session_state.input_source = 'csv'
                        st.session_state.pipeline_done = False
                        go_to('pipeline')
            except Exception as e:
                st.error(f"Erreur lecture CSV : {e}")
        st.markdown("</div>", unsafe_allow_html=True)


# PAGE 3 — PIPELINE ANIMÉ

elif st.session_state.page == 'pipeline':
    page_header("⚙️", "Traitement Analytique & Pipeline de Prédiction",
                "Exécution séquentielle automatique des algorithmes", "PAGE 3 – PIPELINE")

    if st.session_state.input_data is None:
        st.warning("⚠️ Aucune donnée chargée. Retournez à la page Acquisition.")
        if st.button("← Retour Acquisition"):
            go_to('input')
        st.stop()

    step_labels  = ["1. Nettoyage", "2. Ingénierie", "3. Détection", "4. Prédiction"]
    step_descs   = ["Gestion valeurs\naberrantes", "Génération 23\nfeatures avancées", "Score LightGBM\nmulticlasse", "Horizons\n1h · 3h · 6h"]
    step_icons   = ["🔧", "✨", "🎯", "📈"]
    step_msgs    = [
        "Nettoyage des données en cours...",
        "Calcul des features engineering...",
        "Inférence LightGBM — Détection...",
        "Prédiction multi-horizons...",
        "✅ Pipeline terminé avec succès !",
    ]
    step_pcts    = [0, 25, 50, 75, 100]

    if not st.session_state.pipeline_done:
        steps_state = ['pending'] * 4
        ph_steps  = st.empty()
        ph_prog   = st.empty()
        ph_status = st.empty()

        def render_steps(states):
            html = '<div class="card"><div class="card-title">🔄 Étapes du Pipeline</div><div class="pipeline-steps">'
            for i, (lbl, desc, icon, state) in enumerate(zip(step_labels, step_descs, step_icons, states)):
                step_cls = state  # done / active / pending
                html += f'<div class="pipeline-step {step_cls}" id="step-{i}">'
                html += f'<div class="step-circle">{icon}</div>'
                html += f'<div class="step-label">{lbl}</div>'
                html += f'<div class="step-desc">{desc.replace(chr(10),"<br>")}</div>'
                html += '</div>'
            html += '</div></div>'
            return html

        model_det, model_1h, model_3h, model_6h = load_models()
        df_reference = charger_reference()
        df = st.session_state.input_data.copy()

        for step in range(4):
            steps_state[step] = 'active'
            ph_steps.markdown(render_steps(steps_state), unsafe_allow_html=True)
            ph_prog.progress(step_pcts[step] / 100)
            ph_status.markdown(f"""
            <div class="progress-status">
              <span class="status-text"><span class="spinner"></span> &nbsp; {step_msgs[step]}</span>
              <span class="status-pct">{step_pcts[step]}% complété</span>
            </div>""", unsafe_allow_html=True)
            time.sleep(0.9)

            if step == 0:
                # Nettoyage avec référence historique si disponible (identique à test_detection.py)
                df_clean = nettoyer_dataframe(df, df_reference)
            elif step == 1:
                df_feat = calculer_features(df_clean)
                df_pred_feat = calculer_features_prediction(df_clean)
            elif step == 2:
                manquantes = [f for f in FEATURES_DETECTION if f not in df_feat.columns]
                if manquantes:
                    for f in manquantes:
                        df_feat[f] = 0
                X_det = df_feat[FEATURES_DETECTION].fillna(0)
                y_cls  = model_det.predict(X_det)
                y_prob = model_det.predict_proba(X_det)
                st.session_state.detection_result = {
                    'classes': y_cls, 'probas': y_prob,
                    'df': df_feat, 'df_clean': df_clean,
                    'source': st.session_state.input_source
                }
            elif step == 3:
                for f in FEATURES_PREDICTION:
                    if f not in df_pred_feat.columns:
                        df_pred_feat[f] = 0
                X_pred = df_pred_feat[FEATURES_PREDICTION].fillna(0)
                p1_cls  = model_1h.predict(X_pred);  p1_prob = model_1h.predict_proba(X_pred)
                p3_cls  = model_3h.predict(X_pred);  p3_prob = model_3h.predict_proba(X_pred)
                p6_cls  = model_6h.predict(X_pred);  p6_prob = model_6h.predict_proba(X_pred)
                st.session_state.prediction_result = {
                    '1h': {'classes': p1_cls, 'probas': p1_prob},
                    '3h': {'classes': p3_cls, 'probas': p3_prob},
                    '6h': {'classes': p6_cls, 'probas': p6_prob},
                }

            steps_state[step] = 'done'

        ph_steps.markdown(render_steps(steps_state), unsafe_allow_html=True)
        ph_prog.progress(1.0)
        ph_status.markdown(f"""
        <div class="progress-status" style="border-color:var(--green)">
          <span class="status-text" style="color:var(--green)">{step_msgs[4]}</span>
          <span class="status-pct" style="color:var(--green)">100% complété</span>
        </div>""", unsafe_allow_html=True)
        st.session_state.pipeline_done = True

    else:
        # Pipeline déjà exécuté — afficher résumé
        st.markdown("""
        <div class="card" style="border-color:var(--green)">
          <div class="card-title" style="color:var(--green)">✅ Pipeline exécuté avec succès</div>
          <p style='font-size:13px;color:var(--gray-600)'>Les modèles ont bien été appliqués. Consultez les résultats dans les pages Détection et Prédiction.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔍  Voir Résultats Détection →", key="pipe_det", use_container_width=True):
            go_to('detection')
    with c2:
        if st.button("📈  Voir Résultats Prédiction →", key="pipe_pred", use_container_width=True):
            go_to('prediction')
    with c3:
        if st.button("← Nouvelle Analyse", key="pipe_back", use_container_width=True):
            st.session_state.pipeline_done = False
            go_to('input')


# PAGE 4 — DÉTECTION

elif st.session_state.page == 'detection':
    page_header("🔍", "Résultats de Détection — LightGBM",
                "Classification multiclasse instantanée : Normal / Modéré / Congestionné", "PAGE 4 – DÉTECTION")

    if st.session_state.detection_result is None:
        st.warning("⚠️ Aucun résultat disponible. Lancez d'abord l'analyse depuis la page Acquisition.")
        if st.button("← Aller à l'Acquisition"):
            go_to('input')
        st.stop()

    res  = st.session_state.detection_result
    df_r = res['df']
    y_cls  = res['classes']
    y_prob = res['probas']
    src    = res['source']

    if src == 'manual':
        # Résultat ligne unique
        cls   = int(y_cls[0])
        prob  = y_prob[0]
        cell  = df_r['CELLNAME_ID'].iloc[0] if 'CELLNAME_ID' in df_r.columns else "Cellule"
        prb_v = float(df_r['DL_PRB_Usage_Rate'].iloc[0]) if 'DL_PRB_Usage_Rate' in df_r.columns else 0

        badge_html = {
            0: '<span class="badge-normal">✅ CLASSE 0 : NORMAL</span>',
            1: '<span class="badge-modere">⚠️ CLASSE 1 : MODÉRÉ</span>',
            2: '<span class="badge-critique">🔴 CLASSE 2 : CRITIQUE</span>',
        }

        st.markdown(f"""
        <div class="detection-header">
          <div>
            <div class="cell-id-label">Cellule analysée</div>
            <div class="cell-id-value">{cell} &nbsp;<span style='font-size:13px;color:var(--gray-400)'>(Statut instantané)</span></div>
          </div>
          <div style='text-align:right'>
            {badge_html[cls]}
            <div style='font-size:12px;color:var(--gray-400);margin-top:8px'>Score confiance : <strong style='color:var(--blue-dark)'>{prob[cls]:.3f} / 1.000</strong></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""<div class="card"><div class="card-title">📊 Probabilités par Classe</div>""", unsafe_allow_html=True)
            for c_idx, (label, color) in enumerate([('Normal', '#16a34a'), ('Modéré', '#f97316'), ('Congestionné', '#dc2626')]):
                pct = prob[c_idx] * 100
                st.markdown(f"""
                <div class="shap-row" style="margin-bottom:18px">
                  <span class="shap-name">{ICONS[c_idx]} {label}</span>
                  <div class="shap-bar-bg">
                    <div style="height:100%;width:{pct:.1f}%;background:{color};border-radius:4px;opacity:0.8"></div>
                  </div>
                  <span style='font-size:14px;font-weight:700;color:{color};min-width:55px;font-family:"DM Mono",monospace'>{pct:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_b:
            st.markdown("""<div class="card"><div class="card-title">📋 Métriques Clés Observées</div>""", unsafe_allow_html=True)
            prb_val = float(df_r['DL_PRB_Usage_Rate'].iloc[0]) if 'DL_PRB_Usage_Rate' in df_r.columns else 0
            users_val = float(df_r['Avg_User_NB'].iloc[0]) if 'Avg_User_NB' in df_r.columns else 0
            avail_val = float(df_r['Avaibility'].iloc[0]) if 'Avaibility' in df_r.columns else 100
            rolling_val = float(df_r['Rolling_PRB_3h'].iloc[0]) if 'Rolling_PRB_3h' in df_r.columns else prb_val
            grad_val = float(df_r['Gradient_PRB'].iloc[0]) if 'Gradient_PRB' in df_r.columns else 0
            st.markdown(f"""
            <div class="metric-row"><span class="metric-name">DL_PRB_Usage_Rate</span><span class="metric-val orange">{prb_val:.2f}%</span></div>
            <div class="metric-row"><span class="metric-name">Avg_User_NB</span><span class="metric-val">{users_val:.0f}</span></div>
            <div class="metric-row"><span class="metric-name">Availability</span><span class="metric-val">{avail_val:.1f}%</span></div>
            <div class="metric-row"><span class="metric-name">Rolling_PRB_3h</span><span class="metric-val">{rolling_val:.2f}%</span></div>
            <div class="metric-row"><span class="metric-name">Gradient_PRB</span><span class="metric-val">{grad_val:+.2f}</span></div>
            </div>
            """, unsafe_allow_html=True)

        # Interprétation
        interp = {
            0: ("Cellule en fonctionnement NORMAL.", "Aucune action requise. Continuer la surveillance standard."),
            1: ("Congestion MODÉRÉE détectée.", "Surveillance renforcée recommandée. Vérifier l'évolution du taux PRB dans les prochaines heures."),
            2: ("CONGESTION CRITIQUE !", "Intervention immédiate requise. PRB saturé — cellule surchargée. Envisager redistribution du trafic."),
        }
        colors_interp = {0: 'var(--green)', 1: 'var(--orange)', 2: 'var(--red)'}
        st.markdown(f"""
        <div class="synth-box">
          <div class="synth-quote">
            <strong style='color:{colors_interp[cls]}'>{interp[cls][0]}</strong><br>
            {interp[cls][1]}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Jauge PRB
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prb_v,
            delta={'reference': 65, 'valueformat': '.1f'},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#94a3b8'},
                'bar': {'color': '#2563eb'},
                'steps': [
                    {'range': [0, 65], 'color': '#f0fdf4'},
                    {'range': [65, 85], 'color': '#fff7ed'},
                    {'range': [85, 100], 'color': '#fef2f2'}
                ],
                'threshold': {'line': {'color': '#dc2626', 'width': 3}, 'thickness': 0.8, 'value': 85}
            },
            number={'suffix': '%', 'font': {'size': 32, 'family': 'DM Sans'}},
            title={'text': 'DL PRB Usage Rate', 'font': {'size': 14, 'family': 'DM Sans'}},
        ))
        fig_gauge.update_layout(height=240, margin=dict(l=20,r=20,t=40,b=0), paper_bgcolor='white')
        st.markdown("""<div class="card"><div class="card-title">🌡️ Jauge PRB Usage Rate</div>""", unsafe_allow_html=True)
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Mode CSV — résultats multiples
        from collections import Counter
        counts = Counter(y_cls.tolist())
        n = len(y_cls)

        c1, c2, c3 = st.columns(3)
        for col, (cls_i, label, color) in zip([c1,c2,c3], [(0,'Normal','#16a34a'),(1,'Modéré','#f97316'),(2,'Congestionné','#dc2626')]):
            nb  = counts.get(cls_i, 0)
            pct = nb / n * 100
            col.markdown(f"""<div class="kpi-card">
              <div class="kpi-value" style="color:{color}">{nb:,}</div>
              <div class="kpi-label">{ICONS[cls_i]} {label} — {pct:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        fig_bar = go.Figure(go.Bar(
            x=['Normal (0)', 'Modéré (1)', 'Congestionné (2)'],
            y=[counts.get(0,0), counts.get(1,0), counts.get(2,0)],
            marker_color=['#16a34a', '#f97316', '#dc2626'],
            text=[f"{counts.get(i,0):,}\n({counts.get(i,0)/n*100:.1f}%)" for i in range(3)],
            textposition='auto'
        ))
        fig_bar.update_layout(
            title=f"Distribution des classes prédites — {n:,} observations",
            yaxis_title="Nombre d'observations", height=320,
            margin=dict(l=0,r=0,t=50,b=0), paper_bgcolor='white',
            plot_bgcolor='white', font=dict(family='DM Sans')
        )
        st.markdown("""<div class="card"><div class="card-title">📊 Distribution des Résultats de Détection</div>""", unsafe_allow_html=True)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Confidence histo
        max_probs = y_prob.max(axis=1)
        fig_conf = px.histogram(x=max_probs, nbins=40, color_discrete_sequence=['#2563eb'],
                                 labels={'x': 'Probabilité de confiance', 'y': 'Nb observations'},
                                 title="Distribution de la confiance (probabilité max)")
        fig_conf.update_layout(height=280, margin=dict(l=0,r=0,t=50,b=0),
                                paper_bgcolor='white', plot_bgcolor='white',
                                font=dict(family='DM Sans'))
        st.markdown("""<div class="card"><div class="card-title">🎯 Confiance des Prédictions</div>""", unsafe_allow_html=True)
        st.plotly_chart(fig_conf, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Tableau résultats
        df_res = res['df'].copy()
        df_res['Classe_Prédite'] = y_cls
        df_res['Label_Prédit'] = [LABELS[c] for c in y_cls]
        df_res['Proba_Normal'] = y_prob[:,0].round(4)
        df_res['Proba_Modere'] = y_prob[:,1].round(4)
        df_res['Proba_Critique'] = y_prob[:,2].round(4)
        df_res['Confiance'] = y_prob.max(axis=1).round(4)
        st.markdown("""<div class="card"><div class="card-title">📋 Aperçu des Résultats</div>""", unsafe_allow_html=True)
        cols_show = [c for c in ['CELLNAME_ID', 'DATE_', 'DL_PRB_Usage_Rate', 'Avg_User_NB', 'Avaibility', 'Classe_Prédite', 'Label_Prédit', 'Confiance'] if c in df_res.columns]
        st.dataframe(df_res[cols_show].head(20), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📈  Voir Prédiction →", key="det_pred_btn", use_container_width=True):
            go_to('prediction')
    with c2:
        if st.button("🧬  Voir SHAP →", key="det_shap_btn", use_container_width=True):
            go_to('shap')


# PAGE 5 — PRÉDICTION

elif st.session_state.page == 'prediction':
    page_header("📈", "Prédiction Multi-Horizons — LightGBM",
                "Prédiction de la congestion à +1h, +3h et +6h", "PAGE 5 – PRÉDICTION")

    if st.session_state.prediction_result is None:
        st.warning("⚠️ Aucun résultat disponible. Lancez d'abord l'analyse depuis la page Acquisition.")
        if st.button("← Aller à l'Acquisition"):
            go_to('input')
        st.stop()

    pres = st.session_state.prediction_result
    src  = st.session_state.input_source

    horizons = [('1h', '+ 1 Heure', 'blue'), ('3h', '+ 3 Heures', 'orange'), ('6h', '+ 6 Heures', 'green')]

    if src == 'manual':
        col_h1, col_h2, col_h3 = st.columns(3)
        for col, (hkey, hlabel, hcolor) in zip([col_h1, col_h2, col_h3], horizons):
            cls_h = int(pres[hkey]['classes'][0])
            prob_h = pres[hkey]['probas'][0]
            conf = prob_h[cls_h] * 100
            badge_col = {'normal': '#16a34a', 'modere': '#f97316', 'critique': '#dc2626'}
            cls_color = ['#16a34a', '#f97316', '#dc2626'][cls_h]
            bar_w = conf

            col.markdown(f"""
            <div class="horizon-card {hcolor}">
              <div class="horizon-label">{hlabel}</div>
              <div class="horizon-value">{conf:.1f}%</div>
              <div class="horizon-bar-bg">
                <div class="horizon-bar-fill {hcolor}" style="width:{bar_w:.1f}%"></div>
              </div>
              <span class="badge-{'normal' if cls_h==0 else ('modere' if cls_h==1 else 'critique')}">
                {ICONS[cls_h]} {LABELS[cls_h].upper()}
              </span>
            </div>
            """, unsafe_allow_html=True)

        # Graphe évolution
        det_prb = 0
        if st.session_state.detection_result is not None:
            df_det = st.session_state.detection_result['df']
            if 'DL_PRB_Usage_Rate' in df_det.columns:
                det_prb = float(df_det['DL_PRB_Usage_Rate'].iloc[0])

        prob_1h = pres['1h']['probas'][0][int(pres['1h']['classes'][0])] * 100
        prob_3h = pres['3h']['probas'][0][int(pres['3h']['classes'][0])] * 100
        prob_6h = pres['6h']['probas'][0][int(pres['6h']['classes'][0])] * 100

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=['Actuel', '+1h', '+3h', '+6h'],
            y=[det_prb, prob_1h, prob_3h, prob_6h],
            mode='lines+markers',
            name='Score congestion (%)',
            line=dict(color='#2563eb', width=2.5),
            marker=dict(size=8, color='#2563eb'),
            fill='tozeroy', fillcolor='rgba(37,99,235,0.07)'
        ))
        fig_pred.add_hline(y=85, line_dash='dash', line_color='#dc2626', line_width=1.5,
                            annotation_text='Seuil critique (85%)', annotation_position='right')
        fig_pred.add_hline(y=65, line_dash='dash', line_color='#f97316', line_width=1.5,
                            annotation_text='Seuil alerte (65%)', annotation_position='right')
        fig_pred.update_layout(
            title='Évolution du score de congestion sur les 3 horizons',
            yaxis=dict(title='Score (%)', range=[0,105]),
            height=320, margin=dict(l=0,r=80,t=50,b=0),
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(family='DM Sans', size=12),
            legend=dict(orientation='h', y=-0.2)
        )
        st.markdown("""<div class="card"><div class="card-title">📉 Courbe Prédictive — Évolution du Risque de Congestion</div>""", unsafe_allow_html=True)
        st.plotly_chart(fig_pred, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Tableau probabilités
        st.markdown("""<div class="card"><div class="card-title">📋 Détail des Probabilités par Horizon</div>""", unsafe_allow_html=True)
        data_table = []
        for hkey, hlabel, _ in horizons:
            cls_h = int(pres[hkey]['classes'][0])
            prob_h = pres[hkey]['probas'][0]
            data_table.append({
                'Horizon': hlabel,
                'Classe Prédite': f"{ICONS[cls_h]} {LABELS[cls_h]}",
                'P(Normal)': f"{prob_h[0]*100:.2f}%",
                'P(Modéré)': f"{prob_h[1]*100:.2f}%",
                'P(Congestionné)': f"{prob_h[2]*100:.2f}%",
                'Confiance': f"{prob_h[cls_h]*100:.2f}%"
            })
        st.dataframe(pd.DataFrame(data_table), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Analyse de tendance
        cls_list = [int(pres[h]['classes'][0]) for h in ['1h','3h','6h']]
        if max(cls_list) == 0:
            trend_text = "La cellule devrait rester en fonctionnement <strong>normal</strong> sur les 6 prochaines heures. Aucune action requise."
        elif max(cls_list) == 2:
            peak_h = ['1h','3h','6h'][cls_list.index(2)]
            trend_text = f"⚠️ Une <strong>congestion critique</strong> est prédite à l'horizon <strong>+{peak_h}</strong>. Une intervention préventive est recommandée."
        else:
            trend_text = "Un risque de congestion <strong>modérée</strong> est détecté. Surveillance renforcée recommandée pour les prochaines heures."

        st.markdown(f"""
        <div class="trend-box">
          <div class="trend-title">📌 Analyse de Tendance</div>
          <div class="trend-text">{trend_text}</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Mode CSV
        for hkey, hlabel, hcolor in horizons:
            cls_arr = pres[hkey]['classes']
            prob_arr = pres[hkey]['probas']
            counts_h = {i: int(np.sum(cls_arr == i)) for i in [0,1,2]}
            n = len(cls_arr)
            st.markdown(f"""<div class="card"><div class="card-title">📊 Horizon {hlabel} — Résumé</div>""", unsafe_allow_html=True)
            c1h, c2h, c3h = st.columns(3)
            for col_h, (ci, label_i, color_i) in zip([c1h,c2h,c3h], [(0,'Normal','#16a34a'),(1,'Modéré','#f97316'),(2,'Congestionné','#dc2626')]):
                col_h.markdown(f"""<div class="kpi-card" style="border-top:4px solid {color_i}">
                  <div class="kpi-value" style="color:{color_i}">{counts_h[ci]:,}</div>
                  <div class="kpi-label">{label_i} — {counts_h[ci]/n*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Comparatif horizons
        fig_comp = go.Figure()
        categories = ['Normal (0)', 'Modéré (1)', 'Congestionné (2)']
        colors_h = ['#2563eb', '#f97316', '#16a34a']
        for hi, (hkey, hlabel, _) in enumerate(horizons):
            cls_arr = pres[hkey]['classes']
            n_h = len(cls_arr)
            counts_h = [int(np.sum(cls_arr == i)) for i in [0,1,2]]
            fig_comp.add_trace(go.Bar(
                name=f'Horizon {hlabel}', x=categories,
                y=[c/n_h*100 for c in counts_h],
                marker_color=colors_h[hi], opacity=0.85
            ))
        fig_comp.update_layout(
            barmode='group', title='Comparaison des classes prédites par horizon (%)',
            yaxis_title='%', height=340, margin=dict(l=0,r=0,t=50,b=0),
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(family='DM Sans')
        )
        st.markdown("""<div class="card"><div class="card-title">📊 Comparatif Multi-Horizons</div>""", unsafe_allow_html=True)
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button("🧬  Voir SHAP & Rapport →", key="pred_shap", use_container_width=False):
        go_to('shap')

# PAGE 6 — SHAP & RAPPORT

elif st.session_state.page == 'shap':
    page_header("🧬", "SHAP & Explicabilité des Modèles",
                "Analyse de l'importance des features ", "PAGE 6 – SHAP")

    # SHAP saisie manuelle live
    if (st.session_state.detection_result is not None and
            st.session_state.input_source == 'manual'):

        res = st.session_state.detection_result
        df_r = res['df']
        cls = int(res['classes'][0])
        prob = res['probas'][0]
        cell = df_r['CELLNAME_ID'].iloc[0] if 'CELLNAME_ID' in df_r.columns else "Cellule"

        # Valeurs SHAP approximées à partir des feature importances du modèle
        try:
            model_det, *_ = load_models()
            importances = model_det.feature_importances_
            feat_names = FEATURES_DETECTION
            # Pondérer par la valeur courante (approximation visuelle)
            row = df_r[FEATURES_DETECTION].fillna(0).iloc[0].values
            shap_approx = importances * np.abs(row - np.zeros(len(row))) / (np.abs(row).max() + 1e-6)
            shap_approx = shap_approx / shap_approx.max() * 5  # normalisation

            top_idx = np.argsort(shap_approx)[::-1][:8]
            top_names = [feat_names[i] for i in top_idx]
            top_vals  = [shap_approx[i] for i in top_idx]

            st.markdown(f"""
            <div class="synth-box">
              <div class="synth-quote">
                "La cellule <strong>{cell}</strong> est catégorisée en risque <strong>{LABELS[cls].lower()}</strong>
                principalement à cause de <strong>{top_names[0]}</strong> (score SHAP : {top_vals[0]:.2f})
                couplé à <strong>{top_names[1]}</strong> (score SHAP : {top_vals[1]:.2f}).
                Confiance globale du modèle : <strong>{prob[cls]*100:.1f}%</strong>."
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""<div class="card"><div class="card-title">🧬 Importance des Features — Décision Courante (approximation SHAP)</div>""", unsafe_allow_html=True)
            for name, val in zip(top_names, top_vals):
                w = val / max(top_vals) * 100
                is_pos = val > 0
                st.markdown(f"""
                <div class="shap-row">
                  <span class="shap-name">{name}</span>
                  <div class="shap-bar-bg">
                    <div class="{'shap-bar-pos' if is_pos else 'shap-bar-neg'}" style="width:{w:.1f}%"></div>
                  </div>
                  <span class="{'shap-val-pos' if is_pos else 'shap-val-neg'}">{val:+.2f}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.info(f"SHAP approximation non disponible : {e}")

    # ─ Feature importances globales ─
    try:
        model_det, model_1h, model_3h, model_6h = load_models()
        fi_det = model_det.feature_importances_
        top_n = 10
        idx_det = np.argsort(fi_det)[::-1][:top_n]

    # Inversion du tri [::-1] pour avoir la plus grande barre à gauche en vertical
        features_labels = [FEATURES_DETECTION[i] for i in idx_det]
        features_values = [fi_det[i] for i in idx_det]

        fig_fi = go.Figure(go.Bar(
            x=features_labels,                             # Les noms des variables sur l'axe X
            y=features_values,                             # Le gain sur l'axe Y
            marker_color=['#ef4444' if i < 3 else '#2563eb' for i in range(top_n)], # Top 3 en rouge, le reste en bleu
            text=[f"{val:.0f}" for val in features_values],
            textposition='outside'
        ))
    
        fig_fi.update_layout(
            title='Feature Importances  — Modèle Détection LightGBM',
            yaxis_title='Gain',                            # Le titre du gain passe sur l'axe Y
            xaxis_title='Variables',
            height=450,                                    # Augmentation de la hauteur pour laisser de la place aux étiquettes
            margin=dict(l=40, r=40, t=50, b=120),          # Plus de marge en bas (b=120) pour ne pas couper les noms
            paper_bgcolor='white', 
            plot_bgcolor='white',
            font=dict(family='DM Sans'),
            xaxis=dict(
                tickangle=45,                              # Inclinaison à 45° pour que les longs noms de KPIs (PRB, DL_Throughput...) soient lisibles
                automargin=True
            )
        )
    
        st.markdown("""<div class="card"><div class="card-title">📊 Feature Importances Globales — Modèle Détection</div>""", unsafe_allow_html=True)
        st.plotly_chart(fig_fi, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"Erreur feature importance : {e}")

    # ─ Images SHAP réelles ─
    st.markdown("""<div class="card"><div class="card-title">📈 Résultats Optuna — Détection LightGBM</div>""", unsafe_allow_html=True)
    img_opt = os.path.join(BASE_DIR, 'assets', 'lightgbm_optuna_resultats.png')
    if os.path.exists(img_opt):
        st.image(img_opt)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""<div class="card"><div class="card-title">🏅 Métriques Comparatives — 3 Horizons de Prédiction</div>""", unsafe_allow_html=True)
    img_metr = os.path.join(BASE_DIR, 'assets', 'lgbm_metriques_comparatif.png')
    if os.path.exists(img_metr):
        st.image(img_metr)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""<div class="card"><div class="card-title">🧬 SHAP TreeExplainer — 3 Horizons × 3 Classes</div>""", unsafe_allow_html=True)
    img_shap = os.path.join(BASE_DIR, 'assets', 'lgbm_shap_3targets_3classes.png')
    if os.path.exists(img_shap):
        st.image(img_shap)
    st.markdown("</div>", unsafe_allow_html=True)

    # Tableau récap SHAP top features
    st.markdown("""
    <div class="card">
      <div class="card-title">📋 Récapitulatif SHAP — Top Features par Horizon (Classe Congestionné)</div>
      <table class="data-table">
        <thead><tr><th>Rang</th><th>Horizon +1h</th><th>SHAP</th><th>Horizon +3h</th><th>SHAP</th><th>Horizon +6h</th><th>SHAP</th></tr></thead>
        <tbody>
          <tr><td>1</td><td>dl_prb_usage_rate</td><td>2.347</td><td>dl_prb_usage_rate</td><td>1.32</td><td>avaibility</td><td>1.46</td></tr>
          <tr><td>2</td><td>rolling_prb_3h</td><td>0.817</td><td>avg_user_nb</td><td>0.825</td><td>lag_prb_1h</td><td>0.56</td></tr>
          <tr><td>3</td><td>lag_prb_1h</td><td>0.4</td><td>rolling_trafic_3h</td><td>0.78</td><td>dl_prb_usage_rate</td><td>0.434</td></tr>
          <tr><td>4</td><td>hour_cos</td><td>0.25</td><td>rolling_prb_3h</td><td>0.49</td><td>lag_prb_3h</td><td>0.395</td></tr>
          <tr><td>5</td><td>avg_user_nb</td><td>0.2</td><td>lag_prb_3h</td><td>0.22</td><td>lag_prb_2h</td><td>0.297</td></tr>
        </tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    # Résumé performances
    st.markdown("""
    <div class="card">
      <div class="card-title">🏆 Rapport Final — Performances Modèles LightGBM + Optuna</div>
      <table class="data-table">
        <thead><tr><th>Métrique</th><th>Détection</th><th>Prédiction +1h</th><th>Prédiction +3h</th><th>Prédiction +6h</th></tr></thead>
        <tbody>
          <tr><td>Accuracy</td><td><strong>99.92%</strong></td><td>91.84%</td><td>89.39%</td><td>89.46%</td></tr>
          <tr><td>F1-macro</td><td><strong>99.94%</strong></td><td>93.53%</td><td>92.07%</td><td>92.12%</td></tr>
          <tr><td>F1 Normal (0)</td><td>99.930%</td><td>96.06%</td><td>93.36%</td><td>90.22%</td></tr>
          <tr><td>F1 Modéré (1)</td><td>99.993%</td><td>92.86%</td><td>88.14%</td><td>87.25%</td></tr>
          <tr><td>F1 Critique(2)</td><td>99.997%</td><td>99.90%</td><td>99.07%</td><td>98.72%</td></tr>
          <tr><td>ROC-AUC</td><td>—</td><td>99.42%</td><td>99.43%</td><td>97.48%</td></tr>
          <tr><td>Overfitting Gap</td><td>—</td><td>+0.0066 ✅</td><td>+0.029 ✅</td><td>+0.07 ✅</td><td>+0.06 ✅</td></tr>
        </tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🙏  Fin — Remerciements →", key="shap_fin", use_container_width=False):
        go_to('merci')


# PAGE 7 — MERCI

elif st.session_state.page == 'merci':
    st.markdown("""
    <div class="merci-page">
      <h1 class="merci-title">Merci de votre attention</h1>
      <p class="merci-sub">Nous restons disponibles pour vos questions.</p>
      <div class="merci-divider"></div>
      <div class="merci-row">
        <div class="merci-col">
          <div class="merci-label">Réalisé par</div>
          <div class="merci-val">Binôme — Master 2 Big Data<br>USTHB, Alger</div>
        </div>
        <div class="merci-col">
          <div class="merci-label">Encadrement</div>
          <div class="merci-val">Encadrant Académique<br>Département Informatique — USTHB</div>
        </div>
        <div class="merci-col">
          <div class="merci-label">Partenaire Industriel</div>
          <div class="merci-val" style='color:var(--red);font-weight:700'>📡 Djezzy<br><span style='color:var(--gray-600);font-weight:400'>Radio Network Operations</span></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    kpis_fin = [
        ("7.78M", "Données analysées"),
        ("99.94%", "F1-macro détection"),
        ("3 horizons", "Prédiction temporelle"),
        ("✅", "Pas d'overfitting"),
    ]
    for col, (val, lbl) in zip([c1,c2,c3,c4], kpis_fin):
        col.markdown(f"""<div class="kpi-card"><div class="kpi-value">{val}</div><div class="kpi-label">{lbl}</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    if st.button("🔁  Recommencer depuis l'Accueil", key="merci_home", use_container_width=False):
        go_to('home')