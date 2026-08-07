# ============================================================
# LIGHTGBM — RÉENTRAÎNEMENT FINAL + SHAP + TESTS OVERFITTING
# PFE Djezzy — Détection Congestion BTS LTE
#
# CORRECTION : Normalisation shape SHAP (compatibilité multi-versions)
# ============================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, learning_curve
)
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

import sys
sys.stdout.reconfigure(encoding='utf-8')

TEMPS_DEBUT = time.time()

def fmt(s):
    h, m = int(s // 3600), int((s % 3600) // 60)
    sec  = int(s % 60)
    return (f"{h}h {m:02d}min {sec:02d}s" if h > 0 else
            f"{m}min {sec:02d}s"           if m > 0 else
            f"{sec:.1f}s")

def elapsed():
    return fmt(time.time() - TEMPS_DEBUT)

print("=" * 65)
print("  LIGHTGBM FINAL + SHAP + TESTS OVERFITTING")
print("  PFE Djezzy — Détection Congestion BTS LTE")
print(f"  Début : {time.strftime('%H:%M:%S')}")
print("=" * 65)

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
RANDOM_STATE = 42
TARGET       = 'Classe_Congestion'
TEST_SIZE    = 0.20
COLORS       = {0: '#27AE60', 1: '#F39C12', 2: '#E74C3C'}
LABELS       = {0: 'Normal',  1: 'Modéré',  2: 'Critique'}

# ── Meilleurs paramètres trouvés par Optuna ────────────────────
BEST_PARAMS_OPTUNA = {
    "n_estimators"     : 1985,
    "num_leaves"       : 21,
    "learning_rate"    : 0.1511461721588397,
    "subsample"        : 0.6026268487517209,
    "colsample_bytree" : 0.9821888658123518,
    "reg_alpha"        : 0.7179399230647678,
    "reg_lambda"       : 9.511312957867718,
    "min_child_samples": 77,
}

print(f"\n  Paramètres Optuna (F1 CV = 99.909% | 50 trials) :")
for k, v in BEST_PARAMS_OPTUNA.items():
    print(f"    {k:<22} = {v}")

# ══════════════════════════════════════════════════════════════
# [1/8] CHARGEMENT
# ══════════════════════════════════════════════════════════════
t1 = time.time()
print(f"\n[1/8] Chargement...  [{elapsed()}]")

df = pd.read_csv('df_avec_score_kmeans.csv')

FEATURES = [
    'LTE_Setup_Success_Rate','Cell_Traffic_Volume_DL', 
    'Cell_Traffic_Volume_Ul','DL_Average_Throughput', 
    'Ul_Average_Throughput','Avg_User_NB', 'Avaibility',
    'HOUR', 'Spectral_Eff', 'Rolling_PRB_3h','PRB_Z_Score', 
    'Gradient_PRB','DL_PRB_Usage_Rate','PRB_per_User'
]

FEATURES = [f for f in FEATURES if f in df.columns]

X = df[FEATURES].fillna(df[FEATURES].median())
y = df[TARGET].astype(int)

N_TOTAL    = len(df)
N_FEATURES = len(FEATURES)
N_CLASSES  = 3

print(f"  Dataset : {N_TOTAL:,} lignes | {N_FEATURES} features")
print(f"  Shape X : {X.shape}  →  [N_lignes={N_TOTAL:,}, N_features={N_FEATURES}]")
for cls in [0, 1, 2]:
    n = (y == cls).sum()
    print(f"  Classe {cls} {LABELS[cls]:<10}: {n:>10,} ({n/len(y)*100:.1f}%)")
print(f"  Durée : {fmt(time.time()-t1)}")

# ══════════════════════════════════════════════════════════════
# [2/8] SPLIT + POIDS
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# ÉTAPE 2 — SPLIT TRAIN / VALIDATION / TEST (70/15/15)
# ══════════════════════════════════════════════════════════════
t2 = time.time()
print(f"\n[2/8] Split 70/15/15 stratifié...  [{elapsed()}]")

# 70% Train - 30% Temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

# 15% Validation - 15% Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

# Calcul des poids sur le TRAIN uniquement
counts = y_train.value_counts().sort_index()

class_weights = {
    cls: len(y_train) / (N_CLASSES * cnt)
    for cls, cnt in counts.items()
}

sample_weights = y_train.map(class_weights).values

print(f"  X_train shape : {X_train.shape}  →  [N_train={len(X_train):,}, features={N_FEATURES}]")
print(f"  X_val   shape : {X_val.shape}    →  [N_val={len(X_val):,}, features={N_FEATURES}]")
print(f"  X_test  shape : {X_test.shape}   →  [N_test={len(X_test):,}, features={N_FEATURES}]")

print(
    f"  Poids : "
    + " | ".join([f"Cl{c}={w:.3f}" for c, w in class_weights.items()])
)

print(f"  Durée : {fmt(time.time()-t2)}")
# ══════════════════════════════════════════════════════════════
# [3/8] RÉENTRAÎNEMENT FINAL
# ══════════════════════════════════════════════════════════════
t3 = time.time()
print(f"\n[3/8] Réentraînement LightGBM...  [{elapsed()}]")

final_params = {
    **BEST_PARAMS_OPTUNA,
    'objective'    : 'multiclass',
    'num_class'    : N_CLASSES,
    'metric'       : 'multi_logloss',
    'class_weight' : class_weights,
    'n_jobs'       : -1,
    'random_state' : RANDOM_STATE,
    'verbosity'    : -1,
}

lgbm_final = lgb.LGBMClassifier(**final_params)
lgbm_final.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(100)
    ]
)

n_trees = lgbm_final.best_iteration_
print(f"  n_estimators utilisés : {n_trees}")
print(f"  Durée : {fmt(time.time()-t3)}")

# ── Métriques de base ─────────────────────────────────────────
y_pred       = lgbm_final.predict(X_test)
y_proba      = lgbm_final.predict_proba(X_test)
y_train_pred = lgbm_final.predict(X_train)
y_test_np    = np.array(y_test)
y_train_np   = np.array(y_train)

acc_test  = accuracy_score(y_test_np,  y_pred)
acc_train = accuracy_score(y_train_np, y_train_pred)
f1_test   = f1_score(y_test_np,  y_pred,       average='macro')
f1_train  = f1_score(y_train_np, y_train_pred, average='macro')
f1_w      = f1_score(y_test_np,  y_pred,       average='weighted')
f1_cls    = f1_score(y_test_np,  y_pred,       average=None)
roc_auc   = roc_auc_score(y_test_np, y_proba,  multi_class='ovr')

print(f"""
  ╔══════════════════════════════════════════════╗
  ║   LIGHTGBM — RÉSULTATS FINAUX               ║
  ╠══════════════════════════════════════════════╣
  ║  Train Accuracy    : {acc_train*100:>8.3f}%            ║
  ║  Test  Accuracy    : {acc_test*100:>8.3f}%            ║
  ║  Différence        : {abs(acc_train-acc_test)*100:>8.4f}%            ║
  ╠══════════════════════════════════════════════╣
  ║  Train F1-macro    : {f1_train*100:>8.3f}%            ║
  ║  Test  F1-macro    : {f1_test*100:>8.3f}%            ║
  ║  F1 Critique       : {f1_cls[2]*100:>8.3f}%            ║
  ║  ROC-AUC           : {roc_auc*100:>8.3f}%            ║
  ╚══════════════════════════════════════════════╝
""")
print(classification_report(y_test_np, y_pred,
    target_names=[LABELS[i] for i in range(N_CLASSES)]))

# ══════════════════════════════════════════════════════════════
# [4/8] TESTS ANTI-OVERFITTING
# ══════════════════════════════════════════════════════════════
print(f"\n[4/8] Tests anti-overfitting...  [{elapsed()}]")
print("=" * 65)
overfitting_resultats = {}

# ── TEST 1 : Gap Train/Test ───────────────────────────────────
print("\n  TEST 1 — Gap Train vs Test")
print("  " + "-" * 40)
gap_acc = abs(acc_train - acc_test)
gap_f1  = abs(f1_train  - f1_test)
print(f"  Train Accuracy : {acc_train*100:.4f}%")
print(f"  Test  Accuracy : {acc_test*100:.4f}%")
print(f"  GAP Accuracy   : {gap_acc*100:.4f}%")
print(f"  Train F1-macro : {f1_train*100:.4f}%")
print(f"  Test  F1-macro : {f1_test*100:.4f}%")
print(f"  GAP F1-macro   : {gap_f1*100:.4f}%")

if gap_acc < 0.005:
    t1_status = "✅ PASSÉ — Gap < 0.5% (pas d'overfitting)"
elif gap_acc < 0.02:
    t1_status = "⚠️  LÉGÈRE différence (0.5%-2%) — acceptable"
else:
    t1_status = "❌ ÉCHOUÉ — Gap > 2% (overfitting probable)"
print(f"  → {t1_status}")
overfitting_resultats['test1_gap'] = gap_acc

# ── TEST 2 : Cross-validation 5 folds ────────────────────────
print("\n  TEST 2 — Cross-Validation Stratifiée (5 folds)")
print("  " + "-" * 40)
print("  ⏳ Calcul sur sous-échantillon 300k lignes...")

t_cv = time.time()
rng      = np.random.default_rng(42)
idx_0    = np.where(y_train_np == 0)[0]
idx_1    = np.where(y_train_np == 1)[0]
idx_2    = np.where(y_train_np == 2)[0]
n_cv     = 300_000
n0, n1   = int(n_cv * 0.55), int(n_cv * 0.40)
n2       = n_cv - n0 - n1
cv_idx   = np.concatenate([
    rng.choice(idx_0, min(n0, len(idx_0)), replace=False),
    rng.choice(idx_1, min(n1, len(idx_1)), replace=False),
    rng.choice(idx_2, min(n2, len(idx_2)), replace=False),
])
X_cv = X_train.iloc[cv_idx].values
y_cv = y_train_np[cv_idx]

cv_model = lgb.LGBMClassifier(
    n_estimators     = min(500, n_trees),
    num_leaves       = BEST_PARAMS_OPTUNA['num_leaves'],
    learning_rate    = BEST_PARAMS_OPTUNA['learning_rate'],
    reg_alpha        = BEST_PARAMS_OPTUNA['reg_alpha'],
    reg_lambda       = BEST_PARAMS_OPTUNA['reg_lambda'],
    min_child_samples= BEST_PARAMS_OPTUNA['min_child_samples'],
    objective        = 'multiclass',
    num_class        = N_CLASSES,
    n_jobs           = -1,
    random_state     = RANDOM_STATE,
    verbosity        = -1,
)

skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1s  = []
cv_accs = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_cv, y_cv), 1):
    cv_model.fit(X_cv[tr_idx], y_cv[tr_idx])
    preds    = cv_model.predict(X_cv[va_idx])
    f1_fold  = f1_score(y_cv[va_idx], preds, average='macro')
    acc_fold = accuracy_score(y_cv[va_idx], preds)
    cv_f1s.append(f1_fold)
    cv_accs.append(acc_fold)
    print(f"  Fold {fold}/5 | F1={f1_fold*100:.3f}% | Acc={acc_fold*100:.3f}%")

cv_mean = np.mean(cv_f1s)
cv_std  = np.std(cv_f1s)
cv_min  = np.min(cv_f1s)
cv_max  = np.max(cv_f1s)
print(f"\n  CV F1-macro : {cv_mean*100:.3f}% ± {cv_std*100:.3f}%")
print(f"  Min={cv_min*100:.3f}% | Max={cv_max*100:.3f}%")
print(f"  F1 test final : {f1_test*100:.3f}%")
print(f"  Écart CV/Test : {abs(cv_mean - f1_test)*100:.4f}%")

if cv_std < 0.005:
    t2_status = "✅ PASSÉ — Variance CV < 0.5% (modèle stable)"
elif cv_std < 0.015:
    t2_status = "⚠️  VARIANCE MODÉRÉE (0.5%-1.5%) — acceptable"
else:
    t2_status = "❌ ÉCHOUÉ — Variance > 1.5% (instabilité)"
print(f"  → {t2_status}")
overfitting_resultats['test2_cv_mean'] = cv_mean
overfitting_resultats['test2_cv_std']  = cv_std
print(f"  Durée CV : {fmt(time.time()-t_cv)}")

# ── TEST 3 : Learning Curve ───────────────────────────────────
print("\n  TEST 3 — Learning Curve (train size progressif)")
print("  " + "-" * 40)
print("  ⏳ Calcul learning curve sur 100k lignes...")

t_lc = time.time()
idx_lc = rng.choice(len(cv_idx), min(100_000, len(cv_idx)), replace=False)
X_lc   = X_cv[idx_lc]
y_lc   = y_cv[idx_lc]

lc_model = lgb.LGBMClassifier(
    n_estimators=200, num_leaves=BEST_PARAMS_OPTUNA['num_leaves'],
    learning_rate=BEST_PARAMS_OPTUNA['learning_rate'],
    objective='multiclass', num_class=N_CLASSES,
    n_jobs=-1, random_state=42, verbosity=-1,
)

train_sizes_abs, train_scores, val_scores = learning_curve(
    lc_model, X_lc, y_lc,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=3, scoring='f1_macro',
    n_jobs=-1
)

lc_train_mean = train_scores.mean(axis=1)
lc_train_std  = train_scores.std(axis=1)
lc_val_mean   = val_scores.mean(axis=1)
lc_val_std    = val_scores.std(axis=1)
lc_gap_final  = abs(lc_train_mean[-1] - lc_val_mean[-1])

print(f"  Train F1 final (100% data) : {lc_train_mean[-1]*100:.3f}%")
print(f"  Val   F1 final (100% data) : {lc_val_mean[-1]*100:.3f}%")
print(f"  Écart Train/Val final      : {lc_gap_final*100:.3f}%")
print(f"  Tendance : train F1 →  {lc_train_mean[0]*100:.1f}% → {lc_train_mean[-1]*100:.1f}%")
print(f"  Tendance : val   F1 →  {lc_val_mean[0]*100:.1f}% → {lc_val_mean[-1]*100:.1f}%")

if lc_gap_final < 0.02:
    t3_status = "✅ PASSÉ — Courbes convergentes (pas d'overfitting)"
elif lc_gap_final < 0.05:
    t3_status = "⚠️  LÉGER ÉCART — Modèle légèrement surajusté"
else:
    t3_status = "❌ ÉCHOUÉ — Grande divergence train/val"
print(f"  → {t3_status}")
overfitting_resultats['test3_lc_gap'] = lc_gap_final
print(f"  Durée LC : {fmt(time.time()-t_lc)}")

# ── TEST 4 : Stabilité par sous-groupe ───────────────────────
print("\n  TEST 4 — Stabilité sur sous-groupes (robustesse)")
print("  " + "-" * 40)

rng2   = np.random.default_rng(123)
f1_sub = []
for trial in range(5):
    sub_idx  = rng2.choice(len(X_test), size=min(50_000, len(X_test)), replace=False)
    X_sub    = X_test.iloc[sub_idx] if hasattr(X_test, 'iloc') else X_test[sub_idx]
    y_sub    = y_test_np[sub_idx]
    pred_sub = lgbm_final.predict(X_sub)
    f1_sub.append(f1_score(y_sub, pred_sub, average='macro'))
    print(f"  Sous-groupe {trial+1}/5 (50k) | F1={f1_sub[-1]*100:.4f}%")

f1_sub_std = np.std(f1_sub)
print(f"\n  F1 moyen : {np.mean(f1_sub)*100:.4f}% ± {f1_sub_std*100:.4f}%")

if f1_sub_std < 0.003:
    t4_status = "✅ PASSÉ — F1 stable sur tous les sous-groupes"
elif f1_sub_std < 0.01:
    t4_status = "⚠️  LÉGÈRE VARIANCE — acceptable"
else:
    t4_status = "❌ ÉCHOUÉ — F1 instable selon le sous-groupe"
print(f"  → {t4_status}")
overfitting_resultats['test4_stability'] = f1_sub_std

# ── Récapitulatif des tests ───────────────────────────────────
print("\n" + "=" * 65)
print("  RÉCAPITULATIF DES TESTS ANTI-OVERFITTING")
print("=" * 65)
print(f"  Test 1 Gap Train/Test : {gap_f1*100:.4f}%      → {t1_status}")
print(f"  Test 2 CV 5-folds     : {cv_std*100:.4f}% std  → {t2_status}")
print(f"  Test 3 Learning Curve : {lc_gap_final*100:.4f}% gap → {t3_status}")
print(f"  Test 4 Stabilité      : {f1_sub_std*100:.4f}% std  → {t4_status}")
print("=" * 65)

# ══════════════════════════════════════════════════════════════
# [5/8] SHAP — PARAMÈTRES AUTOMATISÉS SELON DATASET
# ══════════════════════════════════════════════════════════════
print(f"\n[5/8] SHAP — Analyse explicabilité...  [{elapsed()}]")
print("=" * 65)

def calculer_params_shap(n_test, n_features):
    if n_test < 100_000:
        n_shap = 5_000
    elif n_test < 500_000:
        n_shap = 10_000
    else:
        n_shap = 20_000
    n_shap = min(n_shap, n_test)
    n_bg   = 100 if n_test < 100_000 else min(500, n_test // 100)
    n_top  = min(n_features, 14)
    return {'n_shap': n_shap, 'n_background': n_bg,
            'n_top_features': n_top, 'stratified': True}

shap_params = calculer_params_shap(len(X_test), N_FEATURES)

print(f"\n  Paramètres SHAP automatisés :")
print(f"    N test total         : {len(X_test):,}")
print(f"    N features           : {N_FEATURES}")
print(f"    n_shap (échantillon) : {shap_params['n_shap']:,}")
print(f"    n_background         : {shap_params['n_background']}")
print(f"    n_top_features       : {shap_params['n_top_features']}")
print(f"    Stratified           : {shap_params['stratified']}")

# ── Échantillon SHAP stratifié ────────────────────────────────
n_shap = shap_params['n_shap']
rng3   = np.random.default_rng(42)
idx_0s = np.where(y_test_np == 0)[0]
idx_1s = np.where(y_test_np == 1)[0]
idx_2s = np.where(y_test_np == 2)[0]
n0s    = int(n_shap * 0.55)
n1s    = int(n_shap * 0.35)
n2s    = n_shap - n0s - n1s

shap_idx = np.concatenate([
    rng3.choice(idx_0s, min(n0s, len(idx_0s)), replace=False),
    rng3.choice(idx_1s, min(n1s, len(idx_1s)), replace=False),
    rng3.choice(idx_2s, min(n2s, len(idx_2s)), replace=False),
])
X_shap = X_test.iloc[shap_idx].values if hasattr(X_test, 'iloc') else X_test[shap_idx]
y_shap = y_test_np[shap_idx]

print(f"\n  Échantillon SHAP : {len(X_shap):,} séquences")
print(f"    Normal   : {(y_shap==0).sum():,} ({(y_shap==0).mean()*100:.1f}%)")
print(f"    Modéré   : {(y_shap==1).sum():,} ({(y_shap==1).mean()*100:.1f}%)")
print(f"    Critique : {(y_shap==2).sum():,} ({(y_shap==2).mean()*100:.1f}%)")

# ══════════════════════════════════════════════════════════════
# CORRECTION PRINCIPALE : Normalisation shape SHAP
# ══════════════════════════════════════════════════════════════
print(f"\n  Calcul TreeExplainer...")
t_shap    = time.time()
explainer = shap.TreeExplainer(lgbm_final)
shap_raw  = explainer.shap_values(X_shap)

# --- Détection et normalisation du format de sortie SHAP ---
# Les versions récentes de SHAP+LightGBM retournent un tableau 3D
# (n_samples, n_features, n_classes) au lieu d'une liste de matrices 2D.
# On normalise ici pour garantir toujours : shap_vals[i].shape == (n_samples, n_features)
if isinstance(shap_raw, np.ndarray):
    if shap_raw.ndim == 3:
        # Format 3D : (n_samples, n_features, n_classes)
        shap_vals = [shap_raw[:, :, i] for i in range(N_CLASSES)]
        print(f"  Format 3D détecté {shap_raw.shape} → converti en liste de {N_CLASSES} matrices 2D")
    elif shap_raw.ndim == 2:
        # Format 2D inattendu pour multiclasse — on tente de découper
        shap_vals = [shap_raw] * N_CLASSES
        print(f"  ⚠️  Format 2D inattendu {shap_raw.shape} — duplication pour {N_CLASSES} classes")
    else:
        raise ValueError(f"Format SHAP numpy inattendu : ndim={shap_raw.ndim}, shape={shap_raw.shape}")
elif isinstance(shap_raw, list):
    # Ancien format : liste de N_CLASSES tableaux (n_samples, n_features)
    shap_vals = shap_raw
    print(f"  Format liste détecté ({len(shap_vals)} classes)")
else:
    raise ValueError(f"Type SHAP inattendu : {type(shap_raw)}")

# Vérification finale
assert len(shap_vals) == N_CLASSES, \
    f"Nombre de classes SHAP incorrect : {len(shap_vals)} ≠ {N_CLASSES}"
assert shap_vals[0].shape == (len(X_shap), N_FEATURES), \
    f"Shape SHAP incorrecte : {shap_vals[0].shape} ≠ ({len(X_shap)}, {N_FEATURES})"

print(f"  ✅ Shape SHAP validée : {N_CLASSES} classes × {shap_vals[0].shape}")
print(f"    shap_vals[0] (Normal)   : {shap_vals[0].shape}  → [n_échant, n_feat]")
print(f"    shap_vals[1] (Modéré)   : {shap_vals[1].shape}  → [n_échant, n_feat]")
print(f"    shap_vals[2] (Critique) : {shap_vals[2].shape}  → [n_échant, n_feat]")
print(f"  Durée TreeExplainer : {fmt(time.time()-t_shap)}")

# ══════════════════════════════════════════════════════════════
# [6/8] GRAPHIQUES SHAP
# ══════════════════════════════════════════════════════════════
print(f"\n[6/8] Graphiques SHAP...  [{elapsed()}]")

n_top     = shap_params['n_top_features']
X_shap_df = pd.DataFrame(X_shap, columns=FEATURES)

# ── G1 : Summary Plot Classe Critique ────────────────────────
print("  Graphique 1/5 : Summary Plot — Classe Critique...")
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_vals[2],   # classe Critique — shape (n_samples, n_features) ✅
    X_shap_df,
    max_display=n_top,
    plot_type='dot',
    show=False,
    color_bar=True,
)
plt.title(
    f'SHAP — Impact des features sur Classe 2 (Critique)\n'
    f'Échantillon = {n_shap:,} obs. (stratifié) | Top {n_top} features',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig('shap_summary_critique.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✅ shap_summary_critique.png")

# ── G2 : Bar Plot toutes classes ─────────────────────────────
print("  Graphique 2/5 : Bar Plot — Importance SHAP toutes classes...")
fig, axes_bar = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle(
    f'SHAP Bar Plot — Importance absolue par classe\n'
    f'Échantillon = {n_shap:,} obs. | Top {n_top} features',
    fontsize=12, fontweight='bold'
)
for cls_i, (cls_name, color) in enumerate(
        [(LABELS[0], '#27AE60'), (LABELS[1], '#F39C12'), (LABELS[2], '#E74C3C')]):
    mean_abs     = np.abs(shap_vals[cls_i]).mean(axis=0)
    feat_imp     = pd.Series(mean_abs, index=FEATURES).sort_values(ascending=True)
    feat_imp_top = feat_imp.tail(n_top)
    axes_bar[cls_i].barh(feat_imp_top.index, feat_imp_top.values,
                          color=color, alpha=0.85, edgecolor='white')
    axes_bar[cls_i].set_title(f'Classe {cls_i} — {cls_name}', fontweight='bold')
    axes_bar[cls_i].set_xlabel('|SHAP| moyen')
    axes_bar[cls_i].grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('shap_bar_par_classe.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✅ shap_bar_par_classe.png")

# ── G3 : Dependence Plot PRB ─────────────────────────────────
print("  Graphique 3/5 : Dependence Plot — PRB vs SHAP Critique...")
if 'DL_PRB_Usage_Rate' in FEATURES:
    prb_idx = FEATURES.index('DL_PRB_Usage_Rate')
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        prb_idx,
        shap_vals[2],
        X_shap_df,
        interaction_index=None,
        show=False,
        alpha=0.3,
        dot_size=5,
    )
    plt.title(
        'SHAP Dependence — DL_PRB_Usage_Rate → Impact sur Classe Critique\n'
        '(+ SHAP = pousse vers Critique | – SHAP = pousse vers Normal)',
        fontsize=11, fontweight='bold'
    )
    plt.xlabel('DL_PRB_Usage_Rate (%)')
    plt.ylabel('Valeur SHAP pour Critique')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('shap_dependence_prb.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✅ shap_dependence_prb.png")

# ── G4 : Waterfall pour 1 exemple Critique ───────────────────
print("  Graphique 4/5 : Waterfall — Un exemple Critique...")
idx_critiques = np.where(y_shap == 2)[0]
if len(idx_critiques) > 0:
    ex_idx  = idx_critiques[0]
    ex_shap = shap.Explanation(
        values      = shap_vals[2][ex_idx],
        base_values = explainer.expected_value[2],
        data        = X_shap[ex_idx],
        feature_names=FEATURES,
    )
    plt.figure(figsize=(12, 7))
    shap.waterfall_plot(ex_shap, max_display=n_top, show=False)
    plt.title(
        f'SHAP Waterfall — Exemple Classe Critique (index {ex_idx})\n'
        f'Montre comment chaque feature pousse vers Critique',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('shap_waterfall_critique.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✅ shap_waterfall_critique.png")

# ── G5 : Heatmap SHAP top features ───────────────────────────
print("  Graphique 5/5 : Heatmap SHAP — Critique...")
plt.figure(figsize=(14, 7))
top_feat_idx   = np.argsort(np.abs(shap_vals[2]).mean(axis=0))[::-1][:8]
top_feat_names = [FEATURES[i] for i in top_feat_idx]
shap_top       = shap_vals[2][:, top_feat_idx]
n_heat         = min(500, len(shap_top))
heat_idx       = rng3.choice(len(shap_top), n_heat, replace=False)
shap_heat      = shap_top[heat_idx]
sns.heatmap(
    shap_heat.T,
    cmap='RdBu_r', center=0,
    xticklabels=False,
    yticklabels=top_feat_names,
    vmin=np.percentile(shap_heat, 5),
    vmax=np.percentile(shap_heat, 95),
)
plt.title(
    f'SHAP Heatmap — Top 8 features pour Classe Critique\n'
    f'{n_heat} observations (rouge=pousse vers Critique, bleu=pousse vers Normal)',
    fontsize=11, fontweight='bold'
)
plt.ylabel('Features')
plt.xlabel(f'Observations (sample de {n_heat})')
plt.tight_layout()
plt.savefig('shap_heatmap_critique.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✅ shap_heatmap_critique.png")

# ── Tableau SHAP importance ───────────────────────────────────
print(f"\n  Top 10 features par SHAP absolu (Critique) :")
mean_abs_crit = np.abs(shap_vals[2]).mean(axis=0)
shap_df = pd.DataFrame({
    'Feature'      : FEATURES,
    'SHAP_Normal'  : np.abs(shap_vals[0]).mean(axis=0),
    'SHAP_Modere'  : np.abs(shap_vals[1]).mean(axis=0),
    'SHAP_Critique': mean_abs_crit,
}).sort_values('SHAP_Critique', ascending=False)
print(shap_df.head(10).to_string(index=False))
shap_df.to_csv('shap_importance_par_classe.csv', index=False)
print("  ✅ shap_importance_par_classe.csv")

# ══════════════════════════════════════════════════════════════
# [7/8] GRAPHIQUES ML CLASSIQUES + OVERFITTING
# ══════════════════════════════════════════════════════════════
print(f"\n[7/8] Graphiques ML + Overfitting...  [{elapsed()}]")

fig = plt.figure(figsize=(24, 16))
fig.patch.set_facecolor('#F0F2F5')
fig.suptitle(
    f'LightGBM Final — Détection Congestion BTS Djezzy\n'
    f'F1-macro={f1_test*100:.3f}% | F1-Critique={f1_cls[2]*100:.3f}% | '
    f'Accuracy={acc_test*100:.3f}% | ROC-AUC={roc_auc*100:.3f}%',
    fontsize=14, fontweight='bold'
)

axes = fig.subplots(3, 4)

# ── G1 : Matrice confusion ────────────────────────────────────
ax = axes[0, 0]
ax.set_facecolor('white')
cm_mat = confusion_matrix(y_test_np, y_pred)
cm_pct = cm_mat.astype(float) / cm_mat.sum(axis=1, keepdims=True) * 100
im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        c = 'white' if cm_pct[i, j] > 60 else 'black'
        ax.text(j, i, f'{cm_pct[i,j]:.1f}%\n({cm_mat[i,j]:,})',
                ha='center', va='center', fontsize=8, color=c, fontweight='bold')
ax.set_xticks(range(N_CLASSES))
ax.set_yticks(range(N_CLASSES))
ax.set_xticklabels([f'Prédit\n{LABELS[i]}' for i in range(N_CLASSES)], fontsize=8)
ax.set_yticklabels([f'Réel\n{LABELS[i]}' for i in range(N_CLASSES)],  fontsize=8)
ax.set_title('Matrice de Confusion (%)', fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046)

# ── G2 : F1 par classe ───────────────────────────────────────
ax = axes[0, 1]
ax.set_facecolor('white')
bars = ax.bar([LABELS[i] for i in range(N_CLASSES)],
              [f1_cls[i]*100 for i in range(N_CLASSES)],
              color=[COLORS[i] for i in range(N_CLASSES)],
              alpha=0.85, edgecolor='white', width=0.5)
for bar, val in zip(bars, f1_cls):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val*100:.3f}%', ha='center', va='bottom',
            fontweight='bold', fontsize=10)
ax.axhline(f1_test*100, color='#2C3E50', linewidth=1.5, linestyle='--',
           label=f'F1 macro={f1_test*100:.3f}%')
ax.set_ylim(0, 115)
ax.set_ylabel('F1-score (%)')
ax.set_title('F1-score par Classe', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# ── G3 : ROC ─────────────────────────────────────────────────
ax = axes[0, 2]
ax.set_facecolor('white')
y_test_bin = label_binarize(y_test_np, classes=[0, 1, 2])
from sklearn.metrics import roc_curve, auc as sk_auc
for cls_i, color in enumerate([COLORS[0], COLORS[1], COLORS[2]]):
    fpr, tpr, _ = roc_curve(y_test_bin[:, cls_i], y_proba[:, cls_i])
    auc_score   = sk_auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f'{LABELS[cls_i]} (AUC={auc_score:.4f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('Courbes ROC par Classe', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# ── G4 : Distribution P(Critique) ────────────────────────────
ax = axes[0, 3]
ax.set_facecolor('white')
for cls in [0, 1, 2]:
    mask = y_test_np == cls
    ax.hist(y_proba[:, 2][mask], bins=60, color=COLORS[cls], alpha=0.70,
            label=f'{LABELS[cls]} ({mask.sum():,})', edgecolor='white',
            linewidth=0.2, density=True)
ax.axvline(0.5, color='black', linewidth=2, linestyle='--', label='Seuil=0.5')
ax.set_title("Distribution P(Critique)\npar classe réelle", fontweight='bold')
ax.set_xlabel("P(Critique)")
ax.set_ylabel('Densité'); ax.set_yscale('log')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── G5 : TEST 1 — Train vs Test ──────────────────────────────
ax = axes[1, 0]
ax.set_facecolor('white')
f1_train_cls = f1_score(y_train_np, y_train_pred, average=None)
metrics    = ['Acc', 'F1-mac', 'F1-Nor', 'F1-Mod', 'F1-Crit']
train_vals = [acc_train*100, f1_train*100,
              f1_train_cls[0]*100, f1_train_cls[1]*100, f1_train_cls[2]*100]
test_vals  = [acc_test*100, f1_test*100,
              f1_cls[0]*100, f1_cls[1]*100, f1_cls[2]*100]
x = np.arange(len(metrics)); w = 0.35
ax.bar(x - w/2, train_vals, w, label='Train', color='steelblue', alpha=0.85)
ax.bar(x + w/2, test_vals,  w, label='Test',  color='orange',   alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=9)
ax.set_ylim([max(0, min(min(train_vals), min(test_vals))-2), 102])
ax.set_title(f'TEST 1 — Train vs Test\nGap={gap_acc*100:.4f}%  {t1_status[:2]}',
             fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

# ── G6 : TEST 2 — CV 5 folds ─────────────────────────────────
ax = axes[1, 1]
ax.set_facecolor('white')
folds   = [f'Fold {i+1}' for i in range(5)]
bars_cv = ax.bar(folds, [f*100 for f in cv_f1s],
                 color='#5B9BD5', alpha=0.85, edgecolor='white')
for bar, val in zip(bars_cv, cv_f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{val*100:.2f}%', ha='center', va='bottom', fontsize=9)
ax.axhline(cv_mean*100, color='red', linewidth=2, linestyle='--',
           label=f'Moy={cv_mean*100:.3f}% ±{cv_std*100:.3f}%')
ax.axhline(f1_test*100, color='green', linewidth=1.5, linestyle=':',
           label=f'Test={f1_test*100:.3f}%')
ax.set_ylim([max(0, cv_mean*100 - 5), min(102, cv_mean*100 + 5)])
ax.set_title(f'TEST 2 — CV 5 folds (300k lignes)\nσ={cv_std*100:.4f}%  {t2_status[:2]}',
             fontweight='bold')
ax.set_ylabel('F1-macro (%)')
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

# ── G7 : TEST 3 — Learning Curve ─────────────────────────────
ax = axes[1, 2]
ax.set_facecolor('white')
train_sizes_pct = train_sizes_abs / train_sizes_abs.max() * 100
ax.plot(train_sizes_pct, lc_train_mean*100, 'o-', color='steelblue',
        linewidth=2, label='Train F1')
ax.fill_between(train_sizes_pct,
                (lc_train_mean - lc_train_std)*100,
                (lc_train_mean + lc_train_std)*100,
                alpha=0.1, color='steelblue')
ax.plot(train_sizes_pct, lc_val_mean*100, 'o--', color='orange',
        linewidth=2, label='Val F1 (CV-3)')
ax.fill_between(train_sizes_pct,
                (lc_val_mean - lc_val_std)*100,
                (lc_val_mean + lc_val_std)*100,
                alpha=0.1, color='orange')
ax.set_xlabel('% données train utilisées')
ax.set_ylabel('F1-macro (%)')
ax.set_title(f'TEST 3 — Learning Curve\nGap final={lc_gap_final*100:.4f}%  {t3_status[:2]}',
             fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# ── G8 : TEST 4 — Stabilité sous-groupes ─────────────────────
ax = axes[1, 3]
ax.set_facecolor('white')
grps    = [f'SG {i+1}\n(50k)' for i in range(5)]
bars_sg = ax.bar(grps, [f*100 for f in f1_sub],
                 color='#9B59B6', alpha=0.85, edgecolor='white')
for bar, val in zip(bars_sg, f1_sub):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val*100:.4f}%', ha='center', va='bottom', fontsize=9)
ax.axhline(np.mean(f1_sub)*100, color='red', linewidth=2, linestyle='--',
           label=f'Moy={np.mean(f1_sub)*100:.4f}% ±{f1_sub_std*100:.4f}%')
ax.set_title(f'TEST 4 — Stabilité sous-groupes\nσ={f1_sub_std*100:.4f}%  {t4_status[:2]}',
             fontweight='bold')
ax.set_ylabel('F1-macro (%)')
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

# ── G9 : Feature Importance LightGBM ─────────────────────────
ax = axes[2, 0]
ax.set_facecolor('white')
fi  = pd.Series(lgbm_final.feature_importances_, index=FEATURES).sort_values()
cfi = ['#E74C3C' if 'PRB' in f else '#3498DB' for f in fi.index]
ax.barh(fi.index, fi.values, color=cfi, alpha=0.85, edgecolor='white')
ax.set_title('Feature Importance (LightGBM gain)', fontweight='bold')
ax.set_xlabel('Importance (gain)')
ax.legend(handles=[
    mpatches.Patch(color='#E74C3C', label='Features PRB'),
    mpatches.Patch(color='#3498DB', label='Autres KPIs')
], fontsize=9)
ax.grid(alpha=0.3, axis='x')

# ── G10 : SHAP bar résumé (Critique) ─────────────────────────
ax = axes[2, 1]
ax.set_facecolor('white')
shap_sorted  = shap_df.sort_values('SHAP_Critique', ascending=True).tail(n_top)
colors_shap  = ['#E74C3C' if 'PRB' in f else '#3498DB'
                for f in shap_sorted['Feature']]
ax.barh(shap_sorted['Feature'], shap_sorted['SHAP_Critique'],
        color=colors_shap, alpha=0.85, edgecolor='white')
ax.set_title('SHAP |moyen| — Classe Critique\n(impact sur prédiction Critique)',
             fontweight='bold')
ax.set_xlabel('|SHAP| moyen')
ax.legend(handles=[
    mpatches.Patch(color='#E74C3C', label='Features PRB'),
    mpatches.Patch(color='#3498DB', label='Autres KPIs')
], fontsize=9)
ax.grid(alpha=0.3, axis='x')

# ── G11 : Tableau récapitulatif overfitting ───────────────────
ax = axes[2, 2]
ax.set_facecolor('white')
ax.axis('off')
recap_ov = [
    ['TEST 1 Gap Acc Train/Test',  f'{gap_acc*100:.4f}%',      t1_status[:2]],
    ['TEST 1 Gap F1  Train/Test',  f'{gap_f1*100:.4f}%',       t1_status[:2]],
    ['TEST 2 CV F1 Moyen',         f'{cv_mean*100:.3f}%',      ''],
    ['TEST 2 CV F1 σ (5 folds)',   f'{cv_std*100:.4f}%',       t2_status[:2]],
    ['TEST 3 LC Gap final',        f'{lc_gap_final*100:.4f}%', t3_status[:2]],
    ['TEST 4 Stabilité σ',         f'{f1_sub_std*100:.4f}%',   t4_status[:2]],
    ['─────────────────', '────────', '──'],
    ['Accuracy Test',       f'{acc_test*100:.3f}%',  ''],
    ['F1-macro Test',       f'{f1_test*100:.3f}%',   ''],
    ['F1 Critique Test',    f'{f1_cls[2]*100:.3f}%', ''],
    ['ROC-AUC',             f'{roc_auc*100:.3f}%',   ''],
    ['n_estimators final',  f'{n_trees}',             ''],
]
tbl = ax.table(cellText=recap_ov,
    colLabels=['Métrique', 'Valeur', 'Status'],
    cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.3)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2E75B6')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#F5F9FF')
    cell.set_edgecolor('#CCCCCC')
ax.set_title('Tests Anti-Overfitting — Récapitulatif', fontweight='bold')

# ── G12 : Tableau SHAP params ─────────────────────────────────
ax = axes[2, 3]
ax.set_facecolor('white')
ax.axis('off')
recap_shap = [
    ['N test total',        f'{len(X_test):,}'],
    ['N features',          f'{N_FEATURES}'],
    ['n_shap (auto)',       f'{shap_params["n_shap"]:,}'],
    ['n_background (auto)', f'{shap_params["n_background"]}'],
    ['n_top_features',      f'{n_top}'],
    ['Stratifié',           str(shap_params['stratified'])],
    ['─────────', '─────────'],
    ['Top SHAP Critique 1', f'{shap_df.iloc[0]["Feature"]}'],
    ['Top SHAP Critique 2', f'{shap_df.iloc[1]["Feature"]}'],
    ['Top SHAP Critique 3', f'{shap_df.iloc[2]["Feature"]}'],
    ['Top SHAP Critique 4', f'{shap_df.iloc[3]["Feature"]}'],
    ['Top SHAP Critique 5', f'{shap_df.iloc[4]["Feature"]}'],
]
tbl2 = ax.table(cellText=recap_shap,
    colLabels=['Paramètre SHAP', 'Valeur'],
    cellLoc='center', loc='center')
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(8.5)
tbl2.scale(1, 1.3)
for (r, c), cell in tbl2.get_celld().items():
    if r == 0:
        cell.set_facecolor('#0F6E56')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#E8F5F0')
    cell.set_edgecolor('#CCCCCC')
ax.set_title('Paramètres SHAP Automatisés', fontweight='bold')

plt.tight_layout()
plt.savefig('lightgbm_final_complet.png', dpi=150,
            bbox_inches='tight', facecolor='#F0F2F5')
plt.close()
print("  ✅ lightgbm_final_complet.png")

# ══════════════════════════════════════════════════════════════
# [8/8] SAUVEGARDE
# ══════════════════════════════════════════════════════════════
print(f"\n[8/8] Sauvegarde...  [{elapsed()}]")

joblib.dump(lgbm_final, 'model_lightgbm_final.pkl')
lgbm_final.booster_.save_model('lightgbm_congestion_final.txt')

df_preds = X_test.copy() if hasattr(X_test, 'copy') else pd.DataFrame(X_test, columns=FEATURES)
df_preds['y_true']  = y_test_np
df_preds['y_pred']  = y_pred
for i in range(N_CLASSES):
    df_preds[f'proba_{i}'] = y_proba[:, i]
df_preds['erreur'] = (y_test_np != y_pred).astype(int)
df_preds.to_csv('predictions_lightgbm_final.csv', index=False)

shap_df.to_csv('shap_importance_par_classe.csv', index=False)

# Vérification PKL
model_charge = joblib.load('model_lightgbm_final.pkl')
f1_verif     = f1_score(y_test_np, model_charge.predict(X_test), average='macro')
assert abs(f1_verif - f1_test) < 1e-6, "❌ PKL corrompu !"

print(f"""
  Fichiers produits :
    model_lightgbm_final.pkl           ← PKL principal ✅
    lightgbm_congestion_final.txt      ← Format natif LightGBM ✅
    predictions_lightgbm_final.csv     ← Prédictions test ✅
    shap_importance_par_classe.csv     ← Importance SHAP ✅
    lightgbm_final_complet.png         ← 12 graphiques ✅
    shap_summary_critique.png          ← Beeswarm SHAP ✅
    shap_bar_par_classe.png            ← Bar SHAP 3 classes ✅
    shap_dependence_prb.png            ← Dependence PRB ✅
    shap_waterfall_critique.png        ← Waterfall exemple ✅
    shap_heatmap_critique.png          ← Heatmap SHAP ✅
  ✅ PKL valide — résultats identiques avant/après sauvegarde
""")

TEMPS_TOTAL = time.time() - TEMPS_DEBUT
print("=" * 65)
print("  RÉSUMÉ FINAL")
print("=" * 65)
print(f"  Accuracy     : {acc_test*100:.3f}%")
print(f"  F1-macro     : {f1_test*100:.3f}%")
print(f"  F1 Critique  : {f1_cls[2]*100:.3f}%")
print(f"  ROC-AUC      : {roc_auc*100:.3f}%")
print(f"  Diff Acc T/T : {gap_acc*100:.4f}%")
print(f"  CV F1        : {cv_mean*100:.3f}% ± {cv_std*100:.3f}%")
print(f"  LC Gap       : {lc_gap_final*100:.4f}%")
print(f"  Stab. σ      : {f1_sub_std*100:.4f}%")
print(f"\n  Tests overfitting :")
print(f"    TEST 1 : {t1_status}")
print(f"    TEST 2 : {t2_status}")
print(f"    TEST 3 : {t3_status}")
print(f"    TEST 4 : {t4_status}")
print(f"\n  Temps total  : {fmt(TEMPS_TOTAL)}")
print(f"  Fin          : {time.strftime('%H:%M:%S')}")
print("=" * 65)
