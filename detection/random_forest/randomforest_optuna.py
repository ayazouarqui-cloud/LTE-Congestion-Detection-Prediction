"""
================================================================
PFE M2 Big Data — Classification Congestion BTS Radio LTE
ALGORITHME : Random Forest + Optuna (Bayesian Hyperparameter Optimization)
Stratégie : Optuna sur 200K lignes → entraînement final sur 100%
Objectif  : Maximiser F1-macro
================================================================
"""

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_val_score, learning_curve
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ── Palette ───────────────────────────────────
COLORS_CLS  = {0: "#27AE60", 1: "#F39C12", 2: "#E74C3C"}
CLASS_NAMES = ["Normal", "Modéré", "Critique"]
GREEN = "#27AE60"; DARK = "#1F3864"; BG = "#F7F9FC"

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25,
    'figure.facecolor': BG, 'axes.facecolor': 'white',
})

print("=" * 65)
print("  RANDOM FOREST + OPTUNA  |  PFE M2 Big Data")
print("  Optimisation Bayésienne des Hyperparamètres")
print("=" * 65)

# ══════════════════════════════════════════════
# 1. CHARGEMENT
# ══════════════════════════════════════════════
print("\n[1/7] Chargement...")
t0 = time.time()
df = pd.read_csv("df_avec_score_kmeans.csv")
print(f"    {df.shape[0]:,} lignes | {df.shape[1]} colonnes | {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════
# 2. FEATURES & SPLIT GLOBAL
# ══════════════════════════════════════════════
print("\n[2/7] Préparation...")
TARGET  = "Classe_Congestion"
FEATURES = [
    'LTE_Setup_Success_Rate','Cell_Traffic_Volume_DL', 
    'Cell_Traffic_Volume_Ul','DL_Average_Throughput', 
    'Ul_Average_Throughput','Avg_User_NB', 'Avaibility',
    'HOUR', 'Spectral_Eff', 'Rolling_PRB_3h','PRB_Z_Score', 
    'Gradient_PRB','DL_PRB_Usage_Rate','PRB_per_User'
]


X = df[FEATURES].values
y = df[TARGET].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

cw      = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw_dict = dict(enumerate(cw))

print(f"    Train={X_train.shape[0]:,} | Val={X_val.shape[0]:,} | Test={X_test.shape[0]:,}")
print(f"    Poids → Normal={cw[0]:.3f} | Modéré={cw[1]:.3f} | Critique={cw[2]:.3f}")

# ══════════════════════════════════════════════
# 3. OPTUNA — SOUS-ÉCHANTILLON STRATIFIÉ
# ══════════════════════════════════════════════
print("\n[3/7] Optuna — Optimisation Bayésienne (50 trials)...")
print("    Sous-échantillon : 200 000 lignes stratifiées")
print("    Métrique : F1-macro (CV-3)")

N_OPTUNA = 200_000
N_TRIALS = 50

# Sous-échantillon stratifié
idx_opt = []
for cls in np.unique(y_train):
    idx_cls = np.where(y_train == cls)[0]
    n_cls   = int(N_OPTUNA * len(idx_cls) / len(y_train))
    idx_opt.extend(np.random.RandomState(42).choice(idx_cls, n_cls, replace=False))
idx_opt = np.array(idx_opt)

X_opt = X_train_s[idx_opt]
y_opt = y_train[idx_opt]
print(f"    Sous-échantillon Optuna : {len(X_opt):,} lignes")

# ── Fonction objectif Optuna ──────────────────
def objective(trial):
    params = {
        "n_estimators"    : trial.suggest_int("n_estimators", 50, 400),
        "max_depth"       : trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 10, 200),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100),
        "max_features"    : trial.suggest_categorical("max_features",
                                                       ["sqrt", "log2", 0.3, 0.5, 0.7]),
        "max_samples"     : trial.suggest_float("max_samples", 0.5, 1.0),
        "class_weight"    : "balanced",
        "n_jobs"          : -1,
        "random_state"    : 42,
    }

    cv      = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    model_t = RandomForestClassifier(**params)
    scores  = cross_val_score(
        model_t, X_opt, y_opt,
        cv=cv, scoring='f1_macro', n_jobs=1
    )
    return scores.mean()

# Lancement Optuna
t_optuna = time.time()
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
)

def callback_progress(study, trial):
    if trial.number % 10 == 0:
        print(f"    Trial {trial.number:>3}/{N_TRIALS} | "
              f"F1={trial.value:.4f} | "
              f"Meilleur={study.best_value:.4f}")

study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback_progress])

t_optuna_elapsed = time.time() - t_optuna
print(f"\n    Optuna terminé en {t_optuna_elapsed:.1f}s ({t_optuna_elapsed/60:.1f} min)")
print(f"    Meilleur F1-macro (CV) : {study.best_value:.4f}")
print(f"\n    ✅ Meilleurs hyperparamètres trouvés :")
best_params = study.best_params
for k, v in best_params.items():
    print(f"       {k:<25} = {v}")

# ══════════════════════════════════════════════
# 4. ENTRAÎNEMENT FINAL
# ══════════════════════════════════════════════
print("\n[4/7] Entraînement final sur 100% du train...")
t_final = time.time()

model = RandomForestClassifier(
    **best_params,
    class_weight='balanced',
    oob_score=True,
    n_jobs=-1,
    random_state=42,
    verbose=1,
)
model.fit(X_train_s, y_train)

t_final_elapsed = time.time() - t_final
print(f"\n    Terminé en {t_final_elapsed:.1f}s ({t_final_elapsed/60:.1f} min)")
print(f"    OOB Score : {model.oob_score_:.4f}")

# ══════════════════════════════════════════════
# 5. ÉVALUATION
# ══════════════════════════════════════════════
print("\n[5/7] Évaluation...")

y_pred      = model.predict(X_test_s)
y_prob      = model.predict_proba(X_test_s)
y_test_bin  = label_binarize(y_test, classes=[0, 1, 2])

acc          = accuracy_score(y_test, y_pred)
f1           = f1_score(y_test, y_pred, average='macro')
auc_score    = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
f1_per_class = f1_score(y_test, y_pred, average=None)

print(f"\n    ┌─────────────────────────────────────────┐")
print(f"    │  RANDOM FOREST + OPTUNA — Résultats      │")
print(f"    │  Accuracy  : {acc:.4f}                    │")
print(f"    │  F1-macro  : {f1:.4f}                    │")
print(f"    │  ROC-AUC   : {auc_score:.4f}                    │")
print(f"    │  OOB Score : {model.oob_score_:.4f}                    │")
print(f"    └─────────────────────────────────────────┘")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

feat_imp = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

# ══════════════════════════════════════════════
# 6. LEARNING CURVE (sous-échantillon)
# ══════════════════════════════════════════════
print("\n[6/7] Learning Curve...")
n_lc   = min(80_000, X_train_s.shape[0])
idx_lc = np.random.RandomState(42).choice(X_train_s.shape[0], n_lc, replace=False)

train_sizes_abs, train_scores, val_scores = learning_curve(
    RandomForestClassifier(
        **best_params, class_weight='balanced', n_jobs=-1, random_state=42
    ),
    X_train_s[idx_lc], y_train[idx_lc],
    train_sizes=np.linspace(0.1, 1.0, 7),
    cv=3, scoring='f1_macro', n_jobs=-1, verbose=0
)

# ══════════════════════════════════════════════
# GRAPHIQUES
# ══════════════════════════════════════════════
print("    Génération des graphiques...")

# ── FIGURE 1 — Optuna + Résultats globaux ─────
fig1, axes = plt.subplots(2, 3, figsize=(20, 13), facecolor=BG)
fig1.suptitle("Random Forest + Optuna — Optimisation Bayésienne | PFE M2 Big Data",
              fontsize=15, fontweight='bold', color=DARK, y=1.01)

# G1 — Convergence Optuna
ax = axes[0, 0]
trials_f1   = [t.value for t in study.trials]
best_so_far = np.maximum.accumulate(trials_f1)
ax.scatter(range(len(trials_f1)), trials_f1, alpha=0.45,
           color=GREEN, s=25, label='F1 trial')
ax.plot(range(len(best_so_far)), best_so_far, color='#E74C3C',
        lw=2.5, label=f'Meilleur = {study.best_value:.4f}')
ax.axhline(study.best_value, color='gray', linestyle=':', lw=1.5)
ax.set_title("Convergence Optuna (50 trials)", fontweight='bold', fontsize=12)
ax.set_xlabel("Trial #"); ax.set_ylabel("F1-macro (CV-3)")
ax.legend(fontsize=10)

# G2 — Importance des hyperparamètres (Optuna)
ax = axes[0, 1]
try:
    param_importance = optuna.importance.get_param_importances(study)
    params_names = list(param_importance.keys())[:8]
    params_vals  = [param_importance[k] for k in params_names]
    ax.barh(params_names[::-1], params_vals[::-1],
            color=GREEN, alpha=0.85, edgecolor='white')
    ax.set_title("Importance des Hyperparamètres\n(Optuna FanovaImportance)",
                 fontweight='bold', fontsize=12)
    ax.set_xlabel("Importance relative")
except Exception:
    ax.text(0.5, 0.5, "Non disponible\n(besoin de plus de trials)",
            ha='center', va='center', transform=ax.transAxes, fontsize=11)

# G3 — Distribution F1 par trial
ax = axes[0, 2]
ax.hist(trials_f1, bins=20, color=GREEN, alpha=0.75, edgecolor='white')
ax.axvline(study.best_value, color='#E74C3C', lw=2.5,
           label=f'Meilleur = {study.best_value:.4f}')
ax.axvline(np.median(trials_f1), color='#F39C12', lw=2, linestyle='--',
           label=f'Médiane = {np.median(trials_f1):.4f}')
ax.set_title("Distribution des F1 — 50 Trials", fontweight='bold', fontsize=12)
ax.set_xlabel("F1-macro (CV-3)"); ax.set_ylabel("Fréquence")
ax.legend(fontsize=10)

# G4 — Matrice de confusion
ax = axes[1, 0]
cm     = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt='.2f', cmap='Greens',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, cbar_kws={'label': '%'}, annot_kws={"size": 13, "weight": "bold"})
ax.set_title("Matrice de Confusion (%)", fontweight='bold', fontsize=12)
ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")

# G5 — Courbes ROC
ax = axes[1, 1]
for cls in [0, 1, 2]:
    fpr, tpr, _ = roc_curve(y_test_bin[:, cls], y_prob[:, cls])
    ax.plot(fpr, tpr, color=list(COLORS_CLS.values())[cls], lw=2.5,
            label=f"{CLASS_NAMES[cls]} (AUC={auc(fpr,tpr):.4f})")
ax.plot([0,1],[0,1],'--', color='gray', lw=1.5, alpha=0.6)
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.set_title("Courbes ROC — One vs Rest", fontweight='bold', fontsize=12)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.legend(fontsize=10, loc='lower right')

# G6 — Learning Curve
ax = axes[1, 2]
train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)
ax.plot(train_sizes_abs, train_mean, 'o-', color=GREEN, lw=2.5, ms=7, label='Train F1')
ax.fill_between(train_sizes_abs, train_mean-train_std, train_mean+train_std,
                alpha=0.15, color=GREEN)
ax.plot(train_sizes_abs, val_mean, 's--', color='#E74C3C', lw=2.5, ms=7, label='Val F1 (CV-3)')
ax.fill_between(train_sizes_abs, val_mean-val_std, val_mean+val_std,
                alpha=0.15, color='#E74C3C')
ax.set_title("Learning Curve — F1-macro", fontweight='bold', fontsize=12)
ax.set_xlabel("Taille train"); ax.set_ylabel("F1-macro")
ax.set_ylim(0.90, 1.01); ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("rf_optuna_fig1.png", dpi=150, bbox_inches='tight', facecolor=BG)
print("    rf_optuna_fig1.png ✓")

# ── FIGURE 2 — Erreurs & Probabilités ─────────
fig2, axes2 = plt.subplots(2, 3, figsize=(20, 13), facecolor=BG)
fig2.suptitle("Random Forest + Optuna — Erreurs & Distributions de Probabilités",
              fontsize=15, fontweight='bold', color=DARK, y=1.01)

# G7 — Distribution probabilités
ax = axes2[0, 0]
for cls in [0, 1, 2]:
    mask = y_test == cls
    ax.hist(y_prob[mask, cls], bins=60, alpha=0.65,
            color=COLORS_CLS[cls], label=CLASS_NAMES[cls],
            edgecolor='white', linewidth=0.4)
ax.axvline(0.5, color='gray', linestyle='--', lw=1.5)
ax.set_title("Distribution P(classe|réel)", fontweight='bold', fontsize=12)
ax.set_xlabel("Probabilité prédite"); ax.set_ylabel("Observations")
ax.legend(fontsize=10)

# G8 — Erreurs hors diagonale
ax = axes2[0, 1]
cm_err = confusion_matrix(y_test, y_pred).copy()
np.fill_diagonal(cm_err, 0)
sns.heatmap(cm_err, annot=True, fmt='d', cmap='Oranges',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, cbar_kws={'label': 'Erreurs'}, annot_kws={"size": 13})
ax.set_title("Carte des Erreurs (diag=0)", fontweight='bold', fontsize=12)
ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")

# G9 — Confiance max correct vs erreur
ax = axes2[0, 2]
max_prob = y_prob.max(axis=1)
correct  = (y_pred == y_test)
ax.hist(max_prob[correct],  bins=60, alpha=0.7, color=GREEN,
        label=f'Correct ({correct.sum():,})', edgecolor='white')
ax.hist(max_prob[~correct], bins=60, alpha=0.7, color='#E74C3C',
        label=f'Erreur ({(~correct).sum():,})', edgecolor='white')
ax.set_title("Confiance Max — Correct vs Erreur", fontweight='bold', fontsize=12)
ax.set_xlabel("Probabilité max prédite"); ax.set_ylabel("Fréquence")
ax.legend(fontsize=10)

# G10 — Calibration
ax = axes2[1, 0]
for cls in [0, 1, 2]:
    frac_pos, mean_pred = calibration_curve(
        (y_test == cls).astype(int), y_prob[:, cls], n_bins=15)
    ax.plot(mean_pred, frac_pos, 's-', color=COLORS_CLS[cls],
            lw=2, ms=5, label=CLASS_NAMES[cls])
ax.plot([0,1],[0,1],'--', color='gray', lw=1.5, label='Parfait')
ax.set_title("Courbes de Calibration", fontweight='bold', fontsize=12)
ax.set_xlabel("Probabilité prédite"); ax.set_ylabel("Fraction réels")
ax.legend(fontsize=10)

# G11 — Feature importance top 12
ax = axes2[1, 1]
top12 = feat_imp.head(12)
ax.barh(top12['Feature'][::-1], top12['Importance'][::-1],
        color=GREEN, alpha=0.85, edgecolor='white')
ax.set_title("Feature Importance Gini (Top 12)", fontweight='bold', fontsize=12)
ax.set_xlabel("Importance")
for i, val in enumerate(top12['Importance'][::-1]):
    ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=8)

# G12 — Tableau hyperparamètres optimaux
ax = axes2[1, 2]
ax.axis('off')
param_rows = [[k, f"{v:.4f}" if isinstance(v, float) else str(v)]
              for k, v in best_params.items()]
tbl = ax.table(
    cellText=param_rows,
    colLabels=["Hyperparamètre", "Valeur optimale"],
    loc='center', cellLoc='center'
)
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
tbl.scale(1.3, 1.9)
for j in range(2):
    tbl[0, j].set_facecolor(DARK)
    tbl[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(param_rows)+1):
    for j in range(2):
        tbl[i, j].set_facecolor('#EAFAF1' if i % 2 == 0 else 'white')
ax.set_title("Hyperparamètres Optimaux — Optuna",
             fontweight='bold', fontsize=12, pad=20)

plt.tight_layout()
plt.savefig("rf_optuna_fig2.png", dpi=150, bbox_inches='tight', facecolor=BG)
print("    rf_optuna_fig2.png ✓")

# ══════════════════════════════════════════════
# 7. SAUVEGARDE
# ══════════════════════════════════════════════
print("\n[7/7] Sauvegarde...")

joblib.dump(model,  "model_rf_optuna.pkl")
joblib.dump(scaler, "scaler_rf_optuna.pkl")
feat_imp.to_csv("feature_importance_rf_optuna.csv", index=False)
pd.DataFrame([best_params]).to_csv("best_params_rf_optuna.csv", index=False)

pd.DataFrame({
    'Modèle': ['RandomForest+Optuna'],
    'Accuracy': [acc], 'F1_macro': [f1], 'ROC_AUC': [auc_score],
    'F1_Normal': [f1_per_class[0]], 'F1_Modere': [f1_per_class[1]],
    'F1_Critique': [f1_per_class[2]],
    'OOB_Score': [model.oob_score_],
    'Optuna_best_cv_f1': [study.best_value],
    'N_trials': [N_TRIALS],
    'Temps_optuna_s': [t_optuna_elapsed],
    'Temps_final_s': [t_final_elapsed],
}).to_csv("resultats_rf_optuna.csv", index=False)

total = t_optuna_elapsed + t_final_elapsed
print("\n  Fichiers produits :")
print("    model_rf_optuna.pkl              ✓")
print("    scaler_rf_optuna.pkl             ✓")
print("    best_params_rf_optuna.csv        ✓")
print("    feature_importance_rf_optuna.csv ✓")
print("    resultats_rf_optuna.csv          ✓")
print("    rf_optuna_fig1.png               ✓  (6 graphiques)")
print("    rf_optuna_fig2.png               ✓  (6 graphiques)")
print(f"\n  Temps total : {total:.1f}s ({total/60:.1f} min)")
print("=" * 65)
