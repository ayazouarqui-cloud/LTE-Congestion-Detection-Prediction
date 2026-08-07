import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING) 
 # Optuna silencieux

import sys
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 65)
print("  RANDOM FOREST + SHAP  |  PFE M2 Big Data — Djezzy")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# [1/7] Chargement
# ─────────────────────────────────────────────────────────────
print("\n[1/7] Chargement...")
t0 = time.time()
df = pd.read_csv("df_avec_score_kmeans.csv")
print(f"    {len(df):,} lignes | {df.shape[1]} colonnes | {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────
# [2/7] Préparation
# ─────────────────────────────────────────────────────────────
print("\n[2/7] Préparation...")

# FIX 1 : Ajout de DL_PRB_Usage_Rate et PRB_per_User qui existent
# dans le dataset mais étaient absents de la liste des features
FEATURES = [
    'LTE_Setup_Success_Rate','Cell_Traffic_Volume_DL', 
    'Cell_Traffic_Volume_Ul','DL_Average_Throughput', 
    'Ul_Average_Throughput','Avg_User_NB', 'Avaibility',
    'HOUR', 'Spectral_Eff', 'Rolling_PRB_3h','PRB_Z_Score', 
    'Gradient_PRB','DL_PRB_Usage_Rate','PRB_per_User'
]

TARGET      = 'Classe_Congestion'
CLASS_NAMES = ['Normal', 'Modere', 'Critique']

# FIX 2 : Vérifier que toutes les features existent bien dans le df
missing_cols = [f for f in FEATURES if f not in df.columns]
if missing_cols:
    print(f"    ⚠ Colonnes absentes du dataset : {missing_cols}")
    FEATURES = [f for f in FEATURES if f in df.columns]
print(f"    Features utilisées : {len(FEATURES)}")

le = LabelEncoder()
df[TARGET] = le.fit_transform(df[TARGET])
print(f"    Classes : {list(le.classes_)}")


X = df[FEATURES]
y = df[TARGET]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print(f"    Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}")

# ─────────────────────────────────────────────────────────────
# [3/7] Optimisation automatique des paramètres RF via Optuna
# ─────────────────────────────────────────────────────────────
print("\n[3/7] Optimisation automatique des hyperparamètres (Optuna)...")

# ── Paramètres de la recherche (modifiables) ─────────────────
N_TRIALS   = 50    # nombre d'essais Optuna (augmenter pour + de précision)
N_CV_FOLDS = 3     # folds de cross-validation interne
TIMEOUT    = 600   # limite en secondes (None = pas de limite)
METRIC     = 'f1_macro'   # métrique optimisée : 'f1_macro' | 'accuracy' | 'roc_auc_ovr'

def objective(trial):
    """Fonction objectif Optuna : espace de recherche complet RF."""
    p = {
        'n_estimators'     : trial.suggest_int  ('n_estimators',      50,   600),
        'max_depth'        : trial.suggest_int  ('max_depth',          5,    40),
        'min_samples_split': trial.suggest_int  ('min_samples_split',  2,    20),
        'min_samples_leaf' : trial.suggest_int  ('min_samples_leaf',   1,    20),
        'max_features'     : trial.suggest_float('max_features',       0.2,  1.0),
        'max_samples'      : trial.suggest_float('max_samples',        0.5,  1.0),
        'criterion'        : trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'class_weight'     : 'balanced',
        'random_state'     : 42,
        'n_jobs'           : -1,
        'oob_score'        : False,   # désactivé en CV pour la vitesse
    }
    clf = RandomForestClassifier(**p)

    # Cross-validation stratifiée sur le train set
    cv  = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)

    if METRIC == 'roc_auc_ovr':
        scores = cross_val_score(clf, X_train, y_train,
                                 cv=cv, scoring='roc_auc_ovr_weighted', n_jobs=1)
    else:
        scores = cross_val_score(clf, X_train, y_train,
                                 cv=cv, scoring=METRIC, n_jobs=1)
    return scores.mean()

t1 = time.time()
study = optuna.create_study(
    direction  = 'maximize',
    sampler    = TPESampler(seed=42),
    study_name = 'RF_Djezzy_Congestion'
)
study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT,
               show_progress_bar=False)

best_params = study.best_params
best_score  = study.best_value

print(f"    Optuna terminé en {time.time()-t1:.1f}s")
print(f"    Meilleur {METRIC} CV : {best_score:.4f}")
print(f"    Meilleurs paramètres :")
for k, v in best_params.items():
    print(f"        {k:22s} = {v}")

# ── Entraînement final avec les meilleurs paramètres ─────────
best_params.update({
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs'      : -1,
    'oob_score'   : True,
    'verbose'     : 0,
})

t2 = time.time()
model = RandomForestClassifier(**best_params)
model.fit(X_train, y_train)
print(f"\n    Entraînement final terminé en {time.time()-t2:.1f}s")
print(f"    OOB Score : {model.oob_score_:.4f}")

# ─────────────────────────────────────────────────────────────
# [4/7] Évaluation
# ─────────────────────────────────────────────────────────────
print("\n[4/7] Évaluation...")

y_pred       = model.predict(X_test)
y_pred_prob  = model.predict_proba(X_test)
y_train_pred = model.predict(X_train)

acc_test  = accuracy_score(y_test,  y_pred)
acc_train = accuracy_score(y_train, y_train_pred)
f1_test   = f1_score(y_test,  y_pred,       average='macro')
f1_train  = f1_score(y_train, y_train_pred, average='macro')
roc_auc   = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')

print(f"""
    ┌──────────────────────────────────────────┐
    │  RANDOM FOREST — Résultats               │
    │  OOB Score      : {model.oob_score_:.4f}            │
    │  Train Accuracy : {acc_train:.4f}            │
    │  Test  Accuracy : {acc_test:.4f}            │
    │  Diff           : {abs(acc_train-acc_test):.4f}            │
    │  Train F1-macro : {f1_train:.4f}            │
    │  Test  F1-macro : {f1_test:.4f}            │
    │  ROC-AUC        : {roc_auc:.4f}            │
    └──────────────────────────────────────────┘
""")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# ─────────────────────────────────────────────────────────────
# [5/7] SHAP — taille d'échantillon automatique
# ─────────────────────────────────────────────────────────────
print("\n[5/7] Calcul SHAP (TreeExplainer)...")

# Taille SHAP automatique : 20% du test set, min 500, max 5000
n_shap   = int(np.clip(len(X_test) * 0.20, 500, 5000))
shap_idx = np.random.choice(len(X_test), n_shap, replace=False)
X_shap   = X_test.iloc[shap_idx]
print(f"    Échantillon SHAP : {n_shap} points (20% de {len(X_test):,})")

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)

# FIX 3 : SHAP >= 0.40 peut retourner un array 3D au lieu d'une liste
# On normalise pour toujours avoir une liste de 3 arrays 2D
if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    # shape (n_samples, n_features, n_classes) → liste de n_classes arrays
    shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
    # Cas binaire inattendu — on duplique pour éviter les crashs
    shap_values = [shap_values, -shap_values]

print(f"    SHAP calculées ✓  shape par classe : {shap_values[0].shape}")

# ─────────────────────────────────────────────────────────────
# [6/7] Graphiques
# ─────────────────────────────────────────────────────────────
print("\n[6/7] Génération des graphiques...")

colors      = ['#2ecc71', '#f39c12', '#e74c3c']
class_names = CLASS_NAMES

# ── Figure 1 : Performance ───────────────────────────────────
fig1, axes = plt.subplots(2, 3, figsize=(20, 12))
fig1.suptitle('Random Forest + SHAP — Performance\nDjezzy BTS Congestion Detection',
              fontsize=14, fontweight='bold')

# 1. Matrice confusion
cm     = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(cm_pct, annot=True, fmt='.2f', ax=axes[0, 0],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap='Greens')
axes[0, 0].set_title('Matrice de Confusion (%)')
axes[0, 0].set_ylabel('Réel')
axes[0, 0].set_xlabel('Prédit')

# 2. Feature Importance Gini
feat_imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
axes[0, 1].barh(feat_imp.index[:12], feat_imp.values[:12], color='#27ae60')
axes[0, 1].set_title(f'Feature Importance Gini (Top 12)\nOOB Score = {model.oob_score_:.4f}')
axes[0, 1].invert_yaxis()

# 3. Courbe de convergence Optuna (automatisée)
trial_values = [t.value for t in study.trials if t.value is not None]
best_so_far  = np.maximum.accumulate(trial_values)
axes[0, 2].plot(range(1, len(trial_values)+1), trial_values,
                alpha=0.4, color='steelblue', label='Score par trial')
axes[0, 2].plot(range(1, len(best_so_far)+1), best_so_far,
                color='#e74c3c', linewidth=2, label=f'Meilleur ({best_score:.4f})')
axes[0, 2].set_title(f'Convergence Optuna ({N_TRIALS} trials)\nMétrique : {METRIC}')
axes[0, 2].set_xlabel('Trial')
axes[0, 2].set_ylabel(METRIC)
axes[0, 2].legend()

# 4. Train vs Test
x = np.arange(2)
w = 0.35
axes[1, 0].bar(x - w/2, [acc_train, f1_train], w, label='Train', color='#27ae60')
axes[1, 0].bar(x + w/2, [acc_test,  f1_test],  w, label='Test',  color='orange')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(['Accuracy', 'F1-macro'])
axes[1, 0].set_ylim([0.995, 1.001])
axes[1, 0].set_title('Train vs Test')
axes[1, 0].legend()

# 5. F1 par classe
f1_scores = f1_score(y_test, y_pred, average=None)
axes[1, 1].bar(CLASS_NAMES, f1_scores, color=colors)
axes[1, 1].set_title('F1-score par Classe')
axes[1, 1].set_ylim([0.99, 1.001])
axes[1, 1].axhline(f1_test, color='navy', linestyle='--',
                   label=f'F1-macro={f1_test:.4f}')
axes[1, 1].legend()

# 6. Récapitulatif + meilleurs paramètres Optuna
axes[1, 2].axis('off')
recap_data = [
    ['OOB Score',       f'{model.oob_score_:.4f}'],
    ['Accuracy',        f'{acc_test:.4f}'],
    ['F1-macro',        f'{f1_test:.4f}'],
    ['ROC-AUC',         f'{roc_auc:.4f}'],
    ['Diff Tr/Test',    f'{abs(acc_train - acc_test):.4f}'],
    ['Erreurs',         f'{sum(y_pred != y_test):,}'],
    ['── Optuna ──',    '──────────'],
    ['Trials',          f'{N_TRIALS}'],
    [f'Best {METRIC}',  f'{best_score:.4f}'],
    ['n_estimators',    f'{best_params["n_estimators"]}'],
    ['max_depth',       f'{best_params["max_depth"]}'],
    ['max_features',    f'{best_params["max_features"]:.3f}'],
    ['max_samples',     f'{best_params["max_samples"]:.4f}'],
    ['criterion',       f'{best_params["criterion"]}'],
]
tbl = axes[1, 2].table(cellText=recap_data,
                        colLabels=['Paramètre / Métrique', 'Valeur'],
                        loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 1.6)
axes[1, 2].set_title('Récapitulatif + Optuna', fontweight='bold')

plt.tight_layout()
plt.savefig('rf_shap_fig1.png', dpi=150, bbox_inches='tight')
print("    rf_shap_fig1.png ✓")
plt.close()

# ── Figure 1b : ROC (séparée pour lisibilité) ─────────────────
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
for i, (cls, col) in enumerate(zip(CLASS_NAMES, colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    ax_roc.plot(fpr, tpr, color=col, label=f'{cls} (AUC={auc(fpr, tpr):.4f})')
ax_roc.plot([0, 1], [0, 1], 'k--')
ax_roc.set_title('Courbes ROC par Classe')
ax_roc.set_xlabel('Taux Faux Positifs')
ax_roc.set_ylabel('Taux Vrais Positifs')
ax_roc.legend()
plt.tight_layout()
plt.savefig('rf_shap_roc.png', dpi=150, bbox_inches='tight')
print("    rf_shap_roc.png ✓")
plt.close()

# ── Figure 2 : SHAP ──────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 3, figsize=(22, 14))
fig2.suptitle('Random Forest — Analyse SHAP\nInterprétabilité des Décisions',
              fontsize=14, fontweight='bold')

# SHAP par classe
for i, (cls, col) in enumerate(zip(class_names, colors)):
    shap_mean = np.abs(shap_values[i]).mean(axis=0)
    shap_df   = pd.Series(shap_mean, index=FEATURES).sort_values(ascending=False)
    axes2[0, i].barh(shap_df.index[:10], shap_df.values[:10], color=col)
    axes2[0, i].set_title(f'SHAP — Classe {cls}', fontweight='bold')
    axes2[0, i].set_xlabel('|SHAP| moyen')
    axes2[0, i].invert_yaxis()

# FIX 4 : SHAP cellule individuelle Critique
# y_test est un Series avec l'index original de X_test (après split)
# On cherche les positions (iloc) dans y_test où la classe == 2
critique_positions = np.where(y_test.values == 2)[0]  # positions dans y_test

if len(critique_positions) > 0:
    # On prend la première position critique dans y_test
    first_critique_pos = critique_positions[0]
    # On cherche si cette position est dans shap_idx
    local_idx = np.where(shap_idx == first_critique_pos)[0]

    if len(local_idx) > 0:
        shap_cell = shap_values[2][local_idx[0]]
        sorted_i  = np.argsort(np.abs(shap_cell))[-10:]
        axes2[1, 0].barh(
            [FEATURES[j] for j in sorted_i],
            shap_cell[sorted_i],
            color=['#e74c3c' if v > 0 else '#2ecc71' for v in shap_cell[sorted_i]]
        )
        axes2[1, 0].axvline(0, color='black', linewidth=0.8)
        axes2[1, 0].set_title('SHAP — 1 Cellule Critique')
        axes2[1, 0].set_xlabel('Valeur SHAP')
    else:
        # La cellule critique n'est pas dans shap_idx → prendre directement
        # un sample Critique parmi ceux dans shap_idx
        shap_critique_positions = [
            j for j, idx in enumerate(shap_idx) if y_test.values[idx] == 2
        ]
        if shap_critique_positions:
            shap_cell = shap_values[2][shap_critique_positions[0]]
            sorted_i  = np.argsort(np.abs(shap_cell))[-10:]
            axes2[1, 0].barh(
                [FEATURES[j] for j in sorted_i],
                shap_cell[sorted_i],
                color=['#e74c3c' if v > 0 else '#2ecc71' for v in shap_cell[sorted_i]]
            )
            axes2[1, 0].axvline(0, color='black', linewidth=0.8)
            axes2[1, 0].set_title('SHAP — 1 Cellule Critique')
            axes2[1, 0].set_xlabel('Valeur SHAP')
        else:
            axes2[1, 0].set_title('SHAP — Pas de cellule Critique dans l\'échantillon')
            axes2[1, 0].axis('off')
else:
    axes2[1, 0].set_title('SHAP — Aucune cellule Critique dans le test set')
    axes2[1, 0].axis('off')

# Distribution P(Critique)
prob_critique = y_pred_prob[:, 2]
for i, (cls, col) in enumerate(zip(class_names, colors)):
    mask = y_test.values == i
    axes2[1, 1].hist(prob_critique[mask], bins=50, alpha=0.6,
                     color=col, label=cls, density=True)
axes2[1, 1].axvline(0.5, color='black', linestyle='--', label='Seuil=0.5')
axes2[1, 1].set_title('Distribution P(Critique)')
axes2[1, 1].set_xlabel('P(Critique)')
axes2[1, 1].set_yscale('log')
axes2[1, 1].legend()

# SHAP global (moyenne sur toutes les classes)
shap_global    = np.mean([np.abs(shap_values[i]).mean(axis=0) for i in range(3)], axis=0)
shap_df_global = pd.Series(shap_global, index=FEATURES).sort_values(ascending=False)
axes2[1, 2].barh(shap_df_global.index[:12], shap_df_global.values[:12], color='purple')
axes2[1, 2].set_title('SHAP Global — Toutes Classes')
axes2[1, 2].set_xlabel('|SHAP| moyen global')
axes2[1, 2].invert_yaxis()

plt.tight_layout()
plt.savefig('rf_shap_fig2.png', dpi=150, bbox_inches='tight')
print("    rf_shap_fig2.png ✓")
plt.close()

# ─────────────────────────────────────────────────────────────
# [7/7] Sauvegarde
# ─────────────────────────────────────────────────────────────
print("\n[7/7] Sauvegarde...")
joblib.dump(model, 'model_rf_shap.pkl')
joblib.dump(le,    'label_encoder_rf.pkl')
joblib.dump(study, 'optuna_study_rf.pkl')   # ← étude Optuna sauvegardée
feat_imp.to_csv('feature_importance_rf_shap.csv')

# Paramètres optimaux sauvegardés
pd.DataFrame([{'Parametre': k, 'Valeur': v} for k, v in best_params.items()]) \
  .to_csv('best_params_optuna_rf.csv', index=False)

pd.DataFrame({
    'Metrique': ['OOB', 'Train_Acc', 'Test_Acc', 'Diff', 'Train_F1', 'Test_F1', 'ROC_AUC',
                 f'Optuna_best_{METRIC}'],
    'Valeur':   [model.oob_score_, acc_train, acc_test,
                 abs(acc_train - acc_test), f1_train, f1_test, roc_auc, best_score]
}).to_csv('resultats_rf_shap.csv', index=False)

print(f"""
  Fichiers produits :
    model_rf_shap.pkl               ✓
    label_encoder_rf.pkl            ✓
    optuna_study_rf.pkl             ✓  (réutilisable / warm start)
    best_params_optuna_rf.csv       ✓
    feature_importance_rf_shap.csv  ✓
    resultats_rf_shap.csv           ✓
    rf_shap_fig1.png                ✓
    rf_shap_roc.png                 ✓
    rf_shap_fig2.png                ✓
  Temps total : {time.time()-t0:.1f}s
""")
print("=" * 65)
