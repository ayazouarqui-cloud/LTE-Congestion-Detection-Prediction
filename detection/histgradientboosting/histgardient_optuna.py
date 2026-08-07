import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, classification_report,
                             confusion_matrix)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import warnings
warnings.filterwarnings('ignore')
import sys
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 65)
print("  HISTGRADIENTBOOSTING + OPTUNA  |  PFE M2 Big Data")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# [1/8] Chargement
# ─────────────────────────────────────────────────────────────
print("\n[1/8] Chargement des données...")
t0 = time.time()
df = pd.read_csv("df_avec_score_kmeans.csv")
print(f"    {len(df):,} lignes | {df.shape[1]} colonnes | {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────
# [2/8] Préparation
# ─────────────────────────────────────────────────────────────
print("\n[2/8] Préparation...")

FEATURES = [
    'LTE_Setup_Success_Rate','Cell_Traffic_Volume_DL', 
    'Cell_Traffic_Volume_Ul','DL_Average_Throughput', 
    'Ul_Average_Throughput','Avg_User_NB', 'Avaibility',
    'HOUR', 'Spectral_Eff', 'Rolling_PRB_3h','PRB_Z_Score', 
    'Gradient_PRB','DL_PRB_Usage_Rate','PRB_per_User'
]

TARGET = 'Classe_Congestion'

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

print(f"    Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}")

# ─────────────────────────────────────────────────────────────
# [3/8] Optuna — Optimisation Bayésienne
# ─────────────────────────────────────────────────────────────
print("\n[3/8] Optuna — Optimisation Bayésienne (50 trials)...")

# Sous-échantillon pour Optuna
sample_idx = np.random.choice(len(X_train), 500_000, replace=False)
X_opt = X_train.iloc[sample_idx]
y_opt = y_train.iloc[sample_idx]

def objective_hgb(trial):
    params = {
        'max_iter'           : trial.suggest_int('max_iter', 200, 1000),
        'max_depth'          : trial.suggest_int('max_depth', 3, 15),
        'learning_rate'      : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_samples_leaf'   : trial.suggest_int('min_samples_leaf', 10, 100),
        'l2_regularization'  : trial.suggest_float('l2_regularization', 1e-4, 10.0, log=True),
        'max_leaf_nodes'     : trial.suggest_int('max_leaf_nodes', 15, 255),
        'max_bins'           : trial.suggest_int('max_bins', 64, 255),
        'random_state'       : 42,
        'early_stopping'     : True,
        'n_iter_no_change'   : 20,
        'validation_fraction': 0.1,
        'class_weight'       : 'balanced'
    }

    model = HistGradientBoostingClassifier(**params)
    scores = cross_val_score(
        model, X_opt, y_opt,
        cv=3, scoring='f1_macro', n_jobs=-1
    )
    return scores.mean()

t1 = time.time()
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_hgb = optuna.create_study(direction='maximize')
study_hgb.optimize(objective_hgb, n_trials=50, show_progress_bar=True)

print(f"\n    Optuna terminé en {time.time()-t1:.1f}s")
print(f"    Meilleur F1-macro (CV) : {study_hgb.best_value:.4f}")
print(f"\n    Meilleurs hyperparamètres :")
for k, v in study_hgb.best_params.items():
    print(f"       {k:<25} = {v}")

# ─────────────────────────────────────────────────────────────
# [4/8] Entraînement final
# ─────────────────────────────────────────────────────────────
print("\n[4/8] Entraînement final sur 100% du train...")
t2 = time.time()

best_params_hgb = study_hgb.best_params
best_params_hgb['random_state']        = 42
best_params_hgb['early_stopping']      = True
best_params_hgb['n_iter_no_change']    = 20
best_params_hgb['validation_fraction'] = 0.1
best_params_hgb['class_weight']        = 'balanced'

model_hgb = HistGradientBoostingClassifier(**best_params_hgb)
model_hgb.fit(X_train, y_train)
print(f"    Terminé en {time.time()-t2:.1f}s")

# ─────────────────────────────────────────────────────────────
# [5/8] Évaluation
# ─────────────────────────────────────────────────────────────
print("\n[5/8] Évaluation...")

y_pred       = model_hgb.predict(X_test)
y_pred_prob  = model_hgb.predict_proba(X_test)
y_train_pred = model_hgb.predict(X_train)

acc_test  = accuracy_score(y_test,  y_pred)
acc_train = accuracy_score(y_train, y_train_pred)
f1_test   = f1_score(y_test,  y_pred,       average='macro')
f1_train  = f1_score(y_train, y_train_pred, average='macro')
roc_auc   = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')

print(f"""
    ┌─────────────────────────────────────────────┐
    │  HISTGRADIENT + OPTUNA — Résultats finaux   │
    │  Train Accuracy : {acc_train:.4f}           │
    │  Test  Accuracy : {acc_test:.4f}            │
    │  Différence     : {abs(acc_train-acc_test):.4f} │
    │  Train F1-macro : {f1_train:.4f}            │
    │  Test  F1-macro : {f1_test:.4f}             │
    │  ROC-AUC        : {roc_auc:.4f}             │
    └─────────────────────────────────────────────┘
""")

target_names_str = [str(c) for c in le.classes_]
print(classification_report(y_test, y_pred, target_names=target_names_str))

# ─────────────────────────────────────────────────────────────
# [6/8] Graphiques
# ─────────────────────────────────────────────────────────────
print("\n[6/8] Génération des graphiques...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('HistGradientBoosting + Optuna — Résultats',
             fontsize=16, fontweight='bold')

colors = ['green', 'orange', 'red']

# 1. Matrice de confusion
cm     = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(cm_pct, annot=True, fmt='.2f', ax=axes[0,0],
            xticklabels=le.classes_, yticklabels=le.classes_,
            cmap='Blues')
axes[0,0].set_title('Matrice de Confusion (%)')
axes[0,0].set_ylabel('Réel')
axes[0,0].set_xlabel('Prédit')

# 2. F1-score par classe
f1_scores = f1_score(y_test, y_pred, average=None)
axes[0,1].bar(le.classes_, f1_scores,
              color=colors[:len(le.classes_)])
axes[0,1].set_title('F1-score par Classe')
axes[0,1].set_ylim([0.99, 1.001])
axes[0,1].axhline(y=f1_test, color='navy',
                  linestyle='--', label=f'F1-macro={f1_test:.4f}')
axes[0,1].legend()

# 3. ROC par classe
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
for i, (cls, color) in enumerate(zip(le.classes_, colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    auc_score   = auc(fpr, tpr)
    axes[0,2].plot(fpr, tpr, color=color,
                   label=f'{cls} (AUC={auc_score:.4f})')
axes[0,2].plot([0,1],[0,1],'k--')
axes[0,2].set_title('Courbes ROC par Classe')
axes[0,2].legend()

# 4. Train vs Test
metrics    = ['Accuracy', 'F1-macro']
train_vals = [acc_train, f1_train]
test_vals  = [acc_test,  f1_test]
x = np.arange(len(metrics))
w = 0.35
axes[1,0].bar(x - w/2, train_vals, w, label='Train', color='steelblue')
axes[1,0].bar(x + w/2, test_vals,  w, label='Test',  color='orange')
axes[1,0].set_xticks(x)
axes[1,0].set_xticklabels(metrics)
axes[1,0].set_ylim([0.995, 1.001])
axes[1,0].set_title('Train vs Test — Overfitting Check')
axes[1,0].legend()

# 5. Convergence Optuna
trial_values = [t.value for t in study_hgb.trials]
best_values  = [max(trial_values[:i+1]) for i in range(len(trial_values))]
axes[1,1].scatter(range(len(trial_values)), trial_values,
                  alpha=0.5, color='steelblue', label='F1 trial')
axes[1,1].plot(range(len(best_values)), best_values,
               color='red', label=f'Meilleur={study_hgb.best_value:.4f}')
axes[1,1].set_title('Convergence Optuna (50 trials)')
axes[1,1].set_xlabel('Trial #')
axes[1,1].set_ylabel('F1-macro (CV-3)')
axes[1,1].legend()

# 6. Récapitulatif
recap_data = {
    'Métrique': ['Accuracy', 'F1-macro', 'ROC-AUC',
                 'Diff Train/Test', 'Erreurs totales'],
    'Valeur':   [f'{acc_test:.4f}', f'{f1_test:.4f}',
                 f'{roc_auc:.4f}',
                 f'{abs(acc_train-acc_test):.4f}',
                 f'{sum(y_pred != y_test):,}']
}
axes[1,2].axis('off')
table = axes[1,2].table(
    cellText=[[r, v] for r, v in zip(recap_data['Métrique'],
                                      recap_data['Valeur'])],
    colLabels=['Métrique', 'Valeur'],
    loc='center', cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)
axes[1,2].set_title('Récapitulatif Final', fontweight='bold')

plt.tight_layout()
plt.savefig('histgradient_optuna_resultats.png',
            dpi=150, bbox_inches='tight')
print("    histgradient_optuna_resultats.png ✓")

# ─────────────────────────────────────────────────────────────
# [7/8] Sauvegarde
# ─────────────────────────────────────────────────────────────
print("\n[7/8] Sauvegarde...")
joblib.dump(model_hgb, 'model_histgradient_optuna.pkl')
joblib.dump(le,        'label_encoder_hgb.pkl')

resultats = pd.DataFrame({
    'Métrique': ['Train_Acc', 'Test_Acc', 'Diff_Acc',
                 'Train_F1',  'Test_F1',  'ROC_AUC'],
    'Valeur':   [acc_train, acc_test, abs(acc_train-acc_test),
                 f1_train,  f1_test,  roc_auc]
})
resultats.to_csv('resultats_histgradient_optuna.csv', index=False)

pd.DataFrame(study_hgb.best_params, index=[0]).to_csv(
    'best_params_histgradient_optuna.csv', index=False
)

print("""
  Fichiers produits :
    model_histgradient_optuna.pkl         ✓
    label_encoder_hgb.pkl                 ✓
    resultats_histgradient_optuna.csv     ✓
    best_params_histgradient_optuna.csv   ✓
    histgradient_optuna_resultats.png     ✓
""")
print(f"  Temps total : {time.time()-t0:.1f}s")
print("=" * 65)