import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
import shap
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import json
import time
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def fmt(s):
    h, m = int(s // 3600), int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h}h {m:02d}min {sec:02d}s" if h > 0 else \
           f"{m}min {sec:02d}s" if m > 0 else f"{sec:.1f}s"


TEMPS_DEBUT = time.time()

RANDOM_STATE = 42
TARGET = 'Classe_Congestion'
CELL_ID_COL = 'CELLNAME_ID'
N_CLASSES = 3
N_TRIALS = 50
CV_FOLDS = 3
N_OPTUNA_SAMPLE = 1_000_000
COLORS = {0: '#27AE60', 1: '#F39C12', 2: '#E74C3C'}
CLASS_NAMES = ['Normal', 'Modere', 'Critique']

# liste de features unifiee (memoir §3.8) - la meme que les autres modeles
FEATURES = [
    'DL_PRB_USAGE_RATE', 'LTE_SETUP_SUCCESS_RATE', 'CELL_TRAFFIC_VOLUME_DL',
    'CELL_TRAFFIC_VOLUME_UL', 'DL_AVERAGE_THROUGHPUT', 'UL_AVERAGE_THROUGHPUT',
    'AVG_USER_NB', 'AVAIBILITY', 'HOUR', 'IS_WEEKEND', 'SPECTRAL_EFF',
    'IS_PEAK_HOUR', 'ROLLING_TRAFIC_3H', 'ROLLING_PRB_3H', 'HOURLY_TREND',
    'ROLLING_MEAN_VOLATILITY', 'PRB_Z_SCORE', 'GRADIENT_PRB'
]

# chargement
print("Chargement...")
df = pd.read_csv('df_avec_score_kmeans.csv')

missing_cols = [f for f in FEATURES if f not in df.columns]
if missing_cols:
    print("attention, colonnes absentes du dataset :", missing_cols)
FEATURES = [f for f in FEATURES if f in df.columns]
TIME_COL = 'DATE_' if 'DATE_' in df.columns else 'HOUR'

df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
df[TARGET] = df[TARGET].astype(int)

print(len(df), "lignes,", len(FEATURES), "features")
for cls, pct in df[TARGET].value_counts(normalize=True).sort_index().items():
    print("classe", cls, CLASS_NAMES[cls], ":", round(pct * 100, 1), "%")


def split_chronologique(data, cell_col=CELL_ID_COL, time_col=TIME_COL,
                         train_frac=0.70, val_frac=0.15):
    """
    Split 70/15/15 chronologique, cellule par cellule : pour chaque
    CELLNAME_ID on trie par ordre temporel puis on coupe en 3 blocs
    (train = debut, val = milieu, test = fin). Evite le data leakage
    temporel d'un split random classique.
    """
    train_parts, val_parts, test_parts = [], [], []

    for _, groupe in data.groupby(cell_col):
        groupe = groupe.sort_values(time_col)
        n = len(groupe)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train_parts.append(groupe.iloc[:n_train])
        val_parts.append(groupe.iloc[n_train:n_train + n_val])
        test_parts.append(groupe.iloc[n_train + n_val:])

    train_df = pd.concat(train_parts).sort_values([cell_col, time_col])
    val_df = pd.concat(val_parts).sort_values([cell_col, time_col])
    test_df = pd.concat(test_parts).sort_values([cell_col, time_col])
    return train_df, val_df, test_df


print("Split chronologique 70/15/15 par cellule...")
train_df, val_df, test_df = split_chronologique(df)

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_val, y_val = val_df[FEATURES], val_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

print("Train :", len(X_train), "| Val :", len(X_val), "| Test :", len(X_test))

counts = y_train.value_counts().sort_index()
class_weights = {cls: len(y_train) / (N_CLASSES * cnt) for cls, cnt in counts.items()}
sample_weights = y_train.map(class_weights).values
print("poids des classes :", {k: round(v, 3) for k, v in class_weights.items()})

# echantillon pour Optuna (recherche d'hyperparametres sur un sous-ensemble, plus rapide)
n_opt = min(N_OPTUNA_SAMPLE, len(X_train))
opt_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train), n_opt, replace=False)
X_opt = X_train.iloc[opt_idx]
y_opt = y_train.iloc[opt_idx]
w_opt = y_opt.map(class_weights).values
print("echantillon Optuna :", len(X_opt), "lignes (", round(len(X_opt) / len(X_train) * 100, 1), "% du train)")

# optimisation des hyperparametres avec optuna (SuccessiveHalvingPruner pour couper les essais faibles tot)
print("Recherche des hyperparametres (Optuna)...")

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'objective': 'multiclass',
        'num_class': N_CLASSES,
        'metric': 'multi_logloss',
        'class_weight': class_weights,
        'n_jobs': -1,
        'random_state': RANDOM_STATE,
        'verbosity': -1,
    }

    f1_scores = []
    for fold_idx, (idx_tr, idx_va) in enumerate(cv.split(X_opt, y_opt)):
        X_f_tr, y_f_tr = X_opt.iloc[idx_tr], y_opt.iloc[idx_tr]
        X_f_va, y_f_va = X_opt.iloc[idx_va], y_opt.iloc[idx_va]
        w_f_tr = w_opt[idx_tr]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_f_tr, y_f_tr,
            sample_weight=w_f_tr,
            eval_set=[(X_f_va, y_f_va)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)]
        )
        preds = model.predict(X_f_va)
        f1 = f1_score(y_f_va, preds, average='macro')
        f1_scores.append(f1)

        trial.report(f1, fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return float(np.mean(f1_scores))


t_optuna = time.time()
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=RANDOM_STATE),
    pruner=SuccessiveHalvingPruner(min_resource=1, reduction_factor=2, min_early_stopping_rate=0)
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

best_params = study.best_params
best_f1_cv = study.best_value
n_completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)

print("Optuna termine en", fmt(time.time() - t_optuna))
print("Complets :", n_completed, "| Pruned :", n_pruned)
print("Meilleur F1-macro CV :", round(best_f1_cv, 4))
print("Meilleurs parametres :", best_params)

with open('lightgbm_best_params.json', 'w') as f:
    json.dump({
        'best_params': best_params,
        'best_f1_cv': best_f1_cv,
        'n_trials': N_TRIALS,
        'n_completed': n_completed,
        'n_pruned': n_pruned,
        'time_minutes': round((time.time() - t_optuna) / 60, 2)
    }, f, indent=2)

# entrainement final sur tout le train, avec early stopping sur le vrai val set
print("Entrainement final...")
t_train = time.time()

final_params = {
    **best_params,
    'objective': 'multiclass',
    'num_class': N_CLASSES,
    'metric': 'multi_logloss',
    'class_weight': class_weights,
    'n_jobs': -1,
    'random_state': RANDOM_STATE,
    'verbosity': -1,
}

lgbm_final = lgb.LGBMClassifier(**final_params)
lgbm_final.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
)
n_trees = lgbm_final.best_iteration_
print("n_estimators utilises :", n_trees)
print("Entrainement termine en", fmt(time.time() - t_train))

# evaluation
y_pred = lgbm_final.predict(X_test)
y_proba = lgbm_final.predict_proba(X_test)
y_train_pred = lgbm_final.predict(X_train)
y_test_np = np.array(y_test)
y_train_np = np.array(y_train)

acc_test = accuracy_score(y_test_np, y_pred)
acc_train = accuracy_score(y_train_np, y_train_pred)
f1_test = f1_score(y_test_np, y_pred, average='macro')
f1_train = f1_score(y_train_np, y_train_pred, average='macro')
f1_cls = f1_score(y_test_np, y_pred, average=None)
roc_auc = roc_auc_score(y_test_np, y_proba, multi_class='ovr')

print("Train accuracy   :", round(acc_train, 4))
print("Test accuracy    :", round(acc_test, 4))
print("Train F1-macro   :", round(f1_train, 4))
print("Test F1-macro    :", round(f1_test, 4))
print("F1 Critique      :", round(f1_cls[2], 4))
print("ROC-AUC          :", round(roc_auc, 4))
print(classification_report(y_test_np, y_pred, target_names=CLASS_NAMES))

# tests anti-overfitting
print("Tests anti-overfitting...")

# test 1 : gap train/test
gap_acc = abs(acc_train - acc_test)
gap_f1 = abs(f1_train - f1_test)
t1_ok = gap_acc < 0.02
print("Test 1 - gap train/test : acc =", round(gap_acc, 4), "f1 =", round(gap_f1, 4), "->", "ok" if t1_ok else "a verifier")

# test 2 : cross-validation 5 folds sur un sous-echantillon du train
print("Test 2 - CV 5 folds (sous-echantillon 300k)...")
rng = np.random.default_rng(RANDOM_STATE)
n_cv = min(300_000, len(X_train))
cv_idx = rng.choice(len(X_train), n_cv, replace=False)
X_cv = X_train.iloc[cv_idx].values
y_cv = y_train_np[cv_idx]

cv_model = lgb.LGBMClassifier(
    n_estimators=min(500, n_trees), num_leaves=best_params['num_leaves'],
    learning_rate=best_params['learning_rate'], reg_alpha=best_params['reg_alpha'],
    reg_lambda=best_params['reg_lambda'], min_child_samples=best_params['min_child_samples'],
    objective='multiclass', num_class=N_CLASSES, n_jobs=-1, random_state=RANDOM_STATE, verbosity=-1,
)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_f1s = []
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_cv, y_cv), 1):
    cv_model.fit(X_cv[tr_idx], y_cv[tr_idx])
    preds = cv_model.predict(X_cv[va_idx])
    f1_fold = f1_score(y_cv[va_idx], preds, average='macro')
    cv_f1s.append(f1_fold)
    print("  fold", fold, "F1 =", round(f1_fold, 4))

cv_mean, cv_std = np.mean(cv_f1s), np.std(cv_f1s)
t2_ok = cv_std < 0.015
print("Test 2 - CV F1 =", round(cv_mean, 4), "+/-", round(cv_std, 4), "->", "ok" if t2_ok else "a verifier")

# test 3 : learning curve
print("Test 3 - learning curve (sous-echantillon 100k)...")
idx_lc = rng.choice(len(cv_idx), min(100_000, len(cv_idx)), replace=False)
X_lc, y_lc = X_cv[idx_lc], y_cv[idx_lc]
lc_model = lgb.LGBMClassifier(
    n_estimators=200, num_leaves=best_params['num_leaves'], learning_rate=best_params['learning_rate'],
    objective='multiclass', num_class=N_CLASSES, n_jobs=-1, random_state=RANDOM_STATE, verbosity=-1,
)
train_sizes_abs, train_scores, val_scores = learning_curve(
    lc_model, X_lc, y_lc, train_sizes=np.linspace(0.1, 1.0, 8), cv=3, scoring='f1_macro', n_jobs=-1
)
lc_train_mean, lc_val_mean = train_scores.mean(axis=1), val_scores.mean(axis=1)
lc_gap_final = abs(lc_train_mean[-1] - lc_val_mean[-1])
t3_ok = lc_gap_final < 0.05
print("Test 3 - gap learning curve final =", round(lc_gap_final, 4), "->", "ok" if t3_ok else "a verifier")

# test 4 : stabilite sur sous-groupes du test
print("Test 4 - stabilite sur sous-groupes...")
rng2 = np.random.default_rng(123)
f1_sub = []
for trial in range(5):
    sub_idx = rng2.choice(len(X_test), size=min(50_000, len(X_test)), replace=False)
    X_sub = X_test.iloc[sub_idx]
    y_sub = y_test_np[sub_idx]
    pred_sub = lgbm_final.predict(X_sub)
    f1_sub.append(f1_score(y_sub, pred_sub, average='macro'))
f1_sub_std = np.std(f1_sub)
t4_ok = f1_sub_std < 0.01
print("Test 4 - F1 moyen =", round(np.mean(f1_sub), 4), "+/-", round(f1_sub_std, 4), "->", "ok" if t4_ok else "a verifier")

# calcul SHAP
print("Calcul SHAP (TreeExplainer)...")
n_shap = min(5000, len(X_test))
idx_shap = np.random.RandomState(RANDOM_STATE).choice(len(X_test), n_shap, replace=False)
X_shap = X_test.iloc[idx_shap].reset_index(drop=True)
y_shap = y_test_np[idx_shap]

explainer = shap.TreeExplainer(lgbm_final)
shap_raw = explainer.shap_values(X_shap)

# selon la version de shap le format de sortie change, on uniformise
if isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
    shap_vals = [shap_raw[:, :, i] for i in range(N_CLASSES)]
elif isinstance(shap_raw, list):
    shap_vals = shap_raw
else:
    raise ValueError(f"format SHAP inattendu : {type(shap_raw)}")

print("SHAP calculees, shape par classe :", shap_vals[0].shape)

shap_df = pd.DataFrame({
    'Feature': FEATURES,
    'SHAP_Normal': np.abs(shap_vals[0]).mean(axis=0),
    'SHAP_Modere': np.abs(shap_vals[1]).mean(axis=0),
    'SHAP_Critique': np.abs(shap_vals[2]).mean(axis=0),
}).sort_values('SHAP_Critique', ascending=False)
shap_df.to_csv('shap_importance_par_classe.csv', index=False)

# graphiques
print("Generation des graphiques...")

fig1, axes = plt.subplots(2, 3, figsize=(20, 12))
fig1.suptitle('LightGBM + Optuna - Performance\nDjezzy BTS Congestion Detection', fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test_np, y_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt='.2f', ax=axes[0, 0], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap='Blues')
axes[0, 0].set_title('Matrice de Confusion (%)')

fi = pd.Series(lgbm_final.feature_importances_, index=FEATURES).sort_values(ascending=False)
axes[0, 1].barh(fi.index[:12], fi.values[:12], color='#3498DB')
axes[0, 1].set_title('Feature Importance LightGBM (Top 12)')
axes[0, 1].invert_yaxis()

y_test_bin = label_binarize(y_test_np, classes=[0, 1, 2])
for i, cls in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    axes[0, 2].plot(fpr, tpr, color=COLORS[i], label=f'{cls} (AUC={auc(fpr, tpr):.4f})')
axes[0, 2].plot([0, 1], [0, 1], 'k--')
axes[0, 2].set_title('Courbes ROC')
axes[0, 2].legend()

x = np.arange(2)
w = 0.35
axes[1, 0].bar(x - w / 2, [acc_train, f1_train], w, label='Train', color='steelblue')
axes[1, 0].bar(x + w / 2, [acc_test, f1_test], w, label='Test', color='orange')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(['Accuracy', 'F1-macro'])
axes[1, 0].set_title('Train vs Test')
axes[1, 0].legend()

axes[1, 1].bar(CLASS_NAMES, f1_cls, color=[COLORS[i] for i in range(3)])
axes[1, 1].set_title('F1-score par Classe')
axes[1, 1].axhline(f1_test, color='navy', linestyle='--', label=f'F1-macro={f1_test:.4f}')
axes[1, 1].legend()

axes[1, 2].axis('off')
recap = [
    ['Accuracy', f'{acc_test:.4f}'], ['F1-macro', f'{f1_test:.4f}'],
    ['F1 Critique', f'{f1_cls[2]:.4f}'], ['ROC-AUC', f'{roc_auc:.4f}'],
    ['Test1 gap acc/f1', f'{gap_acc:.4f} / {gap_f1:.4f}'],
    ['Test2 CV std', f'{cv_std:.4f}'], ['Test3 LC gap', f'{lc_gap_final:.4f}'],
    ['Test4 stabilite std', f'{f1_sub_std:.4f}'], ['n_estimators', f'{n_trees}'],
]
tbl = axes[1, 2].table(cellText=recap, colLabels=['Metrique', 'Valeur'], loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.1, 1.5)
axes[1, 2].set_title('Recapitulatif + Tests Overfitting', fontweight='bold')

plt.tight_layout()
plt.savefig('lightgbm_fig1.png', dpi=150, bbox_inches='tight')
print("lightgbm_fig1.png sauvegarde")
plt.close()

fig2, axes2 = plt.subplots(2, 3, figsize=(22, 14))
fig2.suptitle('LightGBM - Analyse SHAP\nInterpretabilite des Decisions', fontsize=14, fontweight='bold')

for i, cls in enumerate(CLASS_NAMES):
    shap_mean = np.abs(shap_vals[i]).mean(axis=0)
    s = pd.Series(shap_mean, index=FEATURES).sort_values(ascending=False)
    axes2[0, i].barh(s.index[:10], s.values[:10], color=COLORS[i])
    axes2[0, i].set_title(f'SHAP - Classe {cls}', fontweight='bold')
    axes2[0, i].invert_yaxis()

critique_local = np.where(y_shap == 2)[0]
ax_cell = axes2[1, 0]
if len(critique_local) > 0:
    shap_cell = shap_vals[2][critique_local[0]]
    sorted_i = np.argsort(np.abs(shap_cell))[-10:]
    ax_cell.barh([FEATURES[j] for j in sorted_i], shap_cell[sorted_i],
                 color=['#e74c3c' if v > 0 else '#2ecc71' for v in shap_cell[sorted_i]])
    ax_cell.axvline(0, color='black', linewidth=0.8)
    ax_cell.set_title('SHAP - 1 Cellule Critique')
else:
    ax_cell.set_title('Aucune cellule Critique dans l\'echantillon')
    ax_cell.axis('off')

prob_c = y_proba[idx_shap, 2] if y_proba.shape[0] == len(X_test) else lgbm_final.predict_proba(X_shap)[:, 2]
for i, cls in enumerate(CLASS_NAMES):
    mask = y_shap == i
    if mask.sum() > 0:
        axes2[1, 1].hist(prob_c[mask], bins=50, alpha=0.6, color=COLORS[i], label=cls, density=True)
axes2[1, 1].axvline(0.5, color='black', linestyle='--', label='Seuil=0.5')
axes2[1, 1].set_title('Distribution P(Critique)')
axes2[1, 1].set_yscale('log')
axes2[1, 1].legend()

axes2[1, 2].barh(shap_df['Feature'][:12], shap_df['SHAP_Critique'][:12], color='purple')
axes2[1, 2].set_title('SHAP - Top Features Classe Critique')
axes2[1, 2].invert_yaxis()

plt.tight_layout()
plt.savefig('lightgbm_fig2.png', dpi=150, bbox_inches='tight')
print("lightgbm_fig2.png sauvegarde")
plt.close()

# sauvegarde
print("Sauvegarde des fichiers...")
joblib.dump(lgbm_final, 'model_lightgbm_final.pkl')
lgbm_final.booster_.save_model('lightgbm_congestion_final.txt')

df_preds = X_test.copy()
df_preds['y_true'] = y_test_np
df_preds['y_pred'] = y_pred
for i in range(N_CLASSES):
    df_preds[f'proba_{i}'] = y_proba[:, i]
df_preds['erreur'] = (y_test_np != y_pred).astype(int)
df_preds.to_csv('predictions_lightgbm_final.csv', index=False)

pd.DataFrame({
    'Metrique': ['Train_Acc', 'Test_Acc', 'Test_F1', 'F1_Critique', 'ROC_AUC',
                 'Gap_Acc', 'CV_std', 'LC_gap', 'Stabilite_std'],
    'Valeur': [acc_train, acc_test, f1_test, f1_cls[2], roc_auc, gap_acc, cv_std, lc_gap_final, f1_sub_std]
}).to_csv('resultats_lightgbm.csv', index=False)

print("Fichiers generes : model_lightgbm_final.pkl, lightgbm_congestion_final.txt,",
      "predictions_lightgbm_final.csv, shap_importance_par_classe.csv, resultats_lightgbm.csv,",
      "lightgbm_best_params.json, lightgbm_fig1.png, lightgbm_fig2.png")
print("Temps total :", fmt(time.time() - TEMPS_DEBUT))