import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import time
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

t0 = time.time()

RANDOM_STATE = 42
TARGET = 'Classe_Congestion'
CELL_ID_COL = 'CELLNAME_ID'
N_TRIALS = 50
CV_FOLDS = 3
COLORS = {0: '#27AE60', 1: '#F39C12', 2: '#E74C3C'}
CLASS_NAMES = ['Normal', 'Modere', 'Critique']

# liste de features unifiee (memoir §3.8) - la meme que XGBoost / Random Forest
FEATURES = [
    'DL_PRB_USAGE_RATE', 'LTE_SETUP_SUCCESS_RATE', 'CELL_TRAFFIC_VOLUME_DL',
    'CELL_TRAFFIC_VOLUME_UL', 'DL_AVERAGE_THROUGHPUT', 'UL_AVERAGE_THROUGHPUT',
    'AVG_USER_NB', 'AVAIBILITY', 'HOUR', 'IS_WEEKEND', 'SPECTRAL_EFF',
    'IS_PEAK_HOUR', 'ROLLING_TRAFIC_3H', 'ROLLING_PRB_3H', 'HOURLY_TREND',
    'ROLLING_MEAN_VOLATILITY', 'PRB_Z_SCORE', 'GRADIENT_PRB'
]

# chargement
print("Chargement du dataset...")
df = pd.read_csv('df_avec_score_kmeans.csv')

missing_cols = [f for f in FEATURES if f not in df.columns]
if missing_cols:
    print("attention, colonnes absentes du dataset :", missing_cols)
FEATURES = [f for f in FEATURES if f in df.columns]
TIME_COL = 'DATE_' if 'DATE_' in df.columns else 'HOUR'

df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

print(len(df), "lignes,", len(FEATURES), "features")
dist = df[TARGET].value_counts(normalize=True).sort_index()
for cls, pct in dist.items():
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

# poids des classes (dataset desequilibre)
counts = y_train.value_counts().sort_index()
class_weights = {cls: len(y_train) / (3 * cnt) for cls, cnt in counts.items()}
class_weights_list = [class_weights[cls] for cls in range(3)]
print("poids des classes :", {k: round(v, 3) for k, v in class_weights.items()})

# optimisation des hyperparametres avec optuna
print("Recherche des hyperparametres (Optuna)...")

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def objective(trial):
    # bootstrap_type doit etre 'Bernoulli' ou 'MVS' pour pouvoir utiliser
    # subsample (pas possible avec 'Bayesian')
    bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bernoulli', 'MVS'])

    params = {
        'iterations': trial.suggest_int('iterations', 300, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 300),
        'rsm': trial.suggest_float('rsm', 0.5, 1.0),
        'bootstrap_type': bootstrap_type,
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'loss_function': 'MultiClass',
        'eval_metric': 'MultiClass',
        'class_weights': class_weights_list,
        'random_seed': RANDOM_STATE,
        'verbose': False,
        'thread_count': -1,
    }

    clf = CatBoostClassifier(**params)
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=1)
    return scores.mean()


t_optuna = time.time()
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=RANDOM_STATE),
    study_name='CatBoost_Djezzy_Congestion'
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

best_params = study.best_params
best_f1_cv = study.best_value
n_completed = len([t for t in study.trials if t.value is not None])
n_pruned = len(study.trials) - n_completed

print("Optuna termine en", round((time.time() - t_optuna) / 60, 1), "min")
print("Meilleur F1-macro CV :", round(best_f1_cv, 4))
print("Meilleurs parametres :", best_params)

# entrainement final
# note : od_wait et early_stopping_rounds sont des synonymes dans CatBoost,
# on ne garde que early_stopping_rounds (dans le fit) pour eviter le conflit
final_params = dict(best_params)
final_params.update({
    'loss_function': 'MultiClass',
    'eval_metric': 'MultiClass',
    'class_weights': class_weights_list,
    'random_seed': RANDOM_STATE,
    'verbose': False,
    'thread_count': -1,
})

t_train = time.time()
cat_final = CatBoostClassifier(**final_params)
cat_final.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50,
    verbose=False
)
print("Entrainement final termine en", round(time.time() - t_train, 1), "s")

# evaluation
print("Evaluation...")
y_pred = cat_final.predict(X_test).flatten()
y_proba = cat_final.predict_proba(X_test)
y_pred_train = cat_final.predict(X_train).flatten()
y_pred_val = cat_final.predict(X_val).flatten()

acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_cls = f1_score(y_test, y_pred, average=None)
f1_train = f1_score(y_train, y_pred_train, average='macro')
f1_val = f1_score(y_val, y_pred_val, average='macro')
gap_test = f1_train - f1_macro

cm = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
n_erreurs = int((y_pred != y_test.values).sum())

print("Accuracy       :", round(acc * 100, 3), "%")
print("F1 macro       :", round(f1_macro * 100, 3), "%")
print("F1 Critique    :", round(f1_cls[2] * 100, 3), "%")
print("F1 train       :", round(f1_train * 100, 3), "%")
print("F1 val         :", round(f1_val * 100, 3), "%")
print("Gap train/test :", round(gap_test * 100, 3), "%")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# calcul SHAP (implementation native CatBoost)
print("Calcul SHAP...")
n_shap = min(5000, len(X_test))
idx_shap = np.random.RandomState(RANDOM_STATE).choice(len(X_test), n_shap, replace=False)
X_shap = X_test.iloc[idx_shap].reset_index(drop=True)

shap_raw = cat_final.get_feature_importance(Pool(X_shap), type='ShapValues')
n_feat = len(FEATURES)

# la forme du tableau SHAP retourne par CatBoost varie selon la version,
# on detecte le format et on extrait les valeurs par classe (sans la colonne de biais)
if shap_raw.ndim == 3:
    dim0, dim1, dim2 = shap_raw.shape
    if dim2 == 3 and dim1 == n_feat + 1:
        shap_cl0, shap_cl1, shap_cl2 = shap_raw[:, :-1, 0], shap_raw[:, :-1, 1], shap_raw[:, :-1, 2]
    elif dim1 == 3 and dim2 == n_feat + 1:
        shap_cl0, shap_cl1, shap_cl2 = shap_raw[:, 0, :-1], shap_raw[:, 1, :-1], shap_raw[:, 2, :-1]
    elif dim1 == 3 and dim2 == n_feat:
        shap_cl0, shap_cl1, shap_cl2 = shap_raw[:, 0, :], shap_raw[:, 1, :], shap_raw[:, 2, :]
    else:
        print("format SHAP inconnu, fallback sur feature importance")
        fi_vals = cat_final.get_feature_importance()
        shap_cl2 = np.tile(fi_vals, (n_shap, 1))
        shap_cl0, shap_cl1 = shap_cl2.copy(), shap_cl2.copy()
elif shap_raw.ndim == 2:
    print("SHAP 2D, fallback sur feature importance")
    fi_vals = cat_final.get_feature_importance()
    shap_cl2 = np.tile(fi_vals, (n_shap, 1))
    shap_cl0, shap_cl1 = shap_cl2.copy(), shap_cl2.copy()
else:
    raise ValueError(f"shape SHAP inattendue : {shap_raw.shape}")

assert shap_cl2.shape[1] == n_feat, f"shap_cl2.shape[1]={shap_cl2.shape[1]} != n_feat={n_feat}"
shap_values = [shap_cl0, shap_cl1, shap_cl2]
print("SHAP ok, shape :", shap_cl2.shape)

# graphiques
print("Generation des graphiques...")
colors_list = [COLORS[0], COLORS[1], COLORS[2]]

fig1, axes = plt.subplots(2, 3, figsize=(20, 12))
fig1.suptitle('CatBoost + Optuna - Performance\nDjezzy BTS Congestion Detection', fontsize=14, fontweight='bold')

sns.heatmap(cm_pct, annot=True, fmt='.2f', ax=axes[0, 0],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap='Oranges')
axes[0, 0].set_title('Matrice de Confusion (%)')
axes[0, 0].set_ylabel('Reel')
axes[0, 0].set_xlabel('Predit')

feat_imp = pd.Series(cat_final.get_feature_importance(), index=FEATURES).sort_values(ascending=False)
axes[0, 1].barh(feat_imp.index[:12], feat_imp.values[:12], color='#E67E22')
axes[0, 1].set_title('Feature Importance CatBoost (Top 12)')
axes[0, 1].invert_yaxis()

trial_values = [t.value for t in study.trials if t.value is not None]
best_so_far = np.maximum.accumulate(trial_values)
axes[0, 2].plot(range(1, len(trial_values) + 1), trial_values, alpha=0.4, color='steelblue', label='Score par trial')
axes[0, 2].plot(range(1, len(best_so_far) + 1), best_so_far, color='#e74c3c', linewidth=2, label=f'Meilleur ({best_f1_cv:.4f})')
axes[0, 2].set_title(f'Convergence Optuna ({N_TRIALS} trials)')
axes[0, 2].set_xlabel('Trial')
axes[0, 2].set_ylabel('F1-macro CV')
axes[0, 2].legend()

x = np.arange(2)
w = 0.35
axes[1, 0].bar(x - w / 2, [f1_train, f1_train], w, label='Train', color='#E67E22')
axes[1, 0].bar(x + w / 2, [acc, f1_macro], w, label='Test', color='steelblue')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(['Accuracy', 'F1-macro'])
axes[1, 0].set_title('Train vs Test')
axes[1, 0].legend()

axes[1, 1].bar(CLASS_NAMES, f1_cls, color=colors_list)
axes[1, 1].set_title('F1-score par Classe')
axes[1, 1].axhline(f1_macro, color='navy', linestyle='--', label=f'F1-macro={f1_macro:.4f}')
axes[1, 1].legend()

axes[1, 2].axis('off')
recap_data = [
    ['Accuracy', f'{acc:.4f}'],
    ['F1-macro', f'{f1_macro:.4f}'],
    ['F1 Critique', f'{f1_cls[2]:.4f}'],
    ['Gap train/test', f'{gap_test:.4f}'],
    ['Erreurs', f'{n_erreurs:,}'],
    ['Trials Optuna', f'{N_TRIALS}'],
    ['Best F1 CV', f'{best_f1_cv:.4f}'],
    ['iterations', f'{best_params["iterations"]}'],
    ['depth', f'{best_params["depth"]}'],
    ['learning_rate', f'{best_params["learning_rate"]:.4f}'],
]
tbl = axes[1, 2].table(cellText=recap_data, colLabels=['Parametre / Metrique', 'Valeur'], loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 1.6)
axes[1, 2].set_title('Recapitulatif + Optuna', fontweight='bold')

plt.tight_layout()
plt.savefig('catboost_fig1.png', dpi=150, bbox_inches='tight')
print("catboost_fig1.png sauvegarde")
plt.close()

fig2, axes2 = plt.subplots(2, 3, figsize=(22, 14))
fig2.suptitle('CatBoost - Analyse SHAP\nInterpretabilite des Decisions', fontsize=14, fontweight='bold')

for i, cls in enumerate(CLASS_NAMES):
    shap_mean = np.abs(shap_values[i]).mean(axis=0)
    shap_df = pd.Series(shap_mean, index=FEATURES).sort_values(ascending=False)
    axes2[0, i].barh(shap_df.index[:10], shap_df.values[:10], color=COLORS[i])
    axes2[0, i].set_title(f'SHAP - Classe {cls}', fontweight='bold')
    axes2[0, i].invert_yaxis()

y_shap = y_test.values[idx_shap]
critique_positions = np.where(y_shap == 2)[0]
if len(critique_positions) > 0:
    pos = critique_positions[0]
    shap_cell = shap_values[2][pos]
    sorted_i = np.argsort(np.abs(shap_cell))[-10:]
    axes2[1, 0].barh(
        [FEATURES[j] for j in sorted_i],
        shap_cell[sorted_i],
        color=['#e74c3c' if v > 0 else '#2ecc71' for v in shap_cell[sorted_i]]
    )
    axes2[1, 0].axvline(0, color='black', linewidth=0.8)
    axes2[1, 0].set_title('SHAP - 1 Cellule Critique')
else:
    axes2[1, 0].set_title('Aucune cellule Critique dans l\'echantillon SHAP')
    axes2[1, 0].axis('off')

prob_critique = y_proba[idx_shap, 2] if y_proba.shape[0] == len(X_test) else None
if prob_critique is None:
    prob_critique = cat_final.predict_proba(X_shap)[:, 2]
for i, cls in enumerate(CLASS_NAMES):
    mask = y_shap == i
    axes2[1, 1].hist(prob_critique[mask], bins=50, alpha=0.6, color=COLORS[i], label=cls, density=True)
axes2[1, 1].axvline(0.5, color='black', linestyle='--', label='Seuil=0.5')
axes2[1, 1].set_title('Distribution P(Critique)')
axes2[1, 1].set_yscale('log')
axes2[1, 1].legend()

shap_global = np.mean([np.abs(shap_values[i]).mean(axis=0) for i in range(3)], axis=0)
shap_df_global = pd.Series(shap_global, index=FEATURES).sort_values(ascending=False)
axes2[1, 2].barh(shap_df_global.index[:12], shap_df_global.values[:12], color='purple')
axes2[1, 2].set_title('SHAP Global - Toutes Classes')
axes2[1, 2].invert_yaxis()

plt.tight_layout()
plt.savefig('catboost_fig2.png', dpi=150, bbox_inches='tight')
print("catboost_fig2.png sauvegarde")
plt.close()

# sauvegarde
print("Sauvegarde des fichiers...")
cat_final.save_model('catboost_congestion.cbm')
feat_imp.to_csv('feature_importance_catboost.csv')

with open('catboost_best_params.json', 'w') as f:
    json.dump({
        'best_params': best_params,
        'best_f1_cv': best_f1_cv,
        'n_trials': N_TRIALS,
        'n_completed': n_completed,
        'n_pruned': n_pruned,
        'time_minutes': round((time.time() - t_optuna) / 60, 2)
    }, f, indent=2)

pd.DataFrame({
    'Metrique': ['Accuracy', 'F1_macro', 'F1_Critique', 'F1_train', 'F1_val', 'Gap', 'Best_F1_CV'],
    'Valeur': [acc, f1_macro, f1_cls[2], f1_train, f1_val, gap_test, best_f1_cv]
}).to_csv('resultats_catboost.csv', index=False)

print("Fichiers generes : catboost_congestion.cbm, catboost_best_params.json,",
      "feature_importance_catboost.csv, resultats_catboost.csv, catboost_fig1.png, catboost_fig2.png")
print("Temps total :", round((time.time() - t0) / 60, 1), "min")