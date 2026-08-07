import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, classification_report,
                             confusion_matrix, roc_curve, auc)
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

t0 = time.time()

RANDOM_STATE = 42
TARGET = 'Classe_Congestion'
CELL_ID_COL = 'CELLNAME_ID'
N_TRIALS = 30
CLASS_NAMES = ['Normal', 'Modere', 'Critique']

# liste de features unifiee (memoir §3.8) - la meme que XGBoost / Random Forest / CatBoost / HistGradientBoosting
FEATURES = [
    'DL_PRB_USAGE_RATE', 'LTE_SETUP_SUCCESS_RATE', 'CELL_TRAFFIC_VOLUME_DL',
    'CELL_TRAFFIC_VOLUME_UL', 'DL_AVERAGE_THROUGHPUT', 'UL_AVERAGE_THROUGHPUT',
    'AVG_USER_NB', 'AVAIBILITY', 'HOUR', 'IS_WEEKEND', 'SPECTRAL_EFF',
    'IS_PEAK_HOUR', 'ROLLING_TRAFIC_3H', 'ROLLING_PRB_3H', 'HOURLY_TREND',
    'ROLLING_MEAN_VOLATILITY', 'PRB_Z_SCORE', 'GRADIENT_PRB'
]

# chargement
print("Chargement des donnees...")
df = pd.read_csv("df_avec_score_kmeans.csv")
df.columns = df.columns.str.strip()

missing_cols = [f for f in FEATURES if f not in df.columns]
if missing_cols:
    print("attention, colonnes absentes du dataset :", missing_cols)
FEATURES = [f for f in FEATURES if f in df.columns]
TIME_COL = 'DATE_' if 'DATE_' in df.columns else 'HOUR'

TARGET_COL = TARGET
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
df.dropna(subset=[TARGET_COL], inplace=True)
df[TARGET_COL] = df[TARGET_COL].astype(int)

nan_counts = df[FEATURES].isna().sum()
if nan_counts.sum() > 0:
    print("NaN detectes :")
    print(nan_counts[nan_counts > 0].to_string())
    df.dropna(subset=FEATURES, inplace=True)
    print("apres suppression :", len(df), "lignes")

print(len(df), "lignes,", len(FEATURES), "features")
print(df[TARGET_COL].value_counts().sort_index().to_string())


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

X_train_raw, y_train = train_df[FEATURES], train_df[TARGET_COL]
X_val_raw, y_val = val_df[FEATURES], val_df[TARGET_COL]
X_test_raw, y_test = test_df[FEATURES], test_df[TARGET_COL]

print("Train :", len(X_train_raw), "| Val :", len(X_val_raw), "| Test :", len(X_test_raw))

# normalisation obligatoire pour un MLP
print("Normalisation StandardScaler...")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_raw)
X_val_sc = scaler.transform(X_val_raw)
X_test_sc = scaler.transform(X_test_raw)

# DataFrames avec noms de colonnes, utiles pour SHAP
X_train_df = pd.DataFrame(X_train_sc, columns=FEATURES)
X_val_df = pd.DataFrame(X_val_sc, columns=FEATURES)
X_test_df = pd.DataFrame(X_test_sc, columns=FEATURES)

y_train_v = y_train.values
y_val_v = y_val.values
y_test_v = y_test.values

# optimisation des hyperparametres avec optuna
# on tune sur un sous-echantillon du train pour aller plus vite
print("Recherche des hyperparametres (Optuna)...")

rng = np.random.default_rng(RANDOM_STATE)
n_opt_sample = min(200_000, len(X_train_sc))
opt_idx = rng.choice(len(X_train_sc), n_opt_sample, replace=False)
X_opt = X_train_sc[opt_idx]
y_opt = y_train_v[opt_idx]


def objective(trial):
    n_layers = trial.suggest_int('n_layers', 2, 4)
    layers = tuple(trial.suggest_int(f'n_units_l{i}', 64, 512) for i in range(n_layers))

    params = {
        'hidden_layer_sizes': layers,
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'solver': 'adam',
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024]),
        'max_iter': 100,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10,
        'random_state': RANDOM_STATE,
    }
    model = MLPClassifier(**params)
    scores = cross_val_score(model, X_opt, y_opt, cv=3, scoring='f1_macro', n_jobs=-1)
    return scores.mean()


t_optuna = time.time()
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

print("Optuna termine en", round(time.time() - t_optuna, 1), "s")
print("Meilleur F1-macro CV :", round(study.best_value, 4))
print("Meilleurs parametres :", study.best_params)

# reconstruction de l'architecture et entrainement final
print("Entrainement final...")

best = study.best_params
n_layers = best['n_layers']
layers = tuple(best[f'n_units_l{i}'] for i in range(n_layers))

final_params = {
    'hidden_layer_sizes': layers,
    'activation': best['activation'],
    'solver': 'adam',
    'alpha': best['alpha'],
    'learning_rate_init': best['learning_rate_init'],
    'batch_size': best['batch_size'],
    'max_iter': 300,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 20,
    'random_state': RANDOM_STATE,
    'verbose': False,
}

print("Architecture :", layers)
t_train = time.time()
model = MLPClassifier(**final_params)
model.fit(X_train_sc, y_train_v)
print("Entrainement termine en", round(time.time() - t_train, 1), "s")
print("Iterations :", model.n_iter_)

# evaluation
print("Evaluation...")
y_pred = model.predict(X_test_sc)
y_pred_prob = model.predict_proba(X_test_sc)
y_train_pred = model.predict(X_train_sc)

acc_test = accuracy_score(y_test_v, y_pred)
acc_train = accuracy_score(y_train_v, y_train_pred)
f1_test = f1_score(y_test_v, y_pred, average='macro', zero_division=0)
f1_train = f1_score(y_train_v, y_train_pred, average='macro', zero_division=0)

classes_in_test = np.unique(y_test_v)
if len(classes_in_test) == 3:
    roc_auc = roc_auc_score(y_test_v, y_pred_prob, multi_class='ovr')
else:
    roc_auc = float('nan')
    print("ROC-AUC non calculable, une classe est absente du test set")

print("Architecture     :", layers)
print("Train accuracy   :", round(acc_train, 4))
print("Test accuracy    :", round(acc_test, 4))
print("Diff             :", round(abs(acc_train - acc_test), 4))
print("Train F1-macro   :", round(f1_train, 4))
print("Test F1-macro    :", round(f1_test, 4))
print("ROC-AUC          :", round(roc_auc, 4) if not np.isnan(roc_auc) else 'N/A')
print(classification_report(y_test_v, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES, zero_division=0))

# calcul SHAP (KernelExplainer, MLP = boite noire, pas de TreeExplainer possible)
print("Calcul SHAP (KernelExplainer)...")
print("KernelExplainer est lent, on limite a un petit echantillon")

bg_idx = rng.choice(len(X_train_df), 200, replace=False)
X_bg = X_train_df.iloc[bg_idx].reset_index(drop=True)

shap_idx_list = []
for cls in [0, 1, 2]:
    pool = np.where(y_test_v == cls)[0]
    n = min(167, len(pool))
    if n > 0:
        shap_idx_list.extend(rng.choice(pool, n, replace=False).tolist())
shap_idx = np.array(shap_idx_list)

X_shap = X_test_df.iloc[shap_idx].reset_index(drop=True)
y_shap = y_test_v[shap_idx]

print("Background :", len(X_bg), "| SHAP :", len(X_shap))

explainer = shap.KernelExplainer(model.predict_proba, X_bg)
shap_values = explainer.shap_values(X_shap, nsamples=100)

# selon la version, KernelExplainer retourne soit une liste de 3 arrays,
# soit un array 3D (n, features, classes), on uniformise
if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
elif isinstance(shap_values, list):
    pass
else:
    shap_values = [shap_values]

print("SHAP calculees,", len(shap_values), "classes, shape :", shap_values[0].shape)

# graphiques
print("Generation des graphiques...")
colors = ['#2ecc71', '#f39c12', '#e74c3c']

fig1, axes = plt.subplots(2, 3, figsize=(20, 12))
fig1.suptitle('MLP + Optuna + SHAP - Performance\nDjezzy BTS Congestion Detection', fontsize=14, fontweight='bold')

cm_arr = confusion_matrix(y_test_v, y_pred, labels=[0, 1, 2])
cm_pct = cm_arr.astype('float') / cm_arr.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt='.2f', ax=axes[0, 0], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap='Purples')
axes[0, 0].set_title('Matrice de Confusion (%)')
axes[0, 0].set_ylabel('Reel')
axes[0, 0].set_xlabel('Predit')

axes[0, 1].plot(model.loss_curve_, color='purple', label='Train Loss')
axes[0, 1].set_title('Courbe de Loss MLP')
axes[0, 1].set_xlabel('Iterations')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()

y_test_bin = label_binarize(y_test_v, classes=[0, 1, 2])
for i, (cls, col) in enumerate(zip(CLASS_NAMES, colors)):
    if y_test_bin[:, i].sum() > 0:
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        axes[0, 2].plot(fpr, tpr, color=col, label=f'{cls} (AUC={auc(fpr, tpr):.4f})')
axes[0, 2].plot([0, 1], [0, 1], 'k--')
axes[0, 2].set_title('Courbes ROC par Classe')
axes[0, 2].legend(fontsize=8)

x = np.arange(2)
w = 0.35
axes[1, 0].bar(x - w / 2, [acc_train, f1_train], w, label='Train', color='purple')
axes[1, 0].bar(x + w / 2, [acc_test, f1_test], w, label='Test', color='orange')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(['Accuracy', 'F1-macro'])
axes[1, 0].set_title('Train vs Test')
axes[1, 0].legend()

f1_scores = f1_score(y_test_v, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
axes[1, 1].bar(CLASS_NAMES, f1_scores, color=colors)
axes[1, 1].set_title('F1-score par Classe')
axes[1, 1].axhline(f1_test, color='navy', linestyle='--', label=f'F1-macro={f1_test:.4f}')
axes[1, 1].legend()

trial_values = [t.value for t in study.trials if t.value is not None]
best_so_far = np.maximum.accumulate(trial_values)
axes[1, 2].plot(range(1, len(trial_values) + 1), trial_values, alpha=0.4, color='purple', label='Score par trial')
axes[1, 2].plot(range(1, len(best_so_far) + 1), best_so_far, color='red', linewidth=2, label=f'Meilleur ({study.best_value:.4f})')
axes[1, 2].set_title(f'Convergence Optuna ({N_TRIALS} trials)')
axes[1, 2].set_xlabel('Trial')
axes[1, 2].set_ylabel('F1-macro CV')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('mlp_shap_fig1.png', dpi=150, bbox_inches='tight')
print("mlp_shap_fig1.png sauvegarde")
plt.close()

fig2, axes2 = plt.subplots(2, 3, figsize=(22, 14))
fig2.suptitle('MLP - Analyse SHAP\nInterpretabilite des Decisions', fontsize=14, fontweight='bold')

for i, (cls, col) in enumerate(zip(CLASS_NAMES, colors)):
    shap_mean = np.abs(shap_values[i]).mean(axis=0)
    shap_df = pd.Series(shap_mean, index=FEATURES).sort_values(ascending=False)
    top_n = min(10, len(shap_df))
    axes2[0, i].barh(shap_df.index[:top_n], shap_df.values[:top_n], color=col)
    axes2[0, i].set_title(f'SHAP - Classe {cls}', fontweight='bold')
    axes2[0, i].set_xlabel('|SHAP| moyen')
    axes2[0, i].invert_yaxis()

critique_local = np.where(y_shap == 2)[0]
ax_cell = axes2[1, 0]
if len(critique_local) > 0:
    shap_cell = shap_values[2][critique_local[0]]
    top_n_cell = min(10, len(shap_cell))
    sorted_i = np.argsort(np.abs(shap_cell))[-top_n_cell:]
    vals_cell = shap_cell[sorted_i]
    feat_cell = [FEATURES[j] for j in sorted_i]
    bar_col = ['#e74c3c' if v > 0 else '#2ecc71' for v in vals_cell]

    ax_cell.barh(range(top_n_cell), vals_cell, color=bar_col)
    ax_cell.set_yticks(range(top_n_cell))
    ax_cell.set_yticklabels(feat_cell, fontsize=8)
    ax_cell.axvline(0, color='black', linewidth=0.8)
    ax_cell.set_title('SHAP - 1 Cellule Critique')
    ax_cell.set_xlabel('Valeur SHAP')
else:
    ax_cell.text(0.5, 0.5, 'Aucune cellule\nCritique dans\nl\'echantillon', ha='center', va='center', transform=ax_cell.transAxes, fontsize=12)
    ax_cell.axis('off')

prob_c = y_pred_prob[:, 2]
for i, (cls, col) in enumerate(zip(CLASS_NAMES, colors)):
    mask = y_test_v == i
    if mask.sum() > 0:
        axes2[1, 1].hist(prob_c[mask], bins=50, alpha=0.6, color=col, label=cls, density=True)
axes2[1, 1].axvline(0.5, color='black', linestyle='--', label='Seuil=0.5')
axes2[1, 1].set_title('Distribution P(Critique)')
axes2[1, 1].set_yscale('log')
axes2[1, 1].legend()

shap_global = np.mean([np.abs(shap_values[i]).mean(axis=0) for i in range(3)], axis=0)
shap_df_global = pd.Series(shap_global, index=FEATURES).sort_values(ascending=False)
top_g = min(12, len(shap_df_global))
axes2[1, 2].barh(shap_df_global.index[:top_g], shap_df_global.values[:top_g], color='purple')
axes2[1, 2].set_title('SHAP Global - Toutes Classes')
axes2[1, 2].invert_yaxis()

plt.tight_layout()
plt.savefig('mlp_shap_fig2.png', dpi=150, bbox_inches='tight')
print("mlp_shap_fig2.png sauvegarde")
plt.close()

# sauvegarde
print("Sauvegarde des fichiers...")
joblib.dump(model, 'model_mlp_shap.pkl')
joblib.dump(scaler, 'scaler_mlp_shap.pkl')
pd.DataFrame([study.best_params]).to_csv('best_params_mlp_optuna.csv', index=False)

pd.DataFrame({
    'Metrique': ['Train_Acc', 'Test_Acc', 'Diff', 'Train_F1', 'Test_F1', 'ROC_AUC', 'Best_F1_CV'],
    'Valeur': [acc_train, acc_test, abs(acc_train - acc_test), f1_train, f1_test, roc_auc, study.best_value]
}).to_csv('resultats_mlp_shap.csv', index=False)

pd.Series(shap_global, index=FEATURES).sort_values(ascending=False).to_csv(
    'feature_importance_mlp_shap.csv', header=['SHAP_global']
)

print("Fichiers generes : model_mlp_shap.pkl, scaler_mlp_shap.pkl, best_params_mlp_optuna.csv,",
      "resultats_mlp_shap.csv, feature_importance_mlp_shap.csv, mlp_shap_fig1.png, mlp_shap_fig2.png")
print("Temps total :", round((time.time() - t0) / 60, 1), "min")
