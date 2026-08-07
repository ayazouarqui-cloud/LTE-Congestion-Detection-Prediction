import os
import time
import warnings
import json
import re
import tempfile
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize, LabelEncoder

warnings.filterwarnings('ignore')

t0 = time.time()

# chargement des donnees
print("Chargement des donnees...")
df = pd.read_csv("df_avec_score_kmeans.csv")
print("Dataset charge :", df.shape[0], "lignes,", df.shape[1], "colonnes")

# preparation des variables
# liste de features unifiee (memoir §3.8) - a garder identique dans tous les scripts
FEATURES = [
    'LTE_SETUP_SUCCESS_RATE', 'CELL_TRAFFIC_VOLUME_DL',
    'CELL_TRAFFIC_VOLUME_UL', 'DL_AVERAGE_THROUGHPUT', 'UL_AVERAGE_THROUGHPUT',
    'AVG_USER_NB', 'AVAIBILITY', 'HOUR', 'IS_WEEKEND', 'SPECTRAL_EFF',
    'IS_PEAK_HOUR', 'ROLLING_TRAFIC_3H', 'ROLLING_PRB_3H', 'HOURLY_TREND',
    'ROLLING_MEAN_VOLATILITY', 'PRB_Z_SCORE', 'GRADIENT_PRB'
]
TARGET = 'Classe_Congestion'
CELL_ID_COL = 'CELLNAME_ID'
TIME_COL = 'DATE_' if 'DATE_' in df.columns else 'HOUR'
CLASS_NAMES = ['Normal', 'Modere', 'Critique']

manquantes = [f for f in FEATURES if f not in df.columns]
if manquantes:
    print("attention, features manquantes dans le dataset :", manquantes)
FEATURES = [f for f in FEATURES if f in df.columns]

# encodage de la target si besoin
if set(df[TARGET].unique()) == {0, 1, 2}:
    df[TARGET] = df[TARGET].astype(int)
else:
    le = LabelEncoder()
    df[TARGET] = le.fit_transform(df[TARGET])


def split_chronologique(data, cell_col=CELL_ID_COL, time_col=TIME_COL,
                         train_frac=0.70, val_frac=0.15):
    """
    Split 70/15/15 chronologique, applique cellule par cellule :
    pour chaque CELLNAME_ID on trie par ordre temporel puis on
    coupe en 3 blocs (train = debut, val = milieu, test = fin).
    Evite le data leakage temporel d'un split random classique.
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

# entrainement du modele
print("Entrainement de XGBoost...")
params = {
    'n_estimators': 568,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.60,
    'colsample_bytree': 0.70,
    'min_child_weight': 50,
    'gamma': 1.0,
    'reg_alpha': 0.5,
    'reg_lambda': 0.3,
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'tree_method': 'hist',
    'device': 'cpu',
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
    'early_stopping_rounds': 50
}

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

# evaluation
print("Evaluation du modele...")
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)
y_train_pred = model.predict(X_train)

acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_pred)
f1_train = f1_score(y_train, y_train_pred, average='macro')
f1_test = f1_score(y_test, y_pred, average='macro')
f1_par_classe = f1_score(y_test, y_pred, average=None)
roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')

print("Accuracy - Train:", round(acc_train, 4), "| Test:", round(acc_test, 4))
print("F1 Macro - Train:", round(f1_train, 4), "| Test:", round(f1_test, 4))
print("ROC AUC Test:", round(roc_auc, 4))
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# calcul SHAP
print("Calcul des valeurs SHAP...")
np.random.seed(42)
n_shap_samples = int(np.clip(len(X_test) * 0.005, 1000, 10000))
shap_idx = np.random.choice(len(X_test), n_shap_samples, replace=False)
X_shap = X_test.iloc[shap_idx].reset_index(drop=True)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)

# selon la version de shap le format de sortie change, on uniformise
if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

# graphiques
print("Generation des graphiques...")
colors = ['#2ecc71', '#f39c12', '#e74c3c']

# figure 1 : perf globales
fig1, axes = plt.subplots(2, 3, figsize=(20, 12))
fig1.suptitle('Performances du Modele XGBoost - Congestion LTE', fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(cm_pct, annot=True, fmt='.2f', ax=axes[0, 0], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap='Blues')
axes[0, 0].set_title('Matrice de Confusion (%)')

results = model.evals_result()
axes[0, 1].plot(results['validation_0']['mlogloss'], color='orange', label='Val Log-Loss')
axes[0, 1].axvline(x=model.best_iteration, color='red', linestyle='--', label=f'Best Iter: {model.best_iteration}')
axes[0, 1].set_title('Evolution Log-Loss')
axes[0, 1].legend()

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
for i, (cls, col) in enumerate(zip(CLASS_NAMES, colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    axes[0, 2].plot(fpr, tpr, color=col, label=f'{cls} (AUC={auc(fpr, tpr):.4f})')
axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0, 2].set_title('Courbes ROC')
axes[0, 2].legend()

metrics_names = ['Accuracy', 'F1-macro', 'F1-Normal', 'F1-Modere', 'F1-Critique']
f1_train_cls = f1_score(y_train, y_train_pred, average=None)
train_metrics = [acc_train, f1_train, f1_train_cls[0], f1_train_cls[1], f1_train_cls[2]]
test_metrics = [acc_test, f1_test, f1_par_classe[0], f1_par_classe[1], f1_par_classe[2]]

x_axis = np.arange(len(metrics_names))
axes[1, 0].bar(x_axis - 0.175, train_metrics, 0.35, label='Train', color='steelblue')
axes[1, 0].bar(x_axis + 0.175, test_metrics, 0.35, label='Test', color='orange')
axes[1, 0].set_xticks(x_axis)
axes[1, 0].set_xticklabels(metrics_names, rotation=15)
axes[1, 0].set_title('Comparaison Train vs Test')
axes[1, 0].legend()

feat_imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
axes[1, 1].barh(feat_imp.index[:12], feat_imp.values[:12], color='steelblue')
axes[1, 1].set_title('Feature Importance (Top 12)')
axes[1, 1].invert_yaxis()

recap_data = [
    ['Accuracy Test', f'{acc_test:.4f}'],
    ['F1-macro Test', f'{f1_test:.4f}'],
    ['ROC-AUC', f'{roc_auc:.4f}'],
    ['Diff Train/Test', f'{abs(acc_train - acc_test):.4f}'],
    ['Erreurs Totales', f'{sum(y_pred != y_test):,}']
]
axes[1, 2].axis('off')
tbl = axes[1, 2].table(cellText=recap_data, colLabels=['Metrique', 'Valeur'], loc='center', cellLoc='center')
tbl.scale(1.0, 1.5)
axes[1, 2].set_title('Resume des Metriques')

plt.tight_layout()
plt.savefig('xgb_shap_fig1.png', dpi=150)
plt.close()

# figure 2 : SHAP
fig2, axes2 = plt.subplots(2, 3, figsize=(22, 14))
fig2.suptitle('Analyse de l\'explicabilite via SHAP', fontsize=14, fontweight='bold')

for i, (cls, col) in enumerate(zip(CLASS_NAMES, colors)):
    shap_mean = np.abs(shap_values[i]).mean(axis=0)
    shap_df = pd.Series(shap_mean, index=FEATURES).sort_values(ascending=False)
    axes2[0, i].barh(shap_df.index[:10], shap_df.values[:10], color=col)
    axes2[0, i].set_title(f'Importance SHAP - Classe {cls}')
    axes2[0, i].invert_yaxis()

# un cas individuel critique, pour illustrer
y_shap_np = np.array(y_test)[shap_idx]
critiques_idx = np.where(y_shap_np == 2)[0]
if len(critiques_idx) > 0:
    idx_sample = critiques_idx[0]
    cell_shap = shap_values[2][idx_sample]
    sorted_idx = np.argsort(np.abs(cell_shap))[-10:]
    feat_names_s = [FEATURES[j] for j in sorted_idx]
    vals_s = cell_shap[sorted_idx]
    bar_colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in vals_s]
    axes2[1, 0].barh(feat_names_s, vals_s, color=bar_colors)
    axes2[1, 0].axvline(0, color='black', linewidth=0.8)
    axes2[1, 0].set_title('Focus SHAP : 1 Cas Critique')

prob_critique = y_pred_prob[:, 2]
for i, (cls, col) in enumerate(zip(CLASS_NAMES, colors)):
    mask = y_test == i
    if mask.sum() > 0:
        axes2[1, 1].hist(prob_critique[mask], bins=50, alpha=0.5, color=col, label=cls, density=True)
axes2[1, 1].axvline(0.5, color='black', linestyle='--')
axes2[1, 1].set_title('Distribution des Probabilites (Critique)')
axes2[1, 1].set_yscale('log')
axes2[1, 1].legend()

shap_global = np.mean([np.abs(shap_values[i]).mean(axis=0) for i in range(3)], axis=0)
shap_global_df = pd.Series(shap_global, index=FEATURES).sort_values(ascending=False)
axes2[1, 2].barh(shap_global_df.index[:12], shap_global_df.values[:12], color='purple')
axes2[1, 2].set_title('Importance Globale SHAP (Toutes classes)')
axes2[1, 2].invert_yaxis()

plt.tight_layout()
plt.savefig('xgb_shap_fig2.png', dpi=150)
plt.close()

# sauvegarde des resultats
print("Sauvegarde des fichiers de sortie...")
joblib.dump(model, 'model_xgb_shap.pkl')
feat_imp.to_csv('feature_importance_xgb_shap.csv')
shap_global_df.to_csv('shap_importance_globale.csv', header=['SHAP_moyen'])

pd.DataFrame({
    'Metrique': ['Train_Acc', 'Test_Acc', 'Diff_Acc', 'Train_F1', 'Test_F1', 'ROC_AUC'],
    'Valeur': [acc_train, acc_test, abs(acc_train - acc_test), f1_train, f1_test, roc_auc]
}).to_csv('resultats_xgb_shap.csv', index=False)

print("Fichiers generes : model_xgb_shap.pkl, feature_importance_xgb_shap.csv, shap_importance_globale.csv, resultats_xgb_shap.csv, xgb_shap_fig1.png, xgb_shap_fig2.png")
print("Traitement termine en", round((time.time() - t0) / 60, 1), "minutes.")