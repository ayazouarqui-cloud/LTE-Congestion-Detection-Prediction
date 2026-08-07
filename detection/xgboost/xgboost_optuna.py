import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder
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
import sys
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 65)
print("  XGBOOST AJUSTÉ + 3 TESTS OVERFITTING  |  PFE M2 Big Data")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# [1/9] Chargement
# ─────────────────────────────────────────────────────────────
print("\n[1/9] Chargement des données...")
t0 = time.time()
df = pd.read_csv("df_avec_score_kmeans.csv")
print(f"    {len(df):,} lignes | {df.shape[1]} colonnes | {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────
# [2/9] Préparation
# ─────────────────────────────────────────────────────────────
print("\n[2/9] Préparation...")

FEATURES = [
    'LTE_Setup_Success_Rate','Cell_Traffic_Volume_DL', 
    'Cell_Traffic_Volume_Ul','DL_Average_Throughput', 
    'Ul_Average_Throughput','Avg_User_NB', 'Avaibility',
    'HOUR', 'Spectral_Eff', 'Rolling_PRB_3h','PRB_Z_Score', 'Gradient_PRB'
]

TARGET = 'Classe_Congestion'

# Encodage
le = LabelEncoder()
df[TARGET] = le.fit_transform(df[TARGET])
print(f"    Classes : {list(le.classes_)}")

X = df[FEATURES]
y = df[TARGET]

# Split stratifié 70/15/15
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"    Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}")

# ─────────────────────────────────────────────────────────────
# [3/9] Entraînement avec paramètres ajustés
# ─────────────────────────────────────────────────────────────
print("\n[3/9] Entraînement XGBoost paramètres ajustés...")

params = {
    'n_estimators'      : 568,
    'max_depth'         : 6,
    'learning_rate'     : 0.05,
    'subsample'         : 0.6,
    'colsample_bytree'  : 0.7,
    'min_child_weight'  : 50,
    'gamma'             : 1.0,
    'reg_alpha'         : 0.5,
    'reg_lambda'        : 0.3,
    'objective'         : 'multi:softprob',
    'num_class'         : 3,
    'eval_metric'       : 'mlogloss',
    'tree_method'       : 'hist',
    'device'            : 'cpu',
    'random_state'      : 42,
    'n_jobs'            : -1,
    'verbosity'         : 0,
    # AJOUT ICI pour XGBoost 2.0+
    'early_stopping_rounds' : 50
}

t1 = time.time()
# On passe early_stopping_rounds à l'initialisation, pas au fit()
model = xgb.XGBClassifier(**params)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)
print(f"    Terminé en {time.time()-t1:.1f}s")
# Pour récupérer la meilleure itération en version 2.0+
print(f"    Meilleur round : {model.best_iteration}")

# ─────────────────────────────────────────────────────────────
# [4/9] Évaluation standard
# ─────────────────────────────────────────────────────────────
print("\n[4/9] Évaluation standard...")

y_pred       = model.predict(X_test)
y_pred_prob  = model.predict_proba(X_test)
y_train_pred = model.predict(X_train)

acc_test  = accuracy_score(y_test,  y_pred)
acc_train = accuracy_score(y_train, y_train_pred)
f1_test   = f1_score(y_test,  y_pred,       average='macro')
f1_train  = f1_score(y_train, y_train_pred, average='macro')
roc_auc   = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')

print(f"""
    ┌──────────────────────────────────────────────┐
    │  XGBOOST AJUSTÉ — Résultats finaux           │
    │  Train Accuracy : {acc_train:.4f}            │
    │  Test  Accuracy : {acc_test:.4f}             │
    │  Différence     : {abs(acc_train-acc_test):.4f}  │
    │  Train F1-macro : {f1_train:.4f}             │
    │  Test  F1-macro : {f1_test:.4f}              │
    │  ROC-AUC        : {roc_auc:.4f}              │
    └──────────────────────────────────────────────┘
""")
# Conversion explicite des classes en chaînes de caractères
target_names_str = [str(c) for c in le.classes_]
print(classification_report(y_test, y_pred, target_names=target_names_str))

# ─────────────────────────────────────────────────────────────
# ██████████████████████████████████████████████████████████
# TEST 1 : COMPARAISON TRAIN vs TEST
# ██████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  TEST 1 : COMPARAISON TRAIN vs TEST")
print("="*65)

diff_acc = abs(acc_train - acc_test)
diff_f1  = abs(f1_train  - f1_test)

print(f"""
    Train Accuracy : {acc_train:.4f}
    Test  Accuracy : {acc_test:.4f}
    Différence Acc : {diff_acc:.4f}

    Train F1-macro : {f1_train:.4f}
    Test  F1-macro : {f1_test:.4f}
    Différence F1  : {diff_f1:.4f}
""")

if diff_acc < 0.005:
    print("    ✅ TEST 1 PASSÉ — Pas d'overfitting détecté")
    print("       (différence < 0.005 → modèle bien généralisé)")
elif diff_acc < 0.02:
    print("    ⚠️  TEST 1 ATTENTION — Légère différence")
    print("       (différence entre 0.005 et 0.02)")
else:
    print("    ❌ TEST 1 ÉCHOUÉ — Overfitting probable")
    print("       (différence > 0.02 → réduire max_depth)")

# ─────────────────────────────────────────────────────────────
# ██████████████████████████████████████████████████████████
# TEST 2 : DATA LEAKAGE — Corrélation PRB_per_User vs Cible
# ██████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  TEST 2 : DÉTECTION DATA LEAKAGE")
print("="*65)

print("\n  2.1 — Corrélation features vs cible :")
correlations = {}
for feat in FEATURES:
    try:
        corr = abs(df[feat].corr(df[TARGET]))
        correlations[feat] = corr
    except:
        correlations[feat] = 0

corr_series = pd.Series(correlations).sort_values(ascending=False)
print(corr_series.head(10).to_string())

# Seuil critique
leakage_features = corr_series[corr_series > 0.95]
if len(leakage_features) > 0:
    print(f"\n    ❌ DATA LEAKAGE DÉTECTÉ sur : {list(leakage_features.index)}")
    print("       Corrélation > 0.95 avec la cible → retirer ces features")
else:
    print("\n    ✅ TEST 2 PASSÉ — Aucun data leakage critique détecté")
    print("       (aucune feature avec corrélation > 0.95)")

print("\n  2.2 — Distribution PRB_per_User par classe :")
prb_stats = df.groupby(TARGET)['PRB_per_User'].agg(['mean','min','max','std'])
prb_stats.index = le.classes_
print(prb_stats.to_string())

# Vérification séparation parfaite
prb_means = df.groupby(TARGET)['PRB_per_User'].mean()
if prb_means.max() / prb_means.min() > 10:
    print("\n    ⚠️  Séparation très forte entre classes sur PRB_per_User")
    print("       Vérifiez que la cible n'est pas dérivée directement de cette feature")
else:
    print("\n    ✅ Séparation normale entre classes sur PRB_per_User")

print("\n  2.3 — Distribution DL_PRB_Usage_Rate par classe :")
prb_usage = df.groupby(TARGET)['DL_PRB_Usage_Rate'].agg(['mean','min','max','std'])
prb_usage.index = le.classes_
print(prb_usage.to_string())

# ─────────────────────────────────────────────────────────────
# ██████████████████████████████████████████████████████████
# TEST 3 : LEARNING CURVE TEMPORELLE
# ██████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  TEST 3 : LEARNING CURVE — Généralisation")
print("="*65)

# Sous-échantillon pour accélérer le calcul
print("    Calcul de la learning curve (sous-échantillon 100k)...")
sample_idx = np.random.choice(len(X_train), 100_000, replace=False)
X_lc = X_train.iloc[sample_idx]
y_lc = y_train.iloc[sample_idx]

# On retire 'verbosity' et 'early_stopping_rounds' du dictionnaire pour éviter les doublons
params_lc = {k: v for k, v in params.items() if k not in ['n_estimators', 'early_stopping_rounds', 'verbosity']}

model_lc = xgb.XGBClassifier(
    **params_lc,
    n_estimators=200,
    verbosity=0  # Défini une seule fois ici
)

train_sizes, train_scores, val_scores = learning_curve(
    model_lc, X_lc, y_lc,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=3,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=0
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

gap_final = abs(train_mean[-1] - val_mean[-1])
print(f"\n    Train F1 final (LC) : {train_mean[-1]:.4f} ± {train_std[-1]:.4f}")
print(f"    Val   F1 final (LC) : {val_mean[-1]:.4f} ± {val_std[-1]:.4f}")
print(f"    Écart Train/Val     : {gap_final:.4f}")

if gap_final < 0.01:
    print("\n    ✅ TEST 3 PASSÉ — Bonne généralisation confirmée")
elif gap_final < 0.03:
    print("\n    ⚠️  TEST 3 ATTENTION — Légère variance Train/Val")
else:
    print("\n    ❌ TEST 3 ÉCHOUÉ — Overfitting confirmé")

# ─────────────────────────────────────────────────────────────
# [5/9] Résumé des 3 tests
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  RÉSUMÉ DES 3 TESTS OVERFITTING")
print("="*65)
print(f"""
    ┌────────────────────────────────────────────────────┐
    │  TEST 1 — Train vs Test                            │
    │  Diff Accuracy : {diff_acc:.4f}                   │
    │  Diff F1-macro : {diff_f1:.4f}                    │
    │  {'✅ PASSÉ' if diff_acc < 0.005 else '⚠️  ATTENTION' if diff_acc < 0.02 else '❌ ÉCHOUÉ'}                                     │
    ├────────────────────────────────────────────────────┤
    │  TEST 2 — Data Leakage                             │
    │  Features suspectes : {len(leakage_features)}                        │
    │  {'✅ PASSÉ' if len(leakage_features) == 0 else '❌ LEAKAGE DÉTECTÉ'}                               │
    ├────────────────────────────────────────────────────┤
    │  TEST 3 — Learning Curve                           │
    │  Écart Train/Val : {gap_final:.4f}                 │
    │  {'✅ PASSÉ' if gap_final < 0.01 else '⚠️  ATTENTION' if gap_final < 0.03 else '❌ ÉCHOUÉ'}                                     │
    └────────────────────────────────────────────────────┘
""")

# ─────────────────────────────────────────────────────────────
# [6/9] Graphiques
# ─────────────────────────────────────────────────────────────
print("\n[6/9] Génération des graphiques...")

fig, axes = plt.subplots(3, 3, figsize=(20, 18))
fig.suptitle('XGBoost Ajusté — Résultats + Tests Overfitting',
             fontsize=16, fontweight='bold')

# ── Graphique 1 : Feature Importance
feat_imp = pd.Series(
    model.feature_importances_, index=FEATURES
).sort_values(ascending=False)

axes[0,0].barh(feat_imp.index[:12], feat_imp.values[:12], color='steelblue')
axes[0,0].set_title('Feature Importance (Top 12)')
axes[0,0].invert_yaxis()

# ── Graphique 2 : Courbe Log-Loss
results = model.evals_result()
axes[0,1].plot(results['validation_0']['mlogloss'],
               color='orange', label='Val Log-Loss')
axes[0,1].axvline(x=model.best_iteration, color='red',
                  linestyle='--', label=f'Best={model.best_iteration}')
axes[0,1].set_title('Courbe Log-Loss (Validation)')
axes[0,1].set_xlabel('Rounds')
axes[0,1].set_ylabel('Log Loss')
axes[0,1].legend()

# ── Graphique 3 : Matrice de Confusion
cm = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(cm_pct, annot=True, fmt='.2f', ax=axes[0,2],
            xticklabels=le.classes_, yticklabels=le.classes_,
            cmap='Blues')
axes[0,2].set_title('Matrice de Confusion (%)')
axes[0,2].set_ylabel('Réel')
axes[0,2].set_xlabel('Prédit')

# ── Graphique 4 : TEST 1 — Train vs Test Bar
metrics = ['Accuracy', 'F1-macro']
train_vals = [acc_train, f1_train]
test_vals  = [acc_test,  f1_test]

x = np.arange(len(metrics))
w = 0.35
axes[1,0].bar(x - w/2, train_vals, w, label='Train', color='steelblue')
axes[1,0].bar(x + w/2, test_vals,  w, label='Test',  color='orange')
axes[1,0].set_xticks(x)
axes[1,0].set_xticklabels(metrics)
axes[1,0].set_ylim([0.995, 1.001])
axes[1,0].set_title('TEST 1 — Train vs Test')
axes[1,0].legend()
axes[1,0].set_ylabel('Score')

# ── Graphique 5 : TEST 2 — Corrélation features
axes[1,1].barh(corr_series.index[:12],
               corr_series.values[:12], color='tomato')
axes[1,1].axvline(x=0.95, color='red', linestyle='--',
                  label='Seuil leakage (0.95)')
axes[1,1].set_title('TEST 2 — Corrélation Features vs Cible')
axes[1,1].invert_yaxis()
axes[1,1].legend()

# ── Graphique 6 : TEST 3 — Learning Curve
axes[1,2].plot(train_sizes, train_mean, 'o-',
               color='steelblue', label='Train F1')
axes[1,2].fill_between(train_sizes,
                        train_mean - train_std,
                        train_mean + train_std,
                        alpha=0.1, color='steelblue')
axes[1,2].plot(train_sizes, val_mean, 'o--',
               color='red', label='Val F1 (CV-3)')
axes[1,2].fill_between(train_sizes,
                        val_mean - val_std,
                        val_mean + val_std,
                        alpha=0.1, color='red')
axes[1,2].set_title('TEST 3 — Learning Curve')
axes[1,2].set_xlabel('Taille Train')
axes[1,2].set_ylabel('F1-macro')
axes[1,2].legend()

# ── Graphique 7 : ROC par classe
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
colors = ['green', 'orange', 'red']
for i, (cls, color) in enumerate(zip(le.classes_, colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    auc_score   = auc(fpr, tpr)
    axes[2,0].plot(fpr, tpr, color=color,
                   label=f'{cls} (AUC={auc_score:.4f})')
axes[2,0].plot([0,1],[0,1],'k--')
axes[2,0].set_title('Courbes ROC par Classe')
axes[2,0].set_xlabel('FPR')
axes[2,0].set_ylabel('TPR')
axes[2,0].legend()

# ── Graphique 8 : Distribution PRB_per_User par classe
for i, (cls, color) in enumerate(zip(le.classes_, colors)):
    data = df[df[TARGET] == i]['PRB_per_User']
    axes[2,1].hist(data.sample(min(10000, len(data))),
                   bins=50, alpha=0.6,
                   label=cls, color=color)
axes[2,1].set_title('TEST 2 — PRB_per_User par Classe')
axes[2,1].set_xlabel('PRB_per_User')
axes[2,1].set_ylabel('Fréquence')
axes[2,1].legend()

# ── Graphique 9 : Distribution DL_PRB_Usage_Rate par classe
for i, (cls, color) in enumerate(zip(le.classes_, colors)):
    data = df[df[TARGET] == i]['DL_PRB_Usage_Rate']
    axes[2,2].hist(data.sample(min(10000, len(data))),
                   bins=50, alpha=0.6,
                   label=cls, color=color)
axes[2,2].set_title('TEST 2 — DL_PRB_Usage_Rate par Classe')
axes[2,2].set_xlabel('DL_PRB_Usage_Rate')
axes[2,2].set_ylabel('Fréquence')
axes[2,2].legend()

plt.tight_layout()
plt.savefig('xgb_ajuste_overfitting_tests.png',
            dpi=150, bbox_inches='tight')
print("    xgb_ajuste_overfitting_tests.png ✓")

# ─────────────────────────────────────────────────────────────
# [7/9] Sauvegarde
# ─────────────────────────────────────────────────────────────
print("\n[7/9] Sauvegarde...")
joblib.dump(model, 'model_xgboost_ajuste.pkl')
joblib.dump(le,    'label_encoder_ajuste.pkl')
feat_imp.to_csv('feature_importance_ajuste.csv')

resultats = pd.DataFrame({
    'Métrique' : ['Train_Acc', 'Test_Acc', 'Diff_Acc',
                  'Train_F1',  'Test_F1',  'Diff_F1', 'ROC_AUC'],
    'Valeur'   : [acc_train, acc_test, diff_acc,
                  f1_train,  f1_test,  diff_f1,  roc_auc]
})
resultats.to_csv('resultats_xgboost_ajuste.csv', index=False)

print("""
  Fichiers produits :
    model_xgboost_ajuste.pkl              ✓
    label_encoder_ajuste.pkl              ✓
    feature_importance_ajuste.csv         ✓
    resultats_xgboost_ajuste.csv          ✓
    xgb_ajuste_overfitting_tests.png      ✓
""")

print("=" * 65)
print(f"  Temps total : {time.time()-t0:.1f}s")
print("=" * 65)