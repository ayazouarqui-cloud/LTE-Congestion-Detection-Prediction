import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize
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
print("  MLP NEURAL NETWORK + OPTUNA  |  PFE M2 Big Data")
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

# Split stratifié
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"    Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}")

# Normalisation obligatoire pour MLP
print("    Normalisation StandardScaler...")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────
# [3/8] Optuna — sous-échantillon 200k
# ─────────────────────────────────────────────────────────────
print("\n[3/8] Optuna — Optimisation Bayésienne (30 trials)...")
print("    Sous-échantillon : 200,000 lignes")

sample_idx = np.random.choice(len(X_train_sc), 200_000, replace=False)
X_opt = X_train_sc[sample_idx]
y_opt = y_train.iloc[sample_idx]

def objective_mlp(trial):

    # Architecture du réseau
    n_layers = trial.suggest_int('n_layers', 2, 4)
    layers   = tuple([
        trial.suggest_int(f'n_units_l{i}', 64, 512)
        for i in range(n_layers)
    ])

    params = {
        'hidden_layer_sizes' : layers,
        'activation'         : trial.suggest_categorical(
                                'activation', ['relu', 'tanh']),
        'solver'             : 'adam',
        'alpha'              : trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
        'learning_rate_init' : trial.suggest_float('learning_rate_init',
                                                    1e-4, 1e-2, log=True),
        'batch_size'         : trial.suggest_categorical(
                                'batch_size', [256, 512, 1024]),
        'max_iter'           : 100,
        'early_stopping'     : True,
        'validation_fraction': 0.1,
        'n_iter_no_change'   : 10,
        'random_state'       : 42,
    }

    model = MLPClassifier(**params)
    scores = cross_val_score(
        model, X_opt, y_opt,
        cv=3, scoring='f1_macro', n_jobs=-1
    )
    return scores.mean()

t1 = time.time()
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_mlp = optuna.create_study(direction='maximize')
study_mlp.optimize(objective_mlp, n_trials=30, show_progress_bar=True)

print(f"\n    Optuna terminé en {time.time()-t1:.1f}s")
print(f"    Meilleur F1-macro (CV) : {study_mlp.best_value:.4f}")
print(f"\n    Meilleurs hyperparamètres :")
for k, v in study_mlp.best_params.items():
    print(f"       {k:<25} = {v}")

# ─────────────────────────────────────────────────────────────
# [4/8] Reconstruction architecture
# ─────────────────────────────────────────────────────────────
print("\n[4/8] Entraînement final...")

best = study_mlp.best_params
n_layers = best['n_layers']
layers   = tuple([best[f'n_units_l{i}'] for i in range(n_layers)])

final_params = {
    'hidden_layer_sizes' : layers,
    'activation'         : best['activation'],
    'solver'             : 'adam',
    'alpha'              : best['alpha'],
    'learning_rate_init' : best['learning_rate_init'],
    'batch_size'         : best['batch_size'],
    'max_iter'           : 300,
    'early_stopping'     : True,
    'validation_fraction': 0.1,
    'n_iter_no_change'   : 20,
    'random_state'       : 42,
    'verbose'            : True,
}

print(f"    Architecture : {layers}")
t2 = time.time()
model_mlp = MLPClassifier(**final_params)
model_mlp.fit(X_train_sc, y_train)
print(f"    Terminé en {time.time()-t2:.1f}s")
print(f"    Itérations : {model_mlp.n_iter_}")

# ─────────────────────────────────────────────────────────────
# [5/8] Évaluation
# ─────────────────────────────────────────────────────────────
print("\n[5/8] Évaluation...")

y_pred       = model_mlp.predict(X_test_sc)
y_pred_prob  = model_mlp.predict_proba(X_test_sc)
y_train_pred = model_mlp.predict(X_train_sc)

acc_test  = accuracy_score(y_test,  y_pred)
acc_train = accuracy_score(y_train, y_train_pred)
f1_test   = f1_score(y_test,  y_pred,       average='macro')
f1_train  = f1_score(y_train, y_train_pred, average='macro')
roc_auc   = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')

print(f"""
    ┌─────────────────────────────────────────────┐
    │  MLP NEURAL NETWORK — Résultats finaux      │
    │  Architecture      : {layers}
    │  Train Accuracy    : {acc_train:.4f}        │
    │  Test  Accuracy    : {acc_test:.4f}         │
    │  Différence        : {abs(acc_train-acc_test):.4f}    │
    │  Train F1-macro    : {f1_train:.4f}         │
    │  Test  F1-macro    : {f1_test:.4f}          │
    │  ROC-AUC           : {roc_auc:.4f}          │
    └─────────────────────────────────────────────┘
""")

target_names_str = [str(c) for c in le.classes_]
print(classification_report(y_test, y_pred, target_names=target_names_str))

# ─────────────────────────────────────────────────────────────
# [6/8] Graphiques
# ─────────────────────────────────────────────────────────────
print("\n[6/8] Génération des graphiques...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('MLP Neural Network + Optuna — Résultats',
             fontsize=16, fontweight='bold')

colors = ['green', 'orange', 'red']

# 1. Matrice de confusion
cm     = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(cm_pct, annot=True, fmt='.2f', ax=axes[0,0],
            xticklabels=le.classes_, yticklabels=le.classes_,
            cmap='Purples')
axes[0,0].set_title('Matrice de Confusion (%)')
axes[0,0].set_ylabel('Réel')
axes[0,0].set_xlabel('Prédit')

# 2. Courbe de loss MLP
axes[0,1].plot(model_mlp.loss_curve_, color='purple', label='Train Loss')
if hasattr(model_mlp, 'validation_scores_'):
    axes[0,1].plot(model_mlp.validation_scores_,
                   color='red', linestyle='--', label='Val Score')
axes[0,1].set_title('Courbe de Loss MLP')
axes[0,1].set_xlabel('Itérations')
axes[0,1].set_ylabel('Loss')
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
axes[1,0].bar(x - w/2, train_vals, w, label='Train', color='purple')
axes[1,0].bar(x + w/2, test_vals,  w, label='Test',  color='orange')
axes[1,0].set_xticks(x)
axes[1,0].set_xticklabels(metrics)
axes[1,0].set_ylim([0.98, 1.001])
axes[1,0].set_title('Train vs Test — Overfitting Check')
axes[1,0].legend()

# 5. Convergence Optuna
trial_values = [t.value for t in study_mlp.trials
                if t.value is not None]
best_values  = [max(trial_values[:i+1])
                for i in range(len(trial_values))]
axes[1,1].scatter(range(len(trial_values)), trial_values,
                  alpha=0.5, color='purple', label='F1 trial')
axes[1,1].plot(range(len(best_values)), best_values,
               color='red', label=f'Meilleur={study_mlp.best_value:.4f}')
axes[1,1].set_title('Convergence Optuna (30 trials)')
axes[1,1].set_xlabel('Trial #')
axes[1,1].set_ylabel('F1-macro (CV-3)')
axes[1,1].legend()

# 6. Récapitulatif
recap_data = {
    'Métrique': ['Accuracy', 'F1-macro', 'ROC-AUC',
                 'Diff Train/Test', 'Erreurs', 'Architecture'],
    'Valeur':   [f'{acc_test:.4f}', f'{f1_test:.4f}',
                 f'{roc_auc:.4f}',
                 f'{abs(acc_train-acc_test):.4f}',
                 f'{sum(y_pred != y_test):,}',
                 str(layers)]
}
axes[1,2].axis('off')
table = axes[1,2].table(
    cellText=[[r, v] for r, v in zip(recap_data['Métrique'],
                                      recap_data['Valeur'])],
    colLabels=['Métrique', 'Valeur'],
    loc='center', cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2.0)
axes[1,2].set_title('Récapitulatif Final', fontweight='bold')

plt.tight_layout()
plt.savefig('mlp_optuna_resultats.png', dpi=150, bbox_inches='tight')
print("    mlp_optuna_resultats.png ✓")

# ─────────────────────────────────────────────────────────────
# [7/8] Sauvegarde
# ─────────────────────────────────────────────────────────────
print("\n[7/8] Sauvegarde...")
joblib.dump(model_mlp, 'model_mlp_optuna.pkl')
joblib.dump(scaler,    'scaler_mlp_optuna.pkl')
joblib.dump(le,        'label_encoder_mlp.pkl')

resultats = pd.DataFrame({
    'Métrique': ['Train_Acc', 'Test_Acc', 'Diff_Acc',
                 'Train_F1',  'Test_F1',  'ROC_AUC',
                 'Architecture'],
    'Valeur':   [acc_train, acc_test, abs(acc_train-acc_test),
                 f1_train,  f1_test,  roc_auc, str(layers)]
})
resultats.to_csv('resultats_mlp_optuna.csv', index=False)

pd.DataFrame([study_mlp.best_params]).to_csv(
    'best_params_mlp_optuna.csv', index=False
)

print("""
  Fichiers produits :
    model_mlp_optuna.pkl              ✓
    scaler_mlp_optuna.pkl             ✓
    label_encoder_mlp.pkl             ✓
    resultats_mlp_optuna.csv          ✓
    best_params_mlp_optuna.csv        ✓
    mlp_optuna_resultats.png          ✓
""")
print(f"  Temps total : {time.time()-t0:.1f}s")
print("=" * 65)