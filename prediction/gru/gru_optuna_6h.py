import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# CONFIGURATION — GRU target_6h
# ============================================================

TARGET         = 'target_6h'
SEQ_LEN        = 6
N_CLASSES      = 3
BATCH_SIZE     = 2048
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_TRIALS       = 20
OPTUNA_SAMPLE  = 200_000
OPTUNA_EPOCHS  = 5
FINAL_EPOCHS   = 30
FINAL_PATIENCE = 5

# Résultats de référence (pour tableau comparatif final)
REF = {
    'LSTM +1h' : {'acc': 0.9006, 'f1': 0.9139, 'f1cls2': 0.9634},
    'GRU  +3h' : {'acc': None,   'f1': None,   'f1cls2': None  },
    # ← remplis manuellement si tu as les résultats 3h
}

print("=" * 60)
print(f"  GRU + OPTUNA  —  {TARGET}")
print("=" * 60)
print(f"Device            : {DEVICE}")
print(f"Trials Optuna     : {N_TRIALS}")
print(f"Lignes par trial  : {OPTUNA_SAMPLE:,}")
print(f"Epochs par trial  : {OPTUNA_EPOCHS}")
print(f"Réentraînement    : {FINAL_EPOCHS} epochs sur données complètes")
print(f"Fenêtre séquence  : {SEQ_LEN}h d'historique")
print(f"\nGRU vs LSTM : 2 gates (reset + update) vs 3 gates (input + forget + output)")
print(f"             GRU = ~25% moins de paramètres, convergence souvent plus rapide")
print(f"\nHorizon 6h = tâche la plus difficile : prédire 6h à l'avance")
print(f"           → dégradation attendue des métriques vs 1h et 3h")

# ============================================================
# ÉTAPE 1 — CHARGEMENT ET NETTOYAGE
# ============================================================

print("\n[1/8] Chargement et nettoyage...")

df = pd.read_csv("dataset_avec_targets.csv")
df.columns = df.columns.str.lower()
df['date_'] = pd.to_datetime(df['date_'].astype(str).str.strip(), errors='coerce')
df = df.sort_values(['cellname_id', 'date_']).reset_index(drop=True)

print(f"  Lignes brutes : {len(df):,}")
print(f"  Colonnes      : {list(df.columns)}")

COLS_DROP = [
    'time_to_peak', 'peak_trend_interaction', 'traffic_per_user',
    'prb_z_score', 'prb_per_user', 'throughput_per_user'
]
dropped = [c for c in COLS_DROP if c in df.columns]
df.drop(columns=dropped, inplace=True)
print(f"  Colonnes supprimées ({len(dropped)}) : {dropped}")

if 'hour' in df.columns:
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df.drop(columns=['hour'], inplace=True)
    print("  Encodage cyclique heure : hour → hour_sin, hour_cos")

df['lag_prb_1h'] = df.groupby('cellname_id')['dl_prb_usage_rate'].shift(1)
df['lag_prb_2h'] = df.groupby('cellname_id')['dl_prb_usage_rate'].shift(2)
df.dropna(subset=['lag_prb_1h', 'lag_prb_2h', TARGET], inplace=True)

print(f"\nDataset final : {len(df):,} lignes | Cellules : {df['cellname_id'].nunique():,}")

# ============================================================
# ÉTAPE 2 — FEATURES ET SPLIT TEMPOREL 70/15/15
# ============================================================

print("\n[2/8] Features et split temporel (70 / 15 / 15)...")

EXCLURE  = ['date_', 'cellname_id', 'target_1h', 'target_3h',
            'target_6h', 'classe_congestion', 'congestion_score']
FEATURES = [c for c in df.columns if c not in EXCLURE]

print(f"\n  Features retenues ({len(FEATURES)}) :")
for i, f in enumerate(FEATURES):
    print(f"    [{i+1:02d}] {f}")

dates = df['date_'].dropna().sort_values()
d70   = dates.quantile(0.70)
d85   = dates.quantile(0.85)
print(f"\n  Découpe temporelle :")
print(f"    Train  : debut → {d70.date()}  (70%)")
print(f"    Val    : {d70.date()} → {d85.date()}  (15%)")
print(f"    Test   : {d85.date()} → fin  (15%)")

train_df = df[df['date_'] <= d70].copy()
val_df   = df[(df['date_'] > d70) & (df['date_'] <= d85)].copy()
test_df  = df[df['date_'] > d85].copy()

print(f"\n  Train : {len(train_df):,} lignes")
print(f"  Val   : {len(val_df):,} lignes")
print(f"  Test  : {len(test_df):,} lignes")

# Distribution du target
dist_tr = train_df[TARGET].value_counts(normalize=True).sort_index()
print(f"\n  Distribution target_6h (train) :")
for cls, pct in dist_tr.items():
    label = {0: 'Normal', 1: 'Modéré', 2: 'Congestionné'}[cls]
    print(f"    Classe {cls} ({label:>13}) : {pct:.2%}")

# ============================================================
# ÉTAPE 3 — NORMALISATION ET CONSTRUCTION DES SÉQUENCES
# ============================================================

print("\n[3/8] Normalisation StandardScaler + construction séquences...")

scaler = StandardScaler()
X_tr   = scaler.fit_transform(train_df[FEATURES])
X_va   = scaler.transform(val_df[FEATURES])
X_te   = scaler.transform(test_df[FEATURES])

y_tr   = train_df[TARGET].values.astype(int)
y_va   = val_df[TARGET].values.astype(int)
y_te   = test_df[TARGET].values.astype(int)

print(f"\n  Shapes AVANT sliding window (tableaux 2D) :")
print(f"    X_train : {X_tr.shape}  →  [N_lignes, {len(FEATURES)} features]")
print(f"    X_val   : {X_va.shape}")
print(f"    X_test  : {X_te.shape}")
print(f"    y_train : {y_tr.shape}")
print(f"    y_val   : {y_va.shape}")
print(f"    y_test  : {y_te.shape}")

def make_sequences(X, y, seq_len):
    """
    Sliding window : [N, features] → [N-seq_len, seq_len, features]

    Exemple avec seq_len=24 :
      Séquence 0 → X[0:24]   prédit y[24]
      Séquence 1 → X[1:25]   prédit y[25]
      ...
      Séquence k → X[k:k+24] prédit y[k+24]

    Chaque séquence = 24 heures consécutives d'historique
    Target = état de congestion 6h après la fin de la séquence
    """
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)

print(f"\n  Construction sliding window (seq_len={SEQ_LEN}h)...")
print(f"  Logique : chaque séquence = {SEQ_LEN}h d'historique → prédire +6h")

print("    Reshape train...", end=' ', flush=True)
X_tr_s, y_tr_s = make_sequences(X_tr, y_tr, SEQ_LEN)
print(f"done")

print("    Reshape val...", end=' ', flush=True)
X_va_s, y_va_s = make_sequences(X_va, y_va, SEQ_LEN)
print(f"done")

print("    Reshape test...", end=' ', flush=True)
X_te_s, y_te_s = make_sequences(X_te, y_te, SEQ_LEN)
print(f"done")

print(f"\n  Shapes APRÈS sliding window (tenseurs 3D) :")
print(f"    X_train : {X_tr_s.shape}  →  [N_séquences, {SEQ_LEN}h, {len(FEATURES)} features]")
print(f"    X_val   : {X_va_s.shape}")
print(f"    X_test  : {X_te_s.shape}")
print(f"    y_train : {y_tr_s.shape}  →  [N_séquences]")
print(f"    y_val   : {y_va_s.shape}")
print(f"    y_test  : {y_te_s.shape}")

print(f"\n  Mémoire estimée :")
mem_tr = X_tr_s.nbytes / 1024**3
mem_va = X_va_s.nbytes / 1024**3
mem_te = X_te_s.nbytes / 1024**3
print(f"    X_train : {mem_tr:.2f} GB")
print(f"    X_val   : {mem_va:.2f} GB")
print(f"    X_test  : {mem_te:.2f} GB")
print(f"    Total   : {mem_tr + mem_va + mem_te:.2f} GB")

# Poids classes (pour gérer le déséquilibre)
weights_arr = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y_tr_s)
weights_t   = torch.tensor(weights_arr, dtype=torch.float32).to(DEVICE)
print(f"\n  Poids classes (équilibrage déséquilibre) :")
labels_noms = {0: 'Normal', 1: 'Modéré', 2: 'Congestionné'}
for i, w in enumerate(weights_arr):
    print(f"    Classe {i} ({labels_noms[i]:>13}) : {w:.3f}")

# ============================================================
# ÉTAPE 4 — SOUS-ÉCHANTILLON STRATIFIÉ OPTUNA
# ============================================================

print(f"\n[4/8] Sous-échantillon stratifié : {OPTUNA_SAMPLE:,} lignes...")
print(f"  Stratégie : 55% Normal | 35% Modéré | 10% Congestionné")
print(f"  (sur-représentation classe 2 pour guider Optuna)")

rng   = np.random.default_rng(42)
idx_0 = np.where(y_tr_s == 0)[0]
idx_1 = np.where(y_tr_s == 1)[0]
idx_2 = np.where(y_tr_s == 2)[0]

n0 = int(OPTUNA_SAMPLE * 0.55)
n1 = int(OPTUNA_SAMPLE * 0.35)
n2 = OPTUNA_SAMPLE - n0 - n1

sample_idx = np.sort(np.concatenate([
    rng.choice(idx_0, min(n0, len(idx_0)), replace=False),
    rng.choice(idx_1, min(n1, len(idx_1)), replace=False),
    rng.choice(idx_2, min(n2, len(idx_2)), replace=False),
]))
X_opt, y_opt = X_tr_s[sample_idx], y_tr_s[sample_idx]

val_idx  = np.sort(rng.choice(len(X_va_s), min(50_000, len(X_va_s)), replace=False))
X_va_opt = X_va_s[val_idx]
y_va_opt = y_va_s[val_idx]

print(f"\n  Shapes sous-échantillon Optuna :")
print(f"    X_opt    : {X_opt.shape}   ← train Optuna")
print(f"    X_va_opt : {X_va_opt.shape}  ← val Optuna")
print(f"    y_opt    : {y_opt.shape}")

dist_opt = {i: f"{np.mean(y_opt == i):.1%}" for i in range(3)}
print(f"\n  Distribution classes optuna train :")
print(f"    0=Normal : {dist_opt[0]} | 1=Modéré : {dist_opt[1]} | 2=Congestionné : {dist_opt[2]}")

opt_tr_ld  = DataLoader(TensorDataset(torch.from_numpy(X_opt),    torch.from_numpy(y_opt)),
                         batch_size=1024, shuffle=False)
opt_va_ld  = DataLoader(TensorDataset(torch.from_numpy(X_va_opt), torch.from_numpy(y_va_opt)),
                         batch_size=1024, shuffle=False)
full_tr_ld = DataLoader(TensorDataset(torch.from_numpy(X_tr_s),   torch.from_numpy(y_tr_s)),
                         batch_size=BATCH_SIZE, shuffle=False)
full_va_ld = DataLoader(TensorDataset(torch.from_numpy(X_va_s),   torch.from_numpy(y_va_s)),
                         batch_size=BATCH_SIZE, shuffle=False)
test_ld    = DataLoader(TensorDataset(torch.from_numpy(X_te_s),   torch.from_numpy(y_te_s)),
                         batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# ÉTAPE 5 — ARCHITECTURE GRU
# ============================================================
#
#  Schéma du flux de données :
#
#  Input [batch, 24, n_features]
#        ↓
#  ┌─────────────────────────────────────────┐
#  │  GRU (num_layers, hidden_size)          │
#  │  • Reset gate  : z_t = σ(W_z · [h,x])  │
#  │  • Update gate : r_t = σ(W_r · [h,x])  │
#  │  • No cell state (contrairement LSTM)   │
#  └─────────────────────────────────────────┘
#        ↓  out[:, -1, :]  (dernier pas de temps)
#  ┌─────────────────────────────────────────┐
#  │  LayerNorm(hidden_size)                 │
#  └─────────────────────────────────────────┘
#        ↓
#  ┌─────────────────────────────────────────┐
#  │  Dropout(p)                             │
#  └─────────────────────────────────────────┘
#        ↓
#  ┌─────────────────────────────────────────┐
#  │  Linear(hidden_size → fc_size) + ReLU  │
#  └─────────────────────────────────────────┘
#        ↓
#  ┌─────────────────────────────────────────┐
#  │  Dropout(p)                             │
#  └─────────────────────────────────────────┘
#        ↓
#  ┌─────────────────────────────────────────┐
#  │  Linear(fc_size → 3)                   │
#  └─────────────────────────────────────────┘
#        ↓
#  Output [batch, 3]  →  argmax → classe {0, 1, 2}

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, fc_size):
        super().__init__()
        self.gru  = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Sequential(
            nn.Linear(hidden_size, fc_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_size, N_CLASSES)
        )

    def forward(self, x):
        # x    : [batch, 24, n_features]
        out, _ = self.gru(x)       # out : [batch, 24, hidden_size]
        last   = out[:, -1, :]     # dernier pas : [batch, hidden_size]
        last   = self.norm(last)
        last   = self.drop(last)
        return self.fc(last)       # [batch, 3]

# ============================================================
# ÉTAPE 6 — OPTUNA
# ============================================================

print(f"\n[5/8] Optuna — {N_TRIALS} trials × {OPTUNA_EPOCHS} epochs...")
print(f"      Espace de recherche :")
print(f"        hidden_size  : [64, 128, 256]")
print(f"        num_layers   : [1, 2, 3]")
print(f"        dropout      : [0.1, 0.2, 0.3, 0.4, 0.5]")
print(f"        fc_size      : [32, 64, 128]")
print(f"        lr           : [1e-4, 5e-3]  (log)")
print(f"        weight_decay : [1e-5, 1e-3]  (log)")

def objective(trial):
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    num_layers  = trial.suggest_int('num_layers', 1, 3)
    dropout     = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    fc_size     = trial.suggest_categorical('fc_size', [32, 64, 128])
    lr          = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    wd          = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

    model = GRUModel(len(FEATURES), hidden_size, num_layers,
                     dropout, fc_size).to(DEVICE)
    crit  = nn.CrossEntropyLoss(weight=weights_t)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_f1 = 0.0
    for epoch in range(OPTUNA_EPOCHS):
        model.train()
        for xb, yb in opt_tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in opt_va_ld:
                preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
        f1 = f1_score(y_va_opt, preds, average='macro', zero_division=0)

        trial.report(f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        if f1 > best_f1:
            best_f1 = f1

    return best_f1


def print_cb(study, trial):
    if trial.value is not None:
        print(f"  Trial {trial.number:2d} | F1={trial.value:.4f} | "
              f"Best={study.best_value:.4f} | Params={trial.params}")

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
)
study.optimize(objective, n_trials=N_TRIALS, callbacks=[print_cb])

print(f"\n{'='*60}")
print(f"✅ Optuna terminé !")
print(f"   Meilleur F1-macro val : {study.best_value:.4f}")
print(f"   Meilleurs params      : {study.best_params}")
print(f"{'='*60}")

# ============================================================
# ÉTAPE 7 — RÉENTRAÎNEMENT FINAL SUR DONNÉES COMPLÈTES
# ============================================================

print(f"\n[6/8] Réentraînement final — données complètes...")

bp = study.best_params
best_model = GRUModel(
    input_size  = len(FEATURES),
    hidden_size = bp['hidden_size'],
    num_layers  = bp['num_layers'],
    dropout     = bp['dropout'],
    fc_size     = bp['fc_size']
).to(DEVICE)

n_params = sum(p.numel() for p in best_model.parameters() if p.requires_grad)

print(f"\n  Architecture finale :")
print(f"    GRU       : {bp['num_layers']} couche(s) × {bp['hidden_size']} hidden units")
print(f"    FC        : Dense({bp['fc_size']}) → ReLU → Dense(3)")
print(f"    Dropout   : {bp['dropout']}")
print(f"    Paramètres totaux : {n_params:,}")

# Comparaison théorique LSTM vs GRU (même hidden_size)
h, f_in = bp['hidden_size'], len(FEATURES)
lstm_params_th = 4 * (h * f_in + h * h + h)
gru_params_th  = 3 * (h * f_in + h * h + h)
print(f"\n  Comparaison paramètres couche récurrente (hidden={h}) :")
print(f"    LSTM (4 matrices) : ~{lstm_params_th:,} params")
print(f"    GRU  (3 matrices) : ~{gru_params_th:,} params  ({(1-gru_params_th/lstm_params_th):.0%} de moins)")

print(f"\n  Shapes entrée modèle (batch_size=64 exemple) :")
print(f"    Input  : [64, {SEQ_LEN}, {len(FEATURES)}]  →  [batch, seq_len, features]")
print(f"    GRU    : [64, {SEQ_LEN}, {bp['hidden_size']}]  →  [batch, seq_len, hidden]")
print(f"    last   : [64, {bp['hidden_size']}]         →  [batch, hidden]  (dernier pas)")
print(f"    FC     : [64, {bp['fc_size']}]         →  [batch, fc_size]")
print(f"    Output : [64, 3]             →  [batch, n_classes]")

criterion = nn.CrossEntropyLoss(weight=weights_t)
optimizer = torch.optim.Adam(best_model.parameters(),
                              lr=bp['lr'], weight_decay=bp['weight_decay'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=3, factor=0.5, verbose=False)

best_f1_val = 0.0
patience_ct = 0
history     = {'loss': [], 'val_f1': [], 'val_f1_cls2': []}

print(f"\n  Entraînement (max {FINAL_EPOCHS} epochs, early stopping patience={FINAL_PATIENCE}) :")

for epoch in range(1, FINAL_EPOCHS + 1):

    best_model.train()
    total_loss = 0
    for xb, yb in full_tr_ld:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(best_model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(best_model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)

    avg_loss = total_loss / len(y_tr_s)

    best_model.eval()
    preds_v = []
    with torch.no_grad():
        for xb, _ in full_va_ld:
            preds_v.extend(best_model(xb.to(DEVICE)).argmax(1).cpu().numpy())

    val_f1      = f1_score(y_va_s, preds_v, average='macro', zero_division=0)
    val_f1_cls2 = f1_score(y_va_s, preds_v, average=None,    zero_division=0)[2]
    scheduler.step(val_f1)

    history['loss'].append(avg_loss)
    history['val_f1'].append(val_f1)
    history['val_f1_cls2'].append(val_f1_cls2)

    status = ""
    if val_f1 > best_f1_val:
        best_f1_val = val_f1
        torch.save(best_model.state_dict(), 'best_gru_6h_optuna.pt')
        patience_ct = 0
        status = " ✅ saved"
    else:
        patience_ct += 1
        if patience_ct >= FINAL_PATIENCE:
            print(f"  ⏹ Early stopping epoch {epoch}")
            break

    print(f"  Epoch {epoch:3d}/{FINAL_EPOCHS} | Loss: {avg_loss:.4f} | "
          f"Val F1: {val_f1:.4f} | F1-cls2: {val_f1_cls2:.4f}{status}")

# ============================================================
# ÉTAPE 8 — ÉVALUATION FINALE SUR TEST SET
# ============================================================

print(f"\n[7/8] Évaluation sur test set...")
print(f"  Test shape : {X_te_s.shape}  →  [N_séquences, {SEQ_LEN}h, {len(FEATURES)} features]")

best_model.load_state_dict(torch.load('best_gru_6h_optuna.pt', map_location=DEVICE))
best_model.eval()

preds_test = []
with torch.no_grad():
    for xb, _ in test_ld:
        preds_test.extend(best_model(xb.to(DEVICE)).argmax(1).cpu().numpy())
preds_test = np.array(preds_test)

print(f"  Prédictions shape : {preds_test.shape}")
print(f"  Distribution prédictions : " +
      " | ".join([f"cls{i}={np.mean(preds_test==i):.1%}" for i in range(3)]))

print("\n" + "=" * 60)
print(f"RÉSULTATS FINAUX — GRU — {TARGET}")
print("=" * 60)
print(classification_report(
    y_te_s, preds_test,
    target_names=['Normal (0)', 'Modéré (1)', 'Congestionné (2)'],
    digits=4
))

f1_macro = f1_score(y_te_s, preds_test, average='macro',    zero_division=0)
f1_cls2  = f1_score(y_te_s, preds_test, average=None,       zero_division=0)[2]
f1_w     = f1_score(y_te_s, preds_test, average='weighted', zero_division=0)
acc      = (preds_test == y_te_s).mean()

print(f"Accuracy     : {acc:.4f}")
print(f"F1-macro     : {f1_macro:.4f}")
print(f"F1-weighted  : {f1_w:.4f}")
print(f"F1-classe 2  : {f1_cls2:.4f}  ← Congestionné")

# ============================================================
# TABLEAU COMPARATIF MULTI-HORIZONS  (correction bug f-string)
# ============================================================
#
# BUG CORRIGÉ : {val:.4f:>15} → ValueError
# SOLUTION    : formater en str d'abord, puis aligner avec :>15

print("\n" + "=" * 60)
print("COMPARAISON MULTI-HORIZONS — Dégradation avec l'horizon")
print("=" * 60)
print(f"{'Modèle':<16} {'Horizon':>8} {'Accuracy':>10} {'F1-macro':>10} {'F1-cls2':>10} {'Params':>10}")
print("-" * 66)

# Ligne LSTM 1h
print(f"{'LSTM + Optuna':<16} {'+1h':>8} "
      f"{0.9006:>10.4f} {0.9139:>10.4f} {0.9634:>10.4f} {'92,419':>10}")

# Ligne GRU 3h (à remplir avec tes résultats)
gru3_acc = REF['GRU  +3h']['acc']
gru3_f1  = REF['GRU  +3h']['f1']
gru3_f1c = REF['GRU  +3h']['f1cls2']
if gru3_acc is not None:
    print(f"{'GRU + Optuna':<16} {'+3h':>8} "
          f"{gru3_acc:>10.4f} {gru3_f1:>10.4f} {gru3_f1c:>10.4f} {'N/A':>10}")
else:
    print(f"{'GRU + Optuna':<16} {'+3h':>8} "
          f"{'(à remplir)':>10} {'(à remplir)':>10} {'(à remplir)':>10} {'N/A':>10}")

# Ligne GRU 6h (résultats actuels)
acc_str    = f"{acc:.4f}"
f1_str     = f"{f1_macro:.4f}"
f1cls2_str = f"{f1_cls2:.4f}"
npar_str   = f"{n_params:,}"
print(f"{'GRU + Optuna':<16} {'+6h':>8} "
      f"{acc_str:>10} {f1_str:>10} {f1cls2_str:>10} {npar_str:>10}")

print("-" * 66)
print("→ La dégradation des métriques avec l'horizon est normale et attendue")
print("→ F1-cls2 reste la métrique clé (détection congestion)")

# ============================================================
# GRAPHIQUES
# ============================================================

print("\n[8/8] Graphiques...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'GRU + Optuna — {TARGET}  |  Fenêtre={SEQ_LEN}h  |  Features={len(FEATURES)}',
             fontsize=13, fontweight='bold')

# --- 1. Loss entraînement
axes[0, 0].plot(history['loss'], color='steelblue', linewidth=2.5, label='Loss train')
axes[0, 0].set_title('Loss entraînement final', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('CrossEntropy Loss')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# --- 2. F1 validation
axes[0, 1].plot(history['val_f1'],      label='F1-macro',         color='green',  linewidth=2)
axes[0, 1].plot(history['val_f1_cls2'], label='F1-Congestionné',  color='orange', linewidth=2)
axes[0, 1].axhline(f1_macro, color='red', linestyle='--', alpha=0.7,
                    label=f'Test F1-macro: {f1_macro:.4f}')
axes[0, 1].set_title('F1 sur validation', fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('F1-score')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# --- 3. Matrice de confusion
cm     = confusion_matrix(y_te_s, preds_test)
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=['Normal', 'Modéré', 'Congestionné'],
            yticklabels=['Normal', 'Modéré', 'Congestionné'],
            ax=axes[1, 0], cbar=False, annot_kws={'size': 12})
axes[1, 0].set_title(f'Matrice de confusion — GRU {TARGET} (%)', fontweight='bold')
axes[1, 0].set_xlabel('Prédiction')
axes[1, 0].set_ylabel('Réalité')

# --- 4. Comparaison horizons
horizons  = ['+1h\nLSTM', '+6h\nGRU']
f1_macros = [0.9139,  f1_macro]
f1_cls2s  = [0.9634,  f1_cls2]
accs      = [0.9006,  acc]
x = np.arange(len(horizons))
w = 0.25

b1 = axes[1, 1].bar(x - w,   accs,      w, label='Accuracy',     color='steelblue',  alpha=0.85)
b2 = axes[1, 1].bar(x,       f1_macros, w, label='F1-macro',     color='seagreen',   alpha=0.85)
b3 = axes[1, 1].bar(x + w,   f1_cls2s,  w, label='F1-cls2',      color='darkorange', alpha=0.85)

for bars in [b1, b2, b3]:
    for bar in bars:
        h_val = bar.get_height()
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            h_val + 0.003,
            f"{h_val:.4f}",
            ha='center', va='bottom', fontsize=8
        )

axes[1, 1].set_title('Comparaison par horizon (LSTM +1h vs GRU +6h)', fontweight='bold')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(horizons, fontsize=11)
axes[1, 1].set_ylim(0.78, 1.02)
axes[1, 1].set_ylabel('Score')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('gru_6h_optuna_resultats.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ gru_6h_optuna_resultats.png sauvegardé")

# Optuna trials
try:
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    vals = [t.value for t in study.trials if t.value is not None]
    ax2.plot(vals, 'o-', color='teal', linewidth=2, markersize=5)
    ax2.axhline(study.best_value, color='red', linestyle='--',
                label=f'Best: {study.best_value:.4f}')
    ax2.set_title(f'GRU Optuna — {N_TRIALS} trials — {TARGET}')
    ax2.set_xlabel('Trial')
    ax2.set_ylabel('F1-macro (sous-échantillon val)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('gru_6h_optuna_trials.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ gru_6h_optuna_trials.png sauvegardé")
except Exception as e:
    print(f"Trials plot ignoré : {e}")

# ============================================================
# RÉSUMÉ FINAL COMPLET
# ============================================================

print("\n" + "=" * 60)
print("RÉSUMÉ FINAL — GRU 6H")
print("=" * 60)
print(f"Target           : {TARGET}  (prédiction 6h à l'avance)")
print(f"Architecture     : GRU {bp['num_layers']}x{bp['hidden_size']} → Dense({bp['fc_size']}) → Softmax(3)")
print(f"Features ({len(FEATURES)})    : {FEATURES}")
print(f"Séquence         : {SEQ_LEN}h d'historique")
print(f"Shapes finaux    :")
print(f"  X_train        : {X_tr_s.shape}")
print(f"  X_val          : {X_va_s.shape}")
print(f"  X_test         : {X_te_s.shape}")
print(f"Paramètres       : {n_params:,}")
print(f"Trials Optuna    : {N_TRIALS} × {OPTUNA_SAMPLE:,} lignes")
print(f"Meilleurs params : {study.best_params}")
print(f"Accuracy         : {acc:.4f}")
print(f"F1-macro         : {f1_macro:.4f}")
print(f"F1-classe 2      : {f1_cls2:.4f}")
print(f"Modèle sauvé     : best_gru_6h_optuna.pt")
print("=" * 60)
