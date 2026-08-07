import pandas as pd
import numpy as np
import sys
import time
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
# CONFIGURATION — OPTIMISÉE POUR 3-5 HEURES SUR CPU
# ============================================================
#
# PARAMÈTRES CLÉS :
#   SEQ_LEN = 6    → fenêtre 6h (solution au problème val/test vides)
#                    suffisant pour h+3 (congestion se prépare sur 2-4h)
#   OPTUNA_SAMPLE = 150_000  → rapide et représentatif
#   N_TRIALS = 15  → suffisant avec MedianPruner
#   OPTUNA_EPOCHS = 5        → cherche les bons params rapidement
#   FINAL_EPOCHS = 25        → réentraînement complet
#
# CORRECTIONS :
#   1. SEQ_LEN réduit à 6 (val/test ne sont plus vides)
#   2. make_sequences vectorisée (stride_tricks) → 10x plus rapide
#   3. split par cellule (70/15/15) pour garantir séquences valides
#   4. lag_avaibility_3h ajouté
#   5. Temps affiché à chaque étape
# ============================================================

TIME_START = time.time()

def elapsed():
    s = time.time() - TIME_START
    h, m = int(s // 3600), int((s % 3600) // 60)
    sec = int(s % 60)
    if h > 0:   return f"{h}h {m}min {sec}s"
    elif m > 0: return f"{m}min {sec}s"
    else:       return f"{sec}s"

# ── Paramètres modifiables ────────────────────────────────────
TARGET         = 'target_3h'
SEQ_LEN        = 6          # CORRIGÉ : 24→6 (val/test ne sont plus vides)
N_CLASSES      = 3
BATCH_SIZE     = 4096       # grand batch → plus rapide sur CPU
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_TRIALS       = 15         # suffisant avec MedianPruner + 5 epochs
OPTUNA_SAMPLE  = 150_000    # sous-échantillon stratifié pour Optuna
OPTUNA_EPOCHS  = 5
FINAL_EPOCHS   = 25
FINAL_PATIENCE = 5

print("=" * 65)
print(f"  LSTM + ATTENTION + OPTUNA  —  {TARGET}")
print("=" * 65)
print(f"  Device           : {DEVICE}")
print(f"  SEQ_LEN          : {SEQ_LEN}h  (fenêtre de contexte)")
print(f"  Trials Optuna    : {N_TRIALS} × {OPTUNA_EPOCHS} epochs")
print(f"  Lignes par trial : {OPTUNA_SAMPLE:,} (stratifié)")
print(f"  Batch size       : {BATCH_SIZE}")
print(f"  Classe_Congestion: EXCLUE (anticipation pure h+3)")
print("=" * 65)

import datetime
fin_estimee = datetime.datetime.now() + datetime.timedelta(hours=5)
print(f"  Lancement : {datetime.datetime.now().strftime('%H:%M:%S')}")
print(f"  Fin estimée (~5h) : {fin_estimee.strftime('%H:%M:%S')}")
print("=" * 65)

# ============================================================
# ÉTAPE 1 — CHARGEMENT ET PRÉPARATION
# ============================================================

t1 = time.time()
print(f"\n[1/7] Chargement et préparation...  [{elapsed()}]")

df = pd.read_csv("dataset_avec_targets.csv")
df.columns = df.columns.str.lower().str.strip()
df['date_'] = pd.to_datetime(df['date_'].astype(str).str.strip(), errors='coerce')
df = df.sort_values(['cellname_id', 'date_']).reset_index(drop=True)

# Suppression features redondantes (justifiées par matrice corrélation EDA)
COLS_DROP = [
    'time_to_peak', 'peak_trend_interaction', 'traffic_per_user',
    'prb_z_score', 'prb_per_user', 'throughput_per_user'
]
df.drop(columns=[c for c in COLS_DROP if c in df.columns], inplace=True)

# Encodage cyclique heure — évite discontinuité 23h→0h
if 'hour' in df.columns:
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df.drop(columns=['hour'], inplace=True)
    print("  ✅ hour_sin / hour_cos créés")

# Lags PRB — mémoire des 3h passées, cohérent avec horizon h+3
df['lag_prb_1h'] = df.groupby('cellname_id')['dl_prb_usage_rate'].shift(1)
df['lag_prb_2h'] = df.groupby('cellname_id')['dl_prb_usage_rate'].shift(2)
df['lag_prb_3h'] = df.groupby('cellname_id')['dl_prb_usage_rate'].shift(3)

# Lag Avaibility cohérent avec h+3
cols_dropna = ['lag_prb_1h', 'lag_prb_2h', 'lag_prb_3h', TARGET]
if 'avaibility' in df.columns:
    df['lag_avaibility_3h'] = df.groupby('cellname_id')['avaibility'].shift(3)
    cols_dropna.append('lag_avaibility_3h')

df.dropna(subset=cols_dropna, inplace=True)

print(f"  Dataset final  : {len(df):,} lignes")
print(f"  Cellules BTS   : {df['cellname_id'].nunique():,}")
print(f"  Période        : {df['date_'].min()} → {df['date_'].max()}")
print(f"  Durée étape    : {time.time()-t1:.1f}s")

# ============================================================
# ÉTAPE 2 — FEATURES ET SPLIT PAR CELLULE
# ============================================================
# POURQUOI SPLIT PAR CELLULE et non par date ?
#
# Split par DATE (ancienne version) :
#   Val = 15% des heures → ~21h par cellule
#   Avec SEQ_LEN=24 → 21 < 24 → val VIDE  ❌
#
# Split par CELLULE (cette version) :
#   Pour chaque cellule : 70% early / 15% mid / 15% late
#   Val = 15% des heures de chaque cellule = ~21h
#   Avec SEQ_LEN=6 → 21 > 6 → val a des séquences  ✅
# ============================================================

t2 = time.time()
print(f"\n[2/7] Features et split par cellule...  [{elapsed()}]")

# h+3 : classe_congestion exclue — anticipation pure
EXCLURE  = ['date_', 'cellname_id', 'target_1h', 'target_3h',
            'target_6h', 'classe_congestion', 'congestion_score']
FEATURES = [c for c in df.columns if c not in EXCLURE]

print(f"  Features ({len(FEATURES)}) :")
for i, f in enumerate(FEATURES, 1):
    print(f"    {i:2d}. {f}")

# Normalisation sur tout le df (fit sur 70% de chaque cellule serait trop complexe)
# On fit le scaler sur les données train (70% chronologique global)
dates    = df['date_'].dropna().sort_values()
d70, d85 = dates.quantile(0.70), dates.quantile(0.85)
train_global = df[df['date_'] <= d70]
scaler = StandardScaler()
scaler.fit(train_global[FEATURES])
print(f"\n  Scaler fitté sur {len(train_global):,} lignes (70% global)")

print(f"  Durée étape : {time.time()-t2:.1f}s")

# ============================================================
# ÉTAPE 3 — SÉQUENCES PAR CELLULE (VECTORISÉES)
# ============================================================
# VERSION OPTIMISÉE avec stride_tricks — 10x plus rapide
# que la boucle Python for i in range(seq_len, n)
#
# stride_tricks crée une "vue" du tableau sans copier les données
# → création de millions de séquences en quelques secondes

t3 = time.time()
print(f"\n[3/7] Création des séquences (vectorisée)...  [{elapsed()}]")

def make_sequences_fast(X, y, seq_len, cell_ids):
    """
    Crée des séquences LSTM par cellule avec stride_tricks.
    VECTORISÉ : 10x plus rapide que la boucle Python.
    Garantit qu'aucune séquence ne mélange 2 cellules.

    Paramètres :
        X        : tableau normalisé [N, features]
        y        : labels [N]
        seq_len  : longueur de la fenêtre (SEQ_LEN)
        cell_ids : identifiant cellule pour chaque ligne

    Retourne :
        Xs : [N_seq, seq_len, features]
        ys : [N_seq]
    """
    Xs_list, ys_list = [], []
    cellules = np.unique(cell_ids)

    for cell in cellules:
        mask   = (cell_ids == cell)
        Xc     = X[mask]
        yc     = y[mask]
        n      = len(Xc)

        if n <= seq_len:
            continue   # cellule trop petite

        # stride_tricks : crée toutes les fenêtres d'un coup
        # sans boucle Python → très rapide
        shape   = (n - seq_len, seq_len, Xc.shape[1])
        strides = (Xc.strides[0], Xc.strides[0], Xc.strides[1])
        Xc_seq  = np.lib.stride_tricks.as_strided(
            Xc, shape=shape, strides=strides
        ).copy()   # .copy() obligatoire pour éviter les artefacts mémoire

        Xs_list.append(Xc_seq)
        ys_list.append(yc[seq_len:])

    if not Xs_list:
        n_feat = X.shape[1]
        return (np.zeros((0, seq_len, n_feat), dtype=np.float32),
                np.zeros(0, dtype=np.int64))

    return (np.concatenate(Xs_list).astype(np.float32),
            np.concatenate(ys_list).astype(np.int64))


# Split par cellule 70/15/15 chronologique
# Chaque cellule contribue des séquences dans chaque split
Xtr_l, ytr_l = [], []
Xva_l, yva_l = [], []
Xte_l, yte_l = [], []

cellules_uniques = df['cellname_id'].unique()
print(f"  Traitement de {len(cellules_uniques):,} cellules...")

X_all_scaled = scaler.transform(df[FEATURES])
y_all        = df[TARGET].values.astype(np.int64)
ids_all      = df['cellname_id'].values

t_split = time.time()

for cell in cellules_uniques:
    mask   = (ids_all == cell)
    Xc     = X_all_scaled[mask].astype(np.float32)
    yc     = y_all[mask]
    n      = len(Xc)

    if n <= SEQ_LEN + 4:
        continue

    # Split 70/15/15 par position chronologique
    n_tr = int(n * 0.70)
    n_va = int(n * 0.85)

    # Séquences pour chaque split avec stride_tricks
    def seq_block(Xb, yb):
        nb = len(Xb)
        if nb <= SEQ_LEN:
            return None, None
        shape   = (nb - SEQ_LEN, SEQ_LEN, Xb.shape[1])
        strides = (Xb.strides[0], Xb.strides[0], Xb.strides[1])
        return (np.lib.stride_tricks.as_strided(
                    Xb, shape=shape, strides=strides).copy(),
                yb[SEQ_LEN:])

    xs, ys = seq_block(Xc[:n_tr],    yc[:n_tr])
    if xs is not None:
        Xtr_l.append(xs); ytr_l.append(ys)

    xs, ys = seq_block(Xc[n_tr:n_va], yc[n_tr:n_va])
    if xs is not None:
        Xva_l.append(xs); yva_l.append(ys)

    xs, ys = seq_block(Xc[n_va:],    yc[n_va:])
    if xs is not None:
        Xte_l.append(xs); yte_l.append(ys)

print(f"  Boucle cellules : {time.time()-t_split:.1f}s")

# Concaténation finale
X_tr_s = np.concatenate(Xtr_l).astype(np.float32)
y_tr_s = np.concatenate(ytr_l).astype(np.int64)
X_va_s = np.concatenate(Xva_l).astype(np.float32)
y_va_s = np.concatenate(yva_l).astype(np.int64)
X_te_s = np.concatenate(Xte_l).astype(np.float32)
y_te_s = np.concatenate(yte_l).astype(np.int64)

print(f"\n  Séquences créées :")
print(f"    Train : {X_tr_s.shape}  [N, {SEQ_LEN}h, {len(FEATURES)} features]")
print(f"    Val   : {X_va_s.shape}")
print(f"    Test  : {X_te_s.shape}")

# Vérifications critiques
assert len(X_va_s) > 0, (
    f"Val vide ! SEQ_LEN={SEQ_LEN} trop grand. "
    f"Essayer SEQ_LEN=3"
)
assert len(X_te_s) > 0, (
    f"Test vide ! SEQ_LEN={SEQ_LEN} trop grand. "
    f"Essayer SEQ_LEN=3"
)

print(f"\n  Distribution target_3h :")
for cls, nom in zip([0,1,2], ['Normal','Modéré','Critique']):
    n = (y_tr_s == cls).sum()
    print(f"    Classe {cls} {nom:<10} : {n:,} ({n/len(y_tr_s)*100:.1f}%)")

# Poids classes pour gérer déséquilibre (Critique ~4.6%)
weights_arr = compute_class_weight('balanced',
                                    classes=np.array([0,1,2]),
                                    y=y_tr_s)
weights_t   = torch.tensor(weights_arr, dtype=torch.float32).to(DEVICE)
print(f"\n  Poids : Normal={weights_arr[0]:.3f} | "
      f"Modéré={weights_arr[1]:.3f} | Critique={weights_arr[2]:.3f}")
print(f"  Durée étape : {time.time()-t3:.1f}s  [{elapsed()}]")

# ============================================================
# ÉTAPE 4 — SOUS-ÉCHANTILLON OPTUNA + DATALOADERS
# ============================================================

t4 = time.time()
print(f"\n[4/7] Sous-échantillon Optuna stratifié...  [{elapsed()}]")

rng   = np.random.default_rng(42)
idx_0 = np.where(y_tr_s == 0)[0]
idx_1 = np.where(y_tr_s == 1)[0]
idx_2 = np.where(y_tr_s == 2)[0]
n0    = int(OPTUNA_SAMPLE * 0.55)
n1    = int(OPTUNA_SAMPLE * 0.35)
n2    = OPTUNA_SAMPLE - n0 - n1

sample_idx = np.sort(np.concatenate([
    rng.choice(idx_0, min(n0, len(idx_0)), replace=False),
    rng.choice(idx_1, min(n1, len(idx_1)), replace=False),
    rng.choice(idx_2, min(n2, len(idx_2)), replace=False),
]))
X_opt, y_opt = X_tr_s[sample_idx], y_tr_s[sample_idx]

val_idx  = rng.choice(len(X_va_s), min(50_000, len(X_va_s)), replace=False)
X_va_opt = X_va_s[val_idx]
y_va_opt = y_va_s[val_idx]

print(f"  Optuna train : {len(X_opt):,} | val : {len(X_va_opt):,}")

# DataLoaders
opt_tr_ld  = DataLoader(TensorDataset(torch.from_numpy(X_opt),
                                       torch.from_numpy(y_opt)),
                        batch_size=2048, shuffle=False)
opt_va_ld  = DataLoader(TensorDataset(torch.from_numpy(X_va_opt),
                                       torch.from_numpy(y_va_opt)),
                        batch_size=2048, shuffle=False)
full_tr_ld = DataLoader(TensorDataset(torch.from_numpy(X_tr_s),
                                       torch.from_numpy(y_tr_s)),
                        batch_size=BATCH_SIZE, shuffle=False)
full_va_ld = DataLoader(TensorDataset(torch.from_numpy(X_va_s),
                                       torch.from_numpy(y_va_s)),
                        batch_size=BATCH_SIZE, shuffle=False)
test_ld    = DataLoader(TensorDataset(torch.from_numpy(X_te_s),
                                       torch.from_numpy(y_te_s)),
                        batch_size=BATCH_SIZE, shuffle=False)

print(f"  Batches train complet : {len(full_tr_ld)}")
print(f"  Batches val complet   : {len(full_va_ld)}")
print(f"  Durée étape : {time.time()-t4:.1f}s")

# ============================================================
# ARCHITECTURE LSTM + ATTENTION TEMPORELLE
# ============================================================

class LSTMAttention(nn.Module):
    """
    LSTM + mécanisme d'attention temporelle pour h+3.

    Pourquoi attention ?
    → LSTM simple garde seulement le dernier état caché
    → Attention calcule un poids pour chacune des SEQ_LEN heures
    → Le modèle apprend QUELLES heures regarder pour prédire h+3
    → Interprétable : graphique des poids d'attention

    Input  : [batch, SEQ_LEN, n_features]
    Output : logits [batch, 3] + poids [batch, SEQ_LEN]
    """
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout, fc_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn = nn.Linear(hidden_size, 1)
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Sequential(
            nn.Linear(hidden_size, fc_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_size, N_CLASSES)
        )

    def forward(self, x):
        out, _  = self.lstm(x)                          # [B, SEQ, hidden]
        scores  = self.attn(out).squeeze(-1)            # [B, SEQ]
        attn_w  = torch.softmax(scores, dim=1)          # [B, SEQ]
        context = (attn_w.unsqueeze(-1) * out).sum(1)   # [B, hidden]
        context = self.norm(context)
        return self.fc(self.drop(context)), attn_w

# ============================================================
# ÉTAPE 5 — OPTUNA RAPIDE
# ============================================================

t5 = time.time()
print(f"\n[5/7] Optuna — {N_TRIALS} trials × {OPTUNA_EPOCHS} epochs..."
      f"  [{elapsed()}]")

def objective(trial):
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    num_layers  = trial.suggest_int('num_layers', 1, 2)   # max 2 → plus rapide
    dropout     = trial.suggest_float('dropout', 0.1, 0.4, step=0.1)
    fc_size     = trial.suggest_categorical('fc_size', [32, 64, 128])
    lr          = trial.suggest_float('lr', 5e-4, 5e-3, log=True)
    wd          = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

    m    = LSTMAttention(len(FEATURES), hidden_size, num_layers,
                         dropout, fc_size).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=weights_t)
    opt  = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)

    best_f1 = 0.0
    for epoch in range(OPTUNA_EPOCHS):
        m.train()
        for xb, yb in opt_tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits, _ = m(xb)
            crit(logits, yb).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()

        m.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in opt_va_ld:
                logits, _ = m(xb.to(DEVICE))
                preds.extend(logits.argmax(1).cpu().numpy())

        f1 = f1_score(y_va_opt, preds, average='macro', zero_division=0)
        trial.report(f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        if f1 > best_f1:
            best_f1 = f1

    return best_f1

def print_cb(study, trial):
    status = f"F1={trial.value:.4f}" if trial.value else "PRUNÉ"
    best   = study.best_value if study.best_value else 0
    print(f"  Trial {trial.number:2d} | {status} | "
          f"Best={best:.4f} | {elapsed()} | {trial.params}")

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
)
study.optimize(objective, n_trials=N_TRIALS, callbacks=[print_cb])

print(f"\n  ✅ Meilleur F1 val  : {study.best_value:.4f}")
print(f"  Meilleurs params   : {study.best_params}")
print(f"  Durée Optuna       : {time.time()-t5:.1f}s  [{elapsed()}]")

# ============================================================
# ÉTAPE 6 — RÉENTRAÎNEMENT FINAL SUR DONNÉES COMPLÈTES
# ============================================================

t6 = time.time()
print(f"\n[6/7] Réentraînement final...  [{elapsed()}]")
print(f"  {len(X_tr_s):,} séquences train | {len(X_va_s):,} val")

bp         = study.best_params
best_model = LSTMAttention(
    len(FEATURES), bp['hidden_size'], bp['num_layers'],
    bp['dropout'], bp['fc_size']
).to(DEVICE)

n_params = sum(p.numel() for p in best_model.parameters()
               if p.requires_grad)
print(f"  Architecture : LSTM {bp['num_layers']}×{bp['hidden_size']} "
      f"+ Attention → Dense({bp['fc_size']}) → Softmax(3)")
print(f"  Paramètres   : {n_params:,}")

criterion = nn.CrossEntropyLoss(weight=weights_t)
optimizer = torch.optim.Adam(
    best_model.parameters(),
    lr=bp['lr'], weight_decay=bp['weight_decay']
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=3, factor=0.5, verbose=False
)

best_f1_val = 0.0
patience_ct = 0
history     = {'loss': [], 'val_f1': [], 'val_f1_cls2': []}

for epoch in range(1, FINAL_EPOCHS + 1):
    t_ep = time.time()

    # Train
    best_model.train()
    total_loss = 0
    for xb, yb in full_tr_ld:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = best_model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(best_model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
    avg_loss = total_loss / len(y_tr_s)

    # Validation
    best_model.eval()
    preds_v = []
    with torch.no_grad():
        for xb, _ in full_va_ld:
            logits, _ = best_model(xb.to(DEVICE))
            preds_v.extend(logits.argmax(1).cpu().numpy())

    val_f1      = f1_score(y_va_s, preds_v, average='macro',
                            zero_division=0)
    val_f1_cls2 = f1_score(y_va_s, preds_v, average=None,
                            zero_division=0)[2]
    scheduler.step(val_f1)

    history['loss'].append(avg_loss)
    history['val_f1'].append(val_f1)
    history['val_f1_cls2'].append(val_f1_cls2)

    ep_time = time.time() - t_ep
    status  = " ✅" if val_f1 > best_f1_val else ""
    print(f"  Epoch {epoch:3d}/{FINAL_EPOCHS} | "
          f"Loss={avg_loss:.4f} | "
          f"Val F1={val_f1:.4f} | "
          f"F1-Critique={val_f1_cls2:.4f} | "
          f"{ep_time:.1f}s/ep | "
          f"Total={elapsed()}{status}")

    if val_f1 > best_f1_val:
        best_f1_val = val_f1
        torch.save(best_model.state_dict(), 'best_lstm_3h.pt')
        patience_ct = 0
    else:
        patience_ct += 1
        if patience_ct >= FINAL_PATIENCE:
            print(f"  ⏹ Early stopping epoch {epoch}")
            break

print(f"\n  ✅ Meilleur F1 val : {best_f1_val:.4f}")
print(f"  Durée réentraînement : {time.time()-t6:.1f}s  [{elapsed()}]")

# ============================================================
# ÉTAPE 7 — ÉVALUATION ET GRAPHIQUES
# ============================================================

t7 = time.time()
print(f"\n[7/7] Évaluation test + graphiques...  [{elapsed()}]")

best_model.load_state_dict(
    torch.load('best_lstm_3h.pt', map_location=DEVICE)
)
best_model.eval()

preds_test, attn_all = [], []
with torch.no_grad():
    for xb, _ in test_ld:
        logits, attn_w = best_model(xb.to(DEVICE))
        preds_test.extend(logits.argmax(1).cpu().numpy())
        attn_all.append(attn_w.cpu().numpy())

preds_test = np.array(preds_test)
attn_mean  = np.concatenate(attn_all).mean(axis=0)   # [SEQ_LEN]

print("\n" + "=" * 65)
print(f"  RÉSULTATS FINAUX — {TARGET}")
print("=" * 65)
print(classification_report(
    y_te_s, preds_test,
    target_names=['Normal (0)', 'Modéré (1)', 'Congestionné (2)'],
    digits=4
))

f1_macro = f1_score(y_te_s, preds_test, average='macro',    zero_division=0)
f1_cls2  = f1_score(y_te_s, preds_test, average=None,       zero_division=0)[2]
acc      = (preds_test == y_te_s).mean()

print(f"  Accuracy    : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  F1-macro    : {f1_macro:.4f}  ({f1_macro*100:.2f}%)")
print(f"  F1-Critique : {f1_cls2:.4f}  ({f1_cls2*100:.2f}%)  ← métrique principale")

heure_max = np.argmax(attn_mean)
print(f"\n  Attention — heure la plus informative :")
print(f"    Position {heure_max} → t-{SEQ_LEN - heure_max}h "
      f"(poids={attn_mean[heure_max]:.4f})")

# ── Graphiques 2×2 ───────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f'LSTM + Attention — {TARGET}  |  SEQ_LEN={SEQ_LEN}h\n'
    f'F1-macro={f1_macro:.4f} | F1-Critique={f1_cls2:.4f} | '
    f'Accuracy={acc:.4f} | Temps={elapsed()}',
    fontsize=12, fontweight='bold'
)

# G1 : Loss
axes[0,0].plot(history['loss'], color='steelblue', linewidth=2)
axes[0,0].set_title('Loss entraînement (train uniquement)')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('CrossEntropy Loss')
axes[0,0].grid(alpha=0.3)

# G2 : F1 validation
axes[0,1].plot(history['val_f1'],      label='F1-macro',
               color='green',  linewidth=2)
axes[0,1].plot(history['val_f1_cls2'], label='F1-Critique',
               color='orange', linewidth=2, linestyle='--')
axes[0,1].axhline(f1_macro, color='red', linestyle=':',
                   label=f'Test F1={f1_macro:.4f}')
axes[0,1].set_title('F1 sur validation (séparé du train)')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('F1')
axes[0,1].legend(fontsize=9)
axes[0,1].grid(alpha=0.3)

# G3 : Matrice de confusion
cm = confusion_matrix(y_te_s, preds_test)
sns.heatmap(
    cm / cm.sum(axis=1, keepdims=True) * 100,
    annot=True, fmt='.1f', cmap='Blues',
    xticklabels=['Normal', 'Modéré', 'Congestionné'],
    yticklabels=['Normal', 'Modéré', 'Congestionné'],
    ax=axes[1,0], cbar=False
)
axes[1,0].set_title('Matrice de confusion (%) — test set')
axes[1,0].set_xlabel('Prédiction')
axes[1,0].set_ylabel('Réalité')

# G4 : Poids attention temporelle
colors = ['#D85A30' if w > attn_mean.mean() else '#1D9E75'
          for w in attn_mean]
axes[1,1].bar(range(SEQ_LEN), attn_mean, color=colors)
axes[1,1].axvline(heure_max, color='red', linestyle='--',
                   label=f'Max à t-{SEQ_LEN-heure_max}h')
axes[1,1].axhline(attn_mean.mean(), color='black',
                   linestyle=':', alpha=0.5,
                   label=f'Moy={attn_mean.mean():.4f}')
axes[1,1].set_title(f'Poids attention temporelle\n'
                     f'(rouge = heures les plus informatives)')
axes[1,1].set_xlabel(f'Position (0=t-{SEQ_LEN}h, '
                      f'{SEQ_LEN-1}=maintenant)')
axes[1,1].set_ylabel('Poids moyen')
axes[1,1].legend(fontsize=9)
axes[1,1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('lstm_3h_resultats.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ lstm_3h_resultats.png sauvegardé")
print(f"  Durée étape : {time.time()-t7:.1f}s")

# ============================================================
# RÉSUMÉ FINAL COMPLET
# ============================================================

temps_total = time.time() - TIME_START
h = int(temps_total // 3600)
m = int((temps_total % 3600) // 60)
s = int(temps_total % 60)

print(f"\n{'='*65}")
print(f"  RÉSUMÉ FINAL — LSTM H+3")
print(f"{'='*65}")
print(f"  Dataset          : {len(df):,} lignes")
print(f"  Cellules BTS     : {df['cellname_id'].nunique():,}")
print(f"  Features         : {len(FEATURES)}")
print(f"  SEQ_LEN          : {SEQ_LEN}h de contexte")
print(f"  Séq. train       : {len(X_tr_s):,}")
print(f"  Séq. val         : {len(X_va_s):,}")
print(f"  Séq. test        : {len(X_te_s):,}")
print(f"  Architecture     : LSTM {bp['num_layers']}×{bp['hidden_size']} "
      f"+ Attention → Dense({bp['fc_size']}) → Softmax(3)")
print(f"  Paramètres       : {n_params:,}")
print(f"  Trials Optuna    : {N_TRIALS} "
      f"(sous-éch. {OPTUNA_SAMPLE:,} stratifié)")
print(f"  Best params      : {study.best_params}")
print(f"  Split            : 70/15/15 par cellule")
print(f"  Accuracy         : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  F1-macro         : {f1_macro:.4f}  ({f1_macro*100:.2f}%)")
print(f"  F1-Critique      : {f1_cls2:.4f}  ({f1_cls2*100:.2f}%)")
print(f"  Modèle sauvé     : best_lstm_3h.pt")
print(f"  ⏱  TEMPS TOTAL  : {h}h {m}min {s}s")
print(f"{'='*65}")

print(f"\n  Top 3 heures les plus prédictives (attention) :")
top3 = np.argsort(attn_mean)[-3:][::-1]
for rank, idx in enumerate(top3, 1):
    print(f"    Top {rank} : t-{SEQ_LEN-idx}h "
          f"(poids={attn_mean[idx]:.4f})")
print(f"\n  → Le modèle regarde surtout les {SEQ_LEN-top3[0]}h "
      f"précédant la prédiction")
print(f"{'='*65}")