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
import datetime
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# TEMPS D'EXÉCUTION
# ============================================================

TIME_START = time.time()

def elapsed():
    s = time.time() - TIME_START
    h, m = int(s // 3600), int((s % 3600) // 60)
    sec = int(s % 60)
    if h > 0:   return f"{h}h {m}min {sec}s"
    elif m > 0: return f"{m}min {sec}s"
    else:       return f"{sec}s"

# ============================================================
# POURQUOI LSTM + ATTENTION POUR H+1 ?
# ============================================================
# Ta binôme a fait : GRU pour H+6
# Toi tu fais      : LSTM + ATTENTION pour H+1
#
# Différence LSTM vs GRU :
#   GRU  = 2 portes (reset + update)    → moins de paramètres
#   LSTM = 3 portes (input+forget+output)→ plus expressif, meilleur sur séries longues
#
# Pourquoi ATTENTION pour H+1 ?
#   Sans attention : LSTM garde seulement le DERNIER état caché
#   Avec attention : LSTM calcule un score pour CHAQUE heure passée
#   → le modèle apprend QUELLES heures sont les plus prédictives
#   → pour H+1, l'heure la plus récente (t-1h) devrait dominer
#   → ce résultat est interprétable et montrable au jury
#
# SHAPE à retenir :
#   Entrée LSTM  : [batch, SEQ_LEN, n_features]  = [4096, 6, 19]
#   Sortie LSTM  : [batch, SEQ_LEN, hidden_size] = [4096, 6, 128]
#   Attention    : [batch, SEQ_LEN]              = [4096, 6]  poids
#   Contexte     : [batch, hidden_size]          = [4096, 128]
#   Sortie finale: [batch, 3]                    = [4096, 3]  logits
# ============================================================

# ============================================================
# CONFIGURATION — LSTM H+1
# ============================================================

TARGET         = 'target_1h'    # prédiction 1h à l'avance
SEQ_LEN        = 6              # fenêtre de 6h de contexte
N_CLASSES      = 3              # 0=Normal, 1=Modéré, 2=Critique
BATCH_SIZE     = 4096           # grand batch → plus rapide sur CPU
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_TRIALS       = 15             # trials Optuna
OPTUNA_SAMPLE  = 150_000        # sous-échantillon stratifié
OPTUNA_EPOCHS  = 5              # epochs rapides par trial
FINAL_EPOCHS   = 25             # epochs réentraînement final
FINAL_PATIENCE = 5              # early stopping patience

# Résultats de ta binôme (pour le tableau comparatif)
REF = {
    'GRU +6h' : {'acc': None, 'f1': None, 'f1cls2': None},
    # ← remplace None par ses résultats quand tu les as
}

print("=" * 65)
print(f"  LSTM + ATTENTION + OPTUNA  —  {TARGET}")
print("=" * 65)
print(f"  Device           : {DEVICE}")
print(f"  SEQ_LEN          : {SEQ_LEN}h de contexte")
print(f"  Trials Optuna    : {N_TRIALS} × {OPTUNA_EPOCHS} epochs")
print(f"  Sample Optuna    : {OPTUNA_SAMPLE:,} (stratifié)")
print(f"  Batch size       : {BATCH_SIZE}")
print(f"  Classe_Congestion: EXCLUE (anticipation pure h+1)")
print()
print(f"  LSTM vs GRU (ta binôme) :")
print(f"    GRU  : 2 portes → moins de paramètres, convergence rapide")
print(f"    LSTM : 3 portes → plus expressif, meilleur pour séries longues")
print(f"    Attention ajoutée : interprétabilité des heures importantes")
print("=" * 65)
print(f"  Lancement      : {datetime.datetime.now().strftime('%H:%M:%S')}")
fin = datetime.datetime.now() + datetime.timedelta(hours=3)
print(f"  Fin estimée    : {fin.strftime('%H:%M:%S')}  (~3h)")
print("=" * 65)

# ============================================================
# ÉTAPE 1 — CHARGEMENT ET NETTOYAGE
# ============================================================
# On part du dataset nettoyé avec les targets déjà créés.
# Ce fichier contient toutes les features + target_1h/3h/6h.

t1 = time.time()
print(f"\n[1/8] Chargement et nettoyage...  [{elapsed()}]")

df = pd.read_csv("dataset_avec_targets.csv")

# Mise en minuscules pour éviter les erreurs de casse
# (certaines colonnes peuvent être DATE_ ou date_ selon l'export)
df.columns = df.columns.str.lower().str.strip()

# Conversion date
df['date_'] = pd.to_datetime(df['date_'].astype(str).str.strip(), errors='coerce')

# Tri chronologique par cellule — OBLIGATOIRE
# Sans ce tri, les lags et rolling windows seraient faux
df = df.sort_values(['cellname_id', 'date_']).reset_index(drop=True)

print(f"  Lignes brutes  : {len(df):,}")
print(f"  Cellules BTS   : {df['cellname_id'].nunique():,}")
print(f"  Colonnes brutes: {list(df.columns)}")

# Suppression des features redondantes
# Justification par la matrice de corrélation de l'EDA :
#   - time_to_peak, peak_trend_interaction → redondants avec hour_sin/cos
#   - traffic_per_user, throughput_per_user, prb_per_user → ratios redondants
#   - prb_z_score → version normalisée de dl_prb_usage_rate
#   - cell_traffic_volume_dl/ul → corr 0.87 avec avg_user_nb
#   - lte_setup_success_rate → corr 0.95 avec avaibility
COLS_DROP = [
    'time_to_peak', 'peak_trend_interaction', 'traffic_per_user',
    'prb_z_score', 'prb_per_user', 'throughput_per_user',
    'cell_traffic_volume_dl', 'cell_traffic_volume_ul',
    'lte_setup_success_rate'
]
dropped = [c for c in COLS_DROP if c in df.columns]
df.drop(columns=dropped, inplace=True)
print(f"  Features supprimées ({len(dropped)}) : {dropped}")

# Encodage cyclique de l'heure
# Pourquoi sin/cos ?
#   HOUR brut : 0, 1, 2, ..., 23 → le modèle pense que 23h est loin de 0h
#   sin/cos   : encode la nature cyclique → 23h est adjacent à 0h ✅
if 'hour' in df.columns:
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df.drop(columns=['hour'], inplace=True)
    print("  ✅ hour → hour_sin / hour_cos (encodage cyclique)")

# Lags PRB — mémoire des heures passées
# lag_prb_1h : PRB il y a 1h → tendance récente (cohérent avec h+1)
# lag_prb_2h : PRB il y a 2h → confirme ou infirme la tendance
# Pour H+1, lag_prb_1h est la feature la plus importante
df['lag_prb_1h'] = df.groupby('cellname_id')['dl_prb_usage_rate'].shift(1)
df['lag_prb_2h'] = df.groupby('cellname_id')['dl_prb_usage_rate'].shift(2)
print("  ✅ lag_prb_1h / lag_prb_2h créés")

# Lag Avaibility — cohérent avec h+1
# L'état de disponibilité il y a 1h aide à prédire l'état dans 1h
if 'avaibility' in df.columns:
    df['lag_avaibility_1h'] = df.groupby('cellname_id')['avaibility'].shift(1)
    print("  ✅ lag_avaibility_1h créé")

# Suppression des NaN créés par les shifts
# Les 2 premières lignes de chaque cellule n'ont pas de lag valide
cols_dropna = ['lag_prb_1h', 'lag_prb_2h', TARGET]
if 'lag_avaibility_1h' in df.columns:
    cols_dropna.append('lag_avaibility_1h')

avant = len(df)
df.dropna(subset=cols_dropna, inplace=True)
print(f"  Lignes supprimées (NaN lags) : {avant - len(df):,}")
print(f"  Dataset final  : {len(df):,} lignes")
print(f"  Période        : {df['date_'].min()} → {df['date_'].max()}")
print(f"  Durée étape    : {time.time()-t1:.1f}s")

# ============================================================
# ÉTAPE 2 — DÉFINITION DES FEATURES
# ============================================================
# On exclut classe_congestion pour H+1
# Justification : anticipation pure — le modèle ne doit pas
# s'appuyer sur l'état actuel pour prédire l'état dans 1h
# (même si pour H+1 c'est discutable, on reste cohérent
# avec la démarche H+3 et H+6 de ta binôme)

t2 = time.time()
print(f"\n[2/8] Définition des features...  [{elapsed()}]")

EXCLURE  = ['date_', 'cellname_id', 'target_1h', 'target_3h',
            'target_6h', 'classe_congestion', 'congestion_score']
FEATURES = [c for c in df.columns if c not in EXCLURE]

print(f"\n  Features retenues ({len(FEATURES)}) :")
for i, f in enumerate(FEATURES, 1):
    print(f"    [{i:02d}] {f}")

print(f"\n  Durée étape : {time.time()-t2:.1f}s")

# ============================================================
# ÉTAPE 3 — NORMALISATION ET SÉQUENCES PAR CELLULE
# ============================================================
# POURQUOI NORMALISER ?
#   Les features ont des échelles très différentes :
#   dl_prb_usage_rate : 0-100  (pourcentage)
#   dl_average_throughput : 0-263 468  (kbps)
#   Le LSTM traite toutes les features pareil → il faut les
#   ramener à la même échelle sinon le débit "domine" le PRB
#
# POURQUOI StandardScaler et non MinMax ?
#   StandardScaler = (x - mean) / std → robuste aux outliers
#   MinMax         = (x - min) / (max-min) → sensible aux outliers
#   Or ton dataset a des outliers intentionnellement conservés
#   (607 users, 48 GB/h) → StandardScaler est plus adapté
#
# SHAPE EXPLIQUÉ PAS À PAS :
#   Avant normalisation  : df[FEATURES] = [7M lignes, 19 features]
#   Après normalisation  : X_tr = [N_train, 19]  (tableau 2D)
#   Après make_sequences : X_tr_s = [N_seq, 6, 19]  (tableau 3D)
#     dimension 1 : nombre de séquences
#     dimension 2 : SEQ_LEN = 6 heures d'historique
#     dimension 3 : 19 features par heure

t3 = time.time()
print(f"\n[3/8] Normalisation + séquences par cellule...  [{elapsed()}]")

# Scaler fitté uniquement sur les 70% chronologiques
# → évite la fuite d'information du futur vers le passé
dates = df['date_'].dropna().sort_values()
d70   = dates.quantile(0.70)
scaler = StandardScaler()
scaler.fit(df[df['date_'] <= d70][FEATURES])
print(f"  Scaler fitté sur {len(df[df['date_'] <= d70]):,} lignes (70% chronologique)")

# Normalisation de tout le dataset
X_all_sc = scaler.transform(df[FEATURES]).astype(np.float32)
y_all    = df[TARGET].values.astype(np.int64)
ids_all  = df['cellname_id'].values

print(f"\n  Shape AVANT séquences (2D) : {X_all_sc.shape}")
print(f"    → [N_lignes={X_all_sc.shape[0]:,}, N_features={X_all_sc.shape[1]}]")

# ── CRÉATION DES SÉQUENCES PAR CELLULE ───────────────────────
# Pourquoi par cellule et non sur tout le dataset ?
#   Sans cellule : la fenêtre peut contenir la FIN de la cellule 324
#                  ET le DÉBUT de la cellule 325 → physiquement faux
#   Avec cellule : chaque séquence de 6h appartient à UNE SEULE cellule ✅
#
# STRIDE_TRICKS (vectorisé) :
#   au lieu de : for i in range(seq_len, n): → lent
#   on utilise  : stride_tricks.as_strided() → immédiat
#   Économie : 664s → ~30s pour 54 291 cellules

def seq_block(Xb, yb, seq_len):
    """
    Crée toutes les séquences d'un bloc [N, features]
    en une seule opération vectorisée (stride_tricks).

    Exemple avec seq_len=6 et N=10 :
      Séquence 0 : X[0:6]  → prédit y[6]
      Séquence 1 : X[1:7]  → prédit y[7]
      ...
      Séquence 3 : X[3:9]  → prédit y[9]
      Total : N - seq_len = 4 séquences

    SHAPE en sortie : [N-seq_len, seq_len, n_features]
    """
    nb = len(Xb)
    if nb <= seq_len:
        return None, None
    # stride_tricks crée une "vue" sans copier les données
    shape   = (nb - seq_len, seq_len, Xb.shape[1])
    strides = (Xb.strides[0], Xb.strides[0], Xb.strides[1])
    Xs = np.lib.stride_tricks.as_strided(
        Xb, shape=shape, strides=strides
    ).copy()  # .copy() obligatoire pour éviter les artefacts mémoire
    return Xs, yb[seq_len:]

# Split 70/15/15 PAR CELLULE
Xtr_l, ytr_l = [], []
Xva_l, yva_l = [], []
Xte_l, yte_l = [], []

cellules = np.unique(ids_all)
print(f"\n  Traitement de {len(cellules):,} cellules (stride_tricks)...")
t_loop = time.time()

for cell in cellules:
    mask = (ids_all == cell)
    Xc   = X_all_sc[mask]    # toutes les mesures de cette cellule
    yc   = y_all[mask]
    n    = len(Xc)

    if n <= SEQ_LEN + 3:
        continue              # cellule trop petite → ignorée

    # Split chronologique 70/15/15 pour CETTE cellule
    n_tr = int(n * 0.70)     # 70% → train
    n_va = int(n * 0.85)     # 85% → fin val (85-70=15% de val)
                              # reste → test (100-85=15% de test)

    # Séquences pour chaque portion
    xs, ys = seq_block(Xc[:n_tr],      yc[:n_tr],      SEQ_LEN)
    if xs is not None:
        Xtr_l.append(xs); ytr_l.append(ys)

    xs, ys = seq_block(Xc[n_tr:n_va], yc[n_tr:n_va], SEQ_LEN)
    if xs is not None:
        Xva_l.append(xs); yva_l.append(ys)

    xs, ys = seq_block(Xc[n_va:],     yc[n_va:],     SEQ_LEN)
    if xs is not None:
        Xte_l.append(xs); yte_l.append(ys)

print(f"  Boucle cellules terminée en {time.time()-t_loop:.1f}s")

# Concaténation
X_tr_s = np.concatenate(Xtr_l).astype(np.float32)
y_tr_s = np.concatenate(ytr_l).astype(np.int64)
X_va_s = np.concatenate(Xva_l).astype(np.float32)
y_va_s = np.concatenate(yva_l).astype(np.int64)
X_te_s = np.concatenate(Xte_l).astype(np.float32)
y_te_s = np.concatenate(yte_l).astype(np.int64)

# ── AFFICHAGE COMPLET DES SHAPES ─────────────────────────────
print(f"\n  ╔══ SHAPES COMPLETS ══════════════════════════════════╗")
print(f"  ║  AVANT séquences (2D) :                              ║")
print(f"  ║    X_all_sc : {str(X_all_sc.shape):<40}║")
print(f"  ║    Lecture  : [N_lignes, N_features]                 ║")
print(f"  ╠══ APRÈS séquences (3D) ═════════════════════════════╣")
print(f"  ║    X_train  : {str(X_tr_s.shape):<40}║")
print(f"  ║    X_val    : {str(X_va_s.shape):<40}║")
print(f"  ║    X_test   : {str(X_te_s.shape):<40}║")
print(f"  ║    y_train  : {str(y_tr_s.shape):<40}║")
print(f"  ║    y_val    : {str(y_va_s.shape):<40}║")
print(f"  ║    y_test   : {str(y_te_s.shape):<40}║")
print(f"  ║    Lecture  : [N_séquences, SEQ_LEN={SEQ_LEN}h, N_features={len(FEATURES)}]{'':10}║")
print(f"  ╠══ INTERPRÉTATION ══════════════════════════════════╣")
print(f"  ║    Chaque ligne = 1 séquence de {SEQ_LEN}h pour 1 cellule    ║")
print(f"  ║    Le LSTM lit ces {SEQ_LEN}h → prédit ce qui se passe +1h  ║")
print(f"  ╚════════════════════════════════════════════════════╝")

# Mémoire
mem_tr = X_tr_s.nbytes / 1024**3
mem_va = X_va_s.nbytes / 1024**3
mem_te = X_te_s.nbytes / 1024**3
print(f"\n  Mémoire :")
print(f"    X_train : {mem_tr:.2f} GB")
print(f"    X_val   : {mem_va:.2f} GB")
print(f"    X_test  : {mem_te:.2f} GB")
print(f"    Total   : {mem_tr + mem_va + mem_te:.2f} GB")

# Vérifications
assert len(X_va_s) > 0, "Val vide ! Réduire SEQ_LEN"
assert len(X_te_s) > 0, "Test vide ! Réduire SEQ_LEN"

# Distribution des classes
print(f"\n  Distribution target_1h (train) :")
for cls, nom in zip([0,1,2], ['Normal','Modéré','Critique']):
    n = (y_tr_s == cls).sum()
    pct = n / len(y_tr_s) * 100
    print(f"    Classe {cls} {nom:<10} : {n:,} ({pct:.1f}%)")

# Poids de classes — indispensable pour le déséquilibre 4.6% Critique
weights_arr = compute_class_weight('balanced',
                                    classes=np.array([0,1,2]),
                                    y=y_tr_s)
weights_t   = torch.tensor(weights_arr, dtype=torch.float32).to(DEVICE)
print(f"\n  Poids classes (balanced) :")
for cls, nom, w in zip([0,1,2], ['Normal','Modéré','Critique'], weights_arr):
    print(f"    Classe {cls} {nom:<10} : {w:.3f}")

print(f"  Durée étape : {time.time()-t3:.1f}s  [{elapsed()}]")

# ============================================================
# ÉTAPE 4 — SOUS-ÉCHANTILLON STRATIFIÉ POUR OPTUNA
# ============================================================
# Optuna cherche les meilleurs hyperparamètres.
# Faire tourner 15 trials sur 4.7M séquences prendrait des jours.
# On utilise 150k séquences stratifiées (même proportion des 3 classes)
# pour une recherche rapide et représentative.
#
# Après Optuna, le réentraînement final utilise TOUTES les séquences.

t4 = time.time()
print(f"\n[4/8] Sous-échantillon Optuna stratifié...  [{elapsed()}]")

rng   = np.random.default_rng(42)
idx_0 = np.where(y_tr_s == 0)[0]
idx_1 = np.where(y_tr_s == 1)[0]
idx_2 = np.where(y_tr_s == 2)[0]

# Proportions : 55% Normal, 35% Modéré, 10% Critique
# (surreprésente légèrement Critique car classe rare)
n0 = int(OPTUNA_SAMPLE * 0.55)
n1 = int(OPTUNA_SAMPLE * 0.35)
n2 = OPTUNA_SAMPLE - n0 - n1

sample_idx = np.sort(np.concatenate([
    rng.choice(idx_0, min(n0, len(idx_0)), replace=False),
    rng.choice(idx_1, min(n1, len(idx_1)), replace=False),
    rng.choice(idx_2, min(n2, len(idx_2)), replace=False),
]))
X_opt, y_opt = X_tr_s[sample_idx], y_tr_s[sample_idx]

# Val réduite pour Optuna (rapide)
val_idx  = rng.choice(len(X_va_s), min(50_000, len(X_va_s)), replace=False)
X_va_opt = X_va_s[val_idx]
y_va_opt = y_va_s[val_idx]

print(f"  Optuna train : {len(X_opt):,} séquences (stratifié)")
print(f"  Optuna val   : {len(X_va_opt):,} séquences")
print(f"  Shape Optuna train : {X_opt.shape}")
print(f"    → [N_opt={len(X_opt):,}, SEQ_LEN={SEQ_LEN}h, features={len(FEATURES)}]")

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

print(f"\n  Batches train (complet) : {len(full_tr_ld)}")
print(f"  Batches val   (complet) : {len(full_va_ld)}")
print(f"  Batches test            : {len(test_ld)}")
print(f"  Durée étape : {time.time()-t4:.1f}s")

# ============================================================
# ÉTAPE 5 — ARCHITECTURE LSTM + ATTENTION TEMPORELLE
# ============================================================
# FLOW COMPLET DU MODÈLE :
#
# Entrée x : [batch=4096, SEQ_LEN=6, features=19]
#     ↓
# LSTM (2 couches, hidden=128)
#     → out : [batch=4096, SEQ_LEN=6, hidden=128]
#     → chaque ligne de SEQ_LEN contient l'état caché de cette heure
#     ↓
# ATTENTION (nn.Linear(128, 1))
#     → scores : [batch=4096, SEQ_LEN=6]  un score par heure
#     → softmax → weights : [batch=4096, 6]  somme=1 par séquence
#     ↓
# PONDÉRATION (weights × out)
#     → context : [batch=4096, hidden=128]  résumé pondéré des 6h
#     ↓
# LayerNorm → Dropout
#     ↓
# Dense(128→fc_size) → ReLU → Dropout → Dense(fc_size→3)
#     → logits : [batch=4096, 3]
#     ↓
# Softmax (implicite dans CrossEntropyLoss)
#     → probabilités : [batch=4096, 3]  somme=1
#     → prédiction   : argmax → 0, 1 ou 2

class LSTMAttention(nn.Module):
    """
    LSTM + mécanisme d'attention temporelle pour H+1.

    Différence avec LSTM simple :
      LSTM simple → prend seulement le dernier état caché (out[:, -1, :])
      LSTM + Att  → calcule un poids pour chacune des 6 heures
                    → somme pondérée → plus d'information utilisée

    Paramètres :
      input_size  : nombre de features (19)
      hidden_size : taille de l'état caché LSTM (64, 128 ou 256)
      num_layers  : nombre de couches LSTM empilées (1 ou 2)
      dropout     : régularisation (0.1 à 0.4)
      fc_size     : taille de la couche Dense intermédiaire (32, 64, 128)
    """
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout, fc_size):
        super().__init__()

        # Couche LSTM principale
        self.lstm = nn.LSTM(
            input_size  = input_size,    # 19 features en entrée
            hidden_size = hidden_size,   # taille état caché
            num_layers  = num_layers,    # couches empilées
            batch_first = True,          # [batch, seq, features]
            # dropout entre les couches (seulement si >1 couche)
            dropout     = dropout if num_layers > 1 else 0.0
        )

        # Couche d'attention : un scalaire par heure
        # input  : [batch, seq, hidden]
        # output : [batch, seq, 1] → squeeze → [batch, seq]
        self.attn = nn.Linear(hidden_size, 1)

        # Normalisation de la couche (stabilise l'entraînement)
        self.norm = nn.LayerNorm(hidden_size)

        # Dropout sur le contexte
        self.drop = nn.Dropout(dropout)

        # Classifieur final
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_size),  # compression
            nn.ReLU(),                         # non-linéarité
            nn.Dropout(dropout),               # régularisation
            nn.Linear(fc_size, N_CLASSES)      # 3 classes
        )

    def forward(self, x):
        """
        x : [batch, SEQ_LEN, features]

        Retourne :
          logits : [batch, 3]       — scores bruts par classe
          attn_w : [batch, SEQ_LEN] — poids d'attention par heure
        """
        # LSTM : chaque heure génère un état caché
        out, _  = self.lstm(x)                          # [B, 6, hidden]

        # Attention : score d'importance pour chaque heure
        scores  = self.attn(out).squeeze(-1)            # [B, 6]

        # Softmax : les scores deviennent des probabilités (somme=1)
        attn_w  = torch.softmax(scores, dim=1)          # [B, 6]

        # Contexte : somme pondérée des états cachés
        # Chaque heure contribue proportionnellement à son poids
        context = (attn_w.unsqueeze(-1) * out).sum(1)   # [B, hidden]

        # Normalisation et dropout
        context = self.norm(context)

        # Classification finale
        return self.fc(self.drop(context)), attn_w      # [B,3], [B,6]

# ============================================================
# ÉTAPE 6 — OPTUNA RAPIDE
# ============================================================
# Optuna explore l'espace des hyperparamètres avec TPE
# (Tree-structured Parzen Estimator) — une méthode bayésienne.
# MedianPruner arrête les trials mauvais après 3 epochs.
# → gain de temps significatif

t6 = time.time()
print(f"\n[6/8] Optuna — {N_TRIALS} trials × {OPTUNA_EPOCHS} epochs..."
      f"  [{elapsed()}]")
print(f"  Méthode : TPE bayésien + MedianPruner")
print(f"  Chaque trial : {OPTUNA_SAMPLE:,} séquences (sous-échantillon)")

def objective(trial):
    # Hyperparamètres à optimiser
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    num_layers  = trial.suggest_int('num_layers', 1, 2)
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
            logits, _ = m(xb)          # [batch, 3]
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
print(f"  Durée Optuna       : {time.time()-t6:.1f}s  [{elapsed()}]")

# ============================================================
# ÉTAPE 7 — RÉENTRAÎNEMENT FINAL
# ============================================================
# Maintenant on réentraîne avec les MEILLEURS paramètres trouvés
# sur TOUTES les séquences train (4.7M) — pas le sous-échantillon.
# Train sur train uniquement, val sur val séparé → pas de fuite.

t7 = time.time()
print(f"\n[7/8] Réentraînement final...  [{elapsed()}]")
print(f"  Train : {len(X_tr_s):,} séquences")
print(f"  Val   : {len(X_va_s):,} séquences (séparé)")

bp         = study.best_params
best_model = LSTMAttention(
    len(FEATURES), bp['hidden_size'], bp['num_layers'],
    bp['dropout'], bp['fc_size']
).to(DEVICE)

n_params = sum(p.numel() for p in best_model.parameters()
               if p.requires_grad)

print(f"\n  Architecture finale :")
print(f"    LSTM : {bp['num_layers']} couche(s) × {bp['hidden_size']} hidden units")
print(f"    Attention : Linear({bp['hidden_size']}, 1) → Softmax")
print(f"    Dense : {bp['hidden_size']} → {bp['fc_size']} → 3")
print(f"    Paramètres entraînables : {n_params:,}")

# ── SHAPES DANS LE MODÈLE ────────────────────────────────────
print(f"\n  Shapes à travers le modèle (batch_size={BATCH_SIZE}) :")
print(f"    Entrée LSTM  : [{BATCH_SIZE}, {SEQ_LEN}, {len(FEATURES)}]")
print(f"      = [batch, {SEQ_LEN} heures, {len(FEATURES)} features]")
print(f"    Sortie LSTM  : [{BATCH_SIZE}, {SEQ_LEN}, {bp['hidden_size']}]")
print(f"      = [batch, {SEQ_LEN} heures, état caché par heure]")
print(f"    Scores att.  : [{BATCH_SIZE}, {SEQ_LEN}]")
print(f"      = [batch, 1 poids par heure]")
print(f"    Poids att.   : [{BATCH_SIZE}, {SEQ_LEN}]  (après softmax, somme=1)")
print(f"    Contexte     : [{BATCH_SIZE}, {bp['hidden_size']}]")
print(f"      = somme pondérée des états cachés")
print(f"    Sortie finale: [{BATCH_SIZE}, 3]")
print(f"      = [batch, P(Normal), P(Modéré), P(Critique)]")

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

print(f"\n  Entraînement sur {FINAL_EPOCHS} epochs max :")

for epoch in range(1, FINAL_EPOCHS + 1):
    t_ep = time.time()

    # ── Phase train ──────────────────────────────────────────
    best_model.train()
    total_loss = 0
    for xb, yb in full_tr_ld:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = best_model(xb)        # [batch, 3]
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(best_model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
    avg_loss = total_loss / len(y_tr_s)

    # ── Phase validation (par batches — pas tout d'un coup) ──
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
        torch.save(best_model.state_dict(), 'best_lstm_1h.pt')
        patience_ct = 0
    else:
        patience_ct += 1
        if patience_ct >= FINAL_PATIENCE:
            print(f"  ⏹ Early stopping epoch {epoch}")
            break

print(f"\n  ✅ Meilleur F1 val : {best_f1_val:.4f}")
print(f"  Durée réentraînement : {time.time()-t7:.1f}s  [{elapsed()}]")

# ============================================================
# ÉTAPE 8 — ÉVALUATION + GRAPHIQUES
# ============================================================

t8 = time.time()
print(f"\n[8/8] Évaluation test + graphiques...  [{elapsed()}]")

best_model.load_state_dict(
    torch.load('best_lstm_1h.pt', map_location=DEVICE)
)
best_model.eval()

preds_test, attn_all = [], []
with torch.no_grad():
    for xb, _ in test_ld:               # par batches → pas d'OOM
        logits, attn_w = best_model(xb.to(DEVICE))
        preds_test.extend(logits.argmax(1).cpu().numpy())
        attn_all.append(attn_w.cpu().numpy())

preds_test = np.array(preds_test)
attn_mean  = np.concatenate(attn_all).mean(axis=0)   # [SEQ_LEN]

print(f"\n  Shape preds_test : {preds_test.shape}")
print(f"  Shape attn_mean  : {attn_mean.shape}  → poids moyen par heure")

print("\n" + "=" * 65)
print(f"  RÉSULTATS FINAUX — {TARGET.upper()}")
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
print(f"  F1-Critique : {f1_cls2:.4f}  ({f1_cls2*100:.2f}%)  ← principale")

heure_max = np.argmax(attn_mean)
print(f"\n  Attention — heure la plus informative :")
print(f"    Position {heure_max} → t-{SEQ_LEN-heure_max}h "
      f"(poids={attn_mean[heure_max]:.4f})")

# ── Graphiques 2×2 ───────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f'LSTM + Attention — {TARGET.upper()}  |  SEQ_LEN={SEQ_LEN}h\n'
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

# G4 : Attention + comparaison horizons dans le même graphique
# Poids d'attention
colors = ['#D85A30' if w > attn_mean.mean() else '#1D9E75'
          for w in attn_mean]
axes[1,1].bar(range(SEQ_LEN), attn_mean, color=colors)
axes[1,1].axvline(heure_max, color='red', linestyle='--',
                   label=f'Max à t-{SEQ_LEN-heure_max}h')
axes[1,1].axhline(attn_mean.mean(), color='black', linestyle=':',
                   alpha=0.5, label=f'Moy={attn_mean.mean():.4f}')
axes[1,1].set_title('Poids attention temporelle\n'
                     '(rouge=heures les plus informatives)')
axes[1,1].set_xlabel(f'Position (0=t-{SEQ_LEN}h, {SEQ_LEN-1}=maintenant)')
axes[1,1].set_ylabel('Poids moyen')
axes[1,1].legend(fontsize=9)
axes[1,1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('lstm_1h_resultats.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ lstm_1h_resultats.png sauvegardé")

# ── Tableau comparatif multi-horizons ────────────────────────
print(f"\n{'='*65}")
print(f"  TABLEAU COMPARATIF MULTI-HORIZONS")
print(f"{'='*65}")
print(f"  {'Modèle':<18} {'Horizon':>8} {'Accuracy':>10} "
      f"{'F1-macro':>10} {'F1-Crit':>10} {'Params':>10}")
print(f"  {'-'*65}")

# LSTM H+1 (résultats actuels)
print(f"  {'LSTM+Attention':<18} {'+1h':>8} "
      f"{acc:>10.4f} {f1_macro:>10.4f} "
      f"{f1_cls2:>10.4f} {n_params:>10,}")

# LSTM H+3 (tes résultats précédents)
print(f"  {'LSTM+Attention':<18} {'+3h':>8} "
      f"{0.8968:>10.4f} {0.9221:>10.4f} "
      f"{0.9870:>10.4f} {'57,412':>10}")

# LSTM H+6 (tes résultats précédents)
print(f"  {'LSTM+Attention':<18} {'+6h':>8} "
      f"{0.8769:>10.4f} {0.9049:>10.4f} "
      f"{0.9805:>10.4f} {'301,060':>10}")

# GRU H+6 (ta binôme — à remplir)
gru6_acc = REF['GRU +6h']['acc']
if gru6_acc is not None:
    print(f"  {'GRU+Attention':<18} {'+6h':>8} "
          f"{gru6_acc:>10.4f} {REF['GRU +6h']['f1']:>10.4f} "
          f"{REF['GRU +6h']['f1cls2']:>10.4f} {'N/A':>10}")
else:
    print(f"  {'GRU+Attention':<18} {'+6h':>8} "
          f"{'(binôme)':>10} {'(binôme)':>10} "
          f"{'(binôme)':>10} {'N/A':>10}")

print(f"  {'-'*65}")
print(f"  → La dégradation progressive avec l'horizon est normale")
print(f"  → F1-Critique reste la métrique clé (détection congestion)")

# ============================================================
# RÉSUMÉ FINAL
# ============================================================

temps_total = time.time() - TIME_START
h = int(temps_total // 3600)
m = int((temps_total % 3600) // 60)
s = int(temps_total % 60)

print(f"\n{'='*65}")
print(f"  RÉSUMÉ FINAL — LSTM + ATTENTION — {TARGET.upper()}")
print(f"{'='*65}")
print(f"  Dataset          : {len(df):,} lignes")
print(f"  Cellules BTS     : {df['cellname_id'].nunique():,}")
print(f"  Features         : {len(FEATURES)}")
print(f"  SEQ_LEN          : {SEQ_LEN}h de contexte")
print(f"  Shapes finaux    :")
print(f"    X_train        : {X_tr_s.shape}")
print(f"    X_val          : {X_va_s.shape}")
print(f"    X_test         : {X_te_s.shape}")
print(f"  Architecture     : LSTM {bp['num_layers']}×{bp['hidden_size']} "
      f"+ Attention → Dense({bp['fc_size']}) → Softmax(3)")
print(f"  Paramètres       : {n_params:,}")
print(f"  Trials Optuna    : {N_TRIALS} (sous-éch. {OPTUNA_SAMPLE:,} stratifié)")
print(f"  Best params      : {study.best_params}")
print(f"  Split            : 70/15/15 par cellule chronologique")
print(f"  Accuracy         : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  F1-macro         : {f1_macro:.4f}  ({f1_macro*100:.2f}%)")
print(f"  F1-Critique      : {f1_cls2:.4f}  ({f1_cls2*100:.2f}%)")
print(f"  Modèle sauvé     : best_lstm_1h.pt")
print(f"  ⏱  TEMPS TOTAL  : {h}h {m}min {s}s")
print(f"{'='*65}")

print(f"\n  Poids d'attention — Top 3 heures prédictives :")
top3 = np.argsort(attn_mean)[-3:][::-1]
for rank, idx in enumerate(top3, 1):
    print(f"    Top {rank} : t-{SEQ_LEN-idx}h "
          f"(poids={attn_mean[idx]:.4f})")
print(f"\n  → Pour H+1, l'heure la plus récente devrait dominer")
print(f"    (cf. H+3 où t-1h avait poids=0.47)")
print(f"{'='*65}")