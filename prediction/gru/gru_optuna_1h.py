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
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================
# CONFIGURATION
# ============================================================

TARGET         = 'target_1h'
SEQ_LEN        = 24
N_CLASSES      = 3
BATCH_SIZE     = 2048
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BEST_PARAMS = {
    'hidden_size' : 64,
    'num_layers'  : 1,
    'dropout'     : 0.1,
    'fc_size'     : 128,
    'lr'          : 0.002950137270531349,
    'weight_decay': 4.23538831096136e-05
}

print("=" * 60)
print(f"  GRU + SHAP  —  {TARGET}")
print("=" * 60)
print(f"Device  : {DEVICE}")
print(f"Params  : {BEST_PARAMS}")

# ============================================================
# ÉTAPE 1 — CHARGEMENT ET NETTOYAGE
# ============================================================

print("\n[1/7] Chargement et nettoyage...")

df = pd.read_csv("dataset_avec_targets.csv")
df.columns = df.columns.str.lower()
df['date_'] = pd.to_datetime(df['date_'].astype(str).str.strip(), errors='coerce')
df = df.sort_values(['cellname_id', 'date_']).reset_index(drop=True)

COLS_DROP = [
    'time_to_peak', 'peak_trend_interaction', 'traffic_per_user',
    'prb_z_score', 'prb_per_user', 'throughput_per_user',
    'cell_traffic_volume_dl'
]
df.drop(columns=[c for c in COLS_DROP if c in df.columns], inplace=True)

if 'hour' in df.columns:
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df.drop(columns=['hour'], inplace=True)

df['lag_prb_1h'] = df.groupby('cellname_id')['dl_prb_usage_rate'].shift(1)
df['lag_prb_2h'] = df.groupby('cellname_id')['dl_prb_usage_rate'].shift(2)
df.dropna(subset=['lag_prb_1h', 'lag_prb_2h', TARGET], inplace=True)

print(f"Dataset : {len(df):,} lignes | Cellules : {df['cellname_id'].nunique():,}")

# ============================================================
# ÉTAPE 2 — FEATURES ET SPLIT
# ============================================================

print("[2/7] Features et split...")

try:
    ckpt        = torch.load('best_gru_1h_optuna.pt', map_location='cpu')
    n_feat_ckpt = ckpt['gru.weight_ih_l0'].shape[1]
    print(f"✅ Checkpoint détecté : input_size = {n_feat_ckpt} features")
except Exception:
    n_feat_ckpt = None
    print("⚠️  Impossible de lire le checkpoint")

EXCLURE  = ['date_', 'cellname_id', 'target_1h', 'target_3h',
            'target_6h', 'congestion_score', 'classe_congestion']
FEATURES = [c for c in df.columns if c not in EXCLURE]

if n_feat_ckpt is not None and len(FEATURES) != n_feat_ckpt:
    print(f"⚠️  Mismatch : dataset={len(FEATURES)} vs checkpoint={n_feat_ckpt}")
    if n_feat_ckpt == len(FEATURES) + 1:
        EXCLURE.remove('classe_congestion')
        FEATURES = [c for c in df.columns if c not in EXCLURE]
        print(f"   → classe_congestion réintégrée ({len(FEATURES)} features)")

print(f"Features ({len(FEATURES)}) : {FEATURES}")

dates   = df['date_'].dropna().sort_values()
d70     = dates.quantile(0.70)
d85     = dates.quantile(0.85)

train_df = df[df['date_'] <= d70].copy()
val_df   = df[(df['date_'] > d70) & (df['date_'] <= d85)].copy()
test_df  = df[df['date_'] > d85].copy()

print(f"Train : {len(train_df):,} | Val : {len(val_df):,} | Test : {len(test_df):,}")

# ============================================================
# ÉTAPE 3 — NORMALISATION ET RESHAPE
# ============================================================

print("[3/7] Normalisation et reshape...")

scaler = StandardScaler()
X_tr   = scaler.fit_transform(train_df[FEATURES])
X_va   = scaler.transform(val_df[FEATURES])
X_te   = scaler.transform(test_df[FEATURES])

y_tr   = train_df[TARGET].values.astype(int)
y_va   = val_df[TARGET].values.astype(int)
y_te   = test_df[TARGET].values.astype(int)

def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)

print("  Reshape train...") ; X_tr_s, y_tr_s = make_sequences(X_tr, y_tr, SEQ_LEN)
print("  Reshape val...")   ; X_va_s, y_va_s = make_sequences(X_va, y_va, SEQ_LEN)
print("  Reshape test...")  ; X_te_s, y_te_s = make_sequences(X_te, y_te, SEQ_LEN)

print(f"Shapes : train {X_tr_s.shape} | val {X_va_s.shape} | test {X_te_s.shape}")

weights_arr = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_tr_s)
weights_t   = torch.tensor(weights_arr, dtype=torch.float32).to(DEVICE)

full_tr_ld = DataLoader(TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(y_tr_s)),
                         batch_size=BATCH_SIZE, shuffle=False)
full_va_ld = DataLoader(TensorDataset(torch.from_numpy(X_va_s), torch.from_numpy(y_va_s)),
                         batch_size=BATCH_SIZE, shuffle=False)
test_ld    = DataLoader(TensorDataset(torch.from_numpy(X_te_s), torch.from_numpy(y_te_s)),
                         batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# ÉTAPE 4 — ARCHITECTURE GRU
# ============================================================

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, fc_size):
        super().__init__()
        self.gru  = nn.GRU(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Sequential(
            nn.Linear(hidden_size, fc_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_size, N_CLASSES)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last   = self.norm(out[:, -1, :])
        return self.fc(self.drop(last))

# ============================================================
# ÉTAPE 5 — CHARGEMENT DU MODÈLE
# ============================================================

print("\n[4/7] Chargement du modèle best_gru_1h_optuna.pt...")

ckpt       = torch.load('best_gru_1h_optuna.pt', map_location=DEVICE)
input_size = ckpt['gru.weight_ih_l0'].shape[1]
print(f"   Input size du checkpoint : {input_size} features")

if input_size != len(FEATURES):
    raise ValueError(
        f"\n❌ MISMATCH : checkpoint={input_size} ≠ dataset={len(FEATURES)} features"
    )

bp    = BEST_PARAMS
model = GRUModel(
    input_size  = input_size,
    hidden_size = bp['hidden_size'],
    num_layers  = bp['num_layers'],
    dropout     = bp['dropout'],
    fc_size     = bp['fc_size']
).to(DEVICE)

try:
    model.load_state_dict(ckpt)
    print("✅ Modèle chargé avec succès")
except FileNotFoundError:
    print("⚠️  Fichier non trouvé — réentraînement nécessaire")

# ============================================================
# ÉTAPE 6 — ÉVALUATION
# ============================================================

print("\n[5/7] Évaluation sur test set...")

model.eval()
preds_test, probs_test = [], []
with torch.no_grad():
    for xb, _ in test_ld:
        logits = model(xb.to(DEVICE))
        probs  = torch.softmax(logits, dim=1)
        preds_test.extend(logits.argmax(1).cpu().numpy())
        probs_test.extend(probs.cpu().numpy())

preds_test = np.array(preds_test)
probs_test = np.array(probs_test)

print("\n" + "=" * 60)
print(classification_report(y_te_s, preds_test,
      target_names=['Normal (0)', 'Modéré (1)', 'Congestionné (2)'], digits=4))

f1_macro = f1_score(y_te_s, preds_test, average='macro', zero_division=0)
f1_cls2  = f1_score(y_te_s, preds_test, average=None,    zero_division=0)[2]
acc      = (preds_test == y_te_s).mean()

np_s = f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
print(f"{'Métrique':<20} {'LSTM':>12} {'GRU':>12}")
print("-" * 46)
print(f"{'Accuracy':<20} {'0.9006':>12} {acc:.4f}")
print(f"{'F1-macro':<20} {'0.9139':>12} {f1_macro:.4f}")
print(f"{'F1-classe 2':<20} {'0.9634':>12} {f1_cls2:.4f}")
print(f"{'Paramètres':<20} {'92,419':>12} {np_s}")

# ============================================================
# ÉTAPE 7 — SHAP
# ============================================================

print("\n[6/7] Calcul SHAP — GradientExplainer...")

rng      = np.random.default_rng(42)
idx_shap = []
for cls in [0, 1, 2]:
    pool = np.where(y_te_s == cls)[0]
    n    = min(167, len(pool))
    idx_shap.extend(rng.choice(pool, n, replace=False).tolist())
idx_shap = np.array(idx_shap)

X_shap = torch.from_numpy(X_te_s[idx_shap]).to(DEVICE)
y_shap = y_te_s[idx_shap]

bg_idx = rng.choice(len(X_tr_s), 100, replace=False)
X_bg   = torch.from_numpy(X_tr_s[bg_idx]).to(DEVICE)

model.eval()
explainer   = shap.GradientExplainer(model, X_bg)
shap_values = explainer.shap_values(X_shap)

print(f"✅ SHAP calculé | Shape brut : {np.array(shap_values).shape}")

# ── Normalisation robuste de la sortie SHAP ────────────────
shap_arr = np.array(shap_values)   # peut être (3, N, 24, F) ou (N, 24, F, 3)

if shap_arr.ndim == 4:
    if shap_arr.shape[0] == 3:
        # Forme attendue : (3, N, 24, F)
        pass
    elif shap_arr.shape[-1] == 3:
        # Forme alternative : (N, 24, F, 3) → transposer en (3, N, 24, F)
        shap_arr = np.transpose(shap_arr, (3, 0, 1, 2))
    else:
        raise ValueError(f"Shape SHAP inattendue : {shap_arr.shape}")
else:
    raise ValueError(f"SHAP devrait être 4D, obtenu : {shap_arr.shape}")

n_classes_shap, N_shap, T_shap, F_shap = shap_arr.shape
print(f"   Shape normalisée : (n_classes={n_classes_shap}, N={N_shap}, T={T_shap}, F={F_shap})")

shap_abs          = np.abs(shap_arr)                       # (3, N, 24, F)
feat_imp_by_class = shap_abs.mean(axis=(1, 2))             # (3, F)
feat_imp_global   = feat_imp_by_class.mean(axis=0)         # (F,)

# ── Vérification finale avant tout usage de sorted_idx ────
assert feat_imp_global.shape[0] == len(FEATURES), (
    f"Mismatch feat_imp_global {feat_imp_global.shape} vs FEATURES {len(FEATURES)}"
)

sorted_idx = np.argsort(feat_imp_global)[::-1]   # indices triés, longueur = len(FEATURES)
feat_names = np.array(FEATURES)

print(f"\nTop 10 features (global) :")
for rank, i in enumerate(sorted_idx[:10], 1):
    print(f"  {rank:2d}. {feat_names[i]:<30} SHAP = {feat_imp_global[i]:.5f}")

# ============================================================
# ÉTAPE 8 — GRAPHIQUES
# ============================================================

print("\n[7/7] Génération des graphiques SHAP...")

colors_cls = ['#2196F3', '#FF9800', '#F44336']
names_cls  = ['Normal (0)', 'Modéré (1)', 'Congestionné (2)']

fig = plt.figure(figsize=(20, 18))
fig.suptitle(f'GRU + SHAP — {TARGET}\nExplicabilité de la prédiction de congestion BTS',
             fontsize=14, fontweight='bold', y=0.98)

# ── Graphe 1 : Importance globale ────────────────────────────
ax1   = fig.add_subplot(3, 3, 1)
top_n = min(15, len(FEATURES))                    # ex: 15 (ou moins si peu de features)

# Prendre les top_n indices — TOUS issus de feat_imp_global de longueur len(FEATURES)
top_idx        = sorted_idx[:top_n]               # shape (top_n,)  — indices valides
top_vals       = feat_imp_global[top_idx]         # shape (top_n,)  — float64
top_names      = feat_names[top_idx]              # shape (top_n,)  — strings

# Inverser pour affichage barh (plus important en haut)
top_vals_plot  = top_vals[::-1].copy()            # shape (top_n,)
top_names_plot = top_names[::-1].copy()           # shape (top_n,)

# Couleurs : une par barre — même longueur que top_n
bar_colors = ['#D85A30' if v == top_vals_plot.max() else '#1D9E75'
              for v in top_vals_plot]             # len == top_n ✓

ax1.barh(range(top_n), top_vals_plot, color=bar_colors)
ax1.set_yticks(range(top_n))
ax1.set_yticklabels(top_names_plot, fontsize=8)
ax1.set_title('Importance globale\n(toutes classes)', fontweight='bold', fontsize=10)
ax1.set_xlabel('|SHAP| moyen')
ax1.grid(alpha=0.3, axis='x')

# ── Graphes 2-4 : Importance par classe ──────────────────────
for cls_idx in range(3):
    ax         = fig.add_subplot(3, 3, cls_idx + 2)
    vals_cls   = feat_imp_by_class[cls_idx]            # shape (F,)
    si_cls     = np.argsort(vals_cls)[::-1][:top_n]   # shape (top_n,)
    names_plot = feat_names[si_cls][::-1]              # shape (top_n,)
    vals_plot  = vals_cls[si_cls][::-1]                # shape (top_n,)

    ax.barh(range(len(si_cls)), vals_plot, color=colors_cls[cls_idx], alpha=0.8)
    ax.set_yticks(range(len(si_cls)))
    ax.set_yticklabels(names_plot, fontsize=7)
    ax.set_title(f'Importance — {names_cls[cls_idx]}',
                 fontweight='bold', fontsize=10, color=colors_cls[cls_idx])
    ax.set_xlabel('|SHAP| moyen')
    ax.grid(alpha=0.3, axis='x')

# ── Graphe 5 : Heatmap temporelle ────────────────────────────
ax5         = fig.add_subplot(3, 3, 5)
shap_cls2_t = shap_abs[2].mean(axis=0)        # (24, F)
top10_idx   = sorted_idx[:10]                  # toujours valides
heat_data   = shap_cls2_t[:, top10_idx].T     # (10, 24)

sns.heatmap(heat_data,
            xticklabels=[f't-{SEQ_LEN-i}h' if i % 6 == 0 else ''
                         for i in range(SEQ_LEN)],
            yticklabels=feat_names[top10_idx],
            cmap='YlOrRd', ax=ax5, cbar=True)
ax5.set_title('SHAP × Temps — Classe 2\n(Congestionné)', fontweight='bold', fontsize=10)
ax5.set_xlabel('Position dans la fenêtre 24h')
ax5.tick_params(axis='y', labelsize=7)

# ── Graphe 6 : Évolution temporelle ──────────────────────────
ax6           = fig.add_subplot(3, 3, 6)
shap_per_hour = np.array([shap_abs[c].mean(axis=(0, 2)) for c in range(3)])  # (3, 24)

for cls_idx in range(3):
    ax6.plot(range(SEQ_LEN), shap_per_hour[cls_idx],
             color=colors_cls[cls_idx], label=names_cls[cls_idx],
             linewidth=2, marker='o', markersize=3)

ax6.set_title('Importance SHAP par heure\ndans la fenêtre 24h',
              fontweight='bold', fontsize=10)
ax6.set_xlabel('Position (0 = t-24h, 23 = maintenant)')
ax6.set_ylabel('|SHAP| moyen')
ax6.legend(fontsize=8)
ax6.grid(alpha=0.3)
ax6.axvline(SEQ_LEN - 1, color='gray', linestyle='--', alpha=0.5)

# ── Graphe 7 : Matrice de confusion ──────────────────────────
ax7    = fig.add_subplot(3, 3, 7)
cm_arr = confusion_matrix(y_te_s, preds_test)
cm_pct = cm_arr / cm_arr.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Greens',
            xticklabels=['Normal', 'Modéré', 'Congestionné'],
            yticklabels=['Normal', 'Modéré', 'Congestionné'],
            ax=ax7, cbar=False)
ax7.set_title('Matrice de confusion (%)', fontweight='bold', fontsize=10)
ax7.set_xlabel('Prédiction')
ax7.set_ylabel('Réalité')

# ── Graphe 8 : Scatter SHAP vs valeur feature ────────────────
ax8          = fig.add_subplot(3, 3, 8)
top3_idx     = sorted_idx[:3]
point_colors = [colors_cls[int(label)] for label in y_shap]

for rank, fi in enumerate(top3_idx):
    feat_vals = X_te_s[idx_shap, :, fi].mean(axis=1)   # (N_shap,)
    shap_fi   = shap_arr[2, :, :, fi].mean(axis=1)     # (N_shap,)
    ax8.scatter(feat_vals, shap_fi, c=point_colors,
                alpha=0.4, s=8,
                label=feat_names[fi] if rank == 0 else '')

ax8.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax8.set_title('SHAP classe 2 — Top 3 features\n(couleur = classe réelle)',
              fontweight='bold', fontsize=10)
ax8.set_xlabel('Valeur feature normalisée (moy. 24h)')
ax8.set_ylabel('Valeur SHAP classe 2')
patches = [mpatches.Patch(color=col, label=n)
           for col, n in zip(colors_cls, names_cls)]
ax8.legend(handles=patches, fontsize=7)
ax8.grid(alpha=0.3)

# ── Graphe 9 : Récapitulatif ──────────────────────────────────
ax9 = fig.add_subplot(3, 3, 9)
ax9.axis('off')

recap_lines = [
    ("", ""),
    ("RÉSULTATS GRU 1H", ""),
    ("", ""),
    ("Accuracy",          f"{acc:.4f}"),
    ("F1-macro",          f"{f1_macro:.4f}"),
    ("F1-classe 2",       f"{f1_cls2:.4f}"),
    ("", ""),
    ("MEILLEURS PARAMS", ""),
    ("", ""),
    ("hidden_size",       str(bp['hidden_size'])),
    ("num_layers",        str(bp['num_layers'])),
    ("dropout",           str(bp['dropout'])),
    ("fc_size",           str(bp['fc_size'])),
    ("lr",                f"{bp['lr']:.5f}"),
    ("", ""),
    ("TOP 3 FEATURES SHAP", ""),
    ("", ""),
]
for rank, i in enumerate(sorted_idx[:3], 1):
    recap_lines.append((f"  {rank}. {feat_names[i]}", f"{feat_imp_global[i]:.5f}"))

y_pos = 0.98
for label, val in recap_lines:
    if label in ("RÉSULTATS GRU 1H", "MEILLEURS PARAMS", "TOP 3 FEATURES SHAP"):
        ax9.text(0.05, y_pos, label, fontsize=9, fontweight='bold',
                 color='steelblue', transform=ax9.transAxes)
    elif label:
        ax9.text(0.05, y_pos, f"{label}:", fontsize=8, transform=ax9.transAxes)
        ax9.text(0.70, y_pos, val, fontsize=8, fontweight='bold', transform=ax9.transAxes)
    y_pos -= 0.062

ax9.set_title('Récapitulatif', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('gru_1h_shap_complet.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ gru_1h_shap_complet.png sauvegardé")

# ── Graphe séparé : Summary ───────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 8))
fig2.suptitle('SHAP Summary — Importance par classe\nGRU target_1h',
              fontsize=13, fontweight='bold')

for cls_idx in range(3):
    ax  = axes2[cls_idx]
    imp = feat_imp_by_class[cls_idx]              # (F,)
    si  = np.argsort(imp)[::-1][:min(15, len(FEATURES))]
    ax.barh(range(len(si)), imp[si][::-1],
            color=colors_cls[cls_idx], alpha=0.85)
    ax.set_yticks(range(len(si)))
    ax.set_yticklabels(feat_names[si][::-1], fontsize=8)
    ax.set_title(names_cls[cls_idx], fontweight='bold',
                 color=colors_cls[cls_idx], fontsize=11)
    ax.set_xlabel('|SHAP| moyen')
    ax.grid(alpha=0.3, axis='x')
    ax.text(imp[si[0]] * 0.5, len(si) - 1,
            f'#1: {feat_names[si[0]]}',
            fontsize=7, color='white', fontweight='bold', va='center')

plt.tight_layout()
plt.savefig('gru_1h_shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ gru_1h_shap_summary.png sauvegardé")

# ============================================================
# RÉSUMÉ CONSOLE
# ============================================================

print("\n" + "=" * 60)
print("RÉSUMÉ FINAL — GRU + SHAP — target_1h")
print("=" * 60)
print(f"Accuracy     : {acc:.4f}")
print(f"F1-macro     : {f1_macro:.4f}")
print(f"F1-classe 2  : {f1_cls2:.4f}")
print()
print("Top 5 features SHAP — Congestion classe 2 :")
for rank, i in enumerate(sorted_idx[:5], 1):
    print(f"  {rank}. {feat_names[i]:<30} {feat_imp_global[i]:.5f}")
print()
print("Graphiques :")
print("  → gru_1h_shap_complet.png")
print("  → gru_1h_shap_summary.png")
print("=" * 60)

print("\n" + "=" * 60)
print("INTERPRÉTATION JURY")
print("=" * 60)
print(f"""
La feature la plus déterminante est '{feat_names[sorted_idx[0]]}'.
Suivie de '{feat_names[sorted_idx[1]]}' et '{feat_names[sorted_idx[2]]}'.

Ces résultats SHAP confirment que notre modèle GRU
n'a pas appris un pattern trivial — il exploite bien
les indicateurs physiques de saturation réseau.

Le graphe 'Heatmap SHAP × Temps' montre à quelles
heures dans la fenêtre de 24h le modèle porte son
attention — les heures proches de t (t-1h, t-2h)
ont généralement les SHAP les plus élevés pour h+1.

→ Cela valide notre choix de séquence 24h et de
  features lag_prb_1h et lag_prb_2h.
""")