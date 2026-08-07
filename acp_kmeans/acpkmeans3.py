import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score,
                             davies_bouldin_score,
                             calinski_harabasz_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# CHARGEMENT DU DATASET

print("ÉTAPE 1 — Chargement")

df = pd.read_csv('dataset_avec_features2.csv')
print(f"Dataset chargé : {len(df):,} lignes, {df.shape[1]} colonnes")


# DÉFINITION DES KPIs

# KPI_POSITIVE : plus la valeur est haute → plus la congestion est forte
# KPI_NEGATIVE : plus la valeur est haute → moins la congestion est forte
#                (ils seront inversés avant le calcul du score)

print("ÉTAPE 2 — Définition des KPIs")


KPI_POSITIVE = [
    'DL_PRB_Usage_Rate',      # Taux utilisation ressources radio (%) → congestion si élevé
    'PRB_per_User',           # PRB par utilisateur → surcharge si élevé
    'PRB_Z_Score',            # Anomalie PRB vs normale de la cellule
    'Gradient_PRB',           # Vitesse de montée du PRB (diff heure précédente)
    'Rolling_PRB_3h',         # Tendance PRB sur 3h → congestion imminente si monte
    'Cell_Traffic_Volume_DL', # Volume trafic descendant (GB)
    'Avg_User_NB',            # Nombre moyen utilisateurs actifs
]

KPI_NEGATIVE = [
    'DL_Average_Throughput',    # Débit descendant → bas = congestion
    'LTE_Setup_Success_Rate',   # Taux succès connexions → bas = problème
    'Avaibility',               # Disponibilité cellule → 0 = crash
    'Spectral_Eff',             # Efficacité spectrale → bas = saturation
    
]

all_kpis = KPI_POSITIVE + KPI_NEGATIVE

# Vérification que toutes les colonnes existent dans le dataset
manquants = [k for k in all_kpis if k not in df.columns]
if manquants:
    raise SystemExit(f"ERREUR — KPIs manquants dans le dataset : {manquants}")
print(f"KPIs positifs : {len(KPI_POSITIVE)}")
print(f"KPIs négatifs : {len(KPI_NEGATIVE)}")
print(f"Total KPIs    : {len(all_kpis)}")
print("Tous les KPIs présents ")

#  NORMALISATION EN 3 PHASES

# Phase A : RobustScaler
#   → Résistant aux outliers (607 users, 48GB/h conservés intentionnellement)
#   → Utilise la médiane et l'IQR au lieu de la moyenne et l'écart-type
#
# Phase B : Inversion des KPI négatifs
#   → Après inversion, TOUS les KPIs pointent dans le même sens :
#     valeur haute = congestion forte
#   → Indispensable pour que l'ACP construise un axe de congestion cohérent
#
# Phase C : MinMaxScaler [0, 1]
#   → Ramène tout entre 0 et 1 pour que tous les KPIs aient le même poids
#     dans le calcul du score de congestion final


print("ÉTAPE 3 — Normalisation (RobustScaler → inversion → MinMax)")


# Remplacement des NaN résiduels éventuels par la médiane
df_kpi = df[all_kpis].copy().fillna(df[all_kpis].median())

# Phase A : RobustScaler
robust = RobustScaler()
df_scaled = pd.DataFrame(
    robust.fit_transform(df_kpi),
    columns=all_kpis
)

# Phase B : Inversion des KPI négatifs
# Après cette étape : valeur haute = congestion élevée pour TOUS les KPIs
for col in KPI_NEGATIVE:
    df_scaled[col] = -df_scaled[col]

# Phase C : MinMaxScaler → ramène tout entre 0 et 1
minmax = MinMaxScaler()
df_norm = pd.DataFrame(
    minmax.fit_transform(df_scaled),
    columns=all_kpis
)
print("Normalisation terminée ")
print(f"Valeurs min/max après normalisation : {df_norm.values.min():.3f} / {df_norm.values.max():.3f}")


# ACP COMPLÈTE

# L'ACP a deux rôles dans ce pipeline :
#
# Rôle 1 : PC1 → calcul du score de congestion scalaire [0,1]
#   Les loadings de PC1 donnent l'importance de chaque KPI
#   On normalise ces loadings pour obtenir des coefficients de pondération
#   → Le score est une combinaison linéaire pondérée de tous les KPIs
#
# Rôle 2 : PC1 + PC2 → espace de clustering 2D
#   Le K-Means travaille dans cet espace à 2 dimensions
#   qui capture ~85% de l'information totale
#   → C'est la vraie combinaison ACP+KMeans

print("ÉTAPE 4 — ACP complète")


pca = PCA()
pca.fit(df_norm)

# Variance expliquée par composante
cumvar   = np.cumsum(pca.explained_variance_ratio_)
var_pc1  = pca.explained_variance_ratio_[0] * 100
var_pc2  = pca.explained_variance_ratio_[1] * 100
var_total_2d = var_pc1 + var_pc2

print(f"Variance PC1         : {var_pc1:.1f}%")
print(f"Variance PC2         : {var_pc2:.1f}%")
print(f"Variance PC1+PC2     : {var_total_2d:.1f}%  ← utilisée pour K-Means")
print(f"Composantes pour 80% : {np.argmax(cumvar >= 0.80) + 1}")
print(f"Composantes pour 95% : {np.argmax(cumvar >= 0.95) + 1}")

# Loadings PC1 — importance de chaque KPI dans la première composante
# Si la moyenne des loadings est négative, on retourne le signe
# pour que "valeur haute = congestion forte" (convention cohérente)
loadings_pc1 = pca.components_[0].copy()
if loadings_pc1.mean() < 0:
    loadings_pc1 = -loadings_pc1

# Coefficients de pondération = valeur absolue normalisée des loadings
loadings_abs  = np.abs(loadings_pc1)
coefficients  = loadings_abs / loadings_abs.sum()

# DataFrame des coefficients pour analyse et export
coeff_df = pd.DataFrame({
    'KPI'        : all_kpis,
    'Loading_PC1': loadings_pc1,
    'Coefficient': coefficients,
    'Type'       : (['Positive'] * len(KPI_POSITIVE) +
                    ['Negative'] * len(KPI_NEGATIVE))
}).sort_values('Coefficient', ascending=False).reset_index(drop=True)

print("\nImportance des KPIs (PC1) :")
for _, row in coeff_df.iterrows():
    bar = '█' * int(row['Coefficient'] * 100)
    print(f"  {row['KPI']:<30} {row['Coefficient']:.4f}  {bar}")

# Calcule du SCORE DE CONGESTION [0, 1]

# Score = combinaison linéaire pondérée de tous les KPIs normalisés
# Pondération = coefficients dérivés des loadings de PC1
# Normalisation finale → score strictement entre 0 et 1
# 0 = cellule parfaitement saine | 1 = congestion maximale
print("\n" + "=" * 60)
print("ÉTAPE 5 — Score de congestion")
print("=" * 60)

score_raw = df_norm[all_kpis].values @ coefficients
smin, smax = score_raw.min(), score_raw.max()
df['Congestion_Score'] = (score_raw - smin) / (smax - smin)

print(f"Score min  : {df['Congestion_Score'].min():.4f}")
print(f"Score max  : {df['Congestion_Score'].max():.4f}")
print(f"Score moy  : {df['Congestion_Score'].mean():.4f}")
print(f"Score méd  : {df['Congestion_Score'].median():.4f}")

# PROJECTION ACP 2D

# Projection de toutes les observations sur PC1 et PC2
# Ce sont les coordonnées de chaque cellule-heure dans l'espace ACP
# C'est dans CET espace que le K-Means va chercher les clusters

print("\n" + "=" * 60)
print("ÉTAPE 6 — Projection ACP 2D")
print("=" * 60)

X_pca_full = pca.transform(df_norm)
X_pca_2d   = X_pca_full[:, :2]   # On garde seulement PC1 et PC2

print(f"Shape espace ACP 2D : {X_pca_2d.shape}")
print(f"PC1 range : [{X_pca_2d[:,0].min():.3f}, {X_pca_2d[:,0].max():.3f}]")
print(f"PC2 range : [{X_pca_2d[:,1].min():.3f}, {X_pca_2d[:,1].max():.3f}]")


# ÉTAPE 7 — MÉTHODE ELBOW (justification k)

# On calcule l'inertie pour k=1 à 7 dans l'espace ACP 2D
# Sur un sous-échantillon de 500 000 points pour la vitesse
# L'objectif est de trouver le k optimal


print("ÉTAPE 7 — Méthode Elbow (k=1..7)")


sample_size = min(50_000, len(df))
idx_s       = np.random.RandomState(42).choice(len(df), sample_size, replace=False)
X_sample    = X_pca_2d[idx_s]

inertia_vals = []
for k in range(1, 8):
    km_tmp = KMeans(n_clusters=k, n_init=10, max_iter=200, random_state=42)
    km_tmp.fit(X_sample)
    inertia_vals.append(km_tmp.inertia_)
    print(f"  k={k} → inertie = {km_tmp.inertia_:,.1f}")
print("Elbow calculé ")



# K-MEANS k=3 DANS L'ESPACE ACP 2D

# Le K-Means travaille sur les 2 composantes principales
# n_init=20 : 20 initialisations aléatoires → on garde la meilleure
# max_iter=500 : convergence garantie même sur grands datasets
# algorithm='lloyd' : algorithme standard, stable

print("\n" + "=" * 60)
print("ÉTAPE 8 — K-Means k=3 en espace ACP 2D")
print("=" * 60)

kmeans = KMeans(
    n_clusters=3,
    n_init=20,
    max_iter=500,
    random_state=42,
    algorithm='lloyd'
)
labels_raw  = kmeans.fit_predict(X_pca_2d)
centers_2d  = kmeans.cluster_centers_   # Shape (3, 2) — vrais centres K-Means


# ── TRI DES CLASSES PAR SCORE DE CONGESTION MOYEN ────────────
# Le K-Means retourne des labels arbitraires (0, 1, 2)
# On les réordonne pour que :
#   Classe 0 = profil le moins congestionné (score moyen le plus bas)
#   Classe 1 = profil modérément congestionné
#   Classe 2 = profil le plus congestionné (score moyen le plus haut)
score_par_cluster = {
    k: df['Congestion_Score'].values[labels_raw == k].mean()
    for k in range(3)
}
sorted_by_score = sorted(score_par_cluster.items(), key=lambda x: x[1])
label_map = {sorted_by_score[i][0]: i for i in range(3)}
df['Classe_Congestion'] = np.vectorize(label_map.get)(labels_raw)

# Centres triés dans l'espace 2D (pour les graphiques)
centers_sorted_2d = np.array([
    centers_2d[sorted_by_score[i][0]] for i in range(3)
])

# Score moyen par classe (pour les seuils et les graphiques)
scores_par_classe = {
    cls: df.loc[df['Classe_Congestion'] == cls, 'Congestion_Score']
    for cls in [0, 1, 2]
}


# La méthode par scores moyens est plus stable et robuste
seuil_01 = (scores_par_classe[0].mean() + scores_par_classe[1].mean()) / 2
seuil_12 = (scores_par_classe[1].mean() + scores_par_classe[2].mean()) / 2

dist = df['Classe_Congestion'].value_counts(normalize=True).sort_index()
n_total = len(df)

print(f"Classe 0 Normal   : {dist.get(0,0)*100:.1f}%  ({int(dist.get(0,0)*n_total):,} obs.)")
print(f"Classe 1 Modéré   : {dist.get(1,0)*100:.1f}%  ({int(dist.get(1,0)*n_total):,} obs.)")
print(f"Classe 2 Critique : {dist.get(2,0)*100:.1f}%  ({int(dist.get(2,0)*n_total):,} obs.)")
print(f"Seuil 0↔1 : {seuil_01:.4f}")
print(f"Seuil 1↔2 : {seuil_12:.4f}")
print(f"Cohérence des seuils : {' OK' if seuil_01 < seuil_12 else ' PROBLÈME'}")

for cls in [0, 1, 2]:
    print(f"Score moyen Classe {cls} : {scores_par_classe[cls].mean():.4f}")



# MÉTRIQUES DE QUALITÉ DU CLUSTERING

# Silhouette  [−1, 1]  → proche de 1 = clusters bien séparés
# Davies-Bouldin [0, ∞] → proche de 0 = clusters compacts et séparés
# Calinski-Harabasz     → plus élevé = meilleur (pas de borne max)


print("ÉTAPE 9 — Métriques de qualité (espace 2D)")


sil = silhouette_score(X_pca_2d[idx_s], df['Classe_Congestion'].values[idx_s])
db  = davies_bouldin_score(X_pca_2d, df['Classe_Congestion'].values)
ch  = calinski_harabasz_score(X_pca_2d, df['Classe_Congestion'].values)

print(f"Silhouette        = {sil:.4f}  (idéal → 1)   {'BON ' if sil >= 0.5 else 'ACCEPTABLE'}")
print(f"Davies-Bouldin    = {db:.4f}  (idéal → 0)   {'BON ' if db <= 1.0 else 'MOYEN'}")
print(f"Calinski-Harabasz = {ch:,.0f}")

# VISUALISATIONS
COLORS = {0: '#27AE60', 1: '#F39C12', 2: '#E74C3C'}
LABELS = {0: 'Classe 0 — Normal', 1: 'Classe 1 — Modéré', 2: 'Classe 2 — Critique'}

print("ÉTAPE 10 — Génération des graphiques")


fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor('#F0F2F5')
gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)


# ── G1 : NUAGE DE POINTS ACP 2D ──────────────────────────────
# Graphique clé pour la soutenance :
# montre visuellement que les 3 classes sont séparées dans l'espace ACP
# Les étoiles = vrais centres K-Means (pas des centroïdes calculés après)
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('white')

n_viz   = 30_000
rng_viz = np.random.RandomState(42)
idx_viz = rng_viz.choice(len(df), n_viz, replace=False)

# Ordre d'affichage : 0 → 1 → 2 pour que le rouge (classe 2) soit au-dessus
alphas = {0: 0.18, 1: 0.25, 2: 0.70}
sizes  = {0: 2,    1: 2,    2: 5   }
for cls in [0, 1, 2]:
    mask = df['Classe_Congestion'].values[idx_viz] == cls
    ax1.scatter(
        X_pca_2d[idx_viz][mask, 0],
        X_pca_2d[idx_viz][mask, 1],
        c=COLORS[cls], alpha=alphas[cls], s=sizes[cls],
        label=f"{LABELS[cls]} ({dist.get(cls,0)*100:.1f}%)",
        rasterized=True
    )

# Vrais centres K-Means dans l'espace 2D
for cls in range(3):
    cx, cy = centers_sorted_2d[cls]
    ax1.scatter(cx, cy, c=COLORS[cls], s=250, marker='*',
                edgecolors='black', linewidths=1.2, zorder=10)
    ax1.annotate(
        f'  C{cls} ({["Normal","Modéré","Critique"][cls]})',
        (cx, cy), fontsize=8, fontweight='bold', color=COLORS[cls],
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                  alpha=0.8, edgecolor=COLORS[cls])
    )

ax1.set_xlabel(f'PC1 ({var_pc1:.1f}% variance)', fontsize=10)
ax1.set_ylabel(f'PC2 ({var_pc2:.1f}% variance)', fontsize=10)
ax1.set_title(f'Nuage de points ACP 2D\n(PC1+PC2 = {var_total_2d:.1f}% — vrais centres K-Means)',
              fontweight='bold')
ax1.legend(fontsize=8, markerscale=3, loc='upper right')
ax1.grid(alpha=0.25)


# ── G2 : DISTRIBUTION DU SCORE + INSET ZOOMÉ CLASSE 2 ────────
# Axe Y logarithmique pour rendre la Classe 2 visible
# inset zoomé sur la Classe 2 ajouté
# Sans cet inset, ~4-5% de cas critiques sont invisibles
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('white')
score_vals = df['Congestion_Score'].values

for cls in [0, 1, 2]:
    mask = df['Classe_Congestion'] == cls
    ax2.hist(score_vals[mask], bins=120,
             color=COLORS[cls], alpha=0.80,
             label=LABELS[cls], edgecolor='white', linewidth=0.2)

ax2.axvline(seuil_01, color='#2C3E50', linewidth=2.5, linestyle='--',
            label=f'Frontière 0↔1 = {seuil_01:.3f}')
ax2.axvline(seuil_12, color='#2C3E50', linewidth=2.5, linestyle=':',
            label=f'Frontière 1↔2 = {seuil_12:.3f}')

ax2.set_yscale('log')
ax2.set_title('Distribution du score de congestion\n(échelle log — Classe 2 visible en rouge)',
              fontweight='bold')
ax2.set_xlabel('Score de congestion [0 ; 1]')
ax2.set_ylabel('Nombre de cellules-heures (log)')
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(alpha=0.3, which='both')

# Inset zoomé sur la Classe 2
ax_inset = ax2.inset_axes([0.55, 0.45, 0.42, 0.48])
ax_inset.set_facecolor('#FEF9F9')
mask2 = df['Classe_Congestion'] == 2
ax_inset.hist(score_vals[mask2], bins=60,
              color='#E74C3C', alpha=0.85,
              edgecolor='white', linewidth=0.3)
score_centre_c2 = scores_par_classe[2].mean()
ax_inset.axvline(score_centre_c2, color='#8E1010',
                 linewidth=1.5, linestyle='--',
                 label=f'Centre={score_centre_c2:.3f}')
ax_inset.set_title('Zoom Critique', fontsize=8, fontweight='bold', color='#E74C3C')
ax_inset.set_xlabel('Score', fontsize=7)
ax_inset.set_ylabel('Fréquence', fontsize=7)
ax_inset.tick_params(labelsize=7)
ax_inset.legend(fontsize=6)
n2 = mask2.sum()
ax_inset.text(0.05, 0.88, f'{n2:,} obs.\n({n2/n_total*100:.1f}%)',
              transform=ax_inset.transAxes, fontsize=7,
              bbox=dict(boxstyle='round', facecolor='#FDEDEC', alpha=0.9))


# ── G3 : MÉTRIQUES DE QUALITÉ 
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor('white')

ch_norm  = min(ch / 100_000, 1.0)
m_vals   = [sil, max(0, 1 - db), ch_norm]
m_colors = [COLORS[0] if v >= 0.5 else COLORS[1] for v in m_vals]
m_names  = ['Silhouette\n(idéal → 1)', 'Davies-Bouldin\n(inversé, idéal → 1)', 'Calinski-Harabasz\n(normalisé /100k)']
m_raw    = [sil, db, ch]

bars = ax3.bar(m_names, m_vals, color=m_colors, alpha=0.85,
               width=0.5, edgecolor='white')
for bar, orig in zip(bars, m_raw):
    label = f'{orig:.4f}' if orig < 1000 else f'{orig:,.0f}'
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.02, label,
             ha='center', va='bottom', fontweight='bold', fontsize=10)

ax3.axhline(0.5, color='#E74C3C', linewidth=1.2, linestyle='--',
            alpha=0.6, label='Seuil bon clustering')
ax3.set_ylim(0, 1.2)
ax3.set_title('Métriques de qualité\ndu clustering K-Means 2D', fontweight='bold')
ax3.set_ylabel('Score (normalisé)')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3, axis='y')
interp_sil = 'BON' if sil >= 0.5 else 'ACCEPTABLE'
ax3.text(0.04, 0.88,
         f'Silhouette = {sil:.3f} → {interp_sil}\n'
         f'DB = {db:.3f}\nCH = {ch:,.0f}',
         transform=ax3.transAxes, fontsize=9,
         bbox=dict(boxstyle='round', facecolor='#EAFAF1', alpha=0.85))


# ── G4 : MÉTHODE ELBOW 
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor('white')
ax4.plot(range(1, 8), inertia_vals, 'o-', color='#2980B9',
         linewidth=2.5, markersize=8,
         markerfacecolor='white', markeredgewidth=2)
ax4.axvline(3, color='#E74C3C', linewidth=2, linestyle='--', alpha=0.8)
ax4.annotate('k=3\nchoisi', xy=(3, inertia_vals[2]),
             xytext=(4.5, inertia_vals[2] * 1.1),
             fontsize=10, color='#E74C3C', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5))
ax4.set_title('Méthode Elbow\n(justification du choix k=3 en espace ACP 2D)',
              fontweight='bold')
ax4.set_xlabel('Nombre de clusters k')
ax4.set_ylabel('Inertie (intra-cluster)')
ax4.grid(alpha=0.3)


# ── G5 : COMPOSITION DES CLASSES PAR HEURE 
# Barres empilées = chaque heure divisée en 3 couleurs
# Même si Classe 2 est minoritaire, elle apparaît dans chaque barre
# La courbe noire = score moyen par heure (axe secondaire)
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor('white')

heure_cls = df.groupby(['HOUR', 'Classe_Congestion']).size().unstack(fill_value=0)
heure_cls = heure_cls.div(heure_cls.sum(axis=1), axis=0) * 100

hours       = heure_cls.index.tolist()
bottom_vals = np.zeros(len(hours))
for cls in [0, 1, 2]:
    vals = heure_cls.get(cls, pd.Series(0, index=hours)).values
    ax5.bar(hours, vals, bottom=bottom_vals,
            color=COLORS[cls], alpha=0.85,
            label=LABELS[cls], edgecolor='white',
            linewidth=0.3, width=0.75)
    bottom_vals += vals

ax5b = ax5.twinx()
score_heure = df.groupby('HOUR')['Congestion_Score'].mean()
ax5b.plot(score_heure.index, score_heure.values,
          'k-o', linewidth=2, markersize=4, alpha=0.7, label='Score moyen')
ax5b.set_ylabel('Score moyen de congestion', fontsize=9)
ax5b.set_ylim(0, 0.6)
ax5b.axhline(seuil_01, color='#888', linewidth=1, linestyle='--', alpha=0.5)
ax5b.axhline(seuil_12, color='#888', linewidth=1, linestyle=':', alpha=0.5)

ax5.set_title('Composition des classes par heure\n(K-Means 2D — barres empilées)',
              fontweight='bold')
ax5.set_xlabel('Heure de la journée')
ax5.set_ylabel('Proportion des classes (%)')
ax5.set_xticks(range(0, 24))
ax5.set_ylim(0, 100)
ax5.legend(loc='upper left', fontsize=8)
ax5.grid(alpha=0.2, axis='y')


# ── G6 : VIOLIN PLOT PRB PAR CLASSE 
# Violin plot = montre la distribution complète (pas juste la boîte)
# Pour la Classe 2 (PRB≈100%, variance quasi-nulle) :
#   → jitter plot superposé pour montrer la densité malgré la variance nulle
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor('white')

rng      = np.random.RandomState(42)
sample_n = 5000
data_violin = []
for cls in [0, 1, 2]:
    mask  = df['Classe_Congestion'] == cls
    vals  = df.loc[mask, 'DL_PRB_Usage_Rate'].dropna().values
    n_s   = min(sample_n, len(vals))
    data_violin.append(rng.choice(vals, n_s, replace=False))

parts = ax6.violinplot(data_violin, positions=[0, 1, 2],
                       showmedians=True, showextrema=True, widths=0.6)
for i, (pc, cls) in enumerate(zip(parts['bodies'], [0, 1, 2])):
    pc.set_facecolor(COLORS[cls])
    pc.set_edgecolor('white')
    pc.set_alpha(0.80)
parts['cmedians'].set_color('white')
parts['cmedians'].set_linewidth(2)
parts['cmaxes'].set_color('#2C3E50')
parts['cmins'].set_color('#2C3E50')
parts['cbars'].set_color('#2C3E50')

medians = [np.median(d) for d in data_violin]
for cls, med in zip([0, 1, 2], medians):
    ax6.scatter([cls], [med], color='white', zorder=5,
                s=60, linewidths=2, edgecolors=COLORS[cls])
    ax6.text(cls + 0.32, med, f'  méd={med:.0f}%',
             va='center', fontsize=8, color=COLORS[cls], fontweight='bold')

# Jitter pour Classe 2 (variance quasi-nulle → violin dégénéré)
vals2  = data_violin[2]
jitter = rng.uniform(-0.12, 0.12, size=min(300, len(vals2)))
y2     = rng.choice(vals2, size=min(300, len(vals2)), replace=False)
ax6.scatter(2 + jitter, y2, color='#E74C3C', alpha=0.3, s=5, zorder=3)
ax6.text(2, 50, 'PRB ≈ 100%\n(saturation totale)',
         ha='center', fontsize=8, color='#E74C3C', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#FDEDEC', alpha=0.9))

ax6.set_xticks([0, 1, 2])
ax6.set_xticklabels(['Classe 0\nNormal', 'Classe 1\nModéré', 'Classe 2\nCritique'])
ax6.set_title('DL_PRB_Usage_Rate par classe\n(validation métier — PRB Critique ≈ 100%)',
              fontweight='bold')
ax6.set_ylabel('DL PRB Usage Rate (%)')
ax6.set_ylim(-5, 115)
ax6.grid(alpha=0.3, axis='y')


# ── TITRE GLOBAL
fig.suptitle(
    f'Segmentation K-Means 2D — Espace ACP (PC1+PC2 = {var_total_2d:.1f}%)\n'
    f'Score de Congestion BTS  |  '
    f'Silhouette={sil:.3f}  |  DB={db:.3f}  |  '
    f'Classe 2 Critique = {dist.get(2,0)*100:.1f}%',
    fontsize=13, fontweight='bold'
)

plt.savefig('kmeans_final.png', dpi=150,
            bbox_inches='tight', facecolor='#F0F2F5')
print("kmeans_final.png sauvegardé ✓")
plt.close()

# ÉTAPE 11 — VALIDATION MÉTIER


#   Classe 0 : PRB bas, SSR élevé, disponibilité 100% → normal
#   Classe 1 : PRB moyen, légère dégradation → modéré
#   Classe 2 : PRB ≈ 100%, SSR ≈ 0, disponibilité ≈ 0 → critique

print("\n" + "=" * 60)
print("ÉTAPE 11 — Validation métier")
print("=" * 60)

validation_cols = [
    'DL_PRB_Usage_Rate',
    'LTE_Setup_Success_Rate',
    'Avaibility',
    'DL_Average_Throughput',
    'Avg_User_NB',
    'Congestion_Score'
]

for cls in [0, 1, 2]:
    mask = df['Classe_Congestion'] == cls
    print(f"\nClasse {cls} — {LABELS[cls].split('—')[1].strip()}"
          f"  |  {mask.sum():,} obs. ({mask.mean()*100:.1f}%)")
    print("  " + "-" * 50)
    for col in validation_cols:
        if col in df.columns:
            moy = df.loc[mask, col].mean()
            print(f"  {col:<30} moy = {moy:.3f}")


print("ÉTAPE 12 — Sauvegarde des fichiers")


# Dataset complet avec score et classe
df.to_csv('df_avec_score_kmeans.csv', index=False)

# Coefficients ACP (importance de chaque KPI)
coeff_df.to_csv('coefficients_acp_final.csv', index=False)

# Résumé des clusters (pour le rapport)
summary = pd.DataFrame({
    'Classe'          : [0, 1, 2],
    'Libelle'         : ['Normal', 'Modere', 'Critique'],
    'Nb_Observations' : [int((df['Classe_Congestion']==c).sum()) for c in [0,1,2]],
    'Pct_Observations': [round(dist.get(c,0)*100, 2) for c in [0,1,2]],
    'Score_Moyen'     : [round(scores_par_classe[c].mean(), 4) for c in [0,1,2]],
    'Seuil_Score_Bas' : [0.0, round(seuil_01, 4), round(seuil_12, 4)],
    'Seuil_Score_Haut': [round(seuil_01, 4), round(seuil_12, 4), 1.0],
    'Centre_PC1'      : [round(centers_sorted_2d[c][0], 4) for c in [0,1,2]],
    'Centre_PC2'      : [round(centers_sorted_2d[c][1], 4) for c in [0,1,2]],
})
summary.to_csv('kmeans_summary.csv', index=False)

print("df_avec_score_kmeans.csv     (dataset complet avec Classe_Congestion)")
print("coefficients_acp_final.csv   (importance des KPIs)")
print("kmeans_summary.csv           (résumé des 3 clusters)")
print("kmeans_final.png             (6 graphiques)")


print("RÉSUMÉ FINAL")

print(f"Shape dataset final  : {df.shape}")
print(f"NaN restants         : {df.isnull().sum().sum()}")
print(f"Espace ACP utilisé   : PC1 ({var_pc1:.1f}%) + PC2 ({var_pc2:.1f}%) = {var_total_2d:.1f}%")
print(f"Silhouette           : {sil:.4f}")
print(f"Davies-Bouldin       : {db:.4f}")
print(f"Seuil Classe 0 ↔ 1   : {seuil_01:.4f}")
print(f"Seuil Classe 1 ↔ 2   : {seuil_12:.4f}")
print(f"Classe 0 (Normal)    : {dist.get(0,0)*100:.1f}%")
print(f"Classe 1 (Modéré)    : {dist.get(1,0)*100:.1f}%")
print(f"Classe 2 (Critique)  : {dist.get(2,0)*100:.1f}%")