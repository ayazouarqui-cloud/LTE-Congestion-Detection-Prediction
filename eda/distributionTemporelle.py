# ============================================================
# BLOC 3 — DISTRIBUTION TEMPORELLE
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Chargement des données...")
df = pd.read_csv("dataset_bts_nettoye_final.csv")

# Correction conversion DATE_
df['DATE_'] = pd.to_datetime(df['DATE_'], format='mixed', errors='coerce')

print(f"DATE_ NaT : {df['DATE_'].isnull().sum()}")

df['heure']        = df['DATE_'].dt.hour
df['jour_semaine'] = df['DATE_'].dt.dayofweek
df['date_jour']    = df['DATE_'].dt.date

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# --- PRB moyen par heure ---
prb_heure = df.groupby('heure')['DL_PRB_Usage_Rate'].mean()
axes[0,0].plot(prb_heure.index, prb_heure.values,
               color='tomato', marker='o', linewidth=2)
axes[0,0].set_title('PRB moyen par heure')
axes[0,0].set_xlabel('Heure')
axes[0,0].set_ylabel('PRB moyen (%)')
axes[0,0].set_xticks(range(0, 24))
axes[0,0].grid(alpha=0.3)

# --- Utilisateurs moyens par heure ---
user_heure = df.groupby('heure')['Avg_User_NB'].mean()
axes[0,1].plot(user_heure.index, user_heure.values,
               color='steelblue', marker='o', linewidth=2)
axes[0,1].set_title("Utilisateurs moyens par heure")
axes[0,1].set_xlabel('Heure')
axes[0,1].set_ylabel('Nombre utilisateurs')
axes[0,1].set_xticks(range(0, 24))
axes[0,1].grid(alpha=0.3)

# --- Trafic DL moyen par heure ---
trafic_heure = df.groupby('heure')['Cell_Traffic_Volume_DL'].mean()
axes[1,0].bar(trafic_heure.index, trafic_heure.values,
              color='steelblue', alpha=0.8)
axes[1,0].set_title('Trafic DL moyen par heure (GB)')
axes[1,0].set_xlabel('Heure')
axes[1,0].set_ylabel('Trafic DL moyen (GB)')
axes[1,0].set_xticks(range(0, 24))
axes[1,0].grid(alpha=0.3)

# --- Avaibility moyenne par heure ---
avail_heure = df.groupby('heure')['Avaibility'].mean()
axes[1,1].plot(avail_heure.index, avail_heure.values,
               color='green', marker='o', linewidth=2)
axes[1,1].set_title('Disponibilité moyenne par heure (%)')
axes[1,1].set_xlabel('Heure')
axes[1,1].set_ylabel('Avaibility moyenne (%)')
axes[1,1].set_xticks(range(0, 24))
axes[1,1].grid(alpha=0.3)

plt.suptitle('Distribution temporelle — par heure', fontsize=15)
plt.tight_layout()
plt.savefig('B3_distribution_temporelle.png', dpi=150, bbox_inches='tight')
plt.show()
print("Sauvegardé : B3_distribution_temporelle.png ✓")
# ============================================================
# CRASHS PAR HEURE — Avaibility = 0
# ============================================================

crashes_heure = df[df['Avaibility'] == 0].groupby('heure').size()
total_heure   = df.groupby('heure').size()
pct_crashes   = (crashes_heure / total_heure * 100).round(2)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Nombre de crashs par heure
axes[0].bar(crashes_heure.index, crashes_heure.values,
            color='tomato', edgecolor='white')
axes[0].set_title('Nombre de crashs (Avaibility=0) par heure')
axes[0].set_xlabel('Heure')
axes[0].set_ylabel('Nombre de lignes crashées')
axes[0].set_xticks(range(0, 24))
axes[0].grid(alpha=0.3)

# Pourcentage de crashs par heure
axes[1].plot(pct_crashes.index, pct_crashes.values,
             color='red', marker='o', linewidth=2)
axes[1].set_title('% de crashs par heure')
axes[1].set_xlabel('Heure')
axes[1].set_ylabel('% crashs')
axes[1].set_xticks(range(0, 24))
axes[1].grid(alpha=0.3)

# Afficher les valeurs
for x, y in zip(pct_crashes.index, pct_crashes.values):
    axes[1].annotate(f'{y}%', (x, y),
                     textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=8)

plt.suptitle('Distribution des crashs par heure', fontsize=14)
plt.tight_layout()
plt.savefig('B3b_crashs_heure.png', dpi=150, bbox_inches='tight')
plt.show()

print("Crashs par heure :")
print(pd.DataFrame({'Crashs': crashes_heure, '% crashs': pct_crashes}))