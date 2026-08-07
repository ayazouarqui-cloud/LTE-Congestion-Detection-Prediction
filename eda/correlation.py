# ============================================================
# BLOC 4 — MATRICE DE CORRÉLATION
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Chargement des données...")
df = pd.read_csv("dataset_bts_nettoye_final.csv")
df['DATE_'] = pd.to_datetime(df['DATE_'], format='mixed', errors='coerce')

cols_num = ['LTE_Setup_Success_Rate', 'Cell_Traffic_Volume_DL',
            'Cell_Traffic_Volume_UL', 'DL_Average_Throughput',
            'UL_Average_Throughput', 'DL_PRB_Usage_Rate',
            'Avg_User_NB', 'Avaibility']

# Calcul corrélation
corr = df[cols_num].corr()

# Affichage heatmap
plt.figure(figsize=(11, 8))
sns.heatmap(corr,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            annot_kws={'size': 10})
plt.title('Matrice de corrélation entre les colonnes', fontsize=14)
plt.tight_layout()
plt.savefig('B4_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

# Résumé texte des corrélations fortes
print("\nCorrélations fortes (> 0.5) :")
print("-" * 50)
for col1 in cols_num:
    for col2 in cols_num:
        if col1 < col2:
            val = corr.loc[col1, col2]
            if abs(val) > 0.5:
                direction = "positive" if val > 0 else "négative"
                print(f"  {col1} ↔ {col2}")
                print(f"  Corrélation {direction} : {val:.2f}")
                print()

print("\nCorrélations faibles (< 0.1) :")
print("-" * 50)
for col1 in cols_num:
    for col2 in cols_num:
        if col1 < col2:
            val = corr.loc[col1, col2]
            if abs(val) < 0.1:
                print(f"  {col1} ↔ {col2} : {val:.2f}")