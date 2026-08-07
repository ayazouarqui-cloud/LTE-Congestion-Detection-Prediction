import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Chargement du dataset (déjà clusterisé avec KMeans)
df = pd.read_csv("df_avec_score_kmeans.csv")

# La colonne date est en string, faut la convertir en datetime
# On garde toutes les lignes même si certaines dates sont invalides (pas de dropna ici)
df['date_'] = pd.to_datetime(df['date_'].astype(str).str.strip(), errors='coerce')

nb_errors = df['date_'].isna().sum()
if nb_errors > 0:
    print(f"Attention : {nb_errors} dates n'ont pas pu être converties, elles restent dans le df")
else:
    print("Toutes les dates sont converties correctement")

print("Exemple de date :", df['date_'].iloc[0])

# Tri par cellule puis par date -> indispensable avant de faire les shift()
df = df.sort_values(['CELLNAME_ID', 'date_']).reset_index(drop=True)

print(f"\nDataset : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print("Date min :", df['date_'].min())
print("Date max :", df['date_'].max())
print("Nombre de cellules :", df['CELLNAME_ID'].nunique())

# Création des targets à 1h, 3h et 6h dans le futur
# On groupe par cellule pour ne pas mélanger les séries temporelles entre elles
df['target_1h'] = df.groupby('CELLNAME_ID')['Classe_Congestion'].shift(-1)
df['target_3h'] = df.groupby('CELLNAME_ID')['Classe_Congestion'].shift(-3)
df['target_6h'] = df.groupby('CELLNAME_ID')['Classe_Congestion'].shift(-6)

# Les dernières lignes de chaque cellule n'ont pas de futur -> pas le choix, on les enlève
lignes_avant = len(df)
df = df.dropna(subset=['target_1h', 'target_3h', 'target_6h'])
lignes_perdues = lignes_avant - len(df)

print(f"\nLignes supprimées car pas de target dispo : {lignes_perdues}")
print("Lignes restantes :", len(df))

# Les targets étaient en float à cause des NaN, on repasse en int
df['target_1h'] = df['target_1h'].astype(int)
df['target_3h'] = df['target_3h'].astype(int)
df['target_6h'] = df['target_6h'].astype(int)

print("\nTargets créées, vérification de la distribution :")
for target in ['target_1h', 'target_3h', 'target_6h']:
    print(f"\n--- {target} ---")
    print(df[target].value_counts().sort_index())
    print("Proportions :", df[target].value_counts(normalize=True).sort_index().round(4).to_dict())

# Petit exemple visuel sur la première cellule pour vérifier que le shift est cohérent
print("\nExemple sur une cellule :")
cell = df['CELLNAME_ID'].unique()[0]
exemple = df[df['CELLNAME_ID'] == cell].head(8)
print(exemple[['date_', 'CELLNAME_ID', 'Classe_Congestion',
               'target_1h', 'target_3h', 'target_6h']].to_string())

df.to_csv('dataset_avec_targets.csv', index=False)
print(f"\nFichier sauvegardé : dataset_avec_targets.csv ({len(df)} lignes)")

# Vérif rapide qu'il n'y a pas de fuite de données (target trop corrélée à la classe actuelle)
print("\nVérification data leakage :")
for h in [1, 3, 6]:
    col = f'target_{h}h'
    pct = (df[col] == df['Classe_Congestion']).mean()
    print(f"{col} == congestion_class : {pct:.4f} (proche de 0.33 = normal)")