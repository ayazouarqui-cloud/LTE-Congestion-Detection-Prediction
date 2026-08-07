import pandas as pd
import numpy as np
df = pd.read_csv("Datasetbtscongest.csv", sep=",")
print(df.shape)         
print(df.dtypes)        
print(df.isnull().sum()) 
print(df.describe())   
# Supprimer la colonne index inutile
df.drop(columns=["Unnamed: 0"], inplace=True)


print(df.columns.tolist())
print(df.shape)


df['DATE_'] = pd.to_datetime(df['DATE_'], errors='coerce')

print(f"DATE_ type  : {df['DATE_'].dtype}")
print(f"DATE_ NaT   : {df['DATE_'].isnull().sum()}")
print(f"DATE_ aperçu: {df['DATE_'].head(5).tolist()}")


print("  DÉTECTION VALEURS PHYSIQUEMENT IMPOSSIBLES")


# --- LTE_Setup_Success_Rate > 100% ---
ssr_bad = df[df['LTE_Setup_Success_Rate'] > 100]
print(f"\nLTE_Setup_Success_Rate > 100% :")
print(f"  Nombre de lignes : {len(ssr_bad):,}")
print(f"  Valeur max       : {df['LTE_Setup_Success_Rate'].max():.4f}%")
print(f"  Valeur min parmi ces lignes : {ssr_bad['LTE_Setup_Success_Rate'].min():.4f}%")
print(f"  Aperçu des valeurs :")
print(ssr_bad['LTE_Setup_Success_Rate'].value_counts().head(10))

# --- Avaibility > 100% ---
avail_bad = df[df['Avaibility'] > 100]
print(f"\nAvaibility > 100% :")
print(f"  Nombre de lignes : {len(avail_bad):,}")
print(f"  Valeur max       : {df['Avaibility'].max():.4f}%")
if len(avail_bad) > 0:
    print(f"  Aperçu des valeurs :")
    print(avail_bad['Avaibility'].value_counts().head(10))

# --- Résumé ---
print("\n" + "=" * 55)
print(f"  Total lignes à corriger : {len(ssr_bad) + len(avail_bad):,}")
print("=" * 55)
print("--- Type A : Valeurs impossibles ---")

# LTE_Setup_Success_Rate > 100
mask_ssr = df["LTE_Setup_Success_Rate"] > 100
print(f"LTE_Setup_Success_Rate > 100 : {mask_ssr.sum()} lignes")
print(f"  Valeur max avant : {df['LTE_Setup_Success_Rate'].max()}")
df.loc[mask_ssr, "LTE_Setup_Success_Rate"] = 100
print(f"  Valeur max après : {df['LTE_Setup_Success_Rate'].max()} ✅")
print()

# Avaibility > 100
mask_avail = df["Avaibility"] > 100
print(f"Avaibility > 100 : {mask_avail.sum()} lignes")
print(f"  Valeur max avant : {df['Avaibility'].max()}")
df.loc[mask_avail, "Avaibility"] = 100
print(f"  Valeur max après : {df['Avaibility'].max()} ✅")
print()

# Supprimer les doublons exacts
df = df.drop_duplicates()

# Supprimer les doublons de clés (en gardant la première occurrence)
df = df.drop_duplicates(subset=["CELLNAME_ID", "DATE_"], keep='first')

print("Doublons supprimés ! Nouveau nombre de lignes :", len(df))
# Visualiser les valeurs manquantes
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    "Valeurs manquantes": missing,
    "Pourcentage (%)": missing_pct.round(2)
})

print(missing_df[missing_df["Valeurs manquantes"] > 0])
# Vérifier si les NaN coïncident avec trafic = 0
mask_nan = df["LTE_Setup_Success_Rate"].isna()

print("Quand LTE_Setup_Success_Rate est NaN :")
print(df.loc[mask_nan, ["Cell_Traffic_Volume_DL",
                          "Cell_Traffic_Volume_UL",
                          "Avg_User_NB"]].describe())
# Combien de lignes ont NaN MAIS quand même du trafic ?
mask_nan = df["LTE_Setup_Success_Rate"].isna()
mask_trafic = df["Cell_Traffic_Volume_DL"] > 0

cas_suspects = df[mask_nan & mask_trafic]
print(f"Lignes avec NaN mais trafic > 0 : {len(cas_suspects)}")
print(f"Pourcentage sur total NaN : {len(cas_suspects)/mask_nan.sum()*100:.2f}%")
print()
print(cas_suspects[["DATE_", "CELLNAME_ID", "LTE_Setup_Success_Rate",
                     "Cell_Traffic_Volume_DL", "Avg_User_NB"]].head(10))
# Créer un profil pour chaque ligne basé sur les patterns
df["profil"] = ""

# Condition 1 : Trafic nul ou non
df["profil"] += df["Cell_Traffic_Volume_DL"].apply(
    lambda x: "TRAFIC_NUL|" if x == 0 else "TRAFIC_OK|"
)

# Condition 2 : Availability
df["profil"] += df["Avaibility"].apply(
    lambda x: "AVAIL_0|" if x == 0
    else ("AVAIL_100|" if x == 100 else "AVAIL_PARTIEL|")
)

# Condition 3 : SSR manquant ou non
df["profil"] += df["LTE_Setup_Success_Rate"].apply(
    lambda x: "SSR_NAN|" if pd.isna(x) else "SSR_OK|"
)

# Condition 4 : PRB manquant ou non
df["profil"] += df["DL_PRB_Usage_Rate"].apply(
    lambda x: "PRB_NAN|" if pd.isna(x)
    else ("PRB_ELEVE|" if x >= 80 else "PRB_OK|")
)

# Condition 5 : Throughput manquant ou non
df["profil"] += df["DL_Average_Throughput"].apply(
    lambda x: "THP_NAN|" if pd.isna(x) else "THP_OK|"
)

# Afficher tous les types trouvés avec leur count
profils = df["profil"].value_counts()
print(f"Nombre de types distincts trouvés : {len(profils)}")
print()
print(profils.to_string())
print("=== DEBUT DU REMPLISSAGE DES VALEURS MANQUANTES ===\n")


# FAMILLE 2 — Cellules Crashées (AVAIL = 0)
# PRB=100, SSR=0, THP DL=0, THP UL=0

print("Traitement Famille 2 — Cellules Crashées...")

mask_crash = df["Avaibility"] == 0

df.loc[mask_crash, "DL_PRB_Usage_Rate"] = \
    df.loc[mask_crash, "DL_PRB_Usage_Rate"].fillna(100)

df.loc[mask_crash, "LTE_Setup_Success_Rate"] = \
    df.loc[mask_crash, "LTE_Setup_Success_Rate"].fillna(0)

df.loc[mask_crash, "DL_Average_Throughput"] = \
    df.loc[mask_crash, "DL_Average_Throughput"].fillna(0)

df.loc[mask_crash, "UL_Average_Throughput"] = \
    df.loc[mask_crash, "UL_Average_Throughput"].fillna(0)

print(f"   {mask_crash.sum()} lignes traitées\n")



# FAMILLE 3 — Cellules Idle
# AVAIL=100, TRAFIC=0, SSR=NaN
# SSR=0, THP=0, PRB intact

print("Traitement Famille 3 — Cellules Idle...")

mask_idle = (
    (df["Avaibility"] == 100) &
    (df["Cell_Traffic_Volume_DL"] == 0) &
    (df["LTE_Setup_Success_Rate"].isna())
)

df.loc[mask_idle, "LTE_Setup_Success_Rate"] = \
    df.loc[mask_idle, "LTE_Setup_Success_Rate"].fillna(0)

df.loc[mask_idle, "DL_Average_Throughput"] = \
    df.loc[mask_idle, "DL_Average_Throughput"].fillna(0)

df.loc[mask_idle, "UL_Average_Throughput"] = \
    df.loc[mask_idle, "UL_Average_Throughput"].fillna(0)

# PRB → on ne touche pas 

print(f"   {mask_idle.sum()} lignes traitées\n")



# FAMILLES 4 & 5 — Anomalies capteur
# TRAFIC_OK ou SSR_OK → médiane par cellule

print("Traitement Familles 4 & 5 — Anomalies capteur (médiane par cellule)...")
print("   Calcul des médianes par cellule (peut prendre quelques secondes)...")

colonnes_median = [
    "LTE_Setup_Success_Rate",
    "DL_Average_Throughput",
    "UL_Average_Throughput",
    "DL_PRB_Usage_Rate"
]

# Calculer la médiane par cellule pour chaque colonne
for col in colonnes_median:
    # Calculer médiane par CELLNAME_ID
    mediane_par_cellule = df.groupby("CELLNAME_ID")[col].transform("median")

    # Remplir uniquement les NaN restants (F4 et F5)
    df[col] = df[col].fillna(mediane_par_cellule)

    print(f"   {col} → NaN restants : {df[col].isna().sum()}")

print()



# Availability — 178 NaN négligeables

print("Traitement Availability (178 NaN)...")
df["Avaibility"] = df["Avaibility"].fillna(100)
print(f"   Availability → NaN restants : {df['Avaibility'].isna().sum()}\n")



# VÉRIFICATION FINALE

print("=== VÉRIFICATION FINALE ===")
nan_restants = df.isnull().sum()
print(nan_restants)
print(f"\nTotal NaN restants : {nan_restants.sum()}")
print(f"Shape finale : {df.shape}")
# Identifier les cellules dont TOUTES les valeurs sont NaN
cellules_100_nan = df.groupby("CELLNAME_ID")["UL_Average_Throughput"].apply(
    lambda x: x.isna().all()
)

print(f"Cellules avec 100% NaN sur UL_Throughput : {cellules_100_nan.sum()}")
print()

# Voir combien de lignes ça représente
cellules_problematiques = cellules_100_nan[cellules_100_nan].index
lignes_impactees = df[df["CELLNAME_ID"].isin(cellules_problematiques)]
print(f"Lignes impactées : {len(lignes_impactees)}")
print()

# Vérifier leur profil
print("Profils de ces lignes :")
print(lignes_impactees["profil"].value_counts())
# Vérifier les valeurs manquantes par colonne pour ces cellules
cols_a_verifier = ["LTE_Setup_Success_Rate", "DL_Average_Throughput",
                    "UL_Average_Throughput", "DL_PRB_Usage_Rate"]

for col in cols_a_verifier:
    nan_restants = df[col].isna().sum()
    if nan_restants > 0:
        # Calculer la médiane globale sur toutes les lignes non NaN
        mediane_globale = df[col].median()

        print(f"{col}")
        print(f"  NaN restants    : {nan_restants}")
        print(f"  Médiane globale : {mediane_globale:.4f}")

        # Remplir les NaN restants avec la médiane globale
        df[col] = df[col].fillna(mediane_globale)

        print(f"  NaN après fill  : {df[col].isna().sum()} ✅")
        print()

# Vérification finale
print("=== VÉRIFICATION FINALE ===")
print(df.isnull().sum())
print(f"\nTotal NaN restants : {df.isnull().sum().sum()}")
# Supprimer la colonne profil (utilitaire, plus nécessaire)
df.drop(columns=["profil"], inplace=True)

# Vérification
print(df.columns.tolist())
print(df.shape)
df.to_csv('dataset_bts_nettoye_final.csv', index=False)
print(f"Sauvegardé   —  {df.shape[0]:,} lignes, {df.shape[1]} colonnes")


