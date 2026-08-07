import pandas as pd
import numpy as np

def create_features(df):
    # ATTENTION : On utilise 'DATE_' et 'CELLNAME_ID' en MAJUSCULES ici
    df['DATE_DT'] = pd.to_datetime(df['DATE_'], format='mixed', errors='coerce')
    
    # Tri chronologique par cellule
    df = df.sort_values(['CELLNAME_ID', 'DATE_DT']).copy()
    df = df.reset_index(drop=True)

    # ── GROUPE 1 — Features temporelles 
    df['HOUR']       = df['DATE_DT'].dt.hour
    df['IS_WEEKEND'] = df['DATE_DT'].dt.dayofweek.isin([4, 5]).astype(int)

    # ── GROUPE 2 — Features d'efficacité 
    df['PRB_PER_USER'] = (
        df['DL_PRB_USAGE_RATE'] / (df['AVG_USER_NB'] + 1e-6)
    )

    df['SPECTRAL_EFF'] = (
        df['CELL_TRAFFIC_VOLUME_DL'] / (df['DL_PRB_USAGE_RATE'] + 1e-6)
    )

    # ── GROUPE 3 — Features temporelles avancées
    df['IS_PEAK_HOUR'] = df['HOUR'].isin([20, 21, 22, 23]).astype(int)
    
    # ── GROUPE 4 — Features glissantes
    df['ROLLING_TRAFIC_3H'] = df.groupby('CELLNAME_ID')[
        'CELL_TRAFFIC_VOLUME_DL'
    ].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    df['ROLLING_PRB_3H'] = df.groupby('CELLNAME_ID')[
        'DL_PRB_USAGE_RATE'
    ].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    df['HOURLY_TREND'] = df.groupby('CELLNAME_ID')[
        'CELL_TRAFFIC_VOLUME_DL'
    ].diff()
    
    df['ROLLING_MEAN_VOLATILITY'] = df.groupby('CELLNAME_ID')[
        'CELL_TRAFFIC_VOLUME_DL'
    ].transform(lambda x: x.rolling(window=3, min_periods=1).std())

    # ── GROUPE 5 — Features de stabilité et dynamique 
    df['PRB_Z_SCORE'] = df.groupby(['CELLNAME_ID', 'HOUR'])[
        'DL_PRB_USAGE_RATE'
    ].transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))

    df['GRADIENT_PRB'] = df.groupby('CELLNAME_ID')[
        'DL_PRB_USAGE_RATE'
    ].diff()

    # Colonnes numériques seulement pour le fillna
    cols_numeriques = df.select_dtypes(include=[np.number]).columns
    df[cols_numeriques] = df[cols_numeriques].fillna(0)

    # ── NETTOYAGE
    df = df.drop(columns=['DATE_DT'])
    
    # Sécurité : On s'assure que TOUT est en majuscule à la sortie
    df.columns = df.columns.str.upper()
    return df

print("Chargement...")
df = pd.read_csv('dataset_bts_nettoye_final.csv', sep=',', encoding='utf-8')
print(f"Dataset chargé : {len(df):,} lignes, {df.shape[1]} colonnes")

print("\nCalcul des features...")
df_features = create_features(df)

print("\nSauvegarde du fichier...")
df_features.to_csv('dataset_avec_features2.csv', index=False)
print("Dataset sauvegardé : dataset_avec_features2.csv")

print(f"Lignes   : {len(df_features):,}")
print(f"Colonnes : {len(df_features.columns)}")

print(f"\nListe des colonnes :")
for col in df_features.columns:
    print(f"  {col}")

# ── VÉRIFICATION FINALE 
print(f"\nNaN restants : {df_features.isnull().sum().sum()}")
print(f"Doublons     : {df_features.duplicated().sum()}")

print(f"\nAperçu des nouvelles features :")
# Ajout des guillemets ici pour éviter l'erreur NameError
nouvelles = ['HOUR', 'IS_WEEKEND', 'PRB_PER_USER', 'SPECTRAL_EFF',
             'IS_PEAK_HOUR', 'ROLLING_TRAFIC_3H', 'ROLLING_PRB_3H',
             'HOURLY_TREND', 'ROLLING_MEAN_VOLATILITY', 'PRB_Z_SCORE', 'GRADIENT_PRB']

print(df_features[nouvelles].describe().round(3).T[['mean', 'std', 'min', '50%', 'max']].to_string())