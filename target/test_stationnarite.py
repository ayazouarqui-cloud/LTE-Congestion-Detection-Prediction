import pandas as pd

df = pd.read_csv("dataset_avec_targets.csv")

# Avant d'entraîner un modèle, je veux vérifier si les classes changent souvent
# ou si elles restent stables sur de longues périodes (ça influence direct la difficulté du problème)

print("Taux de changement de classe par cellule")

# Pour chaque cellule, on regarde à quelle fréquence la classe change d'une heure à l'autre
transitions = df.groupby('CELLNAME_ID')['Classe_Congestion'].apply(
    lambda x: (x != x.shift(1)).sum() / len(x)
)

print("Taux moyen de changement de classe par heure :", round(transitions.mean(), 4))
print("Médiane :", round(transitions.median(), 4))
print("Max :", round(transitions.max(), 4))
print("Min :", round(transitions.min(), 4))

# Matrice de transition : à quelle classe on passe le plus souvent après 1h
print("\nMatrice de transition (ligne = T, colonne = T+1h)")
transitions_matrix = pd.crosstab(
    df['Classe_Congestion'],
    df['target_1h'],
    normalize='index'
)
print(transitions_matrix.round(4))

# Petit check sur une cellule au hasard pour voir si ça a du sens visuellement
print("\nExemple détaillé (cellule 324)")
cell_324 = df[df['CELLNAME_ID'] == 324.0].head(20)[['date_', 'Classe_Congestion', 'target_1h', 'target_3h', 'target_6h']]
print(cell_324.to_string())

# Durée moyenne pendant laquelle une cellule reste dans la même classe avant de changer
print("\nDurée moyenne dans chaque classe (segments constants)")

def get_durations(group):
    # repère les changements de classe et regroupe les segments constants
    changes = group != group.shift(1)
    group_id = changes.cumsum()
    durations = group.groupby(group_id).size()
    return durations

durations = df.groupby('CELLNAME_ID')['Classe_Congestion'].apply(get_durations)
print("Durée moyenne d'un segment constant :", round(durations.mean(), 2), "heures")
print("Médiane :", round(durations.median(), 2), "heures")
print("Max :", durations.max(), "heures")