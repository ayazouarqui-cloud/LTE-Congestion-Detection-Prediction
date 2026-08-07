# 📡 Détection et Prédiction de Congestion LTE — Réseau BTS Djezzy

> **Pipeline end-to-end de Machine Learning pour la détection en temps réel et la prédiction multi-horizons de la congestion des cellules radio LTE.**

Projet de fin d'études réalisé dans le cadre du **Master Big Data Analytics — USTHB, Faculté d'Électronique et d'Informatique**.

**Réalisé par :**
👩‍💻 **Zouarqui Aya**
👩‍💻 **Khettab Wissam**

---

## 📌 Présentation du projet

La congestion des cellules radio constitue un enjeu majeur pour la qualité de service des réseaux mobiles LTE. Une détection tardive peut entraîner une dégradation de l'expérience utilisateur et rendre les actions correctives essentiellement réactives.

Ce projet propose un **système intelligent de détection, de prédiction et d'explicabilité de la congestion LTE**, appliqué aux données du réseau BTS de **Djezzy — Optimum Télécom Algérie**.

Le système est capable de :

* 🔎 **Détecter** en temps réel l'état de charge d'une cellule LTE
* 🔮 **Prédire** l'état futur à **H+1, H+3 et H+6**
* 📊 **Analyser** les principaux indicateurs de performance réseau
* 💡 **Expliquer** les décisions des modèles grâce à **SHAP**
* 🖥️ **Visualiser** les résultats à travers une application **Streamlit**

### États de congestion

| État            | Description                                                 |
| --------------- | ----------------------------------------------------------- |
| 🟢 **Normal**   | Cellule fonctionnant dans des conditions normales           |
| 🟠 **Modéré**   | Charge élevée nécessitant une surveillance                  |
| 🔴 **Critique** | Niveau de congestion important nécessitant une intervention |

---

# 🎯 Objectifs

Le projet vise à développer un pipeline complet de Machine Learning permettant de passer d'une supervision **réactive** à une supervision **prédictive**.

### Objectifs principaux

1. **Labelliser automatiquement** les cellules selon leur niveau de congestion.
2. Développer un modèle performant pour la **détection de congestion**.
3. Construire des modèles permettant la **prédiction multi-horizons**.
4. Comparer plusieurs algorithmes de Machine Learning et Deep Learning.
5. Optimiser les modèles à l'aide de **Optuna**.
6. Interpréter les prédictions grâce à **SHAP**.
7. Intégrer les modèles dans une **application web interactive**.

---

# 📊 Données

Le jeu de données utilisé contient des mesures horaires provenant du réseau LTE de Djezzy.

| Caractéristique        |                   Valeur |
| ---------------------- | -----------------------: |
| 📈 Enregistrements KPI |            **7 783 150** |
| 📡 Cellules BTS        |               **72 851** |
| 📅 Période             |      **26–31 mars 2026** |
| 🌍 Couverture          | **58 wilayas d'Algérie** |
| 📊 KPI radio           |        **8 indicateurs** |
| 🕐 Granularité         |              **Horaire** |

Les indicateurs comprennent notamment :

* Taux d'utilisation **PRB**
* Disponibilité cellulaire
* Taux de succès d'établissement LTE
* Autres indicateurs de performance radio

> ⚠️ Les données brutes ne sont pas incluses dans ce dépôt pour des raisons de **confidentialité opérateur**.
>
> Le schéma des données et la description des colonnes sont disponibles dans [`data/README.md`](data/README.md).

---

# 🏗️ Architecture du pipeline

```text
                         ┌─────────────────────┐
                         │   Données brutes    │
                         │     KPI LTE         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Nettoyage       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │         EDA         │
                         │ Exploration données │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Feature Engineering │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    ACP + K-Means    │
                         │ Labellisation       │
                         └──────────┬──────────┘
                                    │
                       ┌────────────┴────────────┐
                       │                         │
                       ▼                         ▼
              ┌─────────────────┐      ┌─────────────────────┐
              │    Détection    │      │ Construction cibles │
              │   supervisée    │      │    H+1/H+3/H+6      │
              └────────┬────────┘      └──────────┬──────────┘
                       │                          │
                       │                          ▼
                       │                ┌─────────────────────┐
                       │                │    Prédiction      │
                       │                │    multi-horizons   │
                       │                └──────────┬──────────┘
                       │                           │
                       └─────────────┬─────────────┘
                                     │
                                     ▼
                         ┌─────────────────────┐
                         │       SHAP          │
                         │  Explicabilité IA   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Streamlit       │
                         │ Application Web     │
                         └─────────────────────┘
```

---

# 🔬 Méthodologie

## 1. Labellisation non supervisée — ACP + K-Means

Les données ne disposent pas d'une annotation native indiquant si une cellule est congestionnée.

Une approche non supervisée a donc été utilisée afin d'identifier automatiquement les différents niveaux de congestion.

### ACP

L'Analyse en Composantes Principales permet de réduire la dimensionnalité des données tout en conservant l'essentiel de l'information.

**Résultat :**

* **84,8 %** de variance expliquée
* **2 composantes principales**

### K-Means

Le clustering K-Means permet d'identifier trois groupes correspondant aux niveaux de congestion.

| Classe      |  Proportion |
| ----------- | ----------: |
| 🟢 Normale  | **57,65 %** |
| 🟠 Modérée  | **37,75 %** |
| 🔴 Critique |  **4,59 %** |

### Évaluation du clustering

| Métrique             | Résultat |
| -------------------- | -------: |
| Silhouette Score     | **0,66** |
| Davies-Bouldin Index | **0,36** |

Ces résultats indiquent une séparation relativement nette entre les trois groupes.

---

# 🤖 2. Détection supervisée

Après la labellisation, plusieurs modèles de classification supervisée ont été entraînés afin de détecter automatiquement l'état de congestion d'une cellule.

### Modèles évalués

Six algorithmes ont été comparés :

* XGBoost
* CatBoost
* Random Forest
* HistGradientBoosting
* MLP
* **LightGBM**

L'optimisation des hyperparamètres a été réalisée avec **Optuna**, en utilisant une stratégie TPE.

### Évaluation

Le jeu de test respecte strictement l'ordre chronologique afin d'éviter toute fuite temporelle.

**Nombre d'observations du test set : 1 167 473**

| Modèle               |    F1-macro | F1-classe critique |
| -------------------- | ----------: | -----------------: |
| XGBoost              |           — |                  — |
| CatBoost             |           — |                  — |
| Random Forest        |           — |                  — |
| HistGradientBoosting |           — |                  — |
| MLP                  |           — |                  — |
| 🏆 **LightGBM**      | **99,94 %** |       **99,997 %** |

### 🏆 Meilleur modèle : LightGBM

Le modèle LightGBM obtient :

* **99,94 % de F1-macro**
* **99,997 % de F1 sur la classe critique**
* **+32,76 points** par rapport à une baseline utilisant des seuils fixes

---

# 🔮 3. Prédiction multi-horizons

L'objectif suivant consiste à anticiper l'état futur d'une cellule plutôt que de simplement détecter son état actuel.

Trois horizons de prédiction ont été étudiés :

```text
Temps actuel
     │
     ├──────────► H+1
     │
     ├──────────────────► H+3
     │
     └────────────────────────────► H+6
```

De nouvelles variables temporelles ont été construites afin de capturer l'évolution des KPI dans le temps.

### Modèles comparés

* LSTM avec attention temporelle
* GRU
* LightGBM

### Résultats

| Horizon | F1-macro LightGBM | F1-classe critique |
| ------- | ----------------: | -----------------: |
| **H+1** |       **96,16 %** |      **≥ 98,73 %** |
| **H+3** |       **93,58 %** |      **≥ 98,73 %** |
| **H+6** |       **92,12 %** |      **≥ 98,73 %** |

### Résultat principal

**LightGBM surpasse les architectures récurrentes LSTM-Attention et GRU sur les trois horizons étudiés.**

La performance diminue progressivement lorsque l'horizon augmente, ce qui est cohérent avec la difficulté croissante de prédire l'évolution future du trafic réseau.

---

# 💡 4. Explicabilité avec SHAP

La performance seule ne suffit pas pour une utilisation opérationnelle dans un environnement réseau.

Le projet intègre **SHAP (SHapley Additive exPlanations)** afin d'expliquer les prédictions produites par les modèles.

### Principaux facteurs identifiés

#### 🟢 Normal / 🟠 Modéré

Le **taux d'utilisation des PRB** constitue le principal facteur discriminant.

#### 🔴 Critique

Les facteurs les plus influents comprennent notamment :

* Disponibilité cellulaire
* Taux de succès d'établissement LTE

L'utilisateur peut ainsi comprendre **pourquoi une cellule a été classée comme critique** plutôt que de recevoir uniquement une prédiction.

---

# 🖥️ 5. Application Streamlit

Une application web développée avec **Streamlit** permet d'utiliser les modèles sans nécessiter d'expertise en Data Science.

L'application automatise le pipeline :

```text
KPI bruts
   │
   ▼
Feature Engineering
   │
   ▼
Détection actuelle
   │
   ├──► H+1
   ├──► H+3
   └──► H+6
          │
          ▼
       SHAP
          │
          ▼
   Explication de la décision
```

### Fonctionnalités

* 🏠 Page d'accueil
* 📊 Exploration des données
* 🔎 Détection de congestion
* 🔮 Prédiction H+1 / H+3 / H+6
* 💡 Explicabilité SHAP
* ⚡ Inférence en temps réel
* 📦 Chargement des modèles avec cache

Les **4 modèles LightGBM** sont sérialisés au format `.pkl` avec `joblib` et chargés en cache grâce à :

```python
@st.cache_resource
```

Le Feature Engineering est également exécuté à la volée afin de reproduire le même pipeline que celui utilisé pendant l'entraînement.

---

# 🛠️ Stack technique

### Langage

* 🐍 Python 3.11

### Data Science & Machine Learning

* pandas
* NumPy
* scikit-learn
* LightGBM
* XGBoost
* CatBoost
* Optuna
* SHAP

### Deep Learning

* TensorFlow / PyTorch
* LSTM
* GRU
* Attention temporelle

### Visualisation & Application

* Streamlit
* Plotly
* Joblib

---

# 📁 Structure du repository

```text
LTE-Congestion-Detection-Prediction/
│
├── app/
│   └── app.py
│
├── nettoyage/
│   └── ...
│
├── eda/
│   └── ...
│
├── feature_engineering/
│   └── ...
│
├── acp_kmeans/
│   └── ...
│
├── detection/
│   ├── xgboost/
│   ├── catboost/
│   ├── random_forest/
│   ├── histgradientboosting/
│   ├── mlp/
│   └── lightgbm/
│
├── target/
│   └── ...
│
├── prediction/
│   ├── lstm/
│   ├── gru/
│   └── lightgbm/
│
├── data/
│   └── README.md
│
├── requirements.txt
│
└── README.md
```

---

# 🚀 Installation

## 1. Cloner le repository

```bash
git clone https://github.com/ayazouarqui-cloud/LTE-Congestion-Detection-Prediction.git
```

## 2. Accéder au projet

```bash
cd LTE-Congestion-Detection-Prediction
```

## 3. Installer les dépendances

Il est recommandé d'utiliser un environnement virtuel Python.

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Puis :

```bash
pip install -r requirements.txt
```

---

# ▶️ Lancer l'application

Depuis la racine du projet :

```bash
streamlit run app/app.py
```

L'application Streamlit sera ensuite accessible dans votre navigateur.

---

# 📄 Rapport de projet

Le mémoire complet contient :

* L'état de l'art
* La méthodologie détaillée
* L'analyse exploratoire
* Le Feature Engineering
* La labellisation ACP + K-Means
* L'optimisation des modèles
* Les résultats expérimentaux
* La comparaison des architectures
* L'analyse SHAP
* La conception de l'application

Le rapport principal est disponible dans :

```text
target/rapport_targets_analyse.docx
```

D'autres rapports détaillés sont disponibles dans les différents répertoires du projet.

---

# 🔐 Confidentialité des données

Les données utilisées dans ce projet proviennent d'un environnement opérateur et sont soumises à des contraintes de confidentialité.

Par conséquent :

> **Les données KPI brutes ne sont pas distribuées dans ce repository.**

Seuls les éléments nécessaires à la compréhension, à la reproduction du pipeline et à l'utilisation de l'application sont fournis.

---

# 📈 Principaux résultats

| Étape              | Résultat                         |
| ------------------ | -------------------------------- |
| ACP                | **84,8 %** de variance expliquée |
| K-Means            | **3 classes**                    |
| Silhouette Score   | **0,66**                         |
| Davies-Bouldin     | **0,36**                         |
| Détection LightGBM | **99,94 % F1-macro**             |
| Classe critique    | **99,997 % F1**                  |
| H+1                | **96,16 % F1-macro**             |
| H+3                | **93,58 % F1-macro**             |
| H+6                | **92,12 % F1-macro**             |
| Explicabilité      | **SHAP**                         |
| Interface          | **Streamlit**                    |

---

# 🎓 Contexte académique

**Projet de fin d'études — Master Big Data Analytics**

**Université des Sciences et de la Technologie Houari Boumediene (USTHB)**
Faculté d'Électronique et d'Informatique
Année universitaire **2025–2026**

### Auteurs

**Zouarqui Aya**
**Khettab Wissam**

---

## ⭐ Remerciements

Nous remercions l'ensemble des personnes ayant contribué à la réalisation de ce projet ainsi que l'USTHB pour l'encadrement académique fourni dans le cadre du Master Big Data Analytics.

---

<p align="center">
  <b>Détection intelligente · Prédiction proactive · Explicabilité IA</b>
  <br>
  <br>
  <i>Master Big Data Analytics — USTHB — 2026</i>
</p>

