# 📡 Détection & Prédiction Congestion BTS — Djezzy
## App Streamlit — PFE Master 2 Big Data USTHB

### Structure des fichiers
```
bts_app/
├── app.py                          ← Application principale
├── model_lightgbm_final.pkl        ← Modèle détection (3 classes)
├── lgbm_target_1h.pkl              ← Modèle prédiction +1h
├── lgbm_target_3h.pkl              ← Modèle prédiction +3h
├── lgbm_target_6h.pkl              ← Modèle prédiction +6h
├── assets/
│   ├── B3_distribution_temporelle.png
│   ├── B4_correlations.png
│   ├── lightgbm_optuna_resultats.png
│   ├── lgbm_metriques_comparatif.png
│   └── lgbm_shap_3targets_3classes.png
├── requirements.txt
└── .streamlit/config.toml
```

### Installation
```bash
pip install -r requirements.txt
```

### Lancement
```bash
streamlit run app.py
```
L'app s'ouvre sur http://localhost:8501

### Pages disponibles
| Page | Contenu |
|------|---------|
| 🏠 Accueil | Page de garde, KPIs globaux, architecture pipeline |
| 📊 Dataset & EDA | Stats nettoyage, distributions, corrélations |
| ⌨️ Acquisition | Saisie manuelle OU upload CSV |
| ⚙️ Pipeline | Animation séquentielle du traitement |
| 🔍 Détection | Résultats LightGBM — classe 0/1/2 + probabilités + jauge |
| 📈 Prédiction | Horizons +1h/+3h/+6h + courbe d'évolution |
| 🧬 SHAP | Explicabilité + feature importances + rapport final |
| 🙏 Fin | Page de remerciements |
