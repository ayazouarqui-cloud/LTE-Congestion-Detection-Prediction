import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.stdout.reconfigure(encoding="utf-8")

TARGETS = ["target_1h", "target_3h", "target_6h"]
N_CLASSES = 3
N_TRIALS = 30
OPTUNA_SAMPLE = 300_000
SHAP_SAMPLE = 2_000
SEED = 42

# classe_congestion = l'état actuel du réseau. On la garde seulement
# pour target_1h (info dispo en temps réel) et on la retire pour 3h/6h
# sinon le modèle "trichent" en lisant l'état présent au lieu d'anticiper.
KEEP_CLASSE_CONGESTION = {"target_1h": False, "target_3h": False, "target_6h": False}

COLORS_CLS = ["#2196F3", "#FF9800", "#F44336"]
NAMES_CLS = ["Normal (0)", "Modéré (1)", "Congestionné (2)"]
TARGET_COLS = ["#1D9E75", "#7B4FA6", "#D85A30"]

COLS_A_SUPPRIMER = [
    "time_to_peak", "peak_trend_interaction", "traffic_per_user",
    "prb_z_score", "prb_per_user", "throughput_per_user",
    "cell_traffic_volume_dl", "cell_traffic_volume_ul",
    "lte_setup_success_rate",
]


def charger_dataset(path="dataset_avec_targets.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df["date_"] = pd.to_datetime(df["date_"].astype(str).str.strip(), errors="coerce")
    df = df.sort_values(["cellname_id", "date_"]).reset_index(drop=True)

    df.drop(columns=[c for c in COLS_A_SUPPRIMER if c in df.columns], inplace=True)

    if "hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df.drop(columns=["hour"], inplace=True)

    # mémoire récente de charge PRB
    for lag in (1, 2, 3):
        df[f"lag_prb_{lag}h"] = df.groupby("cellname_id")["dl_prb_usage_rate"].shift(lag)

    df.dropna(subset=["lag_prb_1h", "lag_prb_2h", "lag_prb_3h"] + TARGETS, inplace=True)
    return df


def split_temporel(df):
    dates = df["date_"].dropna().sort_values()
    d70, d85 = dates.quantile(0.70), dates.quantile(0.85)
    train = df[df["date_"] <= d70].copy()
    val = df[(df["date_"] > d70) & (df["date_"] <= d85)].copy()
    test = df[df["date_"] > d85].copy()
    return train, val, test


def get_features(target, colonnes):
    exclure = ["date_", "cellname_id", "target_1h", "target_3h", "target_6h", "congestion_score"]
    if not KEEP_CLASSE_CONGESTION[target]:
        exclure.append("classe_congestion")
    return [c for c in colonnes if c not in exclure]


def chercher_hyperparametres(X_tr, y_tr, X_va, y_va, n_trials):
    """Optuna sur un sous-échantillon stratifié — pas la peine de tourner sur 5M lignes."""
    rng = np.random.default_rng(SEED)
    idx_par_classe = [np.where(y_tr == c)[0] for c in (0, 1, 2)]
    n0, n1 = int(OPTUNA_SAMPLE * 0.55), int(OPTUNA_SAMPLE * 0.35)
    n2 = OPTUNA_SAMPLE - n0 - n1
    idx_sample = np.concatenate([
        rng.choice(idx_par_classe[0], min(n0, len(idx_par_classe[0])), replace=False),
        rng.choice(idx_par_classe[1], min(n1, len(idx_par_classe[1])), replace=False),
        rng.choice(idx_par_classe[2], min(n2, len(idx_par_classe[2])), replace=False),
    ])
    Xs, ys = X_tr.iloc[idx_sample], y_tr.iloc[idx_sample]

    val_idx = rng.choice(len(X_va), min(50_000, len(X_va)), replace=False)
    Xv, yv = X_va.iloc[val_idx], y_va.iloc[val_idx]

    def objective(trial):
        params = {
            "objective": "multiclass",
            "num_class": N_CLASSES,
            "metric": "multi_logloss",
            "verbosity": -1,
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255, step=16),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200, step=20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "class_weight": "balanced",
            "random_state": SEED,
            "n_jobs": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            Xs, ys, eval_set=[(Xv, yv)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )
        preds = model.predict(Xv)
        return f1_score(yv, preds, average="macro", zero_division=0)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def entrainer_final(X_tr, y_tr, X_va, y_va, params):
    X_full = pd.concat([X_tr, X_va])
    y_full = pd.concat([y_tr, y_va])
    model = lgb.LGBMClassifier(
        **params, objective="multiclass", num_class=N_CLASSES,
        class_weight="balanced", random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    model.fit(
        X_full, y_full, eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    return model


def evaluer(model, X_te, y_te):
    preds = model.predict(X_te)
    proba = model.predict_proba(X_te)
    y_bin = label_binarize(y_te, classes=[0, 1, 2])
    return {
        "preds": preds,
        "acc": (preds == y_te).mean(),
        "f1_macro": f1_score(y_te, preds, average="macro", zero_division=0),
        "f1_cls": f1_score(y_te, preds, average=None, zero_division=0),
        "auc": roc_auc_score(y_bin, proba, multi_class="ovr", average="macro"),
        "cm": confusion_matrix(y_te, preds),
        "report": classification_report(
            y_te, preds, target_names=["Normal", "Modéré", "Congestionné"], digits=4
        ),
    }


def calculer_shap(model, X_te, y_te, n_echantillons=SHAP_SAMPLE):
    """
    Renvoie toujours une liste de 3 tableaux (N, n_features), un par classe.

    C'est le point qui plantait avant : TreeExplainer ne renvoie pas le
    même format selon la version de shap/lightgbm installée (liste de 3
    arrays vs. un seul array empilé sur le 3e axe). Sans normalisation,
    sv[2] ne pointait pas forcément vers la classe "Congestionné", ce qui
    faussait silencieusement les graphiques d'importance.
    """
    rng = np.random.default_rng(SEED)
    y_arr = y_te.values if hasattr(y_te, "values") else np.asarray(y_te)

    idx = []
    for cls in (0, 1, 2):
        pool = np.where(y_arr == cls)[0]
        idx.extend(rng.choice(pool, min(n_echantillons // 3, len(pool)), replace=False))
    idx = np.array(idx)

    X_shap = X_te.iloc[idx]
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_shap)

    if isinstance(raw, list):
        sv = [np.asarray(s) for s in raw]
    else:
        arr = np.asarray(raw)
        if arr.ndim == 3 and arr.shape[0] == N_CLASSES:
            sv = [arr[i] for i in range(N_CLASSES)]
        elif arr.ndim == 3 and arr.shape[2] == N_CLASSES:
            sv = [arr[:, :, i] for i in range(N_CLASSES)]
        else:
            raise ValueError(f"Shape SHAP inattendue : {arr.shape}")

    return {"X": X_shap, "y": y_arr[idx], "sv": sv}


def tracer_metriques_et_confusion(results, chemin="lgbm_metriques_comparatif.png"):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "LightGBM — Prédiction Congestion BTS\nMétriques et matrices de confusion",
        fontsize=14, fontweight="bold",
    )
    for col, target in enumerate(TARGETS):
        res = results[target]

        ax = axes[0, col]
        bars = ax.bar(["Normal", "Modéré", "Congestionné"], res["f1_cls"],
                       color=COLORS_CLS, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, res["f1_cls"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(f"{target}\nF1 macro = {res['f1_macro']:.4f}",
                     fontweight="bold", color=TARGET_COLS[col])
        ax.set_ylim(0.5, 1.05)
        ax.set_ylabel("F1-score")
        ax.grid(alpha=0.3, axis="y")

        ax = axes[1, col]
        cm_pct = res["cm"] / res["cm"].sum(axis=1, keepdims=True) * 100
        sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues", cbar=False,
                    xticklabels=["Normal", "Modéré", "Congestionné"],
                    yticklabels=["Normal", "Modéré", "Congestionné"], ax=ax)
        ax.set_title(f"Matrice de confusion — {target}", fontweight="bold", color=TARGET_COLS[col])
        ax.set_xlabel("Prédiction")
        ax.set_ylabel("Réalité")

    plt.tight_layout()
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"figure sauvegardée : {chemin}")


def tracer_shap_par_classe(shap_data, features, chemin="lgbm_shap_3targets_3classes.png"):
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle(
        "SHAP — importance des features\nLightGBM, 3 horizons × 3 classes",
        fontsize=14, fontweight="bold",
    )
    for row, target in enumerate(TARGETS):
        feats = features[target]
        sv = shap_data[target]["sv"]
        for col, cls_idx in enumerate((0, 1, 2)):
            ax = axes[row, col]
            imp = np.abs(sv[cls_idx]).mean(axis=0)
            top_idx = np.argsort(imp)[::-1][:12]
            noms = [feats[i] for i in top_idx]
            valeurs = imp[top_idx]

            couleurs = [COLORS_CLS[cls_idx] if i == 0 else "#AAAAAA" for i in range(len(top_idx))]
            ax.barh(range(len(top_idx)), valeurs[::-1], color=couleurs[::-1], alpha=0.85)
            ax.set_yticks(range(len(top_idx)))
            ax.set_yticklabels(noms[::-1], fontsize=7)
            ax.set_title(f"{target} · {NAMES_CLS[cls_idx]}", fontsize=9,
                         fontweight="bold", color=COLORS_CLS[cls_idx])
            ax.set_xlabel("|SHAP| moyen", fontsize=8)
            ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"figure sauvegardée : {chemin}")


def tracer_shap_comparatif_congestionne(shap_data, features, chemin="lgbm_shap_comparatif_cls2.png"):
    """Compare, sur les 3 horizons, quelles features pèsent le plus pour prédire l'état congestionné."""
    all_feats = sorted(set(f for t in TARGETS for f in features[t]))
    imp_matrix = np.zeros((len(TARGETS), len(all_feats)))

    for row, target in enumerate(TARGETS):
        feats = features[target]
        sv = shap_data[target]["sv"]
        imp = np.abs(sv[2]).mean(axis=0)   # classe 2 = Congestionné
        for fi, f in enumerate(feats):
            imp_matrix[row, all_feats.index(f)] = imp[fi]

    top_order = np.argsort(imp_matrix.mean(axis=0))[::-1][:15]
    imp_plot = imp_matrix[:, top_order]
    feat_plot = [all_feats[i] for i in top_order]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(
        "SHAP — importance classe Congestionné\ncomparaison des 3 horizons",
        fontsize=13, fontweight="bold",
    )
    x = np.arange(len(feat_plot))
    w = 0.25
    for row, (target, offset, col) in enumerate(zip(TARGETS, (-w, 0, w), TARGET_COLS)):
        ax.bar(x + offset, imp_plot[row], w, label=target, color=col, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(feat_plot, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("|SHAP| moyen — Congestionné")
    ax.set_title('Plus la barre est haute, plus la feature prédit "Congestionné"', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"figure sauvegardée : {chemin}")


def tracer_degradation_horizon(results, chemin="lgbm_comparaison_horizons.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("LightGBM — performances selon l'horizon (1h / 3h / 6h)",
                 fontsize=13, fontweight="bold")

    x = np.arange(3)
    w = 0.25
    for i, (target, col) in enumerate(zip(TARGETS, TARGET_COLS)):
        ax1.bar(x + (i - 1) * w, results[target]["f1_cls"], w, label=target, color=col, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Normal", "Modéré", "Congestionné"])
    ax1.set_ylim(0.5, 1.05)
    ax1.set_ylabel("F1-score")
    ax1.set_title("F1 par classe")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, axis="y")

    heures = [1, 3, 6]
    f1_macro = [results[t]["f1_macro"] for t in TARGETS]
    f1_congestion = [results[t]["f1_cls"][2] for t in TARGETS]
    ax2.plot(heures, f1_macro, "o-", color="steelblue", linewidth=2.5, markersize=9, label="F1-macro")
    ax2.plot(heures, f1_congestion, "s-", color="#F44336", linewidth=2.5, markersize=9, label="F1-Congestionné")
    ax2.set_xticks(heures)
    ax2.set_xticklabels(["h+1", "h+3", "h+6"])
    ax2.set_xlabel("Horizon de prédiction")
    ax2.set_ylabel("F1-score")
    ax2.set_title("Dégradation selon l'horizon")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(min(f1_macro) - 0.05, 1.01)

    plt.tight_layout()
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"figure sauvegardée : {chemin}")


def resumer(results, shap_data, features):
    print(f"\n{'Métrique':<20}{'1h':>10}{'3h':>10}{'6h':>10}")
    for label, key, cls_i in [
        ("Accuracy", "acc", None), ("F1-macro", "f1_macro", None),
        ("F1 Normal", "f1_cls", 0), ("F1 Modéré", "f1_cls", 1),
        ("F1 Congestionné", "f1_cls", 2), ("ROC-AUC", "auc", None),
    ]:
        vals = [
            f"{results[t][key][cls_i]:.4f}" if cls_i is not None else f"{results[t][key]:.4f}"
            for t in TARGETS
        ]
        print(f"{label:<20}{vals[0]:>10}{vals[1]:>10}{vals[2]:>10}")

    for target in TARGETS:
        feats = features[target]
        imp = np.abs(shap_data[target]["sv"][2]).mean(axis=0)
        top3 = np.argsort(imp)[::-1][:3]
        print(f"\n{target} — top 3 features (classe Congestionné) :")
        for rank, i in enumerate(top3, 1):
            print(f"  {rank}. {feats[i]} ({imp[i]:.5f})")


def main(chercher_params=True, best_params_connus=None):
    """
    chercher_params=False permet de sauter Optuna si on a déjà les
    meilleurs hyperparamètres d'un run précédent — passe-les dans
    best_params_connus = {"target_1h": {...}, "target_3h": {...}, "target_6h": {...}}
    """
    df = charger_dataset()
    train_df, val_df, test_df = split_temporel(df)
    print(f"Train {len(train_df):,} | Val {len(val_df):,} | Test {len(test_df):,}")

    results, models, features, shap_data = {}, {}, {}, {}

    for target in TARGETS:
        print(f"\n--- {target} ---")
        feats = get_features(target, df.columns.tolist())
        features[target] = feats

        X_tr, y_tr = train_df[feats], train_df[target]
        X_va, y_va = val_df[feats], val_df[target]
        X_te, y_te = test_df[feats], test_df[target]

        if chercher_params:
            params = chercher_hyperparametres(X_tr, y_tr, X_va, y_va, N_TRIALS)
        else:
            params = best_params_connus[target]

        model = entrainer_final(X_tr, y_tr, X_va, y_va, params)
        models[target] = model

        res = evaluer(model, X_te, y_te)
        results[target] = res
        print(res["report"])

        shap_data[target] = calculer_shap(model, X_te, y_te)

    tracer_metriques_et_confusion(results)
    tracer_shap_par_classe(shap_data, features)
    tracer_shap_comparatif_congestionne(shap_data, features)
    tracer_degradation_horizon(results)
    resumer(results, shap_data, features)

    return models, results, shap_data


if __name__ == "__main__":
    main(chercher_params=True)