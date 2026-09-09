# ============================================================================
# 05s_response_curves_v2.py
# СКРИПТ 05s: ВАЖНОСТЬ ПРЕДИКТОРОВ И КРИВЫЕ ОТКЛИКА (RESPONSE CURVES)
#
# Аналог 05_classwise_importance.py, адаптированный под SDM.
#
# ИЗМЕНЕНИЯ В v2:
#   - Исправлена та же ошибка десериализации final_models.pkl, что и в 04s:
#       AttributeError: Can't get attribute 'MaxentJarModel' on <module '__main__'>
#     Класс MaxentJarModel теперь определён на уровне модуля (verbatim из
#     03s_tune_sdm.py), чтобы pickle.load мог восстановить MaxEnt-модель.
#   - Добавлены импорты (csv, tempfile, subprocess, shutil, BaseEstimator,
#     ClassifierMixin), необходимые для работы класса MaxentJarModel.
#
# Отличия от классификации:
#   - Один вид (бинарная задача), без разбивки важности по классам.
#   - Главный артефакт SDM — RESPONSE CURVES: как предсказанная пригодность
#     меняется при изменении одного предиктора (остальные — на медиане).
#   - Permutation importance считается по падению AUC (а не macro-F1).
#
# Функционал:
#   - Загрузка финальной модели (выбор алгоритма) из 03_TUNE_SDM
#   - Permutation importance (по AUC, n повторов), bar-chart
#   - Marginal response curves для каждого предиктора (sigle-variable),
#     остальные предикторы фиксируются на медиане presence-данных
#   - CSV с важностью + PNG графиков
# ============================================================================

import os
import re
import csv
import shutil
import pickle
import tempfile
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score


# ============================================================================
# ОБЁРТКА MAXENT.JAR (sklearn-совместимая)
#
# ВАЖНО (pickle): класс ДОЛЖЕН быть определён на уровне модуля и иметь то же
# имя, что и в 03s_tune_sdm.py, иначе pickle.load(final_models.pkl) не сможет
# восстановить MaxEnt-модель. Определение verbatim из 03s_tune_sdm.py.
# ============================================================================

class MaxentJarModel(BaseEstimator, ClassifierMixin):
    """Обёртка вокруг оригинального maxent.jar (Phillips et al.).

    Обучение и предсказание идут через временные SWD-файлы. Логистический
    выход MaxEnt трактуется как suitability [0,1].
    """

    def __init__(self, maxent_jar=None, feature_names=None, betamultiplier=1.0,
                 features="auto", species="species", java="java"):
        self.maxent_jar = maxent_jar
        self.feature_names = feature_names
        self.betamultiplier = betamultiplier
        self.features = features
        self.species = species
        self.java = java

    def _write_swd(self, path, X, label):
        names = self.feature_names or [f"f{i}" for i in range(X.shape[1])]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["species", "x", "y"] + list(names))
            for i, row in enumerate(X):
                w.writerow([label, i, 0] + [f"{v:.6g}" for v in row])

    def _feature_flags(self):
        if self.features == "auto":
            return []
        flags = []
        allf = {"linear", "quadratic", "product", "threshold", "hinge"}
        chosen = set(s.strip() for s in self.features.split(",")) & allf
        for ft in allf:
            on = ft in chosen
            flags.append(f"{ft}={'true' if on else 'false'}")
        return flags

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        if self.maxent_jar is None or not os.path.exists(self.maxent_jar):
            raise FileNotFoundError(
                f"maxent.jar не найден: {self.maxent_jar}. "
                "Укажите путь через MAXENT_JAR или при запуске."
            )
        self.classes_ = np.array([0, 1])
        self._workdir = tempfile.mkdtemp(prefix="maxent_")
        samples_swd = os.path.join(self._workdir, "samples.csv")
        bg_swd = os.path.join(self._workdir, "background.csv")
        out_dir = os.path.join(self._workdir, "out")
        os.makedirs(out_dir, exist_ok=True)

        self._write_swd(samples_swd, X[y == 1], self.species)
        self._write_swd(bg_swd, X[y == 0], "background")

        cmd = [
            self.java, "-mx2048m", "-jar", self.maxent_jar,
            f"samplesfile={samples_swd}",
            f"environmentallayers={bg_swd}",
            f"outputdirectory={out_dir}",
            "outputformat=logistic",
            f"betamultiplier={self.betamultiplier}",
            "autorun=true", "visible=false", "warnings=false",
            "askoverwrite=false", "writeplotdata=false",
            "responsecurves=false", "jackknife=false", "pictures=false",
            "writebackgroundpredictions=false",
        ] + self._feature_flags()

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"maxent.jar завершился с ошибкой:\n{res.stdout}\n{res.stderr}"
            )
        self._lambdas = os.path.join(out_dir, f"{self.species}.lambdas")
        if not os.path.exists(self._lambdas):
            lam = [f for f in os.listdir(out_dir) if f.endswith(".lambdas")]
            if not lam:
                raise RuntimeError("MaxEnt не создал .lambdas файл.")
            self._lambdas = os.path.join(out_dir, lam[0])
        self._fitted = True
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        proj_dir = tempfile.mkdtemp(prefix="maxent_proj_")
        try:
            proj_swd = os.path.join(proj_dir, "project.csv")
            self._write_swd(proj_swd, X, "project")
            out_csv = os.path.join(proj_dir, "prediction.csv")
            cmd = [
                self.java, "-cp", self.maxent_jar, "density.Project",
                self._lambdas, proj_swd, out_csv, "outputformat=logistic",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0 or not os.path.exists(out_csv):
                raise RuntimeError(
                    f"MaxEnt projection ошибка:\n{res.stdout}\n{res.stderr}"
                )
            pred = pd.read_csv(out_csv)
            suit = pred.iloc[:, -1].values.astype(float)
            suit = np.clip(suit, 0.0, 1.0)
            return np.column_stack([1 - suit, suit])
        finally:
            shutil.rmtree(proj_dir, ignore_errors=True)

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return (p >= 0.5).astype(int)


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ
# ============================================================================

def _select_prev_subdir(base_dir):
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Папка {os.path.basename(base_dir)} не найдена!")
    subdirs = sorted(
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and re.search(r"_v\d{3}$", d)
    )
    if not subdirs:
        raise FileNotFoundError(f"В {os.path.basename(base_dir)} нет результатов.")
    print(f"\nДоступные запуски в {os.path.basename(base_dir)}:\n")
    for i, sd in enumerate(subdirs, 1):
        print(f" [{i}] {sd}")
    raw = input("\nВыберите номер запуска: ").strip() or "1"
    name = subdirs[int(raw) - 1]
    m = re.search(r"(_v\d{3})$", name)
    return os.path.join(base_dir, name), (m.group(1) if m else "_v001")


def _get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return 1.0 / (1.0 + np.exp(-model.decision_function(X)))
    return model.predict(X).astype(float)


def permutation_importance_auc(model, X, y, predictor_names, n_repeats=10, seed=42):
    """Permutation importance по падению AUC при перемешивании предиктора."""
    rng = np.random.default_rng(seed)
    base_auc = roc_auc_score(y, _get_scores(model, X))
    rows = []
    for j, name in enumerate(predictor_names):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            auc_p = roc_auc_score(y, _get_scores(model, Xp))
            drops.append(base_auc - auc_p)
        rows.append({
            "predictor": name,
            "importance_mean": float(np.mean(drops)),
            "importance_sd": float(np.std(drops)),
        })
    df = pd.DataFrame(rows).sort_values("importance_mean", ascending=False)
    return df, base_auc


def response_curve(model, X_pres, predictor_idx, predictor_names, n_points=100):
    """
    Marginal response curve: меняем один предиктор по диапазону,
    остальные фиксируем на медиане presence-данных.
    Возвращает (grid, suitability).
    """
    medians = np.median(X_pres, axis=0)
    col = X_pres[:, predictor_idx]
    lo, hi = np.percentile(col, 1), np.percentile(col, 99)
    if hi <= lo:
        lo, hi = col.min(), col.max() + 1e-6
    grid = np.linspace(lo, hi, n_points)
    Xq = np.tile(medians, (n_points, 1))
    Xq[:, predictor_idx] = grid
    suit = _get_scores(model, Xq)
    return grid, suit


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n============================================================")
    print(" SDM — ВАЖНОСТЬ ПРЕДИКТОРОВ И КРИВЫЕ ОТКЛИКА")
    print("============================================================\n")

    root_dir = os.getcwd()
    TUNE_DIR, suffix = _select_prev_subdir(os.path.join(root_dir, "03_TUNE_SDM"))

    base_out = os.path.join(root_dir, "05_RESPONSE_SDM")
    os.makedirs(base_out, exist_ok=True)
    run_stamp = datetime.now().strftime("%d%m%y_%H%M")
    OUTPUT_DIR = os.path.join(base_out, f"{run_stamp}{suffix}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- модели/данные ---
    with open(os.path.join(TUNE_DIR, "final_models.pkl"), "rb") as f:
        mdl_cfg = pickle.load(f)
    with open(os.path.join(TUNE_DIR, "predictor_names.pkl"), "rb") as f:
        predictor_cols = pickle.load(f)["predictor_cols"]
    with open(os.path.join(TUNE_DIR, "data_info.pkl"), "rb") as f:
        data_info = pickle.load(f)

    final_models = mdl_cfg["final_models"]

    # выбор модели для интерпретации
    keys = list(final_models.keys())
    print("Доступные модели:\n")
    for i, k in enumerate(keys, 1):
        print(f"  [{i}] {k}")
    raw = input("\nВыберите модель для интерпретации (Enter = 1): ").strip() or "1"
    model_key = keys[int(raw) - 1]
    model = final_models[model_key]
    print(f"\n✓ Модель: {model_key}\n")

    # данные
    gdf = gpd.read_file(data_info["samples_path"], layer="samples")
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    X = df[predictor_cols].values.astype(float)
    y = df["pa"].astype(int).values
    X_pres = X[y == 1]

    # ================================================================
    # PERMUTATION IMPORTANCE
    # ================================================================
    print("[1] Permutation importance (по падению AUC)...")
    nrep = int(input("    Число повторов (Enter = 10): ").strip() or "10")
    imp_df, base_auc = permutation_importance_auc(
        model, X, y, predictor_cols, n_repeats=nrep
    )
    imp_df.to_csv(os.path.join(OUTPUT_DIR, f"importance_{model_key}.csv"), index=False)
    print(f"    Базовый AUC: {base_auc:.3f}")
    print(imp_df.head(10).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(predictor_cols))))
    d = imp_df.iloc[::-1]
    ax.barh(d["predictor"], d["importance_mean"], xerr=d["importance_sd"],
            color="steelblue", edgecolor="k")
    ax.set_xlabel("Падение AUC при перемешивании")
    ax.set_title(f"Важность предикторов — {model_key}", fontweight="bold")
    plt.tight_layout()
    imp_png = os.path.join(OUTPUT_DIR, f"importance_{model_key}.png")
    plt.savefig(imp_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    ✓ {os.path.basename(imp_png)}\n")

    # ================================================================
    # RESPONSE CURVES
    # ================================================================
    print("[2] Кривые отклика (response curves)...")
    n_pred = len(predictor_cols)
    ncols = min(4, n_pred)
    nrows = int(np.ceil(n_pred / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()

    # порядок — по важности
    order = [predictor_cols.index(p) for p in imp_df["predictor"]]
    for ax, j in zip(axes, order):
        grid, suit = response_curve(model, X_pres, j, predictor_cols)
        ax.plot(grid, suit, color="darkgreen", lw=2)
        # rug точек присутствия
        pres_vals = X_pres[:, j]
        ax.plot(pres_vals, np.full_like(pres_vals, suit.min()),
                "|", color="firebrick", alpha=0.3, markersize=6)
        ax.set_title(predictor_cols[j], fontsize=9, fontweight="bold")
        ax.set_ylabel("пригодность")
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
    for ax in axes[n_pred:]:
        ax.set_visible(False)
    fig.suptitle(
        f"Кривые отклика — {model_key} (остальные предикторы на медиане presence)",
        fontweight="bold",
    )
    plt.tight_layout()
    rc_png = os.path.join(OUTPUT_DIR, f"response_curves_{model_key}.png")
    plt.savefig(rc_png, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    ✓ {os.path.basename(rc_png)}\n")

    # сохраняем данные кривых
    rc_data = {}
    for j in range(n_pred):
        grid, suit = response_curve(model, X_pres, j, predictor_cols)
        rc_data[predictor_cols[j]] = {"grid": grid.tolist(), "suitability": suit.tolist()}
    with open(os.path.join(OUTPUT_DIR, f"response_curves_{model_key}.pkl"), "wb") as f:
        pickle.dump(rc_data, f)

    print("============================================================")
    print("  ИНТЕРПРЕТАЦИЯ ЗАВЕРШЕНА")
    print("============================================================")
    print(f"Папка: {OUTPUT_DIR}")
    print(f"  - importance_{model_key}.csv / .png")
    print(f"  - response_curves_{model_key}.png / .pkl")
    print("============================================================\n")

    return {"output_dir": OUTPUT_DIR, "model": model_key, "importance": imp_df}


if __name__ == "__main__":
    main()
