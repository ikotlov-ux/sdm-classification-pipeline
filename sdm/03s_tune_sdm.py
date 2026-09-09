# ============================================================================
# 03s_tune_sdm_v1.py
# СКРИПТ 03s: ОБУЧЕНИЕ И ОЦЕНКА SDM-МОДЕЛЕЙ (PRESENCE-BACKGROUND)
#
# Аналог 03_tune_models.py, адаптированный под моделирование встречаемости.
#
# Алгоритмы:
#   - maxent  : MaxEnt через оригинальный maxent.jar (subprocess), обёрнут в
#               sklearn-совместимый класс MaxentJarModel (fit/predict/
#               predict_proba). Логистический выход MaxEnt = suitability [0,1].
#   - lightgbm: LightGBM binary с балансировкой presence/background
#               (scale_pos_weight по соотношению фон/присутствие).
#   - glm     : Логистическая регрессия (sklearn) в Pipeline с
#               StandardScaler + SimpleImputer.
#   - ensemble: взвешенное среднее prоба по AUC моделей (если выбран в 02s).
#
# Оценка — ТОЛЬКО на held-out тестовых фолдах Spatial Block CV (из 02s):
#   - AUC-ROC, AUC-PR (presence-background)
#   - Continuous Boyce Index (CBI) — ключевая метрика presence-only
#   - TSS по оптимальному порогу (max(sensitivity+specificity-1))
#   - Omission rate при пороге 10-го перцентиля присутствия (10pct)
#   Пороги и Boyce считаются на тесте, чтобы оценка не была оптимистичной.
#
# Финальные модели обучаются на ВСЕХ данных и сохраняются для 04s/05s:
#   final_models.pkl, predictor_names.pkl, thresholds.pkl, metrics.csv
#
# ТРЕБОВАНИЕ для MaxEnt: файл maxent.jar (Phillips et al.) + установленная Java.
#   Путь к maxent.jar задаётся переменной окружения MAXENT_JAR или
#   интерактивно при запуске. Скачать: https://biodiversityinformatics.amnh.org/open_source/maxent/
# ============================================================================

import os
import re
import sys
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


# ============================================================================
# ОБЁРТКА MAXENT.JAR (sklearn-совместимая)
# ============================================================================

class MaxentJarModel(BaseEstimator, ClassifierMixin):
    """
    Обёртка вокруг оригинального maxent.jar (Phillips et al.).

    Обучение и предсказание идут через временные SWD-файлы (samples-with-data),
    которые maxent.jar умеет читать напрямую (флаг samplesfile + environmentallayers
    в формате CSV SWD). Логистический выход MaxEnt трактуется как suitability [0,1].

    Параметры
    ---------
    maxent_jar : str   — путь к maxent.jar
    feature_names : list[str] — имена предикторов (порядок столбцов X)
    betamultiplier : float — регуляризация MaxEnt (по умолчанию 1.0)
    features : str — какие признаки MaxEnt включать (по умолчанию авто)
    species : str — имя вида (метка в SWD)
    java : str — команда java
    """

    def __init__(self, maxent_jar=None, feature_names=None, betamultiplier=1.0,
                 features="auto", species="species", java="java"):
        self.maxent_jar = maxent_jar
        self.feature_names = feature_names
        self.betamultiplier = betamultiplier
        self.features = features
        self.species = species
        self.java = java

    # --- SWD-файлы ---
    def _write_swd(self, path, X, label):
        """Запись SWD CSV: species,x,y,<predictors...>. x/y — фиктивные индексы."""
        names = self.feature_names or [f"f{i}" for i in range(X.shape[1])]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["species", "x", "y"] + list(names))
            for i, row in enumerate(X):
                w.writerow([label, i, 0] + [f"{v:.6g}" for v in row])

    def _feature_flags(self):
        """Преобразование строки features в флаги maxent."""
        if self.features == "auto":
            return []  # MaxEnt сам выбирает по числу образцов
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
        # Сохраняем lambdas-файл (нужен для предсказания на новых данных)
        self._lambdas = os.path.join(out_dir, f"{self.species}.lambdas")
        if not os.path.exists(self._lambdas):
            # имя может быть санитизировано — берём первый .lambdas
            lam = [f for f in os.listdir(out_dir) if f.endswith(".lambdas")]
            if not lam:
                raise RuntimeError("MaxEnt не создал .lambdas файл.")
            self._lambdas = os.path.join(out_dir, lam[0])
        self._fitted = True
        return self

    def predict_proba(self, X):
        """Возвращает (n,2): [P(background), P(presence)=suitability]."""
        X = np.asarray(X, dtype=float)
        proj_dir = tempfile.mkdtemp(prefix="maxent_proj_")
        try:
            proj_swd = os.path.join(proj_dir, "project.csv")
            self._write_swd(proj_swd, X, "project")
            out_csv = os.path.join(proj_dir, "prediction.csv")
            # Режим проекции по .lambdas на SWD
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
            # последний столбец — логистическое предсказание
            suit = pred.iloc[:, -1].values.astype(float)
            suit = np.clip(suit, 0.0, 1.0)
            return np.column_stack([1 - suit, suit])
        finally:
            shutil.rmtree(proj_dir, ignore_errors=True)

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return (p >= 0.5).astype(int)


# ============================================================================
# МЕТРИКИ SDM
# ============================================================================

def continuous_boyce_index(y_true, y_score, n_bins=10, window=0.1):
    """
    Continuous Boyce Index (CBI) — корреляция Спирмена между
    predicted/expected ratio (P/E) и предсказанной пригодностью.
    Считается по подвижному окну. Возвращает значение в [-1, 1];
    >0 — модель согласуется с распределением присутствий.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pres = y_score[y_true == 1]
    allp = y_score
    if len(pres) < 3:
        return np.nan

    lo, hi = np.nanmin(allp), np.nanmax(allp)
    if hi <= lo:
        return np.nan
    width = (hi - lo) * window
    centers = np.linspace(lo + width / 2, hi - width / 2, n_bins)

    pe = []
    valid_centers = []
    for c in centers:
        a, b = c - width / 2, c + width / 2
        n_pred_in = np.sum((allp >= a) & (allp <= b))
        n_pres_in = np.sum((pres >= a) & (pres <= b))
        if n_pred_in == 0:
            continue
        expected = n_pred_in / len(allp)
        predicted = n_pres_in / len(pres)
        if expected > 0:
            pe.append(predicted / expected)
            valid_centers.append(c)
    if len(pe) < 3:
        return np.nan
    from scipy.stats import spearmanr
    rho, _ = spearmanr(valid_centers, pe)
    return float(rho)


def tss_and_threshold(y_true, y_score):
    """TSS по порогу max(sens+spec-1). Возвращает (tss, threshold)."""
    y_true = np.asarray(y_true).astype(int)
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr  # = sens + spec - 1
    k = np.argmax(j)
    return float(j[k]), float(thr[k])


def omission_10pct(y_true, y_score):
    """
    Порог 10-го перцентиля присутствия и omission rate при нём.
    Возвращает (threshold_10pct, omission_rate).
    """
    y_true = np.asarray(y_true).astype(int)
    pres = y_score[y_true == 1]
    if len(pres) == 0:
        return np.nan, np.nan
    thr = float(np.percentile(pres, 10))
    omission = float(np.mean(pres < thr))
    return thr, omission


# ============================================================================
# ПОСТРОЕНИЕ МОДЕЛЕЙ
# ============================================================================

def build_model(key, feature_names, n_pres, n_back, maxent_jar=None):
    """Создаёт необученную модель по ключу."""
    if key == "maxent":
        return MaxentJarModel(maxent_jar=maxent_jar, feature_names=feature_names)
    if key == "lightgbm":
        if not LGBM_AVAILABLE:
            raise ImportError("lightgbm не установлен (pip install lightgbm).")
        spw = max(1.0, n_back / max(n_pres, 1))  # балансировка presence/background
        return lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            random_state=42, n_jobs=-1, verbose=-1,
        )
    if key == "glm":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=42
            )),
        ])
    raise ValueError(f"Неизвестный алгоритм: {key}")


def get_scores(model, X):
    """Унифицированное получение suitability [0,1]."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        d = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-d))
    return model.predict(X).astype(float)


# ============================================================================
# ВЫБОР ПОДПАПКИ
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
    raw = input(f"\nВыберите номер запуска: ").strip() or "1"
    name = subdirs[int(raw) - 1]
    m = re.search(r"(_v\d{3})$", name)
    return os.path.join(base_dir, name), (m.group(1) if m else "_v001")


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n============================================================")
    print(" SDM — ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ")
    print("============================================================\n")

    root_dir = os.getcwd()
    SELECT_DIR, suffix = _select_prev_subdir(os.path.join(root_dir, "02_SELECT_SDM"))

    base_out = os.path.join(root_dir, "03_TUNE_SDM")
    os.makedirs(base_out, exist_ok=True)
    run_stamp = datetime.now().strftime("%d%m%y_%H%M")
    OUTPUT_DIR = os.path.join(base_out, f"{run_stamp}{suffix}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Конфигурация из 02s ---
    with open(os.path.join(SELECT_DIR, "selected_algorithms.pkl"), "rb") as f:
        algo_cfg = pickle.load(f)
    with open(os.path.join(SELECT_DIR, "cv_folds.pkl"), "rb") as f:
        cv_cfg = pickle.load(f)
    with open(os.path.join(SELECT_DIR, "data_info.pkl"), "rb") as f:
        data_info = pickle.load(f)

    selected_keys = algo_cfg["selected_keys"]
    build_ensemble = algo_cfg.get("build_ensemble", False)
    folds_train = cv_cfg["folds_train"]
    folds_test = cv_cfg["folds_test"]
    predictor_cols = data_info["predictor_cols"]
    n_folds = data_info["n_folds"]

    print(f"Алгоритмы: {', '.join(selected_keys)}")
    print(f"CV: {data_info['cv_method']} ({n_folds} фолдов)")
    print(f"Предикторов: {len(predictor_cols)}\n")

    # --- MaxEnt jar путь ---
    maxent_jar = None
    if "maxent" in selected_keys:
        maxent_jar = os.environ.get("MAXENT_JAR")
        if not maxent_jar or not os.path.exists(maxent_jar):
            cand = os.path.join(root_dir, "INPUT", "maxent.jar")
            if os.path.exists(cand):
                maxent_jar = cand
        if not maxent_jar or not os.path.exists(maxent_jar):
            maxent_jar = input("Путь к maxent.jar: ").strip()
        if not os.path.exists(maxent_jar):
            raise FileNotFoundError(f"maxent.jar не найден: {maxent_jar}")
        # проверка Java
        try:
            subprocess.run(["java", "-version"], capture_output=True, check=True)
        except Exception:
            raise RuntimeError("Java не найдена в PATH. Установите JRE для maxent.jar.")
        print(f"✓ maxent.jar: {maxent_jar}\n")

    # --- Данные ---
    gdf = gpd.read_file(data_info["samples_path"], layer="samples")
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    X_all = df[predictor_cols].values.astype(float)
    y_all = df["pa"].astype(int).values
    n_pres = int((y_all == 1).sum())
    n_back = int((y_all == 0).sum())

    # ================================================================
    # CV-ОЦЕНКА (на held-out фолдах)
    # ================================================================
    print("============================================================")
    print(" CV-ОЦЕНКА (held-out Spatial Block CV)")
    print("============================================================\n")

    cv_rows = []
    oof_scores = {k: np.full(len(y_all), np.nan) for k in selected_keys}

    for key in selected_keys:
        print(f"--- {key} ---")
        fold_auc, fold_pr, fold_boyce, fold_tss, fold_om = [], [], [], [], []
        for fi in range(n_folds):
            tr = np.array(folds_train[fi])
            te = np.array(folds_test[fi])
            if len(np.unique(y_all[tr])) < 2 or len(np.unique(y_all[te])) < 2:
                continue
            model = build_model(key, predictor_cols, n_pres, n_back, maxent_jar)
            model.fit(X_all[tr], y_all[tr])
            s_te = get_scores(model, X_all[te])
            oof_scores[key][te] = s_te

            try:
                auc = roc_auc_score(y_all[te], s_te)
            except Exception:
                auc = np.nan
            pr = average_precision_score(y_all[te], s_te)
            cbi = continuous_boyce_index(y_all[te], s_te)
            tss, _ = tss_and_threshold(y_all[te], s_te)
            _, om = omission_10pct(y_all[te], s_te)
            fold_auc.append(auc); fold_pr.append(pr); fold_boyce.append(cbi)
            fold_tss.append(tss); fold_om.append(om)
            print(f"  Fold {fi+1}: AUC={auc:.3f}  PR={pr:.3f}  "
                  f"Boyce={cbi:.3f}  TSS={tss:.3f}  Om10={om:.3f}")

        row = {
            "model": key,
            "AUC_mean": np.nanmean(fold_auc), "AUC_sd": np.nanstd(fold_auc),
            "AUCPR_mean": np.nanmean(fold_pr),
            "Boyce_mean": np.nanmean(fold_boyce),
            "TSS_mean": np.nanmean(fold_tss),
            "Omission10_mean": np.nanmean(fold_om),
        }
        cv_rows.append(row)
        print(f"  ИТОГО {key}: AUC={row['AUC_mean']:.3f}±{row['AUC_sd']:.3f}  "
              f"Boyce={row['Boyce_mean']:.3f}  TSS={row['TSS_mean']:.3f}\n")

    # --- Ensemble (OOF, взвешенный по AUC) ---
    if build_ensemble and len(selected_keys) >= 2:
        weights = {}
        for r in cv_rows:
            w = max(r["AUC_mean"] - 0.5, 0.0)  # вес по AUC выше случайного
            weights[r["model"]] = w
        wsum = sum(weights.values()) or 1.0
        ens = np.zeros(len(y_all))
        cnt = np.zeros(len(y_all))
        for k in selected_keys:
            valid = ~np.isnan(oof_scores[k])
            ens[valid] += oof_scores[k][valid] * weights[k] / wsum
            cnt[valid] += weights[k] / wsum
        ens = np.where(cnt > 0, ens / np.where(cnt == 0, 1, cnt), np.nan)
        valid = ~np.isnan(ens)
        cv_rows.append({
            "model": "ensemble",
            "AUC_mean": roc_auc_score(y_all[valid], ens[valid]),
            "AUC_sd": np.nan,
            "AUCPR_mean": average_precision_score(y_all[valid], ens[valid]),
            "Boyce_mean": continuous_boyce_index(y_all[valid], ens[valid]),
            "TSS_mean": tss_and_threshold(y_all[valid], ens[valid])[0],
            "Omission10_mean": omission_10pct(y_all[valid], ens[valid])[1],
        })
        print(f"  ENSEMBLE: AUC={cv_rows[-1]['AUC_mean']:.3f}  "
              f"Boyce={cv_rows[-1]['Boyce_mean']:.3f}\n")

    metrics_df = pd.DataFrame(cv_rows)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)
    print("Сводка метрик (CV):")
    print(metrics_df.to_string(index=False))
    print()

    # ================================================================
    # ФИНАЛЬНЫЕ МОДЕЛИ (на всех данных) + ПОРОГИ
    # ================================================================
    print("============================================================")
    print(" ОБУЧЕНИЕ ФИНАЛЬНЫХ МОДЕЛЕЙ (все данные)")
    print("============================================================\n")

    final_models = {}
    thresholds = {}
    for key in selected_keys:
        print(f"  Обучение финальной модели: {key}...")
        model = build_model(key, predictor_cols, n_pres, n_back, maxent_jar)
        model.fit(X_all, y_all)
        final_models[key] = model
        # пороги считаем по OOF-скорам (честнее, чем на train)
        s = oof_scores[key]
        valid = ~np.isnan(s)
        if valid.sum() > 0:
            tss, thr_mss = tss_and_threshold(y_all[valid], s[valid])
            thr_10, om10 = omission_10pct(y_all[valid], s[valid])
        else:
            thr_mss, thr_10 = 0.5, np.nan
        thresholds[key] = {"max_sss": thr_mss, "p10": thr_10}
        print(f"    Пороги: maxSSS={thr_mss:.3f}, 10pct={thr_10:.3f}")

    # ================================================================
    # ВИЗУАЛИЗАЦИЯ: сравнение метрик
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_df = metrics_df.set_index("model")
    for ax, col, title in zip(
        axes, ["AUC_mean", "Boyce_mean", "TSS_mean"],
        ["AUC-ROC", "Continuous Boyce Index", "TSS"],
    ):
        plot_df[col].plot.bar(ax=ax, color="steelblue", edgecolor="k")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.axhline(0.5 if col == "AUC_mean" else 0, color="grey", ls="--", lw=0.8)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Сравнение SDM-моделей (held-out Spatial Block CV)", fontweight="bold")
    plt.tight_layout()
    png = os.path.join(OUTPUT_DIR, "model_comparison_sdm.png")
    plt.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  ✓ График: {os.path.basename(png)}")

    # ================================================================
    # СОХРАНЕНИЕ
    # ================================================================
    with open(os.path.join(OUTPUT_DIR, "final_models.pkl"), "wb") as f:
        pickle.dump({
            "final_models": final_models,
            "build_ensemble": build_ensemble,
            "ensemble_weights": {
                r["model"]: max(r["AUC_mean"] - 0.5, 0.0) for r in cv_rows
                if r["model"] in selected_keys
            } if build_ensemble else None,
            "maxent_jar": maxent_jar,
        }, f)
    with open(os.path.join(OUTPUT_DIR, "predictor_names.pkl"), "wb") as f:
        pickle.dump({"predictor_cols": predictor_cols}, f)
    with open(os.path.join(OUTPUT_DIR, "thresholds.pkl"), "wb") as f:
        pickle.dump(thresholds, f)
    with open(os.path.join(OUTPUT_DIR, "data_info.pkl"), "wb") as f:
        pickle.dump(data_info, f)

    print("\n✓ Сохранено в 03_TUNE_SDM:")
    print("  - final_models.pkl (обученные модели + веса ensemble)")
    print("  - predictor_names.pkl")
    print("  - thresholds.pkl (maxSSS, 10pct по каждой модели)")
    print("  - metrics.csv, model_comparison_sdm.png\n")
    print("Следующий шаг: запустите 04s_predict_suitability.py")
    print("============================================================\n")

    return {"output_dir": OUTPUT_DIR, "metrics": metrics_df}


if __name__ == "__main__":
    main()
