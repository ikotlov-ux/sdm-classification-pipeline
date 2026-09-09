# ============================================================================
# 04s_predict_suitability_v2.py
# СКРИПТ 04s: КАРТА ПРИГОДНОСТИ МЕСТООБИТАНИЙ (HABITAT SUITABILITY)
#
# Аналог 04_predict_map.py, адаптированный под SDM.
#
# ИЗМЕНЕНИЯ В v2:
#   - Исправлена ошибка десериализации final_models.pkl:
#       AttributeError: Can't get attribute 'MaxentJarModel' on <module '__main__'>
#     Класс MaxentJarModel был определён в __main__ скрипта 03s, поэтому pickle
#     сохранил ссылку как '__main__.MaxentJarModel'. При запуске 04s как другого
#     __main__ класс отсутствовал. Теперь идентичный класс MaxentJarModel
#     определён на уровне модуля и здесь (verbatim из 03s_tune_sdm.py), что
#     позволяет pickle.load корректно восстановить объект MaxEnt-модели.
#   - Добавлены необходимые импорты (csv, tempfile, subprocess, shutil, pandas,
#     BaseEstimator, ClassifierMixin) для работы класса MaxentJarModel.
#
# Отличия от классификации:
#   - Выход — НЕПРЕРЫВНАЯ карта пригодности suitability [0,1] (Float32),
#     а НЕ категориальная карта классов.
#   - Дополнительно — БИНАРНЫЕ карты присутствия по порогам:
#       * maxSSS (max sensitivity + specificity) — из 03s
#       * 10pct (10-й перцентиль присутствия) — из 03s
#   - При наличии ensemble строит усреднённую карту (по весам AUC из 03s).
#
# Функционал:
#   - Загрузка финальных моделей из 03_TUNE_SDM (final_models.pkl)
#   - Поиск растра-композита (raster_info.pkl)
#   - Блочное (по строкам) предсказание на весь растр
#   - suitability_<model>.tif (Float32, LZW)
#   - binary_<model>_maxSSS.tif / binary_<model>_p10.tif (Int8/Byte, LZW)
#   - Карта-превью (PNG)
# ============================================================================

import os
import re
import gc
import csv
import shutil
import pickle
import tempfile
import subprocess
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from sklearn.base import BaseEstimator, ClassifierMixin

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================================
# ОБЁРТКА MAXENT.JAR (sklearn-совместимая)
#
# ВАЖНО (pickle): этот класс ДОЛЖЕН быть определён на уровне модуля и иметь
# то же имя, что и в 03s_tune_sdm.py, иначе pickle.load(final_models.pkl) не
# сможет восстановить объект MaxEnt-модели (AttributeError: Can't get attribute
# 'MaxentJarModel'). Определение приведено verbatim из 03s_tune_sdm.py.
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


def _get_scores_from_model(model, X):
    """Унифицированно получить suitability [0,1] от любой SDM-модели."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        d = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-d))
    return model.predict(X).astype(float)


def _find_raster(tune_dir, root_dir):
    """Поиск растра-композита: raster_info.pkl → 01_PREPARE_SDM → INPUT."""
    # 1) raster_info из 01_PREPARE_SDM (через data_info.data_dir)
    di_path = os.path.join(tune_dir, "data_info.pkl")
    if os.path.exists(di_path):
        with open(di_path, "rb") as f:
            di = pickle.load(f)
        data_dir = di.get("data_dir")
        if data_dir:
            ri = os.path.join(data_dir, "raster_info.pkl")
            if os.path.exists(ri):
                with open(ri, "rb") as f:
                    info = pickle.load(f)
                if os.path.exists(info["raster_path"]):
                    return info["raster_path"]
    # 2) поиск вручную
    import glob
    cands = []
    for d in [os.path.join(root_dir, "INPUT"), root_dir]:
        cands += glob.glob(os.path.join(d, "*.tif"))
    if not cands:
        raise FileNotFoundError("Растр-композит не найден.")
    print("\nВыберите растр-композит:\n")
    for i, c in enumerate(sorted(set(cands)), 1):
        print(f"  [{i}] {os.path.basename(c)}")
    raw = input("Номер (Enter = 1): ").strip() or "1"
    return sorted(set(cands))[int(raw) - 1]


# ============================================================================
# БЛОЧНОЕ ПРЕДСКАЗАНИЕ
# ============================================================================

def _predict_suitability_blockwise(model, raster_path, predictor_names, out_path,
                                    block_rows=256):
    """Предсказание непрерывной пригодности на весь растр блоками по строкам."""
    with rasterio.open(raster_path) as src:
        n_bands = src.count
        width, height = src.width, src.height
        profile = src.profile.copy()
        band_desc = [
            src.descriptions[i] if src.descriptions[i] else f"band{i+1}"
            for i in range(n_bands)
        ]
        nodata = src.nodata

    name_to_idx = {n: i for i, n in enumerate(band_desc)}
    missing = [p for p in predictor_names if p not in name_to_idx]
    if missing:
        raise ValueError(f"В растре нет предикторов модели: {missing}")
    band_indexes = [name_to_idx[p] + 1 for p in predictor_names]
    n_pred = len(predictor_names)

    out_profile = profile.copy()
    out_profile.update(
        dtype="float32", count=1, compress="lzw",
        tiled=True, blockxsize=256, blockysize=256, nodata=-1.0,
    )

    with rasterio.open(raster_path) as src, \
         rasterio.open(out_path, "w", **out_profile) as dst:
        for row_start in range(0, height, block_rows):
            actual = min(block_rows, height - row_start)
            window = Window(0, row_start, width, actual)
            data = src.read(band_indexes, window=window).astype(np.float64)
            pixels = data.reshape(n_pred, -1).T  # (n_pixels, n_pred)

            # маска валидных
            valid = np.all(np.isfinite(pixels), axis=1)
            if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
                valid &= np.all(pixels != nodata, axis=1)

            out = np.full(pixels.shape[0], -1.0, dtype=np.float32)
            if valid.any():
                out[valid] = _get_scores_from_model(model, pixels[valid]).astype(np.float32)
            dst.write(out.reshape(actual, width).astype(np.float32), 1, window=window)
            print(f"\r  Строки {row_start+actual}/{height}", end="", flush=True)
            del data, pixels, out
            gc.collect()
    print()


def _binarize(suit_path, threshold, out_path):
    """Бинаризация карты пригодности по порогу → Byte (0/1, nodata=255)."""
    with rasterio.open(suit_path) as src:
        prof = src.profile.copy()
        arr = src.read(1)
        nodata = src.nodata
    binary = np.where(arr >= threshold, 1, 0).astype(np.uint8)
    binary = np.where(arr == nodata, 255, binary).astype(np.uint8)
    prof.update(dtype="uint8", nodata=255, compress="lzw")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(binary, 1)


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n============================================================")
    print(" SDM — КАРТА ПРИГОДНОСТИ МЕСТООБИТАНИЙ")
    print("============================================================\n")

    root_dir = os.getcwd()
    TUNE_DIR, suffix = _select_prev_subdir(os.path.join(root_dir, "03_TUNE_SDM"))

    base_out = os.path.join(root_dir, "04_FINAL_SDM")
    os.makedirs(base_out, exist_ok=True)
    run_stamp = datetime.now().strftime("%d%m%y_%H%M")
    OUTPUT_DIR = os.path.join(base_out, f"{run_stamp}{suffix}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- модели и пороги ---
    with open(os.path.join(TUNE_DIR, "final_models.pkl"), "rb") as f:
        mdl_cfg = pickle.load(f)
    with open(os.path.join(TUNE_DIR, "predictor_names.pkl"), "rb") as f:
        predictor_cols = pickle.load(f)["predictor_cols"]
    with open(os.path.join(TUNE_DIR, "thresholds.pkl"), "rb") as f:
        thresholds = pickle.load(f)

    final_models = mdl_cfg["final_models"]
    build_ensemble = mdl_cfg.get("build_ensemble", False)
    ens_weights = mdl_cfg.get("ensemble_weights")

    raster_path = _find_raster(TUNE_DIR, root_dir)
    print(f"\n✓ Растр: {os.path.basename(raster_path)}")
    print(f"✓ Предикторов: {len(predictor_cols)}")
    print(f"✓ Модели: {', '.join(final_models.keys())}\n")

    suit_paths = {}

    # --- предсказание по каждой модели ---
    for key, model in final_models.items():
        print(f"--- Карта пригодности: {key} ---")
        out_tif = os.path.join(OUTPUT_DIR, f"suitability_{key}.tif")
        _predict_suitability_blockwise(model, raster_path, predictor_cols, out_tif)
        suit_paths[key] = out_tif
        print(f"  ✓ {os.path.basename(out_tif)}")

        # бинаризация по порогам
        thr = thresholds.get(key, {})
        for tname, tkey in [("maxSSS", "max_sss"), ("p10", "p10")]:
            tv = thr.get(tkey)
            if tv is not None and np.isfinite(tv):
                bpath = os.path.join(OUTPUT_DIR, f"binary_{key}_{tname}.tif")
                _binarize(out_tif, tv, bpath)
                print(f"  ✓ {os.path.basename(bpath)} (порог={tv:.3f})")
        print()

    # --- ensemble карта ---
    if build_ensemble and ens_weights and len(suit_paths) >= 2:
        print("--- Ensemble (взвешенное среднее) ---")
        wsum = sum(max(w, 0) for w in ens_weights.values()) or 1.0
        ens_arr = None
        ref_prof = None
        nodata_mask = None
        for key, sp in suit_paths.items():
            w = max(ens_weights.get(key, 0), 0) / wsum
            with rasterio.open(sp) as src:
                a = src.read(1)
                if ref_prof is None:
                    ref_prof = src.profile.copy()
                    ens_arr = np.zeros_like(a, dtype=np.float32)
                    nodata_mask = (a == src.nodata)
                else:
                    nodata_mask |= (a == src.nodata)
                a = np.where(a < 0, 0, a)
                ens_arr += a * w
        ens_arr = np.where(nodata_mask, -1.0, ens_arr).astype(np.float32)
        ref_prof.update(dtype="float32", count=1, nodata=-1.0, compress="lzw")
        ens_path = os.path.join(OUTPUT_DIR, "suitability_ensemble.tif")
        with rasterio.open(ens_path, "w", **ref_prof) as dst:
            dst.write(ens_arr, 1)
        suit_paths["ensemble"] = ens_path
        print(f"  ✓ {os.path.basename(ens_path)}\n")

    # --- превью PNG ---
    n = len(suit_paths)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (key, sp) in zip(axes, suit_paths.items()):
        with rasterio.open(sp) as src:
            a = src.read(1)
            a = np.where(a < 0, np.nan, a)
        im = ax.imshow(a, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"Suitability — {key}", fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, shrink=0.7, label="пригодность")
    for ax in axes[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    png = os.path.join(OUTPUT_DIR, "suitability_preview.png")
    plt.savefig(png, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Превью: {os.path.basename(png)}")

    print("\n============================================================")
    print("  КАРТЫ ПРИГОДНОСТИ ГОТОВЫ")
    print("============================================================")
    print(f"Папка: {OUTPUT_DIR}")
    print("  - suitability_<model>.tif (Float32, [0,1])")
    print("  - binary_<model>_maxSSS.tif / _p10.tif (Byte 0/1)")
    if "ensemble" in suit_paths:
        print("  - suitability_ensemble.tif")
    print("\nСледующий шаг: запустите 05s_response_curves.py")
    print("============================================================\n")

    return {"output_dir": OUTPUT_DIR, "suit_paths": suit_paths}


if __name__ == "__main__":
    main()
