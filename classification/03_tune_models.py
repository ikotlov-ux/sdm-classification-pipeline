# =============================================================================
# 03_tune_models_v52.py
# СКРИПТ 03: ТЮНИНГ МОДЕЛЕЙ КЛАССИФИКАЦИИ
#
# Функционал:
# - Загрузка данных из 01_PREPARE_DATA и алгоритмов из 02_SELECT
# - Превентивное удаление ZV/NZV-предикторов
# - Проверка sd предикторов по CV-фолдам
# - Три режима сетки гиперпараметров: optimal / extended / maximal + manual
# - Обучение моделей с preProcess (импутация + z-score)
# - Параллельный режим через joblib (внутри sklearn/catboost)
# - Confusion matrix, метрики (Accuracy, Kappa, TSS, F1, AIC)
# - Графики confusion matrix, сравнения метрик, AIC
# - Выбор лучшей модели, сохранение в final_model.pkl
# - Отчёт run_info.txt
#
# Входные данные: 02_SELECT (алгоритмы, фолды) + 01_PREPARE_DATA (samples.gpkg)
# Выходные данные: 03_TUNE/<ддммгг_ччмм_vXXX>/
# =============================================================================

import os
import re
import sys
import pickle
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    classification_report, f1_score, precision_score, recall_score,
    log_loss,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    SVM_KNN_AVAILABLE = True
except ImportError:
    SVM_KNN_AVAILABLE = False

warnings.filterwarnings("ignore")


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def _select_prev_subdir(base_dir: str) -> tuple[str, str]:
    """Интерактивный выбор подпапки. Возвращает (путь, суффикс _vXXX)."""
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(
            f"Папка {os.path.basename(base_dir)} не найдена! "
            f"Сначала запустите предыдущий скрипт."
        )
    subdirs = sorted(
        [
            os.path.join(base_dir, d)
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]
    )
    if not subdirs:
        raise FileNotFoundError(
            f"В {os.path.basename(base_dir)} нет подпапок с результатами."
        )

    print(f"\nДоступные запуски в {os.path.basename(base_dir)}:\n")
    for i, sd in enumerate(subdirs, 1):
        print(f" [{i}] {os.path.basename(sd)}")

    raw = input(
        f"\nВыберите номер запуска из {os.path.basename(base_dir)}: "
    ).strip()
    if raw == "":
        raw = "1"
    idx = int(raw)
    if idx < 1 or idx > len(subdirs):
        raise ValueError("Некорректный выбор запуска.")

    selected = subdirs[idx - 1]
    name = os.path.basename(selected)
    m = re.search(r"(_v\d{3})$", name)
    suffix = m.group(1) if m else "_v001"
    print(f"Выбрана папка: {name}, суффикс: {suffix}\n")
    return selected, suffix


def _near_zero_var(df: pd.DataFrame, cols: list, freq_cut: float = 19.0,
                   unique_cut: float = 10.0) -> list:
    """
    Аналог caret::nearZeroVar.
    freq_cut — отношение самого частого к второму по частоте (95/5 = 19).
    unique_cut — процент уникальных значений.
    Возвращает список имён ZV/NZV-переменных.
    """
    bad = []
    n = len(df)
    for c in cols:
        vals = df[c].dropna()
        n_unique = vals.nunique()
        pct_unique = 100.0 * n_unique / max(n, 1)

        if n_unique <= 1:
            bad.append(c)
            continue

        freq = vals.value_counts()
        if len(freq) >= 2:
            ratio = freq.iloc[0] / max(freq.iloc[1], 1)
        else:
            ratio = float("inf")

        if ratio >= freq_cut and pct_unique <= unique_cut:
            bad.append(c)
    return bad


def _attach_predictor_names(model, predictor_names: list[str]):
    """
    Прикрепляет к Pipeline/модели точный список предикторов, использованный
    при fit. Это нужно 04_predict_map, чтобы читать из Composite только
    правильные band-ы и в правильном порядке.
    """
    names_arr = np.array([str(x) for x in predictor_names], dtype=object)

    try:
        model.feature_names_in_ = names_arr
    except Exception:
        pass

    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            try:
                step.feature_names_in_ = names_arr
            except Exception:
                pass

    return model


# =============================================================================
# TUNING GRIDS
# =============================================================================

def _get_optimal_grid(alg: str, np_: int) -> dict | None:
    """~9–18 комбинаций."""
    if "Random Forest" in alg:
        mtry_vals = sorted(set([
            max(1, int(np_ ** 0.5)),
            max(1, int(np_ / 3)),
            max(1, int(np_ / 2)),
        ]))
        return {
            "model__n_estimators": [500],
            "model__max_features": mtry_vals,
            "model__criterion": ["gini"],
            "model__min_samples_leaf": [1, 5, 10],
            "model__max_samples": [None],
        }

    if "CatBoost" in alg:
        return {
            "model__depth": [4, 6, 8],
            "model__learning_rate": [0.03, 0.1, 0.3],
            "model__iterations": [200, 500],
            "model__l2_leaf_reg": [3],
        }

    if "Ridge" in alg:
        return {
            "model__C": [
                1 / 10, 1 / 5, 1 / 1, 1 / 0.5, 1 / 0.1, 1 / 0.05,
                1 / 0.01, 1 / 0.001, 1 / 0.0001,
            ],
            "model__penalty": ["l2"],
        }

    if "Lasso" in alg:
        return {
            "model__C": [
                1 / 10, 1 / 5, 1 / 1, 1 / 0.5, 1 / 0.1, 1 / 0.05,
                1 / 0.01, 1 / 0.001, 1 / 0.0001,
            ],
            "model__penalty": ["l1"],
        }

    if "Elastic" in alg:
        return {
            "model__C": [1 / 1, 1 / 0.1, 1 / 0.01, 1 / 0.001],
            "model__l1_ratio": [0.2, 0.4, 0.6, 0.8],
            "model__penalty": ["elasticnet"],
        }

    if "RDA" in alg:
        return {
            "model__shrinkage": [
                0, 0.15, 0.35, 0.5, 0.65, 0.85, 1.0,
            ],
        }

    if "LDA" in alg and "RDA" not in alg:
        return None
    if "QDA" in alg:
        return None

    if "LightGBM" in alg:
        return {
            "model__n_estimators": [200, 500],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__num_leaves": [31, 63],
            "model__reg_alpha": [0.0],
        }

    if "KNN" in alg:
        return {
            "model__n_neighbors": [3, 5, 9, 15, 21],
            "model__weights": ["uniform", "distance"],
        }

    if "SVM" in alg:
        return {
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": ["scale", "auto"],
        }

    return None


def _get_extended_grid(alg: str, np_: int) -> dict | None:
    """~20–50 комбинаций."""
    if "Random Forest" in alg:
        mtry_vals = sorted(set([
            max(1, 2),
            max(1, int(np_ ** 0.5)),
            max(1, int(np_ / 4)),
            max(1, int(np_ / 3)),
            max(1, int(np_ / 2)),
        ]))
        return {
            "model__n_estimators": [500],
            "model__max_features": mtry_vals,
            "model__criterion": ["gini"],
            "model__min_samples_leaf": [1, 3, 5, 10, 15],
            "model__max_samples": [None],
        }

    if "CatBoost" in alg:
        return {
            "model__depth": [4, 6, 8, 10],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__iterations": [200, 500],
            "model__l2_leaf_reg": [1, 3],
        }

    if "Ridge" in alg:
        lambdas = [
            0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005,
            0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5,
            10, 25, 50, 100, 500,
        ]
        return {
            "model__C": [1 / l for l in lambdas],
            "model__penalty": ["l2"],
        }

    if "Lasso" in alg:
        lambdas = [
            0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005,
            0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5,
            10, 25, 50, 100, 500,
        ]
        return {
            "model__C": [1 / l for l in lambdas],
            "model__penalty": ["l1"],
        }

    if "Elastic" in alg:
        lambdas = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
        return {
            "model__C": [1 / l for l in lambdas],
            "model__l1_ratio": [0.1, 0.25, 0.5, 0.75, 0.9],
            "model__penalty": ["elasticnet"],
        }

    if "RDA" in alg:
        return {
            "model__shrinkage": [round(x, 4) for x in np.arange(0, 1.001, 1 / 6)],
        }

    if "LDA" in alg and "RDA" not in alg:
        return None
    if "QDA" in alg:
        return None

    if "LightGBM" in alg:
        return {
            "model__n_estimators": [200, 500, 1000],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__num_leaves": [15, 31, 63, 127],
            "model__reg_alpha": [0.0, 0.1, 1.0],
        }

    if "KNN" in alg:
        return {
            "model__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"],
        }

    if "SVM" in alg:
        return {
            "model__C": [0.01, 0.1, 1, 10, 100, 1000],
            "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        }

    return None


def _get_maximal_grid(alg: str, np_: int) -> dict | None:
    """50–150+ комбинаций."""
    if "Random Forest" in alg:
        mtry_vals = sorted(set([
            max(1, 2),
            max(1, int(np_ ** 0.5)),
            max(1, int(np_ / 5)),
            max(1, int(np_ / 4)),
            max(1, int(np_ / 3)),
            max(1, int(np_ / 2)),
            np_,
        ]))
        return {
            "model__n_estimators": [500],
            "model__max_features": mtry_vals,
            "model__criterion": ["gini"],
            "model__min_samples_leaf": [1, 3, 5, 10, 15, 20],
            "model__max_samples": [None],
        }

    if "CatBoost" in alg:
        return {
            "model__depth": [3, 4, 6, 8, 10],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "model__iterations": [200, 500, 1000],
            "model__l2_leaf_reg": [1, 3, 5, 10],
        }

    if "Ridge" in alg:
        lambdas = [
            1e-05, 5e-05, 1e-04, 5e-04, 0.001, 0.005,
            0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
            1, 2, 5, 10, 25, 50, 100, 250, 500, 1000,
        ]
        return {
            "model__C": [1 / l for l in lambdas],
            "model__penalty": ["l2"],
        }

    if "Lasso" in alg:
        lambdas = [
            1e-05, 5e-05, 1e-04, 5e-04, 0.001, 0.005,
            0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
            1, 2, 5, 10, 25, 50, 100, 250, 500, 1000,
        ]
        return {
            "model__C": [1 / l for l in lambdas],
            "model__penalty": ["l1"],
        }

    if "Elastic" in alg:
        lambdas = [1e-04, 5e-04, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
        return {
            "model__C": [1 / l for l in lambdas],
            "model__l1_ratio": [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)],
            "model__penalty": ["elasticnet"],
        }

    if "RDA" in alg:
        return {
            "model__shrinkage": [round(x, 1) for x in np.arange(0, 1.01, 0.1)],
        }

    if "LDA" in alg and "RDA" not in alg:
        return None
    if "QDA" in alg:
        return None

    if "LightGBM" in alg:
        return {
            "model__n_estimators": [200, 500, 1000, 2000],
            "model__learning_rate": [0.005, 0.01, 0.05, 0.1, 0.2],
            "model__num_leaves": [15, 31, 63, 127, 255],
            "model__reg_alpha": [0.0, 0.01, 0.1, 1.0, 10.0],
            "model__reg_lambda": [0.0, 0.1, 1.0],
        }

    if "KNN" in alg:
        return {
            "model__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31, 51],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan", "minkowski"],
        }

    if "SVM" in alg:
        return {
            "model__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "model__gamma": ["scale", "auto", 0.0001, 0.001, 0.01, 0.1, 1.0],
        }

    return None


def _get_tuning_grid(alg: str, np_: int, mode: str = "optimal") -> dict | None:
    if mode == "extended":
        return _get_extended_grid(alg, np_)
    elif mode == "maximal":
        return _get_maximal_grid(alg, np_)
    else:
        return _get_optimal_grid(alg, np_)


def _count_combinations(grid: dict | None) -> int:
    if grid is None:
        return 1
    return len(list(ParameterGrid(grid)))


# =============================================================================
# AIC
# =============================================================================

def _compute_aic(model, X, y_true, n_classes: int) -> float | None:
    """Вычисление AIC через log-loss. Возвращает None при ошибке."""
    try:
        probs = model.predict_proba(X)
        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps)
        ll = 0.0
        for i in range(len(y_true)):
            true_idx = y_true[i]
            ll -= np.log(probs[i, true_idx])
        k = _estimate_nparams(model, X.shape[1], n_classes)
        return 2 * k + 2 * ll
    except Exception:
        return None


def _estimate_nparams(model, n_predictors: int, n_classes: int) -> int:
    """Оценка числа параметров модели."""
    # Извлекаем финальную модель из Pipeline
    if hasattr(model, "named_steps"):
        est = model.named_steps.get("model", model)
    else:
        est = model

    if isinstance(est, RandomForestClassifier):
        mf = est.max_features
        if isinstance(mf, int):
            return mf * 2
        return max(int(n_predictors ** 0.5), 1) * 2

    if isinstance(est, CatBoostClassifier) if CATBOOST_AVAILABLE else False:
        depth = est.get_param("depth") or 6
        iters = est.get_param("iterations") or 500
        return iters * (2 ** depth - 1) + 10

    if isinstance(est, LogisticRegression):
        if hasattr(est, "coef_"):
            return int(np.sum(est.coef_ != 0))
        return n_predictors * (n_classes - 1)

    if isinstance(est, LinearDiscriminantAnalysis):
        return n_predictors * (n_classes - 1) + n_classes

    if isinstance(est, QuadraticDiscriminantAnalysis):
        return n_classes * (n_predictors * (n_predictors + 1) // 2 + n_predictors + 1)

    return n_predictors * (n_classes - 1)


# =============================================================================
# ПОСТРОЕНИЕ SKLEARN-МОДЕЛИ
# =============================================================================

def _build_estimator(algo_key: str, ncores: int):
    """Возвращает (estimator, supports_grid)."""
    if algo_key == "rf":
        return RandomForestClassifier(
            n_estimators=500, random_state=42, n_jobs=ncores
        ), True

    if algo_key == "catboost":
        if not CATBOOST_AVAILABLE:
            return None, False
        return CatBoostClassifier(
            verbose=0, random_seed=42,
            thread_count=ncores, allow_writing_files=False,
        ), True

    if algo_key == "ridge":
        return LogisticRegression(
            penalty="l2", solver="lbfgs",
            max_iter=5000, random_state=42,
        ), True

    if algo_key == "lasso":
        return LogisticRegression(
            penalty="l1", solver="saga",
            max_iter=5000, random_state=42,
        ), True

    if algo_key == "elasticnet":
        return LogisticRegression(
            penalty="elasticnet", solver="saga",
            max_iter=5000, random_state=42,
        ), True

    if algo_key == "lda":
        return LinearDiscriminantAnalysis(), False

    if algo_key == "qda":
        return QuadraticDiscriminantAnalysis(), False

    if algo_key == "rda":
        # RDA = LDA with shrinkage (аналог klaR::rda)
        return LinearDiscriminantAnalysis(solver="lsqr"), True

    if algo_key == "lgbm":
        if not LGBM_AVAILABLE:
            return None, False
        return LGBMClassifier(
            random_state=42, n_jobs=ncores,
            verbose=-1, force_col_wise=True,
        ), True

    if algo_key == "knn":
        if not SVM_KNN_AVAILABLE:
            return None, False
        return KNeighborsClassifier(n_jobs=ncores), True

    if algo_key == "svm":
        if not SVM_KNN_AVAILABLE:
            return None, False
        return SVC(
            kernel="rbf", probability=True, random_state=42,
        ), True

    return None, False


# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================

def main():
    print("\n============================================================")
    print(" 03 - ТЮНИНГ МОДЕЛЕЙ КЛАССИФИКАЦИИ")
    print("============================================================\n")

    root_dir = os.getcwd()

    # --- Выбор подпапки из 02_SELECT ---
    base_select_dir = os.path.join(root_dir, "02_SELECT")
    input_dir, selected_suffix = _select_prev_subdir(base_select_dir)

    # --- Находим соответствующую подпапку в 01_PREPARE_DATA ---
    prepare_base_dir = os.path.join(root_dir, "01_PREPARE_DATA")
    if not os.path.isdir(prepare_base_dir):
        raise FileNotFoundError(
            "Папка 01_PREPARE_DATA не найдена! Сначала запустите 01_prepare_data.py"
        )
    prepare_subdirs = sorted(
        [
            os.path.join(prepare_base_dir, d)
            for d in os.listdir(prepare_base_dir)
            if os.path.isdir(os.path.join(prepare_base_dir, d))
        ]
    )
    prepare_match = [
        sd for sd in prepare_subdirs
        if os.path.basename(sd).endswith(selected_suffix)
    ]
    if not prepare_match:
        raise FileNotFoundError(
            f"Не найдена подпапка в 01_PREPARE_DATA с суффиксом {selected_suffix}."
        )
    prepare_dir = sorted(prepare_match)[-1]

    # --- Создаём папку 03_TUNE ---
    base_output_dir = os.path.join(root_dir, "03_TUNE")
    os.makedirs(base_output_dir, exist_ok=True)

    run_stamp = datetime.now().strftime("%d%m%y_%H%M")
    run_folder = f"{run_stamp}{selected_suffix}"
    output_dir = os.path.join(base_output_dir, run_folder)
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"INPUT 02_SELECT : {input_dir}")
    print(f"INPUT 01_PREPARE: {prepare_dir}")
    print(f"OUTPUT 03_TUNE  : {output_dir}\n")

    # ================================================================
    # ШАГ 1: ЗАГРУЗКА ДАННЫХ
    # ================================================================
    print("============================================================")
    print(" ШАГ 1: ЗАГРУЗКА ДАННЫХ")
    print("============================================================\n")

    samples_path = os.path.join(prepare_dir, "samples.gpkg")
    if not os.path.exists(samples_path):
        raise FileNotFoundError(
            "Файл samples.gpkg не найден в соответствующей подпапке 01_PREPARE_DATA!"
        )

    samples_gdf = gpd.read_file(samples_path, layer="samples")
    samples_df = pd.DataFrame(samples_gdf.drop(columns="geometry"))

    # --- Определяем поле классов ---
    if "class_name" not in samples_df.columns:
        if "abbr2" in samples_df.columns:
            samples_df = samples_df.rename(columns={"abbr2": "class_name"})
        else:
            candidates = [
                c for c in samples_df.columns
                if c not in ("x", "y", "ID", "polygon_id")
                and not pd.api.types.is_numeric_dtype(samples_df[c])
            ]
            if not candidates:
                raise ValueError("Не найдено поле с названиями классов!")
            samples_df = samples_df.rename(columns={candidates[0]: "class_name"})

    samples_df["class_name"] = samples_df["class_name"].astype(str)
    service_cols = {"class_name", "x", "y", "ID", "polygon_id"}
    predictor_cols = [
        c for c in samples_df.columns
        if c not in service_cols and pd.api.types.is_numeric_dtype(samples_df[c])
    ]
    nbands = len(predictor_cols)

    # --- Превентивное удаление ZV/NZV-предикторов ---
    zv_names = _near_zero_var(samples_df, predictor_cols)
    if zv_names:
        print(
            f"  ! Удалены ZV/NZV-предикторы ({len(zv_names)}): "
            f"{', '.join(zv_names)}"
        )
        predictor_cols = [c for c in predictor_cols if c not in zv_names]
        nbands = len(predictor_cols)

    # --- Диагностика NA ---
    na_counts = samples_df[predictor_cols].isna().sum()
    na_cols = na_counts[na_counts > 0]
    if len(na_cols) > 0:
        print(
            f"  ! NA обнаружены в {len(na_cols)} переменных: "
            f"{', '.join(na_cols.index)}"
        )

    # Кодировка классов
    le = LabelEncoder()
    le.fit(sorted(samples_df["class_name"].unique()))
    class_levels = list(le.classes_)

    print(f"  ✓ samples.gpkg: {len(samples_df)} образцов, {nbands} предикторов")
    print(f"  ✓ Классов: {len(class_levels)} ({', '.join(class_levels)})")

    # --- Загрузка CV-фолдов ---
    cv_folds_path = os.path.join(input_dir, "cv_folds.pkl")
    if not os.path.exists(cv_folds_path):
        raise FileNotFoundError("Файл cv_folds.pkl не найден в 02_SELECT!")
    with open(cv_folds_path, "rb") as f:
        cv_data = pickle.load(f)
    folds_train = cv_data["folds_train"]
    folds_test = cv_data["folds_test"]
    n_folds = len(folds_train)
    print(f"  ✓ cv_folds.pkl: {n_folds} фолдов")

    # --- Загрузка data_info ---
    data_info_path = os.path.join(input_dir, "data_info.pkl")
    cv_method_label = "Stratified KFold"
    if os.path.exists(data_info_path):
        with open(data_info_path, "rb") as f:
            data_info = pickle.load(f)
        cv_method = data_info.get("cv_method", "stratified_kfold")
        if cv_method == "spatial_block_cv":
            cv_method_label = "Spatial Block CV + стратификация"
    print(f"  ✓ CV метод: {cv_method_label}")

    # --- Проверка sd предикторов по CV-фолдам ---
    fold_zero_sd = set()
    for fi in range(n_folds):
        fold_data = samples_df.iloc[folds_train[fi]][predictor_cols]
        sds = fold_data.std()
        bad = sds[(sds.isna()) | (sds < 1e-12)].index.tolist()
        fold_zero_sd.update(bad)
    if fold_zero_sd:
        print(
            f"  ! Удалены предикторы с sd=0 в CV-фолдах ({len(fold_zero_sd)}): "
            f"{', '.join(fold_zero_sd)}"
        )
        predictor_cols = [c for c in predictor_cols if c not in fold_zero_sd]
        nbands = len(predictor_cols)
        print(f"  ✓ Осталось предикторов: {nbands}")

    print()

    # ================================================================
    # ШАГ 2: ЗАГРУЗКА ВЫБРАННЫХ АЛГОРИТМОВ ИЗ 02_SELECT
    # ================================================================
    print("============================================================")
    print(" ШАГ 2: ЗАГРУЗКА ВЫБРАННЫХ АЛГОРИТМОВ")
    print("============================================================\n")

    algo_path = os.path.join(input_dir, "selected_algorithms.pkl")
    if not os.path.exists(algo_path):
        raise FileNotFoundError(
            "Файл selected_algorithms.pkl не найден в 02_SELECT!\n"
            "Сначала запустите 02_classification.py"
        )
    with open(algo_path, "rb") as f:
        algo_data = pickle.load(f)

    selected_algorithms = algo_data["selected_algorithms"]
    selected_keys = algo_data["selected_keys"]
    all_selected_names = algo_data["all_selected_names"]

    # Маппинг ключ → отображаемое имя
    key_to_name = dict(zip(selected_keys, all_selected_names))

    print(f"  Загружено алгоритмов: {len(all_selected_names)} (из 02_SELECT)\n")
    for gname, algs in selected_algorithms.items():
        print(f"  [{gname}]")
        for a in algs:
            status = "✓"
            if "catboost" in a.lower() and not CATBOOST_AVAILABLE:
                status = "✗ недоступен"
            if "lightgbm" in a.lower() and not LGBM_AVAILABLE:
                status = "✗ недоступен"
            if "svm rbf" in a.lower() and not SVM_KNN_AVAILABLE:
                status = "✗ недоступен"
            if "knn" in a.lower() and not SVM_KNN_AVAILABLE:
                status = "✗ недоступен"
            print(f"    {status} {a}")

    # Фильтруем недоступные
    available_keys = []
    available_names = []
    for key, name in zip(selected_keys, all_selected_names):
        if key == "catboost" and not CATBOOST_AVAILABLE:
            print(f"\n  ! Пропущен {name} (catboost не установлен)")
            continue
        if key == "lgbm" and not LGBM_AVAILABLE:
            print(f"\n  ! Пропущен {name} (lightgbm не установлен)")
            continue
        if key in ("svm", "knn") and not SVM_KNN_AVAILABLE:
            print(f"\n  ! Пропущен {name} (sklearn недоступен)")
            continue
        available_keys.append(key)
        available_names.append(name)

    if not available_keys:
        raise RuntimeError("Ни один из выбранных алгоритмов недоступен!")

    print(f"\n  ✓ К обучению: {len(available_keys)} алгоритмов\n")

    # ================================================================
    # ШАГ 3: НАСТРОЙКА ПАРАЛЛЕЛИЗАЦИИ
    # ================================================================
    print("============================================================")
    print(" ШАГ 3: НАСТРОЙКА ПАРАЛЛЕЛИЗАЦИИ")
    print("============================================================\n")

    total_cores = os.cpu_count() or 1
    recommended = max(1, total_cores - 1)

    print(f"Обнаружено ядер (потоков): {total_cores}\n")
    print(f"  [1] {recommended} ядер - РЕКОМЕНДУЕМО")
    if total_cores >= 4:
        print(f"  [2] {max(1, total_cores // 2)} ядер - половина")
    print(f"  [3] 1 ядро - последовательный режим")
    print(f"  [4] Ввести своё число (1-{total_cores})\n")

    cores_choice = input(f"Выбор (Enter = {recommended}): ").strip()

    if cores_choice in ("", "1"):
        ncores = recommended
    elif cores_choice == "2" and total_cores >= 4:
        ncores = max(1, total_cores // 2)
    elif cores_choice == "3":
        ncores = 1
    else:
        try:
            ncores = int(cores_choice)
            ncores = max(1, min(ncores, total_cores))
        except ValueError:
            ncores = recommended

    print(f"\n✓ Будет использовано ядер: {ncores} из {total_cores}\n")

    # ================================================================
    # ШАГ 4: НАСТРОЙКА ГИПЕРПАРАМЕТРОВ
    # ================================================================
    print("============================================================")
    print(" ШАГ 4: НАСТРОЙКА ГИПЕРПАРАМЕТРОВ")
    print("============================================================\n")

    print("Для каждого алгоритма выберите режим сетки.")

    model_configs = {}
    run_log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ncores": ncores,
        "n_samples": len(samples_df),
        "n_predictors": nbands,
        "n_classes": len(class_levels),
        "classes": ", ".join(class_levels),
        "cv_method": f"{cv_method_label} — {n_folds}-fold CV",
        "preProcess": "SimpleImputer(median) + StandardScaler (z-score)",
        "algorithms": {},
    }

    for algo_key, algo_name in zip(available_keys, available_names):
        g_opt = _get_optimal_grid(algo_name, nbands)
        g_ext = _get_extended_grid(algo_name, nbands)
        g_max = _get_maximal_grid(algo_name, nbands)

        n1 = _count_combinations(g_opt)
        n2 = _count_combinations(g_ext)
        n3 = _count_combinations(g_max)

        print(f"\n--- {algo_name} ---")
        print(f" [1] ОПТИМАЛЬНЫЙ  - {n1} комбинаций")
        print(f" [2] РАСШИРЕННЫЙ  - {n2} комбинаций (x{n2 / max(n1, 1):.1f})")
        print(f" [3] МАКСИМАЛЬНЫЙ - {n3} комбинаций (x{n3 / max(n1, 1):.1f})")
        print(f" [4] Настроить вручную")
        ch = input(" Выбор (Enter = 1): ").strip()
        if ch == "":
            ch = "1"

        if ch == "4":
            # Ручная настройка
            print("\n Укажите значения через запятую. Enter = по умолчанию.\n")
            ref_grid = g_opt or g_ext or g_max
            if ref_grid is not None:
                manual_grid = {}
                for pname, pvals in ref_grid.items():
                    default_str = ", ".join(str(v) for v in sorted(set(pvals)))
                    ui = input(f" {pname} [{default_str}]: ").strip()
                    if ui == "":
                        manual_grid[pname] = pvals
                    else:
                        parsed = []
                        for v in ui.split(","):
                            v = v.strip()
                            try:
                                parsed.append(float(v))
                                if parsed[-1] == int(parsed[-1]):
                                    parsed[-1] = int(parsed[-1])
                            except ValueError:
                                parsed.append(v)
                        manual_grid[pname] = parsed
                grid = manual_grid
                nc = _count_combinations(grid)
                print(f"\n ✓ Ручная сетка: {nc} комбинаций")
            else:
                grid = None
                print(" (без тюнинга)")
            mode_this = "manual"

        elif ch == "2":
            grid = _get_tuning_grid(algo_name, nbands, "extended")
            mode_this = "extended"
        elif ch == "3":
            grid = _get_tuning_grid(algo_name, nbands, "maximal")
            mode_this = "maximal"
        else:
            grid = _get_tuning_grid(algo_name, nbands, "optimal")
            mode_this = "optimal"

        model_configs[algo_key] = {
            "name": algo_name,
            "grid": grid,
            "mode": mode_this,
        }

        nc = _count_combinations(grid)
        params_str = (
            ", ".join(grid.keys()) if grid else "нет (без тюнинга)"
        )
        grid_detail = ""
        if grid:
            parts = []
            for p, vals in grid.items():
                parts.append(
                    f" {p}: {', '.join(str(v) for v in sorted(set(vals), key=lambda x: (0, x) if isinstance(x, str) else (1, x)))}"
                )
            grid_detail = "\n".join(parts)

        run_log["algorithms"][algo_name] = {
            "mode": mode_this,
            "n_combinations": nc,
            "parameters": params_str,
            "grid_values": grid_detail,
        }

    # Сохраняем конфигурацию
    with open(os.path.join(output_dir, "model_configs.pkl"), "wb") as f:
        pickle.dump(model_configs, f)

    total_combos = sum(
        _count_combinations(cfg["grid"]) for cfg in model_configs.values()
    )
    print(f"\n✓ Конфигурации сохранены. Всего комбинаций: {total_combos}\n")

    # ================================================================
    # ШАГ 5: ОБУЧЕНИЕ МОДЕЛЕЙ
    # ================================================================
    print("============================================================")
    print(" ШАГ 5: ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("============================================================\n")

    print("  Предобработка: SimpleImputer(median) + StandardScaler (z-score)")
    print("  (импутация NA медианой + z-score стандартизация внутри каждого фолда)\n")

    # Подготовка данных
    # Важно: оставляем X_all как DataFrame, а не numpy-массив.
    # Тогда sklearn Pipeline сохраняет feature_names_in_, и 04_predict_map
    # сможет автоматически понять, какие band-ы Composite/Composite_filtered
    # нужны конкретной модели.
    predictor_cols = [str(c) for c in predictor_cols]
    X_all = samples_df[predictor_cols].astype(np.float64)
    y_all_str = samples_df["class_name"].values
    y_all = le.transform(y_all_str)

    # Формируем cv splits для sklearn
    cv_splits = [
        (np.array(folds_train[fi]), np.array(folds_test[fi]))
        for fi in range(n_folds)
    ]

    all_models = {}
    results_list = []
    confusion_matrices = {}
    class_metrics_list = {}
    aic_values = {}

    total_models = len(model_configs)
    current_model = 0

    for algo_key, cfg in model_configs.items():
        current_model += 1
        algo_name = cfg["name"]
        tuning_grid = cfg["grid"]
        n_combos = _count_combinations(tuning_grid)

        grid_label = cfg["mode"].upper()
        total_fits = n_combos * n_folds
        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] "
            f"Модель {current_model}/{total_models}: {algo_name} "
            f"[{grid_label}]"
        )
        print(
            f"  Комбинаций: {n_combos} | Фолдов: {n_folds} | "
            f"Фитов: {total_fits}"
        )

        # --- Строим Pipeline ---
        base_estimator, supports_grid = _build_estimator(algo_key, ncores)
        if base_estimator is None:
            print("  ! Метод недоступен - пропуск.")
            continue

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", base_estimator),
        ])

        # --- Обучение ---
        start_time = time.time()
        trained_model = None

        try:
            if tuning_grid and supports_grid:
                # GridSearchCV с прогресс-баром
                import joblib

                actual_njobs = min(ncores, n_combos)

                pbar = tqdm(
                    total=total_fits,
                    desc=f"  {algo_name}",
                    unit="fit",
                    bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} фитов [{elapsed}<{remaining}, {rate_fmt}]",
                    ncols=100,
                )

                if actual_njobs == 1:
                    # Последовательный режим: вручную перебираем комбинации
                    from sklearn.model_selection import ParameterGrid as PG
                    from sklearn.base import clone

                    best_score_ = -np.inf
                    best_params_ = None
                    all_cv_results = []

                    for params in PG(tuning_grid):
                        fold_scores = []
                        for train_idx, test_idx in cv_splits:
                            pipe_clone = clone(pipeline).set_params(**params)
                            pipe_clone.fit(X_all.iloc[train_idx], y_all[train_idx])
                            score = accuracy_score(
                                y_all[test_idx],
                                pipe_clone.predict(X_all.iloc[test_idx]),
                            )
                            fold_scores.append(score)
                            pbar.update(1)
                        mean_score = np.mean(fold_scores)
                        all_cv_results.append((params, mean_score))
                        if mean_score > best_score_:
                            best_score_ = mean_score
                            best_params_ = params

                    pbar.close()

                    # Refit на всех данных с лучшими параметрами
                    trained_model = clone(pipeline).set_params(**best_params_)
                    trained_model.fit(X_all, y_all)

                    print(f"  Лучшие параметры: ", end="")
                    bp = best_params_
                    parts = [
                        f"{k.replace('model__', '')}={v}"
                        for k, v in bp.items()
                        if "penalty" not in k
                    ]
                    print(", ".join(parts))
                else:
                    # Параллельный режим: патчим joblib для tqdm
                    gs = GridSearchCV(
                        estimator=pipeline,
                        param_grid=tuning_grid,
                        cv=cv_splits,
                        scoring="accuracy",
                        n_jobs=actual_njobs,
                        refit=True,
                        return_train_score=False,
                        error_score="raise",
                        verbose=0,
                    )

                    _orig_cb = joblib.parallel.BatchCompletionCallBack

                    class _TqdmBatchCB(joblib.parallel.BatchCompletionCallBack):
                        def __call__(self, *args, **kwargs):
                            pbar.update(self.batch_size)
                            return super().__call__(*args, **kwargs)

                    joblib.parallel.BatchCompletionCallBack = _TqdmBatchCB
                    try:
                        gs.fit(X_all, y_all)
                    finally:
                        joblib.parallel.BatchCompletionCallBack = _orig_cb
                        pbar.close()

                    trained_model = gs.best_estimator_

                    print(f"  Лучшие параметры: ", end="")
                    bp = gs.best_params_
                    parts = [
                        f"{k.replace('model__', '')}={v}"
                        for k, v in bp.items()
                        if "penalty" not in k
                    ]
                    print(", ".join(parts))
            else:
                # Без тюнинга — обучаем на всех данных
                pbar = tqdm(
                    total=1,
                    desc=f"  {algo_name}",
                    unit="fit",
                    bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} фитов [{elapsed}<{remaining}]",
                    ncols=100,
                )
                pipeline.fit(X_all, y_all)
                pbar.update(1)
                pbar.close()
                trained_model = pipeline

            elapsed = time.time() - start_time
            print(f"  ✓ Время: {elapsed:.1f} сек.")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  ✗ Ошибка: {e}")
            continue

        trained_model = _attach_predictor_names(trained_model, predictor_cols)
        all_models[algo_name] = trained_model

        # --- Метрики через cross_val_predict ---
        try:
            y_pred = cross_val_predict(
                trained_model, X_all, y_all, cv=cv_splits, n_jobs=ncores
            )
            y_proba = cross_val_predict(
                trained_model, X_all, y_all, cv=cv_splits,
                method="predict_proba", n_jobs=ncores,
            )
        except Exception:
            # Fallback: предсказание на трейне (не идеально, но работает)
            y_pred = trained_model.predict(X_all)
            try:
                y_proba = trained_model.predict_proba(X_all)
            except Exception:
                y_proba = None

        overall_acc = accuracy_score(y_all, y_pred)
        kappa = cohen_kappa_score(y_all, y_pred)
        f1_macro = f1_score(y_all, y_pred, average="macro", zero_division=0)
        precision_macro = precision_score(y_all, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_all, y_pred, average="macro", zero_division=0)

        # Sensitivity / Specificity (macro)
        cm_arr = confusion_matrix(y_all, y_pred)
        n_cls = len(class_levels)
        sensitivities = []
        specificities = []
        for ci in range(n_cls):
            tp = cm_arr[ci, ci]
            fn = cm_arr[ci, :].sum() - tp
            fp = cm_arr[:, ci].sum() - tp
            tn = cm_arr.sum() - tp - fn - fp
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            sensitivities.append(sens)
            specificities.append(spec)

        sensitivity = np.mean(sensitivities)
        specificity = np.mean(specificities)
        tss = sensitivity + specificity - 1

        # Balanced Accuracy
        balanced_acc = np.mean(
            [(cm_arr[ci, ci] / max(cm_arr[ci, :].sum(), 1)) for ci in range(n_cls)]
        )

        # AIC
        aic_val = _compute_aic(trained_model, X_all, y_all, n_cls)
        aic_values[algo_name] = aic_val

        # --- Confusion matrix ---
        cm_df = pd.DataFrame(
            cm_arr, index=class_levels, columns=class_levels
        )
        confusion_matrices[algo_name] = cm_df
        safe_name = re.sub(r"[^A-Za-z0-9]", "_", algo_name)
        cm_df.to_csv(os.path.join(output_dir, f"confusion_matrix_{safe_name}.csv"))

        # --- Per-class metrics ---
        report = classification_report(
            y_all, y_pred, target_names=class_levels, output_dict=True,
            zero_division=0,
        )
        class_metrics_df = pd.DataFrame(report).T
        class_metrics_df["Model"] = algo_name
        class_metrics_list[algo_name] = class_metrics_df
        class_metrics_df.to_csv(
            os.path.join(output_dir, f"class_metrics_{safe_name}.csv")
        )

        # --- Результат ---
        results_list.append({
            "Algorithm": algo_name,
            "Method": algo_key,
            "GridMode": grid_label,
            "OverallAccuracy": overall_acc,
            "BalancedAccuracy": balanced_acc,
            "Kappa": kappa,
            "TSS": tss,
            "F1_Macro": f1_macro,
            "Precision": precision_macro,
            "Recall": recall_macro,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "AIC": round(aic_val, 1) if aic_val is not None else None,
            "TrainingTime_Sec": round(elapsed, 1),
        })

        run_log["algorithms"][algo_name]["accuracy"] = round(overall_acc, 4)
        run_log["algorithms"][algo_name]["kappa"] = round(kappa, 4)
        run_log["algorithms"][algo_name]["time_sec"] = round(elapsed, 1)

        print(
            f"  Acc: {overall_acc:.4f} | Kappa: {kappa:.4f} | "
            f"TSS: {tss:.4f} | F1: {f1_macro:.4f} | "
            f"AIC: {'NA' if aic_val is None else f'{aic_val:.1f}'}"
        )

    # ================================================================
    # ШАГ 6: ГРАФИКИ CONFUSION MATRIX
    # ================================================================
    print("\n============================================================")
    print(" ГРАФИКИ CONFUSION MATRIX")
    print("============================================================\n")

    if confusion_matrices:
        for algo_name, cm_df in confusion_matrices.items():
            cm_arr = cm_df.values
            n_cls = len(cm_df)
            cls_names = list(cm_df.index)

            # Нормализация по строкам (%)
            row_sums = cm_arr.sum(axis=1, keepdims=True)
            cm_pct = cm_arr / np.maximum(row_sums, 1) * 100

            fig, ax = plt.subplots(
                figsize=(max(6, n_cls * 0.7 + 2), max(5, n_cls * 0.7 + 1))
            )
            sns.heatmap(
                cm_pct, annot=False, cmap="Blues",
                xticklabels=cls_names, yticklabels=cls_names,
                linewidths=0.5, linecolor="white", ax=ax,
                vmin=0, vmax=100,
            )
            # Аннотации: count + %
            for i in range(n_cls):
                for j in range(n_cls):
                    val = cm_arr[i, j]
                    pct = cm_pct[i, j]
                    if val > 0:
                        color = "white" if pct > 60 else "black"
                        ax.text(
                            j + 0.5, i + 0.5,
                            f"{val}\n{pct:.0f}%",
                            ha="center", va="center",
                            fontsize=max(6, min(10, 100 // n_cls)),
                            color=color,
                        )

            # Accuracy из results_list
            acc_val = None
            for r in results_list:
                if r["Algorithm"] == algo_name:
                    acc_val = r["OverallAccuracy"]
                    break

            title = f"Confusion Matrix: {algo_name}"
            if acc_val is not None:
                title += f"\nOverall Accuracy: {acc_val * 100:.2f}%"
            ax.set_title(title, fontweight="bold", fontsize=11)
            ax.set_xlabel("Reference")
            ax.set_ylabel("Predicted")
            plt.tight_layout()

            safe_name = re.sub(r"[^A-Za-z0-9]", "_", algo_name)
            png_path = os.path.join(plots_dir, f"confusion_matrix_{safe_name}.png")
            plt.savefig(png_path, dpi=250, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"  ✓ {os.path.basename(png_path)}")
    else:
        print("  ! Confusion matrices отсутствуют.")

    # ================================================================
    # ШАГ 7: ИТОГОВАЯ СВОДКА
    # ================================================================
    print("\n============================================================")
    print(" ИТОГОВАЯ СВОДКА")
    print("============================================================\n")

    if not results_list:
        print("  ! Ни одна модель не была успешно обучена.")
        return

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values("OverallAccuracy", ascending=False).reset_index(drop=True)

    print_cols = [
        "Algorithm", "GridMode", "OverallAccuracy", "Kappa",
        "TSS", "F1_Macro", "AIC", "TrainingTime_Sec",
    ]
    print_cols = [c for c in print_cols if c in results_df.columns]
    print(results_df[print_cols].to_string(index=False))
    print()

    results_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

    # Сохраняем все модели
    with open(os.path.join(output_dir, "all_trained_models.pkl"), "wb") as f:
        pickle.dump(all_models, f)
    with open(os.path.join(output_dir, "predictor_names.pkl"), "wb") as f:
        pickle.dump(predictor_cols, f)
    pd.DataFrame({"predictor": predictor_cols}).to_csv(
        os.path.join(output_dir, "predictor_names.csv"), index=False
    )
    with open(os.path.join(output_dir, "predictor_names_for_04.txt"), "w", encoding="utf-8") as f:
        for nm in predictor_cols:
            f.write(f"{nm}\n")
    model_predictors = {name: predictor_cols for name in all_models.keys()}
    with open(os.path.join(output_dir, "model_predictors.pkl"), "wb") as f:
        pickle.dump(model_predictors, f)
    with open(os.path.join(output_dir, "aic_values.pkl"), "wb") as f:
        pickle.dump(aic_values, f)
    with open(os.path.join(output_dir, "confusion_matrices.pkl"), "wb") as f:
        pickle.dump(confusion_matrices, f)

    if class_metrics_list:
        all_class_metrics = pd.concat(class_metrics_list.values())
        all_class_metrics.to_csv(os.path.join(output_dir, "all_class_metrics.csv"))

    # --- Сравнительный график метрик ---
    print("  Строю сравнительные графики...")
    n_models = len(results_df)
    short_names = [re.sub(r" \(.*\)", "", a) for a in results_df["Algorithm"]]

    metrics_to_plot = ["OverallAccuracy", "Kappa", "TSS", "F1_Macro"]
    metric_labels = ["Overall Accuracy", "Kappa", "TSS", "F1 Macro"]
    metric_colors = ["#2166AC", "#4DAF4A", "#FF7F00", "#E41A1C"]

    x = np.arange(n_models)
    bar_width = 0.2
    fig, ax = plt.subplots(figsize=(max(8, n_models * 2), 6))

    for mi, (metric, label, color) in enumerate(
        zip(metrics_to_plot, metric_labels, metric_colors)
    ):
        vals = results_df[metric].values
        bars = ax.bar(
            x + mi * bar_width, vals, bar_width,
            label=label, color=color,
        )
        for bi, v in enumerate(vals):
            ax.text(
                x[bi] + mi * bar_width, v + 0.015,
                f"{v:.3f}", ha="center", fontsize=7, rotation=90,
            )

    ax.set_title("Сравнение метрик качества", fontweight="bold", fontsize=14)
    ax.set_ylabel("Значение метрики")
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(short_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    png_metrics = os.path.join(plots_dir, "model_comparison_metrics.png")
    plt.savefig(png_metrics, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {os.path.basename(png_metrics)}")

    # --- AIC ---
    aic_vals = results_df["AIC"].values
    if any(pd.notna(aic_vals)):
        fig, ax = plt.subplots(figsize=(max(8, n_models * 2), 5))
        aic_plot = np.where(pd.isna(aic_vals), 0, aic_vals)
        bar_cols = ["#B0C4DE"] * len(aic_vals)
        valid_aic = [v for v in aic_vals if pd.notna(v)]
        if valid_aic:
            min_idx = int(np.nanargmin(aic_vals))
            bar_cols[min_idx] = "#2166AC"
        for i in range(len(aic_vals)):
            if pd.isna(aic_vals[i]):
                bar_cols[i] = "gray85"

        ax.bar(short_names, aic_plot, color=bar_cols, edgecolor="none")
        ax.set_title("AIC моделей (меньше = лучше)", fontweight="bold", fontsize=14)
        ax.set_ylabel("AIC")
        plt.xticks(rotation=45, ha="right")

        for i, v in enumerate(aic_plot):
            lbl = "NA" if pd.isna(aic_vals[i]) else f"{aic_vals[i]:.0f}"
            ax.text(i, v + max(aic_plot) * 0.02, lbl, ha="center", fontsize=9)

        plt.tight_layout()
        png_aic = os.path.join(plots_dir, "model_comparison_AIC.png")
        plt.savefig(png_aic, dpi=250, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  ✓ {os.path.basename(png_aic)}")

    print(f"\n✓ Графики: {plots_dir}")

    # --- Выбор лучшей модели ---
    print("\n============================================================")
    print(" ВЫБОР ЛУЧШЕЙ МОДЕЛИ")
    print("============================================================\n")

    # ΔAIC
    aic_series = results_df["AIC"].copy()
    if aic_series.notna().any():
        min_aic = aic_series.min()
        results_df["DeltaAIC"] = aic_series - min_aic
        results_df = results_df.sort_values(
            ["OverallAccuracy", "F1_Macro", "TSS", "DeltaAIC"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
    else:
        results_df["DeltaAIC"] = None
        results_df = results_df.sort_values(
            ["OverallAccuracy", "F1_Macro", "TSS"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    print("Ранжирование по OA, F1_Macro, TSS, ΔAIC:\n")
    for i, row in results_df.iterrows():
        marker = " <- лучшая" if i == 0 else ""
        daic = (
            "NA" if pd.isna(row.get("DeltaAIC"))
            else f"{row['DeltaAIC']:.1f}"
        )
        print(
            f" [{i + 1}] {row['Algorithm']:<30s} "
            f"Acc={row['OverallAccuracy']:.4f} "
            f"F1={row['F1_Macro']:.4f} "
            f"TSS={row['TSS']:.4f} "
            f"ΔAIC={daic}{marker}"
        )

    model_choice = input(f"\nВыберите номер модели (Enter = 1): ").strip()
    if model_choice == "":
        model_choice = "1"
    try:
        model_idx = int(model_choice) - 1
        if model_idx < 0 or model_idx >= len(results_df):
            model_idx = 0
            print("! Некорректный ввод - выбрана лучшая.")
    except ValueError:
        model_idx = 0
        print("! Некорректный ввод - выбрана лучшая.")

    best_model_name = results_df.iloc[model_idx]["Algorithm"]
    best_acc = results_df.iloc[model_idx]["OverallAccuracy"]
    print(f"\n* Выбрана: {best_model_name} (Accuracy = {best_acc:.4f})")

    # Сохраняем финальную модель
    final_model_path = os.path.join(output_dir, "final_model.pkl")
    with open(final_model_path, "wb") as f:
        pickle.dump(all_models[best_model_name], f)
    print(f"✓ Сохранена: {final_model_path}")

    final_predictor_path = os.path.join(output_dir, "final_model_predictors.pkl")
    with open(final_predictor_path, "wb") as f:
        pickle.dump(predictor_cols, f)
    print(f"✓ Предикторы финальной модели: {final_predictor_path}")

    run_log["best_model"] = best_model_name
    run_log["best_accuracy"] = round(best_acc, 4)

    # ================================================================
    # ШАГ 8: ОТЧЁТ (run_info.txt)
    # ================================================================
    info_lines = [
        "============================================================",
        f"  ОТЧЁТ О ЗАПУСКЕ ТЮНИНГА - {run_log['timestamp']}",
        "============================================================",
        "",
        f"Папка результатов: {output_dir}",
        f"Количество ядер: {run_log['ncores']}",
        f"Образцов: {run_log['n_samples']}",
        f"Предикторов: {run_log['n_predictors']}",
        f"Классов: {run_log['n_classes']} ({run_log['classes']})",
        f"Кросс-валидация: {run_log['cv_method']}",
        f"Предобработка: {run_log['preProcess']}",
        "",
        "------------------------------------------------------------",
        "  АЛГОРИТМЫ И ГИПЕРПАРАМЕТРЫ",
        "------------------------------------------------------------",
    ]

    for aname, a in run_log["algorithms"].items():
        info_lines.extend([
            "",
            f"> {aname}",
            f"  Режим: {a['mode'].upper()}",
            f"  Комбинаций: {a['n_combinations']}",
            f"  Параметры: {a['parameters']}",
            a.get("grid_values", ""),
        ])
        if "accuracy" in a:
            info_lines.extend([
                "  --- Результат ---",
                f"  Accuracy: {a['accuracy']:.4f}",
                f"  Kappa: {a['kappa']:.4f}",
                f"  Время: {a['time_sec']:.1f} сек.",
            ])

    if run_log.get("best_model"):
        info_lines.extend([
            "",
            "------------------------------------------------------------",
            "  ВЫБРАННАЯ МОДЕЛЬ",
            "------------------------------------------------------------",
            f"  * {run_log['best_model']} (Accuracy = {run_log['best_accuracy']:.4f})",
        ])

    info_lines.extend([
        "",
        "------------------------------------------------------------",
        f"  Скрипт завершён: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "============================================================",
    ])

    info_path = os.path.join(output_dir, "run_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("\n".join(info_lines))
    print(f"\n✓ Отчёт: {info_path}")

    print(f"\n============================================================")
    print(f"  СКРИПТ 03 ЗАВЕРШЁН - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Результаты: {output_dir}")
    print(f"  Следующий шаг: запустите 04_predict_map.py")
    print(f"============================================================\n")

    return {
        "output_dir": output_dir,
        "best_model_name": best_model_name,
        "best_accuracy": best_acc,
        "results_df": results_df,
        "all_models": all_models,
    }


# ================================================================
# ЗАПУСК
# ================================================================
if __name__ == "__main__":
    main()
