# =============================================================================
# 03_tune_models_v56.py (workspace: 03_tune_models_v6.py)
# v6/v56: режим равной априорной вероятности классов (по умолчанию):
#         sample_weight с одинаковой суммой весов на класс (+ выравнивание
#         полигонов n_pixels^-0.5) для RF/CatBoost/LightGBM/логистических/SVM,
#         priors = 1/K для LDA/QDA/RDA; подбор гиперпараметров и ранжирование
#         моделей по Balanced Accuracy; вывод числа предсказанных классов.
# (история) 03_tune_models_v55.py (workspace: 03_tune_models_v5.py)
# v5/v54: ускорение в 2–20x (предварительная стандартизация по фолдам 1 раз,
#         отказ от 2x cross_val_predict через сбор OOF в процессе CV,
#         быстрый спектральный решатель RDA, однократный поиск соседей в KNN,
#         оценка префиксов деревьев в CatBoost/LightGBM/RF, warm_start по C
#         в Ridge/Lasso/ElasticNet, быстрый ZV/NZV и векторный AIC)
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
    balanced_accuracy_score,
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
    Векторизованный аналог caret::nearZeroVar.
    freq_cut — отношение самого частого к второму по частоте (95/5 = 19).
    unique_cut — процент уникальных значений.
    Возвращает список имён ZV/NZV-переменных.
    """
    bad = []
    n = len(df)
    if n == 0 or not cols:
        return bad
    X_arr = df[cols].to_numpy(dtype=np.float64, copy=False)
    max_u = int(n * unique_cut / 100.0)
    probe_n = min(n, max(max_u * 2, 2000))
    for j, c in enumerate(cols):
        col = X_arr[:, j]
        # Быстрая отсечка явно непрерывных каналов по префиксу без полной сортировки
        probe = col[:probe_n]
        probe_v = probe[~np.isnan(probe)]
        if probe_v.size > max_u and np.unique(probe_v).size > max_u:
            continue
        valid = col[~np.isnan(col)]
        if valid.size <= 1:
            bad.append(c)
            continue
        u, counts = np.unique(valid, return_counts=True)
        n_unique = u.size
        if n_unique <= 1:
            bad.append(c)
            continue
        pct_unique = 100.0 * n_unique / max(n, 1)
        if pct_unique <= unique_cut:
            if counts.size >= 2:
                top2 = np.partition(counts, -2)[-2:]
                ratio = top2[1] / max(top2[0], 1)
            else:
                ratio = float("inf")
            if ratio >= freq_cut:
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

def _compute_aic(model, X, y_true, n_classes: int, X_scaled: np.ndarray | None = None) -> float | None:
    """Векторизованное вычисление AIC через log-loss. Возвращает None при ошибке."""
    try:
        if X_scaled is not None and hasattr(model, "named_steps") and "model" in model.named_steps:
            probs = model.named_steps["model"].predict_proba(X_scaled)
        else:
            probs = model.predict_proba(X)
        eps = 1e-15
        probs = np.clip(probs, eps, 1.0 - eps)
        ll = -float(np.sum(np.log(probs[np.arange(len(y_true)), y_true])))
        k = _estimate_nparams(model, X.shape[1], n_classes)
        return 2.0 * k + 2.0 * ll
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

# =============================================================================
# v56: РАВНАЯ АПРИОРНАЯ ВЕРОЯТНОСТЬ КЛАССОВ
# =============================================================================
# При обучении «пиксель = образец» модель по умолчанию считает, что класс
# встречается так же часто, как в выборке пикселей. Крупные таксационные
# классы (десятки тысяч пикселей) тогда «съедают» редкие классы из кружков
# R = 10 м (единицы пикселей). В режиме равной вероятности сумма весов
# каждого класса одинакова (N/K), то есть априорная вероятность каждого класса
# = 1/K. Дополнительно можно выровнять вклад полигонов внутри класса
# (вес пикселя ~ n_pixels^-alpha), чтобы один огромный выдел не определял
# весь класс.

EQUAL_CLASS_PRIOR = False   # устанавливается в main() по выбору пользователя
POLY_ALPHA = 0.0            # степень выравнивания полигонов (0 = не выравнивать)


def _score_fn(y_true, y_pred) -> float:
    """Критерий выбора гиперпараметров: Balanced Accuracy в режиме равных
    вероятностей классов, иначе Overall Accuracy."""
    if EQUAL_CLASS_PRIOR:
        return float(balanced_accuracy_score(y_true, y_pred))
    return float(accuracy_score(y_true, y_pred))


def _class_balanced_weights(y, n_px=None, alpha: float = 0.0):
    """Веса образцов: сумма весов каждого класса = N/K (средний вес = 1).

    Если alpha > 0 и известен n_px (число пикселей полигона), внутри класса
    вес пикселя пропорционален n_px^-alpha (alpha = 1 — каждый полигон
    весит одинаково, alpha = 0.5 — компромисс), затем нормируется так,
    чтобы сумма по классу осталась N/K.
    """
    y = np.asarray(y)
    N = y.shape[0]
    classes, y_idx = np.unique(y, return_inverse=True)
    K = classes.size
    base = np.ones(N, dtype=np.float64)
    if alpha and n_px is not None:
        npx = np.asarray(n_px, dtype=np.float64)
        npx = np.where(np.isfinite(npx) & (npx > 0), npx, 1.0)
        base = npx ** (-float(alpha))
    sums = np.bincount(y_idx, weights=base, minlength=K)
    sums = np.where(sums > 0, sums, 1.0)
    w = base * (N / float(K)) / sums[y_idx]
    return w


def _fit_w(est, X, y, w=None):
    """fit с учётом режима весов.

    - LDA/QDA/RDA: веса не поддерживаются -> priors = 1/K (по классам,
      присутствующим в обучающей части фолда);
    - KNN: веса не поддерживаются -> обычный fit;
    - остальные (RF, CatBoost, LightGBM, логистические, SVM): sample_weight.
    """
    if isinstance(est, (LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)):
        if EQUAL_CLASS_PRIOR:
            k = int(np.unique(y).size)
            est.set_params(priors=np.full(k, 1.0 / k))
        else:
            est.set_params(priors=None)
        return est.fit(X, y)
    if w is None or (SVM_KNN_AVAILABLE and isinstance(est, KNeighborsClassifier)):
        return est.fit(X, y)
    try:
        return est.fit(X, y, sample_weight=w)
    except TypeError:
        return est.fit(X, y)


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


def _load_samples_safe(data_dir: str, need_xy: bool = True) -> pd.DataFrame:
    """Надёжное чтение обучающей выборки из папки 01_PREPARE_DATA.

    Защищает от ошибки SQLite/GDAL 'attempt to write a readonly database'
    на Яндекс.Диске (когда у samples.gpkg включён WAL или файл удерживается
    синхронизацией):
      1) прямое чтение samples.gpkg;
      2) копирование samples.gpkg в локальный TEMP + сброс флага WAL в заголовке;
      3) резервное чтение samples.csv из той же папки.
    """
    import tempfile as _tempfile, shutil as _shutil, sqlite3 as _sqlite3
    samples_path = os.path.join(data_dir, "samples.gpkg")
    csv_path = os.path.join(data_dir, "samples.csv")
    if not os.path.exists(samples_path) and not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Файлы samples.gpkg и samples.csv не найдены в {data_dir}!\n"
            f"Сначала запустите 01_prepare_data.py"
        )
    gdf = None
    if os.path.exists(samples_path):
        try:
            gdf = gpd.read_file(samples_path, layer="samples")
        except Exception as e1:
            print(f"  ℹ Прямое чтение samples.gpkg не удалось ({e1}); "
                  f"открываю через локальный TEMP...")
            tmp_dir = _tempfile.mkdtemp(prefix="gpkg_read_")
            try:
                tmp_gpkg = os.path.join(tmp_dir, "samples.gpkg")
                _shutil.copyfile(samples_path, tmp_gpkg)
                for ext in ("-wal", "-shm", "-journal"):
                    if os.path.exists(samples_path + ext):
                        try:
                            _shutil.copyfile(samples_path + ext, tmp_gpkg + ext)
                        except Exception:
                            pass
                try:
                    _con = _sqlite3.connect(tmp_gpkg)
                    _con.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                    _con.execute("PRAGMA journal_mode=DELETE;")
                    _con.close()
                except Exception:
                    pass
                try:
                    with open(tmp_gpkg, "r+b") as _fh:
                        _hdr = bytearray(_fh.read(32))
                        if len(_hdr) >= 20 and (_hdr[18] == 2 or _hdr[19] == 2):
                            _hdr[18] = 1
                            _hdr[19] = 1
                            _fh.seek(0)
                            _fh.write(_hdr)
                except Exception:
                    pass
                gdf = gpd.read_file(tmp_gpkg, layer="samples")
                # Попробуем починить заголовок исходного samples.gpkg на диске
                try:
                    with open(samples_path, "r+b") as _fh:
                        _hdr = bytearray(_fh.read(32))
                        if len(_hdr) >= 20 and (_hdr[18] == 2 or _hdr[19] == 2):
                            _hdr[18] = 1
                            _hdr[19] = 1
                            _fh.seek(0)
                            _fh.write(_hdr)
                except Exception:
                    pass
            except Exception as e2:
                print(f"  ⚠ Чтение локальной копии samples.gpkg не удалось ({e2}); "
                      f"перехожу на samples.csv...")
            finally:
                _shutil.rmtree(tmp_dir, ignore_errors=True)

    if gdf is not None:
        df = pd.DataFrame(gdf.drop(columns="geometry"))
        if need_xy:
            df["x"] = gdf.geometry.x.values
            df["y"] = gdf.geometry.y.values
        return df

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"  ✓ Данные загружены из samples.csv ({len(df)} строк)")
        return df

    raise RuntimeError(f"Не удалось прочитать обучающие данные из {data_dir}")



try:
    import functools as _functools
    import sklearn as _sklearn
    from sklearn.base import BaseEstimator as _BaseEstimator
    if not hasattr(_BaseEstimator, "_gpn_cached"):
        _orig_gpn = _BaseEstimator._get_param_names.__func__
        @_functools.lru_cache(maxsize=None)
        def _cached_gpn_impl(cls):
            return _orig_gpn(cls)
        _BaseEstimator._get_param_names = classmethod(_cached_gpn_impl)
        _BaseEstimator._gpn_cached = True
except Exception:
    pass


def _prepare_fold_caches(
    X_df: pd.DataFrame,
    y_all: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
):
    """Однократная импутация (median) и стандартизация (z-score) для всех фолдов и всей выборки.

    Возвращает:
      fold_caches: список словарей по фолдам (X_tr64, X_te64, X_tr32, X_te32, y_tr, y_te, te_idx)
      full_cache: словарь (X_full64, X_full32, imputer_full, scaler_full)
    """
    X_raw = X_df.to_numpy(dtype=np.float64, copy=False)
    has_nan_col = np.isnan(X_raw).any(axis=0)
    any_nan = bool(np.any(has_nan_col))

    fold_caches = []
    for tr_idx, te_idx in cv_splits:
        X_tr = X_raw[tr_idx]
        X_te = X_raw[te_idx]
        if any_nan:
            imp = SimpleImputer(strategy="median")
            scl = StandardScaler()
            X_tr64 = np.ascontiguousarray(scl.fit_transform(imp.fit_transform(X_tr)), dtype=np.float64)
            X_te64 = np.ascontiguousarray(scl.transform(imp.transform(X_te)), dtype=np.float64)
        else:
            mean = X_tr.mean(axis=0)
            std = X_tr.std(axis=0, ddof=0)
            std_safe = np.where(std == 0.0, 1.0, std)
            X_tr64 = np.ascontiguousarray((X_tr - mean) / std_safe, dtype=np.float64)
            X_te64 = np.ascontiguousarray((X_te - mean) / std_safe, dtype=np.float64)
        fold_caches.append({
            "X_tr64": X_tr64,
            "X_te64": X_te64,
            "X_tr32": np.ascontiguousarray(X_tr64, dtype=np.float32),
            "X_te32": np.ascontiguousarray(X_te64, dtype=np.float32),
            "y_tr": y_all[tr_idx],
            "y_te": y_all[te_idx],
            "te_idx": te_idx,
        })

    imputer_full = SimpleImputer(strategy="median")
    scaler_full = StandardScaler()
    X_full64 = np.ascontiguousarray(
        scaler_full.fit_transform(imputer_full.fit_transform(X_df)),
        dtype=np.float64,
    )
    full_cache = {
        "X_full64": X_full64,
        "X_full32": np.ascontiguousarray(X_full64, dtype=np.float32),
        "imputer_full": imputer_full,
        "scaler_full": scaler_full,
    }
    return fold_caches, full_cache


def _build_fitted_pipeline(full_cache: dict, fitted_model, predictor_cols: list[str]) -> Pipeline:
    """Собирает стандартный sklearn Pipeline из уже обученных imputer, scaler и модели."""
    import copy
    pipe = Pipeline([
        ("imputer", copy.deepcopy(full_cache["imputer_full"])),
        ("scaler", copy.deepcopy(full_cache["scaler_full"])),
        ("model", fitted_model),
    ])
    return _attach_predictor_names(pipe, predictor_cols)


def _strip_model_prefix(params: dict) -> dict:
    return {
        (k[7:] if k.startswith("model__") else k): v
        for k, v in params.items()
    }


def _eval_rda_fold_fast(fc: dict, shrinkages: list[float]) -> dict[int, tuple[float, np.ndarray]]:
    """Быстрый точный расчёт RDA (LinearDiscriminantAnalysis solver='lsqr') для всех shrinkage на одном фолде."""
    from scipy import linalg
    X_tr = fc["X_tr64"]
    y_tr = fc["y_tr"]
    X_te = fc["X_te64"]
    y_te = fc["y_te"]

    N, p = X_tr.shape
    classes, y_idx = np.unique(y_tr, return_inverse=True)
    K = classes.size
    counts = np.bincount(y_idx, minlength=K).astype(np.float64)
    if EQUAL_CLASS_PRIOR:
        priors = np.full(K, 1.0 / K)      # v56: равные априорные вероятности
    else:
        priors = counts / float(N)
    log_priors = np.log(priors)

    Y_oh = np.zeros((N, K), dtype=np.float64)
    Y_oh[np.arange(N), y_idx] = 1.0
    means = (Y_oh.T @ X_tr) / counts[:, None]
    # Взвешенная внутриклассовая ковариация: sum_k priors_k * cov_k
    # (как в sklearn LDA с заданными priors); при priors = counts/N совпадает
    # с прежней формулой.
    w_row = (priors / counts)[y_idx]
    cov_base = (X_tr * w_row[:, None]).T @ X_tr - (means * priors[:, None]).T @ means
    mu_trace = np.trace(cov_base) / float(p)

    evals, evecs = linalg.eigh(cov_base)
    means_proj = means @ evecs
    X_te_proj = X_te @ evecs
    eps_cond = np.finfo(np.float64).eps * max(cov_base.shape)

    out = {}
    for si, s in enumerate(shrinkages):
        s_val = float(s)
        d = (1.0 - s_val) * evals + s_val * mu_trace
        cutoff = eps_cond * float(np.max(np.abs(d)))
        inv_d = np.where(d > cutoff, 1.0 / d, 0.0)
        coef_proj = means_proj * inv_d[None, :]
        intercept = -0.5 * np.einsum("ij,ij->i", means_proj, coef_proj) + log_priors
        scores = X_te_proj @ coef_proj.T + intercept
        preds = classes[np.argmax(scores, axis=1)]
        out[si] = (_score_fn(y_te, preds), preds)
    return out


def _eval_knn_fold_fast(
    fc: dict,
    combos: list[dict],
    inner_jobs: int = 1,
) -> dict[int, tuple[float, np.ndarray]]:
    """Быстрый точный расчёт KNN: kneighbors считается один раз на каждую метрику до max(n_neighbors)."""
    X_tr = fc["X_tr64"]
    y_tr = fc["y_tr"]
    X_te = fc["X_te64"]
    y_te = fc["y_te"]

    classes, y_idx = np.unique(y_tr, return_inverse=True)
    n_classes = classes.size

    # Группируем комбинации по metric
    by_metric: dict[str, list[tuple[int, int, str]]] = {}
    for ci, params in enumerate(combos):
        mp = _strip_model_prefix(params)
        metric = mp.get("metric", "minkowski")
        k = int(mp.get("n_neighbors", 5))
        w = mp.get("weights", "uniform")
        by_metric.setdefault(metric, []).append((ci, k, w))

    out = {}
    for metric, items in by_metric.items():
        k_max = max(k for _, k, _ in items)
        knn_base = KNeighborsClassifier(n_neighbors=k_max, metric=metric, n_jobs=inner_jobs)
        knn_base.fit(X_tr, y_tr)
        neigh_dist, neigh_ind = knn_base.kneighbors(X_te, n_neighbors=k_max, return_distance=True)
        neigh_y = y_idx[neigh_ind]

        for ci, k, w in items:
            ny = neigh_y[:, :k]
            if w == "uniform":
                counts = np.zeros((X_te.shape[0], n_classes), dtype=np.int32)
                for c in range(n_classes):
                    counts[:, c] = np.sum(ny == c, axis=1)
                preds = classes[np.argmax(counts, axis=1)]
            else:
                nd = neigh_dist[:, :k]
                with np.errstate(divide="ignore"):
                    w_arr = 1.0 / nd
                inf_mask = np.isinf(w_arr)
                if np.any(inf_mask):
                    w_arr = np.where(inf_mask, 1.0, 0.0)
                w_scores = np.zeros((X_te.shape[0], n_classes), dtype=np.float64)
                for c in range(n_classes):
                    w_scores[:, c] = np.sum(np.where(ny == c, w_arr, 0.0), axis=1)
                preds = classes[np.argmax(w_scores, axis=1)]
            out[ci] = (_score_fn(y_te, preds), preds)
    return out


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

    samples_df = _load_samples_safe(prepare_dir, need_xy=False)

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
    service_cols = {"class_name", "x", "y", "ID", "polygon_id", "n_pixels"}
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

    # --- Быстрая векторная проверка sd предикторов по CV-фолдам ---
    fold_zero_sd = set()
    _X_sd_check = samples_df[predictor_cols].to_numpy(dtype=np.float64, copy=False)
    if n_folds >= 2 and len(folds_test[0]) > 1 and len(folds_test[1]) > 1:
        _t0 = np.asarray(folds_test[0], dtype=np.intp)
        _t1 = np.asarray(folds_test[1], dtype=np.intp)
        _sd0 = np.nanstd(_X_sd_check[_t0], axis=0, ddof=1)
        _sd1 = np.nanstd(_X_sd_check[_t1], axis=0, ddof=1)
        _scale = np.sqrt(max(min(_t0.size, _t1.size) - 1, 1) / max(_X_sd_check.shape[0], 1))
        _suspect = np.where(
            np.isnan(_sd0) | np.isnan(_sd1) | (np.minimum(_sd0, _sd1) * _scale < 1e-11)
        )[0]
    else:
        _suspect = np.arange(len(predictor_cols))
    if _suspect.size > 0:
        _X_sub = _X_sd_check[:, _suspect]
        for fi in range(n_folds):
            _tr_idx = np.asarray(folds_train[fi], dtype=np.intp)
            _sds_f = np.nanstd(_X_sub[_tr_idx], axis=0, ddof=1)
            for _loc_j in np.where(np.isnan(_sds_f) | (_sds_f < 1e-12))[0]:
                fold_zero_sd.add(predictor_cols[int(_suspect[_loc_j])])
    del _X_sd_check
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
    # ШАГ 3b: РЕЖИМ АПРИОРНЫХ ВЕРОЯТНОСТЕЙ КЛАССОВ (v56)
    # ================================================================
    print("============================================================")
    print(" ШАГ 3b: ВЕРОЯТНОСТИ КЛАССОВ ПРИ ОБУЧЕНИИ")
    print("============================================================\n")
    _cls_counts = samples_df["class_name"].value_counts()
    print(
        f"  Пикселей на класс: min={int(_cls_counts.min())} ({_cls_counts.idxmin()}), "
        f"max={int(_cls_counts.max())} ({_cls_counts.idxmax()}), "
        f"отношение max/min = {_cls_counts.max() / max(_cls_counts.min(), 1):.0f}\n"
    )
    print("  [1] Равная вероятность классов + выравнивание полигонов (alpha=0.5) - РЕКОМЕНДУЕМО")
    print("      сумма весов каждого класса одинакова; внутри класса вес пикселя ~ n_pixels^-0.5")
    print("  [2] Равная вероятность классов")
    print("      сумма весов каждого класса одинакова; все пиксели класса равноценны")
    print("  [3] Как в выборке (без весов, прежнее поведение)")
    print("      крупные классы получают преимущество пропорционально числу пикселей\n")
    _wm = input("Выбор (Enter = 1): ").strip()
    global EQUAL_CLASS_PRIOR, POLY_ALPHA
    if _wm == "3":
        EQUAL_CLASS_PRIOR, POLY_ALPHA = False, 0.0
        weight_mode_label = "без весов (вероятности классов как в выборке)"
    elif _wm == "2":
        EQUAL_CLASS_PRIOR, POLY_ALPHA = True, 0.0
        weight_mode_label = "равная вероятность классов"
    else:
        EQUAL_CLASS_PRIOR, POLY_ALPHA = True, 0.5
        weight_mode_label = "равная вероятность классов + выравнивание полигонов (alpha=0.5)"
    if EQUAL_CLASS_PRIOR and POLY_ALPHA > 0 and "n_pixels" not in samples_df.columns:
        if "polygon_id" in samples_df.columns:
            samples_df["n_pixels"] = samples_df.groupby("polygon_id")["polygon_id"].transform("size")
            print("  ℹ n_pixels пересчитан по polygon_id")
        else:
            POLY_ALPHA = 0.0
            weight_mode_label = "равная вероятность классов (нет polygon_id - без выравнивания полигонов)"
    if "n_pixels" in samples_df.columns:
        n_px_all = samples_df["n_pixels"].to_numpy(dtype=np.float64)
    else:
        n_px_all = np.ones(len(samples_df), dtype=np.float64)
    print(f"\n✓ Режим: {weight_mode_label}")
    if EQUAL_CLASS_PRIOR:
        print("  RF/CatBoost/LightGBM/логистические/SVM - sample_weight;")
        print("  LDA/QDA/RDA - priors = 1/K; KNN весов не поддерживает (обучается как раньше).")
        print("  Гиперпараметры подбираются по Balanced Accuracy, модели ранжируются по Balanced Accuracy.\n")
    else:
        print("  Гиперпараметры подбираются по Overall Accuracy.\n")

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
        "class_weighting": weight_mode_label,
        "equal_class_prior": bool(EQUAL_CLASS_PRIOR),
        "poly_alpha": float(POLY_ALPHA),
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

    # Подготовка данных и однократная стандартизация по фолдам
    import joblib
    from sklearn.base import clone

    predictor_cols = [str(c) for c in predictor_cols]
    X_all = samples_df[predictor_cols].astype(np.float64)
    y_all_str = samples_df["class_name"].values
    y_all = le.transform(y_all_str)

    cv_splits = [
        (np.asarray(folds_train[fi], dtype=np.intp), np.asarray(folds_test[fi], dtype=np.intp))
        for fi in range(n_folds)
    ]

    t_prep0 = time.time()
    fold_caches, full_cache = _prepare_fold_caches(X_all, y_all, cv_splits)
    print(f"  ✓ Кэш стандартизации {n_folds} фолдов подготовлен за {time.time() - t_prep0:.2f} сек.")

    # v56: веса образцов по фолдам (считаются только по обучающей части)
    if EQUAL_CLASS_PRIOR:
        for fi, (tr_idx, _te) in enumerate(cv_splits):
            fold_caches[fi]["w_tr"] = _class_balanced_weights(
                y_all[tr_idx], n_px_all[tr_idx], POLY_ALPHA
            )
        full_cache["w_full"] = _class_balanced_weights(y_all, n_px_all, POLY_ALPHA)
        _wf = full_cache["w_full"]
        _cls_sum = np.bincount(y_all, weights=_wf)
        print(
            f"  ✓ Веса образцов: min={_wf.min():.3f}, median={np.median(_wf):.3f}, "
            f"max={_wf.max():.3f}; сумма весов на класс = {_cls_sum.min():.1f}..{_cls_sum.max():.1f}"
        )
    else:
        for fc in fold_caches:
            fc["w_tr"] = None
        full_cache["w_full"] = None

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

        base_estimator, supports_grid = _build_estimator(algo_key, ncores)
        if base_estimator is None:
            print("  ! Метод недоступен - пропуск.")
            continue

        start_time = time.time()
        trained_model = None
        y_pred = None

        try:
            if tuning_grid and supports_grid:
                combos = list(ParameterGrid(tuning_grid))
                import threading as _threading
                _pbar_lock = _threading.Lock()
                pbar = tqdm(
                    total=total_fits,
                    desc=f"  {algo_name}",
                    unit="fit",
                    bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} фитов [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                    ncols=115,
                )

                # Matrix of fold scores (n_combos, n_folds) and OOF predictions per combo
                scores_mat = np.zeros((len(combos), n_folds), dtype=np.float64)
                oof_by_combo = [np.empty_like(y_all) for _ in range(len(combos))]

                if algo_key == "rda" and set(tuning_grid.keys()) == {"model__shrinkage"}:
                    # Быстрый спектральный решатель RDA (1 разложение ковариации на фолд)
                    shrinkages = [float(c["model__shrinkage"]) for c in combos]
                    outer_jobs = min(ncores, n_folds)
                    def _rda_task(fi: int):
                        return fi, _eval_rda_fold_fast(fold_caches[fi], shrinkages)
                    for fi, fdict in joblib.Parallel(
                        n_jobs=outer_jobs, prefer="threads", return_as="generator_unordered"
                    )(joblib.delayed(_rda_task)(fi) for fi in range(n_folds)):
                        te_idx = fold_caches[fi]["te_idx"]
                        for ci, (sc, preds) in fdict.items():
                            scores_mat[ci, fi] = sc
                            oof_by_combo[ci][te_idx] = preds
                        pbar.update(len(combos))

                elif algo_key == "knn":
                    # Быстрый KNN: kneighbors 1 раз на (фолд, метрика) до max(k) со всеми ядрами ncores
                    for fi in range(n_folds):
                        pbar.set_postfix_str(f"фолд {fi+1}/{n_folds}", refresh=True)
                        fdict = _eval_knn_fold_fast(fold_caches[fi], combos, ncores)
                        te_idx = fold_caches[fi]["te_idx"]
                        for ci, (sc, preds) in fdict.items():
                            scores_mat[ci, fi] = sc
                            oof_by_combo[ci][te_idx] = preds
                        pbar.update(len(combos))
                    pbar.set_postfix_str("", refresh=True)

                elif algo_key in ("ridge", "lasso", "elasticnet"):
                    # Warm-start вдоль возрастающего C внутри каждого фолда
                    groups_lr: dict[tuple, list[tuple[int, float, dict]]] = {}
                    for ci, params in enumerate(combos):
                        mp = _strip_model_prefix(params)
                        gkey = (mp.get("penalty", "l2"), mp.get("l1_ratio", None))
                        groups_lr.setdefault(gkey, []).append((ci, float(mp.get("C", 1.0)), mp))
                    for gkey in groups_lr:
                        groups_lr[gkey].sort(key=lambda t: t[1])

                    def _run_lr_fold(fi: int):
                        fc = fold_caches[fi]
                        out = {}
                        for _, items in groups_lr.items():
                            est, _ = _build_estimator(algo_key, 1)
                            est.set_params(warm_start=True)
                            for ci, _, mp in items:
                                est.set_params(**mp)
                                with _sklearn.config_context(assume_finite=True):
                                    _fit_w(est, fc["X_tr64"], fc["y_tr"], fc.get("w_tr"))
                                preds = est.predict(fc["X_te64"])
                                out[ci] = (_score_fn(fc["y_te"], preds), preds)
                                with _pbar_lock:
                                    pbar.update(1)
                        return fi, out

                    outer_jobs = min(ncores, n_folds)
                    for fi, fdict in joblib.Parallel(
                        n_jobs=outer_jobs, prefer="threads", return_as="generator_unordered"
                    )(joblib.delayed(_run_lr_fold)(fi) for fi in range(n_folds)):
                        te_idx = fold_caches[fi]["te_idx"]
                        for ci, (sc, preds) in fdict.items():
                            scores_mat[ci, fi] = sc
                            oof_by_combo[ci][te_idx] = preds

                elif algo_key in ("rf", "catboost", "lgbm"):
                    # Для ансамблей деревьев (RF, CatBoost, LightGBM) обучаем по одному фиту за раз,
                    # отдавая ВСЕ ncores потоков внутрь самой модели (n_jobs=ncores / thread_count=ncores).
                    # Это:
                    # 1) в ncores раз ускоряет завершение первого и каждого следующего фита (прогресс-бар
                    #    начинает двигаться через секунды, а не через минуты ожидания ncores параллельных лесов);
                    # 2) в ncores раз снижает пиковое потребление RAM и полностью исключает борьбу потоков за GIL;
                    # 3) для RF выводит живой счётчик построенных деревьев прямо внутри каждого фолда.
                    iter_key = "iterations" if algo_key == "catboost" else "n_estimators"
                    tree_groups: dict[tuple, list[tuple[int, int, dict]]] = {}
                    for ci, params in enumerate(combos):
                        mp = _strip_model_prefix(params)
                        n_trees = int(mp.get(iter_key, 500))
                        other_items = tuple(sorted((k, v) for k, v in mp.items() if k != iter_key))
                        tree_groups.setdefault(other_items, []).append((ci, n_trees, mp))
                    for gkey in tree_groups:
                        tree_groups[gkey].sort(key=lambda t: t[1])

                    # Сортируем группы так, чтобы быстрые комбинации (меньший max_features/depth, больший min_samples_leaf)
                    # шли первыми и сразу давали быстрый отклик прогресс-бара (результаты записываются по исходным индексам ci).
                    def _group_sort_key(items):
                        mp0 = items[-1][2]
                        mf = mp0.get("max_features", mp0.get("depth", mp0.get("num_leaves", 10)))
                        mf_val = mf if isinstance(mf, (int, float)) else 9999
                        msl = mp0.get("min_samples_leaf", 1)
                        msl_val = msl if isinstance(msl, (int, float)) else 1
                        return (mf_val, -msl_val)

                    group_list = sorted(tree_groups.values(), key=_group_sort_key)
                    use_f32 = algo_key in ("rf", "lgbm")
                    _orig_batch_cb = joblib.parallel.BatchCompletionCallBack

                    for gi, items in enumerate(group_list):
                        max_ci, max_trees, max_mp = items[-1]
                        if algo_key == "rf":
                            hint_str = f"mtry={max_mp.get('max_features')}, leaf={max_mp.get('min_samples_leaf', 1)}"
                        elif algo_key == "catboost":
                            hint_str = f"depth={max_mp.get('depth')}, lr={max_mp.get('learning_rate')}"
                        else:
                            hint_str = f"leaves={max_mp.get('num_leaves')}, lr={max_mp.get('learning_rate')}"

                        for fi in range(n_folds):
                            fc = fold_caches[fi]
                            X_tr = fc["X_tr32"] if use_f32 else fc["X_tr64"]
                            X_te = fc["X_te32"] if use_f32 else fc["X_te64"]
                            y_tr = fc["y_tr"]
                            y_te = fc["y_te"]

                            est, _ = _build_estimator(algo_key, ncores)
                            est.set_params(**max_mp)

                            if algo_key == "rf":
                                trees_done = [0]
                                last_t = [time.monotonic()]
                                pbar.set_postfix_str(
                                    f"фолд {fi+1}/{n_folds}, {hint_str}, дер. 0/{max_trees}",
                                    refresh=True,
                                )

                                class _TreeProgressCB(_orig_batch_cb):
                                    def __call__(self, *args, **kwargs):
                                        trees_done[0] += self.batch_size
                                        now = time.monotonic()
                                        if now - last_t[0] >= 0.25:
                                            last_t[0] = now
                                            pbar.set_postfix_str(
                                                f"фолд {fi+1}/{n_folds}, {hint_str}, дер. {min(trees_done[0], max_trees)}/{max_trees}",
                                                refresh=True,
                                            )
                                        return super().__call__(*args, **kwargs)

                                joblib.parallel.BatchCompletionCallBack = _TreeProgressCB
                                try:
                                    with _sklearn.config_context(assume_finite=True):
                                        _fit_w(est, X_tr, y_tr, fc.get("w_tr"))
                                finally:
                                    joblib.parallel.BatchCompletionCallBack = _orig_batch_cb
                            else:
                                pbar.set_postfix_str(f"фолд {fi+1}/{n_folds}, {hint_str}", refresh=True)
                                with _sklearn.config_context(assume_finite=True):
                                    _fit_w(est, X_tr, y_tr, fc.get("w_tr"))

                            te_idx = fc["te_idx"]
                            if len(items) == 1:
                                preds = np.asarray(est.predict(X_te)).reshape(-1)
                                scores_mat[max_ci, fi] = _score_fn(y_te, preds)
                                oof_by_combo[max_ci][te_idx] = preds
                            else:
                                orig_ests = getattr(est, "estimators_", None) if algo_key == "rf" else None
                                for ci, n_tr, mp in items:
                                    try:
                                        if algo_key == "rf" and orig_ests is not None:
                                            est.estimators_ = orig_ests[:n_tr]
                                            est.n_estimators = n_tr
                                            preds = np.asarray(est.predict(X_te)).reshape(-1)
                                        elif algo_key == "catboost":
                                            preds = np.asarray(est.predict(X_te, ntree_end=n_tr)).reshape(-1)
                                        elif algo_key == "lgbm":
                                            preds = np.asarray(est.predict(X_te, num_iteration=n_tr)).reshape(-1)
                                        else:
                                            preds = np.asarray(est.predict(X_te)).reshape(-1)
                                    except Exception:
                                        est_sub, _ = _build_estimator(algo_key, ncores)
                                        est_sub.set_params(**mp)
                                        with _sklearn.config_context(assume_finite=True):
                                            _fit_w(est_sub, X_tr, y_tr, fc.get("w_tr"))
                                        preds = np.asarray(est_sub.predict(X_te)).reshape(-1)
                                    scores_mat[ci, fi] = _score_fn(y_te, preds)
                                    oof_by_combo[ci][te_idx] = preds
                                if algo_key == "rf" and orig_ests is not None:
                                    est.estimators_ = orig_ests
                                    est.n_estimators = max_trees
                            del est
                            pbar.update(len(items))
                    pbar.set_postfix_str("", refresh=True)

                else:
                    # Универсальный параллельный перебор (SVM и др.) по предмасштабированным фолдам
                    tasks = [(ci, fi) for ci in range(len(combos)) for fi in range(n_folds)]
                    outer_jobs = min(ncores, len(tasks))
                    inner_jobs = max(1, ncores // max(outer_jobs, 1))

                    def _run_generic_fit(ci: int, fi: int):
                        fc = fold_caches[fi]
                        mp = _strip_model_prefix(combos[ci])
                        est, _ = _build_estimator(algo_key, inner_jobs)
                        est.set_params(**mp)
                        with _sklearn.config_context(assume_finite=True):
                                    _fit_w(est, fc["X_tr64"], fc["y_tr"], fc.get("w_tr"))
                        preds = np.asarray(est.predict(fc["X_te64"])).reshape(-1)
                        return ci, fi, _score_fn(fc["y_te"], preds), preds

                    for ci, fi, sc, preds in joblib.Parallel(
                        n_jobs=outer_jobs, prefer="threads", return_as="generator_unordered"
                    )(joblib.delayed(_run_generic_fit)(ci, fi) for ci, fi in tasks):
                        scores_mat[ci, fi] = sc
                        oof_by_combo[ci][fold_caches[fi]["te_idx"]] = preds
                        pbar.update(1)

                mean_scores = scores_mat.mean(axis=1)
                best_ci = int(np.argmax(mean_scores))
                best_params_ = combos[best_ci]
                y_pred = oof_by_combo[best_ci]

                # Финальный refit лучшей модели на всех стандартизованных данных (1 раз)
                pbar.set_postfix_str("финальное обучение на всей выборке...", refresh=True)
                best_mp = _strip_model_prefix(best_params_)
                final_est, _ = _build_estimator(algo_key, ncores)
                final_est.set_params(**best_mp)
                X_fit_full = full_cache["X_full32"] if algo_key in ("rf", "lgbm") else full_cache["X_full64"]
                with _sklearn.config_context(assume_finite=True):
                    _fit_w(final_est, X_fit_full, y_all, full_cache.get("w_full"))
                pbar.set_postfix_str("", refresh=True)
                pbar.close()
                trained_model = _build_fitted_pipeline(full_cache, final_est, predictor_cols)

                print(f"  Лучшие параметры: ", end="")
                parts = [
                    f"{k.replace('model__', '')}={v}"
                    for k, v in best_params_.items()
                    if "penalty" not in k
                ]
                print(", ".join(parts))

            else:
                # Без тюнинга (LDA / QDA): 10 фолдов для OOF y_pred + 1 финальный fit
                pbar = tqdm(
                    total=n_folds + 1,
                    desc=f"  {algo_name}",
                    unit="fit",
                    bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} фитов [{elapsed}<{remaining}]",
                    ncols=100,
                )
                y_pred = np.empty_like(y_all)

                def _run_notune_fold(fi: int):
                    fc = fold_caches[fi]
                    est, _ = _build_estimator(algo_key, 1)
                    with _sklearn.config_context(assume_finite=True):
                                    _fit_w(est, fc["X_tr64"], fc["y_tr"], fc.get("w_tr"))
                    preds = np.asarray(est.predict(fc["X_te64"])).reshape(-1)
                    return fi, preds

                outer_jobs = min(ncores, n_folds)
                for fi, preds in joblib.Parallel(
                    n_jobs=outer_jobs, prefer="threads", return_as="generator_unordered"
                )(joblib.delayed(_run_notune_fold)(fi) for fi in range(n_folds)):
                    y_pred[fold_caches[fi]["te_idx"]] = preds
                    pbar.update(1)

                final_est, _ = _build_estimator(algo_key, ncores)
                _fit_w(final_est, full_cache["X_full64"], y_all, full_cache.get("w_full"))
                trained_model = _build_fitted_pipeline(full_cache, final_est, predictor_cols)
                pbar.update(1)
                pbar.close()

            elapsed = time.time() - start_time
            print(f"  ✓ Время: {elapsed:.1f} сек.")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  ✗ Ошибка: {e}")
            continue

        all_models[algo_name] = trained_model

        # Если по какой-то причине y_pred не заполнен — резервный расчёт
        if y_pred is None:
            try:
                y_pred = cross_val_predict(
                    trained_model, X_all, y_all, cv=cv_splits, n_jobs=ncores
                )
            except Exception:
                y_pred = trained_model.predict(X_all)

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
        aic_val = _compute_aic(trained_model, X_all, y_all, n_cls, X_scaled=full_cache['X_full64'])
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
        run_log["algorithms"][algo_name]["balanced_accuracy"] = round(float(balanced_acc), 4)
        run_log["algorithms"][algo_name]["f1_macro"] = round(float(f1_macro), 4)
        run_log["algorithms"][algo_name]["time_sec"] = round(elapsed, 1)

        print(
            f"  OA: {overall_acc:.4f} | BalAcc: {balanced_acc:.4f} | "
            f"Предсказано классов: {len(np.unique(y_pred))}/{n_cls} | Kappa: {kappa:.4f} | "
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
            plt.savefig(png_path, dpi=(180 if n_cls > 20 else 250), bbox_inches="tight", facecolor="white", pil_kwargs={"compress_level": 1})
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
    rank_metric = "BalancedAccuracy" if EQUAL_CLASS_PRIOR else "OverallAccuracy"
    results_df = results_df.sort_values(rank_metric, ascending=False).reset_index(drop=True)

    print_cols = [
        "Algorithm", "GridMode", "OverallAccuracy", "BalancedAccuracy", "Kappa",
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
            [rank_metric, "F1_Macro", "TSS", "DeltaAIC"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
    else:
        results_df["DeltaAIC"] = None
        results_df = results_df.sort_values(
            [rank_metric, "F1_Macro", "TSS"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    print(f"Ранжирование по {'BalancedAccuracy' if EQUAL_CLASS_PRIOR else 'OA'}, F1_Macro, TSS, ΔAIC:\n")
    for i, row in results_df.iterrows():
        marker = " <- лучшая" if i == 0 else ""
        daic = (
            "NA" if pd.isna(row.get("DeltaAIC"))
            else f"{row['DeltaAIC']:.1f}"
        )
        print(
            f" [{i + 1}] {row['Algorithm']:<30s} "
            f"Acc={row['OverallAccuracy']:.4f} "
            f"BalAcc={row['BalancedAccuracy']:.4f} "
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
    best_bacc = results_df.iloc[model_idx]["BalancedAccuracy"]
    print(f"\n* Выбрана: {best_model_name} (Accuracy = {best_acc:.4f}, Balanced Accuracy = {best_bacc:.4f})")

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
        f"Вероятности классов: {run_log['class_weighting']}",
        f"Критерий подбора гиперпараметров: {'Balanced Accuracy' if run_log['equal_class_prior'] else 'Overall Accuracy'}",
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
                f"  Balanced Accuracy: {a.get('balanced_accuracy', float('nan')):.4f}",
                f"  F1 macro: {a.get('f1_macro', float('nan')):.4f}",
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
