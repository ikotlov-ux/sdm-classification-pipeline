# ============================================================================
# 02_classification_v15.py
# СКРИПТ 02: КЛАССИФИКАЦИЯ — ВЫБОР АЛГОРИТМОВ И НАСТРОЙКА CV
#
# Функционал:
# - Загрузка обучающих данных из 01_PREPARE_DATA (samples.gpkg)
# - Выбор метода кросс-валидации:
#     1) Стратифицированная KFold (классический)
#     2) Spatial Block CV + стратификация (KMeans по координатам)
# - Визуализация распределения фолдов (stacked bar + heatmap + карта блоков)
# - Интерактивный выбор алгоритмов (RF, CatBoost, Ridge/Lasso/EN, LDA/QDA/RDA)
# - Сохранение конфигурации для 03_tune_models.py
# ============================================================================

import os
import re
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupKFold
from sklearn.cluster import KMeans

try:
    import lightgbm  # noqa: F401
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


# ============================================================================
# НАСТРОЙКА ПАПОК
# ============================================================================

def _select_prev_subdir(base_dir: str, label: str) -> tuple[str, str]:
    """Интерактивный выбор подпапки из base_dir. Возвращает (путь, суффикс)."""
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
        raise FileNotFoundError(f"В {os.path.basename(base_dir)} нет подпапок с результатами.")

    print(f"\nДоступные запуски в {os.path.basename(base_dir)}:\n")
    for i, sd in enumerate(subdirs, 1):
        print(f" [{i}] {os.path.basename(sd)}")

    raw = input(f"\nВыберите номер запуска из {os.path.basename(base_dir)}: ").strip()
    if raw == "":
        raw = "1"
    try:
        idx = int(raw)
    except ValueError:
        raise ValueError("Некорректный выбор запуска.")
    if idx < 1 or idx > len(subdirs):
        raise ValueError("Некорректный выбор запуска.")

    selected = subdirs[idx - 1]
    name = os.path.basename(selected)
    m = re.search(r"(_v\d{3})$", name)
    suffix = m.group(1) if m else "_v001"
    print(f"Выбрана папка: {name}, суффикс: {suffix}\n")
    return selected, suffix


# ============================================================================
# SPATIAL BLOCK CV С СТРАТИФИКАЦИЕЙ
# ============================================================================

def _spatial_block_cv(
    coords: np.ndarray,
    y: np.ndarray,
    n_folds: int = 10,
    block_multiplier: int = 5,
    random_state: int = 42,
    groups: np.ndarray | None = None,
) -> tuple[dict, dict, np.ndarray]:
    """
    Spatial Block Cross-Validation со стратификацией.

    Алгоритм:
    1) KMeans кластеризация координат → n_blocks пространственных блоков
    2) Жадное распределение блоков по фолдам:
       - на каждом шаге выбирается фолд с наименьшим числом образцов
       - блок назначается так, чтобы максимизировать покрытие классов
    3) Проверка: все классы должны быть в каждом фолде

    Параметры:
    ----------
    coords : np.ndarray, shape (n_samples, 2)
        Координаты X, Y в метрической СК
    y : np.ndarray, shape (n_samples,)
        Метки классов
    n_folds : int
        Количество фолдов
    block_multiplier : int
        Коэффициент: n_blocks = n_folds * block_multiplier
    random_state : int
        Seed для воспроизводимости

    Возвращает:
    ----------
    folds_train : dict {fold_idx: [indices]}
    folds_test  : dict {fold_idx: [indices]}
    block_labels : np.ndarray — номер блока для каждого образца
    """
    n_samples = len(y)
    n_blocks = n_folds * block_multiplier
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    # --- Шаг 1: KMeans кластеризация (с учётом groups=polygon_id) ---
    if groups is not None:
        # Групповой режим (вариант B): все пиксели одного polygon_id
        # ОБЯЗАНЫ оказаться в одном блоке. Кластеризуем НЕ пиксели, а
        # центроиды полигонов, затем метку блока полигона раздаём всем
        # его пикселям.
        uniq_groups, first_idx = np.unique(groups, return_index=True)
        # Центроид каждого полигона — среднее координат его пикселей
        group_centroids = np.array([
            coords[groups == g].mean(axis=0) for g in uniq_groups
        ])
        n_groups = len(uniq_groups)
        n_blocks = min(n_blocks, n_groups)
        print(
            f"    Групповой режим: {n_groups} полигонов → "
            f"KMeans на центроидах (не пикселях)"
        )
        print(f"    Кластеризация: {n_blocks} пространственных блоков...")
        kmeans = KMeans(
            n_clusters=n_blocks, random_state=random_state,
            n_init=10, max_iter=300,
        )
        group_block_labels = kmeans.fit_predict(group_centroids)
        # Маппинг group_id -> block_id
        g2b = dict(zip(uniq_groups, group_block_labels))
        block_labels = np.array([g2b[g] for g in groups])
    else:
        # Обычный пиксельный режим
        n_blocks = min(n_blocks, n_samples)
        print(f"    KMeans кластеризация: {n_blocks} пространственных блоков...")
        kmeans = KMeans(
            n_clusters=n_blocks, random_state=random_state,
            n_init=10, max_iter=300,
        )
        block_labels = kmeans.fit_predict(coords)

    # --- Сводка по блокам ---
    block_ids = np.unique(block_labels)
    block_info = {}  # block_id -> {class -> count}
    for bid in block_ids:
        mask = block_labels == bid
        classes_in_block = y[mask]
        class_counts = {}
        for cls in unique_classes:
            class_counts[cls] = int((classes_in_block == cls).sum())
        block_info[bid] = {
            "total": int(mask.sum()),
            "class_counts": class_counts,
            "classes_present": set(
                cls for cls, cnt in class_counts.items() if cnt > 0
            ),
        }

    print(f"    Блоков создано: {len(block_ids)}")
    print(
        f"    Размер блоков: "
        f"min={min(b['total'] for b in block_info.values())}, "
        f"max={max(b['total'] for b in block_info.values())}, "
        f"median={int(np.median([b['total'] for b in block_info.values()]))}"
    )

    # --- Шаг 2: жадное распределение блоков по фолдам ---
    # Стратегия: сортируем блоки по убыванию числа уникальных классов
    # (сначала распределяем «богатые» блоки, потом «бедные»)
    sorted_blocks = sorted(
        block_ids,
        key=lambda bid: (
            -len(block_info[bid]["classes_present"]),
            -block_info[bid]["total"],
        ),
    )

    fold_assignments = {fi: [] for fi in range(n_folds)}  # fold -> [block_ids]
    fold_sizes = np.zeros(n_folds, dtype=int)
    fold_class_counts = {fi: {cls: 0 for cls in unique_classes} for fi in range(n_folds)}

    for bid in sorted_blocks:
        binfo = block_info[bid]

        # Оценка: для каждого фолда считаем «полезность» добавления блока
        best_fold = None
        best_score = -np.inf

        for fi in range(n_folds):
            # Штраф за дисбаланс размера (предпочитаем фолд с меньшим числом образцов)
            size_score = -fold_sizes[fi]

            # Бонус за новые классы в фолде
            new_classes = 0
            for cls in binfo["classes_present"]:
                if fold_class_counts[fi][cls] == 0:
                    new_classes += 1
            class_score = new_classes * 1000  # высокий приоритет

            # Бонус за выравнивание классов (добавляем в фолд, где класса меньше всего)
            balance_score = 0
            for cls in binfo["classes_present"]:
                cnt = binfo["class_counts"][cls]
                balance_score -= fold_class_counts[fi][cls] * cnt

            score = size_score + class_score + balance_score

            if score > best_score:
                best_score = score
                best_fold = fi

        fold_assignments[best_fold].append(bid)
        fold_sizes[best_fold] += binfo["total"]
        for cls in unique_classes:
            fold_class_counts[best_fold][cls] += binfo["class_counts"][cls]

    # --- Шаг 3: формирование индексов фолдов ---
    folds_train = {}
    folds_test = {}

    for fi in range(n_folds):
        test_mask = np.isin(block_labels, fold_assignments[fi])
        test_idx = np.where(test_mask)[0].tolist()
        train_idx = np.where(~test_mask)[0].tolist()
        folds_train[fi] = train_idx
        folds_test[fi] = test_idx

    # --- Шаг 4: проверка покрытия классов ---
    all_covered = True
    for fi in range(n_folds):
        fold_classes = set(y[folds_test[fi]])
        missing = set(unique_classes) - fold_classes
        if missing:
            all_covered = False
            print(
                f"    ⚠ Фолд {fi + 1}: отсутствуют классы в тестовой выборке: "
                f"{', '.join(str(c) for c in missing)}"
            )

    if all_covered:
        print("    ✓ Все классы представлены в каждом фолде")
    else:
        print(
            "\n    ⚠ Не все классы представлены во всех фолдах.\n"
            "    Попробуйте уменьшить n_folds или увеличить block_multiplier."
        )

    return folds_train, folds_test, block_labels


# ============================================================================
# ВИЗУАЛИЗАЦИЯ ПРОСТРАНСТВЕННЫХ БЛОКОВ
# ============================================================================

def _plot_spatial_blocks(
    coords: np.ndarray,
    block_labels: np.ndarray,
    fold_assignment_map: np.ndarray,
    class_names: np.ndarray,
    n_folds: int,
    output_dir: str,
):
    """
    Визуализация пространственных блоков и их распределения по фолдам.

    Создаёт 2 графика:
    1) Карта блоков (цвет = блок)
    2) Карта фолдов (цвет = назначенный фолд)
    """
    # --- График 1: Блоки ---
    n_blocks = len(np.unique(block_labels))
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    scatter = ax1.scatter(
        coords[:, 0],
        coords[:, 1],
        c=block_labels,
        cmap="tab20" if n_blocks <= 20 else "nipy_spectral",
        s=3,
        alpha=0.6,
    )
    ax1.set_title(f"Пространственные блоки (KMeans, n={n_blocks})", fontweight="bold")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_aspect("equal")
    plt.colorbar(scatter, ax=ax1, label="Номер блока", shrink=0.8)
    plt.tight_layout()
    png1 = os.path.join(output_dir, "spatial_blocks_map.png")
    plt.savefig(png1, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    print(f"  ✓ Сохранён: {os.path.basename(png1)}")

    # --- График 2: Фолды ---
    cmap_folds = (plt.cm.get_cmap("tab10", n_folds)
                       if hasattr(plt.cm, "get_cmap")
                       else plt.matplotlib.colormaps["tab10"].resampled(n_folds))
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    scatter2 = ax2.scatter(
        coords[:, 0],
        coords[:, 1],
        c=fold_assignment_map,
        cmap=cmap_folds,
        s=3,
        alpha=0.6,
        vmin=0,
        vmax=n_folds - 1,
    )
    ax2.set_title("Распределение образцов по фолдам (Spatial Block CV)", fontweight="bold")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_aspect("equal")
    cbar = plt.colorbar(scatter2, ax=ax2, label="Фолд", shrink=0.8)
    cbar.set_ticks(range(n_folds))
    cbar.set_ticklabels([f"Fold {i + 1}" for i in range(n_folds)])
    plt.tight_layout()
    png2 = os.path.join(output_dir, "spatial_folds_map.png")
    plt.savefig(png2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"  ✓ Сохранён: {os.path.basename(png2)}")

    # --- График 3: Фолды + классы (subplots) ---
    unique_classes = np.unique(class_names)
    n_cls = len(unique_classes)
    ncols = min(4, n_cls)
    nrows = int(np.ceil(n_cls / ncols))
    fig3, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if n_cls == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ci, cls in enumerate(unique_classes):
        ax = axes[ci]
        cls_mask = class_names == cls
        ax.scatter(
            coords[~cls_mask, 0],
            coords[~cls_mask, 1],
            c="lightgrey",
            s=1,
            alpha=0.2,
        )
        sc = ax.scatter(
            coords[cls_mask, 0],
            coords[cls_mask, 1],
            c=fold_assignment_map[cls_mask],
            cmap=cmap_folds,
            s=4,
            alpha=0.7,
            vmin=0,
            vmax=n_folds - 1,
        )
        ax.set_title(f"{cls} (n={cls_mask.sum()})", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)

    # Скрыть пустые subplots
    for ci in range(n_cls, len(axes)):
        axes[ci].set_visible(False)

    fig3.suptitle(
        "Распределение классов по фолдам (Spatial Block CV)", fontweight="bold"
    )
    plt.tight_layout()
    png3 = os.path.join(output_dir, "spatial_folds_by_class.png")
    plt.savefig(png3, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig3)
    print(f"  ✓ Сохранён: {os.path.basename(png3)}")


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================


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


def main():
    print("\n============================================================")
    print(" КЛАССИФИКАЦИЯ С МАШИННЫМ ОБУЧЕНИЕМ")
    print("============================================================\n")

    root_dir = os.getcwd()

    # --- Выбор подпапки из 01_PREPARE_DATA ---
    prev_base_dir = os.path.join(root_dir, "01_PREPARE_DATA")
    DATA_DIR, selected_suffix = _select_prev_subdir(prev_base_dir, "01_PREPARE_DATA")

    # --- Создание подпапки 02_SELECT ---
    base_output_dir = os.path.join(root_dir, "02_SELECT")
    os.makedirs(base_output_dir, exist_ok=True)

    run_stamp = datetime.now().strftime("%d%m%y_%H%M")
    run_folder = f"{run_stamp}{selected_suffix}"
    OUTPUT_DIR = os.path.join(base_output_dir, run_folder)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Данные: {DATA_DIR}")
    print(f"Выход : {OUTPUT_DIR}")

    # ================================================================
    # ШАГ 1: ЗАГРУЗКА ДАННЫХ
    # ================================================================
    print("\n[ШАГ 1] Загрузка обучающих данных...")

    samples_path = os.path.join(DATA_DIR, "samples.gpkg")
    samples_df = _load_samples_safe(DATA_DIR, need_xy=True)
    plots_dir = DATA_DIR if os.access(DATA_DIR, os.W_OK) else OUTPUT_DIR

    # Определяем поле классов
    if "class_name" in samples_df.columns:
        class_col = "class_name"
    elif "abbr2" in samples_df.columns:
        samples_df = samples_df.rename(columns={"abbr2": "class_name"})
        class_col = "class_name"
    else:
        candidates = [
            c
            for c in samples_df.columns
            if c not in ("x", "y", "polygon_id")
            and not pd.api.types.is_numeric_dtype(samples_df[c])
        ]
        if not candidates:
            raise ValueError("Не найдено поле с названиями классов!")
        class_col = candidates[0]
        print(f"  ! Используется поле '{class_col}' как класс")
        samples_df = samples_df.rename(columns={class_col: "class_name"})

    samples_df["class_name"] = samples_df["class_name"].astype(str)
    service_cols = {"class_name", "x", "y", "polygon_id", "n_pixels"}
    predictor_cols = [
        c
        for c in samples_df.columns
        if c not in service_cols and pd.api.types.is_numeric_dtype(samples_df[c])
    ]
    n_predictors = len(predictor_cols)

    class_levels = sorted(samples_df["class_name"].unique())

    print(f"  ✓ Загружено образцов: {len(samples_df)}")
    print(f"  ✓ Количество классов: {len(class_levels)}")
    print(f"  ✓ Классы: {', '.join(class_levels)}")
    print(f"  ✓ Количество предикторов: {n_predictors}\n")

    print("  Распределение образцов по классам:")
    class_counts = samples_df["class_name"].value_counts().sort_index()
    for cls, cnt in class_counts.items():
        print(f"    {cls}: {cnt} образцов")
    print()

    # ================================================================
    # ШАГ 2: ВЫБОР МЕТОДА КРОСС-ВАЛИДАЦИИ
    # ================================================================
    print("============================================================")
    print("[ШАГ 2] Настройка кросс-валидации")
    print("============================================================\n")

    # --- Проверка наличия polygon_id (группы пикселей одного полигона) ---
    has_polygon_id = "polygon_id" in samples_df.columns
    if has_polygon_id:
        n_unique_pids = samples_df["polygon_id"].nunique()
        n_total = len(samples_df)
        # Считаем, что стоит включать групповые опции, если
        # в среднем в группе > 1 пикселя (т.е. вариант B)
        avg_per_group = n_total / max(n_unique_pids, 1)
        group_mode_available = (avg_per_group > 1.05) and (n_unique_pids >= 10)
    else:
        n_unique_pids = 0
        group_mode_available = False

    print("  Доступные методы CV:\n")
    print("  [1] Стратифицированная KFold (классический)")
    print("      Пиксели перемешиваются случайно, классы сбалансированы.")
    print("      НЕ учитывает автокорреляцию внутри полигонов (утечка!).\n")
    print("  [2] Spatial Block CV + стратификация")
    print("      KMeans по координатам → пространственные блоки.")
    print("      Блоки распределяются по фолдам с балансировкой классов.")
    print("      Учитывает пространственную автокорреляцию (но не внутриполигонную).\n")

    if group_mode_available:
        print(
            f"  Обнаружен polygon_id: {n_unique_pids} полигонов, "
            f"~{avg_per_group:.1f} пикселей на полигон.\n"
        )
        print("  [3] Stratified GroupKFold по polygon_id (вариант B, Рекомендуется)")
        print("      Все пиксели одного полигона → в один фолд.")
        print("      Устраняет утечку train/test внутри полигонов.\n")
        print("  [4] Spatial Block CV по polygon_id (вариант B, самый строгий)")
        print("      KMeans по центроидам полигонов → блоки; полигоны целиком в фолды.")
        print("      Учитывает и внутриполигонную, и пространственную автокорреляцию.\n")
        default_choice = "3"
    else:
        if has_polygon_id:
            if n_unique_pids < 10 and avg_per_group > 1.05:
                print(
                    f"  polygon_id обнаружен ({n_unique_pids} полигонов), но для 10-fold GroupCV нужно ≥10 полигонов.\n"
                )
            else:
                print(
                    "  polygon_id обнаружен, но в среднем ≤1 пикселя на полигон\n"
                    "  (это Variant 3 — усреднённые полигоны), групповые методы [3]/[4] не нужны.\n"
                )
        default_choice = "2"

    cv_choice = input(f"Выберите метод CV (Enter = {default_choice}): ").strip()
    if cv_choice == "":
        cv_choice = default_choice
    if cv_choice in ("3", "4") and not group_mode_available:
        print("  ⚠ Групповые методы недоступны — нет polygon_id или он вырожден. "
              f"Фолбек на [{default_choice}].")
        cv_choice = default_choice

    n_folds = 10
    cv_method_name = ""
    block_labels = None  # только для Spatial Block CV

    y = samples_df["class_name"].values
    coords = np.column_stack([samples_df["x"].values, samples_df["y"].values])
    groups = (
        samples_df["polygon_id"].values if group_mode_available else None
    )

    if cv_choice == "1":
        # ---- Стратифицированная KFold ----
        cv_method_name = "stratified_kfold"
        print("\n  Используется СТРАТИФИЦИРОВАННАЯ KFold:")
        print("  - Каждый класс представлен в каждом фолде пропорционально")
        if group_mode_available:
            print(
                "  ⚠ ВНИМАНИЕ: пиксели одного полигона могут попасть и в train, и в test.\n"
                "     Метрики будут ЗАВЫШЕНЫ. Используйте [3] или [4] для честной оценки.\n"
            )
        else:
            print()

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        X_dummy = np.zeros((len(y), 1))

        folds_train = {}
        folds_test = {}
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_dummy, y)):
            folds_train[fold_idx] = train_idx.tolist()
            folds_test[fold_idx] = test_idx.tolist()

    elif cv_choice == "3":
        # ---- Stratified GroupKFold по polygon_id (вариант B) ----
        cv_method_name = "stratified_group_kfold"
        print("\n  Используется Stratified GroupKFold по polygon_id:")
        print("  - Все пиксели одного polygon_id гарантированно в одном фолде")
        print("  - Стратификация по class_name (классы балансируются по фолдам)")
        print("  - Устраняет утечку train/test внутри полигонов (вариант B)\n")

        sgkf = StratifiedGroupKFold(
            n_splits=n_folds, shuffle=True, random_state=42
        )
        X_dummy = np.zeros((len(y), 1))

        folds_train = {}
        folds_test = {}
        for fold_idx, (train_idx, test_idx) in enumerate(
            sgkf.split(X_dummy, y, groups=groups)
        ):
            folds_train[fold_idx] = train_idx.tolist()
            folds_test[fold_idx] = test_idx.tolist()

        # Проверка: ни один polygon_id не должен быть в двух тестовых фолдах
        leak = 0
        for fi in range(n_folds):
            for fj in range(fi + 1, n_folds):
                gi = set(groups[folds_test[fi]])
                gj = set(groups[folds_test[fj]])
                leak += len(gi & gj)
        if leak == 0:
            print("  ✓ Утечка по polygon_id отсутствует.")
        else:
            print(f"  ⚠ Обнаружена утечка: {leak} полигонов в нескольких test-фолдах!")

    else:
        # ---- Spatial Block CV + стратификация ----
        # cv_choice в ("2", "4")
        if cv_choice == "4":
            cv_method_name = "spatial_block_cv_grouped"
            print("\n  Используется SPATIAL BLOCK CV по polygon_id (вариант B):")
            print("  - KMeans по центроидам полигонов (целые полигоны → блоки)")
            print("  - Жадное распределение блоков по фолдам")
            print("  - Учитывает и внутриполигонную, и пространственную автокорреляцию\n")
            use_groups = groups
        else:
            cv_method_name = "spatial_block_cv"
            print("\n  Используется SPATIAL BLOCK CV + стратификация:")
            print("  - KMeans кластеризация по координатам X, Y (пиксели)")
            print("  - Жадное распределение блоков по фолдам с балансировкой классов")
            if group_mode_available:
                print(
                    "  ⚠ Пиксели одного polygon_id могут попасть в разные блоки. "
                    "Используйте [4] для варианта B.\n"
                )
            else:
                print("  - Борьба с пространственной автокорреляцией\n")
            use_groups = None

        # Настройки Spatial Block CV
        block_mult_str = input(
            "  Множитель блоков (n_blocks = n_folds × множитель, Enter = 5): "
        ).strip()
        block_multiplier = int(block_mult_str) if block_mult_str else 5

        print()

        folds_train, folds_test, block_labels = _spatial_block_cv(
            coords=coords,
            y=y,
            n_folds=n_folds,
            block_multiplier=block_multiplier,
            random_state=42,
            groups=use_groups,
        )

        # Проверка утечки polygon_id (только в групповом режиме)
        if use_groups is not None:
            leak = 0
            for fi in range(n_folds):
                for fj in range(fi + 1, n_folds):
                    gi = set(groups[folds_test[fi]])
                    gj = set(groups[folds_test[fj]])
                    leak += len(gi & gj)
            if leak == 0:
                print("  ✓ Утечка по polygon_id отсутствует.")
            else:
                print(
                    f"  ⚠ Обнаружена утечка: {leak} полигонов в нескольких test-фолдах!"
                )

        # Визуализация пространственных блоков
        print("\n  Создание карт пространственных блоков...")

        # Создаём маппинг: sample_idx → fold_idx (для визуализации)
        fold_assignment_map = np.full(len(y), -1, dtype=int)
        for fi in range(n_folds):
            for idx in folds_test[fi]:
                fold_assignment_map[idx] = fi

        _plot_spatial_blocks(
            coords=coords,
            block_labels=block_labels,
            fold_assignment_map=fold_assignment_map,
            class_names=y,
            n_folds=n_folds,
            output_dir=plots_dir,
        )

    # ================================================================
    # ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЯ ПО ФОЛДАМ (общая для обоих методов)
    # ================================================================

    # --- Таблица распределения по фолдам ---
    header = f"  {'Фолд':<10s}"
    for cls in class_levels:
        header += f"  {cls:<10s}"
    header += "  Всего"
    print("\n" + header)
    print("  " + "-" * (10 + 12 * len(class_levels) + 8))

    fold_summary_rows = []
    for fi in range(n_folds):
        test_classes = y[folds_test[fi]]
        row = {"Fold": f"Fold {fi + 1}"}
        line = f"  {'Fold ' + str(fi + 1):<10s}"
        total = 0
        for cls in class_levels:
            cnt = int((test_classes == cls).sum())
            row[cls] = cnt
            total += cnt
            line += f"  {cnt:<10d}"
        row["Total"] = total
        line += f"  {total}"
        print(line)
        fold_summary_rows.append(row)

    fold_summary = pd.DataFrame(fold_summary_rows)

    print("  " + "-" * (10 + 12 * len(class_levels) + 8))
    totals_line = f"  {'ИТОГО':<10s}"
    for cls in class_levels:
        totals_line += f"  {int(fold_summary[cls].sum()):<10d}"
    totals_line += f"  {int(fold_summary['Total'].sum())}"
    print(totals_line)
    print()

    # --- Визуализация фолдов ---
    print("  Создание визуализации фолдов...")

    # Данные для графиков
    fold_long = fold_summary.melt(
        id_vars=["Fold", "Total"],
        value_vars=class_levels,
        var_name="Class",
        value_name="Count",
    )
    fold_long["Fold"] = pd.Categorical(
        fold_long["Fold"],
        categories=[f"Fold {i}" for i in range(1, n_folds + 1)],
        ordered=True,
    )

    # --- График 1: Stacked bar chart ---
    n_cls = len(class_levels)
    cmap = plt.cm.Set2 if n_cls <= 8 else plt.cm.tab20
    colors = [cmap(i / max(n_cls - 1, 1)) for i in range(n_cls)]

    cv_label = (
        "Spatial Block CV + стратификация"
        if cv_method_name == "spatial_block_cv"
        else "Стратифицированная KFold"
    )

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(n_folds)
    fold_labels = [f"Fold {i + 1}" for i in range(n_folds)]

    for ci, cls in enumerate(class_levels):
        vals = [fold_summary.iloc[fi][cls] for fi in range(n_folds)]
        bars = ax1.bar(fold_labels, vals, bottom=bottom, label=cls, color=colors[ci], width=0.7)
        # подписи внутри баров
        for bi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0:
                ax1.text(bi, b + v / 2, str(v), ha="center", va="center",
                         fontsize=7, color="white", fontweight="bold")
        bottom += vals

    ax1.set_title(
        f"{cv_label} — {n_folds}-fold CV",
        fontweight="bold",
    )
    ax1.set_ylabel("Количество образцов (test set)")
    ax1.legend(title="Класс", loc="upper center", ncol=min(n_cls, 6),
               bbox_to_anchor=(0.5, -0.12), fontsize=8)
    fig1.suptitle(
        f"Всего образцов: {len(samples_df)} | Классов: {len(class_levels)} | "
        f"Предикторов: {n_predictors}",
        fontsize=9, color="grey",
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    png1 = os.path.join(plots_dir, "cv_folds_barplot.png")
    plt.savefig(png1, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    print(f"  ✓ Сохранён: {os.path.basename(png1)}")

    # --- График 2: Heatmap ---
    pivot = fold_summary.set_index("Fold")[class_levels]
    fig2, ax2 = plt.subplots(figsize=(max(6, n_cls * 0.9 + 2), max(4, n_folds * 0.45 + 1)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="white",
        ax=ax2,
    )
    ax2.set_title(
        f"Распределение образцов по фолдам и классам ({cv_label})",
        fontweight="bold",
    )
    ax2.set_ylabel("")
    ax2.set_xlabel("Класс")
    plt.tight_layout()

    png2 = os.path.join(plots_dir, "cv_folds_heatmap.png")
    plt.savefig(png2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"  ✓ Сохранён: {os.path.basename(png2)}")

    # CSV
    csv_folds = os.path.join(plots_dir, "cv_folds_summary.csv")
    fold_summary.to_csv(csv_folds, index=False)
    print(f"  ✓ Сохранён: {os.path.basename(csv_folds)}\n")

    print(f"  ✓ Настроена {cv_label} — {n_folds}-fold CV с явными фолдами\n")

    # ================================================================
    # ШАГ 3: ВЫБОР АЛГОРИТМОВ
    # ================================================================
    print("============================================================")
    print(" ВЫБОР АЛГОРИТМОВ ДЛЯ КЛАССИФИКАЦИИ")
    print("============================================================\n")

    # --- Все группы алгоритмов ---
    algo_groups = {
        "TREE-BASED": [
            "Random Forest (sklearn/ranger)",
        ],
        "BOOSTING": [
            "CatBoost (catboost)",
            "LightGBM (lightgbm)",
        ],
        "LOGISTIC REGRESSION": [
            "Ridge (sklearn/glmnet)",
            "Lasso (sklearn/glmnet)",
            "Elastic Net (sklearn/glmnet)",
        ],
        "DISCRIMINANT ANALYSIS": [
            "LDA (sklearn)",
            "QDA (sklearn)",
            "RDA (sklearn, LDA + shrinkage)",
        ],
        "INSTANCE-BASED": [
            "KNN (sklearn)",
        ],
        "SVM": [
            "SVM RBF (sklearn)",
        ],
    }

    optimal_set = {
        "TREE-BASED": ["Random Forest (sklearn/ranger)"],
        "BOOSTING": ["CatBoost (catboost)"],
        "DISCRIMINANT ANALYSIS": ["RDA (sklearn, LDA + shrinkage)"],
    }

    def select_from_group(group_name, algorithms, already_selected=None):
        print(f"\n--- {group_name} ---")
        for i, alg in enumerate(algorithms, 1):
            mark = " ✓ уже выбран" if already_selected and alg in already_selected else ""
            print(f"  {i}. {alg}{mark}")
        print("  0. Пропустить группу")
        choice = input("Ваш выбор (номера через запятую, 0 = пропуск): ").strip()
        if choice in ("0", ""):
            return None
        try:
            indices = [int(x.strip()) for x in choice.split(",")]
            indices = [i for i in indices if 1 <= i <= len(algorithms)]
        except ValueError:
            print("  Некорректный ввод, группа пропущена.")
            return None
        if not indices:
            print("  Некорректный ввод, группа пропущена.")
            return None
        return [algorithms[i - 1] for i in indices]

    # --- Главное меню ---
    print("------------------------------------------------------------")
    print("Режим выбора алгоритмов:\n")
    print("  000 - ВСЕ алгоритмы из всех групп (полный перебор)")
    print("  00  - ОПТИМАЛЬНЫЙ НАБОР: RF + CatBoost + RDA")
    print("        (затем можно дополнить из каждой группы)")
    print("  1   - Ручной выбор по каждой группе")
    print("------------------------------------------------------------")
    mode = input("Ваш выбор (Enter = 00): ").strip()
    if mode == "":
        mode = "00"

    selected_algorithms = {}

    if mode == "000":
        selected_algorithms = {k: list(v) for k, v in algo_groups.items()}
        print("\n✓ Выбраны ВСЕ алгоритмы:")
        idx = 1
        for gname, algs in selected_algorithms.items():
            print(f"\n  [{gname}]")
            for a in algs:
                print(f"    {idx}. {a}")
                idx += 1
        print()

    elif mode == "00":
        selected_algorithms = {k: list(v) for k, v in optimal_set.items()}
        print("\n✓ ОПТИМАЛЬНЫЙ НАБОР:")
        idx = 1
        for gname, algs in selected_algorithms.items():
            for a in algs:
                print(f"  {idx}. {a} [{gname}]")
                idx += 1

        add_more = input("\nДобавить алгоритмы из отдельных групп? (Enter = нет, y = да): ").strip().lower()
        if add_more in ("y", "yes", "д", "да"):
            for gname, algs in algo_groups.items():
                already = selected_algorithms.get(gname, [])
                extra = select_from_group(gname, algs, already_selected=already)
                if extra:
                    current = selected_algorithms.get(gname, [])
                    selected_algorithms[gname] = list(dict.fromkeys(current + extra))
        print()

    else:
        for gname, algs in algo_groups.items():
            chosen = select_from_group(gname, algs)
            if chosen:
                selected_algorithms[gname] = chosen
        print()

    if not selected_algorithms:
        raise ValueError("Не выбран ни один алгоритм! Перезапустите скрипт.")

    # --- Итоговый список ---
    all_selected = []
    for algs in selected_algorithms.values():
        all_selected.extend(algs)

    print("============================================================")
    print(f" ИТОГО ВЫБРАНО АЛГОРИТМОВ: {len(all_selected)}")
    print("============================================================\n")
    for i, a in enumerate(all_selected, 1):
        print(f"  {i}. {a}")
    print()

    # ================================================================
    # ШАГ 4: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # ================================================================

    # Маппинг названий алгоритмов → sklearn/catboost ключи
    algo_key_map = {
        "Random Forest (sklearn/ranger)": "rf",
        "CatBoost (catboost)": "catboost",
        "Ridge (sklearn/glmnet)": "ridge",
        "Lasso (sklearn/glmnet)": "lasso",
        "Elastic Net (sklearn/glmnet)": "elasticnet",
        "LDA (sklearn)": "lda",
        "QDA (sklearn)": "qda",
        "RDA (sklearn, LDA + shrinkage)": "rda",
        "KNN (sklearn)": "knn",
        "SVM RBF (sklearn)": "svm",
        "LightGBM (lightgbm)": "lgbm",
    }

    selected_keys = [algo_key_map.get(a, a) for a in all_selected]

    with open(os.path.join(OUTPUT_DIR, "selected_algorithms.pkl"), "wb") as f:
        pickle.dump(
            {
                "selected_algorithms": selected_algorithms,
                "selected_keys": selected_keys,
                "all_selected_names": all_selected,
            },
            f,
        )

    cv_config = {
        "folds_train": folds_train,
        "folds_test": folds_test,
        "cv_method": cv_method_name,
    }
    if block_labels is not None:
        cv_config["block_labels"] = block_labels.tolist()
    if groups is not None:
        cv_config["groups"] = groups.tolist()  # polygon_id для каждой строки

    with open(os.path.join(OUTPUT_DIR, "cv_folds.pkl"), "wb") as f:
        pickle.dump(cv_config, f)

    with open(os.path.join(OUTPUT_DIR, "data_info.pkl"), "wb") as f:
        pickle.dump(
            {
                "predictor_cols": predictor_cols,
                "n_predictors": n_predictors,
                "class_levels": class_levels,
                "cv_method": cv_method_name,
                "n_folds": n_folds,
                "data_dir": DATA_DIR,
                "samples_path": samples_path,
            },
            f,
        )

    print("✓ Файлы сохранены в 02_SELECT:")
    print("  1. selected_algorithms.pkl")
    print("  2. cv_folds.pkl (индексы фолдов)")
    print("  3. data_info.pkl\n")

    print("✓ Визуализация фолдов в 01_PREPARE_DATA:")
    print("  - cv_folds_barplot.png")
    print("  - cv_folds_heatmap.png")
    print("  - cv_folds_summary.csv")
    if cv_method_name == "spatial_block_cv":
        print("  - spatial_blocks_map.png")
        print("  - spatial_folds_map.png")
        print("  - spatial_folds_by_class.png")
    print()

    print("Настройки кросс-валидации:")
    print(f"  ✓ Метод: {cv_label} — {n_folds}-fold CV (явные фолды)")
    print(f"  ✓ Количество фолдов: {n_folds}")
    if cv_method_name == "spatial_block_cv":
        print(f"  ✓ Пространственных блоков: {len(np.unique(block_labels))}")
    print()

    print("Следующий шаг: запустите 03_tune_models.py")
    print("============================================================\n")

    return {
        "selected_algorithms": selected_algorithms,
        "selected_keys": selected_keys,
        "predictor_cols": predictor_cols,
        "n_folds": n_folds,
        "cv_method": cv_method_name,
        "output_dir": OUTPUT_DIR,
        "data_dir": DATA_DIR,
    }


# ================================================================
# ЗАПУСК
# ================================================================
if __name__ == "__main__":
    main()
