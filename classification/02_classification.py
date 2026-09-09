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
from sklearn.model_selection import StratifiedKFold
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

    # Ограничиваем число блоков (не больше числа образцов)
    n_blocks = min(n_blocks, n_samples)

    print(f"    KMeans кластеризация: {n_blocks} пространственных блоков...")

    # --- Шаг 1: KMeans по координатам ---
    kmeans = KMeans(
        n_clusters=n_blocks, random_state=random_state, n_init=10, max_iter=300
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
    cmap_folds = plt.cm.get_cmap("tab10", n_folds)
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
    if not os.path.exists(samples_path):
        raise FileNotFoundError(
            f"Файл samples.gpkg не найден в {DATA_DIR}!\n"
            f"Сначала запустите 01_prepare_data.py"
        )

    samples_gdf = gpd.read_file(samples_path, layer="samples")
    samples_df = pd.DataFrame(samples_gdf.drop(columns="geometry"))

    # Извлекаем координаты из геометрии
    coords_x = samples_gdf.geometry.x.values
    coords_y = samples_gdf.geometry.y.values
    samples_df["x"] = coords_x
    samples_df["y"] = coords_y

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
    service_cols = {"class_name", "x", "y", "polygon_id"}
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

    print("  Доступные методы CV:\n")
    print("  [1] Стратифицированная KFold (классический)")
    print("      Пиксели перемешиваются случайно, классы сбалансированы.")
    print("      Не учитывает пространственную автокорреляцию.\n")
    print("  [2] Spatial Block CV + стратификация (Рекомендуется)")
    print("      KMeans по координатам → пространственные блоки.")
    print("      Блоки распределяются по фолдам с балансировкой классов.")
    print("      Учитывает пространственную автокорреляцию.\n")

    cv_choice = input("Выберите метод CV (Enter = 2): ").strip()
    if cv_choice == "":
        cv_choice = "2"

    n_folds = 10
    cv_method_name = ""
    block_labels = None  # только для Spatial Block CV

    y = samples_df["class_name"].values
    coords = np.column_stack([samples_df["x"].values, samples_df["y"].values])

    if cv_choice == "1":
        # ---- Стратифицированная KFold ----
        cv_method_name = "stratified_kfold"
        print("\n  Используется СТРАТИФИЦИРОВАННАЯ кросс-валидация:")
        print("  - Каждый класс представлен в каждом фолде пропорционально")
        print("  - Более стабильная оценка качества моделей\n")

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        X_dummy = np.zeros((len(y), 1))

        folds_train = {}
        folds_test = {}
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_dummy, y)):
            folds_train[fold_idx] = train_idx.tolist()
            folds_test[fold_idx] = test_idx.tolist()

    else:
        # ---- Spatial Block CV + стратификация ----
        cv_method_name = "spatial_block_cv"
        print("\n  Используется SPATIAL BLOCK CV + стратификация:")
        print("  - KMeans кластеризация по координатам X, Y")
        print("  - Жадное распределение блоков по фолдам с балансировкой классов")
        print("  - Борьба с пространственной автокорреляцией\n")

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
            output_dir=DATA_DIR,
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

    png1 = os.path.join(DATA_DIR, "cv_folds_barplot.png")
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

    png2 = os.path.join(DATA_DIR, "cv_folds_heatmap.png")
    plt.savefig(png2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"  ✓ Сохранён: {os.path.basename(png2)}")

    # CSV
    csv_folds = os.path.join(DATA_DIR, "cv_folds_summary.csv")
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
    }
    if block_labels is not None:
        cv_config["block_labels"] = block_labels.tolist()

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
