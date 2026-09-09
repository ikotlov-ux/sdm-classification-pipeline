# ============================================================================
# 02s_select_sdm_v2.py
# СКРИПТ 02s: ВЫБОР АЛГОРИТМОВ И НАСТРОЙКА CV ДЛЯ SDM
#
# Аналог 02_classification.py, адаптированный под presence-background.
#
# ИЗМЕНЕНИЯ В v2:
#   - Теперь скрипт читает исходные данные НЕ только из 01_PREPARE_SDM,
#     но И из 01_PREPARE_DATA (вывод пайплайна классификации). При
#     запуске предлагается выбрать источник (01_PREPARE_SDM /
#     01_PREPARE_DATA), затем конкретный запуск.
#   - Если в samples.gpkg НЕТ колонки 'pa' (данные из 01_PREPARE_DATA —
#     только точки присутствия с class_name): скрипт считает все
#     имеющиеся точки присутствием (pa=1) и ГЕНЕРИРУЕТ background
#     (pa=0) через bias-grid (Gaussian KDE плотности точек), как в
#     01s_prepare_sdm.py. Растр-композит берётся из raster_info.pkl
#     выбранного запуска (или ищется Composite_filtered.tif / Composite.tif).
#     Извлечённые предикторы background берутся из того же растра.
#     Если 'pa' уже есть (01_PREPARE_SDM) — данные используются как есть.
#   - Сгенерированные в этом случае background и объединённые samples
#     сохраняются в папку вывода 02_SELECT_SDM (samples_with_bg.gpkg/csv),
#     чтобы 03s мог их использовать.
#
# Отличия от классификации:
#   - Целевая переменная бинарная: pa (1 / 0), а не мультикласс class_name.
#   - Стратификация CV по pa (presence / background).
#   - КРОСС-ВАЛИДАЦИЯ ТОЛЬКО Spatial Block CV (без обычной KFold): для
#     presence-background данных обычная KFold систематически завышает AUC
#     из-за пространственной автокорреляции между близкими точками. Block CV
#     даёт честную оценку переноса модели в пространстве.
#   - Набор алгоритмов: MaxEnt (maxent.jar), LightGBM, GLM (логистическая
#     регрессия). Ensemble настраивается в 03s.
#
# Сохраняет конфигурацию (фолды, алгоритмы, предикторы) для 03s_tune_sdm.py.
# ============================================================================

import os
import re
import glob
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol, xy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

try:
    from scipy.stats import gaussian_kde
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================================
# ВЫБОР ПОДПАПКИ
# ============================================================================

def _select_prev_subdir(base_dir, label):
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Папка {os.path.basename(base_dir)} не найдена!")
    subdirs = sorted(
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    )
    if not subdirs:
        raise FileNotFoundError(f"В {os.path.basename(base_dir)} нет результатов.")
    print(f"\nДоступные запуски в {os.path.basename(base_dir)}:\n")
    for i, sd in enumerate(subdirs, 1):
        print(f" [{i}] {os.path.basename(sd)}")
    raw = input(f"\nВыберите номер запуска из {os.path.basename(base_dir)}: ").strip() or "1"
    idx = int(raw)
    selected = subdirs[idx - 1]
    m = re.search(r"(_v\d{3})$", os.path.basename(selected))
    suffix = m.group(1) if m else "_v001"
    print(f"Выбрана папка: {os.path.basename(selected)}, суффикс: {suffix}\n")
    return selected, suffix


def _select_source_base(root_dir):
    """Выбор источника данных: 01_PREPARE_SDM или 01_PREPARE_DATA.

    Возвращает (base_dir, source_tag), где source_tag ∈ {'sdm','data'}.
    Показываются только те папки, которые реально существуют.
    """
    candidates = [
        ("01_PREPARE_SDM", "sdm", "данные SDM с готовым pa + background"),
        ("01_PREPARE_DATA", "data", "точки присутствия классификации (background будет сгенерирован)"),
    ]
    avail = [
        (name, tag, desc)
        for (name, tag, desc) in candidates
        if os.path.isdir(os.path.join(root_dir, name))
    ]
    if not avail:
        raise FileNotFoundError(
            "Не найдена ни одна папка-источник: 01_PREPARE_SDM или 01_PREPARE_DATA. "
            "Сначала запустите 01s_prepare_sdm.py или 01_prepare_data.py."
        )
    if len(avail) == 1:
        name, tag, _ = avail[0]
        print(f"Источник данных: {name}\n")
        return os.path.join(root_dir, name), tag
    print("Выберите источник данных:\n")
    for i, (name, tag, desc) in enumerate(avail, 1):
        print(f" [{i}] {name}  — {desc}")
    raw = input("\nВаш выбор (Enter = 1): ").strip() or "1"
    name, tag, _ = avail[int(raw) - 1]
    print(f"Выбран источник: {name}\n")
    return os.path.join(root_dir, name), tag


# ============================================================================
# BIAS-GRID И ГЕНЕРАЦИЯ BACKGROUND (при источнике 01_PREPARE_DATA)
#
# Логика перенесена из 01s_prepare_sdm.py, чтобы 02s мог сам добавить
# background-точки (pa=0) к точкам присутствия (pa=1), когда на входе
# только точки присутствия (вывод 01_PREPARE_DATA).
# ============================================================================

def _find_raster(data_dir, root_dir):
    """Поиск растра-композита предикторов.

    Приоритет:
      1) raster_info.pkl в data_dir (ключ 'raster_path');
      2) Composite_filtered.tif / Composite.tif в data_dir, INPUT, root;
      3) любой *.tif в data_dir / INPUT / root (интерактивный выбор).
    """
    info_pkl = os.path.join(data_dir, "raster_info.pkl")
    if os.path.exists(info_pkl):
        try:
            with open(info_pkl, "rb") as f:
                rinfo = pickle.load(f)
            rp = rinfo.get("raster_path")
            if rp and os.path.exists(rp):
                print(f"  ✓ Растр из raster_info.pkl: {os.path.basename(rp)}")
                return rp
            rp2 = os.path.join(data_dir, os.path.basename(rp)) if rp else None
            if rp2 and os.path.exists(rp2):
                print(f"  ✓ Растр рядом с raster_info.pkl: {os.path.basename(rp2)}")
                return rp2
        except Exception:
            pass

    search_dirs = [data_dir, os.path.join(root_dir, "INPUT"), root_dir]
    for pref in ("Composite_filtered.tif", "Composite.tif"):
        for d in search_dirs:
            cand = os.path.join(d, pref)
            if os.path.exists(cand):
                print(f"  ✓ Найден растр: {pref}")
                return cand

    tifs = []
    for d in search_dirs:
        if os.path.isdir(d):
            tifs.extend(glob.glob(os.path.join(d, "*.tif")))
    tifs = sorted(set(tifs))
    if not tifs:
        raise FileNotFoundError(
            "Растр-композит предикторов не найден для генерации background "
            "(нет raster_info.pkl, Composite_filtered.tif или Composite.tif)."
        )
    print("\n  Растр для генерации background не определён однозначно:")
    for i, tf in enumerate(tifs, 1):
        print(f"   [{i}] {os.path.basename(tf)}")
    raw = input("Выберите номер растра (Enter = 1): ").strip() or "1"
    return tifs[int(raw) - 1]


def _build_bias_grid(pres_rc, height, width, bandwidth_px=None):
    """BIAS-GRID через Gaussian KDE плотности точек присутствия.

    Идентично 01s_prepare_sdm.py: нормирован к [0,1], вне валидной
    области обнуляется позже маской.
    """
    xy_pts = np.vstack([pres_rc[:, 1], pres_rc[:, 0]]).astype(float)  # (x=col, y=row)
    if bandwidth_px is not None and bandwidth_px > 0:
        std_x = xy_pts[0].std() if xy_pts[0].std() > 0 else 1.0
        std_y = xy_pts[1].std() if xy_pts[1].std() > 0 else 1.0
        std_mean = (std_x + std_y) / 2.0
        kde = gaussian_kde(xy_pts, bw_method=bandwidth_px / std_mean)
    else:
        kde = gaussian_kde(xy_pts)  # метод Скотта

    step = max(1, int(round(max(height, width) / 600)))
    rows_grid = np.arange(0, height, step)
    cols_grid = np.arange(0, width, step)
    cc, rr = np.meshgrid(cols_grid, rows_grid)
    coords_eval = np.vstack([cc.ravel(), rr.ravel()])
    dens = kde(coords_eval).reshape(len(rows_grid), len(cols_grid))

    row_idx = np.clip((np.arange(height) // step), 0, len(rows_grid) - 1)
    col_idx = np.clip((np.arange(width) // step), 0, len(cols_grid) - 1)
    full = dens[np.ix_(row_idx, col_idx)].astype(np.float32)

    fmin, fmax = float(full.min()), float(full.max())
    if fmax > fmin:
        full = (full - fmin) / (fmax - fmin)
    else:
        full[:] = 1.0
    return full


def _generate_background(presence_df, predictor_cols, raster_path, output_dir, ask=True):
    """Генерация background-точек (pa=0) для точек присутствия.

    presence_df — DataFrame точек присутствия с колонками x, y (CRS растра)
    и предикторами predictor_cols.

    Возвращает объединённый DataFrame (presence + background) с колонкой pa,
    x, y и предикторами; порядок предикторов — как в растре.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "Для генерации background нужен scipy (gaussian_kde). "
            "Установите: pip install scipy"
        )

    with rasterio.open(raster_path) as src:
        n_layers = src.count
        width, height = src.width, src.height
        raster_crs = src.crs
        transform = src.transform
        src_nodata = src.nodata
        src_dtype = src.dtypes[0]
        band_names = [
            src.descriptions[i] if src.descriptions[i] else f"band{i + 1}"
            for i in range(n_layers)
        ]
    print(f"  ✓ Растр: {n_layers} слоёв, {width} x {height}, CRS {raster_crs}")

    # row/col точек присутствия в сетке растра
    xs = presence_df["x"].values.astype(float)
    ys = presence_df["y"].values.astype(float)
    rows, cols = rowcol(transform, xs, ys)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    n_out = int((~in_bounds).sum())
    if n_out:
        print(f"  ! Точек присутствия вне границ растра: {n_out}")
    pres_rc = np.column_stack([rows[in_bounds], cols[in_bounds]])

    # Валидная маска (все предикторы не nodata)
    is_float = np.issubdtype(np.dtype(src_dtype), np.floating)
    valid_mask = np.ones((height, width), dtype=bool)
    with rasterio.open(raster_path) as src:
        for b in range(1, n_layers + 1):
            arr = src.read(b)
            band_valid = np.isfinite(arr) if is_float else np.ones_like(arr, dtype=bool)
            if src_nodata is not None and not (
                isinstance(src_nodata, float) and np.isnan(src_nodata)
            ):
                band_valid &= (arr != src_nodata)
            valid_mask &= band_valid
    n_valid = int(valid_mask.sum())
    print(f"  ✓ Валидных пикселей: {n_valid} из {height * width}")

    # BIAS-GRID
    if ask:
        bw_raw = input("  Ширина ядра KDE в пикселях (Enter = авто): ").strip()
    else:
        bw_raw = ""
    bandwidth_px = float(bw_raw) if bw_raw else None
    bias = _build_bias_grid(pres_rc, height, width, bandwidth_px=bandwidth_px)
    bias = np.where(valid_mask, bias, 0.0).astype(np.float32)

    # Количество background
    if ask:
        nbg_raw = input("  Сколько background-точек (Enter = 10000): ").strip()
        use_bias = (input(
            "  Взвешивать background по bias-grid? (Enter = да, N = равномерно): "
        ).strip().upper() != "N")
    else:
        nbg_raw = ""
        use_bias = True
    n_bg = int(nbg_raw) if nbg_raw else 10000

    valid_flat = np.flatnonzero(valid_mask.ravel())
    n_bg = min(n_bg, len(valid_flat))
    rng = np.random.default_rng(42)
    if use_bias:
        w = bias.ravel()[valid_flat].astype(np.float64)
        probs = (w / w.sum()) if w.sum() > 0 else None
        if probs is None:
            print("  ! Bias пустой, переключаюсь на равномерный семплинг.")
    else:
        probs = None

    bg_choice = rng.choice(valid_flat, size=n_bg, replace=False, p=probs)
    bg_rows, bg_cols = np.unravel_index(bg_choice, (height, width))
    bg_x, bg_y = xy(transform, bg_rows, bg_cols)
    bg_x = np.asarray(bg_x)
    bg_y = np.asarray(bg_y)
    print(f"  ✓ Сгенерировано background: {n_bg} "
          f"({'взвешенно по bias' if probs is not None else 'равномерно'})")

    # Извлекаем предикторы background из растра
    feat_bg = np.full((n_bg, n_layers), np.nan, dtype=np.float64)
    with rasterio.open(raster_path) as src:
        for b in range(1, n_layers + 1):
            arr = src.read(b).astype(np.float64)
            vals = arr[bg_rows, bg_cols]
            if src_nodata is not None and not (
                isinstance(src_nodata, float) and np.isnan(src_nodata)
            ):
                vals = np.where(vals == src_nodata, np.nan, vals)
            feat_bg[:, b - 1] = vals
    bg_df = pd.DataFrame(feat_bg, columns=band_names)
    bg_df["pa"] = 0
    bg_df["x"] = bg_x
    bg_df["y"] = bg_y
    bg_df = bg_df.dropna(subset=band_names).reset_index(drop=True)

    # Собираем presence в том же порядке предикторов (по именам растра)
    missing = [c for c in band_names if c not in presence_df.columns]
    if missing:
        raise KeyError(
            "В точках присутствия отсутствуют предикторы растра: "
            f"{missing}. Проверьте, что samples и растр согласованы."
        )
    pres_out = presence_df.copy()
    pres_out["pa"] = 1
    pres_cols = band_names + ["pa", "x", "y"]
    pres_out = pres_out[pres_cols]
    bg_out = bg_df[pres_cols]

    combined = pd.concat([pres_out, bg_out], ignore_index=True)
    n_pres = int((combined["pa"] == 1).sum())
    n_back = int((combined["pa"] == 0).sum())
    print(f"  ✓ Итог: presence={n_pres}, background={n_back}")

    # Сохраняем объединённые samples и bias в папку вывода 02s
    try:
        geom = gpd.points_from_xy(combined["x"], combined["y"], crs=raster_crs)
        comb_gdf = gpd.GeoDataFrame(
            combined.drop(columns=["x", "y"]), geometry=geom, crs=raster_crs
        )
        gpkg_out = os.path.join(output_dir, "samples_with_bg.gpkg")
        if os.path.exists(gpkg_out):
            os.remove(gpkg_out)
        comb_gdf.to_file(gpkg_out, layer="samples", driver="GPKG")
        combined.to_csv(os.path.join(output_dir, "samples_with_bg.csv"), index=False)
        print(f"  ✓ Сохранено: samples_with_bg.gpkg / .csv (presence + background)")
    except Exception as e:
        print(f"  ! Не удалось сохранить samples_with_bg: {e}")

    bias_profile = {
        "driver": "GTiff", "height": height, "width": width, "count": 1,
        "dtype": "float32", "crs": raster_crs, "transform": transform,
        "compress": "lzw", "nodata": 0.0,
    }
    try:
        bias_path = os.path.join(output_dir, "bias.tif")
        with rasterio.open(bias_path, "w", **bias_profile) as dst:
            dst.write(bias, 1)
        print(f"  ✓ Сохранён bias-grid: bias.tif")
    except Exception as e:
        print(f"  ! Не удалось сохранить bias.tif: {e}")

    return combined, band_names, raster_path


# ============================================================================
# SPATIAL BLOCK CV (бинарная стратификация по pa)
# ============================================================================

def _spatial_block_cv(coords, y, n_folds=5, block_multiplier=5, random_state=42):
    """
    Spatial Block CV: KMeans-блоки по координатам, жадное распределение
    блоков по фолдам с балансировкой presence/background.
    y — бинарный массив pa (1/0).
    """
    n_samples = len(y)
    n_blocks = min(n_folds * block_multiplier, n_samples)
    unique_classes = np.unique(y)

    print(f"    KMeans: {n_blocks} пространственных блоков...")
    kmeans = KMeans(n_clusters=n_blocks, random_state=random_state, n_init=10)
    block_labels = kmeans.fit_predict(coords)

    block_ids = np.unique(block_labels)
    block_info = {}
    for bid in block_ids:
        mask = block_labels == bid
        cc = {cls: int((y[mask] == cls).sum()) for cls in unique_classes}
        block_info[bid] = {
            "total": int(mask.sum()),
            "class_counts": cc,
            "classes_present": {c for c, n in cc.items() if n > 0},
        }

    sorted_blocks = sorted(
        block_ids,
        key=lambda b: (-len(block_info[b]["classes_present"]), -block_info[b]["total"]),
    )
    fold_assignments = {fi: [] for fi in range(n_folds)}
    fold_sizes = np.zeros(n_folds, dtype=int)
    fold_class_counts = {fi: {c: 0 for c in unique_classes} for fi in range(n_folds)}

    for bid in sorted_blocks:
        binfo = block_info[bid]
        best_fold, best_score = None, -np.inf
        for fi in range(n_folds):
            size_score = -fold_sizes[fi]
            new_classes = sum(
                1 for c in binfo["classes_present"] if fold_class_counts[fi][c] == 0
            )
            class_score = new_classes * 1000
            balance_score = -sum(
                fold_class_counts[fi][c] * binfo["class_counts"][c]
                for c in binfo["classes_present"]
            )
            score = size_score + class_score + balance_score
            if score > best_score:
                best_score, best_fold = score, fi
        fold_assignments[best_fold].append(bid)
        fold_sizes[best_fold] += binfo["total"]
        for c in unique_classes:
            fold_class_counts[best_fold][c] += binfo["class_counts"][c]

    folds_train, folds_test = {}, {}
    for fi in range(n_folds):
        test_mask = np.isin(block_labels, fold_assignments[fi])
        folds_test[fi] = np.where(test_mask)[0].tolist()
        folds_train[fi] = np.where(~test_mask)[0].tolist()

    for fi in range(n_folds):
        missing = set(unique_classes) - set(y[folds_test[fi]])
        if missing:
            print(f"    ⚠ Фолд {fi+1}: нет presence/background в тесте: {missing}")
    return folds_train, folds_test, block_labels


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n============================================================")
    print(" SDM — ВЫБОР АЛГОРИТМОВ И НАСТРОЙКА CV")
    print("============================================================\n")

    root_dir = os.getcwd()
    # Источник данных: 01_PREPARE_SDM (готовый pa) или 01_PREPARE_DATA
    # (точки присутствия классификации, background будет сгенерирован).
    prev_base, source_tag = _select_source_base(root_dir)
    DATA_DIR, suffix = _select_prev_subdir(prev_base, os.path.basename(prev_base))

    base_out = os.path.join(root_dir, "02_SELECT_SDM")
    os.makedirs(base_out, exist_ok=True)
    run_stamp = datetime.now().strftime("%d%m%y_%H%M")
    OUTPUT_DIR = os.path.join(base_out, f"{run_stamp}{suffix}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Данные: {DATA_DIR}")
    print(f"Выход : {OUTPUT_DIR}\n")

    # --- Загрузка данных ---
    print("[ШАГ 1] Загрузка данных SDM...")
    samples_path = os.path.join(DATA_DIR, "samples.gpkg")
    if not os.path.exists(samples_path):
        raise FileNotFoundError(
            f"samples.gpkg не найден в {DATA_DIR}. "
            "Сначала 01s_prepare_sdm.py или 01_prepare_data.py"
        )
    gdf = gpd.read_file(samples_path, layer="samples")
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    df["x"] = gdf.geometry.x.values
    df["y"] = gdf.geometry.y.values

    # Служебные колонки, никогда не предикторы
    service_cols = {"pa", "class_name", "x", "y", "polygon_id", "n_pixels", "ID"}

    bg_raster_path = None
    if "pa" not in df.columns:
        # Данные из 01_PREPARE_DATA (только точки присутствия, без pa).
        # Считаем все точки присутствием (pa=1) и генерируем background (pa=0).
        print("\n  Колонка 'pa' отсутствует — источник содержит только точки")
        print("  присутствия. Будут сгенерированы background-точки (pa=0)")
        print("  через bias-grid (Gaussian KDE), как в 01s_prepare_sdm.py.\n")
        if "class_name" in df.columns:
            uniq = df["class_name"].dropna().unique()
            print(f"  Точки присутствия (class_name): {', '.join(str(c) for c in uniq)}")
            print(f"  Все {len(df)} точек будут трактоваться как presence одного вида.\n")

        bg_raster_path = _find_raster(DATA_DIR, root_dir)
        df, predictor_cols, bg_raster_path = _generate_background(
            df, None, bg_raster_path, OUTPUT_DIR, ask=True
        )
        # Обновляем samples_path на сгенерированный файл (для 03s)
        gen_gpkg = os.path.join(OUTPUT_DIR, "samples_with_bg.gpkg")
        if os.path.exists(gen_gpkg):
            samples_path = gen_gpkg
    else:
        predictor_cols = [
            c for c in df.columns
            if c not in service_cols and pd.api.types.is_numeric_dtype(df[c])
        ]
    df["pa"] = df["pa"].astype(int)

    n_pred = len(predictor_cols)
    n_pres = int((df["pa"] == 1).sum())
    n_back = int((df["pa"] == 0).sum())

    print(f"\n  ✓ Точек: {len(df)} (presence={n_pres}, background={n_back})")
    print(f"  ✓ Предикторов: {n_pred}")
    print(f"  ✓ Соотношение presence:background = 1:{n_back/max(n_pres,1):.1f}\n")

    if n_pres < 10:
        print("  ⚠ Менее 10 точек присутствия — модель может быть ненадёжной.\n")
    if n_back == 0:
        raise ValueError(
            "Нет background-точек (pa=0). Для SDM нужен background. "
            "Проверьте исходные данные или генерацию background."
        )

    # ================================================================
    # ШАГ 2: КРОСС-ВАЛИДАЦИЯ (ТОЛЬКО Spatial Block CV)
    # ================================================================
    print("============================================================")
    print("[ШАГ 2] Кросс-валидация: Spatial Block CV")
    print("============================================================\n")
    print("  Для SDM используется ТОЛЬКО Spatial Block CV:")
    print("  - KMeans-блоки по координатам X, Y → пространственные блоки")
    print("  - Жадное распределение блоков по фолдам с балансировкой pa")
    print("  - Честная оценка переноса модели в пространстве; обычная KFold")
    print("    систематически завышает AUC из-за автокорреляции близких точек.\n")

    cv_method = "spatial_block_cv"

    # n_folds ограничиваем по числу presence
    n_folds = min(5, max(2, n_pres))
    if n_folds < 5:
        print(f"  ⚠ n_folds понижен до {n_folds} (мало presence).\n")

    y = df["pa"].values
    coords = np.column_stack([df["x"].values, df["y"].values])

    bm = input("  Множитель блоков (n_blocks = n_folds × множитель, Enter = 5): ").strip()
    block_multiplier = int(bm) if bm else 5
    folds_train, folds_test, block_labels = _spatial_block_cv(
        coords, y, n_folds=n_folds, block_multiplier=block_multiplier
    )

    cv_label = "Spatial Block CV"

    # --- сводка по фолдам ---
    print(f"\n  {'Фолд':<8s}{'presence':>10s}{'background':>12s}{'всего':>8s}")
    print("  " + "-" * 38)
    rows_sum = []
    for fi in range(n_folds):
        yte = y[folds_test[fi]]
        p = int((yte == 1).sum())
        b = int((yte == 0).sum())
        print(f"  Fold {fi+1:<3d}{p:>10d}{b:>12d}{p+b:>8d}")
        rows_sum.append({"Fold": f"Fold {fi+1}", "presence": p, "background": b, "Total": p + b})
    fold_summary = pd.DataFrame(rows_sum)
    fold_summary.to_csv(os.path.join(OUTPUT_DIR, "cv_folds_summary_sdm.csv"), index=False)

    # --- визуализация фолдов в пространстве ---
    fold_map = np.full(len(y), -1, dtype=int)
    for fi in range(n_folds):
        for idx in folds_test[fi]:
            fold_map[idx] = fi
    fig, ax = plt.subplots(figsize=(10, 8))
    # фон — background, поверх — presence
    bg = y == 0
    ax.scatter(coords[bg, 0], coords[bg, 1], c="lightgrey", s=2, alpha=0.3, label="background")
    sc = ax.scatter(
        coords[~bg, 0], coords[~bg, 1], c=fold_map[~bg],
        cmap=plt.cm.get_cmap("tab10", n_folds), s=14, edgecolor="k", linewidth=0.2,
        vmin=0, vmax=n_folds - 1,
    )
    ax.set_title("Spatial Block CV — фолды точек присутствия", fontweight="bold")
    ax.set_aspect("equal")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, label="Фолд")
    cbar.set_ticks(range(n_folds))
    cbar.set_ticklabels([f"Fold {i+1}" for i in range(n_folds)])
    plt.tight_layout()
    png = os.path.join(OUTPUT_DIR, "spatial_folds_map_sdm.png")
    plt.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  ✓ Карта фолдов: {os.path.basename(png)}")

    # ================================================================
    # ШАГ 3: ВЫБОР АЛГОРИТМОВ
    # ================================================================
    print("\n============================================================")
    print(" ВЫБОР АЛГОРИТМОВ SDM")
    print("============================================================\n")

    algos = {
        "1": ("maxent", "MaxEnt (maxent.jar, эталонная реализация)"),
        "2": ("lightgbm", "LightGBM / GBM (gradient boosting)"),
        "3": ("glm", "GLM (логистическая регрессия, sklearn)"),
    }
    print("  Доступные алгоритмы:\n")
    for k, (key, name) in algos.items():
        print(f"  [{k}] {name}")
    print("\n  000 — все три алгоритма (+ ensemble в 03s)")
    print("  Либо номера через запятую (например: 1,2,3)\n")

    raw = input("Ваш выбор (Enter = 000): ").strip() or "000"
    if raw == "000":
        selected_keys = [algos[k][0] for k in ("1", "2", "3")]
    else:
        idxs = [x.strip() for x in raw.split(",") if x.strip() in algos]
        if not idxs:
            raise ValueError("Не выбран ни один алгоритм!")
        selected_keys = [algos[i][0] for i in idxs]

    build_ensemble = False
    if len(selected_keys) >= 2:
        ens = input("\nСтроить ENSEMBLE (взвешенное среднее по AUC)? (Enter = да, N = нет): ").strip().upper()
        build_ensemble = (ens != "N")

    print(f"\n✓ Выбраны алгоритмы: {', '.join(selected_keys)}")
    if build_ensemble:
        print("✓ Ensemble: включён")
    print()

    # ================================================================
    # ШАГ 4: СОХРАНЕНИЕ КОНФИГУРАЦИИ
    # ================================================================
    with open(os.path.join(OUTPUT_DIR, "selected_algorithms.pkl"), "wb") as f:
        pickle.dump(
            {"selected_keys": selected_keys, "build_ensemble": build_ensemble}, f
        )

    cv_config = {"folds_train": folds_train, "folds_test": folds_test}
    if block_labels is not None:
        cv_config["block_labels"] = block_labels.tolist()
    with open(os.path.join(OUTPUT_DIR, "cv_folds.pkl"), "wb") as f:
        pickle.dump(cv_config, f)

    with open(os.path.join(OUTPUT_DIR, "data_info.pkl"), "wb") as f:
        pickle.dump(
            {
                "predictor_cols": predictor_cols,
                "n_predictors": n_pred,
                "target_col": "pa",
                "cv_method": cv_method,
                "n_folds": n_folds,
                "n_presence": n_pres,
                "n_background": n_back,
                "data_dir": DATA_DIR,
                "samples_path": samples_path,
                "source_tag": source_tag,
                "background_generated": (source_tag == "data"),
                "bg_raster_path": bg_raster_path,
            },
            f,
        )

    print("✓ Сохранено в 02_SELECT_SDM:")
    print("  1. selected_algorithms.pkl")
    print("  2. cv_folds.pkl")
    print("  3. data_info.pkl\n")
    print(f"  Метод CV: {cv_label} — {n_folds} фолдов")
    print("\nСледующий шаг: запустите 03s_tune_sdm.py")
    print("============================================================\n")

    return {
        "selected_keys": selected_keys,
        "build_ensemble": build_ensemble,
        "cv_method": cv_method,
        "output_dir": OUTPUT_DIR,
        "data_dir": DATA_DIR,
    }


if __name__ == "__main__":
    main()
