# ============================================================================
# 01s_prepare_sdm_v1.py
# СКРИПТ 01s: ПОДГОТОВКА ДАННЫХ ДЛЯ SDM (PRESENCE-ONLY / PRESENCE-BACKGROUND)
#
# Аналог 01_prepare_data.py из пайплайна классификации, но для
# моделирования встречаемости вида по ТОЧКАМ ПРИСУТСТВИЯ.
#
# Принципиальные отличия от классификации (Variant 3):
#   - Вход: CSV с колонками lon/lat (точки присутствия одного вида),
#     а НЕ полигоны-эталоны нескольких классов.
#   - Целевая переменная: pa (1 = presence, 0 = background), а НЕ class_name.
#   - Один образец = ОДИН ПИКСЕЛЬ (точка), а НЕ среднее по полигону.
#   - Дополнительно генерируются BACKGROUND-точки (по умолчанию 10 000),
#     взвешенные по BIAS-GRID (Gaussian KDE плотности точек сбора),
#     как в SDMToolbox. Это корректирует sampling bias.
#
# Функционал:
#   - Читает настройки слоёв из 00_PREPARE_LAYERS (pipelineconfig.pkl) — опционально
#   - Выбор растра-композита предикторов (Composite / Composite_filtered)
#   - Чтение CSV точек присутствия (lon/lat), проекция в CRS растра
#   - Дедупликация точек до 1 на пиксель (опционально — устранение
#     пространственной псевдо-репликации)
#   - Построение BIAS-GRID: Gaussian KDE плотности точек присутствия,
#     растеризованная в сетку растра (сохраняется как bias.tif)
#   - Генерация N background-точек с вероятностью ~ bias (или равномерно)
#   - Извлечение значений предикторов в точках presence и background
#   - Удаление точек с nodata в любом предикторе
#   - Сохранение: samples.gpkg / samples.csv (с колонкой pa), bias.tif,
#     raster_info.pkl, sdm_meta.pkl
#
# Нормализация НЕ выполняется (z-score делается в 03s через sklearn preProcess;
# для maxent.jar стандартизация не нужна).
# ============================================================================

import os
import sys
import glob
import re
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol, xy
from scipy.stats import gaussian_kde


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def _ask(prompt, default=None):
    val = input(prompt).strip()
    if val == "" and default is not None:
        return default
    return val


def _select_csv(root_dir):
    """Поиск и выбор CSV с точками присутствия."""
    input_dir = os.path.join(root_dir, "INPUT")
    search_dirs = [input_dir, root_dir]
    all_csv = []
    for d in search_dirs:
        if os.path.isdir(d):
            all_csv.extend(glob.glob(os.path.join(d, "*.csv")))
    all_csv = sorted(set(all_csv))
    if not all_csv:
        raise FileNotFoundError(
            "CSV c точками присутствия не найден! "
            "Поместите файл (колонки lon/lat) в папку INPUT."
        )
    print("\nНайдены CSV файлы:\n")
    for i, c in enumerate(all_csv, 1):
        print(f"  [{i}] {c}")
    raw = _ask("\nВыберите номер CSV с точками присутствия: ", "1")
    idx = int(raw)
    if idx < 1 or idx > len(all_csv):
        raise ValueError("Некорректный выбор CSV.")
    return all_csv[idx - 1]


def _detect_lonlat(df):
    """Автоопределение колонок долготы/широты."""
    cols = {c.lower(): c for c in df.columns}
    lon_aliases = ["lon", "long", "longitude", "x", "decimallongitude", "lng"]
    lat_aliases = ["lat", "latitude", "y", "decimallatitude"]
    lon_col = next((cols[a] for a in lon_aliases if a in cols), None)
    lat_col = next((cols[a] for a in lat_aliases if a in cols), None)
    return lon_col, lat_col


def _build_bias_grid(pres_rc, height, width, bandwidth_px=None):
    """
    Построение BIAS-GRID методом Gaussian KDE плотности точек присутствия,
    как в SDMToolbox.

    Параметры
    ---------
    pres_rc : np.ndarray (n, 2) — (row, col) точек присутствия в сетке растра
    height, width : размеры растра
    bandwidth_px : ширина ядра в пикселях (None = автоматически, метод Скотта)

    Возвращает
    ----------
    bias : np.ndarray (height, width), float32, нормирован к [0, 1],
           вне валидной области заполнен 0 (заполняется позже маской).
    """
    # KDE по координатам (col=x, row=y) точек присутствия
    xy_pts = np.vstack([pres_rc[:, 1], pres_rc[:, 0]]).astype(float)  # (2, n): (x=col, y=row)

    # Подгонка KDE
    if bandwidth_px is not None and bandwidth_px > 0:
        # bw_method для gaussian_kde — относительный множитель к std;
        # переведём абсолютную ширину (в пикселях) в относительную
        std_x = xy_pts[0].std() if xy_pts[0].std() > 0 else 1.0
        std_y = xy_pts[1].std() if xy_pts[1].std() > 0 else 1.0
        std_mean = (std_x + std_y) / 2.0
        bw_method = bandwidth_px / std_mean
        kde = gaussian_kde(xy_pts, bw_method=bw_method)
    else:
        kde = gaussian_kde(xy_pts)  # метод Скотта по умолчанию

    # Оцениваем плотность на регулярной сетке растра.
    # Для скорости — на прореженной сетке, затем интерполируем (nearest).
    step = max(1, int(round(max(height, width) / 600)))  # ~600 узлов по длинной стороне
    rows_grid = np.arange(0, height, step)
    cols_grid = np.arange(0, width, step)
    cc, rr = np.meshgrid(cols_grid, rows_grid)
    coords_eval = np.vstack([cc.ravel(), rr.ravel()])

    dens = kde(coords_eval).reshape(len(rows_grid), len(cols_grid))

    # Интерполяция обратно на полную сетку (nearest по индексам)
    full = np.zeros((height, width), dtype=np.float32)
    row_idx = np.clip((np.arange(height) // step), 0, len(rows_grid) - 1)
    col_idx = np.clip((np.arange(width) // step), 0, len(cols_grid) - 1)
    full = dens[np.ix_(row_idx, col_idx)].astype(np.float32)

    # Нормировка к [0, 1]
    fmin, fmax = float(full.min()), float(full.max())
    if fmax > fmin:
        full = (full - fmin) / (fmax - fmin)
    else:
        full[:] = 1.0
    return full


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n============================")
    print("  01s - PREPARE SDM DATA")
    print("============================\n")

    root_dir = os.getcwd()

    # --- (опционально) подхватываем pipelineconfig из 00_PREPARE_LAYERS ---
    selected_prev_dir = None
    selected_suffix = "_v001"
    prev_dir = os.path.join(root_dir, "00_PREPARE_LAYERS")
    config = None
    if os.path.isdir(prev_dir):
        prev_subdirs = sorted(
            os.path.join(prev_dir, d)
            for d in os.listdir(prev_dir)
            if os.path.isdir(os.path.join(prev_dir, d))
        )
        if prev_subdirs:
            print("Доступные запуски в 00_PREPARE_LAYERS:\n")
            for i, sd in enumerate(prev_subdirs, 1):
                print(f" [{i}] {os.path.basename(sd)}")
            choice = int(_ask("\nВыберите номер запуска (Enter = 1): ", "1"))
            selected_prev_dir = prev_subdirs[choice - 1]
            m = re.search(r"(_v\d{3})$", os.path.basename(selected_prev_dir))
            selected_suffix = m.group(1) if m else "_v001"
            cfg_pkl = os.path.join(selected_prev_dir, "pipelineconfig.pkl")
            if os.path.exists(cfg_pkl):
                with open(cfg_pkl, "rb") as f:
                    config = pickle.load(f)
                print(f"✓ pipelineconfig.pkl загружен (CRS: {config.get('target_crs', '?')})")

    # --- выходная папка ---
    base_output_dir = os.path.join(root_dir, "01_PREPARE_SDM")
    os.makedirs(base_output_dir, exist_ok=True)
    run_stamp = datetime.now().strftime("%d%m%y_%H%M")
    output_dir = os.path.join(base_output_dir, f"{run_stamp}{selected_suffix}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Out  : {output_dir}\n")

    # ================================================================
    # ШАГ 1: ВЫБОР РАСТРА-КОМПОЗИТА ПРЕДИКТОРОВ
    # ================================================================
    print("================================================================")
    print("[ШАГ 1] ВЫБОР РАСТРА-КОМПОЗИТА ПРЕДИКТОРОВ")
    print("================================================================\n")

    tif_search = []
    if selected_prev_dir:
        tif_search.append(selected_prev_dir)
    tif_search += [os.path.join(root_dir, "INPUT"), root_dir]
    tif_files = []
    for d in tif_search:
        if os.path.isdir(d):
            tif_files.extend(glob.glob(os.path.join(d, "*.tif")))
    tif_files = sorted(set(tif_files))
    if not tif_files:
        raise FileNotFoundError("TIF-композит не найден (00_PREPARE_LAYERS / INPUT).")

    for i, tf in enumerate(tif_files, 1):
        try:
            with rasterio.open(tf) as src:
                nlyr = src.count
        except Exception:
            nlyr = "?"
        fsize = os.path.getsize(tf) / (1024 ** 2)
        print(f" {i:2d}) {os.path.basename(tf)}  слоёв={nlyr}  {fsize:.1f} MB")

    choice = int(_ask("\nВыберите номер растра (Enter = 1): ", "1"))
    selected_tif = tif_files[choice - 1]
    print(f"\n✓ Выбран: {os.path.basename(selected_tif)}\n")

    with rasterio.open(selected_tif) as src:
        n_layers = src.count
        width = src.width
        height = src.height
        raster_crs = src.crs
        transform = src.transform
        src_nodata = src.nodata
        src_dtype = src.dtypes[0]
        band_names = [
            src.descriptions[i] if src.descriptions[i] else f"band{i + 1}"
            for i in range(n_layers)
        ]
    print(f"  Слоёв: {n_layers}, Размер: {width} x {height}")
    print(f"  CRS: {raster_crs}\n")

    # ================================================================
    # ШАГ 2: ЧТЕНИЕ ТОЧЕК ПРИСУТСТВИЯ (CSV lon/lat)
    # ================================================================
    print("================================================================")
    print("[ШАГ 2] ТОЧКИ ПРИСУТСТВИЯ (CSV lon/lat)")
    print("================================================================\n")

    csv_path = _select_csv(root_dir)
    pres_raw = pd.read_csv(csv_path)
    print(f"  Загружено строк: {len(pres_raw)}")

    lon_col, lat_col = _detect_lonlat(pres_raw)
    if lon_col is None or lat_col is None:
        print(f"  Колонки: {', '.join(pres_raw.columns)}")
        lon_col = _ask("  Имя колонки ДОЛГОТЫ (lon): ")
        lat_col = _ask("  Имя колонки ШИРОТЫ (lat): ")
    print(f"  ✓ Долгота: {lon_col}, Широта: {lat_col}")

    # CRS исходных координат — обычно WGS84 (EPSG:4326)
    src_epsg = _ask("  EPSG координат в CSV (Enter = 4326 / WGS84): ", "4326")
    pres_gdf = gpd.GeoDataFrame(
        pres_raw.copy(),
        geometry=gpd.points_from_xy(pres_raw[lon_col], pres_raw[lat_col]),
        crs=f"EPSG:{src_epsg}",
    )
    pres_gdf = pres_gdf.to_crs(raster_crs)
    print(f"  ✓ Перепроецировано в {raster_crs}")

    # row/col точек в сетке растра
    xs = pres_gdf.geometry.x.values
    ys = pres_gdf.geometry.y.values
    rows, cols = rowcol(transform, xs, ys)
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    # Отбрасываем точки за пределами растра
    in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    n_out = int((~in_bounds).sum())
    if n_out:
        print(f"  ! Вне границ растра отброшено: {n_out}")
    rows, cols = rows[in_bounds], cols[in_bounds]
    xs, ys = xs[in_bounds], ys[in_bounds]

    # Дедупликация: 1 точка на пиксель (устранение псевдо-репликации)
    dedup = _ask("  Оставить 1 точку на пиксель? (Enter = да, N = нет): ", "Y").upper()
    if dedup != "N":
        rc = np.column_stack([rows, cols])
        _, uniq_idx = np.unique(rc, axis=0, return_index=True)
        uniq_idx = np.sort(uniq_idx)
        n_dup = len(rows) - len(uniq_idx)
        rows, cols = rows[uniq_idx], cols[uniq_idx]
        xs, ys = xs[uniq_idx], ys[uniq_idx]
        if n_dup:
            print(f"  ✓ Удалено дублей пикселей: {n_dup}")
    print(f"  ✓ Точек присутствия в работе: {len(rows)}\n")

    pres_rc = np.column_stack([rows, cols])

    # ================================================================
    # ШАГ 3: ВАЛИДНАЯ МАСКА РАСТРА (для семплинга background)
    # ================================================================
    print("================================================================")
    print("[ШАГ 3] МАСКА ВАЛИДНЫХ ПИКСЕЛЕЙ")
    print("================================================================\n")
    print("  Считаем маску: пиксель валиден, если ВСЕ предикторы не nodata.\n")

    is_float = np.issubdtype(np.dtype(src_dtype), np.floating)
    valid_mask = np.ones((height, width), dtype=bool)
    with rasterio.open(selected_tif) as src:
        for b in range(1, n_layers + 1):
            arr = src.read(b)
            band_valid = np.isfinite(arr) if is_float else np.ones_like(arr, dtype=bool)
            if src_nodata is not None and not (
                isinstance(src_nodata, float) and np.isnan(src_nodata)
            ):
                band_valid &= (arr != src_nodata)
            valid_mask &= band_valid
    n_valid = int(valid_mask.sum())
    print(f"  ✓ Валидных пикселей: {n_valid} из {height * width}\n")

    # ================================================================
    # ШАГ 4: BIAS-GRID (Gaussian KDE плотности точек сбора)
    # ================================================================
    print("================================================================")
    print("[ШАГ 4] BIAS-GRID (Gaussian KDE, как в SDMToolbox)")
    print("================================================================\n")

    bw_raw = _ask(
        "  Ширина ядра KDE в пикселях (Enter = авто, метод Скотта): ", ""
    )
    bandwidth_px = float(bw_raw) if bw_raw else None

    print("  Построение bias-grid по KDE плотности точек присутствия...")
    bias = _build_bias_grid(pres_rc, height, width, bandwidth_px=bandwidth_px)
    # Обнуляем bias вне валидной области
    bias = np.where(valid_mask, bias, 0.0).astype(np.float32)

    # Сохраняем bias.tif
    bias_profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": raster_crs,
        "transform": transform,
        "compress": "lzw",
        "nodata": 0.0,
    }
    bias_path = os.path.join(output_dir, "bias.tif")
    with rasterio.open(bias_path, "w", **bias_profile) as dst:
        dst.write(bias, 1)
    print(f"  ✓ Сохранён bias-grid: {os.path.basename(bias_path)}\n")

    # ================================================================
    # ШАГ 5: ГЕНЕРАЦИЯ BACKGROUND-ТОЧЕК (взвешенно по bias)
    # ================================================================
    print("================================================================")
    print("[ШАГ 5] BACKGROUND-ТОЧКИ")
    print("================================================================\n")

    n_bg = int(_ask("  Сколько background-точек (Enter = 10000): ", "10000"))
    use_bias = _ask(
        "  Взвешивать background по bias-grid? (Enter = да, N = равномерно): ", "Y"
    ).upper()

    valid_flat = np.flatnonzero(valid_mask.ravel())
    n_bg = min(n_bg, len(valid_flat))

    rng = np.random.default_rng(42)
    if use_bias != "N":
        w = bias.ravel()[valid_flat].astype(np.float64)
        if w.sum() <= 0:
            print("  ! Bias пустой, переключаюсь на равномерный семплинг.")
            probs = None
        else:
            probs = w / w.sum()
    else:
        probs = None

    bg_choice = rng.choice(valid_flat, size=n_bg, replace=False, p=probs)
    bg_rows, bg_cols = np.unravel_index(bg_choice, (height, width))
    bg_x, bg_y = xy(transform, bg_rows, bg_cols)  # центры пикселей
    bg_x = np.asarray(bg_x)
    bg_y = np.asarray(bg_y)
    print(f"  ✓ Сгенерировано background-точек: {n_bg} "
          f"({'взвешенно по bias' if probs is not None else 'равномерно'})\n")

    # ================================================================
    # ШАГ 6: ИЗВЛЕЧЕНИЕ ПРЕДИКТОРОВ В ТОЧКАХ
    # ================================================================
    print("================================================================")
    print("[ШАГ 6] ИЗВЛЕЧЕНИЕ ПРЕДИКТОРОВ")
    print("================================================================\n")

    all_rows = np.concatenate([pres_rc[:, 0], bg_rows])
    all_cols = np.concatenate([pres_rc[:, 1], bg_cols])
    all_x = np.concatenate([xs, bg_x])
    all_y = np.concatenate([ys, bg_y])
    pa = np.concatenate([
        np.ones(len(pres_rc), dtype=int),
        np.zeros(n_bg, dtype=int),
    ])

    # Читаем значения предикторов по индексам пикселей
    feat = np.full((len(all_rows), n_layers), np.nan, dtype=np.float64)
    with rasterio.open(selected_tif) as src:
        for b in range(1, n_layers + 1):
            arr = src.read(b).astype(np.float64)
            vals = arr[all_rows, all_cols]
            if src_nodata is not None and not (
                isinstance(src_nodata, float) and np.isnan(src_nodata)
            ):
                vals = np.where(vals == src_nodata, np.nan, vals)
            feat[:, b - 1] = vals

    df = pd.DataFrame(feat, columns=band_names)
    df["pa"] = pa
    df["x"] = all_x
    df["y"] = all_y

    # Удаляем строки с nodata в любом предикторе
    before = len(df)
    df = df.dropna(subset=band_names).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  ! Удалено точек с nodata в предикторах: {dropped}")

    n_pres = int((df["pa"] == 1).sum())
    n_back = int((df["pa"] == 0).sum())
    print(f"  ✓ Итог: presence={n_pres}, background={n_back}")
    print(f"  ✓ Предикторы ({n_layers}): {', '.join(band_names)}\n")

    # ================================================================
    # ШАГ 7: СОХРАНЕНИЕ
    # ================================================================
    print("================================================================")
    print("[ШАГ 7] СОХРАНЕНИЕ")
    print("================================================================\n")

    geometry = gpd.points_from_xy(df["x"], df["y"], crs=raster_crs)
    samples_gdf = gpd.GeoDataFrame(
        df.drop(columns=["x", "y"]), geometry=geometry, crs=raster_crs
    )
    gpkg_path = os.path.join(output_dir, "samples.gpkg")
    if os.path.exists(gpkg_path):
        os.remove(gpkg_path)
    samples_gdf.to_file(gpkg_path, layer="samples", driver="GPKG")
    print(f"  ✓ samples.gpkg ({len(samples_gdf)} точек: pa + {n_layers} предикторов)")

    csv_out = os.path.join(output_dir, "samples.csv")
    df.to_csv(csv_out, index=False)
    print(f"  ✓ samples.csv")

    raster_info = {
        "raster_path": selected_tif,
        "raster_name": os.path.basename(selected_tif),
        "n_layers": n_layers,
        "layer_names": band_names,
        "crs": str(raster_crs),
    }
    with open(os.path.join(output_dir, "raster_info.pkl"), "wb") as f:
        pickle.dump(raster_info, f)
    print(f"  ✓ raster_info.pkl")

    sdm_meta = {
        "presence_csv": csv_path,
        "lon_col": lon_col,
        "lat_col": lat_col,
        "src_epsg": src_epsg,
        "n_presence": n_pres,
        "n_background": n_back,
        "bias_path": bias_path,
        "bias_used": (use_bias != "N"),
        "bandwidth_px": bandwidth_px,
        "predictor_names": band_names,
        "target_col": "pa",
    }
    with open(os.path.join(output_dir, "sdm_meta.pkl"), "wb") as f:
        pickle.dump(sdm_meta, f)
    print(f"  ✓ sdm_meta.pkl\n")

    print("================================================================")
    print("  ПОДГОТОВКА ДАННЫХ SDM ЗАВЕРШЕНА")
    print("================================================================\n")
    print("Целевая колонка: pa (1 = presence, 0 = background)")
    print(f"Bias-grid: {os.path.basename(bias_path)} (Gaussian KDE)")
    print("\nСледующий шаг: запустите 02s_select_sdm.py")
    print("================================================================\n")

    return {
        "output_dir": output_dir,
        "n_presence": n_pres,
        "n_background": n_back,
        "predictor_names": band_names,
    }


if __name__ == "__main__":
    main()
