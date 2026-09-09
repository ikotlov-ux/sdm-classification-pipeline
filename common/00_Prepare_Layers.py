# ============================================================================
# 00_Prepare_Layers_v62.py
# Скрипт для объединения TIF растров в один файл (Composite)
#
# v62 (оптимизация диска и памяти):
#   - ZSTD + PREDICTOR=3 + INTERLEAVE=BAND для всех записей float32
#     (вместо LZW + дефолтный PIXEL interleave). Даёт в ~5-10×
#     лучшее сжатие многоканальных растров с NaN (S2 indices: 213 слоёв).
#   - Пропуск aligned-копии, если растр уже на целевой сетке —
#     работаем с исходником напрямую.
#   - Потоковое объединение (band-by-band) вместо загрузки всех
#     302 слоёв в RAM. Экономит ~47 ГБ памяти.
#   - Явный BIGTIFF=YES при >4 ГБ на любом этапе
#   - num_threads=0 в reproject() — параллельность по всем ядрам.
#
# Функционал:
# - Автоопределение UTM зоны по растровым данным
# - Интерактивный выбор проекции (EPSG)
# - Загрузка маски (SHP) — опционально
# - Поиск TIF файлов в INPUT, выбор порядка соединения
# - Выравнивание растров (проекция, экстент, разрешение)
# - Объединение в многослойный Composite.tif
# - Интерактивное переименование слоёв
# - Загрузка эталонных полигонов → извлечение пикселей
# - Тесты Краскела-Уоллиса, графики медиана + CI по классам
# - Матрицы корреляций Спирмена (PNG + CSV)
# - Фильтрация слоёв по взаимной корреляции → Composite_filtered.tif
# ============================================================================

import os
import sys
import glob
import math
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds, array_bounds
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
    transform_bounds,
)
from rasterio.merge import merge
from rasterio.mask import mask as rio_mask
import geopandas as gpd
from shapely.geometry import mapping
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats as sp_stats
from joblib import Parallel, delayed, cpu_count


# ============================================================================
# ФУНКЦИЯ: Автоопределение UTM зоны по данным растра
# ============================================================================

def _unique_tif_list(pattern_list: list[str]) -> list[str]:
    """Убирает дубликаты путей (Windows: *.tif и *.TIF дают одни и те же файлы)."""
    seen = {}
    result = []
    for p in pattern_list:
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen[key] = True
            result.append(p)
    return result


def detect_utm_zone(input_dir: str) -> dict | None:
    """Определяет UTM зону по первому TIF из input_dir."""
    print("\n Автоопределение UTM зоны...")
    tif_files = _unique_tif_list(sorted(
        glob.glob(os.path.join(input_dir, "*.tif"))
        + glob.glob(os.path.join(input_dir, "*.TIF"))
    ))
    if not tif_files:
        print(" ⚠ TIF файлы не найдены, автоопределение невозможно\n")
        return None

    try:
        path0 = tif_files[0]
        print(f" Анализирую: {os.path.basename(path0)}")
        with rasterio.open(path0) as src:
            bounds = src.bounds
            src_crs = src.crs
        # перевод в WGS84
        left, bottom, right, top = transform_bounds(
            src_crs, CRS.from_epsg(4326), *bounds
        )
        center_lon = (left + right) / 2
        center_lat = (bottom + top) / 2
        print(
            f" Центр данных (WGS84): долгота = {center_lon:.4f}, "
            f"широта = {center_lat:.4f}"
        )

        utm_zone = int(math.floor((center_lon + 180) / 6)) + 1
        if center_lat >= 0:
            hemisphere = "N"
            epsg_base = 32600
        else:
            hemisphere = "S"
            epsg_base = 32700

        epsg_main = epsg_base + utm_zone
        zone_prev = utm_zone - 1 if utm_zone > 1 else 60
        zone_next = utm_zone + 1 if utm_zone < 60 else 1

        return {
            "utm_zone": utm_zone,
            "hemisphere": hemisphere,
            "epsg_main": epsg_main,
            "epsg_prev": epsg_base + zone_prev,
            "epsg_next": epsg_base + zone_next,
            "zone_prev": zone_prev,
            "zone_next": zone_next,
            "center_lon": center_lon,
            "center_lat": center_lat,
            "source_file": os.path.basename(path0),
        }
    except Exception as e:
        print(f" ⚠ Ошибка: {e}\n")
        return None


# ============================================================================
# ФУНКЦИЯ: Запрос и валидация кода EPSG (с автоопределением UTM)
# ============================================================================

def select_target_crs(input_dir: str) -> CRS:
    print("\n════════════════════════════════════════════════════════")
    print("ВЫБОР ЦЕЛЕВОЙ ПРОЕКЦИИ (КОД EPSG)")
    print("════════════════════════════════════════════════════════\n")

    utm_info = detect_utm_zone(input_dir)

    print("Введите код EPSG для проекции, в которой будут выполняться расчеты.\n")

    if utm_info:
        print(f" Данные определены по файлу: {utm_info['source_file']}")
        print(
            f" Центр данных: {utm_info['center_lon']:.4f}° долг., "
            f"{utm_info['center_lat']:.4f}° шир.\n"
        )
        print(
            f" {utm_info['epsg_prev']:5d} - UTM Zone {utm_info['zone_prev']}"
            f"{utm_info['hemisphere']} (соседняя -1)"
        )
        print(
            f" >>> {utm_info['epsg_main']:5d} - UTM Zone {utm_info['utm_zone']}"
            f"{utm_info['hemisphere']} <<< РЕКОМЕНДУЕМАЯ"
        )
        print(
            f" {utm_info['epsg_next']:5d} - UTM Zone {utm_info['zone_next']}"
            f"{utm_info['hemisphere']} (соседняя +1)\n"
        )

    print("Другие популярные коды:")
    print(" 4326 - WGS84 (широта/долгота)")
    print(" 3857 - Web Mercator")
    print(" 3395 - World Mercator\n")

    if utm_info:
        prompt = (
            f"Введите код EPSG (Enter = {utm_info['epsg_main']}, "
            f"рекомендуемая UTM): "
        )
    else:
        prompt = "Введите код EPSG (Enter = 4326, WGS84): "

    epsg_input = input(prompt).strip()

    if epsg_input == "":
        if utm_info:
            target_crs = CRS.from_epsg(utm_info["epsg_main"])
            print(
                f"\n✓ Рекомендуемая проекция: UTM Zone {utm_info['utm_zone']}"
                f"{utm_info['hemisphere']} (EPSG:{utm_info['epsg_main']})\n"
            )
        else:
            target_crs = CRS.from_epsg(4326)
            print("\n⚠ Используется WGS84 (EPSG:4326) по умолчанию\n")
    else:
        try:
            code = int(epsg_input)
            target_crs = CRS.from_epsg(code)
            print(f"\n✓ Проекция: EPSG:{code}\n")
        except Exception:
            print(f"\n⚠ Код EPSG:{epsg_input} не распознан.")
            if utm_info:
                target_crs = CRS.from_epsg(utm_info["epsg_main"])
                print(
                    f" Используется UTM Zone {utm_info['utm_zone']}"
                    f"{utm_info['hemisphere']}\n"
                )
            else:
                target_crs = CRS.from_epsg(4326)
                print(" Используется WGS84\n")

    return target_crs


# ============================================================================
# ФУНКЦИЯ: Загрузка маски из SHP файла (опционально)
# ============================================================================

def load_mask(input_dir: str, target_crs: CRS) -> gpd.GeoDataFrame | None:
    print("\n════════════════════════════════════════════════════════")
    print("ЗАГРУЗКА МАСКИ (опционально)")
    print("════════════════════════════════════════════════════════\n")

    shp_files = sorted(
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith(".shp")
    )
    if not shp_files:
        print("SHP файлы не найдены в папке.")
        print("Работу будете выполнять без маски.\n")
        return None

    print(f"Найдено SHP файлов: {len(shp_files)}\n")
    for i, f in enumerate(shp_files, 1):
        print(f"[{i}] {f}")

    print("\nВыберите номер файла для использования в качестве маски.")
    print("(или введите 0 для работы без маски)\n")
    choice = input("Номер файла (0 для пропуска): ").strip()

    try:
        choice = int(choice)
    except ValueError:
        choice = 0

    if choice == 0:
        print("\nМаска не выбрана. Работа без маски.\n")
        return None

    if choice < 1 or choice > len(shp_files):
        print("Некорректный номер! Работа без маски.\n")
        return None

    shp_path = os.path.join(input_dir, shp_files[choice - 1])
    print(f"\nЗагружаю маску: {shp_files[choice - 1]}")

    try:
        gdf = gpd.read_file(shp_path)
        print("✓ Маска загружена успешно")
        print(f" Тип геометрии: {gdf.geom_type.unique()}")
        print(f" Количество объектов: {len(gdf)}")
        print(f" СКО маски: {gdf.crs}")

        if gdf.crs != target_crs:
            print(f"\n ⚠ Маска в другой проекции!")
            print(f" Преобразую маску из {gdf.crs} в {target_crs}")
            gdf = gdf.to_crs(target_crs)
            print(" ✓ Маска переведена в целевую проекцию\n")
        else:
            print()

        return gdf
    except Exception as e:
        print(f"✗ ОШИБКА загрузки маски: {e}\n")
        return None


# ============================================================================
# Вспомогательные функции чтения/записи растров
# ============================================================================

def _read_raster_info(file_path: str) -> dict:
    """Читает метаинформацию о растре."""
    try:
        with rasterio.open(file_path) as src:
            return {
                "path": file_path,
                "file": os.path.basename(file_path),
                "layers": src.count,
                "width": src.width,
                "height": src.height,
                "bounds": src.bounds,
                "res": src.res,
                "crs": src.crs,
                "band_names": list(src.descriptions)
                if any(src.descriptions)
                else [],
                "profile": src.profile.copy(),
                "success": True,
            }
    except Exception as e:
        return {
            "path": file_path,
            "file": os.path.basename(file_path),
            "layers": None,
            "success": False,
            "error": str(e),
        }


def _needs_bigtiff(count: int, height: int, width: int, dtype: str = "float32") -> bool:
    """Проверяет, превысит ли растр порог 4 ГБ (стандартный TIFF лимит)."""
    bytes_per_pixel = np.dtype(dtype).itemsize
    estimated_bytes = int(count) * int(height) * int(width) * bytes_per_pixel
    # Порог 3.5 ГБ (с запасом, т.к. сжатие не гарантирует уменьшение)
    return estimated_bytes > 3.5 * 1024 ** 3


def _write_geotiff(
    data: np.ndarray,
    profile: dict,
    out_path: str,
    band_names: list[str] | None = None,
    block_size: int = 512,
):
    """Записывает numpy-массив в GeoTIFF с LZW, тайлами."""
    p = profile.copy()
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    # v2: ZSTD + PREDICTOR=3 + BAND interleave — лучшее сжатие float с NaN
    p.update(
        driver="GTiff",
        count=data.shape[0],
        height=data.shape[1],
        width=data.shape[2],
        dtype=data.dtype,
        compress="zstd",
        zstd_level=3,
        predictor=3,
        tiled=True,
        blockxsize=block_size,
        blockysize=block_size,
        interleave="band",
    )
    if _needs_bigtiff(data.shape[0], data.shape[1], data.shape[2], str(data.dtype)):
        p["BIGTIFF"] = "YES"
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(data)
        if band_names:
            for i, nm in enumerate(band_names):
                dst.set_band_description(i + 1, nm)


# ============================================================================
# ФУНКЦИЯ: Выравнивание растров по общей сетке и конвертация в целевую проекцию
# ============================================================================

def align_rasters_to_common_grid(
    raster_paths: list[str], target_crs: CRS, temp_dir: str
) -> dict:
    """
    Перепроецирует и выравнивает список растров на общую сетку.
    Возвращает dict с aligned_files, common_extent, common_resolution, template_profile.
    """
    print("════════════════════════════════════════════════════════")
    print("ПРОВЕРКА И КОНВЕРТАЦИЯ ПРОЕКЦИЙ")
    print("════════════════════════════════════════════════════════\n")

    n_rasters = len(raster_paths)
    print(f"Целевая проекция: {target_crs}\n")

    # Шаг 1: перепроецируем каждый растр в target_crs (если нужно)
    reprojected_dir = os.path.join(temp_dir, "terra_align")
    os.makedirs(reprojected_dir, exist_ok=True)

    reproj_paths = []
    n_reprojected = 0
    n_already_ok = 0

    for i, rp in enumerate(raster_paths):
        with rasterio.open(rp) as src:
            src_crs = src.crs
            base = os.path.basename(rp)
            print(f"[{i + 1}] Растр {base}")

            if src_crs == target_crs:
                print(" ✓ Уже в целевой проекции\n")
                reproj_paths.append(rp)
                n_already_ok += 1
            else:
                print(" ⚠ Перевожу в целевую проекцию...")
                out_reproj = os.path.join(
                    reprojected_dir, f"reproj_{i + 1:03d}.tif"
                )
                transform, width, height = calculate_default_transform(
                    src_crs, target_crs, src.width, src.height, *src.bounds
                )
                prof = src.profile.copy()
                # v2: ZSTD + PREDICTOR=3 + BAND interleave
                prof.update(
                    crs=target_crs,
                    transform=transform,
                    width=width,
                    height=height,
                    compress="zstd",
                    zstd_level=1,   # v3.1: level=1 для temp-файлов — 2-3× быстрее записи
                    predictor=3,
                    tiled=True,
                    blockxsize=512,
                    blockysize=512,
                    interleave="band",
                    dtype="float32",
                    nodata=np.nan,
                )
                if _needs_bigtiff(src.count, height, width, "float32"):
                    prof["BIGTIFF"] = "YES"
                with rasterio.open(out_reproj, "w", **prof) as dst:
                    for band in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, band),
                            destination=rasterio.band(dst, band),
                            src_crs=src_crs,
                            dst_crs=target_crs,
                            resampling=Resampling.nearest,
                            dst_nodata=np.nan,
                            num_threads=0,
                        )
                reproj_paths.append(out_reproj)
                n_reprojected += 1
                print(f" ✓ Конвертирована в {target_crs}\n")

    print(
        f"Итого: {n_already_ok} уже в целевой СКО, {n_reprojected} перепроецировано\n"
    )

    # Шаг 2: определяем общий экстент (пересечение) и разрешение
    print("════════════════════════════════════════════════════════")
    print("ВЫРАВНИВАНИЕ РАСТРОВ (ЭКСТЕНТ И РАЗРЕШЕНИЕ)")
    print("════════════════════════════════════════════════════════\n")

    infos = []
    for i, rp in enumerate(reproj_paths):
        with rasterio.open(rp) as src:
            b = src.bounds
            r = src.res
            print(
                f"[{i + 1}] Экстент X: [{b.left:.2f}, {b.right:.2f}], "
                f"Y: [{b.bottom:.2f}, {b.top:.2f}]"
            )
            print(f" Разрешение: X={r[0]:.6f}, Y={r[1]:.6f}")
            print(f" Размер: {src.width} x {src.height} пикселей")
            print(f" СКО: {src.crs}\n")
            infos.append(
                {
                    "bounds": b,
                    "res_x": r[0],
                    "res_y": r[1],
                    "width": src.width,
                    "height": src.height,
                }
            )

    common_xmin = max(info["bounds"].left for info in infos)
    common_xmax = min(info["bounds"].right for info in infos)
    common_ymin = max(info["bounds"].bottom for info in infos)
    common_ymax = min(info["bounds"].top for info in infos)
    common_res_x = min(abs(info["res_x"]) for info in infos)
    common_res_y = min(abs(info["res_y"]) for info in infos)

    print("════════════════════════════════════════════════════════")
    print("ОБЩИЕ ПАРАМЕТРЫ ДЛЯ ВЫРАВНИВАНИЯ")
    print("════════════════════════════════════════════════════════\n")

    print("Экстент (ПЕРЕСЕЧЕНИЕ):")
    print(f" X: [{common_xmin:.2f}, {common_xmax:.2f}]")
    print(f" Y: [{common_ymin:.2f}, {common_ymax:.2f}]\n")

    print("Разрешение (МАКСИМАЛЬНОЕ - наиболее детальное):")
    print(f" X: {common_res_x:.6f}")
    print(f" Y: {common_res_y:.6f}\n")

    extent_w = common_xmax - common_xmin
    extent_h = common_ymax - common_ymin

    print("────────────────────────────────────────────")
    print("ВЫБОР РАЗРЕШЕНИЯ (м/пикс)")
    print("────────────────────────────────────────────\n")

    all_res = sorted(set(round(abs(info["res_x"]), 2) for info in infos))
    print(" Разрешения входных растров (м/пикс):")
    for r in all_res:
        print(f" {r:.1f}")
    print()

    print(" Варианты:")
    print(
        f" >>> [Enter] = {common_res_x:.1f} м (максимальное разрешение) → "
        f"{math.ceil(extent_w / common_res_x)} × "
        f"{math.ceil(extent_h / common_res_y)} пикселей <<<"
    )

    typical = [10, 20, 30, 50, 100, 250, 500, 1000]
    typical = [t for t in typical if t > common_res_x and t <= max(all_res) * 10]
    for tr in typical:
        print(
            f" {tr:5.0f} м → {math.ceil(extent_w / tr)} × "
            f"{math.ceil(extent_h / tr)} пикселей"
        )

    res_input = input(
        f"\nВведите разрешение в м/пикс (Enter = {common_res_x:.1f}): "
    ).strip()
    if res_input:
        try:
            user_res = float(res_input)
            if user_res > 0:
                common_res_x = user_res
                common_res_y = user_res
                print(
                    f"\n✓ Установлено пользовательское разрешение: "
                    f"{common_res_x:.1f} × {common_res_y:.1f} м/пикс"
                )
            else:
                print("\n⚠ Некорректное значение. Используется максимальное разрешение.")
        except ValueError:
            print("\n⚠ Некорректное значение. Используется максимальное разрешение.")
    else:
        print(
            f"\n✓ Используется максимальное разрешение: "
            f"{common_res_x:.1f} × {common_res_y:.1f} м/пикс"
        )

    out_width = math.ceil(extent_w / common_res_x)
    out_height = math.ceil(extent_h / common_res_y)
    print(f" Итоговый размер сетки: {out_width} × {out_height} пикселей\n")

    template_transform = from_bounds(
        common_xmin, common_ymin, common_xmax, common_ymax, out_width, out_height
    )

    print("Переприношу растры на общую сетку:\n")

    aligned_files = []
    # v2: проверяем, совпадает ли исходная сетка с целевой — если да,
    # пропускаем создание aligned-копии и используем исходник напрямую.
    eps_xy = 1e-3       # допуск для сравнения координат (метры)
    eps_res = 1e-6
    n_skipped = 0
    n_aligned = 0
    for i, rp in enumerate(reproj_paths):
        with rasterio.open(rp) as src:
            b = src.bounds
            r = src.res
            grid_match = (
                abs(b.left - common_xmin) < eps_xy
                and abs(b.right - common_xmax) < eps_xy
                and abs(b.bottom - common_ymin) < eps_xy
                and abs(b.top - common_ymax) < eps_xy
                and abs(abs(r[0]) - common_res_x) < eps_res
                and abs(abs(r[1]) - common_res_y) < eps_res
                and src.width == out_width
                and src.height == out_height
            )

        if grid_match:
            print(f" [{i + 1}] Сетка уже совпадает — пропускаю копирование")
            aligned_files.append(rp)
            n_skipped += 1
            continue

        print(f" [{i + 1}] Выравниваю и сохраняю на диск...")
        out_aligned = os.path.join(reprojected_dir, f"aligned_{i + 1:03d}.tif")

        with rasterio.open(rp) as src:
            prof = src.profile.copy()
            # v2: ZSTD + PREDICTOR=3 + BAND interleave — принципиально лучшее
            # сжатие float32 с NaN для многоканальных растров (S2 indices: 213 слоёв).
            prof.update(
                crs=target_crs,
                transform=template_transform,
                width=out_width,
                height=out_height,
                compress="zstd",
                zstd_level=1,   # v3.1: level=1 для temp-файлов — 2-3× быстрее записи
                predictor=3,
                tiled=True,
                blockxsize=512,
                blockysize=512,
                interleave="band",
                dtype="float32",
            )
            prof["nodata"] = np.nan
            if _needs_bigtiff(src.count, out_height, out_width, "float32"):
                prof["BIGTIFF"] = "YES"
            with rasterio.open(out_aligned, "w", **prof) as dst:
                for band in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band),
                        destination=rasterio.band(dst, band),
                        src_crs=src.crs,
                        src_transform=src.transform,
                        dst_crs=target_crs,
                        dst_transform=template_transform,
                        resampling=Resampling.nearest,
                        dst_nodata=np.nan,
                        num_threads=0,  # все доступные ядра
                    )
        aligned_files.append(out_aligned)
        n_aligned += 1

    print(f"\n  Пропущено (сетка уже совпадает): {n_skipped}")
    print(f"  Выравнено с записью на диск:        {n_aligned}")

    print("\n✓ Все растры выравнены и в единой проекции!\n")

    template_profile = {
        "driver": "GTiff",
        "crs": target_crs,
        "transform": template_transform,
        "width": out_width,
        "height": out_height,
        "dtype": "float32",
        "compress": "zstd",
        "zstd_level": 3,
        "predictor": 3,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "interleave": "band",
    }

    return {
        "aligned_files": aligned_files,
        "template_profile": template_profile,
        "common_extent": (common_xmin, common_ymin, common_xmax, common_ymax),
        "common_resolution": (common_res_x, common_res_y),
    }


# ============================================================================
# ФУНКЦИЯ: Применение маски к растру
# ============================================================================

def apply_mask_in_memory(
    composite: np.ndarray,
    mask_gdf: gpd.GeoDataFrame | None,
    transform,
) -> np.ndarray:
    """Применяет маску к numpy-массиву в памяти (до записи на диск).
    composite: (bands, H, W) float32 array.
    Возвращает модифицированный массив (пиксели вне маски = NaN).
    """
    if mask_gdf is None:
        print("Маска не выбрана, пропускаю применение маски.\n")
        return composite

    print("════════════════════════════════════════════════════════")
    print("ПРИМЕНЕНИЕ МАСКИ")
    print("════════════════════════════════════════════════════════\n")
    print("Применяю маску к массиву в памяти...")

    try:
        from rasterio.features import geometry_mask as _geom_mask
        geometries = [mapping(geom) for geom in mask_gdf.geometry]

        # Булева маска: True вне полигонов
        inv_mask = _geom_mask(
            geometries,
            out_shape=(composite.shape[1], composite.shape[2]),
            transform=transform,
            invert=False,
        )
        # Применяем: все бэнды сразу через broadcasting
        composite[:, inv_mask] = np.nan

        n_masked = int(inv_mask.sum())
        n_total = inv_mask.size
        print(f"✓ Маска применена: {n_masked:,} из {n_total:,} пикселей установлены в NaN\n".replace(",", " "))
    except Exception as e:
        print(f"⚠ ОШИБКА при применении маски: {e}")
        print(" Продолжаю работу без маски\n")

    return composite


# ============================================================================
# ФУНКЦИЯ: Поиск TIF файлов
# ============================================================================

def find_tif_files(input_dir: str) -> list[str]:
    tif_files = _unique_tif_list(sorted(
        glob.glob(os.path.join(input_dir, "*.tif"))
        + glob.glob(os.path.join(input_dir, "*.TIF"))
    ))
    import re

    exclude_pat = re.compile(
        r"Composite|CompositeNorm|scaling|boxplot|Spearman", re.IGNORECASE
    )
    tif_files = [f for f in tif_files if not exclude_pat.search(os.path.basename(f))]

    if not tif_files:
        print(" ⚠ TIF не найдены в корне, ищу рекурсивно...")
        tif_files = _unique_tif_list(sorted(
            glob.glob(os.path.join(input_dir, "**", "*.tif"), recursive=True)
            + glob.glob(os.path.join(input_dir, "**", "*.TIF"), recursive=True)
        ))
        tif_files = [
            f for f in tif_files if not exclude_pat.search(os.path.basename(f))
        ]

    return tif_files


# ============================================================================
# ФУНКЦИЯ: Загрузка эталонных полигонов (интерактивный выбор SHP + поля)
# ============================================================================

def load_reference_polygons(
    input_dir: str, target_crs: CRS
) -> list[dict] | None:
    print("\n════════════════════════════════════════════════════════")
    print("ЗАГРУЗКА ЭТАЛОННЫХ ПОЛИГОНОВ")
    print("════════════════════════════════════════════════════════\n")

    shp_files = sorted(
        glob.glob(os.path.join(input_dir, "**", "*.shp"), recursive=True)
    )
    if not shp_files:
        print("SHP-файлы не найдены в INPUT. Пропускаю анализ эталонов.\n")
        return None

    print("Найдены SHP-файлы:\n")
    for i, f in enumerate(shp_files, 1):
        rel = os.path.relpath(f, input_dir)
        print(f"  [{i}] {rel}")
    print("\n  [0] Пропустить (не использовать эталоны)\n")

    choice_str = input("Выберите номер файла с эталонными полигонами: ").strip()
    try:
        choice = int(choice_str)
    except ValueError:
        choice = 0

    if choice == 0:
        print("\nЭталоны не выбраны.\n")
        return None
    if choice < 1 or choice > len(shp_files):
        print("\n⚠ Некорректный номер.\n")
        return None

    shp_path = shp_files[choice - 1]
    print(f"\nЗагружаю: {os.path.basename(shp_path)}")

    try:
        ref_gdf = gpd.read_file(shp_path)
    except Exception as e:
        print(f"✗ Ошибка: {e}\n")
        return None

    print(
        f"  Геометрия: {ref_gdf.geom_type.unique()}, объектов: {len(ref_gdf)}\n"
    )

    # Перевод в целевую проекцию
    if ref_gdf.crs != target_crs:
        ref_gdf = ref_gdf.to_crs(target_crs)

    attr_names = [c for c in ref_gdf.columns if c != "geometry"]
    if not attr_names:
        print("⚠ Нет атрибутивных полей. Все полигоны = класс 'REF'.\n")
        return [{"class_name": "REF", "gdf": ref_gdf, "file": os.path.basename(shp_path)}]

    print("Атрибутивные поля:\n")
    for i, nm in enumerate(attr_names, 1):
        uv = ref_gdf[nm].unique()
        preview = ", ".join(str(v) for v in uv[:6])
        if len(uv) > 6:
            preview += ", ..."
        print(f"  [{i}] {nm:<20s} ({len(uv)} уник.: {preview})")

    fc_str = input("\nВыберите номер поля с названиями классов: ").strip()
    try:
        fc = int(fc_str)
    except ValueError:
        fc = 1

    if fc < 1 or fc > len(attr_names):
        print("\n⚠ Используется первое поле.")
        fc = 1
    class_field = attr_names[fc - 1]
    print(f"\n✓ Поле классов: '{class_field}'\n")

    unique_classes = sorted(ref_gdf[class_field].dropna().unique(), key=str)
    print(f"Классов: {len(unique_classes)}")

    ref_list = []
    for cls in unique_classes:
        sub = ref_gdf[ref_gdf[class_field] == cls]
        ref_list.append(
            {
                "class_name": str(cls),
                "gdf": sub,
                "file": os.path.basename(shp_path),
            }
        )
        print(f"  ✓ '{cls}': {len(sub)} полигонов")

    print(
        f"\n✓ Загружено: {len(ref_list)} классов ({len(ref_gdf)} полигонов)\n"
    )
    return ref_list


# ============================================================================
# ТАБЛИЦА ПИКСЕЛЕЙ ЭТАЛОНОВ
# ============================================================================

def create_reference_points_table(
    ref_info: list[dict] | None,
    composite_path: str,
    band_names: list[str],
) -> pd.DataFrame | None:
    if not ref_info:
        return None

    print("\n════════════════════════════════════════════════════════")
    print("ИЗВЛЕЧЕНИЕ ПИКСЕЛЕЙ ПО ЭТАЛОНАМ")
    print("════════════════════════════════════════════════════════\n")

    all_rows = []
    with rasterio.open(composite_path) as src:
        for ref in ref_info:
            cls = ref["class_name"]
            gdf = ref["gdf"]
            cls_count = 0
            try:
                for _, row_geom in gdf.iterrows():
                    geom = [mapping(row_geom.geometry)]
                    out_img, out_tf = rio_mask(
                        src, geom, crop=True, all_touched=True, nodata=np.nan
                    )
                    # out_img: (bands, h, w)
                    # Пиксель валиден, если хотя бы один слой содержит конечное значение
                    mask_valid = np.any(np.isfinite(out_img), axis=0)
                    coords = np.argwhere(mask_valid)
                    for rc in coords:
                        vals = out_img[:, rc[0], rc[1]]
                        row_dict = {bn: float(vals[bi]) for bi, bn in enumerate(band_names)}
                        row_dict["class_name"] = cls
                        all_rows.append(row_dict)
                        cls_count += 1
                print(f"  ✓ '{cls}': {cls_count} пикселей")
            except Exception as e:
                print(f"  ✗ '{cls}': {e}")

    if not all_rows:
        print("\n⚠ Не извлечено ни одного пикселя! Проверьте пересечение маски и эталонных полигонов.\n")
        return None

    ref_table = pd.DataFrame(all_rows)
    n_classes = ref_table["class_name"].nunique()
    print(f"\n✓ Итого: {len(ref_table)} пикселей, {n_classes} классов")

    # Проверка количества NaN по слоям
    layer_cols = [c for c in ref_table.columns if c != "class_name"]
    na_counts = ref_table[layer_cols].isna().sum()
    na_nonzero = na_counts[na_counts > 0]
    if len(na_nonzero) > 0:
        print(f"  ℹ Слоёв с NaN в эталонах: {len(na_nonzero)}")
        for col, cnt in na_nonzero.items():
            pct = 100.0 * cnt / len(ref_table)
            print(f"      {col}: {cnt} NaN ({pct:.1f}%)")
    print()
    return ref_table


# ============================================================================
# МАТРИЦА КОРРЕЛЯЦИЙ СПИРМЕНА → PNG + CSV
# ============================================================================

def plot_spearman_corr(
    data: pd.DataFrame,
    layer_cols: list[str],
    out_png: str,
    out_csv: str,
    title_text: str,
) -> np.ndarray | None:
    print(f"  Матрица Спирмена ({len(layer_cols)} переменных)...")
    sub = data[layer_cols].dropna()
    if len(sub) < 10:
        print("  ⚠ Недостаточно данных\n")
        return None

    corr_mat = sub.corr(method="spearman").values
    n_vars = len(layer_cols)

    # CSV
    df_corr = pd.DataFrame(corr_mat, index=layer_cols, columns=layer_cols)
    df_corr.to_csv(out_csv)
    print(f"  ✓ {os.path.basename(out_csv)}")

    # PNG
    fig_size = max(8, n_vars * 0.45 + 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.95))
    mask_upper = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)

    annot = n_vars <= 40
    font_size = max(4, min(8, 140 / n_vars))

    sns.heatmap(
        df_corr,
        mask=mask_upper,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        annot=annot,
        fmt=".2f" if annot else "",
        annot_kws={"size": font_size},
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title_text, fontsize=12)
    tick_size = max(5, min(9, 160 / n_vars))
    ax.tick_params(axis="both", labelsize=tick_size)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {os.path.basename(out_png)}\n")

    return corr_mat


# ============================================================================
# ДОВЕРИТЕЛЬНЫЙ ИНТЕРВАЛ МЕДИАНЫ (bootstrap)
# ============================================================================

def calculate_ci_median(x: np.ndarray, conf: float = 0.95, R: int = 1000):
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return {"lower": np.nan, "median": np.nanmedian(x), "upper": np.nan}
    rng = np.random.RandomState(42)
    bm = np.array([np.median(rng.choice(x, size=len(x), replace=True)) for _ in range(R)])
    alpha = (1 - conf) / 2
    lo, med, hi = np.quantile(bm, [alpha, 0.5, 1 - alpha])
    return {"lower": lo, "median": med, "upper": hi}


# ============================================================================
# ТЕСТЫ КРАСКЕЛА-УОЛЛИСА
# ============================================================================

def perform_kruskal_wallis_tests(ref_table: pd.DataFrame) -> pd.DataFrame | None:
    print("\n════════════════════════════════════════════════════════")
    print("ТЕСТЫ КРАСКЕЛА-УОЛЛИСА")
    print("════════════════════════════════════════════════════════\n")

    service_cols = {"class_name", "ID", "id", "geometry"}
    layer_cols = [
        c for c in ref_table.columns
        if c not in service_cols and pd.api.types.is_numeric_dtype(ref_table[c])
    ]
    if not layer_cols or "class_name" not in ref_table.columns:
        print("Недостаточно данных.\n")
        return None

    classes = ref_table["class_name"].unique()
    results = []
    for lc in layer_cols:
        groups = [
            ref_table.loc[ref_table["class_name"] == cls, lc].dropna().values
            for cls in classes
        ]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            continue
        try:
            stat, p = sp_stats.kruskal(*groups)
            if p < 0.001:
                sig = "***"
            elif p < 0.01:
                sig = "**"
            elif p < 0.05:
                sig = "*"
            else:
                sig = "ns"
            results.append(
                {"layer": lc, "statistic": round(stat, 4), "p_value": p, "significant": sig}
            )
        except Exception:
            pass

    if not results:
        print("Нет результатов.\n")
        return None

    df = pd.DataFrame(results)
    n_sig = (df["significant"] != "ns").sum()
    print(f"✓ Тестов: {len(df)}, значимых: {n_sig}\n")
    return df


# ============================================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ KW
# ============================================================================

def save_kruskal_wallis_results(kw_results: pd.DataFrame | None, output_dir: str):
    if kw_results is None or kw_results.empty:
        return
    path = os.path.join(output_dir, "Kruskal_Wallis_results.csv")
    kw_results.to_csv(path, index=False)
    print(f"✓ KW: {os.path.basename(path)}\n")


# ============================================================================
# ГРАФИКИ: МЕДИАНА + CI ПО КЛАССАМ
# ============================================================================

def plot_boxplots_by_class(ref_table: pd.DataFrame, plots_dir: str):
    print("\n════════════════════════════════════════════════════════")
    print("ГРАФИКИ: МЕДИАНА + ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ ПО КЛАССАМ")
    print("════════════════════════════════════════════════════════\n")
    os.makedirs(plots_dir, exist_ok=True)

    service_cols = {"class_name", "ID", "id", "geometry"}
    layer_cols = [
        c for c in ref_table.columns
        if c not in service_cols and pd.api.types.is_numeric_dtype(ref_table[c])
    ]
    if not layer_cols:
        return

    classes = sorted(ref_table["class_name"].unique(), key=str)
    n_classes = len(classes)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_classes, 10)))

    for lc in layer_cols:
        stats_rows = []
        for cls in classes:
            vals = ref_table.loc[ref_table["class_name"] == cls, lc].dropna().values
            n = len(vals)
            med = np.median(vals) if n > 0 else np.nan
            if n >= 3:
                rng = np.random.RandomState(42)
                bm = np.array(
                    [np.median(rng.choice(vals, size=n, replace=True)) for _ in range(2000)]
                )
                ci95 = np.quantile(bm, [0.025, 0.975])
                ci99 = np.quantile(bm, [0.005, 0.995])
            else:
                ci95 = [med, med]
                ci99 = [med, med]
            stats_rows.append(
                {
                    "class": cls,
                    "median": med,
                    "ci95_lo": ci95[0],
                    "ci95_hi": ci95[1],
                    "ci99_lo": ci99[0],
                    "ci99_hi": ci99[1],
                    "n": n,
                }
            )

        sdf = pd.DataFrame(stats_rows)
        fig_w = max(5, n_classes * 0.8 + 2)
        fig, ax = plt.subplots(figsize=(fig_w, 4))

        x = np.arange(len(sdf))
        for j, row in sdf.iterrows():
            ax.plot(
                [j, j],
                [row["ci99_lo"], row["ci99_hi"]],
                color=colors[j % len(colors)],
                linewidth=2.5,
                alpha=0.3,
            )
            ax.plot(
                [j, j],
                [row["ci95_lo"], row["ci95_hi"]],
                color=colors[j % len(colors)],
                linewidth=5,
                alpha=0.5,
            )
            ax.plot(j, row["median"], "ok", markersize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(sdf["class"], rotation=45, ha="right", fontsize=8)
        ax.set_title(lc, fontsize=10)
        ax.set_xlabel("Class")
        ax.set_ylabel("Value")
        fig.suptitle(
            "Median ± CI 0.95 (bold) & CI 0.99 (thin)",
            fontsize=7,
            color="grey",
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, f"ci_{lc}.png"), dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

    print(f"✓ Графиков (медиана + CI): {len(layer_cols)}\n")


# ============================================================================
# СВОДНАЯ ДИАГРАММА KW
# ============================================================================

def plot_kw_summary(kw_results: pd.DataFrame | None, output_dir: str):
    if kw_results is None or kw_results.empty:
        return
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df = kw_results.copy()
    df["log_p"] = -np.log10(np.maximum(df["p_value"].values, 1e-300))

    fig_h = max(4, len(df) * 0.22 + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    color_map = {"***": "#d62728", "**": "#ff7f0e", "*": "#2ca02c", "ns": "#7f7f7f"}
    bar_colors = [color_map.get(s, "#7f7f7f") for s in df["significant"]]

    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["log_p"].values, color=bar_colors, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["layer"].values, fontsize=7)
    ax.axvline(-np.log10(0.05), color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("-log10(p)")
    ax.set_title("Kruskal-Wallis significance")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(
        os.path.join(plots_dir, "Kruskal_Wallis_summary.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"✓ KW summary: Kruskal_Wallis_summary.png\n")


# ============================================================================
# ФИЛЬТРАЦИЯ СЛОЁВ ПО КОРРЕЛЯЦИИ
# ============================================================================

def filter_layers_by_correlation(
    composite_path: str,
    band_names: list[str],
    output_dir: str,
    plots_dir: str,
    ref_table: pd.DataFrame | None = None,
    thresholds: list[float] | None = None,
) -> list[str]:
    """
    Анализирует межслойную корреляцию и позволяет пользователю
    выбрать порог для удаления коррелированных слоёв.
    Возвращает список имён слоёв, прошедших фильтрацию.
    """
    if thresholds is None:
        thresholds = [round(t, 1) for t in np.arange(0.9, 0.0, -0.1)]
    thresholds = sorted(set(thresholds), reverse=True)

    print("\n════════════════════════════════════════════════════════")
    print("АНАЛИЗ ВЗАИМНОЙ КОРРЕЛЯЦИИ СЛОЁВ")
    print("════════════════════════════════════════════════════════\n")

    n_layers = len(band_names)
    if n_layers < 2:
        print("Меньше двух слоёв, фильтрация по корреляции не требуется.\n")
        return band_names

    with rasterio.open(composite_path) as src:
        total_cells = src.width * src.height

    print(f"Всего слоёв в композите: {n_layers}")
    print(f"Всего пикселей в растре: {total_cells:,}\n".replace(",", " "))

    # --- Проверяем эталоны ---
    has_ref = False
    ref_layer_cols = []
    if ref_table is not None and len(ref_table) >= 10:
        service_cols = {"class_name", "ID", "id", "geometry"}
        ref_layer_cols = [
            c for c in ref_table.columns
            if c not in service_cols
            and pd.api.types.is_numeric_dtype(ref_table[c])
            and c in band_names
        ]
        if len(ref_layer_cols) >= 2:
            has_ref = True
            print(f"Эталонных пикселей:     {len(ref_table):,}\n".replace(",", " "))

    # --- Выбор источника данных ---
    print("════════════════════════════════════════════════════════")
    print("ИСТОЧНИК ДАННЫХ ДЛЯ РАСЧЁТА КОРРЕЛЯЦИИ")
    print("════════════════════════════════════════════════════════\n")

    if has_ref:
        print("  [1] Все пиксели растра (случайная выборка)")
        print("  [2] Только эталонные пиксели")
        print("  [3] Оба варианта (две матрицы, фильтрация по выбранной)\n")
        src_choice = input("Выберите вариант (Enter = 1): ").strip()
        if src_choice == "":
            src_choice = "1"
        try:
            src_num = int(src_choice)
        except ValueError:
            src_num = 1
        if src_num < 1 or src_num > 3:
            print("⚠ Некорректный выбор. Используется вариант 1 (все пиксели).\n")
            src_num = 1
    else:
        print("Эталоны не загружены или недостаточно данных.")
        print("Корреляция будет рассчитана по всем пикселям растра.\n")
        src_num = 1

    # --- Вспомогательные ---
    def _estimate_time(n, p):
        t_sec = (n / 100_000) * (p / 20) ** 2
        t_sec = max(t_sec, 0.1)
        return f"~{t_sec:.0f} сек" if t_sec < 60 else f"~{t_sec / 60:.1f} мин"

    def _ask_sample_size(n_total, label):
        print(f"\n── Объём выборки для: {label} ──\n")
        print(f"  Доступно пикселей: {n_total:,}".replace(",", " "))
        print(f"  Слоёв: {n_layers}\n")

        sample_opts = sorted(set(
            [s for s in [1000, 10_000, 100_000, n_total] if s <= n_total]
        ))

        print("  Варианты:\n")
        default_idx = 0
        for si, n_eff in enumerate(sample_opts, 1):
            est = _estimate_time(n_eff, n_layers)
            tag = ""
            if n_eff == n_total:
                tag = "  <<< весь объём"
            elif n_eff == 100_000 and n_total > 100_000:
                tag = "  <<< по умолчанию"
                default_idx = si
            print(f"    [{si}]  {n_eff:>12,} пикселей   ({est}){tag}".replace(",", " "))
        custom_idx = len(sample_opts) + 1
        print(f"    [{custom_idx}]  Ввести своё значение\n")

        if default_idx == 0:
            default_idx = len(sample_opts)

        raw = input(f"Выберите вариант (Enter = {default_idx}): ").strip()
        if raw == "":
            n_use = sample_opts[default_idx - 1]
        else:
            try:
                sc = int(raw)
                if 1 <= sc <= len(sample_opts):
                    n_use = sample_opts[sc - 1]
                elif sc == custom_idx:
                    cust = input("Введите количество пикселей: ").strip()
                    try:
                        cn = int(cust)
                        n_use = min(max(cn, 50), n_total)
                    except ValueError:
                        print("⚠ Некорректное значение. Используется весь объём.")
                        n_use = n_total
                else:
                    # попробовать как прямое число
                    n_use = min(max(sc, 50), n_total) if sc >= 50 else n_total
            except ValueError:
                print("⚠ Некорректный выбор. Используется весь объём.")
                n_use = n_total

        est_f = _estimate_time(n_use, n_layers)
        print(
            f"\n✓ Объём выборки: {n_use:,} из {n_total:,} пикселей "
            f"(оценка: {est_f})\n".replace(",", " ")
        )
        return n_use

    def _sample_raster_vals(composite_path, max_samples, layer_names):
        """Случайная выборка пикселей из многоканального растра."""
        print("Формирую выборку пикселей из растра...")
        with rasterio.open(composite_path) as src:
            h, w = src.height, src.width
            tc = h * w
            n_smp = min(max_samples, tc)
            rng = np.random.RandomState(1)
            idx = rng.choice(tc, size=n_smp, replace=False)
            rows = idx // w
            cols = idx % w

            data = {}
            for band_idx, bn in enumerate(layer_names, 1):
                arr = src.read(band_idx)
                data[bn] = arr[rows, cols].astype(np.float64)

        df = pd.DataFrame(data)
        return df

    def _sample_ref_vals(ref_table, max_samples, layer_cols):
        print("Формирую выборку из эталонных пикселей...")
        if max_samples < len(ref_table):
            df = ref_table[layer_cols].sample(n=max_samples, random_state=42)
        else:
            df = ref_table[layer_cols].copy()
        return df

    def _compute_and_plot_corr(vals, used_cols, suffix, title_tag):
        # Убираем столбцы, где ВСЕ NA
        all_na = vals.isna().all()
        if all_na.any():
            bad = list(all_na[all_na].index)
            print(f"  ⚠ Удалены нечисловые слои ({len(bad)}): {', '.join(bad)}")
            vals = vals.drop(columns=bad)
            used_cols = list(vals.columns)
        vals = vals.dropna()
        print(f"Пикселей после удаления NA: {len(vals)}\n")
        if len(vals) < 10:
            print("⚠ Слишком мало непустых пикселей. Пропускаю.\n")
            return None

        corr_mat = vals.corr(method="spearman").values
        col_names = list(vals.columns)

        csv_name = f"Spearman_layers_{suffix}.csv"
        csv_path = os.path.join(output_dir, csv_name)
        pd.DataFrame(corr_mat, index=col_names, columns=col_names).to_csv(csv_path)
        print(f"✓ Матрица корреляций ({title_tag}): {csv_name}")

        png_name = f"Spearman_layers_{suffix}.png"
        png_path = os.path.join(plots_dir, png_name)
        n_v = len(col_names)
        fig_sz = max(8, n_v * 0.45 + 2)
        fig, ax = plt.subplots(figsize=(fig_sz, fig_sz * 0.95))
        mask_upper = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
        annot = n_v <= 40
        fsize = max(4, min(8, 140 / n_v))

        df_c = pd.DataFrame(corr_mat, index=col_names, columns=col_names)
        sns.heatmap(
            df_c,
            mask=mask_upper,
            cmap="RdBu_r",
            vmin=-1, vmax=1,
            annot=annot,
            fmt=".2f" if annot else "",
            annot_kws={"size": fsize},
            square=True,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(f"Spearman correlation ({title_tag})", fontsize=10)
        tick_sz = max(5, min(9, 160 / n_v))
        ax.tick_params(axis="both", labelsize=tick_sz)
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ PNG: {png_name}\n")

        return corr_mat, col_names

    def _compute_threshold_options(corr_mat, filt_cols, thresholds):
        n_filt = len(filt_cols)
        keep_lists = {}
        n_layers_vec = []
        for thr in thresholds:
            selected = []
            for i in range(n_filt):
                cur = filt_cols[i]
                if not selected:
                    selected.append(cur)
                else:
                    ci = filt_cols.index(cur)
                    max_r = max(
                        abs(corr_mat[ci, filt_cols.index(s)]) for s in selected
                    )
                    if max_r <= thr:
                        selected.append(cur)
            keep_lists[str(thr)] = selected
            n_layers_vec.append(len(selected))
        return {"keep_lists": keep_lists, "n_layers": n_layers_vec}

    # ═══════════════════════════════════════════════
    # ОСНОВНОЙ РАСЧЁТ
    # ═══════════════════════════════════════════════

    corr_all = None
    corr_ref = None
    cols_all = band_names[:]
    cols_ref = ref_layer_cols[:] if has_ref else []

    # Режим 1 или 3: все пиксели
    if src_num in (1, 3):
        n_smp = _ask_sample_size(total_cells, "все пиксели растра")
        vals_all = _sample_raster_vals(composite_path, n_smp, band_names)
        result = _compute_and_plot_corr(vals_all, band_names, "all_pixels", "все пиксели")
        if result is not None:
            corr_all, cols_all = result
        del vals_all

    # Режим 2 или 3: эталоны
    if src_num in (2, 3) and has_ref:
        n_smp_ref = _ask_sample_size(len(ref_table), "эталонные пиксели")
        vals_ref = _sample_ref_vals(ref_table, n_smp_ref, ref_layer_cols)
        result = _compute_and_plot_corr(vals_ref, ref_layer_cols, "reference", "эталоны")
        if result is not None:
            corr_ref, cols_ref = result
        del vals_ref

    # ═══════════════════════════════════════════════
    # ПОРОГОВАЯ ФИЛЬТРАЦИЯ
    # ═══════════════════════════════════════════════

    keep_names = band_names  # по умолчанию — без фильтрации

    if src_num == 3 and corr_all is not None and corr_ref is not None:
        # --- Режим 3: обе матрицы ---
        print("Рассчитываю варианты фильтрации по обеим матрицам...\n")
        res_all = _compute_threshold_options(corr_all, cols_all, thresholds)
        res_ref = _compute_threshold_options(corr_ref, cols_ref, thresholds)

        print("════════════════════════════════════════════════════════")
        print("СРАВНЕНИЕ ВАРИАНТОВ ФИЛЬТРАЦИИ")
        print("════════════════════════════════════════════════════════\n")
        print(f"  Слоёв в композите: {len(cols_all)}\n")
        print(f"  {'Порог |r|':<10s}  {'Все пиксели':>14s}  {'Эталоны':>14s}")
        print(f"  {'──────────':<10s}  {'──────────────':>14s}  {'──────────────':>14s}")
        for k, thr in enumerate(thresholds):
            print(f"  > {thr:<8.2f}  {res_all['n_layers'][k]:>14d}  {res_ref['n_layers'][k]:>14d}")
        print()

        filt_csv = os.path.join(output_dir, "Composite_filtering_options.csv")
        pd.DataFrame({
            "threshold": thresholds,
            "all_pixels": res_all["n_layers"],
            "reference": res_ref["n_layers"],
        }).to_csv(filt_csv, index=False)
        print(f"✓ Таблица вариантов фильтрации: {os.path.basename(filt_csv)}\n")

        print("Выберите порог корреляции для формирования Composite_filtered:")
        print(" (Enter = пропустить фильтрацию, оставить исходный Composite)\n")
        for k, thr in enumerate(thresholds, 1):
            print(
                f" [{k}] |r| > {thr:.2f}  →  все: {res_all['n_layers'][k - 1]} слоёв  |  "
                f"эталоны: {res_ref['n_layers'][k - 1]} слоёв"
            )
        print()

        choice = input("Номер варианта (или Enter для пропуска): ").strip()
        if choice == "":
            print("\nФильтрация по взаимной корреляции не выбрана. Используется полный композит.\n")
            return band_names

        try:
            cn = int(choice)
        except ValueError:
            cn = -1
        if cn < 1 or cn > len(thresholds):
            print("\n⚠ Некорректный выбор. Фильтрация пропущена.\n")
            return band_names

        thr_sel = thresholds[cn - 1]
        keep_all_sel = res_all["keep_lists"][str(thr_sel)]
        keep_ref_sel = res_ref["keep_lists"][str(thr_sel)]

        print(f"\nВыбран порог |r| > {thr_sel:.2f}.")
        print(f"  Все пиксели: {len(keep_all_sel)} слоёв | Эталоны: {len(keep_ref_sel)} слоёв\n")

        # Таблица: все слои с отметкой вхождения в каждую из двух матриц
        max_name_len = max(len(x) for x in cols_all + ["Растровый слой"])
        hdr = f"  {'Растровый слой':<{max_name_len}s}  {'Все пиксели':>14s}  {'Эталоны':>14s}"
        sep = f"  {'─' * max_name_len}  {'─' * 14}  {'─' * 14}"
        print(hdr)
        print(sep)
        for ln in cols_all:
            in_all = "✓" if ln in keep_all_sel else "–"
            in_ref  = "✓" if ln in keep_ref_sel  else "–"
            print(f"  {ln:<{max_name_len}s}  {in_all:>14s}  {in_ref:>14s}")
        print()

        print("  [1] Фильтровать по всем пикселям")
        print("  [2] Фильтровать по эталонам\n")
        src_ch = input("Выберите вариант (Enter = 1): ").strip()
        if src_ch == "2":
            keep_names = keep_ref_sel
            print("\n✓ Фильтрация по матрице эталонов.")
        else:
            keep_names = keep_all_sel
            print("\n✓ Фильтрация по матрице всех пикселей.")

    else:
        # --- Режим 1 или 2: одна матрица ---
        if corr_ref is not None:
            corr_mat = corr_ref
            filt_cols = cols_ref
        elif corr_all is not None:
            corr_mat = corr_all
            filt_cols = cols_all
        else:
            print("⚠ Не удалось рассчитать матрицу корреляций. Пропускаю.\n")
            return band_names

        res = _compute_threshold_options(corr_mat, filt_cols, thresholds)

        print("Варианты числа слоёв при разных порогах |r|:")
        for k, thr in enumerate(thresholds):
            print(f"  > {thr:.2f}  → {res['n_layers'][k]} слоёв")
        print()

        filt_csv = os.path.join(output_dir, "Composite_filtering_options.csv")
        pd.DataFrame({
            "threshold": thresholds,
            "n_layers": res["n_layers"],
        }).to_csv(filt_csv, index=False)
        print(f"✓ Таблица вариантов фильтрации: {os.path.basename(filt_csv)}\n")

        print("Выберите порог корреляции для формирования Composite_filtered:")
        print(" (Enter = пропустить фильтрацию, оставить исходный Composite)\n")
        for k, thr in enumerate(thresholds, 1):
            print(f" [{k}] |r| > {thr:.2f}  → {res['n_layers'][k - 1]} слоёв")
        print()

        choice = input("Номер варианта (или Enter для пропуска): ").strip()
        if choice == "":
            print("\nФильтрация по взаимной корреляции не выбрана. Используется полный композит.\n")
            return band_names

        try:
            cn = int(choice)
        except ValueError:
            cn = -1
        if cn < 1 or cn > len(thresholds):
            print("\n⚠ Некорректный выбор. Фильтрация пропущена.\n")
            return band_names

        thr_sel = thresholds[cn - 1]
        keep_names = res["keep_lists"][str(thr_sel)]

    # Финальный вывод
    print(f"\nВ итоговый Composite_filtered войдёт {len(keep_names)} слоёв:")
    for kn in keep_names:
        print(f"  - {kn}")
    print()

    return keep_names


# ============================================================================
# ОСНОВНОЙ СКРИПТ (main)
# ============================================================================

def main():
    print("\n════════════════════════════════════════════════════════")
    print("Объединение TIF растров в один файл")
    print("ВЕРСИЯ v62 (Python) — ZSTD+PREDICTOR=3, потоковое объединение")
    print("С ВЫБОРОМ ПРОЕКЦИИ, МАСКОЙ, КОРРЕЛЯЦИЯМИ, KRUSKAL-WALLIS И ФИЛЬТРАЦИЕЙ СЛОЁВ")
    print("════════════════════════════════════════════════════════\n")

    root_dir = os.getcwd()
    print(f"Директория: {root_dir}\n")

    # v3.1: увеличиваем блочный кеш GDAL — критично для многослойных
    # растров. По умолчанию ~5% RAM (обычно ~200-500 МБ), при
    # обработке стеков S2-индексов кеш постоянно вытесняется и
    # блоки перечитываются заново → CPU падает до 5%, диск активен,
    # но полезной работы почти нет. 4 ГБ хватает на большинство сцен.
    os.environ["GDAL_CACHEMAX"] = "4096"
    print(f"GDAL_CACHEMAX = 4096 МБ (ускорение чтения многослойных растров)\n")

    input_dir = os.path.join(root_dir, "INPUT")
    if not os.path.isdir(input_dir):
        print("INPUT папка не найдена. Создайте папку INPUT и поместите туда TIF и SHP файлы.")
        os.makedirs(input_dir, exist_ok=True)
        print(f"TIF- и SHP-файлы ожидаются в: {input_dir}")
        return

    base_output_dir = os.path.join(root_dir, "00_PREPARE_LAYERS")
    os.makedirs(base_output_dir, exist_ok=True)

    # счётчик запусков
    counter_file = os.path.join(base_output_dir, "prepare_run_counter.txt")
    run_id = 0
    if os.path.exists(counter_file):
        try:
            run_id = int(open(counter_file).read().strip())
        except ValueError:
            run_id = 0
    run_id += 1
    with open(counter_file, "w") as f:
        f.write(str(run_id))

    from datetime import datetime

    suffix = f"_v{run_id:03d}"
    run_stamp = datetime.now().strftime("%d%m%y_%H%M")
    run_folder = f"{run_stamp}{suffix}"

    output_dir = os.path.join(base_output_dir, run_folder)
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # v3.1: TEMP выносим за пределы Yandex.Disk / OneDrive, иначе
    # облачный клиент перехватывает I/O и запись aligned-файлов
    # блокируется до завершения выгрузки. Если I:\ — SSD, оставляем
    # там же, но в корне диска.
    temp_dir = r"I:\TEMP_terra_align"
    os.makedirs(temp_dir, exist_ok=True)

    print(f"INPUT : {input_dir}")
    print(f"OUTPUT: {output_dir}")
    print(f"PLOTS : {plots_dir}")
    print(f"TEMP  : {temp_dir}\n")

    # Удаление старых Composite*.tif из корня и output
    import re as _re

    for d in [root_dir, base_output_dir]:
        for f in os.listdir(d):
            if _re.match(r"Composite.*\.tif$", f, _re.IGNORECASE):
                try:
                    os.remove(os.path.join(d, f))
                except OSError:
                    pass

    # Потоки
    print("════════════════════════════════════════════════════════")
    print("КОЛИЧЕСТВО ПОТОКОВ (ПАРАЛЛЕЛИЗАЦИЯ)")
    print("════════════════════════════════════════════════════════\n")

    n_cores_available = cpu_count()
    n_cores_physical = os.cpu_count() or 1
    print(f" Доступно ядер (логических): {n_cores_available}")
    print(f" Доступно ядер (физических): {n_cores_physical}\n")

    recommended = max(1, n_cores_physical - 1)
    print(f" Рекомендуемое количество потоков: {recommended}")
    print(" (физические ядра - 1, чтобы система оставалась отзывчивой)\n")

    n_threads_input = input(
        f"Введите количество потоков (Enter = {recommended}): "
    ).strip()
    if n_threads_input == "":
        n_threads = recommended
    else:
        try:
            n_threads = int(n_threads_input)
            if n_threads < 1:
                n_threads = recommended
        except ValueError:
            print("\n⚠ Некорректное значение. Используется рекомендуемое.")
            n_threads = recommended

    if n_threads > n_cores_available:
        print(
            f"\n⚠ Указано больше доступных ядер ({n_cores_available}). "
            f"Ограничиваю до {n_cores_available}."
        )
        n_threads = n_cores_available

    # Устанавливаем переменные окружения для GDAL
    os.environ["GDAL_NUM_THREADS"] = str(n_threads)
    print(f"\n✓ Установлено потоков: {n_threads}")
    print(f" GDAL_NUM_THREADS = {n_threads}\n")

    # --- Выбор проекции ---
    target_crs = select_target_crs(input_dir)

    # --- Маска ---
    mask_gdf = load_mask(input_dir, target_crs)

    # --- Поиск TIF ---
    print("════════════════════════════════════════════════════════")
    print("ПОИСК TIF ФАЙЛОВ")
    print("════════════════════════════════════════════════════════\n")
    print("Сканирую папки в поиске TIF файлов...")

    tif_files = find_tif_files(input_dir)
    if not tif_files:
        print("TIF файлы не найдены!")
        return

    print(f"Найдено {len(tif_files)} TIF файлов\n")
    print("Анализирую растры...\n")

    raster_info_list = [_read_raster_info(f) for f in tif_files]

    print("НАЙДЕННЫЕ РАСТРЫ:")
    print("─────────────────\n")
    for i, info in enumerate(raster_info_list, 1):
        print(f"[{i:2d}] {info['file']}")
        print(f" Путь: {info['path']}")
        if info["success"]:
            print(f" Слоёв: {info['layers']}\n")
        else:
            print(f" ОШИБКА: {info['error']}\n")

    valid_rasters = [r for r in raster_info_list if r["success"]]
    if not valid_rasters:
        print("Не удалось загрузить ни один растр!")
        return

    # --- Порядок соединения ---
    print("════════════════════════════════════════════════════════")
    print("ПОРЯДОК СОЕДИНЕНИЯ")
    print("════════════════════════════════════════════════════════\n")
    print("Введите номера растров в желаемом порядке (через запятую).")
    print("Например: 1,3,2 или просто Enter для порядка по умолчанию\n")

    order_input = input("Порядок: ").strip()
    if order_input == "":
        file_order = list(range(len(valid_rasters)))
    else:
        try:
            file_order = [int(x.strip()) - 1 for x in order_input.split(",")]
            if any(i < 0 or i >= len(valid_rasters) for i in file_order):
                raise ValueError
        except (ValueError, IndexError):
            print("Некорректные номера! Используется порядок по умолчанию.\n")
            file_order = list(range(len(valid_rasters)))

    ordered = [valid_rasters[i] for i in file_order]

    print("\nПорядок соединения:")
    for i, r in enumerate(ordered, 1):
        print(f" {i}. {r['file']} ({r['layers']} слой(ев))")

    # --- Названия слоёв ---
    print("\n════════════════════════════════════════════════════════")
    print("НАЗВАНИЯ СЛОЁВ")
    print("════════════════════════════════════════════════════════\n")

    total_layers = sum(r["layers"] for r in ordered)
    print(f"Всего будет {total_layers} слоёв\n")

    layer_names = []
    for r in ordered:
        n_ly = r["layers"]
        bnames = r.get("band_names", [])
        base_name = os.path.splitext(r["file"])[0]

        if n_ly == 1:
            auto = (
                bnames[0]
                if bnames and bnames[0] and bnames[0].lower() != "constant"
                else base_name
            )
            layer_names.append(auto if auto else base_name)
        else:
            for j in range(n_ly):
                if (
                    j < len(bnames)
                    and bnames[j]
                    and bnames[j].lower() != "constant"
                ):
                    layer_names.append(bnames[j])
                else:
                    layer_names.append(f"{base_name}_band{j + 1}")

    print("Автоматически определённые названия слоёв:")
    print("─────────────────────────────────────────────\n")

    idx = 0
    for r in ordered:
        n_ly = r["layers"]
        print(f"Растр: {r['file']} ({n_ly} слой(ев))")
        for j in range(n_ly):
            print(f" [{idx + 1:2d}] {layer_names[idx]}")
            idx += 1
        print()

    print("════════════════════════════════════════════════════════")
    print("Выберите действие:\n")
    print(" [Enter] = Принять все названия как есть")
    print(" [R]     = Переименовать отдельные слои\n")

    action = input("Ваш выбор (Enter/R): ").strip().upper()

    if action == "R":
        print("\nВведите новое название для слоя или Enter чтобы оставить текущее:\n")
        idx = 0
        for r in ordered:
            n_ly = r["layers"]
            print(f"Растр: {r['file']} ({n_ly} слой(ев))")
            for j in range(n_ly):
                print(f" [{idx + 1:2d}] Текущее: {layer_names[idx]}")
                new_name = input(" Новое (Enter = оставить): ").strip()
                if new_name:
                    layer_names[idx] = new_name
                    print(f"  → {new_name}")
                idx += 1
            print()
    else:
        print("\n✓ Названия приняты без изменений\n")

    # Уникальность
    if len(set(layer_names)) < len(layer_names):
        print(" ⚠ Обнаружены дублированные имена слоёв! Переименовываю...")
        seen = {}
        for i, nm in enumerate(layer_names):
            if nm in seen:
                seen[nm] += 1
                layer_names[i] = f"{nm}_{seen[nm]}"
            else:
                seen[nm] = 0
        print(f" ✓ Все {len(layer_names)} имён теперь уникальны")

    print("────────────────────────────────────────────")
    print("Итоговые названия слоёв:\n")
    for i, nm in enumerate(layer_names, 1):
        print(f" [{i:2d}] {nm}")
    print()

    # --- Загрузка и выравнивание ---
    print("════════════════════════════════════════════════════════")
    print("ЗАГРУЗКА И ВЫРАВНИВАНИЕ РАСТРОВ")
    print("════════════════════════════════════════════════════════\n")

    raster_paths = [r["path"] for r in ordered]
    alignment = align_rasters_to_common_grid(raster_paths, target_crs, temp_dir)
    aligned_files = alignment["aligned_files"]
    tpl = alignment["template_profile"]

    # --- Объединение ---
    print("════════════════════════════════════════════════════════")
    print("ОБЪЕДИНЕНИЕ РАСТРОВ (потоковое)")
    print("════════════════════════════════════════════════════════\n")

    output_name = "Composite"
    output_path = os.path.join(output_dir, f"{output_name}.tif")
    composite_path = output_path

    # v2: пишем композит послойно (band-by-band) — без полной загрузки
    # в RAM. Экономит десятки ГБ памяти (302 слоя × 6339×6116×4 б ≈ 47 ГБ).
    total_bands = sum(rasterio.open(af).count for af in aligned_files)
    if total_bands != len(layer_names):
        print(
            f" ⚠ Несоответствие: слоёв в файлах {total_bands}, имён {len(layer_names)}"
        )

    profile = tpl.copy()
    # v2: финальный композит — ZSTD+PREDICTOR=3+BAND interleave
    profile.update(
        count=total_bands,
        dtype="float32",
        compress="zstd",
        zstd_level=3,
        predictor=3,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        interleave="band",
        nodata=np.nan,
    )
    if _needs_bigtiff(total_bands, tpl["height"], tpl["width"]):
        profile["BIGTIFF"] = "YES"
        print(" ℹ Размер > 4 ГБ — используется BigTIFF")

    print("Сохранение композитного растра...")
    print(f" File: {os.path.basename(output_path)}")
    print(f" Datatype: float32 (FLT4S), слоёв: {total_bands}")

    # Маска применяется послойно (расстризуем однократно)
    mask_array_2d = None
    if mask_gdf is not None:
        from rasterio.features import geometry_mask
        mask_array_2d = geometry_mask(
            mask_gdf.geometry,
            out_shape=(tpl["height"], tpl["width"]),
            transform=tpl["transform"],
            invert=True,
        )

    out_band_idx = 1
    with rasterio.open(output_path, "w", **profile) as dst:
        for af_idx, af in enumerate(aligned_files):
            with rasterio.open(af) as src:
                for band in range(1, src.count + 1):
                    arr = src.read(band).astype(np.float32)
                    if mask_array_2d is not None:
                        arr = np.where(mask_array_2d, arr, np.nan).astype(np.float32)
                    dst.write(arr, out_band_idx)
                    if out_band_idx - 1 < len(layer_names):
                        dst.set_band_description(
                            out_band_idx, layer_names[out_band_idx - 1]
                        )
                    if out_band_idx % 20 == 0 or out_band_idx == total_bands:
                        print(f"   Слой {out_band_idx}/{total_bands}")
                    out_band_idx += 1

    print(f" ✓ Композит записан\n")

    file_size_mb = os.path.getsize(output_path) / (1024 ** 2)
    with rasterio.open(composite_path) as src:
        n_bands = src.count
    print("COMPOSITE RASTER:")
    print(f" File: {os.path.basename(output_path)}")
    print(f" Size: {file_size_mb:.2f} MB")
    print(f" Layers: {n_bands}")
    print(f" Datatype: float32 (FLT4S)")
    print(f" CRS: {target_crs}\n")

    # ═══════════════════════════════════════════════
    # ЭТАЛОНЫ, KRUSKAL-WALLIS
    # ═══════════════════════════════════════════════

    ref_info = load_reference_polygons(input_dir, target_crs)
    ref_table = create_reference_points_table(ref_info, composite_path, layer_names)

    if ref_table is not None:
        ref_csv = os.path.join(output_dir, "reference_points_table.csv")
        ref_table.to_csv(ref_csv, index=False)
        print(f"✓ Таблица эталонов: {os.path.basename(ref_csv)}\n")

        print(f"  Данных для KW и боксплотов: {len(ref_table):,} пикселей\n".replace(",", " "))

        kw_results = perform_kruskal_wallis_tests(ref_table)
        save_kruskal_wallis_results(kw_results, output_dir)
        plot_kw_summary(kw_results, output_dir)
        plot_boxplots_by_class(ref_table, plots_dir)

    # ═══════════════════════════════════════════════
    # АНАЛИЗ КОРРЕЛЯЦИИ И ФИЛЬТРАЦИЯ СЛОЁВ
    # ═══════════════════════════════════════════════

    keep_names = filter_layers_by_correlation(
        composite_path=composite_path,
        band_names=layer_names,
        output_dir=output_dir,
        plots_dir=plots_dir,
        ref_table=ref_table,
        thresholds=[round(t, 1) for t in np.arange(0.9, 0.0, -0.1)],
    )

    # Сохранение отфильтрованного композита
    filtered_path = os.path.join(output_dir, "Composite_filtered.tif")
    print("Сохраняю Composite_filtered...")

    keep_indices = [layer_names.index(nm) + 1 for nm in keep_names]  # 1-based

    with rasterio.open(composite_path) as src:
        prof = src.profile.copy()
        # v2: ZSTD + PREDICTOR=3 + BAND interleave
        prof.update(
            count=len(keep_indices),
            compress="zstd",
            zstd_level=3,
            predictor=3,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            interleave="band",
        )
        if _needs_bigtiff(len(keep_indices), src.height, src.width, str(prof.get("dtype", "float32"))):
            prof["BIGTIFF"] = "YES"
        with rasterio.open(filtered_path, "w", **prof) as dst:
            for new_idx, old_idx in enumerate(keep_indices, 1):
                dst.write(src.read(old_idx), new_idx)
                dst.set_band_description(new_idx, layer_names[old_idx - 1])

    print(f"✓ Composite_filtered сохранён: {os.path.basename(filtered_path)}")
    print(f"  Слоёв: {len(keep_names)}\n")

    print("════════════════════════════════════════════════════════")
    print("✓ ГОТОВО! Все результаты в папке:")
    print(f"  {output_dir}")
    print("════════════════════════════════════════════════════════\n")


# Запуск
if __name__ == "__main__":
    main()
