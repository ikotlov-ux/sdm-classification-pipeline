# ============================================================================
# 01_prepare_data_v12.py
# СКРИПТ 01: ПОДГОТОВКА ДАННЫХ И ИЗВЛЕЧЕНИЕ ОБУЧАЮЩИХ ОБРАЗЦОВ
#
# Функционал:
# - Читает настройки из 00_PREPARE_LAYERS (pipelineconfig.pkl)
# - Выбор растра (Composite / Composite_filtered)
# - Загрузка обучающих полигонов (SHP)
# - Извлечение пиксельных значений по полигонам
# - Сохранение в GeoPackage, CSV и raster_info.pkl
# - Нормализация НЕ выполняется (z-score делается в sklearn preProcess)
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
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n============================")
    print("  01 - PREPARE DATA")
    print("============================\n")

    root_dir = os.getcwd()

    prev_dir = os.path.join(root_dir, "00_PREPARE_LAYERS")
    if not os.path.isdir(prev_dir):
        raise FileNotFoundError(
            "Папка 00_PREPARE_LAYERS не найдена. "
            "Сначала запустите 00_Prepare_Layers_v61.py"
        )

    # --- выбор подпапки внутри 00_PREPARE_LAYERS ---
    prev_subdirs = sorted(
        [
            os.path.join(prev_dir, d)
            for d in os.listdir(prev_dir)
            if os.path.isdir(os.path.join(prev_dir, d))
        ]
    )
    if not prev_subdirs:
        raise FileNotFoundError("В 00_PREPARE_LAYERS нет подпапок с результатами.")

    print("Доступные запуски в 00_PREPARE_LAYERS:\n")
    for i, sd in enumerate(prev_subdirs, 1):
        print(f" [{i}] {os.path.basename(sd)}")

    choice_raw = input("\nВыберите номер запуска из 00_PREPARE_LAYERS: ").strip()
    if choice_raw == "":
        choice_raw = "1"
    try:
        choice = int(choice_raw)
    except ValueError:
        raise ValueError("Некорректный выбор запуска.")

    if choice < 1 or choice > len(prev_subdirs):
        raise ValueError("Некорректный выбор запуска.")

    selected_prev_dir = prev_subdirs[choice - 1]
    selected_prev_name = os.path.basename(selected_prev_dir)

    # извлекаем суффикс "_vXXX" из имени папки
    m = re.search(r"(_v\d{3})$", selected_prev_name)
    selected_suffix = m.group(1) if m else "_v001"

    print(f"Выбрана папка: {selected_prev_name}, суффикс: {selected_suffix}\n")

    # --- 01_PREPARE_DATA: создание подпапки ---
    base_output_dir = os.path.join(root_dir, "01_PREPARE_DATA")
    os.makedirs(base_output_dir, exist_ok=True)

    run_stamp = datetime.now().strftime("%d%m%y_%H%M")
    run_folder_name = f"{run_stamp}{selected_suffix}"
    output_dir = os.path.join(base_output_dir, run_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Prev : {selected_prev_dir}")
    print(f"Out  : {output_dir}")

    # ================================================================
    # ЗАГРУЗКА НАСТРОЕК ПАЙПЛАЙНА (pipelineconfig.pkl)
    # ================================================================
    config = None

    # Пробуем pkl (Python), затем rds (если есть конвертер)
    config_path_pkl = os.path.join(selected_prev_dir, "pipelineconfig.pkl")
    config_path_rds = os.path.join(selected_prev_dir, "pipelineconfig.rds")

    if os.path.exists(config_path_pkl):
        with open(config_path_pkl, "rb") as f:
            config = pickle.load(f)
        print("✓ pipelineconfig.pkl")
        print(f"CRS: {config.get('target_crs', '?')}")
        layer_names_cfg = config.get("layer_names", [])
        print(f"Слоёв: {config.get('nlayers', '?')}; имена: {', '.join(layer_names_cfg)}")
        if config.get("ref_shp_file"):
            print(f"REF SHP: {config['ref_shp_file']}")
            print(f"REF CLASS FIELD: {config.get('ref_class_field', '?')}")
    elif os.path.exists(config_path_rds):
        print(
            "! pipelineconfig.rds найден, но Python не читает .rds напрямую.\n"
            "  Некоторые параметры будут определены автоматически."
        )
    else:
        print("! pipelineconfig не найден в выбранной подпапке 00_PREPARE_LAYERS.")
        print("  Некоторые параметры будут определены автоматически.")

    # ================================================================
    # ШАГ 1: ВЫБОР РАСТРА ДЛЯ АНАЛИЗА
    # ================================================================
    print("\n================================================================")
    print("[ШАГ 1] ВЫБОР РАСТРА")
    print("================================================================\n")

    print("-- INPUT (TIF) --")
    tif_files = sorted(glob.glob(os.path.join(selected_prev_dir, "*.tif")))
    if not tif_files:
        raise FileNotFoundError("TIF не найдены в выбранной подпапке 00_PREPARE_LAYERS!")

    print(f"00_PREPARE_LAYERS ({selected_prev_name}):")
    for i, tf in enumerate(tif_files, 1):
        fsize = os.path.getsize(tf) / (1024 ** 2)
        try:
            with rasterio.open(tf) as src:
                nlyr = src.count
        except Exception:
            nlyr = "?"
        print(f" {i:2d}) {os.path.basename(tf)}  {nlyr}  {fsize:.2f} MB")

    choice_str = input("\nВыберите номер растра (Enter = 1): ").strip()
    if choice_str == "":
        choice_str = "1"
    try:
        choice = int(choice_str)
    except ValueError:
        print("! Некорректный выбор, используется первый файл.")
        choice = 1

    if choice < 1 or choice > len(tif_files):
        print("! Некорректный выбор, используется первый файл.")
        choice = 1

    selected_tif = tif_files[choice - 1]
    print(f"\n✓ Выбран: {os.path.basename(selected_tif)}\n")

    # Загрузка растра — читаем метаданные
    with rasterio.open(selected_tif) as src:
        n_layers = src.count
        width = src.width
        height = src.height
        raster_crs = src.crs
        band_descriptions = [
            src.descriptions[i] if src.descriptions[i] else f"band{i + 1}"
            for i in range(n_layers)
        ]

        print(f"  Слоёв: {n_layers}, Размер: {width} x {height} пикселей")
        crs_code = raster_crs.to_epsg() if raster_crs.to_epsg() else str(raster_crs)
        print(f"  Проекция: EPSG:{crs_code}")

        # Краткий обзор диапазонов значений
        print("\n  Диапазоны значений:")
        for b in range(1, n_layers + 1):
            arr = src.read(b)
            valid = arr[np.isfinite(arr)]
            if len(valid) > 0:
                mn, mx = valid.min(), valid.max()
            else:
                mn, mx = np.nan, np.nan
            print(
                f"    [{b}] {band_descriptions[b - 1]:<20s} "
                f"min = {mn:10.4f}, max = {mx:10.4f}"
            )

    print("\n  Нормализация растра НЕ выполняется.")
    print("  Стандартизация (z-score) будет применена в sklearn (preProcess).\n")

    # ================================================================
    # ШАГ 2: ЗАГРУЗКА ОБУЧАЮЩИХ ПОЛИГОНОВ
    # ================================================================
    print("================================================================")
    print("[ШАГ 2] ЗАГРУЗКА ОБУЧАЮЩИХ ПОЛИГОНОВ")
    print("================================================================\n")

    ref_shp_path = None
    class_field = None

    if config and config.get("ref_shp_path"):
        candidate = config["ref_shp_path"]
        if os.path.exists(candidate):
            ref_shp_path = candidate
            class_field = config.get("ref_class_field")
            print("Из pipeline_config:")
            print(f"  SHP: {os.path.basename(ref_shp_path)}")
            print(f"  Поле: {class_field}\n")
            confirm = input("Использовать? (Enter = да, N = выбрать другой): ").strip().upper()
            if confirm == "N":
                ref_shp_path = None
                class_field = None
        else:
            print(f"! Файл из config не найден: {candidate}")
            ref_shp_path = None

    if ref_shp_path is None:
        input_dir = os.path.join(root_dir, "INPUT")
        shp_dirs = [input_dir, prev_dir, root_dir]
        all_shp = []
        for d in shp_dirs:
            if os.path.isdir(d):
                all_shp.extend(glob.glob(os.path.join(d, "*.shp")))
        all_shp = sorted(set(all_shp))

        if not all_shp:
            raise FileNotFoundError(
                "SHP файлы не найдены! Поместите слой эталонов в папку INPUT."
            )

        print("Найдены SHP файлы:\n")
        for i, s in enumerate(all_shp, 1):
            print(f"  [{i}] {s}")

        shp_choice_str = input("\nВыберите номер файла с обучающими полигонами: ").strip()
        try:
            shp_choice = int(shp_choice_str)
        except ValueError:
            raise ValueError("Некорректный выбор!")

        if shp_choice < 1 or shp_choice > len(all_shp):
            raise ValueError("Некорректный выбор!")

        ref_shp_path = all_shp[shp_choice - 1]

        # Выбор поля классов
        tmp_gdf = gpd.read_file(ref_shp_path)
        flds = [c for c in tmp_gdf.columns if c != "geometry"]

        print("\nПоля атрибутивной таблицы:")
        for i, f in enumerate(flds, 1):
            print(f"  [{i}] {f}")

        cf_input = input(
            "\nВ каком поле находятся НАЗВАНИЯ КЛАССОВ? (имя или номер): "
        ).strip()

        if cf_input in flds:
            class_field = cf_input
        else:
            try:
                idx = int(cf_input)
                if 1 <= idx <= len(flds):
                    class_field = flds[idx - 1]
                else:
                    raise ValueError("Некорректный выбор поля!")
            except ValueError:
                raise ValueError("Некорректный выбор поля!")
        del tmp_gdf

    print(f"\n✓ Эталоны: {os.path.basename(ref_shp_path)}")
    print(f"✓ Поле классов: {class_field}\n")

    training_poly = gpd.read_file(ref_shp_path)

    if class_field not in training_poly.columns:
        raise KeyError(
            f"Поле '{class_field}' не найдено в {os.path.basename(ref_shp_path)}!"
        )

    print(f"  Загружено полигонов: {len(training_poly)}")
    unique_classes = training_poly[class_field].unique()
    print(f"  Классы ({class_field}): {', '.join(str(c) for c in unique_classes)}")

    # Согласование СКО
    training_poly = training_poly.to_crs(raster_crs)
    print("  ✓ Система координат согласована\n")

    # ================================================================
    # ШАГ 3: ИЗВЛЕЧЕНИЕ ЗНАЧЕНИЙ ИЗ РАСТРА ПО ПОЛИГОНАМ
    # ================================================================
    print("================================================================")
    print("[ШАГ 3] ИЗВЛЕЧЕНИЕ ЗНАЧЕНИЙ ИЗ РАСТРА ПО ПОЛИГОНАМ")
    print("================================================================\n")

    print(f"  Источник: {os.path.basename(selected_tif)} ({n_layers} слоёв)")
    print("  (это может занять время в зависимости от размера данных)\n")

    samples_list = []
    with rasterio.open(selected_tif) as src:
        transform = src.transform

        for idx_row, row in training_poly.iterrows():
            cls = row[class_field]
            geom = [mapping(row.geometry)]

            print(
                f"\r  Полигон {idx_row + 1}/{len(training_poly)} "
                f"(класс: {cls})...   ",
                end="",
                flush=True,
            )

            try:
                out_image, out_transform = rio_mask(
                    src, geom, crop=True, all_touched=True, nodata=np.nan
                )
            except Exception:
                continue

            # out_image: (n_bands, h, w)
            # маска валидных пикселей — все бенды конечные
            valid_mask = np.all(np.isfinite(out_image), axis=0)
            if not valid_mask.any():
                continue

            coords = np.argwhere(valid_mask)  # (row, col) внутри кропа
            for rc in coords:
                r, c = rc[0], rc[1]
                vals = out_image[:, r, c]
                # координаты x, y
                x_coord = out_transform.c + c * out_transform.a + r * out_transform.b
                y_coord = out_transform.f + c * out_transform.d + r * out_transform.e
                row_dict = {}
                for bi, bn in enumerate(band_descriptions):
                    row_dict[bn] = float(vals[bi])
                row_dict["class_name"] = str(cls)
                row_dict["polygon_id"] = int(idx_row)
                row_dict["x"] = x_coord
                row_dict["y"] = y_coord
                samples_list.append(row_dict)

    print("\n")

    if not samples_list:
        raise RuntimeError("Не удалось извлечь образцы!")

    samples_df = pd.DataFrame(samples_list)

    # --- Определяем бенд-колонки ---
    service_cols = {
        "ID", "cell", "x", "y", "class_name", "polygon_id",
        "FID", "fid", "OBJECTID", "GID", "gid",
    }
    band_cols = [
        c
        for c in samples_df.columns
        if c not in service_cols and pd.api.types.is_numeric_dtype(samples_df[c])
    ]

    # Проверка соответствия числа бандов
    if len(band_cols) != n_layers:
        print(
            f"  ⚠ Число бандов в образцах ({len(band_cols)}) "
            f"не совпадает со слоями растра ({n_layers})!"
        )
    else:
        print(
            f"  ✓ Проверка: {len(band_cols)} бандов в образцах = "
            f"{n_layers} слоёв растра"
        )

    # Оставляем бенды + class_name + polygon_id + x + y
    keep_cols = band_cols + ["class_name", "polygon_id", "x", "y"]
    extra_cols = [c for c in samples_df.columns if c not in keep_cols]
    if extra_cols:
        print(f"  Удалены лишние поля: {', '.join(extra_cols)}")
    samples_df = samples_df[keep_cols]

    print(f"\n  ✓ Итого образцов: {len(samples_df)}")
    print(f"  ✓ Банды: {', '.join(band_cols)}")
    print(f"  ✓ Поля в samples_df: {', '.join(samples_df.columns)}")

    # Статистика по классам
    class_counts = samples_df["class_name"].value_counts().sort_index()
    print("\n  Распределение по классам:")
    for cls, cnt in class_counts.items():
        print(f"    {cls}: {cnt} образцов")

    # ================================================================
    # ШАГ 4: СОХРАНЕНИЕ ОБРАЗЦОВ
    # ================================================================
    print("\n================================================================")
    print("[ШАГ 4] СОХРАНЕНИЕ ОБРАЗЦОВ")
    print("================================================================\n")

    # --- GeoPackage ---
    geometry = gpd.points_from_xy(samples_df["x"], samples_df["y"], crs=raster_crs)
    samples_gdf = gpd.GeoDataFrame(
        samples_df.drop(columns=["x", "y"]),  # polygon_id сохраняется
        geometry=geometry,
        crs=raster_crs,
    )

    gpkg_path = os.path.join(output_dir, "samples.gpkg")
    if os.path.exists(gpkg_path):
        os.remove(gpkg_path)
    samples_gdf.to_file(gpkg_path, layer="samples", driver="GPKG")
    print(
        f" ✓ {os.path.basename(gpkg_path)} "
        f"({len(samples_gdf)} точек, поля: {', '.join(band_cols)} + class_name + polygon_id)"
    )

    # --- CSV (с координатами для справки) ---
    csv_path = os.path.join(output_dir, "samples.csv")
    samples_df.to_csv(csv_path, index=False)
    print(f"  ✓ {os.path.basename(csv_path)}")

    # --- Сохраняем информацию о растре (pkl вместо rds) ---
    raster_info = {
        "raster_path": selected_tif,
        "raster_name": os.path.basename(selected_tif),
        "n_layers": n_layers,
        "layer_names": band_descriptions,
        "crs": str(raster_crs),
    }
    raster_info_path = os.path.join(output_dir, "raster_info.pkl")
    with open(raster_info_path, "wb") as f:
        pickle.dump(raster_info, f)
    print(f"  ✓ raster_info.pkl (путь к растру для скрипта 04)")

    # ================================================================
    # ФИНАЛЬНАЯ СВОДКА
    # ================================================================
    print("\n================================================================")
    print("  ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА")
    print("================================================================\n")

    print("Файлы в 01_PREPARE_DATA:")
    print(
        f"  1. samples.gpkg - обучающие образцы "
        f"({len(samples_gdf)} точек, {len(band_cols)} бандов + polygon_id)"
    )
    print("  2. samples.csv - обучающие образцы (таблица)")
    print("  3. raster_info.pkl - информация о растре")
    print(
        f"\n  Растр: {os.path.basename(selected_tif)} "
        f"(используется как есть, без нормализации)"
    )
    print("  Стандартизация будет выполнена в 03_tune_models (preProcess)")
    print("\nСледующий шаг: запустите 02_classification.py")
    print("================================================================\n")

    return {
        "raster_path": selected_tif,
        "samples_shp_path": ref_shp_path,
        "samples_csv_path": csv_path,
        "n_samples": len(samples_gdf),
        "class_field": "class_name",
        "classes": list(class_counts.index),
        "n_layers": n_layers,
        "layer_names": band_descriptions,
        "band_cols": band_cols,
    }


# ================================================================
# ЗАПУСК
# ================================================================
if __name__ == "__main__":
    main()
