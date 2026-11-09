# ============================================================================
# 01_prepare_data_v14.py (workspace: 01_prepare_data_v8.py)
# v8/v14: запись samples.gpkg через локальный TEMP с journal_mode=DELETE (без WAL)
# ИЗМЕНЕНИЯ: в списке растров запуска 00 видны и .vrt — виртуальные
# композиты (Composite.vrt / Composite_filtered.vrt из 00 v66) читаются
# как обычные растры.
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

# Отключаем буферизацию stdout (важно для Jupyter/IPython через %run)
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n============================", flush=True)
    print("  01 - PREPARE DATA", flush=True)
    print("============================\n", flush=True)
    print(f"Рабочая папка: {os.getcwd()}", flush=True)
    print(f"Python: {sys.executable}", flush=True)
    print(flush=True)

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

    print("-- INPUT (TIF / VRT) --")
    # v7: .vrt (виртуальный композит из 00 v66) читается как обычный растр
    tif_files = sorted(
        glob.glob(os.path.join(selected_prev_dir, "*.tif")) +
        glob.glob(os.path.join(selected_prev_dir, "*.vrt"))
    )
    if not tif_files:
        raise FileNotFoundError(
            "TIF/VRT не найдены в выбранной подпапке 00_PREPARE_LAYERS!")

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

    # --- ПРАВКА (a): стабильный polygon_id ---
    # Если в исходнике есть свое поле polygon_id / ID / OBJECTID / FID с уникальными
    # значениями — уважаем его. Иначе создаём свой через reset_index,
    # чтобы polygon_id не зависел от внешней сортировки/фильтрации.
    _pid_candidates = ["polygon_id", "ID", "OBJECTID", "FID"]
    _pid_source = None
    for _c in _pid_candidates:
        if _c in training_poly.columns and training_poly[_c].is_unique:
            _pid_source = _c
            break
    if _pid_source is not None:
        training_poly = training_poly.copy()
        training_poly["polygon_id"] = training_poly[_pid_source].astype(int)
        print(f"  ✓ polygon_id взят из исходного поля '{_pid_source}'")
    else:
        training_poly = training_poly.reset_index(drop=True).copy()
        training_poly["polygon_id"] = training_poly.index.astype(int)
        print("  ✓ polygon_id сгенерирован автоматически (0..N-1)")

    # --- ПРАВКА (a-bis): геометрия — определение типа (точки/полигоны) ---
    _geom_types = set(training_poly.geometry.geom_type.unique())
    _has_points = bool(_geom_types & {"Point", "MultiPoint"})
    _has_polys = bool(_geom_types & {"Polygon", "MultiPolygon"})
    if _has_points and _has_polys:
        print(
            "  ⚠ Смешанная геометрия (точки + полигоны) — будет "
            "обработано через rio_mask для всех геометрий единообразно."
        )
    elif _has_points:
        print("  ✓ Геометрия: точки (1 точка = 1 пиксель)")
    else:
        print("  ✓ Геометрия: полигоны (много пикселей на полигон)")
    print()

    # ================================================================
    # ШАГ 3: ИЗВЛЕЧЕНИЕ ЗНАЧЕНИЙ ИЗ РАСТРА ПО ПОЛИГОНАМ
    # ================================================================
    print("================================================================")
    print("[ШАГ 3] ИЗВЛЕЧЕНИЕ ЗНАЧЕНИЙ ИЗ РАСТРА ПО ПОЛИГОНАМ")
    print("================================================================\n")

    print(f"  Источник: {os.path.basename(selected_tif)} ({n_layers} слоёв)")
    print("  (это может занять время в зависимости от размера данных)\n")

    samples_list = []
    # Статистика по polygon_id: сколько пикселей извлечено (ПРАВКА b)
    n_pixels_by_pid = {}
    n_skipped_empty = 0
    n_skipped_error = 0
    with rasterio.open(selected_tif) as src:
        transform = src.transform

        for idx_row, row in training_poly.iterrows():
            cls = row[class_field]
            pid = int(row["polygon_id"])  # ПРАВКА (a): берём из колонки
            geom = [mapping(row.geometry)]

            print(
                f"\r  Геометрия {idx_row + 1}/{len(training_poly)} "
                f"(polygon_id={pid}, класс: {cls})...   ",
                end="",
                flush=True,
            )

            try:
                out_image, out_transform = rio_mask(
                    src, geom, crop=True, all_touched=True, nodata=np.nan
                )
            except Exception:
                n_skipped_error += 1
                continue

            # out_image: (n_bands, h, w)
            # маска валидных пикселей — все бенды конечные
            valid_mask = np.all(np.isfinite(out_image), axis=0)
            if not valid_mask.any():
                n_skipped_empty += 1
                continue

            coords = np.argwhere(valid_mask)  # (row, col) внутри кропа
            n_pixels_by_pid[pid] = int(len(coords))
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
                row_dict["polygon_id"] = pid  # стабильный ID
                row_dict["n_pixels"] = int(len(coords))  # ПРАВКА (b)
                row_dict["x"] = x_coord
                row_dict["y"] = y_coord
                samples_list.append(row_dict)

    # Диагностика отброшенных геометрий (ПРАВКА b)
    print()
    if n_skipped_error > 0:
        print(f"  ⚠ Отброшено по ошибке rio_mask: {n_skipped_error} геометрий")
    if n_skipped_empty > 0:
        print(
            f"  ⚠ Отброшено без валидных пикселей (NaN во всех бендах): "
            f"{n_skipped_empty} геометрий"
        )
    if n_pixels_by_pid:
        _counts = np.array(list(n_pixels_by_pid.values()))
        print(
            f"  ✓ Пикселей на полигон: min={_counts.min()}, "
            f"median={int(np.median(_counts))}, mean={_counts.mean():.1f}, "
            f"max={_counts.max()}"
        )
        _tiny = int((_counts <= 2).sum())
        _huge = int((_counts > 500).sum())
        if _tiny > 0:
            print(f"    • мелких (≤2 пикс.): {_tiny}")
        if _huge > 0:
            print(f"    • крупных (>500 пикс.): {_huge}")

    print("\n")

    if not samples_list:
        raise RuntimeError("Не удалось извлечь образцы!")

    samples_df = pd.DataFrame(samples_list)

    # --- Определяем бенд-колонки ---
    service_cols = {
        "ID", "cell", "x", "y", "class_name", "polygon_id", "n_pixels",
        "FID", "fid", "OBJECTID", "GID", "gid",
    }
    band_cols = [
        c
        for c in samples_df.columns
        if c not in service_cols and pd.api.types.is_numeric_dtype(samples_df[c])
    ]

    # --- ПРАВКА (c): де-конфликт case-insensitive имён ---
    # GPKG/Shapefile приводят имена полей к единому регистру. Если в растре
    # есть 'S2_BAI' и 'S2_BaI' драйвер падает с "Error adding field...".
    # Найдём такие пары и переименуем второе вхождение → добавим суффикс.
    _seen_lower = {}
    _rename_map = {}
    for _bc in band_cols:
        _low = _bc.lower()
        if _low in _seen_lower:
            # Конфликт — генерируем уникальное имя
            _n = 2
            while f"{_bc}_{_n}".lower() in _seen_lower:
                _n += 1
            _new = f"{_bc}_{_n}"
            _rename_map[_bc] = _new
            _seen_lower[_new.lower()] = _new
            print(
                f"  ⚠ Конфликт имён (case-insensitive): '{_bc}' vs "
                f"'{_seen_lower[_low]}' → переименовано в '{_new}'"
            )
        else:
            _seen_lower[_low] = _bc
    if _rename_map:
        samples_df = samples_df.rename(columns=_rename_map)
        band_cols = [_rename_map.get(c, c) for c in band_cols]
        # Сообщаем наружу, чтобы band_descriptions в raster_info тоже обновить
        band_descriptions = [_rename_map.get(bn, bn) for bn in band_descriptions]

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

    # Оставляем бенды + class_name + polygon_id + n_pixels + x + y (ПРАВКА b)
    keep_cols = band_cols + ["class_name", "polygon_id", "n_pixels", "x", "y"]
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
    # --- ПРАВКА (c-bis): финальный барьер против case-insensitive дублей ---
    _final_lower = {}
    _dup_final = []
    for _c in samples_gdf.columns:
        _l = _c.lower()
        if _l in _final_lower and _final_lower[_l] != _c:
            _dup_final.append((_final_lower[_l], _c))
        else:
            _final_lower[_l] = _c
    if _dup_final:
        print(
            f"  ⚠ Обнаружены оставшиеся case-insensitive дубли перед записью: "
            f"{_dup_final}"
        )
    # v8: запись GPKG через локальный TEMP + journal_mode=DELETE (без WAL),
    # чтобы Яндекс.Диск не блокировал -wal/-shm и 02/03 не падали с
    # "attempt to write a readonly database".
    os.environ["OGR_SQLITE_JOURNAL"] = "DELETE"
    import tempfile as _tempfile, shutil as _shutil, sqlite3 as _sqlite3
    _tmp_dir = _tempfile.mkdtemp(prefix="gpkg_write_")
    _tmp_gpkg = os.path.join(_tmp_dir, "samples.gpkg")
    try:
        try:
            samples_gdf.to_file(_tmp_gpkg, layer="samples", driver="GPKG")
        except Exception as _e:
            print(f"  ⚠ Ошибка записи GPKG: {_e}. Пробуем fallback (все имена → lower).")
            _fallback_map = {}
            _used = set()
            for _c in samples_gdf.columns:
                _base = _c.lower()
                _n = _base
                _i = 2
                while _n in _used:
                    _n = f"{_base}_{_i}"
                    _i += 1
                _fallback_map[_c] = _n
                _used.add(_n)
            samples_gdf = samples_gdf.rename(columns=_fallback_map)
            samples_gdf.to_file(_tmp_gpkg, layer="samples", driver="GPKG")
            print("  ✓ GPKG сохранён через fallback")
        try:
            _con = _sqlite3.connect(_tmp_gpkg)
            _con.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            _con.execute("PRAGMA journal_mode=DELETE;")
            _con.close()
        except Exception:
            pass
        try:
            with open(_tmp_gpkg, "r+b") as _fh:
                _hdr = bytearray(_fh.read(32))
                if len(_hdr) >= 20 and (_hdr[18] == 2 or _hdr[19] == 2):
                    _hdr[18] = 1
                    _hdr[19] = 1
                    _fh.seek(0)
                    _fh.write(_hdr)
        except Exception:
            pass
        _shutil.copyfile(_tmp_gpkg, gpkg_path)
    finally:
        _shutil.rmtree(_tmp_dir, ignore_errors=True)
    print(
        f" ✓ {os.path.basename(gpkg_path)} "
        f"({len(samples_gdf)} строк, поля: {len(band_cols)} бандов + "
        f"class_name + polygon_id + n_pixels)"
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
        f"({len(samples_gdf)} строк, {len(band_cols)} бандов + polygon_id + n_pixels)"
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
