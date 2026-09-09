# =============================================================================
# 04_predict_map_v5.py
#
# СКРИПТ 4: ЭКСТРАПОЛЯЦИЯ МОДЕЛИ НА ВЕСЬ РАСТР
#
# Функционал:
# - Загрузка финальной модели из 03_TUNE (final_model.pkl)
# - Поиск исходного растра (raster_info.pkl / 00_PREPARE_LAYERS)
# - Предсказание классов (categorial GeoTIFF, INT16, LZW)
# - Предсказание макс. вероятности (probability GeoTIFF, FLT32, LZW)
# - Confusion matrix + per-class метрики (heatmap PNG)
# - Карта классификации (PNG)
# - CSV со статистикой
# - Проект QGIS (.qgs) с готовой раскраской категориального растра
# - Сохранение данных для 05 (final_model.pkl, samples_df.pkl, etc.)
#
# ИЗМЕНЕНИЯ В v5:
# - Нумерация категорий в растре теперь 0-based (ID класса = индекс в
#   class_levels, БЕЗ +1). Значения растра: 0..(n_classes-1).
#   nodata = -9999 (не пересекается с классом 0).
# - Формируется QGIS-проект <base_name>.qgs с palettedраскраской
#   категориального растра по фиксированной палитре (24 класса).
# =============================================================================

import os
import re
import sys
import gc
import glob
import pickle
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    classification_report, f1_score,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# =============================================================================
# ФИКСИРОВАННАЯ ПАЛИТРА КЛАССОВ (для QGIS)
# =============================================================================
# Ключ — номер класса НА ИСХОДНОЙ ПАЛИТРЕ (1..24, как на картинке).
# Значение — HEX-цвет заливки.
# В растре v5 нумерация 0-based, поэтому классу палитры k
# соответствует значение растра (k-1) — см. _build_palette_for_raster().
PALETTE_HEX = {
    1:  "#FF99FF",
    2:  "#FF99FF",
    3:  "#FECCFF",
    4:  "#FECCFF",
    5:  "#B1A1C8",
    6:  "#B1A1C8",
    7:  "#E3DFED",
    8:  "#E3DFED",
    9:  "#E9ECF5",
    10: "#E9ECF5",
    11: "#E9ECF5",
    12: "#E9ECF5",
    13: "#F1DCDB",
    14: "#F1DCDB",
    15: "#FFFF00",
    16: "#C3D79A",
    17: "#E9ECF5",
    18: "#F1DCDB",
    19: "#CDC0DA",
    20: "#F1DCDB",
    21: "#95B3D7",
    22: "#F1DCDB",
    23: "#95B3D7",
    24: "#95B3D7",
}

# Цвет по умолчанию для классов сверх таблицы (если классов > 24)
PALETTE_FALLBACK = "#BDBDBD"


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
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
            and re.search(r"_v\d{3}$", d)
        ]
    )
    if not subdirs:
        raise FileNotFoundError(
            f"В {os.path.basename(base_dir)} нет подпапок "
            f"с результатами (формат ддммгг_ччмм_vXXX)."
        )

    print(f"\nДоступные запуски в {os.path.basename(base_dir)}:\n")
    for i, sd in enumerate(subdirs, 1):
        print(f" [{i}] {sd}")

    raw = input(
        f"\nВыберите номер запуска из {os.path.basename(base_dir)}: "
    ).strip()
    if raw == "":
        raw = "1"
    idx = int(raw)
    if idx < 1 or idx > len(subdirs):
        raise ValueError("Некорректный выбор запуска.")

    selected_name = subdirs[idx - 1]
    selected_path = os.path.join(base_dir, selected_name)

    m = re.search(r"(_v\d{3})$", selected_name)
    suffix = m.group(1) if m else "_v001"
    print(f"Выбрана папка: {selected_name}, суффикс: {suffix}\n")
    return selected_path, suffix


def _get_algo_abbr(name: str) -> str:
    """Аббревиатура алгоритма для имени файла."""
    mapping = {
        "Random Forest": "RFo", "ranger": "RFo",
        "XGBoost": "XGb", "xgbTree": "XGb",
        "GBM": "GBM", "gbm": "GBM",
        "Ridge": "Rid", "Lasso": "Las", "Elastic": "ENet",
        "glmnet": "GLn",
        "LDA": "LDA", "lda": "LDA",
        "QDA": "QDA", "qda": "QDA",
        "RDA": "RDA", "rda": "RDA",
        "catboost": "CBo", "CatBoost": "CBo",
    }
    for pattern, abbr in mapping.items():
        if pattern.lower() in name.lower():
            return abbr
    clean = re.sub(r"[^A-Za-z0-9]", "", name)
    return clean[:3] if clean else "Mdl"


def _interp_oa(x):
    if x is None or x == 0:
        return "н/д"
    if x >= 0.9:
        return "отлично"
    if x >= 0.8:
        return "хорошо"
    if x >= 0.7:
        return "удовл."
    if x >= 0.6:
        return "посредств."
    return "плохо"


def _interp_kappa(x):
    if x is None or x == 0:
        return "н/д"
    if x >= 0.8:
        return "отлично"
    if x >= 0.6:
        return "хорошо"
    if x >= 0.4:
        return "удовл."
    if x >= 0.2:
        return "слабо"
    return "плохо"


def _interp_daic(x):
    if x is None:
        return "н/д"
    if x <= 0:
        return "лучшая"
    if x <= 2:
        return "практ.равная"
    if x <= 4:
        return "заметно хуже"
    if x <= 7:
        return "значит.хуже"
    return "несопоставима"


def _color_for_metric(value, metric_name: str) -> str:
    """Цвет фона для метрики (traffic-light)."""
    if value is None or np.isnan(value):
        return "#CCCCCC"
    v = float(value)
    if metric_name == "dAIC":
        if v <= 0:
            return "#00AA00"
        elif v <= 2:
            return "#88CC00"
        elif v <= 4:
            return "#CCCC00"
        elif v <= 7:
            return "#FFAA00"
        return "#FF0000"
    else:
        if v >= 0.9:
            return "#00AA00"
        elif v >= 0.8:
            return "#88CC00"
        elif v >= 0.7:
            return "#CCCC00"
        elif v >= 0.6:
            return "#FFAA00"
        return "#FF0000"


def _get_model_n_features(model) -> int | None:
    """Возвращает фактическое число признаков, ожидаемое сохранённой моделью."""
    if hasattr(model, "n_features_in_"):
        try:
            return int(model.n_features_in_)
        except Exception:
            pass

    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            n = _get_model_n_features(step)
            if n is not None:
                return n

    if hasattr(model, "steps"):
        for _, step in model.steps:
            n = _get_model_n_features(step)
            if n is not None:
                return n

    return None


def _get_model_feature_names(model) -> list[str] | None:
    """Пытается извлечь имена признаков прямо из модели / pipeline."""
    if hasattr(model, "feature_names_in_"):
        try:
            return [str(x) for x in list(model.feature_names_in_)]
        except Exception:
            pass

    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            names = _get_model_feature_names(step)
            if names:
                return names

    if hasattr(model, "steps"):
        for _, step in model.steps:
            names = _get_model_feature_names(step)
            if names:
                return names

    return None


def _is_feature_list(value, band_names: list[str], expected_n: int | None) -> bool:
    """Проверяет, похож ли объект на список имён предикторов растра."""
    if not isinstance(value, (list, tuple, np.ndarray, pd.Index)):
        return False
    vals = [str(v) for v in list(value)]
    if expected_n is not None and len(vals) != expected_n:
        return False
    if len(vals) == 0:
        return False
    return set(vals).issubset(set(band_names))


def _find_feature_list_in_object(obj, band_names: list[str], expected_n: int | None):
    """Рекурсивно ищет список имён признаков в pkl-объектах."""
    if _is_feature_list(obj, band_names, expected_n):
        return [str(v) for v in list(obj)]

    if isinstance(obj, dict):
        priority_keys = [
            "predictor_cols", "predictors", "selected_features",
            "feature_cols", "feature_names", "features", "keep_names",
        ]
        for key in priority_keys:
            if key in obj:
                found = _find_feature_list_in_object(obj[key], band_names, expected_n)
                if found:
                    return found
        for value in obj.values():
            found = _find_feature_list_in_object(value, band_names, expected_n)
            if found:
                return found

    if isinstance(obj, pd.DataFrame):
        for col in [
            "predictor", "predictors", "feature", "features",
            "Feature", "Predictor", "layer", "Layer", "band", "Band", "name", "Name",
        ]:
            if col in obj.columns:
                vals = obj[col].dropna().astype(str).tolist()
                if _is_feature_list(vals, band_names, expected_n):
                    return vals

    return None


def _find_predictors_in_artifacts(
    search_dirs: list[str],
    band_names: list[str],
    expected_n: int | None,
) -> tuple[list[str] | None, str | None]:
    """Ищет точный список предикторов в pkl/csv/json артефактах предыдущих этапов."""
    for search_dir in search_dirs:
        if not search_dir or not os.path.isdir(search_dir):
            continue

        for root, _, files in os.walk(search_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                low = fn.lower()
                try:
                    if low.endswith(".pkl"):
                        with open(fp, "rb") as f:
                            obj = pickle.load(f)
                        found = _find_feature_list_in_object(obj, band_names, expected_n)
                        if found:
                            return found, fp
                    elif low.endswith(".csv"):
                        df = pd.read_csv(fp)
                        found = _find_feature_list_in_object(df, band_names, expected_n)
                        if found:
                            return found, fp
                    elif low.endswith(".json"):
                        df = pd.read_json(fp)
                        found = _find_feature_list_in_object(df, band_names, expected_n)
                        if found:
                            return found, fp
                except Exception:
                    continue

    return None, None


def _parse_predictor_selection(raw: str, band_names: list[str]) -> list[str]:
    """Парсит ручной ввод признаков: номера band-ов или имена через запятую/пробел."""
    tokens = re.split(r"[,;\s]+", raw.strip())
    tokens = [t for t in tokens if t]
    selected = []

    for token in tokens:
        if token.isdigit():
            idx = int(token) - 1
            if idx < 0 or idx >= len(band_names):
                raise ValueError(f"Номер band-а вне диапазона: {token}")
            selected.append(band_names[idx])
        else:
            if token not in band_names:
                raise ValueError(f"Имя отсутствует в растре: {token}")
            selected.append(token)

    return selected


def _manual_predictor_selection(
    band_names: list[str],
    expected_n: int | None,
    selected_tune_dir: str,
) -> list[str]:
    """Интерактивно выбирает признаки, если модель не сохранила их имена."""
    print("\n" + "=" * 60)
    print("  РУЧНОЙ ВЫБОР ПРЕДИКТОРОВ ДЛЯ МОДЕЛИ")
    print("=" * 60)
    print(
        "Модель хранит только количество признаков, но не их имена. "
        "Нужно указать, какие слои растра использовались при обучении."
    )
    if expected_n is not None:
        print(f"Ожидается признаков: {expected_n}")
    print("\nДоступные слои растра:")
    for i, nm in enumerate(band_names, 1):
        print(f" [{i:2d}] {nm}")

    print(
        "\nВведите номера или имена нужных слоёв через запятую/пробел "
        "в порядке обучения модели."
    )
    print("Например: 1,2,3,5,7,...")

    while True:
        raw = input("\nПредикторы модели: ").strip()
        if raw == "":
            print("  Ввод пустой. Нужно указать список признаков.")
            continue

        try:
            selected = _parse_predictor_selection(raw, band_names)
        except ValueError as exc:
            print(f"  Ошибка: {exc}")
            continue

        if expected_n is not None and len(selected) != expected_n:
            print(
                f"  Выбрано {len(selected)} признаков, "
                f"а модель ожидает {expected_n}. Повторите ввод."
            )
            continue

        if len(set(selected)) != len(selected):
            print("  В списке есть дубликаты. Повторите ввод.")
            continue

        save_path = os.path.join(selected_tune_dir, "predictor_names_for_04.txt")
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                for nm in selected:
                    f.write(f"{nm}\n")
            print(f"  ✓ Список сохранён: {save_path}")
        except Exception as exc:
            print(f"  ! Не удалось сохранить список: {exc}")

        return selected


def _load_predictor_names_txt(
    selected_tune_dir: str,
    band_names: list[str],
    expected_n: int | None,
) -> list[str] | None:
    """Загружает ранее вручную сохранённый список признаков для 04."""
    fp = os.path.join(selected_tune_dir, "predictor_names_for_04.txt")
    if not os.path.exists(fp):
        return None

    try:
        with open(fp, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
    except Exception:
        return None

    if expected_n is not None and len(names) != expected_n:
        print(
            f"  ! predictor_names_for_04.txt содержит {len(names)} признаков, "
            f"а модель ожидает {expected_n}; файл не используется."
        )
        return None

    missing = [nm for nm in names if nm not in band_names]
    if missing:
        print(
            "  ! predictor_names_for_04.txt содержит признаки, отсутствующие в растре: "
            + ", ".join(missing)
        )
        return None

    print(f"  ✓ Список предикторов загружен: {fp}")
    return names


def _resolve_predictor_names(
    final_model,
    predictor_cols: list | None,
    band_names: list[str],
    selected_tune_dir: str,
    select_dir: str | None,
) -> list[str]:
    """Определяет имена и порядок предикторов, которые надо читать из растра."""
    expected_n = _get_model_n_features(final_model)
    model_names = _get_model_feature_names(final_model)

    txt_names = _load_predictor_names_txt(selected_tune_dir, band_names, expected_n)
    if txt_names:
        return txt_names

    candidates = []
    if model_names:
        candidates.append(("модель", model_names))
    if predictor_cols:
        candidates.append(("data_info.pkl", [str(x) for x in predictor_cols]))

    for source, names in candidates:
        if expected_n is not None and len(names) != expected_n:
            print(
                f"  ! {source}: найдено {len(names)} признаков, "
                f"а модель ожидает {expected_n}; список не используется."
            )
            continue
        missing = [nm for nm in names if nm not in band_names]
        if missing:
            print(
                f"  ! {source}: {len(missing)} признаков отсутствуют в растре; "
                f"список не используется."
            )
            if len(missing) <= 10:
                print(f"    Нет в растре: {', '.join(missing)}")
            continue
        return names

    artifact_dirs = [selected_tune_dir]
    if select_dir:
        artifact_dirs.append(select_dir)
    found, found_path = _find_predictors_in_artifacts(artifact_dirs, band_names, expected_n)
    if found:
        print(f"  ✓ Список предикторов найден в артефакте: {found_path}")
        return found

    if expected_n is not None and expected_n == len(band_names):
        return band_names

    print("\n  ! Не удалось автоматически определить точный набор предикторов.")
    print(f"    Растр содержит слоёв: {len(band_names)}")
    print(
        "    Модель ожидает признаков: "
        f"{expected_n if expected_n is not None else 'неизвестно'}"
    )
    if predictor_cols:
        print(f"    В data_info.pkl найдено признаков: {len(predictor_cols)}")

    return _manual_predictor_selection(band_names, expected_n, selected_tune_dir)


def _read_raster_brief(path: str) -> dict | None:
    """Читает краткие метаданные TIF для выбора подходящего композита."""
    try:
        with rasterio.open(path) as src:
            names = [
                src.descriptions[i] if src.descriptions[i] else f"band{i + 1}"
                for i in range(src.count)
            ]
            return {
                "path": path,
                "n_bands": src.count,
                "height": src.height,
                "width": src.width,
                "band_names": names,
                "size_mb": os.path.getsize(path) / (1024 ** 2),
            }
    except Exception:
        return None


def _find_matching_composite_raster(
    root_dir: str,
    version_suffix: str,
    expected_n: int | None,
    current_path: str | None = None,
) -> tuple[str | None, str | None]:
    """Ищет Composite*.tif с числом слоёв, совпадающим с моделью."""
    if expected_n is None:
        return current_path, None

    search_roots = []
    for dirname in ("01_PREPARE_DATA", "00_PREPARE_LAYERS"):
        base = os.path.join(root_dir, dirname)
        if not os.path.isdir(base):
            continue
        subdirs = sorted([
            d for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d)) and d.endswith(version_suffix)
        ])
        if subdirs:
            search_roots.append((dirname, os.path.join(base, subdirs[-1])))

    candidates = {}
    if current_path and os.path.exists(current_path):
        candidates[os.path.abspath(current_path)] = ("текущий raster_info", current_path)

    for source, search_root in search_roots:
        patterns = [
            os.path.join(search_root, "Composite*.tif"),
            os.path.join(search_root, "**", "Composite*.tif"),
            os.path.join(search_root, "*.tif"),
        ]
        for pattern in patterns:
            for fp in glob.glob(pattern, recursive=True):
                if os.path.exists(fp):
                    candidates[os.path.abspath(fp)] = (source, fp)

    matched = []
    scanned = []
    for _, (source, fp) in candidates.items():
        brief = _read_raster_brief(fp)
        if brief is None:
            continue
        scanned.append((source, brief))
        if brief["n_bands"] == expected_n:
            matched.append((source, brief))

    if current_path:
        cur_abs = os.path.abspath(current_path)
        for source, brief in matched:
            if os.path.abspath(brief["path"]) == cur_abs:
                return current_path, None

    if len(matched) == 1:
        source, brief = matched[0]
        print(
            f"  ✓ Найден растр под модель ({expected_n} слоёв): "
            f"{os.path.basename(brief['path'])} ({source})"
        )
        return brief["path"], source

    if len(matched) > 1:
        def _priority(item):
            source, brief = item
            name = os.path.basename(brief["path"]).lower()
            if name == "composite_filtered.tif":
                p = 0
            elif name == "composite.tif":
                p = 1
            elif name.startswith("composite"):
                p = 2
            else:
                p = 3
            return (p, -brief["size_mb"], brief["path"])

        matched = sorted(matched, key=_priority)
        source, brief = matched[0]
        print(f"  ! Найдено несколько растров с {expected_n} слоями:")
        for j, (src_name, b) in enumerate(matched, 1):
            print(
                f"    [{j}] {os.path.basename(b['path'])} "
                f"({src_name}, {b['size_mb']:.1f} MB)"
            )
        print(f"  ✓ Автоматически выбран: {os.path.basename(brief['path'])}")
        return brief["path"], source

    if scanned:
        print(f"  ! Растр с {expected_n} слоями не найден. Проверенные TIF:")
        for source, brief in scanned:
            print(
                f"    - {os.path.basename(brief['path'])}: "
                f"{brief['n_bands']} слоёв ({source})"
            )

    return current_path, None


# =============================================================================
# QGIS-ПРОЕКТ + СТИЛЬ (paletted раскраска категориального растра)
# =============================================================================

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _build_palette_for_raster(class_levels: list) -> list[dict]:
    """
    Сопоставляет каждому ЗНАЧЕНИЮ растра (0-based) цвет и метку.

    Растр v5: value = index в class_levels (0..n-1).
    Исходная палитра пронумерована 1..24, поэтому
    для value=v берём цвет класса палитры (v+1).

    Возвращает список словарей: {value, label, hex, r, g, b}.
    """
    entries = []
    for value, name in enumerate(class_levels):
        palette_key = value + 1                       # 0-based value -> 1-based класс палитры
        hex_color = PALETTE_HEX.get(palette_key, PALETTE_FALLBACK)
        r, g, b = _hex_to_rgb(hex_color)
        entries.append({
            "value": value,
            "label": f"{value} - {name}",
            "hex": hex_color,
            "r": r, "g": g, "b": b,
        })
    return entries


def _paletted_renderer_xml(palette_entries: list[dict], band: int = 1) -> str:
    """XML блок rasterrenderer type=paletted для QGIS (.qgs / .qml)."""
    pal_items = []
    for e in palette_entries:
        pal_items.append(
            f'          <paletteEntry '
            f'value="{e["value"]}" '
            f'color="{e["hex"]}" '
            f'alpha="255" '
            f'label="{_xml_escape(e["label"])}"/>'
        )
    pal_block = "\n".join(pal_items)

    color_ramp_items = []
    for e in palette_entries:
        color_ramp_items.append(
            f'          <item '
            f'value="{e["value"]}" '
            f'color="{e["hex"]}" '
            f'alpha="255" '
            f'label="{_xml_escape(e["label"])}"/>'
        )
    ramp_block = "\n".join(color_ramp_items)

    return (
        f'      <rasterrenderer type="paletted" '
        f'band="{band}" opacity="1" alphaBand="-1" nodataColor="">\n'
        f'        <rasterTransparency/>\n'
        f'        <minMaxOrigin>\n'
        f'          <limits>None</limits>\n'
        f'          <extent>WholeRaster</extent>\n'
        f'          <statAccuracy>Estimated</statAccuracy>\n'
        f'          <cumulativeCutLower>0.02</cumulativeCutLower>\n'
        f'          <cumulativeCutUpper>0.98</cumulativeCutUpper>\n'
        f'          <stdDevFactor>2</stdDevFactor>\n'
        f'        </minMaxOrigin>\n'
        f'        <colorPalette>\n{pal_block}\n        </colorPalette>\n'
        f'        <colorramp type="randomcolors" name="[source]"/>\n'
        f'      </rasterrenderer>'
    )


def _xml_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _write_qml_style(qml_path: str, palette_entries: list[dict]) -> None:
    """Пишет QGIS QML (стиль слоя) — переносимый сайдкар для растра."""
    renderer = _paletted_renderer_xml(palette_entries, band=1)
    qml = (
        '<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'>\n'
        '<qgis version="3.28" styleCategories="AllStyleCategories">\n'
        '  <pipe>\n'
        f'{renderer}\n'
        '    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>\n'
        '    <huesaturation saturation="0" grayscaleMode="0" colorizeOn="0"/>\n'
        '    <rasterresampler maxOversampling="2"/>\n'
        '  </pipe>\n'
        '  <blendMode>0</blendMode>\n'
        '</qgis>\n'
    )
    with open(qml_path, "w", encoding="utf-8") as f:
        f.write(qml)


def _write_qgis_project(
    qgs_path: str,
    raster_path: str,
    layer_name: str,
    palette_entries: list[dict],
    crs_wkt: str,
    crs_authid: str,
    extent: tuple[float, float, float, float],
) -> None:
    """
    Пишет минимальный, но корректный QGIS-проект (.qgs) с одним
    растровым слоем и paletted-раскраской.

    Путь к растру записывается ОТНОСИТЕЛЬНым (рядом с проектом),
    чтобы папку 04_FINAL можно было переносить целиком.
    """
    import uuid
    xmin, ymin, xmax, ymax = extent
    layer_id = "raster_" + uuid.uuid4().hex
    rel_source = os.path.basename(raster_path)     # растр лежит в той же папке
    renderer = _paletted_renderer_xml(palette_entries, band=1)

    srs_block = (
        f'        <spatialrefsys>\n'
        f'          <wkt>{_xml_escape(crs_wkt)}</wkt>\n'
        f'          <authid>{_xml_escape(crs_authid)}</authid>\n'
        f'          <srsid>0</srsid>\n'
        f'          <geographicflag>false</geographicflag>\n'
        f'        </spatialrefsys>'
    )

    proj = f"""<?xml version="1.0" encoding="UTF-8"?>
<qgis projectname="{_xml_escape(layer_name)}" version="3.28.0">
  <homePath path=""/>
  <title>{_xml_escape(layer_name)}</title>
  <transaction mode="Disabled"/>
  <projectFlags set=""/>
  <projectCrs>
    <spatialrefsys>
      <wkt>{_xml_escape(crs_wkt)}</wkt>
      <authid>{_xml_escape(crs_authid)}</authid>
      <geographicflag>false</geographicflag>
    </spatialrefsys>
  </projectCrs>
  <layer-tree-group>
    <customproperties/>
    <layer-tree-layer id="{layer_id}" name="{_xml_escape(layer_name)}"
        source="./{_xml_escape(rel_source)}" providerKey="gdal" checked="Qt::Checked"
        expanded="1" legend_exp="" patch_size="-1,-1">
      <customproperties/>
    </layer-tree-layer>
    <custom-order enabled="0">
      <item>{layer_id}</item>
    </custom-order>
  </layer-tree-group>
  <snapping-settings enabled="0"/>
  <mapcanvas name="theMapCanvas" annotationsVisible="1">
    <units>meters</units>
    <extent>
      <xmin>{xmin}</xmin>
      <ymin>{ymin}</ymin>
      <xmax>{xmax}</xmax>
      <ymax>{ymax}</ymax>
    </extent>
    <rotation>0</rotation>
    <destinationsrs>
      <spatialrefsys>
        <wkt>{_xml_escape(crs_wkt)}</wkt>
        <authid>{_xml_escape(crs_authid)}</authid>
        <geographicflag>false</geographicflag>
      </spatialrefsys>
    </destinationsrs>
  </mapcanvas>
  <legend updateDrawingOrder="true">
    <legendlayer drawingOrder="-1" open="true" checked="Qt::Checked"
        name="{_xml_escape(layer_name)}" showFeatureCount="0">
      <filegroup open="true" hidden="false">
        <legendlayerfile isInOverview="0" layerid="{layer_id}" visible="1"/>
      </filegroup>
    </legendlayer>
  </legend>
  <projectlayers>
    <maplayer type="raster" hasScaleBasedVisibilityFlag="0"
        minScale="1e+08" maxScale="0" autoRefreshTime="0"
        autoRefreshMode="Disabled" refreshOnNotifyEnabled="0">
      <id>{layer_id}</id>
      <datasource>./{_xml_escape(rel_source)}</datasource>
      <layername>{_xml_escape(layer_name)}</layername>
      <provider>gdal</provider>
      <srs>
{srs_block}
      </srs>
      <extent>
        <xmin>{xmin}</xmin>
        <ymin>{ymin}</ymin>
        <xmax>{xmax}</xmax>
        <ymax>{ymax}</ymax>
      </extent>
      <customproperties>
        <Option type="Map">
          <Option name="identify/format" value="Value" type="QString"/>
        </Option>
      </customproperties>
      <pipe>
{renderer}
        <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
        <huesaturation saturation="0" grayscaleMode="0" colorizeOn="0"
            colorizeRed="255" colorizeGreen="128" colorizeBlue="128"
            colorizeStrength="100"/>
        <rasterresampler maxOversampling="2"/>
      </pipe>
      <blendMode>0</blendMode>
    </maplayer>
  </projectlayers>
  <properties/>
</qgis>
"""
    with open(qgs_path, "w", encoding="utf-8") as f:
        f.write(proj)


# =============================================================================
# PREDICT ПО БЛОКАМ (чтобы не загружать весь растр в память)
# =============================================================================

def _predict_raster_blockwise(
    raster_path: str,
    model,
    predictor_names: list,
    class_levels: list,
    le: LabelEncoder,
    n_cores: int,
    block_rows: int = 512,
) -> tuple[str, str]:
    """
    Предсказание по блокам. Возвращает пути к двум temp-файлам:
    (categorical_tif, probability_tif).
    Они будут потом скопированы в output_dir с нужным именем.
    """
    with rasterio.open(raster_path) as src:
        height = src.height
        width = src.width
        profile = src.profile.copy()

        band_names = [
            src.descriptions[i] if src.descriptions[i] else f"band{i + 1}"
            for i in range(src.count)
        ]

        missing = [nm for nm in predictor_names if nm not in band_names]
        if missing:
            raise RuntimeError(
                "В растре отсутствуют слои, необходимые модели: "
                + ", ".join(missing)
            )

        # rasterio использует 1-based индексы band-ов. Читаем только те слои,
        # которые реально ожидает модель, и строго в порядке обучения.
        band_indexes = [band_names.index(nm) + 1 for nm in predictor_names]
        n_bands = len(band_indexes)

        if n_bands != len(predictor_names):
            raise RuntimeError("Внутренняя ошибка сопоставления band-ов и предикторов.")

    # Создаём выходные файлы (temp)
    cat_profile = profile.copy()
    cat_profile.update(
        count=1, dtype="int16", nodata=-9999,
        compress="lzw", tiled=True, blockxsize=256, blockysize=256,
    )

    prob_profile = profile.copy()
    prob_profile.update(
        count=1, dtype="float32", nodata=-9999.0,
        compress="lzw", tiled=True, blockxsize=256, blockysize=256,
    )

    cat_tmp = raster_path + "_cat_tmp.tif"
    prob_tmp = raster_path + "_prob_tmp.tif"

    total_blocks = int(np.ceil(height / block_rows))
    processed_pixels = 0

    with rasterio.open(raster_path) as src, \
         rasterio.open(cat_tmp, "w", **cat_profile) as dst_cat, \
         rasterio.open(prob_tmp, "w", **prob_profile) as dst_prob:

        for block_idx in range(total_blocks):
            row_start = block_idx * block_rows
            row_end = min(row_start + block_rows, height)
            actual_rows = row_end - row_start

            window = Window(0, row_start, width, actual_rows)

            # Читаем только предикторы модели: (n_bands, actual_rows, width)
            data = src.read(indexes=band_indexes, window=window).astype(np.float32)

            # Reshape: (n_pixels, n_bands)
            pixels = data.reshape(n_bands, -1).T  # (n_pixels, n_bands)
            n_pixels = pixels.shape[0]

            # Маска валидных пикселей (все бенды конечные)
            valid_mask = np.all(np.isfinite(pixels), axis=1)

            # Выходные массивы
            cat_block = np.full(n_pixels, -9999, dtype=np.int16)
            prob_block = np.full(n_pixels, -9999.0, dtype=np.float32)

            valid_count = valid_mask.sum()
            if valid_count > 0:
                X_valid = pixels[valid_mask]

                # Предсказание классов
                y_pred = model.predict(X_valid)
                # Обратно в строковые метки -> int ID
                y_labels = le.inverse_transform(y_pred)
                # ID = индекс в class_levels (0-based, БЕЗ +1)
                cat_vals = np.array(
                    [class_levels.index(lbl) for lbl in y_labels],
                    dtype=np.int16,
                )
                cat_block[valid_mask] = cat_vals

                # Вероятности -> макс.
                try:
                    y_proba = model.predict_proba(X_valid)
                    prob_block[valid_mask] = y_proba.max(axis=1).astype(np.float32)
                except Exception:
                    # Если predict_proba не поддерживается
                    prob_block[valid_mask] = 1.0

            # Записываем в GeoTIFF
            cat_2d = cat_block.reshape(actual_rows, width)
            prob_2d = prob_block.reshape(actual_rows, width)

            dst_cat.write(cat_2d.astype(np.int16), 1, window=window)
            dst_prob.write(prob_2d.astype(np.float32), 1, window=window)

            processed_pixels += n_pixels
            pct = (block_idx + 1) / total_blocks * 100
            print(
                f"\r  Блок {block_idx + 1}/{total_blocks} ({pct:.0f}%)   ",
                end="", flush=True,
            )

    print()
    return cat_tmp, prob_tmp


# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================

def main():
    print("\n============================================================")
    print("  04 - ЭКСТРАПОЛЯЦИЯ КЛАССИФИКАЦИИ")
    print("============================================================\n")

    root_dir = os.getcwd()

    # ================================================================
    # ШАГ 1: ВЫБОР ПАПКИ 03_TUNE
    # ================================================================
    print("============================")
    print("  03_TUNE: выбор запуска")
    print("============================")

    tune_base_dir = os.path.join(root_dir, "03_TUNE")
    selected_tune_dir, version_suffix = _select_prev_subdir(tune_base_dir)
    tune_basename = os.path.basename(selected_tune_dir)

    # --- Создание подпапки 04_FINAL ---
    output_base = os.path.join(root_dir, "04_FINAL")
    os.makedirs(output_base, exist_ok=True)

    run_stamp = datetime.now().strftime("%d%m%y_%H%M")
    run_folder = f"{run_stamp}{version_suffix}"
    output_dir = os.path.join(output_base, run_folder)
    os.makedirs(output_dir, exist_ok=True)

    print(f"INPUT 03_TUNE : {selected_tune_dir}")
    print(f"OUTPUT 04_FINAL: {output_dir}\n")

    # ================================================================
    # ШАГ 2: ЗАГРУЗКА МОДЕЛИ
    # ================================================================
    print("============================================================")
    print("  ЗАГРУЗКА МОДЕЛИ")
    print("============================================================\n")

    model_file = os.path.join(selected_tune_dir, "final_model.pkl")
    final_model = None
    model_name = "model"

    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            final_model = pickle.load(f)
        # Пытаемся определить имя модели
        if hasattr(final_model, "named_steps"):
            est = final_model.named_steps.get("model", final_model)
            model_name = type(est).__name__
        else:
            model_name = type(final_model).__name__
        print(f"  ✓ Модель: {model_name}")
    else:
        all_models_file = os.path.join(selected_tune_dir, "all_trained_models.pkl")
        if os.path.exists(all_models_file):
            print("  ! final_model.pkl не найден, загружаю all_trained_models.pkl\n")
            with open(all_models_file, "rb") as f:
                all_models = pickle.load(f)
            names = list(all_models.keys())
            for i, n in enumerate(names, 1):
                print(f"  [{i}] {n}")
            mod_ch = input("\n  Выберите модель (Enter = 1): ").strip()
            if mod_ch == "":
                mod_ch = "1"
            mod_idx = int(mod_ch) - 1
            if mod_idx < 0 or mod_idx >= len(names):
                mod_idx = 0
            model_name = names[mod_idx]
            final_model = all_models[model_name]
            print(f"\n  ✓ Выбрана: {model_name}")
        else:
            raise FileNotFoundError(
                "Не найден ни final_model.pkl, ни all_trained_models.pkl!"
            )

    # --- Метрики из model_comparison.csv ---
    accuracy_value = None
    kappa_value = None
    aic_value = None
    delta_aic = None

    model_comp_file = os.path.join(selected_tune_dir, "model_comparison.csv")
    if os.path.exists(model_comp_file):
        mc = pd.read_csv(model_comp_file)
        print(f"  ✓ model_comparison.csv: {len(mc)} моделей")

        # Ищем строку с нашей моделью
        model_row = None
        for _, row in mc.iterrows():
            if (model_name.lower() in str(row.get("Algorithm", "")).lower()
                    or model_name.lower() in str(row.get("Method", "")).lower()):
                model_row = row
                break
        if model_row is None and len(mc) > 0:
            model_row = mc.iloc[0]

        if model_row is not None:
            accuracy_value = round(model_row.get("OverallAccuracy", 0), 3)
            kappa_value = round(model_row.get("Kappa", 0), 3)
            if "AIC" in model_row and pd.notna(model_row["AIC"]):
                aic_value = model_row["AIC"]
                aic_all = mc["AIC"].dropna()
                if len(aic_all) > 0:
                    delta_aic = round(aic_value - aic_all.min(), 1)

    if accuracy_value is None:
        accuracy_value = 0
    if kappa_value is None:
        kappa_value = 0

    # --- Confusion matrix из 03_TUNE ---
    cm_arr = None
    cm_file = os.path.join(selected_tune_dir, "confusion_matrices.pkl")
    if os.path.exists(cm_file):
        with open(cm_file, "rb") as f:
            all_cms = pickle.load(f)
        if all_cms:
            # Ищем по имени модели
            cm_key = None
            for k in all_cms:
                if model_name.lower() in k.lower():
                    cm_key = k
                    break
            if cm_key is None:
                cm_key = list(all_cms.keys())[0]
            cm_df = all_cms[cm_key]
            if isinstance(cm_df, pd.DataFrame):
                cm_arr = cm_df.values

    # --- Предикторы ---
    # Извлекаем из data_info.pkl (02_SELECT) или из модели
    data_info_path = None
    # Ищем data_info.pkl в 02_SELECT с тем же суффиксом
    select_base = os.path.join(root_dir, "02_SELECT")
    predictor_cols = None
    class_levels = None
    selected_select_dir = None

    if os.path.isdir(select_base):
        select_subdirs = sorted([
            d for d in os.listdir(select_base)
            if os.path.isdir(os.path.join(select_base, d))
            and d.endswith(version_suffix)
        ])
        if select_subdirs:
            selected_select_dir = os.path.join(select_base, select_subdirs[-1])
            di_path = os.path.join(selected_select_dir, "data_info.pkl")
            if os.path.exists(di_path):
                with open(di_path, "rb") as f:
                    data_info = pickle.load(f)
                predictor_cols = data_info.get("predictor_cols")
                class_levels = data_info.get("class_levels")

    # Fallback: из модели features
    if predictor_cols is None and hasattr(final_model, "feature_names_in_"):
        predictor_cols = list(final_model.feature_names_in_)

    if predictor_cols:
        print(f"  ✓ Предикторов в модели: {len(predictor_cols)}")
    if class_levels:
        print(f"  ✓ Классов: {len(class_levels)} ({', '.join(class_levels)})")

    daic_str = "N/A" if delta_aic is None else str(delta_aic)
    print(f"  === OA={accuracy_value} | Kappa={kappa_value} | dAIC={daic_str} ===")

    # ================================================================
    # ШАГ 3: РАСТР
    # ================================================================
    print("\n============================================================")
    print("  ЗАГРУЗКА РАСТРА")
    print("============================================================\n")

    raster_path = None
    raster_source = ""

    # 1) Ищем raster_info.pkl в 01_PREPARE_DATA с нужным суффиксом
    prepare_base = os.path.join(root_dir, "01_PREPARE_DATA")
    if os.path.isdir(prepare_base):
        prep_subdirs = sorted([
            d for d in os.listdir(prepare_base)
            if os.path.isdir(os.path.join(prepare_base, d))
            and d.endswith(version_suffix)
        ])
        if prep_subdirs:
            ri_path = os.path.join(prepare_base, prep_subdirs[-1], "raster_info.pkl")
            if os.path.exists(ri_path):
                with open(ri_path, "rb") as f:
                    raster_info = pickle.load(f)
                print(f"✓ raster_info.pkl")
                print(f"  {raster_info.get('raster_name', '?')}")
                print(
                    f"  {raster_info.get('n_layers', '?')} слоёв: "
                    f"{', '.join(raster_info.get('layer_names', []))}"
                )
                candidate = raster_info.get("raster_path", "")
                if os.path.exists(candidate):
                    raster_path = candidate
                    raster_source = "raster_info.pkl (01)"
                else:
                    print(f"  ! {candidate} не найден.")

    # 2) Fallback: ищем TIF в 00_PREPARE_LAYERS
    if raster_path is None:
        print("! raster_info.pkl не найден или не подходит.")
        print("  Ищу TIF в 00_PREPARE_LAYERS...\n")

        prep_layers_root = os.path.join(root_dir, "00_PREPARE_LAYERS")
        if not os.path.isdir(prep_layers_root):
            raise FileNotFoundError("Папка 00_PREPARE_LAYERS не найдена!")

        prep_layer_subdirs = sorted([
            d for d in os.listdir(prep_layers_root)
            if os.path.isdir(os.path.join(prep_layers_root, d))
            and d.endswith(version_suffix)
        ])
        if not prep_layer_subdirs:
            raise FileNotFoundError(
                f"Не найдена подпапка в 00_PREPARE_LAYERS с суффиксом {version_suffix}."
            )
        prep_dir = os.path.join(prep_layers_root, prep_layer_subdirs[-1])
        print(f"Использую папку 00_PREPARE_LAYERS: {prep_dir}")

        import glob
        tif_files = sorted(glob.glob(os.path.join(prep_dir, "*.tif")))
        if not tif_files:
            raise FileNotFoundError(
                "TIF не найдены в соответствующей подпапке 00_PREPARE_LAYERS!"
            )

        print("Найдены TIF:")
        for i, tf in enumerate(tif_files, 1):
            fsize = os.path.getsize(tf) / (1024 ** 2)
            try:
                with rasterio.open(tf) as src:
                    nlyr = src.count
            except Exception:
                nlyr = "?"
            print(f" {i:2d}) {os.path.basename(tf)}  {nlyr}  {fsize:.1f} MB")

        rchoice = input("Выберите номер TIF (Enter = 1): ").strip()
        if rchoice == "":
            rchoice = "1"
        ridx = int(rchoice) - 1
        if ridx < 0 or ridx >= len(tif_files):
            ridx = 0

        raster_path = tif_files[ridx]
        raster_source = f"00_PREPARE_LAYERS/{os.path.basename(prep_dir)}"

    if raster_path is None:
        raise RuntimeError("Не удалось определить входной TIF для предсказания.")

    expected_n_features = _get_model_n_features(final_model)
    if expected_n_features is not None:
        current_brief = _read_raster_brief(raster_path)
        if current_brief is not None and current_brief["n_bands"] != expected_n_features:
            print(
                f"  ! Выбранный растр содержит {current_brief['n_bands']} слоёв, "
                f"а модель ожидает {expected_n_features}."
            )
            matched_raster, matched_source = _find_matching_composite_raster(
                root_dir=root_dir,
                version_suffix=version_suffix,
                expected_n=expected_n_features,
                current_path=raster_path,
            )
            if matched_raster and os.path.abspath(matched_raster) != os.path.abspath(raster_path):
                raster_path = matched_raster
                raster_source = matched_source or "auto-match"
                print(f"  ✓ Переключаюсь на: {raster_path}")

    # Читаем метаданные растра
    with rasterio.open(raster_path) as src:
        raster_n_bands = src.count
        raster_height = src.height
        raster_width = src.width
        total_cells = raster_height * raster_width
        band_names = [
            src.descriptions[i] if src.descriptions[i] else f"band{i + 1}"
            for i in range(raster_n_bands)
        ]

    predictor_names = _resolve_predictor_names(
        final_model=final_model,
        predictor_cols=predictor_cols,
        band_names=band_names,
        selected_tune_dir=selected_tune_dir,
        select_dir=selected_select_dir,
    )
    n_predictors = len(predictor_names)

    print(
        f"✓ {os.path.basename(raster_path)}: "
        f"{raster_height} x {raster_width}, {raster_n_bands} слоёв "
        f"({raster_source})\n"
    )
    if n_predictors != raster_n_bands:
        skipped = [nm for nm in band_names if nm not in predictor_names]
        print(
            f"  ✓ Для модели будет использовано {n_predictors} из "
            f"{raster_n_bands} слоёв растра."
        )
        if skipped:
            print(f"  ! Не используются: {', '.join(skipped)}")
        print()
    else:
        print(f"  ✓ Для модели будет использовано {n_predictors} слоёв.\n")

    # ================================================================
    # ШАГ 4: ПАРАЛЛЕЛИЗАЦИЯ
    # ================================================================
    print("============================================================")
    print("  НАСТРОЙКА ПАРАЛЛЕЛИЗАЦИИ")
    print("============================================================\n")

    total_cores = os.cpu_count() or 1
    recommended = max(1, total_cores - 1)

    print(f"Обнаружено ядер (потоков): {total_cores}\n")
    print(f"  [1] {recommended} ядер - РЕКОМЕНДУЕМО")
    if total_cores >= 4:
        print(f"  [2] {max(1, total_cores // 2)} ядер - половина")
    print(f"  [3] 1 ядро - последовательный режим")
    print(f"  [4] Ввести число (1-{total_cores})\n")

    cores_choice = input(f"Выбор (Enter = {recommended}): ").strip()

    if cores_choice in ("", "1"):
        n_cores = recommended
    elif cores_choice == "2" and total_cores >= 4:
        n_cores = max(1, total_cores // 2)
    elif cores_choice == "3":
        n_cores = 1
    else:
        try:
            n_cores = int(cores_choice)
            n_cores = max(1, min(n_cores, total_cores))
        except ValueError:
            n_cores = recommended

    print(f"\n✓ Ядер: {n_cores} из {total_cores}\n")

    # ================================================================
    # ШАГ 5: ИМЕНА ФАЙЛОВ
    # ================================================================
    print("[ШАГ 5] Имена файлов...")

    algo_abbr = _get_algo_abbr(model_name)
    oa_code = f"{round(accuracy_value * 1000):03d}"
    date_str = datetime.now().strftime("%y%m%d")
    time_str = datetime.now().strftime("%H%M")
    base_name = f"{algo_abbr}_{oa_code}_{date_str}_{time_str}"

    file_categorical = os.path.join(output_dir, f"{base_name}_cat.tif")
    file_probability = os.path.join(output_dir, f"{base_name}_prob.tif")
    file_map = os.path.join(output_dir, f"{base_name}_map.png")
    file_confmat = os.path.join(output_dir, f"{base_name}_confmat.png")
    file_stats = os.path.join(output_dir, f"{base_name}_stats.csv")
    file_qgs = os.path.join(output_dir, f"{base_name}.qgs")
    file_qml = os.path.join(output_dir, f"{base_name}_cat.tif.qml")

    print(f"  ✓ {base_name}")

    # ================================================================
    # ШАГ 6: КЛАССИФИКАЦИЯ
    # ================================================================
    print(f"\n[ШАГ 6] Классификация...")

    # Подготовка LabelEncoder
    if class_levels is None:
        # Fallback: попробуем из модели
        if hasattr(final_model, "classes_"):
            class_levels = [str(c) for c in final_model.classes_]
        elif hasattr(final_model, "named_steps"):
            est = final_model.named_steps.get("model", final_model)
            if hasattr(est, "classes_"):
                class_levels = [str(c) for c in est.classes_]

    if class_levels is None:
        raise RuntimeError(
            "Не удалось определить список классов. "
            "Проверьте data_info.pkl в 02_SELECT."
        )

    le = LabelEncoder()
    le.fit(class_levels)

    # Оценка времени
    speed_map = {
        "RandomForest": 0.08, "CatBoost": 0.12,
        "LogisticRegression": 0.05, "LinearDiscriminant": 0.04,
        "QuadraticDiscriminant": 0.04,
    }
    spd = 0.15
    for pattern, s in speed_map.items():
        if pattern.lower() in model_name.lower():
            spd = s
            break

    core_f = max(0.3, 1 / (n_cores * 0.6)) if n_cores > 1 else 1
    pred_f = max(0.8, 1 + (n_predictors - 6) * 0.05)
    est_min = (total_cells / 1000) * spd * core_f * pred_f / 60

    if est_min < 1:
        time_est = "< 1 мин"
    elif est_min < 60:
        time_est = f"~{int(np.ceil(est_min))} мин"
    else:
        time_est = f"~{int(est_min // 60)} ч {int(np.ceil(est_min % 60))} мин"

    print(
        f"  Ядер: {n_cores} | "
        f"Ячеек: {total_cells:,} | "
        f"Метод: {model_name} | {time_est}"
    )

    start_time = time.time()

    cat_tmp, prob_tmp = _predict_raster_blockwise(
        raster_path=raster_path,
        model=final_model,
        predictor_names=predictor_names,
        class_levels=class_levels,
        le=le,
        n_cores=n_cores,
        block_rows=512,
    )

    elapsed = (time.time() - start_time) / 60
    print(f"  ✓ Классификация за {elapsed:.2f} мин")

    # ================================================================
    # ШАГ 7: СОХРАНЕНИЕ РАСТРОВ
    # ================================================================
    print(f"\n[ШАГ 7] Растры...")

    # Переименование temp → final
    import shutil
    shutil.move(cat_tmp, file_categorical)
    shutil.move(prob_tmp, file_probability)

    n_classes = len(class_levels)
    class_table = pd.DataFrame({
        "ID": range(0, n_classes),          # 0-based
        "Class": class_levels,
    })

    print(
        f"  ✓ Классов: {n_classes} ({', '.join(class_levels)})"
    )
    print(f"  ✓ {os.path.basename(file_categorical)}")
    print(f"  ✓ {os.path.basename(file_probability)}")

    # Статистика вероятностей
    with rasterio.open(file_probability) as src:
        prob_data = src.read(1)
    valid_probs = prob_data[prob_data > -9998]
    if len(valid_probs) > 0:
        print(
            f"  Вероятности: min={valid_probs.min():.3f}, "
            f"mean={valid_probs.mean():.3f}, max={valid_probs.max():.3f}"
        )

    # ================================================================
    # ШАГ 7b: QGIS-ПРОЕКТ + СТИЛЬ (раскраска по палитре)
    # ================================================================
    print(f"\n[ШАГ 7b] QGIS-проект...")
    try:
        palette_entries = _build_palette_for_raster(class_levels)

        # CRS и extent берём из готового категориального растра
        with rasterio.open(file_categorical) as src:
            crs = src.crs
            b = src.bounds
        crs_wkt = crs.to_wkt() if crs else ""
        try:
            crs_authid = crs.to_authority() if crs else None
            crs_authid = ":".join(crs_authid) if crs_authid else ""
        except Exception:
            crs_authid = ""
        extent = (b.left, b.bottom, b.right, b.top)

        layer_name = f"{base_name}_cat"

        # 1) QML-сайдкар (автоподхватывается QGIS рядом с .tif)
        _write_qml_style(file_qml, palette_entries)
        print(f"  ✓ {os.path.basename(file_qml)} (стиль слоя)")

        # 2) QGIS-проект .qgs с встроенной раскраской
        _write_qgis_project(
            qgs_path=file_qgs,
            raster_path=file_categorical,
            layer_name=layer_name,
            palette_entries=palette_entries,
            crs_wkt=crs_wkt,
            crs_authid=crs_authid,
            extent=extent,
        )
        print(f"  ✓ {os.path.basename(file_qgs)} (проект QGIS)")
        print(f"    Цветов в палитре: {len(palette_entries)} (значения 0..{n_classes - 1})")
    except Exception as e:
        print(f"  ! QGIS-проект не создан: {e}")

    # ================================================================
    # ШАГ 8: СТАТИСТИКА КЛАССОВ
    # ================================================================
    print(f"\n[ШАГ 8] Статистика...\n")

    with rasterio.open(file_categorical) as src:
        cat_data = src.read(1)

    # Подсчёт пикселей по классам (0-based: валидными считаем всё, кроме nodata)
    valid_cat = cat_data[cat_data != -9999]
    total_pixels = len(valid_cat)

    class_freq_rows = []
    print("--------------------------------------------------")
    print(f"{'Класс':<20s} {'Пикселей':>12s} {'%':>10s}")
    print("--------------------------------------------------")
    for cid in range(0, n_classes):
        count = int((valid_cat == cid).sum())
        cname = class_levels[cid]
        pct = count / max(total_pixels, 1) * 100
        print(f"{cname:<20s} {count:>12d} {pct:>9.2f}%")
        class_freq_rows.append({
            "value": cid, "Class": cname, "count": count, "Percent": pct,
        })
    print("--------------------------------------------------")
    print(f"{'ВСЕГО':<20s} {total_pixels:>12d} {'100.00':>9s}%")
    print("--------------------------------------------------\n")

    class_freq = pd.DataFrame(class_freq_rows)

    # ================================================================
    # ШАГ 9: CONFUSION MATRIX
    # ================================================================
    print("[ШАГ 9] Confusion Matrix...")

    conf_matrix_created = False
    class_metrics_df = None

    if cm_arr is not None:
        try:
            n_cls = cm_arr.shape[0]
            cm_norm = cm_arr / np.maximum(cm_arr.sum(axis=1, keepdims=True), 1)

            # Per-class метрики
            sensitivities = []
            specificities = []
            precisions_arr = []
            f1s = []
            balanced_accs = []

            for ci in range(n_cls):
                tp = cm_arr[ci, ci]
                fn = cm_arr[ci, :].sum() - tp
                fp = cm_arr[:, ci].sum() - tp
                tn = cm_arr.sum() - tp - fn - fp
                sens = tp / max(tp + fn, 1)
                spec = tn / max(tn + fp, 1)
                prec = tp / max(tp + fp, 1)
                f1 = 2 * prec * sens / max(prec + sens, 1e-10)
                ba = (sens + spec) / 2
                sensitivities.append(round(sens, 3))
                specificities.append(round(spec, 3))
                precisions_arr.append(round(prec, 3))
                f1s.append(round(f1, 3))
                balanced_accs.append(round(ba, 3))

            class_metrics_df = pd.DataFrame({
                "Class": class_levels[:n_cls],
                "Prec": precisions_arr,
                "Recall": sensitivities,
                "F1": f1s,
                "Sens": sensitivities,
                "Spec": specificities,
                "BA": balanced_accs,
            })

            # --- Рисуем Confusion Matrix PNG ---
            metric_cols = ["Prec", "Recall", "F1", "Sens", "Spec", "BA"]
            n_met = len(metric_cols)
            total_rows = n_met + 1 + n_cls  # metrics + separator + CM

            fig_h = max(8, (total_rows) * 0.6 + 3)
            fig_w = max(8, n_cls * 0.9 + 4)

            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            # Формируем матрицу для отображения
            disp = np.zeros((total_rows, n_cls))
            for i in range(n_cls):
                for j in range(n_cls):
                    disp[n_met + 1 + i, j] = cm_norm[i, j]
            for m_idx, mc in enumerate(metric_cols):
                for j in range(n_cls):
                    disp[m_idx, j] = class_metrics_df.iloc[j][mc]

            # Background
            cm_cmap = mcolors.LinearSegmentedColormap.from_list(
                "cm", ["white", "#C6DBEF", "#4292C6", "#08519C"]
            )
            ax.imshow(disp, cmap=cm_cmap, aspect="auto", vmin=0, vmax=1)

            # Метрики (верхняя часть) — цветные фоны
            for m_idx, mc in enumerate(metric_cols):
                for j in range(n_cls):
                    val = class_metrics_df.iloc[j][mc]
                    bg_color = _color_for_metric(val, mc)
                    rect = plt.Rectangle(
                        (j - 0.5, m_idx - 0.5), 1, 1,
                        facecolor=bg_color, edgecolor="white", linewidth=0.5,
                    )
                    ax.add_patch(rect)
                    ax.text(
                        j, m_idx, f"{val:.3f}",
                        ha="center", va="center",
                        fontsize=max(6, 9 - n_cls * 0.2),
                        fontweight="bold", color="black",
                    )

            # Разделительная линия
            ax.axhline(y=n_met - 0.5 + 0.5, color="red", linewidth=2)

            # CM (нижняя часть)
            for i in range(n_cls):
                for j in range(n_cls):
                    row_pos = n_met + 1 + i
                    val = cm_arr[i, j]
                    pct = cm_norm[i, j] * 100
                    ax.text(
                        j, row_pos,
                        f"{val}\n({pct:.1f}%)",
                        ha="center", va="center",
                        fontsize=max(6, 9 - n_cls * 0.2),
                        color="black",
                    )

            # Оси
            y_labels = metric_cols + ["---"] + class_levels[:n_cls]
            ax.set_yticks(range(total_rows))
            ax.set_yticklabels(y_labels, fontsize=8)
            ax.set_xticks(range(n_cls))
            ax.set_xticklabels(class_levels[:n_cls], rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("Predicted")

            aic_txt = "" if delta_aic is None else f" | dAIC: {delta_aic}"
            ax.set_title(
                f"Confusion Matrix + Metrics\n"
                f"{model_name} | OA: {accuracy_value} | Kappa: {kappa_value}{aic_txt}",
                fontweight="bold", fontsize=11,
            )

            # Сетка
            for h in np.arange(n_met + 1.5, n_met + n_cls + 0.5):
                ax.axhline(y=h, color="gray", linewidth=0.3, linestyle="--")
            for v in np.arange(-0.5, n_cls + 0.5):
                ax.axvline(x=v, color="gray", linewidth=0.3, linestyle="--")

            plt.tight_layout()
            plt.savefig(file_confmat, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)

            print(f"  ✓ {os.path.basename(file_confmat)}")
            conf_matrix_created = True

        except Exception as e:
            print(f"  ! CM: {e}")
    else:
        print("  ! Confusion matrix не найдена в 03_TUNE.")

    # ================================================================
    # ШАГ 10: КАРТА
    # ================================================================
    print(f"\n[ШАГ 10] Карта...")

    try:
        with rasterio.open(file_categorical) as src:
            cat_data = src.read(1)
            transform = src.transform
            bounds = src.bounds

        # Цвета для классов
        n_cls = len(class_levels)
        if n_cls <= 10:
            cmap_cls = plt.cm.Set3
        elif n_cls <= 20:
            cmap_cls = plt.cm.tab20
        else:
            cmap_cls = plt.cm.nipy_spectral
        colors = [cmap_cls(i / max(n_cls - 1, 1)) for i in range(n_cls)]

        # Маска nodata (0-based: класс 0 валиден, прячем только -9999)
        cat_float = cat_data.astype(float)
        cat_float[cat_data == -9999] = np.nan

        fig, ax = plt.subplots(figsize=(12, 10))

        # Создаём custom colormap
        from matplotlib.colors import ListedColormap, BoundaryNorm
        cmap_custom = ListedColormap(colors)
        # 0-based: границы -0.5 .. (n_cls-0.5), чтобы значения 0..n_cls-1
        # попадали ровно в свои ячейки цветовой шкалы
        bounds_arr = np.arange(-0.5, n_cls + 0.5, 1)
        norm = BoundaryNorm(bounds_arr, cmap_custom.N)

        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        im = ax.imshow(
            cat_float, cmap=cmap_custom, norm=norm,
            extent=extent, interpolation="nearest",
        )

        # Легенда
        patches = [
            mpatches.Patch(color=colors[i], label=class_levels[i])
            for i in range(n_cls)
        ]
        ax.legend(
            handles=patches, loc="upper right", title="Классы",
            fontsize=8, title_fontsize=9,
        )

        aic_txt = "N/A" if delta_aic is None else str(delta_aic)
        ax.set_title(
            f"Classification: {model_name}\n"
            f"OA: {accuracy_value} | Kappa: {kappa_value} | dAIC: {aic_txt}",
            fontweight="bold", fontsize=11,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        plt.tight_layout()
        plt.savefig(file_map, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  ✓ {os.path.basename(file_map)}")

    except Exception as e:
        print(f"  ! Карта: {e}")

    # ================================================================
    # ШАГ 11: CSV
    # ================================================================
    print(f"\n[ШАГ 11] CSV...")

    metadata = pd.DataFrame({
        "Parameter": [
            "Model", "OverallAccuracy", "Kappa", "dAIC",
            "SourceFolder", "Date", "Time",
        ],
        "Value": [
            model_name, accuracy_value, kappa_value,
            "N/A" if delta_aic is None else str(delta_aic),
            tune_basename,
            datetime.now().strftime("%Y-%m-%d"),
            datetime.now().strftime("%H:%M:%S"),
        ],
    })
    metadata.to_csv(file_stats, index=False)

    # Дописываем статистику классов
    with open(file_stats, "a") as f:
        f.write("\n")
    class_freq.to_csv(file_stats, mode="a", index=False)

    if class_metrics_df is not None:
        with open(file_stats, "a") as f:
            f.write("\n")
        class_metrics_df.to_csv(file_stats, mode="a", index=False)

    print(f"  ✓ {os.path.basename(file_stats)}")

    # ================================================================
    # ШАГ 12: СОХРАНЕНИЕ ДАННЫХ ДЛЯ 05
    # ================================================================
    print(f"\n[ШАГ 12] Сохранение данных для скрипта 05...")

    # 12a. final_model.pkl
    with open(os.path.join(output_dir, "final_model.pkl"), "wb") as f:
        pickle.dump(final_model, f)
    print("  ✓ final_model.pkl")

    # 12b. samples_df.pkl (обучающие данные из 01_PREPARE_DATA)
    samples_saved = False
    if os.path.isdir(prepare_base):
        prep_subdirs = sorted([
            d for d in os.listdir(prepare_base)
            if os.path.isdir(os.path.join(prepare_base, d))
            and d.endswith(version_suffix)
        ])
        if prep_subdirs:
            samples_gpkg = os.path.join(
                prepare_base, prep_subdirs[-1], "samples.gpkg"
            )
            if os.path.exists(samples_gpkg):
                gdf = gpd.read_file(samples_gpkg, layer="samples")
                df = pd.DataFrame(gdf.drop(columns="geometry"))
                with open(os.path.join(output_dir, "samples_df.pkl"), "wb") as f:
                    pickle.dump(df, f)
                print(
                    f"  ✓ samples_df.pkl ({len(df)} obs, {len(df.columns)} cols)"
                )
                samples_saved = True
    if not samples_saved:
        print("  ! samples_df.pkl не сохранён (samples.gpkg не найден)")

    # 12c. class_table.pkl
    with open(os.path.join(output_dir, "class_table.pkl"), "wb") as f:
        pickle.dump(class_table, f)
    print("  ✓ class_table.pkl")

    # 12d. run_meta.pkl
    run_meta = {
        "model_name": model_name,
        "accuracy": accuracy_value,
        "kappa": kappa_value,
        "delta_aic": delta_aic,
        "predictors": predictor_names,
        "class_levels": class_levels,
        "source_tune": tune_basename,
        "base_name": base_name,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "run_meta.pkl"), "wb") as f:
        pickle.dump(run_meta, f)
    print("  ✓ run_meta.pkl")

    # ================================================================
    # ИТОГО
    # ================================================================
    print("\n============================================================")
    print("  ЗАВЕРШЕНО")
    print("============================================================\n")

    print(f"  OA:    {accuracy_value} ({_interp_oa(accuracy_value)})")
    print(f"  Kappa: {kappa_value} ({_interp_kappa(kappa_value)})")
    print(
        f"  dAIC:  {'N/A' if delta_aic is None else delta_aic} "
        f"({_interp_daic(delta_aic)})"
    )
    print(f"\n  Источник: {selected_tune_dir}")
    print(f"  Папка:    {output_dir}\n")

    print("  Файлы:")
    fnum = 1
    print(f"  {fnum}. {os.path.basename(file_categorical)} (классы)"); fnum += 1
    print(f"  {fnum}. {os.path.basename(file_probability)} (макс. вероятность)"); fnum += 1
    print(f"  {fnum}. {os.path.basename(file_map)}"); fnum += 1
    if conf_matrix_created:
        print(f"  {fnum}. {os.path.basename(file_confmat)}"); fnum += 1
    print(f"  {fnum}. {os.path.basename(file_stats)}"); fnum += 1
    print(f"  {fnum}. {os.path.basename(file_qgs)} (проект QGIS с раскраской)"); fnum += 1
    print(f"  {fnum}. {os.path.basename(file_qml)} (стиль слоя QGIS)"); fnum += 1
    print(f"  {fnum}. final_model.pkl (для 05)"); fnum += 1
    print(f"  {fnum}. samples_df.pkl  (для 05)"); fnum += 1
    print(f"  {fnum}. class_table.pkl (для 05)"); fnum += 1
    print(f"  {fnum}. run_meta.pkl    (для 05)")

    print("\n  -> Запустите 05_classwise_importance.py для анализа важности")
    print("============================================================\n")

    return {
        "output_dir": output_dir,
        "model_name": model_name,
        "accuracy": accuracy_value,
        "kappa": kappa_value,
        "class_levels": class_levels,
    }


# ================================================================
# ЗАПУСК
# ================================================================
if __name__ == "__main__":
    main()
