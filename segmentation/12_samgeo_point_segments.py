# -*- coding: utf-8 -*-
"""
12_samgeo_point_segments.py
===========================
Сегмент КАЖДОЙ точки по фрагментам, созданным 10_download_sat_chips.py:
SAM 2 с подсказкой-точкой (samgeo SamGeo2, automatic=False) и спектральная
фильтрация масок по расстоянию Махаланобиса — алгоритм sam_segment_v12.py,
перенесённый на фрагменты подложки, с параллельной работой на GPU.

Для каждой точки (chips_points.gpkg):
  1. Из её фрагмента читается квадрат 2R x 2R вокруг точки, R = MAX_RADIUS_M
     (радиус поиска — наибольший радиус объекта от точки).
  2. SAM 2 получает точку как положительную подсказку и возвращает 3 маски.
  3. Эталон — яркости RGB в окне радиусом REF_RADIUS_M вокруг точки (или из
     поля REF_RADIUS_FIELD). Для каждого пикселя считается расстояние
     Махаланобиса d² до эталона; «хорошие» пиксели: d² <= χ²(CONFIDENCE, 3).
     CONFIDENCE — доля пикселей самого эталона, которые проходят порог: чем
     ВЫШЕ, тем МЯГЧЕ отбор (0.99 пропускает почти всё похожее и не очень),
     чем НИЖЕ — тем строже (0.80, 0.50). При SMOOTH_M > 0 яркости перед этим
     сглаживаются (Гаусс, сигма SMOOTH_M м): сравниваются участки, а не
     отдельные пиксели, поэтому непохожие участки (дороги, прогалины, вырубки)
     отсекаются целиком, а не мелкими дырками.
  4. Для каждой маски: доля = (хорошие пиксели маски внутри круга R) / πR².
     Доля считается от площади круга, а не маски. Выбирается маска с
     наибольшей долей; если она меньше MIN_GOOD_RATIO, точка отклоняется.
     Пример: R = 150 м, 0.25 -> у объекта должно быть не меньше
     0.25 · π · 150² ≈ 1.77 га хороших пикселей.
  5. Сегмент = выбранная маска ∩ круг R ∩ хорошие пиксели; дырки меньше
     MIN_HOLE_M2 заливаются, островки меньше MIN_PART_M2 удаляются (на
     шумных снимках 0,3 м без этого в сегменте тысячи дырок-пикселей).
  6. Перекрытия сегментов разных точек устраняются (OVERLAP_MODE):
     "voronoi" — пиксель отходит ближайшей точке из тех, чьи сегменты его
     покрывают (перекрытие делится серединным перпендикуляром); "centroid" —
     как в v12, перекрытие целиком отходит точке, ближайшей к его центру
     (у плотных профилей точек оставляет обрезки в десятки м²). Точки ближе
     COINCIDENT_M друг к другу (повторные описания одного места) перекрытия
     не делят — у каждой остаётся свой сегмент. Внутри фрагмента это
     делается на его растре (быстро и точно), между соседними фрагментами —
     по полигонам при сборке результата.
  7. Части меньше MIN_PART_M2 и меньше MIN_PART_SHARE площади полигона
     удаляются (KEEP_POINT_PART — оставить только часть с точкой).
  8. В полигоны переносятся все атрибуты точки (из chips_points.gpkg).

Результат — point_segments.gpkg (UTM 37N):
  слой segments — один полигон на точку: seg_id, все атрибуты точки,
    area_m2, area_raw_m2 (до деления перекрытий), good_ratio, mask_idx,
    sam_score, ratios (доли 3 масок), ref_r_m, ref_src (field/default),
    max_r_m, edge (круг R вышел за край фрагмента);
  слой rejected — точки без сегмента с причиной и долями масок.

Параллельная работа: GPU — WORKERS_PER_GPU процессов на карту; CPU —
физические ядра / THREADS_PER_WORKER. Процесс берёт фрагмент целиком (все
его точки), фрагменты раздаются динамически, крупные — первыми.
Прогресс: точки (шт. и %), фрагменты, выделено/отклонено, скорость,
оставшееся время; по каждой точке — площадь, доля, маска или причина отказа.
Пробный запуск: TEST_N = 20 -> случайные фрагменты, в сумме ~20 точек, и
прогноз на все.
Настройки: при запуске (ASK_SETTINGS = True) после выбора папки открывается окно
со всеми параметрами сегментов, модели и запуска, с пояснениями. В нём сразу
видны площадь круга поиска, минимальный объект, порог χ² и сколько точек уже
готово с такими параметрами. Кнопки: «Как в v12» (очистка 0, centroid),
«Из шапки скрипта», «Запустить» (Enter), «Отмена» (Esc). Введённые значения
сохраняются в _points/settings12.json и подставляются при следующем запуске
в этой папке. Без графического окружения те же вопросы задаются в консоли.
Повторный запуск продолжает с места остановки. Если изменить параметры
шагов 1–6 (модель, MAX_RADIUS_M, CONFIDENCE, MIN_GOOD_RATIO, REF_RADIUS_M,
REF_RADIUS_FIELD, SMOOTH_M, ENHANCE, MIN_SAM_SCORE, MIN_HOLE_M2, MIN_PART_M2,
REMOVE_OVERLAPS, OVERLAP_MODE, COINCIDENT_M), точки пересчитываются заново;
MIN_PART_SHARE, KEEP_POINT_PART и SIMPLIFY_M меняются без пересчёта —
достаточно запустить скрипт ещё раз.

Служебные файлы — в <папка фрагментов>/_points/ (status.csv, parts/),
логи — в _logs/.

Установка:
  pip install segment-geospatial sam2 geopandas rasterio scipy psutil   # + PyTorch с CUDA
Проверка видеокарты: nvidia-smi
"""

import csv
import gc
import glob
import hashlib
import math
import multiprocessing as mp
import os
import sys
import time
import traceback

# Hugging Face на Windows без прав на символические ссылки хранит копии файлов —
# работает, но каждый раз пишет длинное предупреждение
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from shapely import wkb
from shapely.geometry import shape, MultiPolygon, Polygon
from shapely.ops import unary_union, transform as shp_transform
from pyproj import CRS, Transformer, Geod

# ============================================================================
# НАСТРОЙКИ
# ============================================================================
CHIPS_DIR = None            # None -> окно выбора папки с chips_index.gpkg
ASK_SETTINGS = True         # окно со всеми настройками при запуске (значения ниже —
                            # исходные; последние введённые хранятся в
                            # <папка фрагментов>/_points/settings12.json и
                            # подставляются в окно при следующем запуске)
SAM2_MODEL_ID = "sam2-hiera-large"   # sam2-hiera-tiny / -small / -base-plus / -large
DEVICE = None               # None -> авто (cuda, если есть), "cpu", "cuda"

N_WORKERS = "auto"          # "auto" или число процессов (1 -> один процесс)
THREADS_PER_WORKER = 8      # потоков PyTorch на процесс
WORKERS_PER_GPU = 2         # процессов на видеокарту при N_WORKERS="auto"
MEM_PER_WORKER_GB = 6       # запас ОЗУ на процесс при N_WORKERS="auto" на CPU
TEST_N = 20                 # ТЕСТОВЫЙ ЗАПУСК на ~20 точках; после проверки лога -> 0 (все)

# --- параметры сегмента (как в sam_segment_v12.py) ---
MAX_RADIUS_M = 150.0        # радиус поиска: наибольший радиус объекта от точки, м
                            # (кадр SAM — квадрат 2R x 2R; R не больше RADIUS_M фрагментов)
CONFIDENCE = 0.95           # порог χ² (доля пикселей эталона, проходящих порог):
                            # ВЫШЕ — МЯГЧЕ (0.99), НИЖЕ — СТРОЖЕ (0.80, 0.50)
MIN_GOOD_RATIO = 0.25       # мин. доля хороших пикселей маски от площади круга πR² (0–1)
REF_RADIUS_M = 20.0         # радиус эталонного окна вокруг точки, м
REF_RADIUS_FIELD = ""       # поле точки с радиусом эталона, м ("" -> общий REF_RADIUS_M;
                            # пустое значение или <= 0 -> тоже общий)
ENHANCE = False             # растяжка контраста 2–98% кадра для SAM (Махаланобис —
                            # всегда по исходным яркостям)
SMOOTH_M = 0.0              # сглаживание яркостей перед Махаланобисом, м (сигма Гаусса):
                            # сравниваются участки, а не отдельные пиксели — отсев
                            # непохожих участков целиком (0 -> по пикселям, как в v12)
MIN_SAM_SCORE = 0.0         # маски SAM с оценкой ниже не рассматриваются (0 -> как в v12)
MIN_HOLE_M2 = 10.0          # дырки сегмента меньше (м²) заливаются (0 -> как в v12)
MIN_PART_M2 = 10.0          # островки сегмента меньше (м²) удаляются (0 -> как в v12)

# --- перекрытия и очистка ---
REMOVE_OVERLAPS = True      # устранять перекрытия сегментов разных точек
OVERLAP_MODE = "voronoi"    # "voronoi": перекрытие делится серединным перпендикуляром —
                            #   каждый пиксель отходит ближайшей точке;
                            # "centroid": перекрытие целиком отходит точке, ближайшей
                            #   к его центру (как в v12; у плотных профилей точек дробит)
COINCIDENT_M = 1.0          # точки ближе (м) — одно место: перекрытия не делятся
# --- меняются без пересчёта точек ---
MIN_PART_SHARE = 0.01       # удалить части полигона меньше этой доли его площади (v12)
KEEP_POINT_PART = False     # оставить только часть, в которой лежит точка (или ближайшую)
SIMPLIFY_M = 0.0            # упрощение контуров, м (0 -> пиксельные контуры, как в v12)
OUT_EPSG = 32637            # UTM 37N — вся Московская обл.
OUT_NAME = "point_segments.gpkg"

# ============================================================================



# ============================================================================
# ЛОГ: всё, что выводится в консоль, дублируется в файл
# ============================================================================
class _Tee:
    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        self.fh.flush()

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def __getattr__(self, name):          # encoding, isatty и т.п. — от исходного потока
        return getattr(self.stream, name)


_LOG = {}


def start_log(folder, prefix):
    os.makedirs(folder, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(folder, f"{prefix}_{stamp}.log")
    fh = open(path, "w", encoding="utf-8")
    _LOG["orig"] = (sys.stdout, sys.stderr)
    _LOG["fh"] = fh
    # оборачиваем ТЕКУЩИЙ вывод (в Jupyter это ячейка, в консоли — терминал)
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    return path, stamp


def stop_log():
    if "orig" in _LOG:
        sys.stdout, sys.stderr = _LOG.pop("orig")
        _LOG.pop("fh").close()


def system_info():
    import platform
    lines = [f"Дата: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"Компьютер: {platform.node()} | ОС: {platform.platform()}",
             f"Python: {sys.version.split()[0]}",
             f"Процессор: {platform.processor() or platform.machine()} | "
             f"логических ядер: {os.cpu_count()}"]
    try:
        import psutil
        vm = psutil.virtual_memory()
        lines.append(f"Физических ядер: {psutil.cpu_count(logical=False)} | "
                     f"ОЗУ: {vm.total / 2**30:.0f} ГБ (свободно {vm.available / 2**30:.0f} ГБ)")
    except Exception:
        lines.append("psutil не установлен (pip install psutil) — нет данных об ОЗУ")
    from importlib import metadata
    for dist in ("numpy", "pandas", "geopandas", "rasterio", "shapely", "pyproj", "requests",
                 "pillow", "openpyxl", "psutil", "earthengine-api", "torch", "torchvision",
                 "segment-geospatial", "sam2", "scipy"):
        try:
            lines.append(f"  {dist} {metadata.version(dist)}")
        except Exception:
            pass
    return "\n".join(lines)


def settings_info(names):
    g = globals()
    return "\n".join(f"  {n} = {g[n]!r}" for n in names if n in g)


def pick_dir():
    if CHIPS_DIR:
        return CHIPS_DIR
    if len(sys.argv) > 1:
        return sys.argv[1]
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    d = filedialog.askdirectory(title="Папка с фрагментами (chips_index.gpkg)")
    root.destroy()
    return d


# ============================================================================
# ОКНО НАСТРОЕК: все параметры сегментов спрашиваются при каждом запуске
# ============================================================================
# (группа, имя, подпись, тип, варианты/диапазон, пояснение)
SPEC = [
    ("Сегмент точки", "MAX_RADIUS_M", "Радиус поиска R, м", "float", (1, None),
     "наибольший радиус объекта от точки; кадр SAM — квадрат 2R × 2R"),
    ("Сегмент точки", "CONFIDENCE", "Достоверность (хи-квадрат)", "conf",
     ["0.50", "0.80", "0.90", "0.95", "0.99"],
     "НИЖЕ — СТРОЖЕ: меньше непохожих пикселей (0.80, 0.50); выше — мягче (0.99)"),
    ("Сегмент точки", "MIN_GOOD_RATIO", "Мин. доля хороших пикселей", "float", (0, 1),
     "от площади круга поиска; если меньше — точка отклоняется"),
    ("Сегмент точки", "REF_RADIUS_M", "Радиус эталона, м", "float", (0.5, None),
     "окно вокруг точки, по которому считается спектральный эталон"),
    ("Сегмент точки", "REF_RADIUS_FIELD", "Поле радиуса эталона", "field", None,
     "свой радиус эталона у точки; пусто или ≤ 0 — общий радиус"),
    ("Сегмент точки", "SMOOTH_M", "Сглаживание, м", "float", (0, 20),
     "1–3 м: непохожие участки отсекаются целиком, а не мелкими дырками; 0 — как в v12"),
    ("Сегмент точки", "MIN_SAM_SCORE", "Мин. оценка маски SAM", "float", (0, 1),
     "маски с оценкой ниже не рассматриваются; 0 — как в v12"),
    ("Сегмент точки", "ENHANCE", "Растяжка контраста для SAM", "bool", None,
     "2–98% кадра; Махаланобис всегда по исходным яркостям"),
    ("Очистка сегмента", "MIN_HOLE_M2", "Заливать дырки до, м²", "float", (0, None),
     "0 — как в v12 (пиксельные дырки остаются)"),
    ("Очистка сегмента", "MIN_PART_M2", "Удалять островки до, м²", "float", (0, None),
     "0 — как в v12"),
    ("Очистка сегмента", "MIN_PART_SHARE", "Удалять части меньше доли", "float", (0, 1),
     "от площади полигона; в v12 — 0.01"),
    ("Очистка сегмента", "KEEP_POINT_PART", "Только часть с точкой", "bool", None,
     "из многочастного полигона оставить часть, в которой лежит точка"),
    ("Очистка сегмента", "SIMPLIFY_M", "Упрощение контура, м", "float", (0, None),
     "0 — пиксельные контуры, как в v12"),
    ("Перекрытия", "REMOVE_OVERLAPS", "Устранять перекрытия", "bool", None,
     "сегменты разных точек не перекрываются"),
    ("Перекрытия", "OVERLAP_MODE", "Способ деления", "choice", ["voronoi", "centroid"],
     "voronoi — каждый пиксель ближайшей точке; centroid — всё перекрытие точке, "
     "ближайшей к его центру (как в v12)"),
    ("Перекрытия", "COINCIDENT_M", "Одно место, если ближе, м", "float", (0, None),
     "повторные описания одного места не делят перекрытие — у каждого свой сегмент"),
    ("Модель и запуск", "SAM2_MODEL_ID", "Модель SAM 2", "choice",
     ["sam2-hiera-large", "sam2-hiera-base-plus", "sam2-hiera-small", "sam2-hiera-tiny"],
     "large — точнее, tiny — быстрее"),
    ("Модель и запуск", "TEST_N", "Пробный запуск, точек", "int", (0, None),
     "примерно столько точек (целыми фрагментами) и прогноз времени на все; 0 — все"),
    ("Модель и запуск", "DEVICE", "Устройство", "device", ["авто", "cuda", "cpu"],
     "авто — видеокарта, если доступна PyTorch"),
    ("Модель и запуск", "N_WORKERS", "Процессов", "workers", None,
     "auto или число; auto: на GPU — по числу карт × процессов на карту"),
    ("Модель и запуск", "WORKERS_PER_GPU", "Процессов на видеокарту", "int", (1, 16),
     "при «Процессов» = auto"),
    ("Модель и запуск", "THREADS_PER_WORKER", "Потоков на процесс", "int", (1, 256),
     "потоков PyTorch в каждом процессе"),
    ("Результат", "OUT_NAME", "Файл результата", "str", None,
     "GeoPackage в папке фрагментов"),
    ("Результат", "OUT_EPSG", "EPSG результата", "int", (1024, 999999),
     "32637 — UTM 37N (вся Московская обл.)"),
]
SETTING_NAMES = [s[1] for s in SPEC]
SEG_PARAMS = ["SAM2_MODEL_ID", "MAX_RADIUS_M", "CONFIDENCE", "MIN_GOOD_RATIO", "REF_RADIUS_M",
              "REF_RADIUS_FIELD", "SMOOTH_M", "ENHANCE", "MIN_SAM_SCORE", "MIN_HOLE_M2", "MIN_PART_M2",
              "REMOVE_OVERLAPS", "OVERLAP_MODE", "COINCIDENT_M"]    # их изменение -> пересчёт
V12 = dict(SMOOTH_M=0.0, MIN_SAM_SCORE=0.0, MIN_HOLE_M2=0.0, MIN_PART_M2=0.0, MIN_PART_SHARE=0.01,
           KEEP_POINT_PART=False, SIMPLIFY_M=0.0, ENHANCE=False, REMOVE_OVERLAPS=True,
           OVERLAP_MODE="centroid")
HEADER = {n: globals()[n] for n in SETTING_NAMES}      # значения из шапки скрипта
SETTINGS_FILE = "settings12.json"                       # в <папка фрагментов>/_points/


def settings_sig(cfg=None):
    """Подпись параметров сегмента: при их изменении точки пересчитываются."""
    c = cfg or globals()
    s = repr((c["SAM2_MODEL_ID"], float(c["MAX_RADIUS_M"]), float(c["CONFIDENCE"]),
              float(c["MIN_GOOD_RATIO"]), float(c["REF_RADIUS_M"]), c["REF_RADIUS_FIELD"] or "",
              bool(c["ENHANCE"]), float(c["MIN_SAM_SCORE"]), float(c["MIN_HOLE_M2"]),
              float(c["MIN_PART_M2"]), bool(c["REMOVE_OVERLAPS"]), c["OVERLAP_MODE"],
              float(c["COINCIDENT_M"]))
             + ((float(c["SMOOTH_M"]),) if float(c.get("SMOOTH_M", 0) or 0) else ()))
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]


def current_settings():
    return {n: globals()[n] for n in SETTING_NAMES}


def apply_settings(cfg):
    globals().update({k: v for k, v in cfg.items() if k in SETTING_NAMES})


def _to_text(kind, v):
    if kind == "device":
        return "авто" if v in (None, "", "auto") else str(v)
    if kind == "conf":
        return f"{float(v):.2f}"
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _parse(spec, v):
    """Значение из окна/консоли -> значение параметра; ValueError с пояснением."""
    _, name, label, kind, opt, _ = spec
    if kind == "bool":
        if isinstance(v, str):
            t = v.strip().lower()
            if t in ("1", "y", "yes", "true", "да", "д", "+"):
                return True
            if t in ("0", "n", "no", "false", "нет", "н", "-"):
                return False
            raise ValueError(f"{label}: да / нет")
        return bool(v)
    t = str(v).strip().replace(",", ".")
    if kind == "str":
        if not t:
            raise ValueError(f"{label}: пусто")
        return t if t.lower().endswith(".gpkg") else t + ".gpkg"
    if kind == "field":
        return t
    if kind == "choice":                        # точное имя, номер (1, 2, ...) или часть имени
        if t in opt:
            return t
        if t.isdigit() and 1 <= int(t) <= len(opt):
            return opt[int(t) - 1]
        hit = [o for o in opt if t.lower() and t.lower() in o.lower()]
        if len(hit) == 1:
            return hit[0]
        raise ValueError(f"{label}: один из {', '.join(opt)} (или номер 1–{len(opt)})")
    if kind == "device":
        t = t.lower()
        if t in ("", "авто", "auto", "none"):
            return None
        if t not in ("cuda", "cpu"):
            raise ValueError(f"{label}: авто, cuda или cpu")
        return t
    if kind == "workers":
        if t.lower() in ("auto", "авто", ""):
            return "auto"
        try:
            n = int(t)
        except ValueError:
            raise ValueError(f"{label}: auto или целое число") from None
        if n < 1:
            raise ValueError(f"{label}: не меньше 1")
        return n
    if kind == "conf":
        try:
            x = float(t)
        except ValueError:
            raise ValueError(f"{label}: число, например 0.95") from None
        if x > 1:                                   # 95 -> 0.95
            x /= 100
        if not 0.05 <= x < 1:
            raise ValueError(f"{label}: от 0.05 до 0.999")
        return x
    try:
        x = int(t) if kind == "int" else float(t)
    except ValueError:
        raise ValueError(f"{label}: {'целое ' if kind == 'int' else ''}число") from None
    lo, hi = opt
    if (lo is not None and x < lo) or (hi is not None and x > hi):
        rng = f"от {lo:g}" if hi is None else f"от {lo:g} до {hi:g}"
        raise ValueError(f"{label}: {rng}")
    return x


def _settings_path(d):
    return os.path.join(d, "_points", SETTINGS_FILE)


def load_saved_settings(d):
    """Последние введённые настройки для этой папки (или {})."""
    import json
    p = _settings_path(d)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    out = {}
    for sp in SPEC:
        if sp[1] in raw:
            try:
                out[sp[1]] = _parse(sp, raw[sp[1]] if sp[3] != "device" else
                                    _to_text("device", raw[sp[1]]))
            except ValueError:
                pass
    return out


def save_settings(d, cfg):
    import json
    os.makedirs(os.path.join(d, "_points"), exist_ok=True)
    with open(_settings_path(d), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    return _settings_path(d)


def _folder_facts(d):
    """Поля точек, число точек/фрагментов, радиус фрагментов, готовые точки по подписи."""
    facts = dict(fields=[], n_pts=None, n_chips=None, r_chip=None, done={})
    try:
        import pyogrio
        info = pyogrio.read_info(os.path.join(d, "chips_points.gpkg"))
        facts["fields"] = [c for c in info["fields"] if c not in ("point_id", "chip_id")]
        facts["n_pts"] = int(info["features"])
    except Exception:
        try:
            g = gpd.read_file(os.path.join(d, "chips_points.gpkg"))
            facts["fields"] = [c for c in g.columns if c not in ("geometry", "point_id",
                                                                 "chip_id")]
            facts["n_pts"] = len(g)
        except Exception:
            pass
    try:
        idx = gpd.read_file(os.path.join(d, "chips_index.gpkg"), ignore_geometry=True)
        facts["n_chips"] = int((idx["file"].fillna("") != "").sum())
        if "radius_m" in idx:
            facts["r_chip"] = float(pd.to_numeric(idx["radius_m"], errors="coerce").min())
        f0 = next(f for f in idx["file"].fillna("") if f)
        with rasterio.open(os.path.join(d, f0)) as src:        # пиксель на местности
            res, epsg = float(src.res[0]), src.crs.to_epsg()
            cx, cy = (src.bounds.left + src.bounds.right) / 2, (src.bounds.bottom + src.bounds.top) / 2
            if src.crs.is_geographic:
                gpx = None
            elif epsg in (3857, 3395, 900913):
                lat = Transformer.from_crs(src.crs, 4326, always_xy=True).transform(cx, cy)[1]
                gpx = res * math.cos(math.radians(lat))
            else:
                gpx = res
        facts.update(res=res, epsg=epsg, px_m=gpx)
    except Exception:
        pass
    try:
        st = read_status(os.path.join(d, "_points", "status.csv"))
        st = st[st.status.isin(["ok", "rejected"])].drop_duplicates("token")
        facts["done"] = st.groupby("sig").size().to_dict()
    except Exception:
        pass
    return facts


def _info_text(cfg, facts):
    """Пояснение к текущим значениям: площади, порог, готовые точки."""
    from scipy.stats import chi2
    R = float(cfg["MAX_RADIUS_M"])
    circ = math.pi * R ** 2
    lines = [f"Круг поиска: {circ / 1e4:.2f} га; минимальный объект: "
             f"{cfg['MIN_GOOD_RATIO'] * circ / 1e4:.2f} га хороших пикселей; порог хи-квадрат "
             f"{chi2.ppf(cfg['CONFIDENCE'], df=3):.3f}; окно эталона "
             f"{math.pi * cfg['REF_RADIUS_M'] ** 2:,.0f} м²".replace(",", " ")]
    g = facts.get("px_m")
    if g:
        a = g * g
        npx = lambda m2: f"{m2 / a:,.0f}".replace(",", " ")
        lines.append(
            f"Пиксель снимка: {facts['res']:.2f} ед. EPSG:{facts['epsg']} = {g:.2f} м на "
            f"местности ({a:.2f} м²). Пороги — в м² на местности: дырки до "
            f"{cfg['MIN_HOLE_M2']:g} м² ~ {npx(cfg['MIN_HOLE_M2'])} пикс., островки до "
            f"{cfg['MIN_PART_M2']:g} м² ~ {npx(cfg['MIN_PART_M2'])} пикс.; сглаживание "
            f"{cfg['SMOOTH_M']:g} м ~ {cfg['SMOOTH_M'] / g:.1f} пикс.")
    warn = []
    if facts.get("r_chip") and R > facts["r_chip"]:
        warn.append(f"R больше радиуса фрагментов ({facts['r_chip']:g} м) — круг будет "
                    f"обрезан краем")
    if cfg["REF_RADIUS_M"] >= R:
        warn.append("радиус эталона не меньше радиуса поиска")
    sig = settings_sig(cfg)
    n_done = facts.get("done", {}).get(sig, 0)
    n_pts = facts.get("n_pts")
    if n_done:
        lines.append(f"С этими параметрами сегмента уже готово {n_done} точек — они не "
                     f"пересчитываются.")
    elif facts.get("done"):
        lines.append("С этими параметрами сегмента готовых точек нет: все точки будут "
                     "посчитаны заново (прежние результаты в _points сохраняются).")
    if cfg["TEST_N"]:
        lines.append(f"Пробный запуск: ~{cfg['TEST_N']} точек"
                     f"{f' из {n_pts}' if n_pts else ''} и прогноз времени на все.")
    else:
        lines.append(f"Полный запуск: все точки{f' ({n_pts})' if n_pts else ''}.")
    if warn:
        lines.append("Внимание: " + "; ".join(warn) + ".")
    return "\n".join(lines)


def _tk_dpi():
    if os.name == "nt":                    # чёткие шрифты на мониторах с масштабом > 100%
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


def gui_settings(d, init, facts):
    """Окно со всеми настройками. Возвращает dict или None (Отмена)."""
    import tkinter as tk
    from tkinter import ttk, messagebox
    _tk_dpi()
    root = tk.Tk()
    root.title("12_samgeo_point_segments — настройки сегментов")
    root.attributes("-topmost", True)
    root.after(800, lambda: root.attributes("-topmost", False))
    try:
        ttk.Style().theme_use("vista" if os.name == "nt" else "clam")
    except Exception:
        pass
    fam = "Segoe UI" if os.name == "nt" else "DejaVu Sans"
    try:
        from tkinter import font as tkfont
        for fn in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            tkfont.nametofont(fn).configure(family=fam, size=9)
    except Exception:
        pass
    small = (fam, 8)
    out = {"cfg": None}
    vars_, widgets = {}, {}
    FIELD_NONE = "(общий радиус)"

    top = ttk.Frame(root, padding=(10, 8, 10, 0))
    top.grid(row=0, column=0, sticky="ew")
    src = []
    if facts.get("n_pts") is not None:
        src.append(f"точек: {facts['n_pts']}")
    if facts.get("n_chips") is not None:
        src.append(f"фрагментов: {facts['n_chips']}")
    if facts.get("r_chip"):
        src.append(f"радиус фрагментов: {facts['r_chip']:g} м")
    ttk.Label(top, text=f"Папка: {os.path.abspath(d)}").grid(row=0, column=0, sticky="w")
    ttk.Label(top, text="; ".join(src) + (f".  Значения — с прошлого запуска в этой папке "
                                          f"(_points/{SETTINGS_FILE})."
                                          if init.get("_saved") else
                                          ".  Значения — из шапки скрипта."),
              foreground="#555").grid(row=1, column=0, sticky="w")

    body = ttk.Frame(root, padding=10)
    body.grid(row=1, column=0, sticky="nsew")
    cols = {"Сегмент точки": 0, "Очистка сегмента": 0,
            "Перекрытия": 1, "Модель и запуск": 1, "Результат": 1}
    frames, rows, col_row = {}, {}, {0: 0, 1: 0}
    for grp, name, label, kind, opt, hint in SPEC:
        if grp not in frames:
            c = cols[grp]
            fr = ttk.LabelFrame(body, text=grp, padding=(8, 4))
            fr.grid(row=col_row[c], column=c, sticky="new", padx=6, pady=4)
            col_row[c] += 1
            frames[grp], rows[grp] = fr, 0
        fr, r = frames[grp], rows[grp]
        rows[grp] += 1
        val = init[name]
        if kind == "bool":
            v = tk.BooleanVar(value=bool(val))
            w = ttk.Checkbutton(fr, variable=v)
        else:
            v = tk.StringVar(value=_to_text(kind, val))
            if kind in ("choice", "device"):
                w = ttk.Combobox(fr, textvariable=v, values=opt, state="readonly", width=20)
            elif kind == "conf":
                w = ttk.Combobox(fr, textvariable=v, values=opt, width=20)
            elif kind == "field":
                fl = [FIELD_NONE] + facts.get("fields", [])
                if val and val not in fl:
                    fl.append(val)
                v.set(val or FIELD_NONE)
                w = ttk.Combobox(fr, textvariable=v, values=fl, state="readonly", width=20)
            else:
                w = ttk.Entry(fr, textvariable=v, width=34 if name == "OUT_NAME" else 22)
        ttk.Label(fr, text=label, width=28).grid(row=r, column=0, sticky="w", pady=2)
        w.grid(row=r, column=1, sticky="w", padx=6, pady=2)
        ttk.Label(fr, text=hint, foreground="#666", font=small, wraplength=300,
                  justify="left").grid(row=r, column=2, sticky="w", pady=2)
        vars_[name], widgets[name] = v, w

    info = tk.StringVar()
    ttk.Label(root, textvariable=info, padding=(16, 0), justify="left", wraplength=1150,
              foreground="#1a4d8f").grid(row=2, column=0, sticky="w")

    def read():
        cfg, errs = {}, []
        for sp in SPEC:
            name, kind = sp[1], sp[3]
            raw = vars_[name].get()
            if kind == "field" and raw == FIELD_NONE:
                raw = ""
            try:
                cfg[name] = _parse(sp, raw)
            except ValueError as e:
                errs.append(str(e))
                cfg[name] = init[name]
        return cfg, errs

    def refresh(*_):
        cfg, errs = read()
        on = bool(vars_["REMOVE_OVERLAPS"].get())
        for n in ("OVERLAP_MODE", "COINCIDENT_M"):
            widgets[n].configure(state=("readonly" if n == "OVERLAP_MODE" else "normal")
                                 if on else "disabled")
        try:
            txt = _info_text(cfg, facts)
        except Exception:
            txt = ""
        info.set(txt + (("\nНеверные значения: " + "; ".join(errs)) if errs else ""))

    def set_values(vals):
        for sp in SPEC:
            name, kind = sp[1], sp[3]
            if name not in vals:
                continue
            if kind == "bool":
                vars_[name].set(bool(vals[name]))
            elif kind == "field":
                vars_[name].set(vals[name] or FIELD_NONE)
            else:
                vars_[name].set(_to_text(kind, vals[name]))

    def start(*_):
        cfg, errs = read()
        if errs:
            messagebox.showerror("Проверьте значения", "\n".join(errs), parent=root)
            return
        out["cfg"] = cfg
        root.destroy()

    def cancel(*_):
        root.destroy()

    btn = ttk.Frame(root, padding=(10, 6, 10, 10))
    btn.grid(row=3, column=0, sticky="ew")
    btn.columnconfigure(2, weight=1)
    ttk.Button(btn, text="Как в v12", command=lambda: set_values(V12)).grid(row=0, column=0,
                                                                            padx=4)
    ttk.Button(btn, text="Из шапки скрипта",
               command=lambda: set_values(HEADER)).grid(row=0, column=1, padx=4)
    ttk.Button(btn, text="Запустить", command=start).grid(row=0, column=3, padx=4)
    ttk.Button(btn, text="Отмена", command=cancel).grid(row=0, column=4, padx=4)
    for v in vars_.values():
        v.trace_add("write", refresh)
    refresh()
    root.bind("<Return>", start)
    root.bind("<Escape>", cancel)
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.update_idletasks()
    w, h = root.winfo_reqwidth(), root.winfo_reqheight()
    x = max(0, (root.winfo_screenwidth() - w) // 2)
    y = max(0, (root.winfo_screenheight() - h) // 3)
    root.geometry(f"+{x}+{y}")
    root.mainloop()
    return out["cfg"]


def console_settings(init, facts):
    """Те же вопросы в консоли (если окно не открывается). Enter — оставить значение."""
    print("\nНастройки сегментов (Enter — оставить значение в скобках):")
    cfg, grp = {}, None
    for sp in SPEC:
        g, name, label, kind, opt, hint = sp
        if g != grp:
            print(f"\n  --- {g} ---")
            grp = g
        cur = _to_text(kind, init[name]) if kind != "bool" else ("да" if init[name] else "нет")
        if kind == "choice":
            var = " (" + ", ".join(f"{i}: {o}" for i, o in enumerate(opt, 1)) + ")"
        elif kind in ("device", "conf"):
            var = f" ({'/'.join(opt)})"
        else:
            var = " (да/нет)" if kind == "bool" else ""
        if kind == "field" and facts.get("fields"):
            print(f"    поля точек: {', '.join(facts['fields'])} ('-' — общий радиус)")
        while True:
            t = input(f"  {label}{var} — {hint} [{cur}]: ").strip().strip('"')
            if not t:
                cfg[name] = init[name]
                break
            try:
                cfg[name] = _parse(sp, "" if kind == "field" and t == "-" else t)
                break
            except ValueError as e:
                print(f"    {e}")
    print("\n" + _info_text(cfg, facts))
    if (input("\nНачать? (y/n) [y]: ").strip().lower() or "y") not in ("y", "д", "да", "yes"):
        return None
    return cfg


def ask_settings(d):
    """Окно (или консоль) со всеми настройками; значения — с прошлого запуска в этой
    папке, иначе из шапки. Возвращает dict или None (отмена)."""
    init = dict(HEADER)
    saved = load_saved_settings(d)
    init.update(saved)
    facts = _folder_facts(d)
    try:
        cfg = gui_settings(d, dict(init, _saved=bool(saved)), facts)
    except Exception as e:                  # нет графического окружения
        if sys.stdin is not None and sys.stdin.isatty():
            print(f"[i] Окно настроек недоступно ({e.__class__.__name__}) — вопросы в консоли.")
            cfg = console_settings(init, facts)
        else:
            print(f"[i] Окно настроек недоступно ({e.__class__.__name__}), консоль без ввода — "
                  f"используются значения {'с прошлого запуска' if saved else 'из шапки'}.")
            cfg = init
    if cfg is not None:
        cfg.pop("_saved", None)
    return cfg


def gpu_count(verbose=True):
    """Определение видеокарт NVIDIA: через PyTorch и (для диагностики) nvidia-smi.
    Предупреждает, если карта в системе есть, а PyTorch установлен без CUDA."""
    import shutil
    import subprocess
    smi = []
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
            smi = [l.strip() for l in r.stdout.splitlines() if l.strip()]
        except Exception:
            pass
    try:
        import torch
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        build = torch.version.cuda
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        tv = torch.__version__
    except Exception as e:
        n, build, names, tv = 0, None, [], f"не импортируется ({e})"
    if verbose:
        print(f"PyTorch {tv}, сборка CUDA: {build or 'нет (CPU-версия)'}")
        if smi:
            print("nvidia-smi: " + "; ".join(smi))
        else:
            print("nvidia-smi: видеокарт NVIDIA не найдено")
        if n:
            for i, nm in enumerate(names):
                print(f"  GPU {i}: {nm}")
        elif smi:
            print("[!] Видеокарта есть, но PyTorch её не видит: установлена CPU-версия.\n"
                  "    Закройте Jupyter и другие процессы Python и выполните в PowerShell\n"
                  "    от имени администратора (скачивается ~3 ГБ):\n    "
                  + torch_cuda_cmd()
                  + "\n    Другие версии: https://pytorch.org/get-started/locally/")
    return n


def torch_cuda_cmd():
    """Команда PowerShell для замены CPU-сборки PyTorch на сборку с CUDA 12.6 той же версии
    (сборки cu126 есть для PyTorch 2.6–2.9; драйвер NVIDIA для CUDA 12.x: от 527.41)."""
    from importlib import metadata
    pk = []
    for name in ("torch", "torchvision"):
        try:
            pk.append(f"{name}=={metadata.version(name).split('+')[0]}+cu126")
        except Exception:
            pass
    return (f'& "{sys.executable}" -m pip install --force-reinstall --no-deps '
            f'{" ".join(pk) or "torch torchvision"} --index-url https://download.pytorch.org/whl/cu126')


def physical_cores():
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) // 2)



def load_model(device):
    from samgeo import SamGeo2
    return SamGeo2(model_id=SAM2_MODEL_ID, automatic=False, device=device)


def prefetch_weights():
    """Скачивает веса модели один раз в главном процессе (до запуска рабочих).
    Возвращает True, если веса есть в кэше и рабочие процессы могут работать без сети."""
    t = time.time()
    try:
        from huggingface_hub import hf_hub_download
        from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES
        repo = SAM2_MODEL_ID if "/" in SAM2_MODEL_ID else f"facebook/{SAM2_MODEL_ID}"
        fn = HF_MODEL_ID_TO_FILENAMES[repo][1]
        try:                                    # уже скачаны -> без обращения к сети
            path = hf_hub_download(repo_id=repo, filename=fn, local_files_only=True)
        except Exception:
            path = hf_hub_download(repo_id=repo, filename=fn)
        print(f"Веса модели: {path} ({time.time() - t:.0f} с)", flush=True)
        return True
    except Exception as e:
        print(f"[!] Не удалось заранее загрузить веса модели: {e!r}\n"
              f"    Рабочие процессы попробуют загрузить их сами.", flush=True)
        return False


def stretch(img, lo=2, hi=98):
    out = np.empty_like(img)
    for b in range(img.shape[2]):
        v = img[:, :, b]
        valid = v[v > 0] if (v > 0).any() else v
        p1, p2 = np.percentile(valid, (lo, hi))
        out[:, :, b] = np.clip((v.astype(np.float32) - p1) / max(p2 - p1, 1) * 255, 0, 255)
    return out


# ============================================================================
# СЕГМЕНТ ОДНОЙ ТОЧКИ
# ============================================================================
_W = {}
_GEOD = Geod(ellps="WGS84")


def _transformers(crs):
    """Кэш преобразований WGS84 <-> CRS фрагмента и CRS фрагмента -> OUT_EPSG."""
    key = crs.to_wkt()
    tr = _W.setdefault("tr", {})
    if key not in tr:
        c = CRS.from_user_input(crs)
        tr[key] = (Transformer.from_crs(4326, c, always_xy=True),
                   Transformer.from_crs(c, 4326, always_xy=True),
                   Transformer.from_crs(c, OUT_EPSG, always_xy=True))
    return tr[key]


def ground_pixel_m(T, col, row, to_ll):
    """Размер пикселя на местности, м (для EPSG:3857 / 3395 это res·cos(φ), а не res)."""
    x0, y0 = T * (col, row)
    x1, y1 = T * (col + 1, row)
    x2, y2 = T * (col, row + 1)
    (lo0, la0), (lo1, la1), (lo2, la2) = [to_ll.transform(x, y) for x, y in
                                          ((x0, y0), (x1, y1), (x2, y2))]
    dx = _GEOD.inv(lo0, la0, lo1, la1)[2]
    dy = _GEOD.inv(lo0, la0, lo2, la2)[2]
    return (dx + dy) / 2.0


def _empty_res():
    return dict(status="rejected", reason="", ratios="", scores="", mask_idx=-1,
                sam_score=np.nan, good_ratio=np.nan, n_px=0, area_m2=0.0, area_raw_m2=0.0,
                ref_r_m=np.nan, ref_src="", max_r_m=float(MAX_RADIUS_M), px_m=np.nan, edge=0)


def _clean(seg, px_m, holes=True, parts=True):
    """Заливка дырок < MIN_HOLE_M2 и удаление островков < MIN_PART_M2 (связность 4)."""
    from scipy import ndimage
    pa = px_m ** 2
    if holes and MIN_HOLE_M2:
        lab, _ = ndimage.label(~seg)
        small = np.bincount(lab.ravel()) < MIN_HOLE_M2 / pa
        small[0] = False
        seg = seg | small[lab]
    if parts and MIN_PART_M2:
        lab, _ = ndimage.label(seg)
        small = np.bincount(lab.ravel()) < MIN_PART_M2 / pa
        small[0] = False
        seg = seg & ~small[lab]
    return seg


def point_mask(sam, raw_full, T, to_ll, colf, rowf, ref_r):
    """Сегмент одной точки на фрагменте (логика sam_segment_v12).
    Возвращает (res, (r0, c0, seg) или None)."""
    from scipy.stats import chi2
    res = _empty_res()
    _, H, W = raw_full.shape
    pc, pr = int(math.floor(colf)), int(math.floor(rowf))
    if not (0 <= pr < H and 0 <= pc < W):
        res["reason"] = "точка вне фрагмента"
        return res, None
    px_m = ground_pixel_m(T, pc, pr, to_ll)
    half = int(math.ceil(MAX_RADIUS_M / px_m))
    r0, c0 = max(0, pr - half), max(0, pc - half)
    r1, c1 = min(H, pr + half), min(W, pc + half)
    res["edge"] = int(pr - half < 0 or pc - half < 0 or pr + half > H or pc + half > W)
    raw = raw_full[:, r0:r1, c0:c1]                         # (3, h, w), исходные яркости
    res["px_m"] = round(px_m, 4)
    nb, h, w = raw.shape
    lr, lc = pr - r0, pc - c0

    # радиус эталона: из поля точки или общий
    ref_m, src_r = float(REF_RADIUS_M), "default"
    try:
        v = float(ref_r)
        if np.isfinite(v) and v > 0:
            ref_m, src_r = v, "field"
    except (TypeError, ValueError):
        pass
    ref_px = max(1, int(math.ceil(ref_m / px_m)))
    res["ref_r_m"], res["ref_src"] = ref_m, src_r

    # SAM 2: точка — положительная подсказка, 3 маски
    img = np.ascontiguousarray(np.transpose(raw, (1, 2, 0)))
    if ENHANCE:
        img = stretch(img)
    sam.predictor.set_image(img)
    masks, scores, _ = sam.predictor.predict(
        point_coords=np.array([[lc, lr]], dtype=np.float32),
        point_labels=np.array([1]), multimask_output=True)
    masks = np.asarray(masks).astype(bool)
    scores = np.asarray(scores, dtype=float)
    res["scores"] = ";".join(f"{s:.3f}" for s in scores)

    # эталон и расстояние Махаланобиса для всех пикселей кадра
    spec_img = raw.astype(np.float64)
    if SMOOTH_M and SMOOTH_M > 0:            # сглаживание: сравниваются участки, а не пиксели
        from scipy import ndimage
        sg = SMOOTH_M / px_m
        spec_img = np.stack([ndimage.gaussian_filter(b, sg, mode="nearest") for b in spec_img])
    ref = spec_img[:, max(0, lr - ref_px):min(h, lr + ref_px + 1),
                   max(0, lc - ref_px):min(w, lc + ref_px + 1)].reshape(nb, -1).T
    if ref.shape[0] < nb + 1:
        res["reason"] = "мало пикселей в эталонном окне"
        return res, None
    mean = ref.mean(axis=0)
    cov = np.cov(ref, rowvar=False) + np.eye(nb) * 1e-6
    try:
        icov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        res["reason"] = "вырожденная ковариация эталона"
        return res, None
    diff = spec_img.reshape(nb, -1).T - mean
    d2 = np.einsum("ij,jk,ik->i", diff, icov, diff).reshape(h, w)
    good = d2 <= chi2.ppf(CONFIDENCE, df=nb)

    yy, xx = np.ogrid[:h, :w]
    circle = ((yy - lr) ** 2 + (xx - lc) ** 2) <= half ** 2
    circle_area = math.pi * half ** 2
    gc_ = good & circle
    ratios = [float((m & gc_).sum()) / circle_area if m.any() else -1.0 for m in masks]
    res["ratios"] = ";".join("empty" if r < 0 else f"{r:.3f}" for r in ratios)
    cand = [r if sc >= MIN_SAM_SCORE else -1.0 for r, sc in zip(ratios, scores)]
    best = int(np.argmax(cand))                             # первая из максимальных, как в v12
    if cand[best] < 0 and max(ratios) >= 0:
        res["reason"] = f"оценка SAM всех масок < {MIN_SAM_SCORE}"
        return res, None
    if ratios[best] < MIN_GOOD_RATIO:
        res["reason"] = f"доля хороших пикселей {max(ratios[best], 0):.2f} < {MIN_GOOD_RATIO}"
        return res, None
    seg = _clean(masks[best] & gc_, px_m)
    if not seg.any():
        res["reason"] = "пустой сегмент"
        return res, None
    res.update(status="ok", mask_idx=best + 1, sam_score=round(float(scores[best]), 3),
               good_ratio=round(ratios[best], 3))
    return res, (r0, c0, seg)


def _resolve_chip(items, pxy, grp):
    """Перекрытия сегментов точек одного фрагмента — на растре фрагмента (быстро и точно).
    items[i] = (r0, c0, seg) или None; pxy[i] = (row, col) точки; grp[i] — номер места
    (точки ближе COINCIDENT_M — одно место, между собой не делят). Изменяет items.
    voronoi: пиксель отходит ближайшей точке из тех, чьи сегменты его покрывают."""
    idx = [i for i, it in enumerate(items) if it is not None]
    if len(idx) < 2 or len(set(grp[i] for i in idx)) < 2:
        return
    R0 = min(items[i][0] for i in idx)
    C0 = min(items[i][1] for i in idx)
    R1 = max(items[i][0] + items[i][2].shape[0] for i in idx)
    C1 = max(items[i][1] + items[i][2].shape[1] for i in idx)
    if OVERLAP_MODE == "voronoi":
        best = np.full((R1 - R0, C1 - C0), np.inf, dtype=np.float32)
        owner = np.full((R1 - R0, C1 - C0), -1, dtype=np.int32)
        for i in idx:
            r0, c0, m = items[i]
            h, w = m.shape
            sl = (slice(r0 - R0, r0 - R0 + h), slice(c0 - C0, c0 - C0 + w))
            yy, xx = np.ogrid[r0:r0 + h, c0:c0 + w]
            d = ((yy + 0.5 - pxy[i][0]) ** 2 + (xx + 0.5 - pxy[i][1]) ** 2).astype(np.float32)
            b, o = best[sl], owner[sl]
            upd = m & (d < b)
            b[upd] = d[upd]
            o[upd] = grp[i]
        for i in idx:
            r0, c0, m = items[i]
            h, w = m.shape
            sl = (slice(r0 - R0, r0 - R0 + h), slice(c0 - C0, c0 - C0 + w))
            items[i] = (r0, c0, m & (owner[sl] == grp[i]))
    else:                                                   # "centroid", как в v12
        for _pass in range(2):                              # второй проход — тройные
            n = 0
            for ia, a in enumerate(idx):
                for b in idx[ia + 1:]:
                    if items[a] is None or items[b] is None or grp[a] == grp[b]:
                        continue
                    (ra, ca, ma), (rb, cb, mb) = items[a], items[b]
                    rr0, rr1 = max(ra, rb), min(ra + ma.shape[0], rb + mb.shape[0])
                    cc0, cc1 = max(ca, cb), min(ca + ma.shape[1], cb + mb.shape[1])
                    if rr0 >= rr1 or cc0 >= cc1:
                        continue
                    sa = ma[rr0 - ra:rr1 - ra, cc0 - ca:cc1 - ca]
                    sb = mb[rr0 - rb:rr1 - rb, cc0 - cb:cc1 - cb]
                    ov = sa & sb
                    if not ov.any():
                        continue
                    n += 1
                    rows, cols = np.nonzero(ov)
                    cy, cx = rows.mean() + rr0 + 0.5, cols.mean() + cc0 + 0.5
                    da = (cy - pxy[a][0]) ** 2 + (cx - pxy[a][1]) ** 2
                    db = (cy - pxy[b][0]) ** 2 + (cx - pxy[b][1]) ** 2
                    if da <= db:
                        sb &= ~ov                            # вид на массив mb
                    else:
                        sa &= ~ov
            if n == 0:
                break


def segment_chip(sam, path, pts):
    """Сегменты всех точек фрагмента. pts — список dict(point_id, lon, lat, ref_r).
    Возвращает (список res по точкам, список WKB или None, число сегментов, обрезанных
    соседними)."""
    with rasterio.open(path) as src:
        T, crs = src.transform, src.crs
        raw_full = src.read([1, 2, 3])
    to_crs, to_ll, to_out = _transformers(crs)
    results, items, pxy, pxm = [], [], [], []
    for p in pts:
        t1 = time.time()
        x, y = to_crs.transform(p["lon"], p["lat"])
        colf, rowf = ~T * (x, y)
        if not (np.isfinite(colf) and np.isfinite(rowf)):
            res, it = _empty_res(), None
            res["reason"] = "некорректные координаты"
        else:
            res, it = point_mask(sam, raw_full, T, to_ll, colf, rowf, p.get("ref_r"))
        res["seconds"] = round(time.time() - t1, 2)
        results.append(res)
        items.append(it)
        pxy.append((rowf, colf))
        pxm.append(res["px_m"] if np.isfinite(res["px_m"]) else 0.3)
    del raw_full
    # места: точки ближе COINCIDENT_M — одна группа
    grp = list(range(len(pts)))
    for i in range(len(pts)):
        for j in range(i):
            if (grp[j] == j and np.isfinite(pxy[i][0]) and np.isfinite(pxy[j][0]) and
                    math.hypot(pxy[i][0] - pxy[j][0], pxy[i][1] - pxy[j][1]) * pxm[i]
                    < COINCIDENT_M):
                grp[i] = j
                break
    raw_px = [int(it[2].sum()) if it is not None else 0 for it in items]
    for i, it in enumerate(items):
        if it is not None:
            results[i]["area_raw_m2"] = round(raw_px[i] * pxm[i] ** 2, 1)
    if REMOVE_OVERLAPS:
        _resolve_chip(items, pxy, grp)
    n_ov = sum(1 for i, it in enumerate(items) if it is not None and int(it[2].sum()) < raw_px[i])
    geoms = []
    for i, it in enumerate(items):
        res = results[i]
        if it is None:
            geoms.append(None)
            continue
        r0, c0, seg = it
        seg = _clean(seg, pxm[i], holes=False)              # обрезки после деления перекрытий
        n_px = int(seg.sum())
        if n_px == 0:
            res.update(status="rejected", reason="сегмент целиком отошёл соседним точкам")
            geoms.append(None)
            continue
        wt = T * rasterio.Affine.translation(c0, r0)
        polys = [shape(g) for g, val in
                 features.shapes(seg.astype(np.uint8), mask=seg, transform=wt) if val == 1]
        geom = shp_transform(to_out.transform, unary_union(polys))
        res.update(n_px=n_px, area_m2=round(geom.area, 1))
        geoms.append(geom.wkb)
    return results, geoms, n_ov


# ============================================================================
# РАБОЧИЙ ПРОЦЕСС
# ============================================================================
def init_worker(wid, chips_dir, n_gpu, threads):
    device = DEVICE
    if n_gpu > 0 and DEVICE in (None, "cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(wid % n_gpu)
        device = "cuda"
    elif DEVICE is None:
        device = "cpu"
    if device == "cpu":
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[v] = str(threads)
    import torch
    torch.set_num_threads(threads)      # и на GPU: иначе каждый процесс берёт все 128 потоков
    if device == "cpu":
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    import warnings
    import logging
    warnings.filterwarnings("ignore")
    logging.getLogger("rasterio._env").setLevel(logging.ERROR)   # GDAL 3.11: 'Memory' -> 'MEM'
    for attempt in range(3):            # файлы кэша на Windows бывают кратковременно заняты
        try:
            _W["sam"] = load_model(device)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(5 + 2 * attempt + wid)
    _W["dir"] = chips_dir
    os.makedirs(os.path.join(chips_dir, "_points", "parts"), exist_ok=True)
    _W["part"] = os.path.join(chips_dir, "_points", "parts", f"seg_{os.getpid()}.gpkg")


def peak_ram_gb():
    """Пиковая память процесса, ГБ (Windows: peak working set; Linux/macOS: ru_maxrss)."""
    try:
        import psutil
        mi = psutil.Process().memory_info()
        if hasattr(mi, "peak_wset"):
            return mi.peak_wset / 2**30
    except Exception:
        pass
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 2**30 if sys.platform == "darwin" else r / 2**20
    except Exception:
        return float("nan")


def process_chip(task):
    sam, d = _W["sam"], _W["dir"]
    t1 = time.time()
    err, n_ov = None, 0
    try:
        results, geoms, n_ov = segment_chip(sam, os.path.join(d, task["file"]), task["pts"])
        ok = [i for i, g in enumerate(geoms) if g is not None]
        if ok:
            gdf = gpd.GeoDataFrame(
                {"token": [task["pts"][i]["token"] for i in ok],
                 "point_id": [task["pts"][i]["point_id"] for i in ok],
                 "sig": [task["sig"]] * len(ok)},
                geometry=[wkb.loads(geoms[i]) for i in ok], crs=OUT_EPSG)
            gdf.to_file(_W["part"], layer="seg", driver="GPKG",
                        mode="a" if os.path.exists(_W["part"]) else "w")
    except Exception as e:
        err = f"{e!r} | {traceback.format_exc(limit=3)}"
        results = [dict(status="error", reason=err, seconds=0) for _ in task["pts"]]
    rss = round(peak_ram_gb(), 2)
    for r in results:
        r["pid"], r["peak_ram_gb"] = os.getpid(), rss
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    return task["chip_id"], results, n_ov, time.time() - t1, err


def worker_loop(wid, task_q, res_q, chips_dir, n_gpu, threads, cfg):
    apply_settings(cfg)                 # настройки из окна (процесс заново импортирует модуль)
    t = time.time()
    try:
        init_worker(wid, chips_dir, n_gpu, threads)
    except Exception as e:
        res_q.put(("fail", wid, f"{e!r} | {traceback.format_exc(limit=3)}"))
        return
    res_q.put(("ready", wid, (os.getpid(), time.time() - t)))
    while True:
        task = task_q.get()
        if task is None:
            break
        res_q.put(("start", wid, task["chip_id"]))
        res_q.put(("done", wid, process_chip(task)))


# ============================================================================
# ЗАГРУЗКА ВИДЕОКАРТ (nvidia-smi раз в GPU_SAMPLE_S секунд, итог — в конце)
# ============================================================================
class GpuMonitor:
    def __init__(self, every=10):
        import shutil
        import threading
        self.ok = bool(shutil.which("nvidia-smi"))
        self.every, self.samples = every, {}
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self):
        if self.ok:
            self._t.start()
        return self

    def _run(self):
        import subprocess
        while not self._stop.wait(self.every):
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,"
                                    "memory.total", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=15)
                for line in r.stdout.splitlines():
                    i, u, m, mt = [v.strip() for v in line.split(",")]
                    self.samples.setdefault(int(i), []).append((float(u), float(m), float(mt)))
            except Exception:
                pass

    def report(self, n_gpu, per_gpu):
        self._stop.set()
        if not self.samples:
            return
        low, room = True, True
        for i, ss in sorted(self.samples.items()):
            if i >= n_gpu:
                continue
            u = [x[0] for x in ss]
            mem, tot = max(x[1] for x in ss), ss[0][2]
            print(f"Загрузка GPU {i}: средняя {np.mean(u):.0f}%, медиана {np.median(u):.0f}%, "
                  f"память до {mem / 1024:.1f} из {tot / 1024:.0f} ГБ ({len(ss)} замеров)")
            low &= np.median(u) < 70
            room &= mem / per_gpu * (per_gpu + 1) < 0.85 * tot
        if low and room:
            print(f"  Видеокарты загружены не полностью и памяти хватает: можно поставить "
                  f"WORKERS_PER_GPU = {per_gpu + 1} (сейчас {per_gpu}).")




# ============================================================================
# СТАТУС ТОЧЕК И СЛИЯНИЕ
# ============================================================================
STATUS_COLS = ["token", "chip_token", "sig", "point_id", "chip_id", "status", "reason", "ratios",
               "scores", "mask_idx", "sam_score", "good_ratio", "n_px", "area_m2", "area_raw_m2",
               "ref_r_m", "ref_src", "max_r_m", "px_m", "edge", "seconds", "pid", "peak_ram_gb",
               "time"]
SERVICE_COLS = ["seg_id", "area_m2", "area_raw_m2", "good_ratio", "mask_idx", "sam_score", "ratios",
                "ref_r_m", "ref_src", "max_r_m", "edge"]


def read_status(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=STATUS_COLS)
    st = pd.read_csv(path, encoding="utf-8-sig", dtype={"token": str, "chip_token": str,
                                                        "sig": str, "point_id": str,
                                                        "chip_id": str})
    for c in STATUS_COLS:
        if c not in st:
            st[c] = ""
    return st


def _parts(g):
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    if hasattr(g, "geoms"):                          # GeometryCollection после difference
        return [p for gg in g.geoms for p in _parts(gg)]
    return []


def _polys_only(g, min_share=0.0, min_hole=0.0, min_part=0.0, keep_xy=None):
    """Только полигоны: дырки меньше min_hole заливаются; части меньше min_part или
    min_share общей площади удаляются (как в v12); keep_xy -> только часть с точкой."""
    parts = [p for p in _parts(g) if p.area > 0]
    if min_hole:
        parts = [Polygon(p.exterior, [r for r in p.interiors if Polygon(r).area >= min_hole])
                 if p.interiors else p for p in parts]
    if not parts:
        return None
    tot = sum(p.area for p in parts)
    parts = [p for p in parts if p.area > tot * min_share and p.area >= min_part] or []
    if keep_xy is not None and len(parts) > 1:
        from shapely.geometry import Point
        pt = Point(keep_xy)
        parts = [min(parts, key=lambda q: q.distance(pt))]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def _half_plane(a, b, L=1e6):
    """Полуплоскость точек, которые ближе к a, чем к b (большой прямоугольник)."""
    (xa, ya), (xb, yb) = a, b
    mx, my = (xa + xb) / 2, (ya + yb) / 2
    dx, dy = xb - xa, yb - ya
    n = math.hypot(dx, dy)
    nx, ny = dx / n, dy / n                          # направление a -> b
    px, py = -ny, nx                                 # вдоль серединного перпендикуляра
    return Polygon([(mx + px * L, my + py * L), (mx + px * L - nx * L, my + py * L - ny * L),
                    (mx - px * L - nx * L, my - py * L - ny * L), (mx - px * L, my - py * L)])


def resolve_overlaps(geoms, pxy, chips):
    """Перекрытия сегментов точек из РАЗНЫХ фрагментов (внутри фрагмента они уже
    устранены на растре): OVERLAP_MODE, два прохода. Точки ближе COINCIDENT_M
    перекрытия не делят."""
    from shapely.strtree import STRtree
    geoms = list(geoms)
    n_total = 0
    for _pass in range(2):
        alive = [i for i, g in enumerate(geoms) if g is not None and not g.is_empty]
        if len(alive) < 2:
            break
        tree = STRtree([geoms[i] for i in alive])
        n = 0
        for a in alive:
            if geoms[a] is None or geoms[a].is_empty:
                continue
            for jb in tree.query(geoms[a]):
                b = alive[int(jb)]
                if a >= b or chips[a] == chips[b]:
                    continue
                ga, gb = geoms[a], geoms[b]
                if ga is None or ga.is_empty:
                    break
                if gb is None or gb.is_empty or not ga.intersects(gb):
                    continue
                (xa, ya), (xb, yb) = pxy[a], pxy[b]
                if math.hypot(xa - xb, ya - yb) < COINCIDENT_M:
                    continue
                ov = ga.intersection(gb)
                if ov.is_empty or ov.area < 1e-6:
                    continue
                n += 1
                if OVERLAP_MODE == "voronoi":
                    ha = _half_plane(pxy[a], pxy[b])
                    geoms[a] = _polys_only(ga.difference(ov.difference(ha)))
                    geoms[b] = _polys_only(gb.difference(ov.intersection(ha)))
                else:                                # "centroid", как в v12
                    c = ov.centroid
                    da = (c.x - xa) ** 2 + (c.y - ya) ** 2
                    db = (c.x - xb) ** 2 + (c.y - yb) ** 2
                    if da <= db:
                        geoms[b] = _polys_only(gb.difference(ov))
                    else:
                        geoms[a] = _polys_only(ga.difference(ov))
        n_total += n
        if n == 0:
            break
    return geoms, n_total


def merge_results(d, pts, sig, stamp):
    """pts — точки с фрагментами (GeoDataFrame, OUT_EPSG), колонка token."""
    st = read_status(os.path.join(d, "_points", "status.csv"))
    st = st[(st.sig == sig) & st.token.isin(set(pts.token)) & st.status.isin(["ok", "rejected"])]
    st = st.drop_duplicates("token", keep="last").set_index("token")
    ok_tok = set(st.index[st.status == "ok"])
    parts = sorted(glob.glob(os.path.join(d, "_points", "parts", "seg_*.gpkg")))
    chunks = []
    for p in parts:
        try:
            g = gpd.read_file(p, layer="seg")
        except Exception as e:
            print(f"  [!] часть {os.path.basename(p)} не читается: {e}")
            continue
        chunks.append(g[g.token.isin(ok_tok)])
    seg = (pd.concat([c for c in chunks if len(c)], ignore_index=True)
           if any(len(c) for c in chunks) else pd.DataFrame(columns=["token", "geometry"]))
    seg = seg.drop_duplicates("token", keep="first")
    geom_by_tok = dict(zip(seg.token, seg.geometry))

    attrs = pts.drop(columns=[c for c in ("file", "key", "radius_m", "token", "chip_token", "ref_r")
                              if c in pts.columns])
    ren = {c: f"{c}_pt" for c in attrs.columns if c in SERVICE_COLS}
    if ren:
        attrs = attrs.rename(columns=ren)
    ok_pts = pts[pts.token.isin(geom_by_tok)]
    geoms = [geom_by_tok[t] for t in ok_pts.token]
    pxy = [(g.x, g.y) for g in ok_pts.geometry]
    n_ov = 0
    t_ov = time.time()
    if REMOVE_OVERLAPS and len(geoms) > 1:
        geoms, n_ov = resolve_overlaps(geoms, pxy, list(ok_pts.chip_id))
    geoms = [_polys_only(g, MIN_PART_SHARE, 0, MIN_PART_M2, xy if KEEP_POINT_PART else None)
             for g, xy in zip(geoms, pxy)]
    if SIMPLIFY_M:
        geoms = [g.simplify(SIMPLIFY_M, preserve_topology=True) if g is not None else None
                 for g in geoms]
    keep = [i for i, g in enumerate(geoms) if g is not None and not g.is_empty]
    lost = len(geoms) - len(keep)

    out = attrs.loc[ok_pts.index[keep]].drop(columns="geometry").reset_index(drop=True)
    s = st.loc[ok_pts.token.iloc[keep]].reset_index(drop=True)
    out = gpd.GeoDataFrame(out, geometry=[geoms[i] for i in keep], crs=OUT_EPSG)
    out.insert(0, "seg_id", np.arange(1, len(out) + 1))
    out["area_m2"] = out.area.round(1)
    for c in SERVICE_COLS[2:]:
        out[c] = s[c].values if c in s else np.nan
    out["mask_idx"] = pd.to_numeric(out["mask_idx"], errors="coerce").astype("Int64")
    out["edge"] = pd.to_numeric(out["edge"], errors="coerce").astype("Int64")
    for c in ("area_raw_m2", "good_ratio", "sam_score", "ref_r_m", "max_r_m"):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    rej_tok = st.index[st.status == "rejected"]
    rej = attrs.loc[pts.index[pts.token.isin(rej_tok)]].copy()
    rs = st.loc[pts.token[pts.token.isin(rej_tok)]]
    for c in ("reason", "ratios", "scores", "ref_r_m"):
        rej[c] = rs[c].values
    rej = gpd.GeoDataFrame(rej, geometry="geometry", crs=OUT_EPSG)

    path = os.path.join(d, OUT_NAME)
    tmp = path + ".tmp.gpkg"
    if os.path.exists(tmp):
        os.remove(tmp)
    if len(out):
        out.to_file(tmp, layer="segments", driver="GPKG")
    if len(rej):
        rej.to_file(tmp, layer="rejected", driver="GPKG",
                    mode="a" if os.path.exists(tmp) else "w")
    if not os.path.exists(tmp):
        print("Нет готовых сегментов для слияния.")
        return
    try:
        if os.path.exists(path):
            os.remove(path)
        os.replace(tmp, path)
    except PermissionError:                      # файл открыт в ArcGIS / QGIS
        alt = os.path.join(d, f"{os.path.splitext(OUT_NAME)[0]}_{stamp}.gpkg")
        os.replace(tmp, alt)
        print(f"[!] {path} занят другой программой — результат записан в {alt}")
        path = alt
    print(f"Результат: {path}")
    print(f"  слой segments: {len(out)} полигонов (по одному на точку); перекрытий между "
          f"соседними фрагментами: {n_ov} ({time.time() - t_ov:.0f} с)"
          f"{f'; {lost} сегментов целиком ушли соседним точкам' if lost else ''}")
    if len(out):
        a = out.area_m2
        print(f"  площадь сегмента, га: медиана {a.median() / 1e4:.2f}, мин {a.min() / 1e4:.2f}, "
              f"макс {a.max() / 1e4:.2f}; сумма {a.sum() / 1e4:,.1f} га")
    print(f"  слой rejected: {len(rej)} точек без сегмента")


# ============================================================================
# MAIN
# ============================================================================
def _key(v):
    return v if isinstance(v, str) and v and v.lower() != "nan" else ""


def fmt_t(sec):
    sec = int(sec)
    return f"{sec // 3600} ч {sec % 3600 // 60:02d} мин" if sec >= 3600 else f"{sec // 60} мин {sec % 60:02d} с"


def main():
    d = pick_dir()
    if not d:
        print("Папка не выбрана."); return
    if not os.path.exists(os.path.join(d, "chips_index.gpkg")):
        print(f"В папке нет chips_index.gpkg: {d}"); return
    src = "шапка скрипта"
    if os.environ.get("SEG12_SETTINGS"):            # запуск из Jupyter: окно уже было
        import json
        with open(os.environ["SEG12_SETTINGS"], encoding="utf-8") as f:
            apply_settings(json.load(f))
        src = "окно настроек"
    elif ASK_SETTINGS:
        cfg = ask_settings(d)
        if cfg is None:
            print("Отменено."); return
        apply_settings(cfg)
        save_settings(d, cfg)
        src = "окно настроек"
    log_path, stamp = start_log(os.path.join(d, "_logs"), "12_point_segments")
    print("=" * 78)
    print("12_samgeo_point_segments.py" + ("  [ТЕСТОВЫЙ ЗАПУСК]" if TEST_N else ""))
    print("=" * 78)
    print(system_info())
    print(f"Настройки ({src}):")
    print(settings_info(["SAM2_MODEL_ID", "DEVICE", "N_WORKERS", "THREADS_PER_WORKER",
                         "WORKERS_PER_GPU", "TEST_N", "MAX_RADIUS_M", "CONFIDENCE",
                         "MIN_GOOD_RATIO", "REF_RADIUS_M", "REF_RADIUS_FIELD", "SMOOTH_M",
                         "ENHANCE",
                         "MIN_SAM_SCORE", "REMOVE_OVERLAPS", "OVERLAP_MODE", "COINCIDENT_M",
                         "MIN_HOLE_M2", "MIN_PART_M2", "MIN_PART_SHARE", "KEEP_POINT_PART",
                         "SIMPLIFY_M",
                         "OUT_EPSG", "OUT_NAME"]))
    from scipy.stats import chi2
    sig = settings_sig()
    print(f"  χ²-порог ({CONFIDENCE * 100:.0f}%, 3 канала): {chi2.ppf(CONFIDENCE, df=3):.3f}; "
          f"круг R = {MAX_RADIUS_M:g} м: {math.pi * MAX_RADIUS_M ** 2 / 1e4:.2f} га, минимум "
          f"хороших пикселей {MIN_GOOD_RATIO * math.pi * MAX_RADIUS_M ** 2 / 1e4:.2f} га; "
          f"подпись параметров {sig}")
    print(f"Папка фрагментов: {os.path.abspath(d)}")
    print(f"Лог: {log_path}")

    idx = gpd.read_file(os.path.join(d, "chips_index.gpkg"), ignore_geometry=True)
    idx = idx[idx["file"].fillna("") != ""].copy()
    idx = idx[[os.path.exists(os.path.join(d, f)) for f in idx["file"]]]
    if "key" not in idx:
        idx["key"] = ""
    if "radius_m" not in idx:
        idx["radius_m"] = np.nan
    pp = os.path.join(d, "chips_points.gpkg")
    if not os.path.exists(pp):
        print("Нет chips_points.gpkg — запустите 10_download_sat_chips.py текущей версии.")
        return
    pts = gpd.read_file(pp)
    pts["point_id"] = pts["point_id"].astype(str)
    pts = pts.to_crs(4326)
    n_all = len(pts)
    pts = pts.merge(idx[["chip_id", "file", "key", "radius_m"]], on="chip_id", how="left")
    no_chip = pts["file"].isna()
    if no_chip.any():
        ids = pts.loc[no_chip, "point_id"].tolist()
        print(f"[i] Точек без скачанного фрагмента: {len(ids)} (пропущены): "
              f"{', '.join(ids[:10])}{' ...' if len(ids) > 10 else ''}")
    pts = pts[~no_chip].copy()
    pts.index = np.arange(len(pts))
    pts["key"] = pts["key"].map(_key)
    pts["chip_token"] = [f"{c}|{k}|{sig}" for c, k in zip(pts.chip_id, pts.key)]
    pts["token"] = [f"{p}|{t}" for p, t in zip(pts.point_id, pts.chip_token)]
    dup = pts.token.duplicated()
    if dup.any():
        print(f"[!] Повторяющиеся point_id: {pts.loc[dup, 'point_id'].nunique()} — "
              f"обрабатывается первая точка с таким ID")
        pts = pts[~dup].copy()
    rf = REF_RADIUS_FIELD
    if rf and rf not in pts.columns:
        print(f"[!] Поля '{rf}' нет в chips_points.gpkg — используется общий радиус "
              f"эталона {REF_RADIUS_M:g} м")
        rf = ""
    elif rf:
        v = pd.to_numeric(pts[rf], errors="coerce")
        print(f"  Радиус эталона из поля '{rf}': у {int((v > 0).sum())} из {len(pts)} точек, "
              f"у остальных {REF_RADIUS_M:g} м")
    pts["ref_r"] = pts[rf] if rf else np.nan
    rmin = pd.to_numeric(pts["radius_m"], errors="coerce").min()
    if np.isfinite(rmin) and MAX_RADIUS_M > rmin:
        print(f"[!] MAX_RADIUS_M = {MAX_RADIUS_M:g} м больше радиуса фрагментов ({rmin:g} м): "
              f"круг поиска будет обрезан краем фрагмента (поле edge = 1)")

    os.makedirs(os.path.join(d, "_points", "parts"), exist_ok=True)
    status_f = os.path.join(d, "_points", "status.csv")
    st = read_status(status_f)
    done_chips = set(st.chip_token[st.status.isin(["ok", "rejected"])])
    is_done = pts.chip_token.isin(done_chips)
    chips = (pts[~is_done].groupby("chip_token", sort=False)
             .agg(chip_id=("chip_id", "first"), file=("file", "first"), n=("token", "size"))
             .reset_index())
    if TEST_N and TEST_N > 0:
        # случайные фрагменты целиком (перекрытия делятся внутри фрагмента), ~TEST_N точек
        sh = chips.sample(frac=1, random_state=0)
        take, tot = [], 0
        for r in sh.itertuples():
            if tot == 0 or tot + r.n <= TEST_N:
                take.append(r.Index)
                tot += r.n
            if tot >= TEST_N:
                break
        chips = chips.loc[take]
    N_pt, done_pt = len(pts), int(is_done.sum())
    n_todo = int(chips.n.sum())
    print(f"Точек: {n_all}, из них с фрагментами {N_pt} (фрагментов {idx.chip_id.nunique()}); "
          f"готово ранее с этими параметрами: {done_pt}; к обработке сейчас: {n_todo} точек "
          f"в {len(chips)} фрагментах{' [ПРОБНЫЙ ЗАПУСК]' if TEST_N else ''}")
    pts_out = pts.to_crs(OUT_EPSG)
    if chips.empty:
        merge_results(d, pts_out, sig, stamp); return

    n_gpu = gpu_count() if DEVICE in (None, "cuda") else 0
    if DEVICE == "cuda" and n_gpu == 0:
        raise RuntimeError("DEVICE = 'cuda', но видеокарта недоступна PyTorch")
    if N_WORKERS == "auto":
        nw = (n_gpu * max(1, int(WORKERS_PER_GPU)) if n_gpu > 0
              else max(1, physical_cores() // THREADS_PER_WORKER))
        if n_gpu == 0:
            try:
                import psutil
                free = psutil.virtual_memory().available / 2**30
                cap = max(1, int(free // MEM_PER_WORKER_GB))
                if cap < nw:
                    print(f"Ограничение по ОЗУ: свободно {free:.0f} ГБ -> {cap} процессов "
                          f"вместо {nw}")
                    nw = cap
            except Exception:
                pass
    else:
        nw = int(N_WORKERS)
    nw = max(1, min(nw, len(chips)))
    dev = (f"GPU x{n_gpu}, процессов на карту: {math.ceil(nw / n_gpu)}" if n_gpu
           else f"CPU, {THREADS_PER_WORKER} потоков на процесс")
    print(f"Процессов: {nw} ({dev}); модель {SAM2_MODEL_ID}. Загрузка моделей...")

    tasks = []
    by_chip = {k: g for k, g in pts[pts.chip_token.isin(set(chips.chip_token))]
               .groupby("chip_token", sort=False)}
    for r in chips.sort_values("n", ascending=False).itertuples():   # крупные — первыми
        g = by_chip[r.chip_token]
        tasks.append(dict(chip_id=r.chip_id, chip_token=r.chip_token, file=r.file, sig=sig,
                          pts=[dict(point_id=q.point_id, token=q.token,
                                    lon=float(q.geometry.x), lat=float(q.geometry.y),
                                    ref_r=getattr(q, "ref_r")) for q in g.itertuples()]))
    task_by_id = {t["chip_id"]: t for t in tasks}

    if prefetch_weights():
        os.environ["HF_HUB_OFFLINE"] = "1"      # рабочие процессы: только кэш, без записи и сети
    ctx = mp.get_context("spawn")
    task_q, res_q = ctx.Queue(), ctx.Queue()
    for t in tasks:
        task_q.put(t)
    retries, fails = {}, {}

    def spawn(wid):
        p = ctx.Process(target=worker_loop, daemon=True,
                        args=(wid, task_q, res_q, d, n_gpu, THREADS_PER_WORKER,
                              current_settings()))
        p.start()
        return p

    if not os.path.exists(status_f):
        with open(status_f, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(STATUS_COLS)
    workers = {w: spawn(w) for w in range(nw)}
    inflight = {}
    print(f"Модели загружаются в {nw} процессах (первые точки появятся после загрузки)...",
          flush=True)
    gmon = GpuMonitor().start() if n_gpu else None
    t0 = time.time()
    k = kc = n_ok = n_rej = errs = 0
    n_ready = 0
    reasons = {}
    secs, rss_l = [], []
    import queue as _queue

    def finish(chip_id, results, n_ov, sec_chip, err):
        nonlocal k, kc, n_ok, n_rej, errs
        t = task_by_id[chip_id]
        kc += 1
        stamp_t = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(status_f, "a", newline="", encoding="utf-8-sig") as f:
            wr = csv.writer(f)
            for p, res in zip(t["pts"], results):
                row = dict(res, token=p["token"], chip_token=t["chip_token"], sig=sig,
                           point_id=p["point_id"], chip_id=chip_id, time=stamp_t)
                wr.writerow([str(row.get(c, ""))[:400] for c in STATUS_COLS])
        lines = []
        for p, res in zip(t["pts"], results):
            k += 1
            stt = res["status"]
            if stt == "ok":
                n_ok += 1
                cut = (f", до деления перекрытий {res['area_raw_m2'] / 1e4:.2f} га"
                       if res.get("area_raw_m2", 0) > res["area_m2"] * 1.01 else "")
                msg = (f"сегмент {res['area_m2'] / 1e4:.2f} га{cut}, доля {res['good_ratio']:.2f}, "
                       f"маска {res['mask_idx']}/3 (score {res['sam_score']:.2f})")
            elif stt == "rejected":
                n_rej += 1
                key = res["reason"]
                if key.startswith("доля хороших"):
                    key = f"доля хороших пикселей < {MIN_GOOD_RATIO}"
                reasons[key] = reasons.get(key, 0) + 1
                msg = f"ОТКЛОНЕНА: {res['reason']} (доли масок {res.get('ratios') or '-'})"
            else:
                errs += 1
                msg = f"ОШИБКА {res['reason'].splitlines()[0][:150]}"
            if stt in ("ok", "rejected"):
                secs.append(float(res.get("seconds", 0) or 0))
                rss_l.append(float(res.get("peak_ram_gb", np.nan)))
            lines.append(f"    {p['point_id']}: {msg}, {res.get('seconds', 0):.1f} с")
        el = time.time() - t0
        rate = k / el if el > 0 else 0
        eta = (n_todo - k) / rate if rate else 0
        pt_all = done_pt + n_ok + n_rej
        print(f"[точки {k}/{n_todo} {k / n_todo:6.1%} | всего {pt_all}/{N_pt} {pt_all / N_pt:6.1%}"
              f" | фрагм. {kc}/{len(tasks)}] {chip_id}: {len(results)} т., {sec_chip:.0f} с"
              f"{f', обрезано соседями {n_ov}' if n_ov else ''} | сегм. {n_ok} | откл. {n_rej} | "
              f"ош. {errs} | {rate * 3600:.0f} т/ч | осталось ~{fmt_t(eta)}", flush=True)
        print("\n".join(lines), flush=True)

    while kc < len(tasks):
        try:
            kind, wid, data = res_q.get(timeout=5)
        except _queue.Empty:
            for wid, p in list(workers.items()):
                if p.is_alive() or (p.exitcode == 0 and wid not in inflight):
                    continue
                cid = inflight.pop(wid, None)
                print(f"[!] Процесс {wid} завершился аварийно (код {p.exitcode}) на фрагменте "
                      f"{cid}. Перезапуск.", flush=True)
                if cid is not None:
                    retries[cid] = retries.get(cid, 0) + 1
                    if retries[cid] <= 1:
                        task_q.put(task_by_id[cid])
                    else:
                        er = f"процесс аварийно завершён дважды (код {p.exitcode})"
                        finish(cid, [dict(status="error", seconds=0, reason=er)
                                     for _ in task_by_id[cid]["pts"]], 0, 0.0, er)
                workers[wid] = spawn(wid)
            continue
        if kind == "ready":
            n_ready += 1
            tag = f"[{n_ready}/{nw}]" if n_ready <= nw else "[перезапуск]"
            print(f"  процесс {wid} (pid {data[0]}) готов, модель загружена за {data[1]:.0f} с "
                  f"{tag}", flush=True)
            if n_ready == 1:
                t0 = time.time()                 # скорость — без времени загрузки моделей
        elif kind == "fail":
            fails[wid] = fails.get(wid, 0) + 1
            print(f"[!] Процесс {wid} не смог загрузить модель (попытка {fails[wid]}): "
                  f"{data.splitlines()[0]}", flush=True)
            if fails[wid] == 1:
                print("    Повторный запуск процесса.", flush=True)
                workers[wid] = spawn(wid)
                continue
            print(data)
            workers.pop(wid, None)
            if not workers:
                print("Ни один процесс не загрузил модель — остановлено.")
                raise SystemExit(1)
            print(f"    Работа продолжается без него, процессов: {len(workers)}.", flush=True)
        elif kind == "start":
            inflight[wid] = data
        elif kind == "done":
            inflight.pop(wid, None)
            finish(*data)

    for _ in workers:
        task_q.put(None)
    for p in workers.values():
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    el = time.time() - t0
    print(f"\nОбработано {k} точек ({kc} фрагментов) за {fmt_t(el)}: сегментов {n_ok}, "
          f"отклонено {n_rej}, ошибок {errs}.")
    if reasons:
        print("  причины отклонения: " + "; ".join(f"{r} — {n}" for r, n in reasons.items()))
    if gmon:
        gmon.report(n_gpu, math.ceil(nw / n_gpu))
    if secs:
        s = np.array(secs)
        print(f"Время на точку в одном процессе: медиана {np.median(s):.1f} с, мин {s.min():.1f}, "
              f"макс {s.max():.1f}; пик ОЗУ процесса {np.nanmax(rss_l):.1f} ГБ")
    if TEST_N and k:
        rest = N_pt - done_pt - n_ok - n_rej
        thr = k / el
        print(f"Тестовый запуск: {thr * 3600:.0f} точек/ч при {nw} процессах. Прогноз на "
              f"оставшиеся {rest} точек: ~{fmt_t(rest / thr)} (плюс ~1 мин на загрузку моделей)")
        if no_chip.any():
            print(f"  Ещё {int(no_chip.sum())} точек без фрагментов — сначала 10_ с TEST_N = 0.")
    merge_results(d, pts_out, sig, stamp)
    print(f"Статус точек: {status_f}")
    print(f"Лог: {log_path}")
    if TEST_N:
        print("Тест завершён. Пришлите лог; для полного запуска в окне настроек поставьте "
              "«Пробный запуск, точек» = 0.")


def _in_jupyter():
    return "ipykernel" in sys.modules and os.environ.get("SEG12_CHILD") != "1"


def run_from_jupyter():
    """В Jupyter (%run) параллельные процессы Windows не могут импортировать функции
    из ячейки, поэтому скрипт перезапускает сам себя обычным python и выводит его
    строки в ячейку по мере работы."""
    import subprocess
    d = pick_dir()
    if not d:
        print("Папка не выбрана."); return
    if not os.path.exists(os.path.join(d, "chips_index.gpkg")):
        print(f"В папке нет chips_index.gpkg: {d}"); return
    env = dict(os.environ, SEG12_CHILD="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    if ASK_SETTINGS:                        # окно — здесь: у дочернего процесса нет ввода
        cfg = ask_settings(d)
        if cfg is None:
            print("Отменено."); return
        env["SEG12_SETTINGS"] = save_settings(d, cfg)
    print(f"Запуск в отдельном процессе: {sys.executable} {os.path.abspath(__file__)}", flush=True)
    p = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), d], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         encoding="utf-8", errors="replace", bufsize=1)
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
    except KeyboardInterrupt:
        if os.name == "nt":             # вместе с рабочими процессами (иначе они держат GPU)
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
        else:
            p.terminate()
        print("\nОстановлено пользователем (готовые точки сохранены; "
              "повторный запуск продолжит с места остановки).")
    p.wait()


if __name__ == "__main__":
    mp.freeze_support()
    if _in_jupyter():
        run_from_jupyter()
    else:
        try:
            main()
        finally:
            stop_log()
