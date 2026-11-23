#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04b_add_landcover_classes.py — добавление нелесных категорий 20–23 в
классифицированную карту из 04_predict_map и стиль для ArcGIS Pro / QGIS.

  20  Пашни (включая залежи на ранних стадиях)
  21  Застройка и дороги
  22  Водные объекты
  23  Карьеры и торфоразработки

ИСТОЧНИК: ESA WorldCover 2021 v200 (10 м, Sentinel-1/2, открытая лицензия
CC BY 4.0). Плитки 3°×3° скачиваются один раз в 00_LANDCOVER/ESA_WorldCover_2021
и дальше берутся с диска. Для каждой ячейки карты (обычно 60 м) берётся
ПРЕОБЛАДАЮЩИЙ класс WorldCover из 10-м пикселей внутри неё (GDAL mode); этот
слой на сетке карты тоже кэшируется (WC2021_mode_*.tif), повторный запуск на той
же сетке не пересчитывает его.

ПРАВИЛА (таблица RULES ниже, их можно менять):
  WorldCover 50 застройка        -> 21  для всех классов модели
  WorldCover 80 вода             -> 22  для всех классов
  WorldCover 60 открытый грунт   -> 23  для всех классов (карьеры, торфоразработки)
  WorldCover 40 пашни            -> 20  для классов 1–17 (луга 18–19 сохраняются)
  WorldCover 30 травяные         -> 20  только для лесных классов 1–13
                                        (залежи на ранних стадиях; 14–19 сохраняются)
  WorldCover 90 травяные болота  -> 19  только для лесных классов 1–13
  WorldCover 10/20/95/100        -> класс модели не меняется

Дополнительно можно подложить СВОЙ растр (любая проекция и разрешение):
пиксели со значениями 20–23 записываются поверх результата WorldCover для всех
классов, остальные значения игнорируются. Так можно добавить, например,
маску торфоразработок или дорог OSM, растеризованных заранее.

ЧТО ПИШЕТСЯ рядом с исходным <base>_cat.tif:
  <base>_lc_cat.tif     карта 1–23 (int16, nodata как у исходной, LZW, пирамиды)
  <base>_lc_cat.lyrx    стиль ArcGIS Pro (уникальные значения, цвета по кодам);
                        лежит рядом с растром и ссылается на него относительным
                        путём — можно перетащить в карту, а можно применить как
                        шаблон: Symbology -> меню (≡) -> Import from layer file,
                        или инструмент Apply Symbology From Layer
  <base>_lc_cat.clr     цветовая карта «код R G B» (Symbology -> Import color map,
                        инструмент Add Colormap)
  <base>_lc_cat.qml     стиль QGIS (подхватывается автоматически)
  <base>_lc_classes.csv площади по классам до/после и таблица переходов
  <base>_lc_log.txt     лог запуска

Стиль подходит и для исходных карт 1–19 (лишние классы просто не встретятся).

ЗАПУСК
  python 04b_add_landcover_classes.py              окно Проводника (мультивыбор)
  python 04b_add_landcover_classes.py a_cat.tif b_cat.tif [--extra mask.tif]
  python 04b_add_landcover_classes.py --style-only out_dir   только стиль

Рабочая папка = папка проекта CLASS (там лежат 04_FINAL и 00_LANDCOVER).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

VERSION = "v1"

# ───────────────────────── настройки ─────────────────────────

FINAL_BASE_NAME = "04_FINAL"
LC_BASE_NAME = "00_LANDCOVER"
WC_DIR_NAME = "ESA_WorldCover_2021"
WC_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
          "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")

FOREST = list(range(1, 14))       # лесные классы модели
NONMEADOW = list(range(1, 18))    # всё, кроме лугов 18–19

# (код WorldCover, новый код, какие классы модели заменяются; None = все)
RULES = [
    (50, 21, None),        # застройка -> 21
    (80, 22, None),        # вода -> 22
    (60, 23, None),        # открытый грунт / редкая растительность -> 23
    (40, 20, NONMEADOW),   # пашни -> 20, луга 18–19 сохраняются
    (30, 20, FOREST),      # травяные на месте лесных классов -> 20 (залежи)
    (90, 19, FOREST),      # травяные болота на месте лесных классов -> 19
]

WC_NAMES = {
    0: "нет данных", 10: "древесная растительность", 20: "кустарники",
    30: "травяные", 40: "пашни", 50: "застройка", 60: "открытый грунт",
    70: "снег и лёд", 80: "вода", 90: "травяные болота", 95: "мангры",
    100: "мхи и лишайники",
}

# код -> (полное название = подпись в легенде, цвет HEX). Названия — из эталонов
# (поле 4_type_f_1 point_segments), цвета — из Book2.xlsx.
CLASS_INFO = {
    1: ("Еловые и хвойно-широколиственные — бореальные (кустарничково-мелкотравно-зеленомошные, мелкотравные)",
         "7030A0"),
    2: ("Еловые и хвойно-широколиственные — неморальные (мелкотравно-широкотравные, широкотравные, разнотравные)",
         "FF33CC"),
    3: ("Сосново-еловые — бореальные (кустарничково-мелкотравно-зеленомошные, мелкотравные)",
         "FF6600"),
    4: ("Сосново-еловые — неморальные (мелкотравно-широкотравные, широкотравные, разнотравные)",
         "FF9953"),
    5: ("Сосново-еловые — олиготрофные (травяно-болотные, кустарничково-травяно-сфагновые)",
         "FFE1CD"),
    6: ("Дубовые — неморальные (мелкотравно-широкотравные, широкотравные, разнотравные)",
         "006600"),
    7: ("Липовые — неморальные (мелкотравно-широкотравные, широкотравные, разнотравные)",
         "FFFF00"),
    8: ("Широколиственно-еловые — неморальные (мелкотравно-широкотравные, широкотравные, разнотравные)",
         "C00000"),
    9: ("Березово-осиновые с елью и осиной, местами с дубом и липой — неморальные (мелкотравно-широкотравные, широкотравные, разнотравные)",
         "0091C4"),
    10: ("Березово-осиновые с елью и осиной, местами с дубом и липой — олиготрофные (травяно-болотные, кустарничково-травяно-сфагновые)",
         "8BE1FF"),
    11: ("Осинники с елью и дубом — неморальные (мелкотравно-широкотравные, широкотравные, разнотравные) / нитрофильные (влажнотравно-широкотравные)",
         "00B400"),
    12: ("Сероольшаники — нитрофильные (влажнотравно-широкотравные)",
         "808080"),
    13: ("Черноольшаники — нитрофильные (влажнотравно-широкотравные)",
         "404040"),
    14: ("Распавшиеся ельники — бореальные (кустарничково-мелкотравно-зеленомошные, мелкотравные)",
         "FFCCFF"),
    15: ("Распавшиеся ельники — неморальные (мелкотравно-широкотравные, широкотравные, разнотравные)",
         "E2EFDA"),
    16: ("Ивняки — нитрофильные (влажнотравно-широкотравные)",
         "C6E0B4"),
    17: ("Мелколиственная поросль и вырубки — нитрофильные (влажнотравно-широкотравные)",
         "A9D08E"),
    18: ("Луга суходольные — разнотравные",
         "CCFFFF"),
    19: ("Луга увлажнённые и открытые заболоченные местообитания — олиготрофные (травяно-болотные, кустарничково-травяно-сфагновые)",
         "CCECFF"),
    20: ("Пашни (включая залежи на ранних стадиях)",
         "FFFFCC"),
    21: ("Застройка и дороги",
         "0D0D0D"),
    22: ("Водные объекты",
         "4472C4"),
    23: ("Карьеры и торфоразработки",
         "FF0000"),
}
NEW_CODES = (20, 21, 22, 23)
LABEL_WITH_CODE = False           # True — подпись «1. Еловые…» (код перед названием)

# ───────────────────────── лог ─────────────────────────

_LOG_FH = None


def log(msg: str = "", end: str = "\n") -> None:
    """Печать в консоль и в лог (строки прогресса с \\r в лог не пишутся)."""
    print(msg, end=end, flush=True)
    if _LOG_FH is not None and end == "\n" and not msg.startswith("\r"):
        _LOG_FH.write(msg + "\n")
        _LOG_FH.flush()


def log_open(path: str) -> None:
    global _LOG_FH
    log_close()
    _LOG_FH = open(path, "w", encoding="utf-8")
    _LOG_FH.write(f"04b_add_landcover_classes.py {VERSION}  "
                  f"{_dt.datetime.now():%Y-%m-%d %H:%M:%S}\n")


def log_close() -> None:
    global _LOG_FH
    if _LOG_FH is not None:
        _LOG_FH.close()
        _LOG_FH = None


def _progress(prefix: str, done: float, total: float, t0: float) -> None:
    pct = 100.0 * done / total if total else 100.0
    el = time.time() - t0
    eta = el / done * (total - done) if done > 0 else 0
    print(f"\r  {prefix}: {pct:5.1f}%  ({el:5.0f} с, осталось ~{eta:4.0f} с)   ",
          end="", flush=True)


# ───────────────────────── WorldCover ─────────────────────────

def wc_tiles_for_bounds(lon0, lat0, lon1, lat1) -> list[str]:
    """Имена плиток WorldCover 3°×3° (по нижнему левому углу)."""
    tiles = []
    for la in range(int(np.floor(lat0 / 3) * 3), int(np.floor(lat1 / 3) * 3) + 1, 3):
        for lo in range(int(np.floor(lon0 / 3) * 3), int(np.floor(lon1 / 3) * 3) + 1, 3):
            ns = "N" if la >= 0 else "S"
            ew = "E" if lo >= 0 else "W"
            tiles.append(f"{ns}{abs(la):02d}{ew}{abs(lo):03d}")
    return tiles


def _http_size(url: str):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "04b_lc"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers.get("Content-Length", 0))
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return None
        raise


def download_tile(tile: str, cache_dir: str, retries: int = 5):
    """Скачивает плитку с докачкой (.part). None, если плитки нет (море)."""
    os.makedirs(cache_dir, exist_ok=True)
    url = WC_URL.format(tile=tile)
    dst = os.path.join(cache_dir, os.path.basename(url))
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        log(f"  {tile}: есть на диске ({os.path.getsize(dst) / 2**20:.0f} МБ)")
        return dst
    size = _http_size(url)
    if size is None:
        log(f"  {tile}: плитки нет в WorldCover (вне суши) — пропуск")
        return None
    part = dst + ".part"
    t0 = time.time()
    for attempt in range(1, retries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have >= size:
            break
        req = urllib.request.Request(url, headers={"User-Agent": "04b_lc",
                                                   "Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab") as f:
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
                    have += len(buf)
                    print(f"\r  {tile}: {have / 2**20:6.1f} / {size / 2**20:.1f} МБ "
                          f"({100 * have / size:5.1f}%)   ", end="", flush=True)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print()
            log(f"  {tile}: обрыв ({e}), попытка {attempt}/{retries}")
            time.sleep(min(30, 3 * attempt))
    print()
    if not os.path.exists(part) or os.path.getsize(part) != size:
        raise RuntimeError(f"Не удалось скачать {url}")
    os.replace(part, dst)
    log(f"  {tile}: скачано {size / 2**20:.0f} МБ за {time.time() - t0:.0f} с")
    return dst


def _grid_key(crs, transform, shape) -> str:
    s = json.dumps([crs.to_wkt(), list(transform)[:6], list(shape)])
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]


def wc_on_grid(ref_path: str, root_dir: str, threads: int | None = None) -> np.ndarray:
    """Преобладающий класс WorldCover (uint8) на сетке ref_path, с кэшем."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject, transform_bounds

    with rasterio.open(ref_path) as ref:
        crs, tr, shape = ref.crs, ref.transform, (ref.height, ref.width)
        bounds = ref.bounds
    lc_dir = os.path.join(root_dir, LC_BASE_NAME)
    os.makedirs(lc_dir, exist_ok=True)
    res = abs(tr.a)
    cache = os.path.join(lc_dir, f"WC2021_mode_{shape[1]}x{shape[0]}_"
                                 f"{res:g}m_{_grid_key(crs, tr, shape)}.tif")
    if os.path.exists(cache):
        with rasterio.open(cache) as c:
            if c.crs == crs and c.transform == tr and (c.height, c.width) == shape:
                log(f"  WorldCover на сетке карты: из кэша {os.path.basename(cache)}")
                return c.read(1)

    lon0, lat0, lon1, lat1 = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=101)
    tiles = wc_tiles_for_bounds(lon0 - 0.01, lat0 - 0.01, lon1 + 0.01, lat1 + 0.01)
    log(f"  Охват {lon0:.2f}–{lon1:.2f}° в.д., {lat0:.2f}–{lat1:.2f}° с.ш.; "
        f"плиток WorldCover: {len(tiles)} ({', '.join(tiles)})")
    wc_dir = os.path.join(lc_dir, WC_DIR_NAME)
    paths = []
    for i, t in enumerate(tiles, 1):
        log(f"  [{i}/{len(tiles)}] загрузка {t}")
        p = download_tile(t, wc_dir)
        if p:
            paths.append((t, p))
    if not paths:
        raise RuntimeError("Нет ни одной плитки WorldCover для охвата карты")

    out = np.zeros(shape, np.uint8)
    nthr = threads or min(os.cpu_count() or 4, 16)
    t0 = time.time()
    for i, (t, p) in enumerate(paths, 1):
        t1 = time.time()
        tmp = np.zeros(shape, np.uint8)
        with rasterio.open(p) as src:
            reproject(rasterio.band(src, 1), tmp, src_nodata=0, dst_nodata=0,
                      dst_transform=tr, dst_crs=crs, resampling=Resampling.mode,
                      num_threads=nthr, warp_mem_limit=2048)
        fill = (out == 0) & (tmp > 0)
        out[fill] = tmp[fill]
        _progress(f"пересчёт 10 м -> {res:g} м, плитка {i}/{len(paths)} {t}",
                  i, len(paths), t0)
        print()
        log(f"  [{i}/{len(paths)}] {t}: {time.time() - t1:.0f} с, "
            f"заполнено {100 * (out > 0).mean():.1f}% ячеек")

    prof = dict(driver="GTiff", width=shape[1], height=shape[0], count=1,
                dtype="uint8", crs=crs, transform=tr, nodata=0, compress="lzw",
                tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(cache, "w", **prof) as dst:
        dst.write(out, 1)
    log(f"  Кэш сохранён: {cache}")
    return out


def extra_on_grid(extra_path: str, ref_path: str) -> np.ndarray:
    """Свой растр с кодами 20–23 на сетке карты (mode), прочие значения -> 0."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    with rasterio.open(ref_path) as ref:
        crs, tr, shape = ref.crs, ref.transform, (ref.height, ref.width)
    out = np.zeros(shape, np.int16)
    with rasterio.open(extra_path) as src:
        same = (src.crs == crs and src.transform == tr
                and (src.height, src.width) == shape)
        if same:
            out = src.read(1).astype(np.int16)
        else:
            nd = src.nodata
            reproject(rasterio.band(src, 1), out, src_nodata=nd, dst_nodata=0,
                      dst_transform=tr, dst_crs=crs, resampling=Resampling.mode,
                      num_threads=min(os.cpu_count() or 4, 16), warp_mem_limit=2048)
    out[~np.isin(out, NEW_CODES)] = 0
    return out


# ───────────────────────── правила ─────────────────────────

def apply_rules(cat: np.ndarray, wc: np.ndarray, nodata, extra=None):
    """Возвращает новую карту и сводку по правилам."""
    valid = np.ones(cat.shape, bool) if nodata is None else (cat != nodata)
    out = cat.copy()
    summary = []
    for wc_code, new, only in RULES:
        m = valid & (wc == wc_code)
        if only is not None:
            m &= np.isin(cat, only)
        m &= (out == cat)                 # каждое правило — по исходной карте
        n = int(m.sum())
        out[m] = new
        summary.append((f"WorldCover {wc_code} {WC_NAMES.get(wc_code, '')}",
                        new, "все" if only is None else _ranges(only), n))
    if extra is not None:
        for code in NEW_CODES:
            m = valid & (extra == code) & (out != code)
            n = int(m.sum())
            out[m] = code
            summary.append((f"свой растр, значение {code}", code, "все", n))
    return out, summary


def _ranges(codes) -> str:
    codes = sorted(set(int(c) for c in codes))
    parts, s = [], codes[0]
    prev = s
    for c in codes[1:] + [None]:
        if c is not None and c == prev + 1:
            prev = c
            continue
        parts.append(f"{s}–{prev}" if prev > s else f"{s}")
        if c is not None:
            s = prev = c
    return ", ".join(parts)


# ───────────────────────── стили ─────────────────────────

def _rgb(hexs: str):
    hexs = hexs.lstrip("#")
    return int(hexs[0:2], 16), int(hexs[2:4], 16), int(hexs[4:6], 16)


def _label(code: int) -> str:
    name = CLASS_INFO.get(code, (f"класс {code}", "BFBFBF"))[0]
    return f"{code}. {name}" if LABEL_WITH_CODE else name


def _codes_for_style(codes=None) -> list[int]:
    if codes is None:
        return sorted(CLASS_INFO)
    return sorted(set(int(c) for c in codes) | set(CLASS_INFO))


def write_lyrx(path: str, raster_name: str | None, codes=None,
               layer_name: str | None = None) -> None:
    """Стиль ArcGIS Pro: CIMRasterLayer + CIMRasterUniqueValueColorizer.

    raster_name — имя файла рядом с .lyrx (относительная ссылка DATABASE=.);
    None — только шаблон символики без источника данных.
    """
    classes = []
    for c in _codes_for_style(codes):
        hexs = CLASS_INFO.get(c, ("", "BFBFBF"))[1]
        r, g, b = _rgb(hexs)
        classes.append({
            "type": "CIMRasterUniqueValueClass",
            "values": [str(c)],
            "label": _label(c),
            "color": {"type": "CIMRGBColor", "values": [r, g, b, 100]},
        })
    name = layer_name or (os.path.splitext(raster_name)[0] if raster_name
                          else "Классы растительности 1–23")
    layer = {
        "type": "CIMRasterLayer",
        "name": name,
        "uRI": "CIMPATH=map/classes_lc.xml",
        "sourceModifiedTime": {"type": "TimeInstant"},
        "useSourceMetadata": True,
        "description": name,
        "layerType": "Operational",
        "showLegends": True,
        "visibility": True,
        "displayCacheType": "Permanent",
        "maxDisplayCacheAge": 5,
        "showPopups": True,
        "serviceLayerID": -1,
        "refreshRate": -1,
        "refreshRateUnit": "esriTimeUnitsSeconds",
        "blendingMode": "Alpha",
    }
    if raster_name:
        layer["dataConnection"] = {
            "type": "CIMStandardDataConnection",
            "workspaceConnectionString": "DATABASE=.",
            "workspaceFactory": "Raster",
            "dataset": raster_name,
            "datasetType": "esriDTAny",
        }
    layer["colorizer"] = {
        "type": "CIMRasterUniqueValueColorizer",
        "resamplingType": "NearestNeighbor",
        "noDataColor": {"type": "CIMRGBColor", "values": [255, 255, 255, 0]},
        "defaultColor": {"type": "CIMRGBColor", "values": [191, 191, 191, 100]},
        "fieldName": "Value",
        "groups": [{"type": "CIMRasterUniqueValueGroup", "heading": "Value",
                    "classes": classes}],
        "colorScheme": {
            "type": "CIMRandomHSVColorRamp",
            "colorSpace": {"type": "CIMICCColorSpace", "url": "Default RGB"},
            "maxH": 360, "minS": 15, "maxS": 30, "minV": 99, "maxV": 100,
            "minAlpha": 100, "maxAlpha": 100,
        },
    }
    doc = {
        "type": "CIMLayerDocument",
        "version": "2.9.0",
        "build": 32739,
        "layers": ["CIMPATH=map/classes_lc.xml"],
        "layerDefinitions": [layer],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=True, indent=2)   # кириллица экранируется (ASCII)


def write_clr(path: str, codes=None) -> None:
    """Цветовая карта ArcGIS: «код R G B» по строке."""
    with open(path, "w", encoding="ascii") as f:
        for c in _codes_for_style(codes):
            r, g, b = _rgb(CLASS_INFO.get(c, ("", "BFBFBF"))[1])
            f.write(f"{c} {r} {g} {b}\n")


def _xml_escape(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&apos;"))


def write_qml(path: str, codes=None) -> None:
    """Стиль QGIS (paletted)."""
    items = []
    for c in _codes_for_style(codes):
        hexs = "#" + CLASS_INFO.get(c, ("", "BFBFBF"))[1].lower()
        items.append(f'          <paletteEntry value="{c}" color="{hexs}" alpha="255" '
                     f'label="{_xml_escape(_label(c))}"/>')
    qml = (
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
        '<qgis version="3.28" styleCategories="AllStyleCategories">\n'
        "  <pipe>\n"
        '      <rasterrenderer type="paletted" band="1" opacity="1" alphaBand="-1" '
        'nodataColor="">\n'
        "        <rasterTransparency/>\n"
        "        <colorPalette>\n" + "\n".join(items) + "\n        </colorPalette>\n"
        '        <colorramp type="randomcolors" name="[source]"/>\n'
        "      </rasterrenderer>\n"
        '    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>\n'
        '    <huesaturation saturation="0" grayscaleMode="0" colorizeOn="0"/>\n'
        '    <rasterresampler maxOversampling="2"/>\n'
        "  </pipe>\n"
        "  <blendMode>0</blendMode>\n"
        "</qgis>\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(qml)


def write_styles(out_dir: str, stem: str, raster_name: str | None, codes=None) -> list:
    files = [os.path.join(out_dir, stem + ext) for ext in (".lyrx", ".clr", ".qml")]
    write_lyrx(files[0], raster_name, codes)
    write_clr(files[1], codes)
    write_qml(files[2], codes)
    return files


# ───────────────────────── обработка файла ─────────────────────────

def _out_stem(in_path: str) -> str:
    stem = os.path.splitext(os.path.basename(in_path))[0]
    if stem.endswith("_cat"):
        stem = stem[:-4]
    return stem


def process_file(in_path: str, root_dir: str, extra_path: str | None = None) -> dict:
    import pandas as pd
    import rasterio
    from rasterio.enums import Resampling

    t0 = time.time()
    out_dir = os.path.dirname(os.path.abspath(in_path))
    stem = _out_stem(in_path)
    out_tif = os.path.join(out_dir, f"{stem}_lc_cat.tif")
    log_open(os.path.join(out_dir, f"{stem}_lc_log.txt"))
    log(f"Входной растр: {in_path}")

    with rasterio.open(in_path) as src:
        cat = src.read(1)
        prof = src.profile.copy()
        nodata = src.nodata
        px_km2 = abs(src.transform.a * src.transform.e) / 1e6
        log(f"  {src.width} x {src.height} пикс., {abs(src.transform.a):g} м, "
            f"{src.crs.to_string()}, {src.dtypes[0]}, nodata={nodata}")
    if not np.issubdtype(cat.dtype, np.integer):
        raise ValueError("Ожидается целочисленный растр классов (*_cat.tif)")
    valid = np.ones(cat.shape, bool) if nodata is None else (cat != nodata)
    n_valid = int(valid.sum())
    present = np.unique(cat[valid])
    log(f"  Классы в карте: {_ranges(present) if present.size else '—'}; "
        f"ячеек с данными: {n_valid:,}".replace(",", " "))
    clash = [c for c in present if c in NEW_CODES]
    if clash:
        log(f"  ВНИМАНИЕ: коды {clash} уже есть в карте — они будут дополнены, "
            "а не заменены")

    log("\n[1/3] WorldCover 2021 на сетке карты")
    wc = wc_on_grid(in_path, root_dir)
    extra = None
    if extra_path:
        log(f"  Свой растр поверх: {extra_path}")
        extra = extra_on_grid(extra_path, in_path)
        log(f"  в нём ячеек с кодами 20–23: {int((extra > 0).sum()):,}".replace(",", " "))

    log("\n[2/3] Применение правил")
    out, summary = apply_rules(cat, wc, nodata, extra)
    w = max(len(s[0]) for s in summary)
    for src_name, new, only, n in summary:
        log(f"  {src_name:<{w}} -> {new:>2}  (классы {only:<6}): "
            f"{n:>10,} ячеек  {100 * n / max(n_valid, 1):5.1f}%".replace(",", " "))
    changed = valid & (out != cat)
    log(f"  Всего изменено: {int(changed.sum()):,} ячеек "
        f"({100 * changed.mean() / max(valid.mean(), 1e-9):.1f}% карты)".replace(",", " "))

    log("\n[3/3] Запись результата")
    prof.update(driver="GTiff", compress="lzw", tiled=True, blockxsize=512,
                blockysize=512, BIGTIFF="IF_SAFER")
    prof.pop("predictor", None)
    if out.dtype != cat.dtype:
        out = out.astype(cat.dtype)
    with rasterio.open(out_tif, "w", **prof) as dst:
        dst.write(out, 1)
        dst.update_tags(SOURCE=os.path.basename(in_path),
                        LANDCOVER="ESA WorldCover 2021 v200 (mode 10 m -> grid)",
                        SCRIPT=f"04b_add_landcover_classes.py {VERSION}")
        for c in _codes_for_style(np.unique(out[valid])):
            dst.update_tags(1, **{f"CLASS_{c}": _label(c)})
        lv = [f for f in (2, 4, 8, 16, 32) if min(out.shape) // f >= 64]
        if lv:
            dst.build_overviews(lv, Resampling.nearest)
            dst.update_tags(ns="rio_overview", resampling="nearest")
    log(f"  {out_tif}")

    codes_out = np.unique(out[valid])
    styles = write_styles(out_dir, f"{stem}_lc_cat", os.path.basename(out_tif), codes_out)
    for s in styles:
        log(f"  {s}")

    # таблица классов и переходов
    vb = cat[valid].astype(np.int64)
    va = out[valid].astype(np.int64)
    off = int(min(vb.min(), va.min())) if vb.size else 0
    cb = np.bincount(vb - off)
    ca = np.bincount(va - off)
    rows = []
    for c in _codes_for_style(np.union1d(present, codes_out)):
        nb = int(cb[c - off]) if 0 <= c - off < cb.size else 0
        na = int(ca[c - off]) if 0 <= c - off < ca.size else 0
        if nb == 0 and na == 0:
            continue
        rows.append({"Value": c, "Класс": CLASS_INFO.get(c, (f"класс {c}",))[0],
                     "Цвет": "#" + CLASS_INFO.get(c, ("", "BFBFBF"))[1],
                     "Ячеек до": nb, "% до": round(100 * nb / n_valid, 2),
                     "Ячеек после": na, "% после": round(100 * na / n_valid, 2),
                     "км² после": round(na * px_km2, 2)})
    df = pd.DataFrame(rows)
    ch_b, ch_a = cat[changed].astype(np.int64), out[changed].astype(np.int64)
    if ch_b.size:
        key = ch_b * 1000 + ch_a
        k, n = np.unique(key, return_counts=True)
        tr_df = pd.DataFrame({"Было": k // 1000, "Стало": k % 1000, "Ячеек": n})
        tr_df["км²"] = (tr_df["Ячеек"] * px_km2).round(2)
        tr_df["% исходного класса"] = [
            round(100 * x / max(int(cb[b - off]), 1), 1)
            for b, x in zip(tr_df["Было"], tr_df["Ячеек"])]
        tr_df = tr_df.sort_values(["Было", "Стало"])
    else:
        tr_df = pd.DataFrame(columns=["Было", "Стало", "Ячеек", "км²", "% исходного класса"])
    csv_path = os.path.join(out_dir, f"{stem}_lc_classes.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        f.write("# Классы до и после добавления категорий 20–23\n")
        df.to_csv(f, index=False, sep=";", decimal=",")
        f.write("\n# Переходы (было -> стало)\n")
        tr_df.to_csv(f, index=False, sep=";", decimal=",")
    log(f"  {csv_path}")

    log("\n  Код   до, %  после, %  Класс")
    for r in rows:
        log(f"  {r['Value']:>3}  {r['% до']:6.2f}  {r['% после']:8.2f}  {r['Класс']}")
    log(f"\nГотово за {time.time() - t0:.0f} с")
    log_close()
    return {"out": out_tif, "csv": csv_path, "styles": styles}


# ───────────────────────── запуск ─────────────────────────

def _in_notebook() -> bool:
    return "ipykernel" in sys.modules


def _initial_dir(root_dir: str) -> str:
    fb = os.path.join(root_dir, FINAL_BASE_NAME)
    if os.path.isdir(fb):
        subs = [os.path.join(fb, d) for d in os.listdir(fb)
                if os.path.isdir(os.path.join(fb, d))]
        return max(subs, key=os.path.getmtime) if subs else fb
    return root_dir


def _ask_files(root_dir: str) -> list[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        win = tk.Tk()
        win.withdraw()
        win.attributes("-topmost", True)
        paths = filedialog.askopenfilenames(
            parent=win, title="Классифицированные растры (*_cat.tif), можно несколько",
            initialdir=_initial_dir(root_dir),
            filetypes=[("Классифицированные растры", "*_cat.tif"),
                       ("GeoTIFF", "*.tif *.tiff"), ("Все файлы", "*.*")])
        win.destroy()
        return [os.path.normpath(p) for p in paths]
    except Exception as e:  # нет дисплея
        print(f"  Окно выбора недоступно ({e}). Пути через ';':")
        raw = input("  > ").strip().strip('"')
        return [p.strip().strip('"') for p in raw.split(";") if p.strip()]


def _ask_extra(root_dir: str):
    try:
        import tkinter as tk
        from tkinter import filedialog
        win = tk.Tk()
        win.withdraw()
        win.attributes("-topmost", True)
        p = filedialog.askopenfilename(
            parent=win, title="Свой растр с кодами 20–23",
            initialdir=os.path.join(root_dir, LC_BASE_NAME)
            if os.path.isdir(os.path.join(root_dir, LC_BASE_NAME)) else root_dir,
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("Все файлы", "*.*")])
        win.destroy()
        return os.path.normpath(p) if p else None
    except Exception:
        p = input("  Путь к своему растру: ").strip().strip('"')
        return p or None


def _print_rules() -> None:
    print("\nПравила (RULES в начале скрипта):")
    for wc_code, new, only in RULES:
        print(f"  WorldCover {wc_code:>3} {WC_NAMES.get(wc_code, ''):<16} -> {new}  "
              f"для классов {'все' if only is None else _ranges(only)}")
    print("  остальное (деревья, кустарники, мхи) — класс модели без изменений")


def _run_interactive() -> None:
    root_dir = os.getcwd()
    print("=" * 72)
    print(f"04b_add_landcover_classes.py {VERSION} — категории 20–23 + стиль ArcGIS/QGIS")
    print("=" * 72)
    for c in NEW_CODES:
        print(f"  {c}  {CLASS_INFO[c][0]}")
    _print_rules()
    files = _ask_files(root_dir)
    if not files:
        print("Файлы не выбраны — выход.")
        return
    print(f"\nВыбрано файлов: {len(files)}")
    for p in files:
        print(f"  {p}")
    raw = input("\nИсточник: [1] только WorldCover 2021 (Enter)  "
                "[2] WorldCover + свой растр 20–23 поверх: ").strip()
    extra = _ask_extra(root_dir) if raw == "2" else None
    _run_files(files, root_dir, extra)
    if not _in_notebook() and sys.stdin and sys.stdin.isatty():
        input("\nEnter — закрыть окно")


def _run_files(files, root_dir, extra=None) -> None:
    done, t0 = [], time.time()
    for i, p in enumerate(files, 1):
        print(f"\n{'─' * 72}\nФайл {i}/{len(files)} ({100 * (i - 1) / len(files):.0f}% "
              f"выполнено): {os.path.basename(p)}")
        try:
            done.append(process_file(p, root_dir, extra))
        except Exception as e:
            log(f"ОШИБКА: {e}")
            log_close()
            import traceback
            traceback.print_exc()
    print(f"\n{'=' * 72}\nОбработано {len(done)}/{len(files)} файлов (100%) "
          f"за {time.time() - t0:.0f} с")
    for d in done:
        print(f"  {d['out']}")
        print(f"     стиль ArcGIS Pro: {d['styles'][0]}")


def main(argv=None) -> None:
    if argv is None:
        argv = sys.argv[1:]
        prog = os.path.basename(sys.argv[0]) if sys.argv else ""
        if "ipykernel" in prog.lower() or "kernel" in prog.lower() or _in_notebook():
            argv = [a for a in argv if not (a.startswith("-f") or a.endswith(".json"))]
    if not argv:
        _run_interactive()
        return
    ap = argparse.ArgumentParser(description="Категории 20–23 из WorldCover + стиль")
    ap.add_argument("inputs", nargs="*", help="*_cat.tif")
    ap.add_argument("--extra", default=None, help="свой растр с кодами 20–23")
    ap.add_argument("--root", default=os.getcwd(), help="папка проекта (00_LANDCOVER)")
    ap.add_argument("--style-only", default=None, metavar="DIR",
                    help="только записать classes_1_23.lyrx/.clr/.qml в папку")
    a = ap.parse_args(argv)
    if a.style_only:
        os.makedirs(a.style_only, exist_ok=True)
        for f in write_styles(a.style_only, "classes_1_23", None):
            print(f)
        return
    files = []
    for p in a.inputs:
        files.extend(sorted(glob.glob(p)) or [p])
    _run_files(files, a.root, a.extra)


if __name__ == "__main__":
    main()
