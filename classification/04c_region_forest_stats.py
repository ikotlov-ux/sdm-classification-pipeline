#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04c_region_forest_stats.py — площади и доли лесных категорий и лесистость
по субъектам РФ (областям), на которые попадает классифицированная карта.

ВХОД: карта *_cat.tif из 04_predict_map / 04b_add_landcover_classes (коды 1–23).

ГРАНИЦЫ: по умолчанию субъекты РФ geoBoundaries gbOpen RUS ADM1 (OpenStreetMap,
ODbL; состояние ~2017 г., Москва — в границах после 2012 г., с Новой Москвой).
Файл скачивается один раз в 00_BOUNDARIES. Можно указать свой слой границ
(GPKG / SHP / GeoJSON) и поле с названием.

ЛЕСНЫЕ КАТЕГОРИИ (настройка FOREST_CODES): 1–17 — все типы древесной
растительности, включая распавшиеся ельники (14–15), ивняки (16) и
мелколиственную поросль с вырубками (17). Нелесные: 18–23 (луга, болота,
пашни, застройка, вода, карьеры). Лесистость дополнительно считается в узком
варианте STRICT_CODES = 1–13 (сформированные леса без 14–17).

ПЛОЩАДИ — истинные, с поправкой на искажение площадей проекции (для UTM на
краю зоны до ~0.2 %): масштаб площадей считается pyproj на сетке 64×64 пикс.
Пиксель относится к субъекту по положению центра.

ВЫХОД рядом с растром: <base>_regions_forest.xlsx
  Лесистость       по субъектам: площадь субъекта, охват картой, лес 1–17 и
                   1–13 (км² и % от площади субъекта в пределах карты)
  Лесные, км²      площади классов 1–17 по субъектам
  Лесные, %        доли классов 1–17 (лесные = 100 % в каждом субъекте)
  Все классы, км²  площади всех классов 1–23
Строки «Москва и Московская область» и «Вся карта» — сводные. Для сравнения
добавлена официальная лесистость Рослесхоза (OFFICIAL_FOREST_COVER).
Лесистость частично покрытых субъектов относится только к части в пределах
карты (столбец «Охват картой, %»).
  <base>_regions_forest_log.txt — лог

ЗАПУСК
  python 04c_region_forest_stats.py                окно Проводника (мультивыбор)
  python 04c_region_forest_stats.py a_cat.tif [--regions my.gpkg --name-field NAME]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import os
import sys
import time
import urllib.request

import numpy as np

VERSION = "v1"

FINAL_BASE_NAME = "04_FINAL"
BND_BASE_NAME = "00_BOUNDARIES"
GB_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/"
          "gbOpen/RUS/ADM1/geoBoundaries-RUS-ADM1.geojson")
GB_FILE = "geoBoundaries-RUS-ADM1.geojson"

FOREST_CODES = list(range(1, 18))     # лесные = 100 %
STRICT_CODES = list(range(1, 14))     # узкий вариант лесистости
NONFOREST_CODES = list(range(18, 24))

# объединённая строка (ISO-коды субъектов)
COMBINED = {"Москва и Московская область": ["RU-MOW", "RU-MOS"]}

EQUAL_AREA = ("+proj=aea +lat_1=52 +lat_2=58 +lat_0=55 +lon_0=37 "
              "+datum=WGS84 +units=m +no_defs")

RU_NAMES = {
    "RU-MOW": "Москва", "RU-MOS": "Московская область", "RU-TVE": "Тверская область",
    "RU-SMO": "Смоленская область", "RU-KLU": "Калужская область",
    "RU-TUL": "Тульская область", "RU-RYA": "Рязанская область",
    "RU-VLA": "Владимирская область", "RU-YAR": "Ярославская область",
    "RU-IVA": "Ивановская область", "RU-KOS": "Костромская область",
    "RU-BRY": "Брянская область", "RU-ORL": "Орловская область",
    "RU-LIP": "Липецкая область", "RU-NGR": "Новгородская область",
    "RU-PSK": "Псковская область", "RU-VGG": "Волгоградская область",
    "RU-VOR": "Воронежская область", "RU-TAM": "Тамбовская область",
    "RU-PNZ": "Пензенская область", "RU-NIZ": "Нижегородская область",
    "RU-KRS": "Курская область", "RU-BEL": "Белгородская область",
    "RU-VLG": "Вологодская область", "RU-LEN": "Ленинградская область",
    "RU-SPE": "Санкт-Петербург",
}

# Официальная лесистость (Рослесхоз, открытые данные 7705598840-forestcover,
# на 01.01.2026, %) — для сравнения; относится ко ВСЕЙ площади субъекта
# и только к покрытым лесной растительностью землям госреестра.
OFFICIAL_DATE = "01.01.2026"
OFFICIAL_FOREST_COVER = {
    "RU-BEL": 8.8, "RU-BRY": 32.9, "RU-VLA": 51.7, "RU-VOR": 8.3, "RU-IVA": 45.6,
    "RU-KLU": 45.4, "RU-KOS": 73.0, "RU-KRS": 8.2, "RU-LIP": 8.8, "RU-MOS": 41.8,
    "RU-ORL": 8.1, "RU-RYA": 25.4, "RU-SMO": 42.0, "RU-TAM": 10.6, "RU-TVE": 54.7,
    "RU-TUL": 14.3, "RU-YAR": 45.4, "RU-MOW": 1.0, "RU-VLG": 68.4, "RU-LEN": 58.0,
    "RU-NGR": 64.7, "RU-PSK": 39.1, "RU-MOR": 27.1, "RU-NIZ": 47.8, "RU-PNZ": 21.0,
}

CLASS_NAMES = {
    1: "Еловые и хвойно-широколиственные — бореальные (кустарничково-мелкотравно-зеленомошные, мелкотравные)",
    2: "Еловые и хвойно-широколиственные — неморальные (мелкотравно-широкотравные, широкотравные, разнотравные)",
    3: "Сосново-еловые — бореальные (кустарничково-мелкотравно-зеленомошные, мелкотравные)",
    4: "Сосново-еловые — неморальные (мелкотравно-широкотравные, широкотравные, разнотравные)",
    5: "Сосново-еловые — олиготрофные (травяно-болотные, кустарничково-травяно-сфагновые)",
    6: "Дубовые — неморальные (мелкотравно-широкотравные, широкотравные, разнотравные)",
    7: "Липовые — неморальные (мелкотравно-широкотравные, широкотравные, разнотравные)",
    8: "Широколиственно-еловые — неморальные (мелкотравно-широкотравные, широкотравные, разнотравные)",
    9: "Березово-осиновые с елью и осиной, местами с дубом и липой — неморальные (мелкотравно-широкотравные, широкотравные, разнотравные)",
    10: "Березово-осиновые с елью и осиной, местами с дубом и липой — олиготрофные (травяно-болотные, кустарничково-травяно-сфагновые)",
    11: "Осинники с елью и дубом — неморальные (мелкотравно-широкотравные, широкотравные, разнотравные) / нитрофильные (влажнотравно-широкотравные)",
    12: "Сероольшаники — нитрофильные (влажнотравно-широкотравные)",
    13: "Черноольшаники — нитрофильные (влажнотравно-широкотравные)",
    14: "Распавшиеся ельники — бореальные (кустарничково-мелкотравно-зеленомошные, мелкотравные)",
    15: "Распавшиеся ельники — неморальные (мелкотравно-широкотравные, широкотравные, разнотравные)",
    16: "Ивняки — нитрофильные (влажнотравно-широкотравные)",
    17: "Мелколиственная поросль и вырубки — нитрофильные (влажнотравно-широкотравные)",
    18: "Луга суходольные — разнотравные",
    19: "Луга увлажнённые и открытые заболоченные местообитания — олиготрофные (травяно-болотные, кустарничково-травяно-сфагновые)",
    20: "Пашни (включая залежи на ранних стадиях)",
    21: "Застройка и дороги",
    22: "Водные объекты",
    23: "Карьеры и торфоразработки",
}

# ───────────────────────── лог ─────────────────────────

_LOG = None


def log(msg: str = "") -> None:
    print(msg, flush=True)
    if _LOG is not None:
        _LOG.write(msg + "\n")
        _LOG.flush()


def _rng(codes) -> str:
    c = sorted(codes)
    return f"{c[0]}–{c[-1]}" if c == list(range(c[0], c[-1] + 1)) else ", ".join(map(str, c))


# ───────────────────────── границы ─────────────────────────

def get_boundaries(root_dir: str) -> str:
    d = os.path.join(root_dir, BND_BASE_NAME)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, GB_FILE)
    if os.path.exists(p) and os.path.getsize(p) > 0:
        log(f"  Границы субъектов: {p}")
        return p
    log(f"  Загрузка границ субъектов РФ (geoBoundaries, ~60 МБ): {GB_URL}")
    tmp = p + ".part"
    req = urllib.request.Request(GB_URL, headers={"User-Agent": "04c_region_stats"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0) or 0)
        done = 0
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            pct = f" ({100 * done / total:5.1f}%)" if total else ""
            print(f"\r  {done / 2**20:6.1f} МБ{pct}   ", end="", flush=True)
    print()
    os.replace(tmp, p)
    log(f"  Сохранено за {time.time() - t0:.0f} с: {p}")
    return p


def load_regions(path: str, name_field: str | None, crs, bounds):
    """Субъекты, пересекающие охват карты, в CRS карты: GeoDataFrame [key, name, geometry, area_km2]."""
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    frame = gpd.GeoSeries([box(*bounds)], crs=crs)
    gdf = gdf[gdf.intersects(frame.to_crs(gdf.crs).iloc[0])].copy()
    if gdf.empty:
        raise RuntimeError("Ни один полигон границ не пересекает карту")
    if name_field is None and "shapeISO" in gdf.columns:
        gdf["key"] = gdf["shapeISO"].astype(str)
        gdf["name"] = [RU_NAMES.get(k, n) for k, n in zip(gdf["shapeISO"], gdf["shapeName"])]
    else:
        fld = name_field or next(c for c in gdf.columns if c != "geometry"
                                 and gdf[c].dtype == object)
        gdf["name"] = gdf[fld].astype(str)
        gdf["key"] = gdf["name"]
        gdf = gdf.dissolve(by="key", as_index=False)
    gdf["geometry"] = gdf.geometry.make_valid() if hasattr(gdf.geometry, "make_valid") \
        else gdf.geometry.buffer(0)
    gdf["area_km2"] = gdf.to_crs(EQUAL_AREA).area / 1e6
    gdf = gdf.to_crs(crs)
    return gdf[["key", "name", "area_km2", "geometry"]].reset_index(drop=True)


# ───────────────────────── площади пикселей ─────────────────────────

def pixel_area_grid(src, step: int = 64) -> np.ndarray:
    """Истинная площадь пикселя (км²) на грубой сетке step×step пикс."""
    from pyproj import CRS, Proj, Transformer

    h, w = src.height, src.width
    tr = src.transform
    rows = np.arange(step / 2, h + step / 2, step).clip(0, h - 0.5)
    cols = np.arange(step / 2, w + step / 2, step).clip(0, w - 0.5)
    cc, rr = np.meshgrid(cols, rows)
    xs = tr.c + cc * tr.a + rr * tr.b
    ys = tr.f + cc * tr.d + rr * tr.e
    crs = CRS.from_user_input(src.crs.to_wkt())
    nominal = abs(tr.a * tr.e - tr.b * tr.d) / 1e6
    if crs.is_geographic:
        lat = np.radians(ys)
        dlon, dlat = np.radians(abs(tr.a)), np.radians(abs(tr.e))
        R = 6371.0072
        return (R ** 2 * dlon * np.abs(np.sin(lat + dlat / 2) - np.sin(lat - dlat / 2)))
    to_ll = Transformer.from_crs(crs, 4326, always_xy=True)
    lon, lat = to_ll.transform(xs, ys)
    f = Proj(crs).get_factors(lon, lat)
    return nominal / np.asarray(f.areal_scale)


# ───────────────────────── расчёт ─────────────────────────

def compute(raster: str, regions, step: int = 64):
    import rasterio
    from rasterio import features

    with rasterio.open(raster) as src:
        h, w = src.height, src.width
        nodata = src.nodata
        area_c = pixel_area_grid(src, step)
        log(f"  Площадь пикселя: {area_c.min() * 1e6:.1f}–{area_c.max() * 1e6:.1f} м² "
            f"(номинал {abs(src.transform.a * src.transform.e):.0f} м²)")
        shapes = [(g, i + 1) for i, g in enumerate(regions.geometry)]
        rid = features.rasterize(shapes, out_shape=(h, w), transform=src.transform,
                                 fill=0, dtype="int32", all_touched=False)
        n_reg = len(regions) + 1
        K = 256
        acc = np.zeros(n_reg * K, np.float64)
        cnt = np.zeros(n_reg * K, np.int64)
        strip = step * 8
        t0 = time.time()
        for r0 in range(0, h, strip):
            r1 = min(h, r0 + strip)
            a = src.read(1, window=((r0, r1), (0, w))).astype(np.int64)
            valid = (a >= 0) & (a < K)
            if nodata is not None:
                valid &= a != nodata
            wa = np.repeat(np.repeat(area_c[r0 // step:(r1 + step - 1) // step], step, 0),
                           step, 1)
            wa = wa[: r1 - r0, :w]
            key = (rid[r0:r1] * K + a)[valid]
            acc += np.bincount(key, weights=wa[valid], minlength=n_reg * K)
            cnt += np.bincount(key, minlength=n_reg * K)
            pct = 100 * r1 / h
            el = time.time() - t0
            print(f"\r  Подсчёт: {pct:5.1f}%  ({el:4.0f} с)   ", end="", flush=True)
        print()
    return acc.reshape(n_reg, K), cnt.reshape(n_reg, K)


def _add_row(df, values):
    import pandas as pd
    return pd.concat([df.astype(object), pd.DataFrame([values], columns=df.columns)],
                     ignore_index=True)


def build_tables(acc, cnt, regions):
    import pandas as pd

    codes_all = [c for c in range(1, acc.shape[1]) if acc[:, c].sum() > 0]
    cols = []    # (name, key-list, area_km2)
    order = np.argsort(-acc[1:].sum(1))
    for i in order:
        r = regions.iloc[i]
        cols.append((r["name"], [i + 1], float(r["area_km2"]), r["key"]))
    keys = list(regions["key"])
    for nm, isos in COMBINED.items():
        idx = [keys.index(k) + 1 for k in isos if k in keys]
        if len(idx) == len(isos):
            cols.append((nm, idx, float(sum(regions.iloc[j - 1]["area_km2"] for j in idx)), ""))
    cols.append(("Вся карта", list(range(acc.shape[0])), np.nan, ""))

    def s(idx, codes):
        return float(acc[np.ix_(idx, codes)].sum())

    rows = []
    for nm, idx, a_reg, key in cols:
        mapped = s(idx, list(range(1, acc.shape[1])))
        forest = s(idx, FOREST_CODES)
        strict = s(idx, STRICT_CODES)
        nonf = s(idx, NONFOREST_CODES)
        cov = 100 * mapped / a_reg if a_reg and a_reg > 0 else np.nan
        rows.append({
            "Субъект": nm, "ISO": key,
            "Площадь субъекта, км²": round(a_reg, 1) if a_reg == a_reg else None,
            "В пределах карты, км²": round(mapped, 1),
            "Охват картой, %": round(min(cov, 100.0), 1) if cov == cov else None,
            f"Лес {_rng(FOREST_CODES)}, км²": round(forest, 1),
            f"Лесистость {_rng(FOREST_CODES)}, %": round(100 * forest / mapped, 2) if mapped else None,
            f"Лес {_rng(STRICT_CODES)}, км²": round(strict, 1),
            f"Лесистость {_rng(STRICT_CODES)}, %": round(100 * strict / mapped, 2) if mapped else None,
            f"Нелесные {_rng(NONFOREST_CODES)}, км²": round(nonf, 1),
            "Охват": ("" if cov != cov else "полностью" if cov >= 99.5 else "частично"),
            f"Лесистость по Рослесхозу на {OFFICIAL_DATE} (весь субъект), %":
                OFFICIAL_FOREST_COVER.get(key),
        })
    t_cover = pd.DataFrame(rows)

    names = [c[0] for c in cols]
    km = pd.DataFrame({nm: [s(idx, [c]) for c in FOREST_CODES] for nm, idx, *_ in cols})
    km.insert(0, "Класс", [CLASS_NAMES.get(c, f"класс {c}") for c in FOREST_CODES])
    km.insert(0, "Код", FOREST_CODES)
    tot = km[names].sum()
    pct = km.copy()
    for nm in names:
        pct[nm] = (100 * km[nm] / tot[nm]).round(2) if tot[nm] > 0 else 0.0
    km[names] = km[names].round(1)
    km = _add_row(km, ["", f"Итого лесные ({_rng(FOREST_CODES)})"]
                  + [round(tot[n], 1) for n in names])
    pct = _add_row(pct, ["", f"Итого лесные ({_rng(FOREST_CODES)})"]
                   + [100.0 if tot[n] > 0 else 0.0 for n in names])

    allc = pd.DataFrame({nm: [round(s(idx, [c]), 1) for c in codes_all] for nm, idx, *_ in cols})
    allc.insert(0, "Класс", [CLASS_NAMES.get(c, f"класс {c}") for c in codes_all])
    allc.insert(0, "Код", codes_all)
    allc = _add_row(allc, ["", "Итого"] + [round(s(idx, codes_all), 1) for _, idx, *_ in cols])
    return t_cover, km, pct, allc


def write_xlsx(path, t_cover, km, pct, allc, meta: list[str]):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    bold = Font(bold=True)
    head_fill = PatternFill("solid", fgColor="DDEBF7")
    tot_fill = PatternFill("solid", fgColor="F2F2F2")
    thin = Side(style="thin", color="BFBFBF")
    border = Border(top=thin, bottom=thin, left=thin, right=thin)

    def sheet(ws, df, num_fmt, title, name_col=None):
        ws.append([title])
        ws["A1"].font = Font(bold=True, size=12)
        ws.append([])
        ws.append(list(df.columns))
        for c in ws[3]:
            c.font = bold
            c.fill = head_fill
            c.border = border
            c.alignment = Alignment(vertical="center")
        for row in df.itertuples(index=False):
            ws.append([None if (isinstance(v, float) and v != v) else v for v in row])
        last = ws.max_row
        for r in ws.iter_rows(min_row=4, max_row=last):
            for c in r:
                c.border = border
                if isinstance(c.value, (int, float)) and not isinstance(c.value, bool):
                    c.number_format = num_fmt.get(ws.cell(3, c.column).value, "#,##0.0")
        # сводные строки
        for r in range(4, last + 1):
            first = str(ws.cell(r, 1).value or "") + str(ws.cell(r, 2).value or "")
            if "Итого" in first or "Вся карта" in first or " и " in str(ws.cell(r, 1).value or ""):
                for c in ws[r]:
                    c.fill = tot_fill
                    c.font = bold
        # ширина без переносов
        for j, col in enumerate(df.columns, 1):
            # заголовок полужирный, кириллица шире цифр — запас по ширине
            vals = [str(v) for v in df.iloc[:, j - 1].tolist()]
            wdt = max([len(str(col)) * 1.25] + [len(v) * 1.15 for v in vals]) + 3
            ws.column_dimensions[get_column_letter(j)].width = min(max(wdt, 8), 150)
        ws.freeze_panes = ws.cell(4, 3 if name_col else 2)

    ws = wb.active
    ws.title = "Лесистость"
    fmt = {c: ("0.0" if "Рослесхоз" in c else
               "0.00" if c.endswith(", %") and "Лесистость" in c else
               "0.0" if c.endswith(", %") else "#,##0.0") for c in t_cover.columns}
    sheet(ws, t_cover, fmt, "Лесистость субъектов в пределах карты "
          f"(лесные {_rng(FOREST_CODES)}; узкий вариант {_rng(STRICT_CODES)})")
    r = ws.max_row + 2
    for m in meta:
        ws.cell(r, 1, m)
        r += 1

    ws2 = wb.create_sheet("Лесные, км²")
    sheet(ws2, km, {}, f"Площади лесных категорий {_rng(FOREST_CODES)}, км²", name_col=True)
    ws3 = wb.create_sheet("Лесные, %")
    sheet(ws3, pct, {c: "0.00" for c in pct.columns},
          f"Доли лесных категорий, % (лесные {_rng(FOREST_CODES)} = 100 % в каждом субъекте)",
          name_col=True)
    ws4 = wb.create_sheet("Все классы, км²")
    sheet(ws4, allc, {}, "Площади всех классов карты, км²", name_col=True)
    wb.save(path)


def process(raster: str, root_dir: str, regions_path: str | None = None,
            name_field: str | None = None) -> str:
    global _LOG
    import rasterio

    out_dir = os.path.dirname(os.path.abspath(raster))
    stem = os.path.splitext(os.path.basename(raster))[0]
    if stem.endswith("_cat"):
        stem = stem[:-4]
    out_xlsx = os.path.join(out_dir, f"{stem}_regions_forest.xlsx")
    _LOG = open(os.path.join(out_dir, f"{stem}_regions_forest_log.txt"), "w", encoding="utf-8")
    log(f"04c_region_forest_stats.py {VERSION}  {_dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"Растр: {raster}")
    t0 = time.time()
    with rasterio.open(raster) as src:
        crs, bounds = src.crs, src.bounds
        log(f"  {src.width} x {src.height} пикс., {abs(src.transform.a):g} м, {crs.to_string()}")
    log("\n[1/3] Границы субъектов")
    bpath = regions_path or get_boundaries(root_dir)
    regions = load_regions(bpath, name_field, crs, bounds)
    log(f"  Пересекают карту: {len(regions)} — " + ", ".join(regions["name"]))
    log("\n[2/3] Площади классов по субъектам")
    acc, cnt = compute(raster, regions)
    log("\n[3/3] Таблицы")
    t_cover, km, pct, allc = build_tables(acc, cnt, regions)
    src_txt = ("geoBoundaries gbOpen RUS ADM1 (OpenStreetMap, ODbL, ~2017)"
               if regions_path is None else os.path.basename(regions_path))
    meta = [f"Карта: {os.path.basename(raster)}",
            f"Границы: {src_txt}",
            f"Лесные категории: {_rng(FOREST_CODES)}; узкий вариант лесистости: "
            f"{_rng(STRICT_CODES)}; нелесные: {_rng(NONFOREST_CODES)}",
            "Лесистость = лес / площадь субъекта в пределах карты; площади истинные "
            "(поправка на искажение проекции), пиксель — по центру",
            f"Лесистость по Рослесхозу на {OFFICIAL_DATE}: rosleshoz.gov.ru/opendata/"
            "7705598840-forestcover — покрытые лесной растительностью земли госреестра, "
            "весь субъект; по карте учитывается вся древесная растительность "
            "(в т.ч. зарастающие земли вне лесного фонда, городские леса)",
            f"04c_region_forest_stats.py {VERSION}, {_dt.datetime.now():%d.%m.%Y %H:%M}"]
    write_xlsx(out_xlsx, t_cover, km, pct, allc, meta)
    fc, sc = f"Лесистость {_rng(FOREST_CODES)}, %", f"Лесистость {_rng(STRICT_CODES)}, %"
    log(f"\n  {'Субъект':<30} {'охват, %':>9} {'лес, км²':>10} {fc[11:]:>12} {sc[11:]:>12}")
    for _, r in t_cover.iterrows():
        cov = "" if r["Охват картой, %"] is None or r["Охват картой, %"] != r["Охват картой, %"] \
            else f"{r['Охват картой, %']:.1f}"
        log(f"  {r['Субъект']:<30} {cov:>9} {r[f'Лес {_rng(FOREST_CODES)}, км²']:>10,.0f} "
            f"{r[fc]:>12.2f} {r[sc]:>12.2f}".replace(",", " "))
    log(f"\n  {out_xlsx}\nГотово за {time.time() - t0:.0f} с")
    _LOG.close()
    _LOG = None
    return out_xlsx


# ───────────────────────── запуск ─────────────────────────

def _in_notebook() -> bool:
    return "ipykernel" in sys.modules


def _initial_dir(root_dir: str) -> str:
    fb = os.path.join(root_dir, FINAL_BASE_NAME)
    if os.path.isdir(fb):
        subs = [os.path.join(fb, d) for d in os.listdir(fb) if os.path.isdir(os.path.join(fb, d))]
        return max(subs, key=os.path.getmtime) if subs else fb
    return root_dir


def _dialog(multi: bool, title: str, initialdir: str, types):
    try:
        import tkinter as tk
        from tkinter import filedialog
        win = tk.Tk()
        win.withdraw()
        win.attributes("-topmost", True)
        if multi:
            r = filedialog.askopenfilenames(parent=win, title=title, initialdir=initialdir,
                                            filetypes=types)
            r = [os.path.normpath(p) for p in r]
        else:
            r = filedialog.askopenfilename(parent=win, title=title, initialdir=initialdir,
                                           filetypes=types)
            r = os.path.normpath(r) if r else None
        win.destroy()
        return r
    except Exception as e:
        print(f"  Окно выбора недоступно ({e}).")
        raw = input("  Путь(и) через ';': ").strip()
        ps = [p.strip().strip('"') for p in raw.split(";") if p.strip()]
        return ps if multi else (ps[0] if ps else None)


def _run_interactive() -> None:
    root_dir = os.getcwd()
    print("=" * 72)
    print(f"04c_region_forest_stats.py {VERSION} — лесные категории и лесистость по субъектам")
    print("=" * 72)
    print(f"  лесные = {_rng(FOREST_CODES)} (= 100 %), узкий вариант лесистости "
          f"{_rng(STRICT_CODES)}, нелесные {_rng(NONFOREST_CODES)}")
    files = _dialog(True, "Карты *_cat.tif (можно несколько)", _initial_dir(root_dir),
                    [("Классифицированные растры", "*_cat.tif"), ("GeoTIFF", "*.tif *.tiff"),
                     ("Все файлы", "*.*")])
    if not files:
        print("Файлы не выбраны — выход.")
        return
    raw = input("\nГраницы: [1] субъекты РФ geoBoundaries (Enter)  [2] свой слой: ").strip()
    rpath, fld = None, None
    if raw == "2":
        rpath = _dialog(False, "Слой границ", root_dir,
                        [("Векторные слои", "*.gpkg *.shp *.geojson *.json"), ("Все файлы", "*.*")])
        if rpath:
            import pyogrio
            info = pyogrio.read_info(rpath)
            flds = list(info["fields"])
            for i, f in enumerate(flds, 1):
                print(f"   [{i}] {f}")
            k = input("  Поле с названием (номер, Enter = 1): ").strip()
            fld = flds[(int(k) if k else 1) - 1]
    _run_files(files, root_dir, rpath, fld)
    if not _in_notebook() and sys.stdin and sys.stdin.isatty():
        input("\nEnter — закрыть окно")


def _run_files(files, root_dir, rpath=None, fld=None) -> None:
    t0 = time.time()
    out = []
    for i, p in enumerate(files, 1):
        print(f"\n{'─' * 72}\nФайл {i}/{len(files)} ({100 * (i - 1) / len(files):.0f}% выполнено): "
              f"{os.path.basename(p)}")
        try:
            out.append(process(p, root_dir, rpath, fld))
        except Exception as e:
            log(f"ОШИБКА: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{'=' * 72}\nОбработано {len(out)}/{len(files)} (100%) за {time.time() - t0:.0f} с")
    for o in out:
        print(f"  {o}")


def main(argv=None) -> None:
    if argv is None:
        argv = sys.argv[1:]
        prog = os.path.basename(sys.argv[0]) if sys.argv else ""
        if "ipykernel" in prog.lower() or "kernel" in prog.lower() or _in_notebook():
            argv = [a for a in argv if not (a.startswith("-f") or a.endswith(".json"))]
    if not argv:
        _run_interactive()
        return
    ap = argparse.ArgumentParser(description="Лесные категории и лесистость по субъектам")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--regions", default=None, help="свой слой границ")
    ap.add_argument("--name-field", default=None)
    ap.add_argument("--root", default=os.getcwd())
    a = ap.parse_args(argv)
    files = []
    for p in a.inputs:
        files.extend(sorted(glob.glob(p)) or [p])
    _run_files(files, a.root, a.regions, a.name_field)


if __name__ == "__main__":
    main()
