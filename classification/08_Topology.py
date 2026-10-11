#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_Topology.py — топологическая коррекция векторной карты классов.

Полигоны слоя П (например, результат 07_generalization) выборочно заменяются
объектами целевых слоёв ЦС (например, водоёмы и болота из OSM).

Алгоритм
--------
1. Пользователь выбирает слой П, поле класса (числовое или текстовое) и классы
   для замены (например, 300 — вода, 400 — болота).
2. Каждому классу назначаются один или несколько слоёв ЦС (можно с фильтром
   по атрибуту, например natural = water). Порядок строк задаёт приоритет,
   если объекты ЦС разных классов перекрываются: выше в списке — главнее.
3. Для каждого класса K:
   3.1 Объекты K в П, пересекающиеся с ЦС(K), удаляются целиком; на их место
       встают объекты ЦС. Полигоны других классов, перекрытые новым объектом,
       обрезаются по его границе. Если после удаления остаётся пустота, она
       делится между соседними полигонами П по принципу ближайшего соседа
       (диаграмма Вороного по точкам общих границ): каждая точка пустоты
       достаётся тому соседу, чья граница ближе.
   3.2 Там, где K в П не было, а в ЦС он есть, объект ЦС переносится в П,
       другие классы обрезаются.
   Объекты ЦС обрезаются по охвату П (за пределы исходной карты не выходят).
   Новые объекты получают значение класса в выбранном поле П; остальные поля,
   постоянные для этого класса в П (например, class_name), заполняются тем же
   значением. Поле topo_src: "P" — объект П, "TS:<слой>" — объект ЦС.
4. Проверка и исправление топологии с допуском (м):
   • невалидные геометрии исправляются (make_valid);
   • перекрытия убираются (ЦС главнее П; между двумя объектами П перекрытие
     остаётся у большего);
   • дыры внутри охвата П заполняются соседями (тем же способом, что в 3.1);
   • узкие полигоны П (уже допуска) и мелкие (площадь меньше допуска в
     квадрате) присоединяются к соседу П с самой длинной общей границей;
   • итоговая проверка: невалидные, перекрытия, дыры, узкие полигоны.
   Объекты ЦС при исправлении не меняются, кроме снятия перекрытий между ними.

Если СК слоя П географическая (градусы), расчёт ведётся в UTM-зоне по центру
данных, результат возвращается в исходную СК.

Запуск из папки проекта:
    %run 08_Topology.py          # Jupyter — откроется окно
    python 08_Topology.py        # терминал — откроется окно

Выход: 08_TOPOLOGY/<ддммгг_ччмм><_vXXX источника>/
    <имя П>_topo.gpkg   — исправленный слой (имя слоя как у П)
    run_info.txt        — параметры и отчёт проверки
    слой topology_issues в том же GPKG — только по флажку

CLI:
    python 08_Topology.py P.gpkg --layer classes --field class_value \
        --target "300|water.gpkg|water_polygons|natural=water,reservoir" \
        --target "400|wetlands.shp" --tol 1.0
    Формат --target: КЛАСС|ФАЙЛ[|СЛОЙ][|ПОЛЕ=ЗНАЧ1,ЗНАЧ2]

Зависимости:
    pip install numpy pandas shapely>=2.0 geopandas pyogrio pyproj
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

OUT_BASE_NAME = "08_TOPOLOGY"
SRC_FIELD = "topo_src"
VECTOR_TYPES = [("Векторные слои", "*.gpkg *.shp *.geojson *.json *.fgb *.gml *.kml *.sqlite"),
                ("GeoPackage", "*.gpkg"), ("Shapefile", "*.shp"), ("Все файлы", "*.*")]


def log(msg=""):
    print(msg, flush=True)


# ===========================================================================
# 1. ГЕОМЕТРИЧЕСКИЕ УТИЛИТЫ
# ===========================================================================

def _polys(geom):
    """Только полигональная часть геометрии (Polygon/MultiPolygon) или None."""
    import shapely
    from shapely.geometry import MultiPolygon, Polygon
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = shapely.make_valid(geom)
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom if not geom.is_empty else None
    parts = []
    for g in getattr(geom, "geoms", []):
        p = _polys(g)
        if p is not None:
            parts.extend(p.geoms if isinstance(p, MultiPolygon) else [p])
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def _parts(geom):
    """Список отдельных полигонов."""
    from shapely.geometry import MultiPolygon
    g = _polys(geom)
    if g is None:
        return []
    return list(g.geoms) if isinstance(g, MultiPolygon) else [g]


GRID = 1e-3   # округление координат (м) при повторе операции после сбоя GEOS


def _union(geoms, coverage=False):
    """Объединение. coverage=True — полигоны не перекрываются (быстрый метод).
    При сбое GEOS — повтор с исправлением геометрий и округлением до GRID."""
    import shapely
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return None
    arr = np.array(geoms, dtype=object)
    if coverage:
        try:
            u = _polys(shapely.coverage_union_all(arr))
            tot = float(shapely.area(arr).sum())
            # быстрый метод верен только для точного разбиения — проверяем площадь
            if u is not None and u.is_valid and abs(u.area - tot) <= 1e-7 * max(tot, 1.0):
                return u
        except Exception:
            pass
    try:
        return _polys(shapely.union_all(arr))
    except shapely.errors.GEOSException:
        arr = shapely.make_valid(arr)
        return _polys(shapely.union_all(arr, grid_size=GRID))


def _inter(a, b):
    import shapely
    try:
        return _polys(shapely.intersection(a, b))
    except shapely.errors.GEOSException:
        return _polys(shapely.intersection(shapely.make_valid(a), shapely.make_valid(b),
                                           grid_size=GRID))


def _diff(a, b):
    import shapely
    try:
        return _polys(shapely.difference(a, b))
    except shapely.errors.GEOSException:
        return _polys(shapely.difference(shapely.make_valid(a), shapely.make_valid(b),
                                         grid_size=GRID))


def _clean_gdf(gdf):
    """make_valid + только полигоны + разбиение на части."""
    gdf = gdf.copy()
    gdf["geometry"] = [_polys(g) for g in gdf.geometry]
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    return gdf[~gdf.geometry.is_empty].reset_index(drop=True)


# ===========================================================================
# 2. ЗАПОЛНЕНИЕ ПУСТОТ ПО БЛИЖАЙШЕМУ СОСЕДУ
# ===========================================================================

def _voronoi_split(gap, pts, labels):
    """Делит полигон gap между метками по ближайшей точке границы
    (диаграмма Вороного). pts — точки на границе gap, labels — их метки."""
    import shapely
    from shapely.geometry import MultiPoint

    if len(set(labels.tolist())) == 1:
        return {labels[0]: gap}

    # Точки вдоль прямых границ лежат на одной линии; на таких вырожденных
    # данных GEOS строит перекрывающиеся ячейки. Сдвиг точек на доли мм
    # (детерминированный) убирает вырожденность; результат контролируется.
    rng = np.random.default_rng(0)
    for jit in (1e-4, 1e-3, 1e-2):
        p = pts + rng.uniform(-jit, jit, pts.shape)
        try:
            cells = list(shapely.voronoi_polygons(MultiPoint(p), extend_to=gap,
                                                  ordered=True).geoms)
            cell_labels = labels
        except TypeError:  # shapely < 2.1: нет ordered — сопоставляем через точки
            cells = list(shapely.voronoi_polygons(MultiPoint(p), extend_to=gap).geoms)
            tree = shapely.STRtree(shapely.points(p))
            cell_labels = []
            for c in cells:
                hit = tree.query(c, predicate="intersects")
                cell_labels.append(labels[hit[0]] if len(hit) else labels[0])
            cell_labels = np.array(cell_labels, dtype=object)
        arr = np.array(cells, dtype=object)
        tot = float(shapely.area(arr).sum())
        if abs(tot - shapely.union_all(arr).area) <= 1e-6 * max(tot, 1.0):
            break

    out = {}
    for lab in set(cell_labels.tolist()):
        u = _union([cells[i] for i in np.where(cell_labels == lab)[0]], coverage=True)
        if u is not None:
            piece = _inter(u, gap)
            if piece is not None:
                out[lab] = piece
    return out


def _label_boundary(gap, cand_geoms, touch, max_pts=20000):
    """Точки вдоль границы пустоты с меткой ближайшего соседа.
    Точки дальше touch от всех соседей (граница с ЦС или с краем) не берутся."""
    import shapely
    step = max(gap.boundary.length / 2000.0, touch / 5.0, 1e-3)
    coords = shapely.get_coordinates(shapely.segmentize(gap.boundary, step))
    if len(coords) > max_pts:
        coords = coords[np.linspace(0, len(coords) - 1, max_pts).astype(int)]
    coords = np.unique(np.round(coords, 6), axis=0)
    pts = shapely.points(coords)
    keys = list(cand_geoms.keys())
    dist = np.vstack([shapely.distance(pts, cand_geoms[k]) for k in keys])
    best = dist.argmin(axis=0)
    ok = dist[best, np.arange(len(coords))] <= touch
    labels = np.array([keys[b] for b in best[ok]], dtype=object)
    return coords[ok], labels


def _reattach_detached(split, geoms, touch, eps):
    """Части ячеек, оторванные от «своего» соседа (бывает у пустот сложной
    формы), отдаются соседу с самой длинной общей границей."""
    kept, detached = {}, []
    for i, piece in split.items():
        for part in _parts(piece):
            if part.distance(geoms[i]) <= touch:
                kept.setdefault(i, []).append(part)
            else:
                detached.append(part)
    for _ in range(3):              # оторванные куски могут цепляться друг за друга
        rest = []
        for part in detached:
            zone = part.buffer(eps)
            best, best_len = None, eps
            for i in split:
                ln = sum(part.boundary.intersection(g.buffer(eps)).length
                         for g in [geoms[i]] + kept.get(i, []) if g.intersects(zone))
                if ln > best_len:
                    best, best_len = i, ln
            if best is None:
                rest.append(part)
            else:
                kept.setdefault(best, []).append(part)
        if not rest or len(rest) == len(detached):
            detached = rest
            break
        detached = rest
    for part in detached:           # совсем изолированные — ближайшему соседу
        best = min(split, key=lambda i: part.distance(geoms[i]))
        kept.setdefault(best, []).append(part)
    return kept


def fill_gaps(gdf, gaps, eps, touch=None, prefer_p=True, protect_ts=True, min_area=None):
    """Распределяет пустоты между соседями по ближайшему соседу. Меняет gdf
    на месте. touch — расстояние, на котором сосед считается примыкающим.
    Возвращает (заполнено шт., площадь, список незаполненных пустот)."""
    import shapely

    if not gaps:
        return 0, 0.0, []
    touch = max(touch if touch is not None else eps, eps)
    min_area = eps * eps if min_area is None else min_area
    geoms = list(gdf.geometry.values)
    src = gdf[SRC_FIELD].values
    tree = shapely.STRtree(np.array(geoms, dtype=object))
    pieces = {}
    n_done, area_done, unfilled = 0, 0.0, []

    for gap in gaps:
        if gap is None or gap.is_empty or gap.area <= min_area:
            continue
        cand = tree.query(gap.buffer(touch), predicate="intersects").tolist()
        cand_p = [i for i in cand if src[i] == "P"]
        use = cand_p if (prefer_p and cand_p) else cand
        if protect_ts and not cand_p:
            use = []   # пустоту, окружённую только ЦС, ЦС не отдаём
        if not use:
            unfilled.append(gap)
            continue
        pts, labels = _label_boundary(gap, {i: geoms[i] for i in use}, touch)
        if len(labels) == 0:
            unfilled.append(gap)
            continue
        split = _voronoi_split(gap, pts, labels)
        for i, plist in _reattach_detached(split, geoms, touch, eps).items():
            pieces.setdefault(i, []).extend(plist)
        n_done += 1
        area_done += gap.area

    for i, plist in pieces.items():
        u = _union([geoms[i]] + plist)
        if u is not None:
            geoms[i] = u
    gdf["geometry"] = geoms
    return n_done, area_done, unfilled


# ===========================================================================
# 3. ПРОВЕРКА И ИСПРАВЛЕНИЕ ТОПОЛОГИИ
# ===========================================================================

def _overlap_pairs(geoms):
    """Пары индексов (i<j), у которых пересекаются внутренности."""
    import shapely
    arr = np.array(geoms, dtype=object)
    tree = shapely.STRtree(arr)
    a, b = tree.query(arr, predicate="intersects")
    m = a < b
    a, b = a[m], b[m]
    if len(a) == 0:
        return a, b
    inter = shapely.relate_pattern(arr[a], arr[b], "T********")
    return a[inter], b[inter]


def _remove_overlaps(gdf, eps):
    geoms = list(gdf.geometry.values)
    src = gdf[SRC_FIELD].values
    a, b = _overlap_pairs(geoms)
    n, area, issues = 0, 0.0, []
    for i, j in zip(a.tolist(), b.tolist()):
        gi, gj = geoms[i], geoms[j]
        if gi is None or gj is None:
            continue
        inter = _inter(gi, gj)
        if inter is None or inter.area <= eps * eps:
            continue
        # кто уступает: П уступает ЦС; среди равных — меньший уступает большему
        pi, pj = src[i] != "P", src[j] != "P"
        if pi != pj:
            loser = j if pi else i
        else:
            loser = i if gi.area < gj.area else j
        geoms[loser] = _diff(geoms[loser], inter)
        n += 1
        area += inter.area
        issues.append(("overlap_removed", inter))
    gdf["geometry"] = geoms
    gdf = gdf[gdf.geometry.notna()]
    return _clean_gdf(gdf), n, area, issues


def _is_sliver(g, tol):
    if tol <= 0:
        return False
    return g.area < tol * tol or g.buffer(-tol / 2.0).is_empty


def _merge_slivers(gdf, tol, eps):
    """Узкие (уже допуска) и мелкие (площадь < допуска²) полигоны П убираются,
    а их место делится между соседями П по ближайшему соседу — так же, как
    пустоты. Смежные узкие фрагменты обрабатываются вместе. Объекты ЦС
    не трогаются."""
    if tol <= 0:
        return gdf, 0, 0.0, []
    src = gdf[SRC_FIELD].values
    mask = np.array([src[i] == "P" and _is_sliver(g, tol)
                     for i, g in enumerate(gdf.geometry.values)], dtype=bool)
    if not mask.any():
        return gdf, 0, 0.0, []
    if mask.all():
        return gdf, 0, 0.0, []
    # крошки (периметр меньше допуска) — целиком соседу с длиннейшей общей границей
    import shapely
    geoms_all = list(gdf.geometry.values)
    tree_all = shapely.STRtree(np.array(geoms_all, dtype=object))
    micro_done = 0
    micro_area = 0.0
    for i in np.where(mask)[0]:
        g = geoms_all[i]
        if g is None or g.length >= tol:
            continue
        best, best_len = None, -1.0
        for k in tree_all.query(g.buffer(eps), predicate="intersects"):
            if k == i or mask[k] or src[k] != "P" or geoms_all[k] is None:
                continue
            ln = g.boundary.intersection(geoms_all[k].buffer(eps)).length
            if ln > best_len:
                best, best_len = k, ln
        if best is not None:
            geoms_all[best] = _union([geoms_all[best], g])
            geoms_all[i] = None
            mask[i] = False
            micro_done += 1
            micro_area += g.area
    gdf = gdf.copy()
    gdf["geometry"] = geoms_all
    gone = np.array([g is None for g in geoms_all])
    gdf = gdf[~gone].reset_index(drop=True)
    mask = mask[~gone]
    if not mask.any():
        return _clean_gdf(gdf), micro_done, micro_area, []
    sl = list(gdf.geometry.values[mask])
    rest = gdf[~mask].reset_index(drop=True)
    gaps = _parts(_union(sl))
    n_f, a_f, unfilled = fill_gaps(rest, gaps, eps, touch=max(tol / 2.0, eps), min_area=0.0)
    n_back = 0
    if unfilled:
        # без соседей П (например, внутри ЦС) — возвращаем как было
        back = gdf[mask].copy()
        keep_back = [g for g in back.geometry
                     if any(g.intersects(u) for u in unfilled)]
        back = back[back.geometry.isin(keep_back)] if keep_back else back.iloc[0:0]
        n_back = len(back)
        rest = gpd_concat(rest, back)
    issues = [("sliver_merged", g) for g in sl]
    return (_clean_gdf(rest), int(mask.sum()) - n_back + micro_done,
            float(sum(g.area for g in sl)) - float(sum(g.area for g in unfilled)) + micro_area,
            issues)


def gpd_concat(a, b):
    import pandas as pd
    import geopandas as gpd
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return gpd.GeoDataFrame(pd.concat([a, b], ignore_index=True),
                                geometry="geometry", crs=a.crs)


def _close_small_holes(coverage, tol):
    """Охват П без внутренних дыр уже допуска (они считаются ошибками).
    Возвращает (охват, число оставленных дыр больше допуска)."""
    from shapely.geometry import Polygon
    if coverage is None or tol <= 0:
        return coverage, 0
    fill, big = [], 0
    for poly in _parts(coverage):
        for ring in poly.interiors:
            h = Polygon(ring)
            if _is_sliver(h, tol):
                fill.append(h)
            else:
                big += 1
    if fill:
        coverage = _union([coverage] + fill)
    return coverage, big


def _find_gaps(gdf, coverage, eps):
    if coverage is None:
        return []
    u = _union(list(gdf.geometry.values))
    if u is None:
        return _parts(coverage)
    return [g for g in _parts(_diff(coverage, u)) if g.area > eps * eps]


def check_topology(gdf, coverage, tol, eps):
    """Итоговая проверка без исправления. Возвращает словарь."""
    geoms = list(gdf.geometry.values)
    invalid = int(sum(1 for g in geoms if not g.is_valid))
    a, b = _overlap_pairs(geoms)
    ov_area, ov_n = 0.0, 0
    for i, j in zip(a.tolist(), b.tolist()):
        inter = _inter(geoms[i], geoms[j])
        if inter is not None and inter.area > eps * eps:
            ov_area += inter.area
            ov_n += 1
    gaps = _find_gaps(gdf, coverage, eps)
    src = gdf[SRC_FIELD].values
    slivers = int(sum(1 for k, g in enumerate(geoms)
                      if src[k] == "P" and _is_sliver(g, tol)))
    return {
        "invalid": invalid,
        "overlap_pairs": ov_n,
        "overlap_area": ov_area,
        "gaps": len(gaps),
        "gap_area": float(sum(g.area for g in gaps)),
        "slivers": slivers,
    }


def fix_topology(gdf, coverage, tol, eps, max_iter=3):
    """Исправление: валидность, перекрытия, дыры, узкие полигоны."""
    rep = {"invalid_fixed": 0, "overlaps_removed": 0, "overlap_area": 0.0,
           "gaps_filled": 0, "gap_area": 0.0, "slivers_merged": 0,
           "sliver_area": 0.0, "unfilled": []}
    issues = []
    rep["invalid_fixed"] = int(sum(1 for g in gdf.geometry if not g.is_valid))
    gdf = _clean_gdf(gdf)

    for it in range(1, max_iter + 1):
        gdf, n_ov, a_ov, iss = _remove_overlaps(gdf, eps)
        issues += iss
        gaps = _find_gaps(gdf, coverage, eps)
        n_g, a_g, unfilled = fill_gaps(gdf, gaps, eps, touch=max(tol / 2.0, eps))
        gdf = _clean_gdf(gdf)
        uf = {id(g) for g in unfilled}
        issues += [("gap_filled", g) for g in gaps if id(g) not in uf]
        gdf, n_s, a_s, iss = _merge_slivers(gdf, tol, eps)
        issues += iss
        rep["overlaps_removed"] += n_ov
        rep["overlap_area"] += a_ov
        rep["gaps_filled"] += n_g
        rep["gap_area"] += a_g
        rep["slivers_merged"] += n_s
        rep["sliver_area"] += a_s
        rep["unfilled"] = unfilled
        log(f"  [проверка {it}] перекрытий снято {n_ov}, дыр заполнено {n_g}, "
            f"узких полигонов присоединено {n_s}")
        if n_ov == 0 and n_g == 0 and n_s == 0:
            break
    issues += [("gap_unfilled", g) for g in rep["unfilled"]]
    return gdf, rep, issues


# ===========================================================================
# 4. ОСНОВНАЯ КОРРЕКЦИЯ
# ===========================================================================

def _class_constants(P, field, value):
    """Поля, постоянные для класса value в П (например, class_name)."""
    sub = P[P[field] == value]
    out = {}
    if len(sub) == 0:
        return out
    for c in P.columns:
        if c in (field, "geometry", SRC_FIELD) or c.lower() in ("area", "shape_area",
                                                                  "shape_leng", "perimeter",
                                                                  "fid", "id", "objectid"):
            continue
        vals = sub[c].dropna().unique()
        if len(vals) == 1:
            out[c] = vals[0]
    return out


def topology_correct(P, field, targets, tol=1.0, drop_unmatched=False):
    """
    P       — GeoDataFrame слоя П.
    field   — поле класса.
    targets — список словарей {"value": значение класса, "gdf": GeoDataFrame ЦС,
              "name": подпись}. Порядок = приоритет.
    Возвращает (результат, отчёт, слой проблем).
    """
    import geopandas as gpd

    t0 = time.time()
    if P.crs is None:
        raise ValueError("У слоя П не задана система координат.")
    src_crs = P.crs
    work_crs = src_crs
    if src_crs.is_geographic:
        work_crs = P.estimate_utm_crs()
        log(f"  СК П географическая ({src_crs.to_string()}); расчёт в {work_crs.to_string()}")
        P = P.to_crs(work_crs)

    eps = max(tol * 0.01, 1e-6)
    log("  Подготовка слоя П (исправление геометрий)...")
    P = _clean_gdf(P)
    P[SRC_FIELD] = "P"
    n_p0 = len(P)
    coverage = _union(list(P.geometry.values))
    if coverage is None:
        raise ValueError("Слой П пуст.")
    coverage, big_holes = _close_small_holes(coverage, tol)
    log(f"  П: объектов {n_p0}, площадь охвата {coverage.area / 1e6:.3f} км²"
        + (f"; внутренних дыр больше допуска (оставлены как есть): {big_holes}" if big_holes else ""))

    # --- ЦС по классам с учётом приоритета ---
    taken = None
    t_by_class = []          # (value, name, geometry)
    for t in targets:
        ts = t["gdf"]
        if ts.crs is None:
            log(f"  ! У слоя ЦС «{t['name']}» нет СК — считаю, что она как у П")
            ts = ts.set_crs(work_crs, allow_override=True)
        ts = ts.to_crs(work_crs)
        T = _union([_polys(g) for g in ts.geometry])
        if T is None:
            log(f"  ! ЦС «{t['name']}»: нет полигонов")
            continue
        T = _inter(T, coverage)
        if T is not None and taken is not None:
            T = _diff(T, taken)
        if T is None:
            log(f"  ! ЦС «{t['name']}»: нет объектов в охвате П")
            continue
        taken = T if taken is None else _union([taken, T])
        t_by_class.append((t["value"], t["name"], T))
        log(f"  ЦС «{t['name']}» → класс {t['value']}: "
            f"площадь {T.area / 1e6:.3f} км², частей {len(_parts(T))}")

    if not t_by_class:
        raise ValueError("Ни один слой ЦС не дал объектов в охвате П.")
    T_all = taken

    # --- какие объекты П удаляются ---
    import shapely
    geoms = np.array(list(P.geometry.values), dtype=object)
    tree = shapely.STRtree(geoms)
    remove = np.zeros(len(P), dtype=bool)
    for value in {v for v, _, _ in t_by_class}:
        is_cls = (P[field] == value).values
        T_v = _union([T for v, _, T in t_by_class if v == value])
        if drop_unmatched:
            remove |= is_cls
        else:
            hit = tree.query(T_v, predicate="intersects")
            hit = [i for i in hit if is_cls[i]
                   and shapely.relate_pattern(geoms[i], T_v, "T********")]
            remove[hit] = True
        log(f"  Класс {value}: в П объектов {int(is_cls.sum())}, "
            f"удаляется {int((remove & is_cls).sum())}")

    removed_union = _union(list(geoms[remove]))
    constants = {v: _class_constants(P, field, v) for v, _, _ in t_by_class}

    # --- обрезка оставшихся по ЦС ---
    keep = P[~remove].copy()
    kg = list(keep.geometry.values)
    hit = set(shapely.STRtree(np.array(kg, dtype=object))
              .query(T_all, predicate="intersects").tolist()) if kg else set()
    n_cut = 0
    for i in hit:
        if shapely.relate_pattern(kg[i], T_all, "T********"):
            kg[i] = _diff(kg[i], T_all)
            n_cut += 1
    keep["geometry"] = kg
    keep = keep[keep.geometry.notna()]
    log(f"  Обрезано по границам ЦС объектов П: {n_cut}")

    # --- новые объекты ЦС ---
    rows = []
    for value, name, T in t_by_class:
        for part in _parts(T):
            r = {c: None for c in P.columns if c != "geometry"}
            r.update(constants.get(value, {}))
            r[field] = value
            r[SRC_FIELD] = f"TS:{name}"
            r["geometry"] = part
            rows.append(r)
    new = gpd.GeoDataFrame(rows, geometry="geometry", crs=work_crs)
    log(f"  Перенесено объектов ЦС: {len(new)}")

    import pandas as pd
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        res = gpd.GeoDataFrame(pd.concat([keep, new], ignore_index=True),
                               geometry="geometry", crs=work_crs)
    res = _clean_gdf(res)
    for c in P.columns:          # возвращаем исходные типы полей, где возможно
        if c in res.columns and c != "geometry":
            try:
                res[c] = res[c].astype(P[c].dtype)
            except (TypeError, ValueError):
                pass

    # --- пустоты на месте удалённых объектов ---
    voids = []
    if removed_union is not None:
        voids = [g for g in _parts(_diff(removed_union, T_all)) if g.area > eps * eps]
    log(f"  Пустот после удаления: {len(voids)}, "
        f"площадь {sum(g.area for g in voids) / 1e4:.2f} га")
    n_v, a_v, unf = fill_gaps(res, voids, eps, touch=max(tol / 2.0, eps))
    res = _clean_gdf(res)
    log(f"  Пустот заполнено соседями П: {n_v} ({a_v / 1e4:.2f} га); не заполнено: {len(unf)}")
    uf = {id(g) for g in unf}
    issues = [("void_filled", g) for g in voids if id(g) not in uf]

    # --- топология ---
    log("\n  Проверка и исправление топологии...")
    res, rep, iss = fix_topology(res, coverage, tol, eps)
    issues += iss
    final = check_topology(res, coverage, tol, eps)

    for c in res.columns:        # пересчёт площадей, если такие поля есть
        if c.lower() in ("area", "shape_area"):
            res[c] = res.geometry.area
    if src_crs != work_crs:
        res = res.to_crs(src_crs)

    report = {
        "p_in": n_p0, "p_removed": int(remove.sum()), "p_cut": n_cut,
        "ts_added": len(new), "voids": len(voids), "voids_filled": n_v,
        "void_area": a_v, "fix": rep, "final": final, "out": len(res),
        "coverage_area": coverage.area, "big_holes": big_holes,
        "out_area": float(res.to_crs(work_crs).area.sum()) if src_crs != work_crs
        else float(res.area.sum()),
        "time_min": (time.time() - t0) / 60.0,
    }
    iss_gdf = None
    if issues:
        iss_gdf = gpd.GeoDataFrame({"issue": [k for k, _ in issues],
                                    "area": [g.area for _, g in issues]},
                                   geometry=[g for _, g in issues], crs=work_crs)
        if src_crs != work_crs:
            iss_gdf = iss_gdf.to_crs(src_crs)
    return res, report, iss_gdf


# ===========================================================================
# 5. ЧТЕНИЕ ДАННЫХ
# ===========================================================================

def list_layers(path):
    try:
        import geopandas as gpd
        df = gpd.list_layers(path)
        return [str(x) for x in df["name"]]
    except Exception:
        pass
    try:
        import pyogrio
        return [str(x[0]) for x in pyogrio.list_layers(path)]
    except Exception:
        pass
    try:
        import fiona
        return list(fiona.listlayers(path))
    except Exception:
        return [os.path.splitext(os.path.basename(path))[0]]


def read_layer(path, layer=None, attrs_only=False):
    import geopandas as gpd
    kw = {}
    if layer and path.lower().endswith((".gpkg", ".sqlite", ".gdb", ".gml", ".kml")):
        kw["layer"] = layer
    if attrs_only:
        kw["ignore_geometry"] = True
    return gpd.read_file(path, **kw)


def apply_filter(gdf, ffield, fvalues):
    """Фильтр ЦС: поле in {значения}; сравнение как строк."""
    if not ffield or not fvalues:
        return gdf
    if ffield not in gdf.columns:
        raise ValueError(f"В слое ЦС нет поля {ffield}")
    s = {str(v).strip() for v in fvalues if str(v).strip()}
    return gdf[gdf[ffield].astype(str).str.strip().isin(s)]


def coerce_value(text, series):
    """Строку класса привести к типу поля П."""
    import pandas as pd
    if pd.api.types.is_integer_dtype(series):
        return int(float(text))
    if pd.api.types.is_float_dtype(series):
        f = float(text)
        return f
    return str(text)


def _fmt(v):
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


# ===========================================================================
# 6. ЗАПУСК: ЧТЕНИЕ → КОРРЕКЦИЯ → ЗАПИСЬ
# ===========================================================================

def run_job(cfg):
    """cfg: p_path, p_layer, field, targets[{value, path, layer, ffield, fvalues}],
    tol, drop_unmatched, write_issues, out_path."""
    line = "=" * 60
    log(f"\n{line}\n  08 - ТОПОЛОГИЧЕСКАЯ КОРРЕКЦИЯ\n{line}")
    log(f"  П     : {cfg['p_path']}" + (f" (слой {cfg['p_layer']})" if cfg.get("p_layer") else ""))
    log(f"  Поле  : {cfg['field']}")
    log(f"  Допуск: {cfg['tol']} м")

    P = read_layer(cfg["p_path"], cfg.get("p_layer"))
    if cfg["field"] not in P.columns:
        raise ValueError(f"В слое П нет поля {cfg['field']}")

    targets = []
    for t in cfg["targets"]:
        ts = read_layer(t["path"], t.get("layer"))
        n0 = len(ts)
        ts = apply_filter(ts, t.get("ffield"), t.get("fvalues"))
        name = t.get("layer") or os.path.splitext(os.path.basename(t["path"]))[0]
        flt = f", фильтр {t['ffield']} = {','.join(t['fvalues'])}" if t.get("ffield") and t.get("fvalues") else ""
        log(f"  ЦС    : класс {_fmt(t['value'])} ← {t['path']} ({name}{flt}): "
            f"объектов {len(ts)} из {n0}")
        targets.append({"value": t["value"], "gdf": ts, "name": name})

    log("")
    res, rep, issues = topology_correct(P, cfg["field"], targets, tol=cfg["tol"],
                                        drop_unmatched=cfg.get("drop_unmatched", False))

    out_path = cfg["out_path"]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    layer_name = cfg.get("p_layer") or os.path.splitext(os.path.basename(cfg["p_path"]))[0]
    if os.path.isfile(out_path):
        os.remove(out_path)
    res.to_file(out_path, layer=layer_name, driver="GPKG")
    if cfg.get("write_issues") and issues is not None and len(issues):
        issues.to_file(out_path, layer="topology_issues", driver="GPKG")

    f, fx = rep["final"], rep["fix"]
    lines = [
        "08 - Topology (топологическая коррекция)",
        f"Время запуска : {datetime.now().strftime('%d.%m.%Y %H:%M')}",
        f"Слой П        : {cfg['p_path']} | {cfg.get('p_layer') or ''}",
        f"Поле класса   : {cfg['field']}",
        f"Допуск, м     : {cfg['tol']}",
        f"Удалять непересекающиеся объекты классов: {cfg.get('drop_unmatched', False)}",
        "Слои ЦС (порядок = приоритет):",
    ] + [f"  {_fmt(t['value'])} <- {t['path']} | {t.get('layer') or ''}"
         + (f" | {t['ffield']}={','.join(t['fvalues'])}" if t.get('ffield') and t.get('fvalues') else "")
         for t in cfg["targets"]] + [
        "",
        "Результат:",
        f"  объектов П на входе           : {rep['p_in']}",
        f"  удалено объектов П            : {rep['p_removed']}",
        f"  обрезано по границам ЦС       : {rep['p_cut']}",
        f"  перенесено объектов ЦС        : {rep['ts_added']}",
        f"  пустот заполнено соседями     : {rep['voids_filled']} из {rep['voids']} "
        f"({rep['void_area'] / 1e4:.2f} га)",
        f"  объектов на выходе            : {rep['out']}",
        f"  дыр в охвате П больше допуска : {rep['big_holes']} (оставлены; меньше допуска — заполнены)",
        f"  площадь охвата П / результата : {rep['coverage_area'] / 1e6:.4f} / "
        f"{rep['out_area'] / 1e6:.4f} км²",
        "",
        "Исправление топологии:",
        f"  невалидных геометрий          : {fx['invalid_fixed']}",
        f"  снято перекрытий              : {fx['overlaps_removed']} ({fx['overlap_area']:.2f} м²)",
        f"  заполнено дыр                 : {fx['gaps_filled']} ({fx['gap_area']:.2f} м²)",
        f"  присоединено узких полигонов  : {fx['slivers_merged']} ({fx['sliver_area']:.2f} м²)",
        "",
        "Итоговая проверка:",
        f"  невалидных                    : {f['invalid']}",
        f"  перекрытий                    : {f['overlap_pairs']} ({f['overlap_area']:.4f} м²)",
        f"  дыр                           : {f['gaps']} ({f['gap_area']:.4f} м²)",
        f"  узких полигонов П (< допуска) : {f['slivers']}",
        f"Время работы  : {rep['time_min']:.1f} мин",
        f"Выход         : {out_path} (слой {layer_name})",
    ]
    with open(os.path.join(os.path.dirname(os.path.abspath(out_path)), "run_info.txt"),
              "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    log(f"\n{line}")
    for s in lines[lines.index("Результат:"):]:
        log("  " + s)
    log(line)
    ok = f["invalid"] == 0 and f["overlap_pairs"] == 0 and f["gaps"] == 0
    log("  Топология в порядке." if ok else
        "  ! Остались ошибки топологии — см. run_info.txt"
        + (" и слой topology_issues" if cfg.get("write_issues") else ""))
    return out_path


def default_out_path(p_path, p_layer=None, root_dir=None):
    root_dir = root_dir or os.getcwd()
    stamp = datetime.now().strftime("%d%m%y_%H%M")
    m = re.search(r"(_v\d{3})$", os.path.basename(os.path.dirname(os.path.abspath(p_path))))
    folder = os.path.join(root_dir, OUT_BASE_NAME, stamp + (m.group(1) if m else ""))
    base = os.path.splitext(os.path.basename(p_path))[0]
    return os.path.join(folder, f"{base}_topo.gpkg")


# ===========================================================================
# 7. ОКНО (tkinter)
# ===========================================================================

def _initial_dir(root_dir):
    for name in ("07_GENERALIZATION", "06_SIEVE_MARCHING_SQUARES"):
        b = os.path.join(root_dir, name)
        if os.path.isdir(b):
            subs = [os.path.join(b, d) for d in os.listdir(b) if os.path.isdir(os.path.join(b, d))]
            return max(subs, key=os.path.getmtime) if subs else b
    return root_dir


def _gui_dialog(root_dir):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    state = {"attrs": None, "values": {}, "last_dir": _initial_dir(root_dir)}
    targets = []         # список словарей, как в cfg["targets"]
    result = {}

    win = tk.Tk()
    win.title("08 — Топологическая коррекция")
    win.minsize(780, 0)
    win.attributes("-topmost", True)
    win.after(500, lambda: win.attributes("-topmost", False))
    pad = {"padx": 8, "pady": 4}

    # ---------- слой П ----------
    fr_p = ttk.LabelFrame(win, text="1. Слой П (полигоны классов)")
    fr_p.pack(fill="x", **pad)
    v_p = tk.StringVar()
    v_pl = tk.StringVar()
    v_fld = tk.StringVar()
    ttk.Label(fr_p, text="Файл").grid(row=0, column=0, sticky="w", padx=6, pady=3)
    ttk.Entry(fr_p, textvariable=v_p, width=70).grid(row=0, column=1, sticky="we", padx=6)
    ttk.Label(fr_p, text="Слой").grid(row=1, column=0, sticky="w", padx=6, pady=3)
    cb_pl = ttk.Combobox(fr_p, textvariable=v_pl, state="readonly", width=40)
    cb_pl.grid(row=1, column=1, sticky="w", padx=6)
    ttk.Label(fr_p, text="Поле класса").grid(row=2, column=0, sticky="w", padx=6, pady=3)
    cb_fld = ttk.Combobox(fr_p, textvariable=v_fld, state="readonly", width=40)
    cb_fld.grid(row=2, column=1, sticky="w", padx=6)
    fr_p.columnconfigure(1, weight=1)

    ttk.Label(fr_p, text="Классы для замены\n(Ctrl/Shift — несколько)").grid(
        row=3, column=0, sticky="nw", padx=6, pady=3)
    fr_cls = ttk.Frame(fr_p)
    fr_cls.grid(row=3, column=1, columnspan=2, sticky="we", padx=6, pady=3)
    lb_cls = tk.Listbox(fr_cls, selectmode="extended", height=6, exportselection=False)
    sb_cls = ttk.Scrollbar(fr_cls, orient="vertical", command=lb_cls.yview)
    lb_cls.configure(yscrollcommand=sb_cls.set)
    lb_cls.pack(side="left", fill="both", expand=True)
    sb_cls.pack(side="right", fill="y")
    cls_keys = []        # исходные значения в порядке списка

    def load_classes(*_):
        lb_cls.delete(0, "end")
        cls_keys.clear()
        df, fld = state["attrs"], v_fld.get()
        if df is None or not fld:
            return
        name_col = next((c for c in df.columns if c != fld and c.lower() in
                         ("class_name", "name", "название", "label", "class")), None)
        counts = df[fld].value_counts(dropna=True)
        try:
            keys = sorted(counts.index.tolist())
        except TypeError:
            keys = sorted(counts.index.tolist(), key=str)
        for k in keys:
            label = _fmt(k)
            if name_col is not None:
                names = df.loc[df[fld] == k, name_col].dropna().unique()
                if len(names) == 1 and str(names[0]) != label:
                    label += f" — {names[0]}"
            lb_cls.insert("end", f"{label}   ({counts[k]} шт.)")
            cls_keys.append(k)

    def load_layer(*_):
        try:
            df = read_layer(v_p.get(), v_pl.get(), attrs_only=True)
        except Exception as e:
            messagebox.showerror("Ошибка чтения", str(e), parent=win)
            return
        state["attrs"] = df
        fields = [c for c in df.columns]
        cb_fld["values"] = fields
        guess = next((c for c in fields if c.lower() in ("class_value", "class", "code", "gridcode")), fields[0] if fields else "")
        v_fld.set(guess)
        load_classes()

    def browse_p():
        p = filedialog.askopenfilename(parent=win, title="Слой П", initialdir=state["last_dir"],
                                       filetypes=VECTOR_TYPES)
        if not p:
            return
        p = os.path.normpath(p)
        state["last_dir"] = os.path.dirname(p)
        v_p.set(p)
        layers = list_layers(p)
        cb_pl["values"] = layers
        v_pl.set(layers[0] if layers else "")
        load_layer()
        v_out.set(default_out_path(p, v_pl.get(), root_dir))

    ttk.Button(fr_p, text="Обзор…", command=browse_p).grid(row=0, column=2, padx=6)
    cb_pl.bind("<<ComboboxSelected>>", load_layer)
    cb_fld.bind("<<ComboboxSelected>>", load_classes)

    def selected_classes():
        return [cls_keys[i] for i in lb_cls.curselection()]

    # ---------- слои ЦС ----------
    fr_t = ttk.LabelFrame(win, text="2. Слои ЦС для выбранных классов (порядок = приоритет)")
    fr_t.pack(fill="both", expand=True, **pad)
    cols = ("cls", "file", "layer", "flt")
    tv = ttk.Treeview(fr_t, columns=cols, show="headings", height=6)
    for c, h, w in zip(cols, ("Класс П", "Файл ЦС", "Слой", "Фильтр"), (90, 360, 130, 180)):
        tv.heading(c, text=h)
        tv.column(c, width=w, anchor="w")
    tv.pack(fill="both", expand=True, padx=6, pady=6)

    def refresh_tv():
        tv.delete(*tv.get_children())
        for i, t in enumerate(targets):
            flt = f"{t['ffield']} = {', '.join(t['fvalues'])}" if t.get("ffield") and t.get("fvalues") else "—"
            tv.insert("", "end", iid=str(i),
                      values=(_fmt(t["value"]), t["path"], t.get("layer") or "", flt))

    def add_target():
        if state["attrs"] is None:
            messagebox.showwarning("Нет слоя П", "Сначала выберите слой П и поле класса.", parent=win)
            return
        sel = selected_classes() or list(cls_keys)
        dlg = tk.Toplevel(win)
        dlg.title("Слой ЦС")
        dlg.transient(win)
        dlg.grab_set()
        d_cls, d_path, d_lay = tk.StringVar(value=_fmt(sel[0]) if sel else ""), tk.StringVar(), tk.StringVar()
        d_ff, d_fv = tk.StringVar(), tk.StringVar()
        cache = {"df": None}

        ttk.Label(dlg, text="Класс П (можно ввести новый)").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Combobox(dlg, textvariable=d_cls, values=[_fmt(k) for k in sel], width=20).grid(
            row=0, column=1, sticky="w", padx=8)
        ttk.Label(dlg, text="Файлы ЦС").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(dlg, textvariable=d_path, width=60).grid(row=1, column=1, sticky="we", padx=8)
        ttk.Label(dlg, text="Слой").grid(row=2, column=0, sticky="w", padx=8, pady=4)
        cb_l = ttk.Combobox(dlg, textvariable=d_lay, state="readonly", width=40)
        cb_l.grid(row=2, column=1, sticky="w", padx=8)
        ttk.Label(dlg, text="Фильтр: поле (необязательно)").grid(row=3, column=0, sticky="w", padx=8, pady=4)
        cb_ff = ttk.Combobox(dlg, textvariable=d_ff, width=30)
        cb_ff.grid(row=3, column=1, sticky="w", padx=8)
        ttk.Label(dlg, text="Фильтр: значения через запятую").grid(row=4, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(dlg, textvariable=d_fv, width=40).grid(row=4, column=1, sticky="w", padx=8)
        lb_hint = ttk.Label(dlg, text="", foreground="#555")
        lb_hint.grid(row=5, column=0, columnspan=3, sticky="w", padx=8)
        files = []

        def load_ts_fields(*_):
            if not files:
                return
            try:
                df = read_layer(files[0], d_lay.get(), attrs_only=True)
            except Exception as e:
                lb_hint.config(text=f"Ошибка чтения: {e}")
                return
            cache["df"] = df
            cb_ff["values"] = [""] + list(df.columns)
            lb_hint.config(text=f"Объектов в слое: {len(df)}")

        def show_values(*_):
            df, ff = cache["df"], d_ff.get()
            if df is not None and ff in df.columns:
                vals = df[ff].dropna().astype(str).value_counts().head(15)
                lb_hint.config(text="Частые значения: " + ", ".join(f"{k} ({n})" for k, n in vals.items()))

        def browse_ts():
            ps = filedialog.askopenfilenames(parent=dlg, title="Слои ЦС", initialdir=state["last_dir"],
                                             filetypes=VECTOR_TYPES)
            if not ps:
                return
            files.clear()
            files.extend(os.path.normpath(p) for p in ps)
            state["last_dir"] = os.path.dirname(files[0])
            d_path.set("; ".join(files))
            layers = list_layers(files[0])
            cb_l["values"] = layers
            d_lay.set(layers[0] if layers else "")
            load_ts_fields()

        ttk.Button(dlg, text="Обзор…", command=browse_ts).grid(row=1, column=2, padx=8)
        cb_l.bind("<<ComboboxSelected>>", load_ts_fields)
        cb_ff.bind("<<ComboboxSelected>>", show_values)

        def ok():
            if not files or not d_cls.get().strip():
                messagebox.showwarning("Не заполнено", "Укажите класс и файл ЦС.", parent=dlg)
                return
            txt = d_cls.get().strip()
            key = next((k for k in cls_keys if _fmt(k) == txt), None)
            if key is None:
                try:
                    key = coerce_value(txt, state["attrs"][v_fld.get()])
                except ValueError:
                    messagebox.showerror("Ошибка", f"Значение «{txt}» не подходит к типу поля.", parent=dlg)
                    return
            fvals = [s.strip() for s in d_fv.get().split(",") if s.strip()]
            for i, p in enumerate(files):
                # у второго и следующих файлов слой берётся первый в файле
                lay = d_lay.get() if i == 0 else (list_layers(p) or [""])[0]
                targets.append({"value": key, "path": p, "layer": lay,
                                "ffield": d_ff.get().strip(), "fvalues": fvals})
            refresh_tv()
            dlg.destroy()

        fr_b = ttk.Frame(dlg)
        fr_b.grid(row=6, column=0, columnspan=3, sticky="e", padx=8, pady=8)
        ttk.Button(fr_b, text="Отмена", command=dlg.destroy).pack(side="right")
        ttk.Button(fr_b, text="Добавить", command=ok).pack(side="right", padx=6)

    def del_target():
        for iid in sorted((int(i) for i in tv.selection()), reverse=True):
            targets.pop(iid)
        refresh_tv()

    def move(delta):
        sel = [int(i) for i in tv.selection()]
        if len(sel) != 1:
            return
        i, j = sel[0], sel[0] + delta
        if 0 <= j < len(targets):
            targets[i], targets[j] = targets[j], targets[i]
            refresh_tv()
            tv.selection_set(str(j))

    fr_tb = ttk.Frame(fr_t)
    fr_tb.pack(fill="x", padx=6, pady=(0, 6))
    ttk.Button(fr_tb, text="Добавить слой ЦС…", command=add_target).pack(side="left")
    ttk.Button(fr_tb, text="Удалить", command=del_target).pack(side="left", padx=6)
    ttk.Button(fr_tb, text="Выше", command=lambda: move(-1)).pack(side="left")
    ttk.Button(fr_tb, text="Ниже", command=lambda: move(1)).pack(side="left", padx=6)

    # ---------- параметры ----------
    fr_o = ttk.LabelFrame(win, text="3. Параметры и результат")
    fr_o.pack(fill="x", **pad)
    v_tol = tk.StringVar(value="1.0")
    v_drop = tk.BooleanVar(value=False)
    v_iss = tk.BooleanVar(value=False)
    v_out = tk.StringVar()
    ttk.Label(fr_o, text="Допуск ошибки топологии, м").grid(row=0, column=0, sticky="w", padx=6, pady=3)
    ttk.Entry(fr_o, textvariable=v_tol, width=10).grid(row=0, column=1, sticky="w", padx=6)
    ttk.Checkbutton(fr_o, text="Удалять объекты выбранных классов П, не пересекающиеся с ЦС",
                    variable=v_drop).grid(row=1, column=0, columnspan=3, sticky="w", padx=6, pady=3)
    ttk.Checkbutton(fr_o, text="Записать слой проблем topology_issues в тот же GPKG",
                    variable=v_iss).grid(row=2, column=0, columnspan=3, sticky="w", padx=6, pady=3)
    ttk.Label(fr_o, text="Выходной GPKG").grid(row=3, column=0, sticky="w", padx=6, pady=3)
    ttk.Entry(fr_o, textvariable=v_out, width=70).grid(row=3, column=1, sticky="we", padx=6)

    def browse_out():
        p = filedialog.asksaveasfilename(parent=win, title="Выходной GPKG", defaultextension=".gpkg",
                                         filetypes=[("GeoPackage", "*.gpkg")],
                                         initialdir=os.path.dirname(v_out.get()) or root_dir,
                                         initialfile=os.path.basename(v_out.get()))
        if p:
            v_out.set(os.path.normpath(p))

    ttk.Button(fr_o, text="Обзор…", command=browse_out).grid(row=3, column=2, padx=6)
    fr_o.columnconfigure(1, weight=1)

    def on_run():
        if state["attrs"] is None or not v_fld.get():
            messagebox.showwarning("Нет слоя П", "Выберите слой П и поле класса.", parent=win)
            return
        if not targets:
            messagebox.showwarning("Нет слоёв ЦС", "Добавьте хотя бы один слой ЦС.", parent=win)
            return
        try:
            tol = float(v_tol.get().replace(",", "."))
            if tol < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Допуск — неотрицательное число (м).", parent=win)
            return
        result.update(cfg={
            "p_path": v_p.get(), "p_layer": v_pl.get(), "field": v_fld.get(),
            "targets": list(targets), "tol": tol, "drop_unmatched": bool(v_drop.get()),
            "write_issues": bool(v_iss.get()),
            "out_path": v_out.get().strip() or default_out_path(v_p.get(), v_pl.get(), root_dir),
        })
        win.destroy()

    fr_r = ttk.Frame(win)
    fr_r.pack(fill="x", **pad)
    ttk.Label(fr_r, text="Ход обработки выводится в консоль.").pack(side="left")
    ttk.Button(fr_r, text="Закрыть", command=win.destroy).pack(side="right")
    ttk.Button(fr_r, text="Запустить", command=on_run).pack(side="right", padx=6)

    win.mainloop()
    return result.get("cfg")


# ===========================================================================
# 8. MAIN
# ===========================================================================

def _parse_target(s, p_series):
    parts = s.split("|")
    if len(parts) < 2:
        raise ValueError(f"--target «{s}»: нужен формат КЛАСС|ФАЙЛ[|СЛОЙ][|ПОЛЕ=ЗНАЧ,...]")
    value = coerce_value(parts[0].strip(), p_series)
    t = {"value": value, "path": parts[1].strip(), "layer": None, "ffield": "", "fvalues": []}
    for extra in parts[2:]:
        extra = extra.strip()
        if "=" in extra:
            ff, fv = extra.split("=", 1)
            t["ffield"], t["fvalues"] = ff.strip(), [v.strip() for v in fv.split(",") if v.strip()]
        elif extra:
            t["layer"] = extra
    return t


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        log("08 - Топологическая коррекция. Рабочая папка: " + os.getcwd())
        try:
            cfg = _gui_dialog(os.getcwd())
        except Exception as e:
            log(f"Графический интерфейс недоступен ({e}). Используйте CLI: "
                f"python 08_Topology.py --help")
            return
        if not cfg:
            log("Окно закрыто без запуска.")
            return
        run_job(cfg)
        return

    p = argparse.ArgumentParser(description="Топологическая коррекция слоя П по слоям ЦС")
    p.add_argument("p_path", help="слой П")
    p.add_argument("--layer", default=None, help="слой внутри GPKG")
    p.add_argument("--field", required=True, help="поле класса")
    p.add_argument("--target", action="append", required=True,
                   help="КЛАСС|ФАЙЛ[|СЛОЙ][|ПОЛЕ=ЗНАЧ1,ЗНАЧ2]; можно несколько, порядок = приоритет")
    p.add_argument("--tol", type=float, default=1.0, help="допуск, м")
    p.add_argument("--drop-unmatched", action="store_true",
                   help="удалять объекты классов П, не пересекающиеся с ЦС")
    p.add_argument("--issues", action="store_true", help="записать слой topology_issues")
    p.add_argument("--out", default=None, help="выходной GPKG")
    a = p.parse_args(argv)

    attrs = read_layer(a.p_path, a.layer, attrs_only=True)
    if a.field not in attrs.columns:
        p.error(f"в слое П нет поля {a.field}")
    cfg = {
        "p_path": a.p_path, "p_layer": a.layer, "field": a.field,
        "targets": [_parse_target(s, attrs[a.field]) for s in a.target],
        "tol": a.tol, "drop_unmatched": a.drop_unmatched, "write_issues": a.issues,
        "out_path": a.out or default_out_path(a.p_path, a.layer),
    }
    run_job(cfg)


if __name__ == "__main__":
    main()
