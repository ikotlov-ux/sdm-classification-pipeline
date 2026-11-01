#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_generalization.py — генерализация классифицированной карты из 04_predict_map.

Окно для выбора файлов и параметров + быстрые алгоритмы (векторизованный sieve,
тайловая векторизация marching squares), перенесённые из
06_sieve_marching_squares.py без изменений.

1) SIEVE: связные группы пикселей одного класса меньше min_size
   присоединяются к преобладающему соседнему классу (numpy, без цикла
   по компонентам). Соседство 4 или 8.
2) ВЕКТОРИЗАЦИЯ тайлами (marching squares, уровень 0.5 по маске класса):
   узлы контура лежат между центрами пикселей, граница не «лестничная».
   Чайкин, Douglas-Peucker, разбиение без зазоров внутри тайла (repair),
   параллельные процессы (workers). Упрощение DP применяется к СЕТИ ЛИНИЙ
   внутри тайла, а не к полигонам по отдельности: общая граница двух
   классов упрощается один раз, поэтому наложений между классами не
   возникает (проверено попарно).

3) НЕГЕНЕРАЛИЗУЕМЫЕ СЛОИ (вода, болота и т.п.): векторные полигоны
   замещают собой классификацию и не генерализуются — их границы остаются
   точно такими, как во входном слое. Остальные категории подстраиваются:
   их полигоны обрезаются по границам негенерализуемых слоёв. Слоёв можно
   добавить несколько («Добавить ещё…»): файл + (для GPKG) имя слоя + номер
   класса в классификации. Слои, пересекающиеся между собой: приоритет у
   верхнего в списке. При другом CRS слой перепроецируется в CRS растра;
   за пределы растра полигоны обрезаются.

   СТИРАНИЕ КЛАССОВ РАСТРА («стираемые классы» у каждого слоя): пиксели
   перечисленных классов растра передаются ПРЕОБЛАДАЮЩЕМУ соседнему классу
   (механизм sieve), поэтому генерализация по ним не проводится и дыр
   не остаётся; категорию предоставляют только полигоны слоя. Пример:
   слой «озёра OSM» с классом 400 и стираемыми классами 1400,2400,3400,4400
   убирает из классификации ВСЮ растровую воду (включая «бахрому» вдоль
   берегов, которую сглаживание превращало в артефакты) — вода в карте
   остаётся только по контурам OSM. По умолчанию стирается сам класс слоя.

   Если у класса есть такой слой, классифицированные пиксели этого класса
   остаются только там, где их не накрывает слой.

4) КЛАССЫ БЕЗ РАСШИРЕНИЯ ГРАНИЦ (--no-expand): полигоны перечисленных
   классов обрезаются по пиксельным границам класса (с допуском
   --expand-tol пикселей) — для классов, у которых нет векторного слоя.

Запуск из папки проекта:
    %run 07_generalization.py        # Jupyter — откроется окно
    python 07_generalization.py      # терминал — откроется окно

В окне:
  • «Добавить файлы…» — Проводник с мультивыбором (*_cat.tif из 04_FINAL);
  • параметры с дефолтами; ход обработки выводится в консоль;
  • по умолчанию сохраняется ТОЛЬКО вектор GPKG. Очищенный растр нужен
    тайловой векторизации как промежуточный файл и удаляется после неё,
    если не отмечен флажок «Сохранить очищенный растр».

Выход: 07_GENERALIZATION/<ддммгг_ччмм><_vXXX источника>/
    <base>_sieveN_cM_smK_ref6-7.gpkg   (ref — номера негенерализуемых классов)
    <base>_sieveN_cM_smK_nxCLS_cat.tif  только по флажку
    run_info.txt

Без tkinter/дисплея — консольный режим. CLI сохранён:
    python 07_generalization.py class.tif --min-size 5 --connectivity 4 \
        --out-raster class_sieved.tif --out-vector class.gpkg --smooth 2 \
        --tile 256 --workers 8 --ref "D:\\geo\\lakes.gpkg|water|6" \\
        --ref "D:\\geo\\bogs.gpkg|7"

Производительность (растр 6174x6408, 25 классов, 2 ядра): sieve ~2.5 мин;
тайл 256 px с repair ~4 c, 512 px ~46 c — тайл с repair не увеличивать.
При workers > 1 используется multiprocessing: в JupyterLab/Spyder это может
не работать (spawn), многопоточный прогон надёжнее из cmd/Anaconda Prompt.

Зависимости:
    pip install numpy scipy scikit-image rasterio shapely geopandas pandas pyarrow
"""

from __future__ import annotations

import argparse
import re
import glob
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from datetime import datetime

import numpy as np

# ===========================================================================
# 1. SIEVE — векторизованный
# ===========================================================================


def _structure(connectivity: int) -> np.ndarray:
    if connectivity == 4:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    if connectivity == 8:
        return np.ones((3, 3), dtype=bool)
    raise ValueError("connectivity должен быть 4 или 8")


def _shifts(connectivity: int):
    s4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 4:
        return s4
    return s4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _pair_slices(dr, dc, shape):
    """Пара согласованных срезов: окно пикселя и окно его соседа (dr, dc)."""
    h, w = shape
    r_a = slice(max(0, -dr), h - max(0, dr))
    r_b = slice(max(0, dr), h - max(0, -dr))
    c_a = slice(max(0, -dc), w - max(0, dc))
    c_b = slice(max(0, dc), w - max(0, -dc))
    return (r_a, c_a), (r_b, c_b)


def _sieve_pass_window(out, valid, r0, r1, inner_lo, inner_hi, min_size,
                       struct, shifts, stage=0, force_classes=None):
    """
    Один проход по окну строк [r0, r1) массива out.

    Обрабатываются только группы, ПЕРВЫЙ пиксель которых лежит в
    «своей» полосе [inner_lo, inner_hi) окна — это гарантирует, что каждая
    группа обрабатывается ровно одним блоком. Так как группа размером
    < min_size занимает не более min_size строк, при перекрытии >= min_size
    она целиком попадает в окно, то есть результат совпадает с обработкой
    всего растра сразу.

    stage=0 — голосуют только НЕмелкие соседи. Это принципиально: если
    разрешить мелким группам голосовать друг за друга, две соседние мелкие
    группы будут вечно обмениваться классами (при одновременном обновлении
    процесс не сходится).
    stage=1 — режим слипания для скоплений, у которых все соседи тоже мелкие:
    группа присоединяется только к СТРОГО СТАРШЕЙ соседней группе, где
    старшинство = (размер, затем меньший номер метки). Отношение «старше»
    ациклично, поэтому обмена классами не возникает, а мелкие скопления
    быстро сливаются в группы, которые затем расшивает stage=0.
    """
    from scipy import ndimage as ndi

    win = out[r0:r1]
    vw = valid[r0:r1]
    hh, ww = win.shape

    classes = np.unique(win[vw])
    if classes.size < 2:
        return 0, 0
    n_cls = classes.size
    force = (set(int(c) for c in force_classes) if force_classes else set()
    ) & set(int(c) for c in classes)
    lut_off = int(classes.min())
    lut = np.zeros(int(classes.max()) - lut_off + 1, np.int32)
    lut[classes - lut_off] = np.arange(n_cls, dtype=np.int32)

    # нумерация связных областей по всем классам окна
    lab = np.zeros((hh, ww), np.int32)
    force_lbl = np.zeros(0, dtype=np.int64)
    offset = 0
    for c in classes:
        m = (win == c) & vw
        if not m.any():
            continue
        l, n = ndi.label(m, structure=struct)
        if n:
            lab[m] = l[m] + offset
            if c in force:
                force_lbl = np.r_[force_lbl,
                                  np.arange(offset + 1, offset + n + 1)]
            offset += n
    if offset == 0:
        return 0, 0
    forced = np.zeros(offset + 1, dtype=bool)
    if force_lbl.size:
        forced[force_lbl] = True

    flat = lab.ravel()
    sizes = np.bincount(flat, minlength=offset + 1)
    sizes[0] = 0
    # строка первого пикселя каждой метки (scipy нумерует в порядке обхода)
    ulab, ufirst = np.unique(flat, return_index=True)
    first_row = np.zeros(offset + 1, np.int32)
    first_row[ulab] = (ufirst // ww).astype(np.int32)

    ok = (sizes > 0) & (sizes < min_size)
    ok |= forced                      # классы на стирание — любого размера
    ok[0] = False
    ok &= (first_row >= inner_lo) & (first_row < inner_hi)
    small_lbl = np.nonzero(ok)[0]
    if small_lbl.size == 0:
        return 0, 0

    remap = np.full(offset + 1, -1, np.int32)
    remap[small_lbl] = np.arange(small_lbl.size, dtype=np.int32)
    sidx = remap[lab]
    del flat, remap, ulab, ufirst

    if stage == 1:
        # старшинство: сначала размер, при равенстве — меньший номер метки
        rank = sizes.astype(np.int64) * (offset + 1) - np.arange(offset + 1)
    else:
        rank = None

    keys_parts, cnt_parts = [], []
    for dr, dc in shifts:
        (ra, ca), (rb, cb) = _pair_slices(dr, dc, (hh, ww))
        a_idx = sidx[ra, ca]
        sel = a_idx >= 0
        if not sel.any():
            continue
        a_cls = win[ra, ca][sel]
        b_cls = win[rb, cb][sel]
        keep = vw[rb, cb][sel] & (b_cls != a_cls)
        if stage == 0:
            # голосуют только соседи, не входящие в мелкие группы
            keep &= sidx[rb, cb][sel] < 0
        else:
            keep &= rank[lab[rb, cb][sel]] > rank[lab[ra, ca][sel]]
        if not keep.any():
            continue
        k = (a_idx[sel][keep].astype(np.int64) * n_cls
             + lut[b_cls[keep] - lut_off])
        uk, uc = np.unique(k, return_counts=True)
        keys_parts.append(uk)
        cnt_parts.append(uc.astype(np.int32))

    if not keys_parts:
        return 0, small_lbl.size

    keys = np.concatenate(keys_parts)
    cnts = np.concatenate(cnt_parts)
    del keys_parts, cnt_parts

    o = np.argsort(keys, kind="stable")
    keys, cnts = keys[o], cnts[o]
    starts = np.nonzero(np.r_[True, keys[1:] != keys[:-1]])[0]
    keys = keys[starts]
    cnts = np.add.reduceat(cnts, starts)

    g_small = keys // n_cls
    g_cls = keys % n_cls
    # максимум по числу контактов, при равенстве — по значению класса
    o2 = np.lexsort((g_cls, cnts, g_small))
    gs, gc = g_small[o2], g_cls[o2]
    last = np.nonzero(np.r_[gs[1:] != gs[:-1], True])[0]
    winner = np.full(small_lbl.size, -1, np.int64)
    winner[gs[last]] = classes[gc[last]]

    px = sidx >= 0
    if not px.any():
        return 0, 0
    vals = winner[sidx[px]]
    good = vals >= 0
    if not good.any():
        return 0, small_lbl.size
    idx = np.nonzero(px.ravel())[0][good]
    flat_win = win.ravel()
    newvals = vals[good].astype(win.dtype)
    changed = int((flat_win[idx] != newvals).sum())
    flat_win[idx] = newvals
    unresolved = int(small_lbl.size - np.unique(sidx[px][good]).size)
    return changed, unresolved


def sieve_majority_fast(arr: np.ndarray,
                        min_size: int,
                        connectivity: int = 8,
                        nodata_mask: np.ndarray | None = None,
                        max_iter: int = 30,
                        chunk_rows: int = 1024,
                        force_classes=None,
                        verbose: bool = True) -> np.ndarray:
    """
    Группы связных пикселей одного класса площадью < min_size получают
    ПРЕОБЛАДАЮЩИЙ соседний класс (голосование по числу примыкающих пикселей).

    force_classes — классы, чьи группы передаются соседям ЛЮБОГО размера
    (стирание класса из растра перед заменой его векторным слоем).

    Отличия от версии v1:
      * нет Python-цикла по компонентам (при 13.7 млн групп это часы) —
        голоса всех групп считаются векторно через сдвиги массива;
      * обработка блоками строк с перекрытием min_size + 2, поэтому память
        не зависит от числа групп; результат при этом точный.

    Отличие от gdal.SieveFilter: там группа отдаётся самому БОЛЬШОМУ соседу,
    здесь — соседу с наибольшей длиной общей границы.
    """
    only_force = bool(force_classes) and (min_size is None or min_size <= 1)
    if (min_size is None or min_size <= 1) and not only_force:
        if verbose:
            print("[sieve] min_size <= 1 — фильтрация пропущена")
        return arr.copy()

    out = np.ascontiguousarray(arr.copy())
    valid = np.ones(arr.shape, bool) if nodata_mask is None else ~nodata_mask
    struct = _structure(connectivity)
    shifts = _shifts(connectivity)
    H = out.shape[0]
    ov = max((min_size if min_size and min_size > 1 else 1) + 2, 3)
    step = max(chunk_rows, ov * 4)

    stage = 0
    for it in range(1, max_iter + 1):
        t0 = time.time()
        changed = 0
        unresolved = 0
        for lo in range(0, H, step):
            hi = min(H, lo + step)
            r0 = max(0, lo - ov)
            r1 = min(H, hi + ov)
            ch, un = _sieve_pass_window(out, valid, r0, r1, lo - r0, hi - r0,
                                        min_size, struct, shifts, stage,
                                        force_classes=force_classes)
            changed += ch
            unresolved += un
        if verbose:
            tag = "" if stage == 0 else " [резервный режим]"
            tag2 = " (стирание классов)" if force_classes else ""
            print(f"[sieve] итерация {it}{tag}{tag2}: переназначено {changed}, "
                  f"осталось групп без крупного соседа {unresolved}, "
                  f"{time.time() - t0:.1f} c")
        if changed == 0 and unresolved == 0:
            if verbose:
                print("[sieve] мелких групп больше нет — стоп")
            break
        # чередуем расшивание по границе с крупными контурами (stage 0)
        # и слипание мелких скоплений между собой (stage 1)
        if changed == 0 and stage == 1:
            if verbose:
                print(f"[sieve] осталось {unresolved} нерасшиваемых групп — стоп")
            break
        stage = 1 if stage == 0 else 0

    return out


# ===========================================================================
# 2. ВЕКТОРИЗАЦИЯ marching squares (ядро, работает по одному окну)
# ===========================================================================


def _chaikin(coords: np.ndarray, iterations: int = 1) -> np.ndarray:
    pts = coords
    for _ in range(iterations):
        n = len(pts)
        if n < 3:
            return pts
        new = np.empty((2 * n, 2), dtype=float)
        p = pts
        q = np.roll(pts, -1, axis=0)
        new[0::2] = 0.75 * p + 0.25 * q
        new[1::2] = 0.25 * p + 0.75 * q
        pts = new
    return pts


def _rc_to_xy(contour_rc: np.ndarray, transform, pad: int) -> np.ndarray:
    rr = contour_rc[:, 0] - pad
    cc = contour_rc[:, 1] - pad
    xs, ys = transform * (cc + 0.5, rr + 0.5)
    return np.column_stack([np.asarray(xs), np.asarray(ys)])


def _rings_to_polygons(rings):
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.geometry.polygon import orient

    polys = [Polygon(r) for r in rings]
    polys = [p for p in polys if p.area > 0]
    if not polys:
        return None
    order = np.argsort([-p.area for p in polys])
    polys = [polys[i] for i in order]

    reps = [p.representative_point() for p in polys]
    depth = [0] * len(polys)
    # индекс по дереву, иначе O(n^2) на тысячах контуров
    from shapely import STRtree
    tree = STRtree(polys)
    for i, rp in enumerate(reps):
        for j in tree.query(rp):
            if j < i and polys[j].contains(rp):
                depth[i] += 1

    shells = [p for p, d in zip(polys, depth) if d % 2 == 0]
    holes = [p for p, d in zip(polys, depth) if d % 2 == 1]

    built = []
    if holes:
        htree = STRtree(holes)
        for sh in shells:
            cand = [holes[j] for j in htree.query(sh)]
            sh_holes = [h.exterior.coords for h in cand
                        if sh.contains(h.representative_point())]
            built.append(Polygon(sh.exterior.coords, sh_holes))
    else:
        built = shells

    geoms = []
    for p in built:
        if not p.is_valid:
            p = p.buffer(0)
        for q in _polygons(p):
            geoms.append(orient(q, 1.0))
    if not geoms:
        return None
    return MultiPolygon(geoms) if len(geoms) > 1 else geoms[0]


def _polygons(geom):
    """
    Все полигоны из любой геометрии.

    Важно: пересечение полигона с рамкой тайла часто даёт
    GeometryCollection (полигоны + линии/точки касания по рамке). Такую
    геометрию нельзя отбрасывать целиком и нельзя разбирать проверкой
    geom_type.startswith("Multi").
    """
    if geom is None or geom.is_empty:
        return []
    gt = geom.geom_type
    if gt == "Polygon":
        return [geom] if geom.area > 0 else []
    if gt in ("MultiPolygon", "GeometryCollection"):
        out = []
        for g in geom.geoms:
            out.extend(_polygons(g))
        return out
    return []


def _repair_topology(feats, arr, transform, frame, nodata_mask=None,
                     simplify=0.0):
    """
    Строгое разбиение без зазоров и перекрытий: все границы классов + рамка
    объединяются в линейную сеть, сеть разбивается на элементарные грани
    (polygonize), каждая грань получает класс по значению растра в её
    представительной точке, затем грани сливаются по классам.

    simplify > 0: упрощение Douglas-Peucker применяется к СЕТИ ЛИНИЙ
    (общая граница двух классов — одна линия, упрощается один раз), а не
    к полигонам по отдельности — иначе общие границы соседних классов
    расходятся и появляются наложения. Разбиение остаётся строгим.

    Вызывается ТОЛЬКО в пределах одного тайла — на весь растр эта операция
    неподъёмна.
    """
    from shapely.ops import polygonize, unary_union

    if len(feats) < 2:
        return feats

    lines = [g.boundary for _, g in feats]
    lines.append(frame.exterior)
    net = unary_union(lines)
    if simplify and simplify > 0 and not net.is_empty:
        net = net.simplify(simplify)
        if net.is_empty:
            return feats
    faces = list(polygonize(net))
    if not faces:
        return feats

    h, w = arr.shape
    inv = ~transform
    buckets = {}
    for f in faces:
        if f.is_empty or f.area <= 0:
            continue
        pt = f.representative_point()
        c, r = inv * (pt.x, pt.y)
        r = min(max(int(np.floor(r)), 0), h - 1)
        c = min(max(int(np.floor(c)), 0), w - 1)
        if nodata_mask is not None and nodata_mask[r, c]:
            continue
        buckets.setdefault(int(arr[r, c]), []).append(f)

    out = []
    for cls in sorted(buckets):
        parts = _polygons(unary_union(buckets[cls]))
        if parts:
            from shapely.geometry import MultiPolygon
            out.append((cls, parts[0] if len(parts) == 1
                        else MultiPolygon(parts)))
    return out


def vectorize_window(arr, transform, nodata_mask=None, smooth=0,
                     edge_flags=(False, False, False, False),
                     edge_mode="full", repair=True, simplify=0.0):
    """
    Векторизация одного окна методом marching squares.

    edge_flags = (top, bottom, left, right): True там, где сторона окна
    совпадает с краем всего растра. Только на этих сторонах применяется
    edge_mode='full' (репликация краевых пикселей), на внутренних сторонах
    контур всё равно будет обрезан вызывающим кодом.
    """
    from shapely.geometry import MultiPolygon, box
    from skimage import measure

    valid = np.ones(arr.shape, bool) if nodata_mask is None else ~nodata_mask
    classes = np.unique(arr[valid])
    if classes.size == 0:
        return []

    h, w = arr.shape
    x0, y0 = transform * (0, 0)
    x1, y1 = transform * (w, h)
    frame = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    top, bottom, left, right = edge_flags
    results = []
    for cls in classes:
        mask = ((arr == cls) & valid).astype(np.float32)
        if edge_mode == "full" and any(edge_flags):
            m = np.pad(mask, 1, mode="constant", constant_values=0.0)
            if top:
                m[0, :] = m[1, :]
            if bottom:
                m[-1, :] = m[-2, :]
            if left:
                m[:, 0] = m[:, 1]
            if right:
                m[:, -1] = m[:, -2]
            padded = np.pad(m, 1, mode="constant", constant_values=0.0)
            pad = 2
        else:
            padded = np.pad(mask, 1, mode="constant", constant_values=0.0)
            pad = 1

        contours = measure.find_contours(padded, level=0.5)
        if not contours:
            continue
        rings = []
        for c in contours:
            xy = _rc_to_xy(c, transform, pad=pad)
            if len(xy) > 1 and np.allclose(xy[0], xy[-1]):
                xy = xy[:-1]
            if len(xy) < 3:
                continue
            if smooth > 0:
                xy = _chaikin(xy, smooth)
            rings.append(np.vstack([xy, xy[0]]))

        geom = _rings_to_polygons(rings)
        if geom is None or geom.is_empty:
            continue
        parts = _polygons(geom.intersection(frame))
        if parts:
            geom = parts[0] if len(parts) == 1 else MultiPolygon(parts)
            results.append((int(cls), geom))

    if repair:
        results = _repair_topology(results, arr, transform, frame, nodata_mask,
                                   simplify=simplify)
    elif simplify and simplify > 0:
        # без repair упрощаем каждый полигон отдельно: у соседних классов
        # общие границы могут разойтись (возможны наложения/зазоры)
        out = []
        for cls, g in results:
            g = g.simplify(simplify, preserve_topology=True)
            if not g.is_empty:
                out.append((cls, g))
        results = out
    return results


# ===========================================================================
# 3. ТАЙЛОВАЯ ВЕКТОРИЗАЦИЯ ВСЕГО РАСТРА
# ===========================================================================

_JOB = {}


def _clamp_workers(workers, n_tasks, verbose=True):
    """Приведение числа процессов к допустимому.

    На Windows ProcessPoolExecutor не принимает больше 61 процесса
    (ограничение WaitForMultipleObjects), поэтому 128-ядерная машина
    требует явного ограничения. Больше процессов, чем тайлов, тоже
    бессмысленно.
    """
    hard = 60 if sys.platform == "win32" else 4096
    cpu = os.cpu_count() or 1
    w = max(1, min(int(workers), hard, cpu, max(1, n_tasks)))
    if verbose and w != int(workers):
        why = []
        if int(workers) > hard:
            why.append(f"предел Windows {hard}")
        if int(workers) > cpu:
            why.append(f"ядер {cpu}")
        if int(workers) > n_tasks:
            why.append(f"тайлов {n_tasks}")
        print(f"[tiles] workers {workers} -> {w} ({', '.join(why)})",
              flush=True)
    return w


def _init_worker(path, margin, smooth, edge_mode, repair, simplify, parts_dir):
    _JOB.update(path=path, margin=margin, smooth=smooth,
                edge_mode=edge_mode, repair=repair, simplify=simplify,
                parts_dir=parts_dir)


def _tile_worker(task):
    """Обрабатывает один тайл и пишет результат в parquet-часть."""
    import geopandas as gpd
    import rasterio
    from rasterio.windows import Window
    from shapely.geometry import box

    tid, row_off, col_off, th, tw = task
    j = _JOB
    out_path = os.path.join(j["parts_dir"], f"part_{tid:06d}.parquet")
    if os.path.exists(out_path):
        return tid, -1, 0.0

    t0 = time.time()
    m = j["margin"]
    with rasterio.open(j["path"]) as src:
        H, W = src.height, src.width
        r0, c0 = max(0, row_off - m), max(0, col_off - m)
        r1, c1 = min(H, row_off + th + m), min(W, col_off + tw + m)
        big = Window(c0, r0, c1 - c0, r1 - r0)
        arr = src.read(1, window=big)
        wt = src.window_transform(big)
        inner = src.window_transform(Window(col_off, row_off, tw, th))
        nodata = src.nodata
        crs = src.crs

    nd_mask = None
    if nodata is not None:
        nd_mask = np.isnan(arr) if np.isnan(nodata) else (arr == nodata)
        if not nd_mask.any():
            nd_mask = None
        elif nd_mask.all():
            gpd.GeoDataFrame({"class_value": []}, geometry=[], crs=crs
                             ).to_parquet(out_path)
            return tid, 0, time.time() - t0

    ix0, iy0 = inner * (0, 0)
    ix1, iy1 = inner * (tw, th)
    inner_box = box(min(ix0, ix1), min(iy0, iy1), max(ix0, ix1), max(iy0, iy1))

    edge_flags = (r0 == 0, r1 == H, c0 == 0, c1 == W)
    feats = vectorize_window(arr, wt, nd_mask, smooth=j["smooth"],
                             edge_flags=edge_flags, edge_mode=j["edge_mode"],
                             repair=j["repair"], simplify=j.get("simplify", 0.0))

    rows_cls, rows_geom = [], []
    for cls, g in feats:
        for p in _polygons(g.intersection(inner_box)):
            rows_cls.append(cls)
            rows_geom.append(p)

    gpd.GeoDataFrame({"class_value": rows_cls}, geometry=rows_geom,
                     crs=crs).to_parquet(out_path)
    return tid, len(rows_geom), time.time() - t0


def _parse_ref_spec(spec: str) -> dict:
    """'файл|класс' | 'файл|слой|класс' | 'файл|слой|класс|стираемые'
    -> {path, layer, class_value, erase}.

    стираемые — коды классов растра через запятую: их пиксели передаются
    преобладающему соседнему классу ещё до sieve (генерализация по ним не
    проводится, категорию даёт только сам слой). Пустой слой:
    'файл||класс|стираемые'."""
    parts = [p for p in str(spec).split("|")]
    erase = None
    if len(parts) == 4:
        *parts, er = parts
        try:
            erase = [int(x) for x in re.split(r"[,;\s]+", er) if x.strip()]
        except ValueError:
            raise ValueError(f"Слой-эталон «{spec}»: стираемые классы — "
                             f"целые числа через запятую, получено «{er}»")
    if len(parts) == 2:
        path, cls = parts
        layer = None
    elif len(parts) == 3:
        path, layer, cls = parts
    else:
        raise ValueError(f"Слой-эталон «{spec}»: ожидается файл|класс, "
                         f"файл|имя слоя|класс или файл|слой|класс|стираемые")
    path = path.strip()
    cls = cls.strip()
    if not path or not cls:
        raise ValueError(f"Слой-эталон «{spec}»: не заданы файл или класс")
    try:
        cv = int(float(cls))
    except ValueError:
        raise ValueError(f"Слой-эталон «{spec}»: номер класса должен быть "
                         f"числом, получено «{cls}»")
    return {"path": path, "layer": layer or None, "class_value": cv,
            "erase": erase}


def _apply_ref_layers(gdf, path, refs, verbose=True):
    """
    Негенерализуемые слои: замещают классификацию, границы не меняются.

    refs — список {path, layer, class_value} в порядке приоритета: верхний
    слой в списке главнее (при пересечении из нижнего вычитается верхний).
    Возвращает (gdf, суммарная площадь слоёв).
    """
    import geopandas as gpd
    import pandas as pd
    import rasterio
    from shapely import STRtree
    from shapely.geometry import MultiPolygon
    from shapely.ops import unary_union

    refs = list(refs)
    with rasterio.open(path) as src:
        bbox = src.bounds
        crs = src.crs
    frame = MultiPolygon() if False else None
    from shapely.geometry import box
    frame = box(*bbox)

    kept = []          # [(class_value, geom)]
    total_area = 0.0
    cumulative = None
    for i, ref in enumerate(refs, 1):
        try:
            r = gpd.read_file(ref["path"], layer=ref.get("layer") or None)
        except Exception as e:
            raise RuntimeError(f"Не удалось прочитать слой-эталон "
                               f"{ref['path']}: {e}")
        if r.crs is not None and r.crs != crs:
            r = r.to_crs(crs)
        geoms = [g for g in r.geometry if g is not None and not g.is_empty]
        if not geoms:
            if verbose:
                print(f"[ref] слой {i}: {ref['path']} — пустая геометрия, "
                      f"пропущен", flush=True)
            continue
        g = unary_union(geoms).intersection(frame)
        if g.is_empty:
            if verbose:
                print(f"[ref] слой {i}: {ref['path']} — вне растра, пропущен",
                      flush=True)
            continue
        if cumulative is not None:
            g = g.difference(cumulative)
            if g.is_empty:
                if verbose:
                    print(f"[ref] слой {i}: {ref['path']} — полностью перекрыт "
                          f"слоями выше, пропущен", flush=True)
                continue
        cumulative = g if cumulative is None else unary_union([cumulative, g])
        total_area += float(g.area)
        n = len(getattr(g, "geoms", [g]))
        if verbose:
            print(f"[ref] слой {i}: {os.path.basename(ref['path'])}"
                  + (f" ({ref['layer']})" if ref.get("layer") else "")
                  + f" -> класс {ref['class_value']}: {n} полигонов, "
                  f"{g.area:.1f} (единиц CRS²)", flush=True)
        kept.append((int(ref["class_value"]), g))

    if not kept:
        return gdf, 0.0

    ref_union = cumulative
    comps = list(getattr(ref_union, "geoms", [ref_union]))
    tree = STRtree(comps)

    # 1) классификация подстраивается: вычитаем эталон из всех полигонов
    new_geom, keep = [], []
    n_cut = 0
    area_cut = 0.0
    for cls_val, g in zip(gdf["class_value"], gdf.geometry):
        if g is None or g.is_empty:
            new_geom.append(g)
            keep.append(True)
            continue
        cand = [comps[i] for i in tree.query(g)]
        if not cand:
            new_geom.append(g)
            keep.append(True)
            continue
        lim = cand[0] if len(cand) == 1 else unary_union(cand)
        diff = g.difference(lim)
        if diff.is_empty or diff.area <= 1e-9:
            new_geom.append(None)
            keep.append(False)
            continue
        parts = _polygons(diff)
        if not parts:
            new_geom.append(None)
            keep.append(False)
            continue
        area_cut += float(g.area - diff.area)
        n_cut += 1
        new_geom.append(parts[0] if len(parts) == 1 else MultiPolygon(parts))
        keep.append(True)
    gdf = gdf.copy()
    gdf["geometry"] = new_geom
    gdf = gdf[np.array(keep)]
    gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)
    if verbose:
        print(f"[ref] классификация подстроена: обрезано полигонов {n_cut}, "
              f"изъято {area_cut:.1f} (единиц CRS²)", flush=True)

    # 2) эталоны вставляются как есть, граница не меняется
    rows = [{"class_value": cv, "geometry": g} for cv, g in kept]
    ref_gdf = gpd.GeoDataFrame(rows, crs=gdf.crs)
    gdf = gpd.GeoDataFrame(
        pd.concat([ref_gdf, gdf], ignore_index=True), geometry="geometry",
        crs=gdf.crs)
    return gdf, total_area


def _class_footprints(path, classes, verbose=True):
    """
    Пиксельные границы (footprint) классов: class -> список полигонов
    (по одному на связную область, вершины — по краям пикселей).

    Растр читается окнами, поэтому память не зависит от его размера.
    """
    import rasterio
    from rasterio import features
    from rasterio.windows import Window
    from shapely.geometry import shape

    classes = [int(c) for c in classes]
    out = {c: [] for c in classes}
    with rasterio.open(path) as src:
        H, W = src.height, src.width
        nodata = src.nodata
        step = 2048
        for r0 in range(0, H, step):
            for c0 in range(0, W, step):
                win = Window(c0, r0, min(step, W - c0), min(step, H - r0))
                arr = src.read(1, window=win)
                ok = np.ones(arr.shape, bool)
                if nodata is not None:
                    ok &= ~(np.isnan(arr) if np.isnan(nodata)
                            else (arr == nodata))
                for cls in classes:
                    m = (arr == cls) & ok
                    if not m.any():
                        continue
                    for g, v in features.shapes(
                            m.astype(np.uint8), mask=m,
                            transform=src.window_transform(win)):
                        if v == 1:
                            out[cls].append(shape(g))
    if verbose:
        for cls in classes:
            n = len(out[cls])
            if n:
                print(f"[keep] класс {cls}: {n} областей в пиксельных границах")
            else:
                print(f"[keep] класс {cls}: пикселей нет — ограничение не действует")
    return out


def _clip_no_expand(gdf, path, no_expand, expand_tol=0.0, verbose=True):
    """
    Обрезать полигоны перечисленных классов по пиксельным границам класса.

    Гарантия: полигон класса из no_expand не выходит за пределы пикселей
    этого класса в очищенном растре (с допуском expand_tol пикселей, если
    он > 0). Возвращает (gdf, число обрезанных полигонов, срезанная площадь).
    """
    import rasterio
    from shapely import STRtree
    from shapely.geometry import MultiPolygon
    from shapely.ops import unary_union

    no_expand = [int(c) for c in no_expand]
    fps = _class_footprints(path, no_expand, verbose=verbose)
    with rasterio.open(path) as src:
        px = abs(src.transform.a)
    tol = max(float(expand_tol), 0.0) * px

    trees, comps = {}, {}
    for cls in no_expand:
        comps[cls] = fps[cls]
        trees[cls] = STRtree(fps[cls]) if fps[cls] else None

    n_clip = 0
    area_cut = 0.0
    new_geom, keep = [], []
    for cls_val, g in zip(gdf["class_value"], gdf.geometry):
        cls_val = int(cls_val)
        if (cls_val not in trees or trees[cls_val] is None
                or g is None or g.is_empty):
            new_geom.append(g)
            keep.append(True)
            continue
        tree, cl = trees[cls_val], comps[cls_val]
        if tol > 0:
            cand = [cl[i].buffer(tol) for i in tree.query(g)]
        else:
            cand = [cl[i] for i in tree.query(g)]
        if not cand:
            new_geom.append(g)
            keep.append(True)
            continue
        lim = cand[0] if len(cand) == 1 else unary_union(cand)
        diff = g.difference(lim)
        if diff.is_empty or diff.area <= 1e-9:
            new_geom.append(g)
            keep.append(True)
            continue
        parts = _polygons(g.intersection(lim))
        if not parts:                       # полигон целиком за границей
            area_cut += float(g.area)
            n_clip += 1
            keep.append(False)
            continue
        g2 = parts[0] if len(parts) == 1 else MultiPolygon(parts)
        if not g2.is_valid:
            g2 = g2.buffer(0)
        area_cut += float(diff.area)
        n_clip += 1
        new_geom.append(g2)
        keep.append(True)

    gdf = gdf.copy()
    gdf["geometry"] = new_geom
    gdf = gdf[np.array(keep)]
    gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)
    if verbose:
        print(f"[keep] обрезано полигонов {n_clip}, срезано выступов "
              f"{area_cut:.1f} (единиц CRS²)")
    return gdf, n_clip, area_cut


def vectorize_raster_tiled(path, out_vector, tile=256, margin=16, smooth=0,
                           simplify=0.0, edge_mode="full", repair=True,
                           no_expand=None, expand_tol=0.0, refs=None,
                           workers=1, output_mode="patch", layer="classes",
                           parts_dir=None, resume=False, verbose=True):
    """
    Векторизация всего растра тайлами.

    output_mode:
      'patch' — слить части по классам (лечит швы между тайлами) и разбить
                на отдельные полигоны: одна строка = один контур;
      'class' — одна строка на класс (мультиполигон);
      'raw'   — как посчитано по тайлам, без слияния (самый быстрый и самый
                экономный по памяти; видны разрезы по границам тайлов).

    no_expand — список классов, границы которых нельзя расширять: их
                полигоны после сборки обрезаются по пиксельным границам
                класса (с допуском expand_tol пикселей).
    refs      — негенерализуемые слои [{path, layer, class_value}, ...]:
                замещают классификацию, границы не меняются, остальные
                классы обрезаются по ним (приоритет — порядок в списке).
    """
    import geopandas as gpd
    import pandas as pd
    import rasterio

    with rasterio.open(path) as src:
        H, W = src.height, src.width

    if parts_dir is None:
        base = os.path.splitext(out_vector)[0]
        parts_dir = base + "_parts"
    if os.path.isdir(parts_dir) and not resume:
        shutil.rmtree(parts_dir)
    os.makedirs(parts_dir, exist_ok=True)

    tasks = []
    tid = 0
    for r in range(0, H, tile):
        for c in range(0, W, tile):
            tasks.append((tid, r, c, min(tile, H - r), min(tile, W - c)))
            tid += 1
    workers = _clamp_workers(workers, len(tasks), verbose=verbose)
    if verbose:
        print(f"[tiles] тайлов {len(tasks)} ({tile}x{tile}, margin={margin}), "
              f"воркеров {workers}")

    t_start = time.time()
    done = 0
    n_geom = 0

    if workers <= 1:
        _init_worker(path, margin, smooth, edge_mode, repair, simplify, parts_dir)
        for t in tasks:
            _, n, dt = _tile_worker(t)
            done += 1
            n_geom += max(n, 0)
            if verbose and (done % 10 == 0 or done == len(tasks)):
                el = time.time() - t_start
                eta = el / done * (len(tasks) - done)
                print(f"[tiles] {done}/{len(tasks)} полигонов {n_geom} "
                      f"прошло {el/60:.1f} мин, осталось ~{eta/60:.1f} мин")
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker,
            initargs=(path, margin, smooth, edge_mode, repair, simplify,
                      parts_dir),
        ) as ex:
            futs = [ex.submit(_tile_worker, t) for t in tasks]
            for f in as_completed(futs):
                _, n, dt = f.result()
                done += 1
                n_geom += max(n, 0)
                if verbose and (done % 10 == 0 or done == len(tasks)):
                    el = time.time() - t_start
                    eta = el / done * (len(tasks) - done)
                    print(f"[tiles] {done}/{len(tasks)} полигонов {n_geom} "
                          f"прошло {el/60:.1f} мин, осталось ~{eta/60:.1f} мин")

    if verbose:
        print(f"[tiles] готово за {(time.time()-t_start)/60:.1f} мин")

    # --- сборка
    files = sorted(os.path.join(parts_dir, f) for f in os.listdir(parts_dir)
                   if f.endswith(".parquet"))
    gdfs = [gpd.read_parquet(f) for f in files]
    gdfs = [g for g in gdfs if len(g)]
    if not gdfs:
        print("[io] пусто, нечего писать", file=sys.stderr)
        return None
    gdf = pd.concat(gdfs, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdfs[0].crs)
    if verbose:
        print(f"[merge] частей {len(gdf)}, режим сборки: {output_mode}")

    if output_mode in ("patch", "class"):
        gdf = gdf.dissolve(by="class_value", as_index=False)
        if output_mode == "patch":
            gdf = gdf.explode(index_parts=False, ignore_index=True)

    # упрощение DP выполняется внутри тайлов (по сети линий, если repair) —
    # глобальное упрощение полигонов создало бы наложения соседних классов

    if no_expand:
        gdf, n_clip, area_cut = _clip_no_expand(gdf, path, no_expand,
                                                expand_tol, verbose=verbose)

    if refs:
        gdf, ref_area = _apply_ref_layers(gdf, path, refs, verbose=verbose)

    gdf["area"] = gdf.geometry.area
    if str(out_vector).lower().endswith(".gpkg"):
        gdf.to_file(out_vector, layer=layer, driver="GPKG")
    else:
        gdf.to_file(out_vector)
    if verbose:
        print(f"[io] вектор сохранён: {out_vector} ({len(gdf)} объектов, "
              f"площадь {gdf.area.sum():.1f})")
    return gdf


# ===========================================================================
# CLI
# ===========================================================================


def run(in_raster, min_size=3, connectivity=8, out_raster=None,
        out_vector=None, smooth=0, simplify=0.0, edge_mode="full",
        repair=True, tile=256, margin=16, no_expand=None, expand_tol=0.0,
        refs=None, workers=1, output_mode="patch",
        layer="classes", skip_vector=False, resume=False, verbose=True):
    import rasterio

    src_for_vector = in_raster

    erase_classes = sorted({int(c) for ref in (refs or []) for c in
                            (ref.get("erase") or [ref["class_value"]])})

    if (min_size and min_size > 1) or erase_classes:
        with rasterio.open(in_raster) as src:
            arr = src.read(1)
            profile = src.profile
            nodata = src.nodata
        nd = None
        if nodata is not None:
            nd = np.isnan(arr) if np.isnan(nodata) else (arr == nodata)
            if not nd.any():
                nd = None
            elif verbose:
                print(f"[io] nodata={nodata}, пикселей nodata: {nd.sum()}")
        if verbose:
            print(f"[io] растр {arr.shape}, классов {np.unique(arr).size}, "
                  f"dtype={arr.dtype}")

        if erase_classes:
            before = int(np.isin(arr, erase_classes).sum())
            print(f"[erase] классы растра {erase_classes}: {before} пикс. "
                  f"передаётся преобладающим соседям")
        cleaned = sieve_majority_fast(arr, min_size, connectivity, nd,
                                      force_classes=erase_classes or None,
                                      verbose=verbose)
        if erase_classes and verbose:
            print(f"[erase] остаток стираемых классов в растре: "
                  f"{int(np.isin(cleaned, erase_classes).sum())} пикс.")
        if out_raster:
            profile.update(count=1, dtype=cleaned.dtype, compress="lzw",
                           tiled=True, blockxsize=512, blockysize=512)
            with rasterio.open(out_raster, "w", **profile) as dst:
                dst.write(cleaned, 1)
            if verbose:
                print(f"[io] растр сохранён: {out_raster}")
            src_for_vector = out_raster
        elif not skip_vector:
            raise SystemExit("для тайловой векторизации нужен --out-raster "
                             "(тайлы читаются с диска; при --sieve "
                             "или стираемых классах)")
        del arr, cleaned

    if out_vector and not skip_vector:
        vectorize_raster_tiled(
            src_for_vector, out_vector, tile=tile, margin=margin,
            smooth=smooth, simplify=simplify, edge_mode=edge_mode,
            repair=repair, no_expand=no_expand, expand_tol=expand_tol,
            refs=refs, workers=workers, output_mode=output_mode,
            layer=layer, resume=resume, verbose=verbose,
        )



# ===========================================================================
# ИНТЕРАКТИВНЫЙ РЕЖИМ (запуск без аргументов, как в пайплайне classification)
# ===========================================================================

def _prompt(prompt, default):
    raw = input(f"  {prompt} (Enter = {default}): ").strip()
    return raw if raw else str(default)


def _prompt_int(prompt, default):
    return int(_prompt(prompt, default))


def _prompt_float(prompt, default):
    return float(_prompt(prompt, default))


def _prompt_choice(prompt, choices, default):
    raw = _prompt(f"{prompt} {choices}", default)
    return raw if raw in choices else default


def _prompt_yes(prompt, default=True):
    raw = _prompt(prompt + " (y/n)", "y" if default else "n").lower()
    return raw not in ("n", "no", "0")


def _select_prev_subdir(base_dir):
    """Интерактивный выбор подпапки запуска. Возвращает (путь, суффикс)."""
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(
            f"Папка {os.path.basename(base_dir)} не найдена "
            f"в {os.getcwd()}")
    subdirs = sorted(d for d in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, d))
                     and re.search(r"_v\d{3}$", d))
    if not subdirs:
        raise FileNotFoundError(
            f"В {os.path.basename(base_dir)} нет подпапок вида ддммгг_ччмм_vXXX")

    print(f"\nДоступные запуски в {os.path.basename(base_dir)}:\n", flush=True)
    for i, sd in enumerate(subdirs, 1):
        print(f" [{i}] {sd}", flush=True)
    raw = input(f"\nВыберите номер (Enter = {len(subdirs)}, самый свежий): ").strip()
    idx = int(raw) if raw else len(subdirs)
    if idx < 1 or idx > len(subdirs):
        raise ValueError("Некорректный выбор запуска.")
    name = subdirs[idx - 1]
    m = re.search(r"(_v\d{3})$", name)
    suffix = m.group(1) if m else "_v001"
    print(f"Выбрана папка: {name}, суффикс: {suffix}\n", flush=True)
    return os.path.join(base_dir, name), suffix


def _select_cat_raster(folder):
    """Выбор категориального растра *_cat.tif в папке запуска."""
    cats = sorted(glob.glob(os.path.join(folder, "*_cat.tif")))
    if not cats:
        cats = sorted(glob.glob(os.path.join(folder, "*.tif")))
    if not cats:
        raise FileNotFoundError(f"В {folder} нет файлов *.tif")
    if len(cats) == 1:
        print(f"  ✓ Растр: {os.path.basename(cats[0])}", flush=True)
        return cats[0]
    print("\n  Найдено несколько растров:", flush=True)
    for i, c in enumerate(cats, 1):
        print(f"   [{i}] {os.path.basename(c)}", flush=True)
    raw = input("  Выберите номер (Enter = 1): ").strip()
    return cats[(int(raw) if raw else 1) - 1]


def _read_class_table(folder):
    """Чтение *_stats.csv для маппинга ID -> class_name.

    Разделитель и лишние колонки определяются автоматически: в файлах
    пайплайна встречаются и запятая, и точка с запятой, и строки с другим
    числом полей (из-за них раньше падало чтение целиком).
    """
    files = sorted(glob.glob(os.path.join(folder, "*_stats.csv")))
    if not files:
        # ищем на уровень глубже: 04_FINAL/*/..._stats.csv
        files = sorted(glob.glob(os.path.join(folder, "*", "*_stats.csv")),
                       reverse=True)
    if not files:
        return None
    import pandas as pd
    last_err = None
    for path in files:
        for kwargs in ({"sep": None, "engine": "python"},
                       {"sep": ";"}, {"sep": ","}, {"sep": "\t"}):
            try:
                df = pd.read_csv(path, on_bad_lines="skip",
                                 encoding="utf-8-sig", **kwargs)
            except Exception as e:
                last_err = e
                continue
            cols = {c.strip().lower(): c for c in df.columns}
            id_col = next((cols[k] for k in ("id", "class_value", "value",
                                             "class_id", "code") if k in cols),
                          None)
            nm_col = next((cols[k] for k in ("class_name", "name", "class",
                                             "label", "название") if k in cols),
                          None)
            if id_col is None or nm_col is None:
                continue
            sub = df[[id_col, nm_col]].dropna()
            out = {}
            for v, n in zip(sub[id_col], sub[nm_col]):
                try:
                    out[int(float(str(v).strip()))] = str(n).strip()
                except (TypeError, ValueError):
                    continue
            if out:
                print(f"  ✓ Таблица классов: {os.path.basename(path)}, "
                      f"{len(out)} классов", flush=True)
                return out
    if last_err is not None:
        print(f"  ⚠ Не удалось прочитать stats.csv: {last_err}", flush=True)
    return None


# ===========================================================================
# ГЕНЕРАЛИЗАЦИЯ СПИСКА ФАЙЛОВ (общая часть для окна и консоли)
# ===========================================================================

OUT_BASE_NAME = "07_GENERALIZATION"
FINAL_BASE_NAME = "04_FINAL"


def _in_notebook() -> bool:
    return "ipykernel" in sys.modules


def _default_params() -> dict:
    cpu = os.cpu_count() or 1
    cap = min(cpu, 60 if sys.platform == "win32" else cpu)
    return {
        "min_size": 5,
        "connectivity": 4,
        "smooth": 2,
        "simplify": 0.0,
        "edge_mode": "full",
        "repair": True,
        "tile": 256,
        "margin": 16,
        # в Jupyter multiprocessing (spawn) часто не работает
        "workers": 1 if _in_notebook() else min(cap, 8),
        "output_mode": "patch",
        "save_raster": False,   # по умолчанию только вектор
        "no_expand": [],        # классы, границы которых нельзя расширять
        "expand_tol": 0.0,      # допуск расширения, пикселей (0 = строго)
        "refs": [],             # негенерализуемые слои [{path, layer, class_value}]
    }


def _version_suffix(path: str) -> str:
    m = re.search(r"(_v\d{3})$", os.path.basename(os.path.dirname(path)))
    return m.group(1) if m else ""


def _make_tag(p: dict) -> str:
    tag = f"sieve{p['min_size']}_c{p['connectivity']}_sm{p['smooth']}"
    if p["simplify"] > 0:
        tag += f"_dp{p['simplify']:g}"
    if not p["repair"]:
        tag += "_norep"
    if p.get("no_expand"):
        tag += "_nx" + "-".join(str(int(c)) for c in p["no_expand"])
        if p.get("expand_tol", 0.0) > 0:
            tag += f"t{p['expand_tol']:g}"
    if p.get("refs"):
        tag += "_ref" + "-".join(str(int(r_["class_value"])) for r_ in p["refs"])
    return tag


def _add_class_names(out_vector: str, id_to_name: dict) -> None:
    import geopandas as gpd
    gdf = gpd.read_file(out_vector, layer="classes")
    gdf["class_name"] = gdf["class_value"].map(
        lambda v: id_to_name.get(int(v), str(int(v))))
    cols = [c for c in ("class_value", "class_name", "area") if c in gdf.columns]
    cols += [c for c in gdf.columns if c not in cols and c != "geometry"]
    gdf[cols + ["geometry"]].to_file(out_vector, layer="classes", driver="GPKG")


def generalize_files(files: list[str], params: dict, out_base: str) -> list[str]:
    """Обрабатывает растры по очереди. Возвращает список созданных GPKG."""
    line = "=" * 60
    os.makedirs(out_base, exist_ok=True)
    stamp = datetime.now().strftime("%d%m%y_%H%M")
    tag = _make_tag(params)
    created = []

    for k, in_raster in enumerate(files, 1):
        print(f"\n{line}\n  [{k}/{len(files)}] {in_raster}\n{line}", flush=True)
        out_dir = os.path.join(out_base, stamp + _version_suffix(in_raster))
        os.makedirs(out_dir, exist_ok=True)

        base = re.sub(r"_cat$", "", os.path.splitext(os.path.basename(in_raster))[0])
        base = re.sub(r"_sieve\d+_c\d+_sm\d+(_dp[\d.]+)?(_norep)?$", "", base)
        out_vector = os.path.join(out_dir, f"{base}_{tag}.gpkg")

        # refs по умолчанию стирают свой класс (см. run) — нужен растр на диске
        do_sieve = (params["min_size"] > 1
                    or bool(params.get("refs")))
        out_raster = None
        tmp_raster = False
        if do_sieve:
            if params["save_raster"]:
                out_raster = os.path.join(out_dir, f"{base}_{tag}_cat.tif")
            else:
                # тайловая векторизация читает растр с диска — временный файл
                out_raster = os.path.join(out_dir, f"_tmp_{base}_{tag}_sieved.tif")
                tmp_raster = True

        t0 = time.time()
        try:
            run(in_raster=in_raster,
                min_size=params["min_size"],
                connectivity=params["connectivity"],
                out_raster=out_raster,
                out_vector=out_vector,
                smooth=params["smooth"],
                simplify=params["simplify"],
                edge_mode=params["edge_mode"],
                repair=params["repair"],
                tile=params["tile"],
                margin=params["margin"],
                no_expand=params.get("no_expand") or None,
                expand_tol=params.get("expand_tol", 0.0),
                refs=params.get("refs") or None,
                workers=params["workers"],
                output_mode=params["output_mode"],
                layer="classes",
                skip_vector=False,
                resume=False,
                verbose=True)
        except Exception as e:
            print(f"  ! Ошибка обработки {os.path.basename(in_raster)}: {e}", flush=True)
            continue
        finally:
            if tmp_raster and out_raster and os.path.isfile(out_raster):
                try:
                    os.remove(out_raster)
                except OSError:
                    pass
        dt = time.time() - t0

        if not os.path.isfile(out_vector):
            print("  ! Вектор не создан.", flush=True)
            continue

        # промежуточные тайлы больше не нужны
        parts = os.path.splitext(out_vector)[0] + "_parts"
        if os.path.isdir(parts):
            shutil.rmtree(parts, ignore_errors=True)

        id_to_name = _read_class_table(os.path.dirname(in_raster))
        if id_to_name is None:
            id_to_name = _read_class_table(os.path.join(os.getcwd(), FINAL_BASE_NAME))
        if id_to_name:
            try:
                _add_class_names(out_vector, id_to_name)
                print("  ✓ Добавлено поле class_name", flush=True)
            except Exception as e:
                print(f"  ! Не удалось добавить class_name: {e}", flush=True)
        else:
            print("  • таблица классов не найдена — в векторе только class_value",
                  flush=True)

        with open(os.path.join(out_dir, "run_info.txt"), "a", encoding="utf-8") as f:
            f.write("07 - Generalization (sieve + marching squares)\n")
            f.write(f"Запуск       : {stamp}\n")
            f.write(f"Входной растр: {in_raster}\n")
            for key, val in params.items():
                f.write(f"  {key:12s}= {val}\n")
            f.write(f"Вектор       : {os.path.basename(out_vector)}\n")
            if out_raster and not tmp_raster:
                f.write(f"Растр        : {os.path.basename(out_raster)}\n")
            f.write(f"Время работы : {dt / 60:.1f} мин\n\n")

        created.append(out_vector)

    print(f"\n{line}\n  ГОТОВО: векторов {len(created)} из {len(files)}\n{line}",
          flush=True)
    for c in created:
        print(f"  {c}", flush=True)
    return created


# ===========================================================================
# ОКНО (tkinter)
# ===========================================================================

def _initial_dir(root_dir: str) -> str:
    """Самая свежая подпапка 04_FINAL, иначе 04_FINAL, иначе рабочая папка."""
    fb = os.path.join(root_dir, FINAL_BASE_NAME)
    if os.path.isdir(fb):
        subs = [os.path.join(fb, d) for d in os.listdir(fb)
                if os.path.isdir(os.path.join(fb, d))]
        return max(subs, key=os.path.getmtime) if subs else fb
    return root_dir


def _gui_dialog(root_dir: str):
    """Возвращает (files, params, out_base) или None, если окно закрыто."""
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    d = _default_params()
    result = {}
    files: list[str] = []
    last_dir = {"path": _initial_dir(root_dir)}

    win = tk.Tk()
    win.title("07 — Генерализация классифицированной карты")
    win.minsize(700, 0)
    win.attributes("-topmost", True)          # в Jupyter окно иначе бывает позади
    win.after(500, lambda: win.attributes("-topmost", False))
    pad = {"padx": 8, "pady": 4}

    # --- файлы ---
    fr_f = ttk.LabelFrame(win, text="Входные растры (*_cat.tif из 04_FINAL)")
    fr_f.pack(fill="both", expand=True, **pad)
    fr_lb = ttk.Frame(fr_f)
    fr_lb.pack(fill="both", expand=True, padx=6, pady=6)
    lb = tk.Listbox(fr_lb, selectmode="extended", height=7, width=95)
    sb = ttk.Scrollbar(fr_lb, orient="vertical", command=lb.yview)
    lb.configure(yscrollcommand=sb.set)
    lb.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")

    def add_files():
        paths = filedialog.askopenfilenames(
            parent=win, title="Выберите классифицированные растры",
            initialdir=last_dir["path"],
            filetypes=[("Классифицированные растры", "*_cat.tif"),
                       ("GeoTIFF", "*.tif *.tiff"), ("Все файлы", "*.*")])
        for p in paths:
            p = os.path.normpath(p)
            if p not in files:
                files.append(p)
                lb.insert("end", p)
        if paths:
            last_dir["path"] = os.path.dirname(paths[0])

    def remove_sel():
        for i in reversed(lb.curselection()):
            lb.delete(i)
            files.pop(i)

    def clear_all():
        lb.delete(0, "end")
        files.clear()

    fr_b = ttk.Frame(fr_f)
    fr_b.pack(fill="x", padx=6, pady=(0, 6))
    ttk.Button(fr_b, text="Добавить файлы…", command=add_files).pack(side="left")
    ttk.Button(fr_b, text="Удалить выбранные", command=remove_sel).pack(side="left", padx=6)
    ttk.Button(fr_b, text="Очистить", command=clear_all).pack(side="left")

    # --- параметры ---
    v = {
        "min_size": tk.StringVar(value=str(d["min_size"])),
        "connectivity": tk.IntVar(value=d["connectivity"]),
        "smooth": tk.StringVar(value=str(d["smooth"])),
        "simplify": tk.StringVar(value=str(d["simplify"])),
        "edge_mode": tk.StringVar(value=d["edge_mode"]),
        "repair": tk.BooleanVar(value=d["repair"]),
        "tile": tk.StringVar(value=str(d["tile"])),
        "margin": tk.StringVar(value=str(d["margin"])),
        "workers": tk.StringVar(value=str(d["workers"])),
        "output_mode": tk.StringVar(value=d["output_mode"]),
        "save_raster": tk.BooleanVar(value=d["save_raster"]),
        "no_expand": tk.StringVar(value=""),
        "expand_tol": tk.StringVar(value="0"),
    }

    def radio(parent, var, opts):
        f = ttk.Frame(parent)
        for txt, val in opts:
            ttk.Radiobutton(f, text=txt, variable=var, value=val).pack(side="left", padx=(0, 10))
        return f

    fr_s = ttk.LabelFrame(win, text="Sieve")
    fr_s.pack(fill="x", **pad)
    ttk.Label(fr_s, text="Мин. размер группы, пикс. (0/1 — без sieve)").grid(row=0, column=0, sticky="w", padx=6, pady=3)
    ttk.Spinbox(fr_s, from_=0, to=100000, textvariable=v["min_size"], width=10).grid(row=0, column=1, sticky="w", padx=6)
    ttk.Label(fr_s, text="Соседство").grid(row=1, column=0, sticky="w", padx=6, pady=3)
    radio(fr_s, v["connectivity"], [("4", 4), ("8", 8)]).grid(row=1, column=1, sticky="w", padx=6)

    fr_v = ttk.LabelFrame(win, text="Векторизация")
    fr_v.pack(fill="x", **pad)
    rows = [
        ("Сглаживание Чайкина, итераций (0 — выкл, обычно 1-2)",
         lambda p: ttk.Spinbox(p, from_=0, to=10, textvariable=v["smooth"], width=10)),
        ("Упрощение Douglas-Peucker, ед. CRS (0 — выкл)",
         lambda p: ttk.Entry(p, textvariable=v["simplify"], width=12)),
        ("Поведение на краю растра",
         lambda p: ttk.Combobox(p, textvariable=v["edge_mode"], values=["full", "shrink"],
                                state="readonly", width=10)),
        ("Сборка: patch — контуры, class — мультиполигон на класс, raw — без слияния",
         lambda p: ttk.Combobox(p, textvariable=v["output_mode"],
                                values=["patch", "class", "raw"], state="readonly", width=10)),
        ("Тайл, пикс. (с repair не выше 256-384)",
         lambda p: ttk.Spinbox(p, from_=64, to=4096, increment=64, textvariable=v["tile"], width=10)),
        ("Запас вокруг тайла, пикс. (>= 8 при сглаживании)",
         lambda p: ttk.Spinbox(p, from_=0, to=256, textvariable=v["margin"], width=10)),
        ("Процессов" + (" (в Jupyter надёжно только 1)" if _in_notebook() else ""),
         lambda p: ttk.Spinbox(p, from_=1, to=64, textvariable=v["workers"], width=10)),
        ("Классы без расширения границ (вода и т.п.), через запятую/пробел",
         lambda p: ttk.Entry(p, textvariable=v["no_expand"], width=24)),
        ("Допуск расширения, пикс. (0 — строго; 0.2-0.5 — глаже)",
         lambda p: ttk.Entry(p, textvariable=v["expand_tol"], width=12)),
    ]
    for r, (label, make) in enumerate(rows):
        ttk.Label(fr_v, text=label).grid(row=r, column=0, sticky="w", padx=6, pady=3)
        make(fr_v).grid(row=r, column=1, sticky="w", padx=6, pady=3)
    ttk.Checkbutton(fr_v, text="Убирать зазоры между полигонами (repair; дольше)",
                    variable=v["repair"]).grid(row=len(rows), column=0, columnspan=2,
                                               sticky="w", padx=6, pady=3)

    # --- негенерализуемые слои ---
    fr_ref = ttk.LabelFrame(
        win, text="Негенерализуемые слои (вода, болота): замещают классификацию, границы не меняются")
    fr_ref.pack(fill="x", **pad)
    ref_entries: list[dict] = []          # [{path, layer, class_value}]

    lb_ref = tk.Listbox(fr_ref, height=4, activestyle="dotbox")
    lb_ref.pack(fill="x", padx=6, pady=(6, 2))

    fr_rf1 = ttk.Frame(fr_ref)
    fr_rf1.pack(fill="x", padx=6)
    ttk.Label(fr_rf1, text="Файл").grid(row=0, column=0, sticky="w", pady=2)
    v_ref_path = tk.StringVar()
    ttk.Entry(fr_rf1, textvariable=v_ref_path, width=58).grid(
        row=0, column=1, columnspan=3, sticky="we", padx=4)
    def browse_ref():
        pth = filedialog.askopenfilename(
            parent=win, title="Слой негенерализуемых полигонов",
            filetypes=[("Векторные слои", "*.gpkg *.shp *.geojson *.fgb"),
                       ("Все файлы", "*.*")])
        if pth:
            v_ref_path.set(pth)
            try:
                import pyogrio
                layers = [l[0] for l in pyogrio.list_layers(pth)]
                if len(layers) == 1:
                    v_ref_layer.set(layers[0])
                cb_ref_layer.configure(values=layers)
            except Exception:
                pass
    ttk.Button(fr_rf1, text="Обзор…", command=browse_ref).grid(row=0, column=4, padx=4)

    ttk.Label(fr_rf1, text="Слой (GPKG)").grid(row=1, column=0, sticky="w", pady=2)
    v_ref_layer = tk.StringVar()
    cb_ref_layer = ttk.Combobox(fr_rf1, textvariable=v_ref_layer, width=24)
    cb_ref_layer.grid(row=1, column=1, sticky="w", padx=4)
    ttk.Label(fr_rf1, text="Номер класса").grid(row=1, column=2, sticky="e", padx=(10, 0))
    v_ref_cls = tk.StringVar()
    ttk.Entry(fr_rf1, textvariable=v_ref_cls, width=8).grid(row=1, column=3, sticky="w", padx=4)

    ttk.Label(fr_rf1, text="Стираемые классы растра").grid(
        row=2, column=0, sticky="w", pady=2)
    v_ref_erase = tk.StringVar()
    ttk.Entry(fr_rf1, textvariable=v_ref_erase, width=24).grid(
        row=2, column=1, sticky="w", padx=4)
    ttk.Label(fr_rf1, text="через запятую; пусто = только класс слоя").grid(
        row=2, column=2, columnspan=2, sticky="w", padx=(10, 0))

    def _ref_item(e):
        return (f"{os.path.basename(e['path'])}"
                + (f" | {e['layer']}" if e.get("layer") else "")
                + f" | класс {e['class_value']}"
                + (f" | стирает {','.join(str(int(c)) for c in e['erase'])}"
                   if e.get("erase") else ""))
    def _ref_refresh():
        lb_ref.delete(0, "end")
        for e in ref_entries:
            lb_ref.insert("end", _ref_item(e))
    def _ref_fields(e):
        v_ref_path.set(e["path"])
        v_ref_layer.set(e.get("layer") or "")
        v_ref_cls.set(str(e["class_value"]))
        v_ref_erase.set(",".join(str(int(c)) for c in e["erase"])
                        if e.get("erase") else "")
    def _ref_from_fields():
        pth = v_ref_path.get().strip()
        cls = v_ref_cls.get().strip()
        if not pth:
            messagebox.showwarning("Слой", "Выберите файл слоя.", parent=win)
            return None
        try:
            cv = int(float(cls))
        except ValueError:
            messagebox.showwarning("Слой", "Номер класса должен быть числом.",
                                   parent=win)
            return None
        erase = None
        raw_er = v_ref_erase.get().strip()
        if raw_er:
            try:
                erase = [int(x) for x in re.split(r"[,;\s]+", raw_er)
                         if x.strip()]
            except ValueError:
                messagebox.showwarning(
                    "Слой", "Стираемые классы — целые числа через запятую.",
                    parent=win)
                return None
        return {"path": pth, "layer": v_ref_layer.get().strip() or None,
                "class_value": cv, "erase": erase}
    def ref_add():
        e = _ref_from_fields()
        if e is None:
            return
        ref_entries.append(e)
        _ref_refresh()
        lb_ref.see("end")
        v_ref_path.set(""); v_ref_layer.set(""); v_ref_cls.set("")
        v_ref_erase.set("")
    def ref_change():
        sel = lb_ref.curselection()
        if not sel:
            return
        e = _ref_from_fields()
        if e is None:
            return
        ref_entries[sel[0]] = e
        _ref_refresh()
    def ref_del():
        for i in reversed(lb_ref.curselection()):
            ref_entries.pop(i)
        _ref_refresh()
    def ref_edit(_ev):
        sel = lb_ref.curselection()
        if sel:
            _ref_fields(ref_entries[sel[0]])
    lb_ref.bind("<Double-Button-1>", ref_edit)

    fr_rf2 = ttk.Frame(fr_ref)
    fr_rf2.pack(fill="x", padx=6, pady=(2, 6))
    ttk.Button(fr_rf2, text="Добавить ещё", command=ref_add).pack(side="left")
    ttk.Button(fr_rf2, text="Изменить", command=ref_change).pack(side="left", padx=6)
    ttk.Button(fr_rf2, text="Удалить выбранные", command=ref_del).pack(side="left")
    ttk.Label(fr_rf2, text="двойной щелчок — правка строки; верхний слой в списке главнее"
              ).pack(side="right")

    # --- вывод ---
    fr_o = ttk.LabelFrame(win, text="Результаты")
    fr_o.pack(fill="x", **pad)
    v_out = tk.StringVar(value=os.path.join(root_dir, OUT_BASE_NAME))
    ttk.Entry(fr_o, textvariable=v_out).grid(row=0, column=0, sticky="we", padx=6, pady=6)

    def choose_out():
        p = filedialog.askdirectory(parent=win, title="Папка результатов",
                                    initialdir=root_dir)
        if p:
            v_out.set(os.path.normpath(p))

    ttk.Button(fr_o, text="Обзор…", command=choose_out).grid(row=0, column=1, padx=6)
    ttk.Checkbutton(fr_o, text="Сохранить также очищенный растр (*_cat.tif)",
                    variable=v["save_raster"]).grid(row=1, column=0, columnspan=2,
                                                    sticky="w", padx=6, pady=(0, 6))
    fr_o.columnconfigure(0, weight=1)

    def on_run():
        if not files:
            messagebox.showwarning("Нет файлов", "Добавьте хотя бы один растр.", parent=win)
            return
        try:
            params = {
                "min_size": int(v["min_size"].get()),
                "connectivity": int(v["connectivity"].get()),
                "smooth": int(v["smooth"].get()),
                "simplify": max(float(v["simplify"].get().replace(",", ".")), 0.0),
                "edge_mode": v["edge_mode"].get(),
                "repair": bool(v["repair"].get()),
                "tile": int(v["tile"].get()),
                "margin": int(v["margin"].get()),
                "workers": max(int(v["workers"].get()), 1),
                "output_mode": v["output_mode"].get(),
                "save_raster": bool(v["save_raster"].get()),
                "no_expand": [int(x) for x in re.split(r"[,;\s]+", v["no_expand"].get())
                              if x.strip()],
                "expand_tol": max(float(v["expand_tol"].get().replace(",", ".")), 0.0),
                "refs": list(ref_entries),
            }
        except (ValueError, tk.TclError):
            messagebox.showerror("Ошибка",
                                 "Проверьте числовые параметры.\n"
                                 "Классы — целые числа через запятую, "
                                 "допуск — число пикселей.", parent=win)
            return
        result.update(files=list(files), params=params,
                      out_base=v_out.get().strip() or os.path.join(root_dir, OUT_BASE_NAME))
        win.destroy()

    fr_r = ttk.Frame(win)
    fr_r.pack(fill="x", **pad)
    ttk.Label(fr_r, text="Ход обработки выводится в консоль.").pack(side="left")
    ttk.Button(fr_r, text="Закрыть", command=win.destroy).pack(side="right")
    ttk.Button(fr_r, text="Запустить", command=on_run).pack(side="right", padx=6)

    win.mainloop()
    if not result:
        return None
    return result["files"], result["params"], result["out_base"]


# ===========================================================================
# КОНСОЛЬНЫЙ РЕЖИМ (нет tkinter или дисплея)
# ===========================================================================

def _console_dialog(root_dir: str):
    src_folder, _ = _select_prev_subdir(os.path.join(root_dir, FINAL_BASE_NAME))
    files = [_select_cat_raster(src_folder)]
    d = _default_params()
    p = dict(d)
    p["min_size"] = _prompt_int("min_size (0/1 = без sieve)", d["min_size"])
    p["connectivity"] = _prompt_int("connectivity (4 или 8)", d["connectivity"])
    p["smooth"] = _prompt_int("smooth (итераций Чайкина)", d["smooth"])
    p["simplify"] = _prompt_float("simplify (Douglas-Peucker, 0 = выкл)", d["simplify"])
    p["repair"] = _prompt_yes("repair (без зазоров; дольше)", d["repair"])
    p["workers"] = _prompt_int("workers (процессов)", d["workers"])
    raw = input("  классы без расширения границ (вода и т.п.; через запятую, "
                "Enter = нет): ").strip()
    p["no_expand"] = [int(x) for x in re.split(r"[,;\s]+", raw) if x.strip()]
    p["expand_tol"] = _prompt_float("expand_tol (допуск расширения, пикс.; 0 = строго)",
                                    d["expand_tol"])
    refs = []
    while True:
        raw = input("  негенерализуемый слой: файл (Enter = больше не добавлять): ").strip()
        if not raw:
            break
        layer = input("    имя слоя в файле (Enter = первый/единственный): ").strip()
        cls = input("    номер класса в классификации (например 6): ").strip()
        raw_er = input("    стираемые классы растра через запятую "
                       "(Enter = только класс слоя): ").strip()
        try:
            cv = int(float(cls))
        except ValueError:
            print("    ! номер класса должен быть числом — слой пропущен")
            continue
        erase = None
        if raw_er:
            try:
                erase = [int(x) for x in re.split(r"[,;\s]+", raw_er)
                         if x.strip()]
            except ValueError:
                print("    ! стираемые классы должны быть числами — "
                      "стирается только класс слоя")
        refs.append({"path": raw, "layer": layer or None,
                     "class_value": cv, "erase": erase})
    p["refs"] = refs
    p["save_raster"] = _prompt_yes("сохранить очищенный растр", False)
    return files, p, os.path.join(root_dir, OUT_BASE_NAME)


def _run_interactive():
    root_dir = os.getcwd()
    line = "=" * 60
    print(f"\n{line}\n  07 - ГЕНЕРАЛИЗАЦИЯ (sieve + marching squares)\n{line}", flush=True)
    print(f"Рабочая папка: {root_dir}", flush=True)
    try:
        choice = _gui_dialog(root_dir)
        if choice is None:
            print("Окно закрыто без запуска.", flush=True)
            return
    except Exception as e:  # нет tkinter / дисплея
        print(f"Графический интерфейс недоступен ({e}). Консольный режим.", flush=True)
        choice = _console_dialog(root_dir)

    files, params, out_base = choice
    print(f"\nФайлов: {len(files)}", flush=True)
    for key, val in params.items():
        print(f"  {key:12s}= {val}", flush=True)
    print(f"Папка результатов: {out_base}", flush=True)
    generalize_files(files, params, out_base)


# ===========================================================================
# MAIN
# ===========================================================================

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    # запуск без аргументов (кнопка Run, %run в Jupyter) — окно
    if not argv:
        _run_interactive()
        return

    p = argparse.ArgumentParser(
        description="Генерализация: быстрый sieve + тайловая векторизация marching squares")
    p.add_argument("input")
    p.add_argument("--min-size", type=int, default=3)
    p.add_argument("--connectivity", type=int, choices=[4, 8], default=8)
    p.add_argument("--out-raster", default=None)
    p.add_argument("--out-vector", default=None)
    p.add_argument("--smooth", type=int, default=0,
                   help="итераций Чайкина (0 = выкл, 1-2 достаточно)")
    p.add_argument("--simplify", type=float, default=0.0)
    p.add_argument("--edge-mode", choices=["full", "shrink"], default="full")
    p.add_argument("--no-repair", action="store_true",
                   help="не приводить к разбиению без зазоров внутри тайла")
    p.add_argument("--tile", type=int, default=256,
                   help="сторона тайла в пикселях; с repair 256 px ~ 4 c, 512 px ~ 46 c")
    p.add_argument("--margin", type=int, default=16,
                   help="запас вокруг тайла в пикселях (>= 8 при smooth > 0)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output-mode", choices=["patch", "class", "raw"], default="patch")
    p.add_argument("--no-expand", default="",
                   help="классы, границы которых нельзя расширять (через запятую), "
                        "например 6 или 6,12 — обычно вода")
    p.add_argument("--expand-tol", type=float, default=0.0,
                   help="допуск расширения в пикселях (0 = строго; 0.2-0.5 — глаже)")
    p.add_argument("--ref", action="append", default=[],
                   metavar="ФАЙЛ|СЛОЙ|КЛАСС|СТИРАЕМЫЕ",
                   help="негенерализуемый слой: замещает классификацию, границы не "
                        "меняются (можно несколько раз; СЛОЙ можно опустить: "
                        "ФАЙЛ|КЛАСС; СТИРАЕМЫЕ — классы растра, чьи пиксели "
                        "передаются соседям до генерализации, через запятую: "
                        "ФАЙЛ|СЛОЙ|400|1400,2400,3400,4400)")
    p.add_argument("--layer", default="classes")
    p.add_argument("--skip-vector", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="не удалять уже посчитанные тайлы")
    p.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args(argv)

    if not a.out_raster and not a.out_vector:
        p.error("укажите --out-raster и/или --out-vector")

    no_expand = [int(x) for x in re.split(r"[,;\s]+", a.no_expand) if x.strip()]
    refs = [_parse_ref_spec(spec) for spec in a.ref]
    run(a.input, a.min_size, a.connectivity, a.out_raster, a.out_vector,
        a.smooth, a.simplify, a.edge_mode, not a.no_repair, a.tile, a.margin,
        no_expand, a.expand_tol, refs,
        a.workers, a.output_mode, a.layer, a.skip_vector, a.resume,
        verbose=not a.quiet)


if __name__ == "__main__":
    main()
