# -*- coding: utf-8 -*-
"""
10_download_sat_chips.py
========================
Выгрузка фрагментов спутниковых подложек высокого разрешения (Google / Bing /
Yandex / ESRI World Imagery) вокруг точек: окружность радиусом RADIUS_M,
каждый фрагмент — отдельный лёгкий файл (GeoTIFF с JPEG-сжатием, либо
JPG/PNG/WebP + world-file). Предназначено для последующей сегментации SAM
(segment-geospatial / samgeo, SAM2 и т.п.).

Что делает:
  1. Окно Проводника (мультивыбор) -> файлы точек: CSV / XLSX / GPKG / SHP /
     GeoJSON / KML. Для CSV/XLSX столбцы lon/lat определяются автоматически.
  2. (опционально) отбрасывает точки вне AOI: ассет GEE
     projects/steel-capsule-438113-h8/assets/MO_AOI26 (через earthengine-api)
     или локальный файл AOI.
  3. Группирует близкие точки (GROUP_MODE = "merge"): точки ближе GROUP_LINK_M
     друг к другу попадают в ОДИН общий фрагмент, который охватывает круги
     RADIUS_M всех точек группы (у каждой точки остаётся не меньше RADIUS_M
     окружения). Длинная сторона общего фрагмента не больше MAX_CHIP_M:
     крупные скопления делятся на части так, чтобы суммарная площадь была
     минимальной. Одиночная точка -> обычный фрагмент 2R x 2R.
  4. Для каждого фрагмента: круги R м (в локальной азимутальной проекции ->
     точный радиус в метрах), подбор XYZ-тайлов на уровне ZOOM, загрузка с
     кэшем (SQLite, повторно не качаются), сшивка, обрезка по прямоугольнику,
     описанному вокруг кругов, (опц.) маска вне кругов, (опц.) перепроецирование.
  5. Пишет фрагменты + таблицы:
       chips_index.gpkg / .csv  — фрагменты (охват, файл, точки, размер пикселя,
                                  доля пропущенных тайлов); геометрия — охват;
       chips_points.gpkg / .csv — все точки с chip_id своего фрагмента и
                                  исходными атрибутами (для 11_ и дальнейшей работы).

Имена фрагментов: одиночная точка -> её ID (напр. 108.tif); общий
фрагмент -> ID первой точки группы + число точек (напр. 108_7p.tif).
Готовый фрагмент не перекачивается, если состав его точек и охват не
изменились (ключ key в chips_index).

Размеры (R = 500 м, квадрат 1x1 км, широта ~55.7°):
  zoom 17 -> ~0.67 м/пикс, ~1500x1500 px, ~0.4-0.8 МБ
  zoom 18 -> ~0.34 м/пикс, ~3000x3000 px, ~1.5-3 МБ   (по умолчанию)
  zoom 19 -> ~0.17 м/пикс, ~6000x6000 px, ~5-10 МБ (не везде есть данные)

ВНИМАНИЕ: условия использования Google / Bing / Yandex запрещают массовую
загрузку тайлов вне их API. Для научной работы безопаснее ESRI World Imagery
(с атрибуцией) или ограниченные объёмы. Ответственность — на пользователе.

Зависимости: pip install geopandas rasterio requests pillow shapely pyproj
             (опционально) earthengine-api openpyxl
"""

import io
import math
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from PIL import Image
from pyproj import CRS, Transformer
from shapely.geometry import Point
from shapely.ops import transform as shp_transform

import rasterio
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling

# ============================================================================
# НАСТРОЙКИ
# ============================================================================
PROVIDER = "google"        # "google" | "bing" | "yandex" | "esri"
ZOOM = 18                  # 17 / 18 / 19
RADIUS_M = 500.0           # радиус окружности, м

OUT_DIR = None             # None -> папка "sat_chips_<provider>_z<zoom>" рядом с файлом точек
OUT_FORMAT = "gtiff"       # "gtiff" (JPEG-сжатие, гео-привязка внутри; лучший вариант для samgeo)
                           # "jpg" | "png" | "webp" (+ world-file .jgw/.pgw/.wld и .prj)
JPEG_QUALITY = 90

CIRCLE_MASK = False        # True -> пиксели вне кругов = 0 (+ маска/альфа). Для SAM обычно
                           # лучше False (квадрат без "чёрных углов"), а круг применять к маскам SAM.
REPROJECT_EPSG = None      # None -> оставить Web Mercator (3857, для Yandex — 3395);
                           # "utm" -> авто-UTM по точке; или код, напр. 32637

# AOI-фильтр (None, "gee" или путь к файлу GPKG/SHP/GeoJSON)
AOI_SOURCE = None
GEE_PROJECT = "steel-capsule-438113-h8"
GEE_AOI_ASSET = "projects/steel-capsule-438113-h8/assets/MO_AOI26"

SHEET_NAME = None          # для XLSX: None -> первый лист с координатами (у вас это "Class_all")
ID_FIELD = None            # None -> авто (id / point_id / NAME... ), иначе имя столбца
GROUP_MODE = "merge"       # "merge": близкие точки -> один общий фрагмент (рекомендуется);
                           # "dedup": свой фрагмент 2R x 2R на каждое место, объединяются
                           # только точки ближе DEDUP_M
GROUP_LINK_M = 500.0       # merge: точки ближе этого расстояния (цепочкой) -> одна группа
MAX_CHIP_M = 2000.0        # merge: наибольшая сторона общего фрагмента, м (больше -> делится)
DEDUP_M = 25.0             # dedup: точки ближе DEDUP_M м -> один фрагмент
VALID_BBOX = (33.0, 53.5, 42.0, 57.5)  # lon_min, lat_min, lon_max, lat_max: отсев ошибочных
                           # координат (0, 535 и т.п.); None -> без проверки
TEST_N = 20                # ТЕСТОВЫЙ ЗАПУСК: 20 случайных мест; после проверки -> 0 (все).
                           # Готовые фрагменты при полном запуске не перекачиваются.
SKIP_EXISTING = True       # не пересоздавать готовые фрагменты
N_THREADS = 6              # параллельные загрузки тайлов
REQUEST_DELAY = 0.02       # пауза между запросами в потоке, с
TIMEOUT = 20
RETRIES = 3

# ============================================================================
# ПРОВАЙДЕРЫ
# ============================================================================
HALF = 20037508.342789244          # половина длины экватора Меркатора, м
TILE = 256
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

PROVIDERS = {
    "google": dict(url="https://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                   subs="0123", epsg=3857),
    "bing":   dict(url="https://ecn.t{s}.tiles.virtualearth.net/tiles/a{q}.jpeg?g=14500",
                   subs="0123", epsg=3857),
    # Yandex использует эллипсоидальный Меркатор EPSG:3395
    "yandex": dict(url="https://core-sat.maps.yandex.net/tiles?l=sat&v=3.1&x={x}&y={y}&z={z}&lang=ru_RU",
                   subs="", epsg=3395),
    "esri":   dict(url="https://server.arcgisonline.com/ArcGIS/rest/services/"
                       "World_Imagery/MapServer/tile/{z}/{y}/{x}",
                   subs="", epsg=3857),
}


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
                 "segment-geospatial", "sam2", "segment-anything"):
        try:
            lines.append(f"  {dist} {metadata.version(dist)}")
        except Exception:
            pass
    return "\n".join(lines)


def settings_info(names):
    g = globals()
    return "\n".join(f"  {n} = {g[n]!r}" for n in names if n in g)


def quadkey(x, y, z):
    q = []
    for i in range(z, 0, -1):
        d = 0
        m = 1 << (i - 1)
        if x & m:
            d += 1
        if y & m:
            d += 2
        q.append(str(d))
    return "".join(q)


def tile_url(provider, x, y, z):
    p = PROVIDERS[provider]
    s = p["subs"][(x + y) % len(p["subs"])] if p["subs"] else ""
    return p["url"].format(s=s, x=x, y=y, z=z, q=quadkey(x, y, z))


# ============================================================================
# КЭШ ТАЙЛОВ (SQLite, формат как в MBTiles, но с XYZ-нумерацией)
# ============================================================================
class TileCache:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.execute("CREATE TABLE IF NOT EXISTS tiles (z INT, x INT, y INT, "
                         "data BLOB, PRIMARY KEY (z, x, y))")
        self.con.commit()

    def get(self, z, x, y):
        with self.lock:
            r = self.con.execute("SELECT data FROM tiles WHERE z=? AND x=? AND y=?",
                                 (z, x, y)).fetchone()
        return r[0] if r else None

    def put(self, z, x, y, data):
        with self.lock:
            self.con.execute("INSERT OR REPLACE INTO tiles VALUES (?,?,?,?)",
                             (z, x, y, sqlite3.Binary(data)))
            self.con.commit()


_local = threading.local()


def _session():
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.headers.update({"User-Agent": UA, "Accept": "image/*,*/*;q=0.8"})
        _local.s = s
    return _local.s


def fetch_tile(provider, cache, x, y, z):
    """Возвращает RGB-массив 256x256x3 или None (тайл недоступен)."""
    data = cache.get(z, x, y)
    if data is None:
        url = tile_url(provider, x, y, z)
        for attempt in range(RETRIES):
            try:
                r = _session().get(url, timeout=TIMEOUT)
                if r.status_code == 200 and r.content and \
                        r.headers.get("Content-Type", "image").startswith("image"):
                    data = r.content
                    cache.put(z, x, y, data)
                    break
                if r.status_code in (403, 404, 204):
                    break
            except requests.RequestException:
                pass
            time.sleep(1.0 * (attempt + 1))
        time.sleep(REQUEST_DELAY)
    if data is None:
        return None
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
        if im.size != (TILE, TILE):
            im = im.resize((TILE, TILE))
        return np.asarray(im)
    except Exception:
        return None


# ============================================================================
# ВХОДНЫЕ ДАННЫЕ
# ============================================================================
LON_NAMES = ["lon", "lng", "long", "longitude", "x", "x_wgs84", "lon_wgs84", "долгота", "e", "east"]
LAT_NAMES = ["lat", "latitude", "y", "y_wgs84", "lat_wgs84", "широта", "n", "north"]
ID_NAMES = ["id", "point_id", "pid", "fid", "name", "код", "номер", "point", "polygon_id"]


def pick_files():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        files = filedialog.askopenfilenames(
            title="Выберите файлы точек",
            filetypes=[("Точки", "*.csv *.xlsx *.xls *.gpkg *.shp *.geojson *.json *.kml"),
                       ("Все файлы", "*.*")])
        root.destroy()
        return list(files)
    except Exception as e:
        print(f"[!] Окно выбора недоступно ({e}). Укажите пути через аргументы.")
        return sys.argv[1:]


def _find_col(cols, names, prefix=False):
    low = {str(c).lower().strip(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    if prefix:                       # "NAME РЫЖИМ_2010-2026" -> name
        for n in names:
            for lc, c in low.items():
                if lc.startswith(n + " ") or lc.startswith(n + "_"):
                    return c
    return None


def _reject(df, bad, path):
    """Строки без координат / вне VALID_BBOX -> rejected_points.csv рядом с файлом точек."""
    print(f"  [!] {os.path.basename(path)}: {int(bad.sum())} строк без координат или вне "
          f"VALID_BBOX -> rejected_points.csv")
    rp = os.path.join(os.path.dirname(os.path.abspath(path)), "rejected_points.csv")
    rej = pd.DataFrame(df[bad]).copy()
    if "geometry" in rej:
        rej["geometry"] = rej["geometry"].astype(str)
    rej.assign(src_file=os.path.basename(path)).to_csv(
        rp, mode="a", index=False, encoding="utf-8-sig", header=not os.path.exists(rp))


def read_points(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".txt", ".xlsx", ".xls"):
        if ext in (".xlsx", ".xls"):
            if SHEET_NAME is not None:
                df = pd.read_excel(path, sheet_name=SHEET_NAME)
            else:
                df = None
                for sh, d in pd.read_excel(path, sheet_name=None).items():
                    if _find_col(d.columns, LON_NAMES) and _find_col(d.columns, LAT_NAMES):
                        df = d
                        print(f"  {os.path.basename(path)}: лист '{sh}'")
                        break
                if df is None:
                    raise ValueError(f"{path}: нет листа со столбцами lon/lat")
        else:
            df = pd.read_csv(path, sep=None, engine="python", encoding_errors="replace")
        cx, cy = _find_col(df.columns, LON_NAMES), _find_col(df.columns, LAT_NAMES)
        if cx is None or cy is None:
            raise ValueError(f"{path}: не найдены столбцы lon/lat, есть {list(df.columns)}")
        for c in (cx, cy):
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
        lon, lat = df[cx], df[cy]
    else:
        df = gpd.read_file(path)
        if df.crs is None:
            df = df.set_crs(4326)
        df = df[df.geometry.notna() & ~df.geometry.is_empty].to_crs(4326)
        df["geometry"] = df.geometry.representative_point()  # полигоны -> точка
        lon, lat = df.geometry.x, df.geometry.y
    bad = lon.isna() | lat.isna()
    if VALID_BBOX is not None:
        x0, y0, x1, y1 = VALID_BBOX
        bad |= ~lon.between(x0, x1) | ~lat.between(y0, y1)
    if bad.any():
        _reject(df, bad, path)
    df, lon, lat = df[~bad], lon[~bad], lat[~bad]
    gdf = gpd.GeoDataFrame(pd.DataFrame(df).drop(columns="geometry", errors="ignore"),
                           geometry=gpd.points_from_xy(lon, lat), crs=4326)
    gdf["src_file"] = os.path.basename(path)
    return gdf


def assign_ids(gdf):
    col = ID_FIELD or _find_col(gdf.columns, ID_NAMES, prefix=True)
    if col:
        ids = gdf[col].astype(str)
    else:
        ids = pd.Series([f"pt{i:05d}" for i in range(len(gdf))], index=gdf.index)
    ids = ids.str.replace(r"[^\w\-.]+", "_", regex=True)
    # уникальность при совпадениях
    dup = ids.duplicated(keep=False)
    if dup.any():
        ids[dup] = ids[dup] + "_" + ids[dup].groupby(ids[dup]).cumcount().astype(str)
    gdf["point_id"] = ids.values
    return gdf


# ============================================================================
# ГРУППИРОВКА ТОЧЕК ВО ФРАГМЕНТЫ
# ============================================================================
def _components(xy, link):
    """Группы точек, связанных цепочкой расстояний <= link (м)."""
    n = len(xy)
    par = list(range(n))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    grid = {}
    for i, p in enumerate(xy):
        c = (int(p[0] // link), int(p[1] // link))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((c[0] + dx, c[1] + dy), ()):
                    if math.hypot(*(xy[j] - p)) <= link:
                        a, b = find(i), find(j)
                        if a != b:
                            par[a] = b
        grid.setdefault(c, []).append(i)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [np.array(v) for v in groups.values()]


def _extent(xy, idx, r):
    p = xy[idx]
    w = p[:, 0].max() - p[:, 0].min() + 2 * r
    h = p[:, 1].max() - p[:, 1].min() + 2 * r
    return w * h, max(w, h)


def _best_split(xy, idx, r):
    """Разрез группы по X или Y с минимальной суммарной площадью двух охватов."""
    best = None
    for ax in (0, 1):
        o = idx[np.argsort(xy[idx, ax], kind="stable")]
        px, py = xy[o, 0], xy[o, 1]
        acc = lambda f, v: f.accumulate(v)
        lw = acc(np.maximum, px) - acc(np.minimum, px) + 2 * r
        lh = acc(np.maximum, py) - acc(np.minimum, py) + 2 * r
        rw = (acc(np.maximum, px[::-1]) - acc(np.minimum, px[::-1]))[::-1] + 2 * r
        rh = (acc(np.maximum, py[::-1]) - acc(np.minimum, py[::-1]))[::-1] + 2 * r
        tot = lw[:-1] * lh[:-1] + rw[1:] * rh[1:]          # разрез перед элементом k = 1..n-1
        v = xy[o, ax]
        tot[v[1:] == v[:-1]] = np.inf                       # не резать между равными координатами
        k = int(np.argmin(tot))
        if np.isfinite(tot[k]) and (best is None or tot[k] < best[0]):
            best = (tot[k], o[:k + 1], o[k + 1:])
    return best


def _split_groups(xy, idx, r, max_side, gain=0.9):
    """Делит группу, пока сторона охвата > max_side или разрез экономит > 10% площади."""
    out, stack = [], [idx]
    while stack:
        g = stack.pop()
        if len(g) == 1:
            out.append(g)
            continue
        area, side = _extent(xy, g, r)
        bs = _best_split(xy, g, r)
        if bs is not None and (side > max_side or bs[0] < gain * area):
            stack += [bs[2], bs[1]]
        else:
            out.append(g)
    return out


def _dedup_groups(xy, dist):
    """Режим dedup: точки не дальше dist от первой точки группы (без цепочек)."""
    if not dist or dist <= 0:
        return [np.array([i]) for i in range(len(xy))]
    leaders, grid, group = [], {}, {}
    for i, p in enumerate(xy):
        cx, cy = (p // dist).astype(int)
        best, bd = None, dist
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for L in grid.get((cx + dx, cy + dy), []):
                    d = np.hypot(*(xy[L] - p))
                    if d <= bd:
                        best, bd = L, d
        if best is None:
            best = i
            leaders.append(i)
            grid.setdefault((cx, cy), []).append(i)
        group.setdefault(best, []).append(i)
    return [np.array(group[L]) for L in leaders]


def plan_chips(pts):
    """Точки -> фрагменты. Возвращает (chips: GeoDataFrame по фрагменту, pts с chip_id)."""
    import hashlib
    utm = pts.estimate_utm_crs()
    m = pts.to_crs(utm)
    xy = np.c_[m.geometry.x.values, m.geometry.y.values]
    if GROUP_MODE == "merge":
        max_side = max(float(MAX_CHIP_M), 2 * RADIUS_M)
        groups = []
        for comp in _components(xy, GROUP_LINK_M):
            groups += _split_groups(xy, comp, RADIUS_M, max_side)
    elif GROUP_MODE == "dedup":
        groups = _dedup_groups(xy, DEDUP_M)
    else:
        raise ValueError('GROUP_MODE должен быть "merge" или "dedup"')
    groups = sorted((np.sort(g) for g in groups), key=lambda g: g[0])
    pid = pts["point_id"].values
    src = pts["src_file"].values
    to_ll = Transformer.from_crs(utm, 4326, always_xy=True).transform
    recs, chip_of, used = [], np.empty(len(pts), dtype=object), set()
    for g in groups:
        cid = pid[g[0]] if len(g) == 1 else f"{pid[g[0]]}_{len(g)}p"
        base, k = cid, 1
        while cid in used:
            cid, k = f"{base}_{k}", k + 1
        used.add(cid)
        chip_of[g] = cid
        p = xy[g]
        x0, y0 = p.min(0) - RADIUS_M
        x1, y1 = p.max(0) + RADIUS_M
        lon, lat = to_ll((x0 + x1) / 2, (y0 + y1) / 2)
        members = ";".join(pid[g])
        key = hashlib.md5(f"{PROVIDER}|{ZOOM}|{RADIUS_M}|{REPROJECT_EPSG}|{members}"
                          .encode("utf-8")).hexdigest()[:10]
        recs.append(dict(chip_id=cid, n_points=len(g), members=members,
                         src_file=";".join(dict.fromkeys(src[g])),
                         lon=lon, lat=lat, width_m=round(x1 - x0, 1),
                         height_m=round(y1 - y0, 1), key=key, idx_pts=g))
    chips = pd.DataFrame(recs)
    pts = pts.copy()
    pts["chip_id"] = chip_of
    return chips, pts


def load_aoi():
    if AOI_SOURCE is None:
        return None
    if AOI_SOURCE == "gee":
        import ee
        try:
            ee.Initialize(project=GEE_PROJECT)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=GEE_PROJECT)
        fc = ee.FeatureCollection(GEE_AOI_ASSET)
        gj = fc.geometry().getInfo()
        from shapely.geometry import shape
        return gpd.GeoSeries([shape(gj)], crs=4326).union_all()
    aoi = gpd.read_file(AOI_SOURCE).to_crs(4326)
    return aoi.union_all()


# ============================================================================
# ГЕОМЕТРИЯ ФРАГМЕНТА
# ============================================================================
def circle_in_crs(lon, lat, r, epsg):
    """Круг радиусом r (м) в локальной AEQD-проекции -> в CRS тайлов."""
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m")
    tr = Transformer.from_crs(aeqd, CRS.from_epsg(epsg), always_xy=True).transform
    return shp_transform(tr, Point(0, 0).buffer(r, 128))


def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def chip_geom(lons, lats, r, epsg):
    """Объединение кругов r (м) вокруг всех точек фрагмента -> в CRS тайлов."""
    from shapely.ops import unary_union
    return unary_union([circle_in_crs(lo, la, r, epsg) for lo, la in zip(lons, lats)])


def chip_window(geom, z):
    """Окно фрагмента в сетке тайлов: тайлы, пиксельная обрезка, привязка, охват."""
    minx, miny, maxx, maxy = geom.bounds
    tsize = 2 * HALF / (1 << z)          # размер тайла, м
    res = tsize / TILE                   # размер пикселя (в единицах проекции)
    tx0 = int(math.floor((minx + HALF) / tsize))
    tx1 = int(math.floor((maxx + HALF) / tsize))
    ty0 = int(math.floor((HALF - maxy) / tsize))
    ty1 = int(math.floor((HALF - miny) / tsize))
    ox, oy = -HALF + tx0 * tsize, HALF - ty0 * tsize
    c0 = int(math.floor((minx - ox) / res)); c1 = int(math.ceil((maxx - ox) / res))
    r0 = int(math.floor((oy - maxy) / res)); r1 = int(math.ceil((oy - miny) / res))
    bounds = (ox + c0 * res, oy - r1 * res, ox + c1 * res, oy - r0 * res)
    return dict(tx0=tx0, tx1=tx1, ty0=ty0, ty1=ty1, ox=ox, oy=oy, res=res,
                c0=c0, c1=c1, r0=r0, r1=r1, bounds=bounds, w=c1 - c0, h=r1 - r0)


def build_chip(provider, cache, geom, z, pool):
    epsg = PROVIDERS[provider]["epsg"]
    wd = chip_window(geom, z)
    tx0, tx1, ty0, ty1 = wd["tx0"], wd["tx1"], wd["ty0"], wd["ty1"]
    nx, ny = tx1 - tx0 + 1, ty1 - ty0 + 1
    mosaic = np.zeros((ny * TILE, nx * TILE, 3), dtype=np.uint8)
    jobs = {(tx, ty): pool.submit(fetch_tile, provider, cache, tx, ty, z)
            for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)}
    missing = 0
    for (tx, ty), f in jobs.items():
        arr = f.result()
        if arr is None:
            missing += 1
            continue
        i, j = (ty - ty0) * TILE, (tx - tx0) * TILE
        mosaic[i:i + TILE, j:j + TILE] = arr
    # обрезка до прямоугольника, описанного вокруг кругов
    img = mosaic[wd["r0"]:wd["r1"], wd["c0"]:wd["c1"]]
    res = wd["res"]
    transform = from_origin(wd["ox"] + wd["c0"] * res, wd["oy"] - wd["r0"] * res, res, res)
    return img, transform, epsg, geom, missing / len(jobs)


def maybe_reproject(img, transform, src_epsg, lon, lat, geom):
    if REPROJECT_EPSG is None:
        return img, transform, src_epsg, geom
    dst_epsg = utm_epsg(lon, lat) if REPROJECT_EPSG == "utm" else int(REPROJECT_EPSG)
    h, w = img.shape[:2]
    left, top = transform.c, transform.f
    right, bottom = left + w * transform.a, top + h * transform.e
    dst_t, dw, dh = calculate_default_transform(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}",
                                                w, h, left, bottom, right, top)
    out = np.zeros((dh, dw, 3), dtype=np.uint8)
    for b in range(3):
        reproject(img[:, :, b], out[:, :, b], src_transform=transform,
                  src_crs=f"EPSG:{src_epsg}", dst_transform=dst_t,
                  dst_crs=f"EPSG:{dst_epsg}", resampling=Resampling.bilinear)
    tr = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True).transform
    return out, dst_t, dst_epsg, shp_transform(tr, geom)


# ============================================================================
# ЗАПИСЬ
# ============================================================================
def write_chip(path, img, transform, epsg, geom):
    h, w = img.shape[:2]
    inside = None
    if CIRCLE_MASK:
        inside = ~geometry_mask([geom], out_shape=(h, w), transform=transform)
        img = img.copy()
        img[~inside] = 0

    if OUT_FORMAT == "gtiff":
        prof = dict(driver="GTiff", height=h, width=w, count=3, dtype="uint8",
                    crs=f"EPSG:{epsg}", transform=transform, compress="JPEG",
                    jpeg_quality=JPEG_QUALITY, photometric="YCbCr",
                    tiled=True, blockxsize=512, blockysize=512, interleave="pixel")
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(path, "w", **prof) as dst:
                dst.write(np.moveaxis(img, 2, 0))
                if inside is not None:
                    dst.write_mask(inside.astype(np.uint8) * 255)
        return

    # JPG / PNG / WebP + world-file + .prj
    if OUT_FORMAT == "png" and inside is not None:
        Image.fromarray(np.dstack([img, inside.astype(np.uint8) * 255]), "RGBA").save(path, optimize=True)
    elif OUT_FORMAT == "png":
        Image.fromarray(img).save(path, optimize=True)
    elif OUT_FORMAT == "webp":
        Image.fromarray(img).save(path, quality=JPEG_QUALITY, method=6)
    else:
        Image.fromarray(img).save(path, quality=JPEG_QUALITY, optimize=True)
    wext = {"jpg": ".jgw", "png": ".pgw", "webp": ".wld"}[OUT_FORMAT]
    a, e = transform.a, transform.e
    cx, cy = transform.c + a / 2, transform.f + e / 2      # центр левого верхнего пикселя
    with open(os.path.splitext(path)[0] + wext, "w") as f:
        f.write(f"{a:.10f}\n0.0\n0.0\n{e:.10f}\n{cx:.6f}\n{cy:.6f}\n")
    with open(os.path.splitext(path)[0] + ".prj", "w") as f:
        f.write(CRS.from_epsg(epsg).to_wkt(version="WKT1_ESRI"))


# ============================================================================
# MAIN
# ============================================================================
def _footprint(bounds, epsg):
    """Охват фрагмента (прямоугольник) -> полигон WGS84."""
    from shapely.geometry import box
    return gpd.GeoSeries([box(*bounds).segmentize(max(bounds[2] - bounds[0], 1) / 8)],
                         crs=epsg).to_crs(4326).iloc[0]


def _file_info(fn, lat):
    """Размер (px), пиксель на местности (м), охват и EPSG готового файла."""
    try:
        with rasterio.open(fn) as src:
            epsg = src.crs.to_epsg() if src.crs else None
            k = math.cos(math.radians(lat)) if epsg in (3857, 3395, None) else 1.0
            return (f"{src.width}x{src.height}", round(src.transform.a * k, 3),
                    tuple(src.bounds), epsg)
    except Exception:
        return None, None, None, None


def _chip_is_current(fn, c, old_keys, geom):
    """Готовый фрагмент годится, если не изменились состав точек и охват."""
    if not os.path.exists(fn):
        return False
    k = old_keys.get(c.chip_id)
    if isinstance(k, str) and k:
        return k == c.key
    # фрагменты прежних версий (без ключа): сверка охвата
    if OUT_FORMAT == "gtiff" and REPROJECT_EPSG is None:
        wd = chip_window(geom, ZOOM)
        _, _, b, _ = _file_info(fn, c.lat)
        return b is not None and all(abs(u - v) < 1.5 * wd["res"]
                                     for u, v in zip(b, wd["bounds"]))
    return True


def _clean_attrs(df):
    """Исходные атрибуты точек для GPKG: без конфликтующих имён, смешанные типы -> текст."""
    reserved = {"point_id", "chip_id", "src_file", "lon", "lat", "geometry", "fid"}
    out, seen = pd.DataFrame(index=df.index), set()
    for c in df.columns:
        name = str(c)
        if name.lower() in reserved:
            name = f"{name}_src"
        while name.lower() in seen or name.lower() in reserved:
            name += "_"
        seen.add(name.lower())
        v = df[c]
        if v.dtype == object:
            v = v.map(lambda x: x if x is None or (isinstance(x, float) and math.isnan(x))
                      else str(x))
        out[name] = v
    return out


def main():
    if PROVIDER not in PROVIDERS:
        raise ValueError(f"PROVIDER должен быть одним из {list(PROVIDERS)}")
    files = pick_files()
    if not files:
        print("Файлы не выбраны."); return

    out_dir = OUT_DIR or os.path.join(os.path.dirname(os.path.abspath(files[0])),
                                      f"sat_chips_{PROVIDER}_z{ZOOM}")
    log_path, stamp = start_log(os.path.join(out_dir, "_logs"), "10_download")
    print("=" * 78)
    print("10_download_sat_chips.py" + ("  [ТЕСТОВЫЙ ЗАПУСК]" if TEST_N else ""))
    print("=" * 78)
    print(system_info())
    print("Настройки:")
    print(settings_info(["PROVIDER", "ZOOM", "RADIUS_M", "OUT_FORMAT", "JPEG_QUALITY",
                         "CIRCLE_MASK", "REPROJECT_EPSG", "AOI_SOURCE", "SHEET_NAME", "ID_FIELD",
                         "GROUP_MODE", "GROUP_LINK_M", "MAX_CHIP_M", "DEDUP_M", "VALID_BBOX",
                         "TEST_N", "N_THREADS"]))
    print("Файлы точек: " + "; ".join(files))
    print(f"Лог: {log_path}")
    pts = pd.concat([read_points(f) for f in files], ignore_index=True)
    pts = gpd.GeoDataFrame(pts, geometry="geometry", crs=4326)
    pts = assign_ids(pts)
    print(f"Точек загружено: {len(pts)}")

    aoi = load_aoi()
    if aoi is not None:
        n0 = len(pts)
        pts = pts[pts.within(aoi)].reset_index(drop=True)
        print(f"В пределах AOI: {len(pts)} из {n0}")

    # ---- план фрагментов ----------------------------------------------------
    chips, pts = plan_chips(pts)
    n_all, n_pts = len(chips), len(pts)
    single_km2 = n_pts * (2 * RADIUS_M) ** 2 / 1e6
    chips["area_km2"] = chips.width_m * chips.height_m / 1e6
    plan_km2 = chips.area_km2.sum()
    side = np.maximum(chips.width_m, chips.height_m)
    n1 = int((chips.n_points == 1).sum())
    print(f"Фрагментов: {n_all} из {n_pts} точек (GROUP_MODE = {GROUP_MODE}): "
          f"одиночных {n1}, общих {n_all - n1}")
    print("Точек на фрагмент: " + str(chips.n_points.value_counts().sort_index().to_dict()))
    print(f"Длинная сторона фрагмента: медиана {side.median():.0f} м, макс. {side.max():.0f} м; "
          f"суммарная площадь {plan_km2:.0f} км² (по фрагменту на каждую точку было бы "
          f"{single_km2:.0f} км²)")
    os.makedirs(out_dir, exist_ok=True)
    epsg_t = PROVIDERS[PROVIDER]["epsg"]
    res_g = 2 * HALF / (1 << ZOOM) / TILE          # пиксель в единицах проекции
    plan_px = {r.chip_id: [int(r.width_m / (res_g * math.cos(math.radians(r.lat)))),
                           int(r.height_m / (res_g * math.cos(math.radians(r.lat))))]
               for r in chips.itertuples()}
    import json
    with open(os.path.join(out_dir, "_plan.json"), "w", encoding="utf-8") as f:
        json.dump({"n_places": int(n_all), "n_points": int(n_pts), "group_mode": GROUP_MODE,
                   "chips": plan_px}, f)

    ip = os.path.join(out_dir, "chips_index.gpkg")
    old = gpd.read_file(ip) if os.path.exists(ip) else None
    old_keys = dict(zip(old.chip_id, old["key"])) if old is not None and "key" in old else {}

    sel = chips
    if TEST_N and TEST_N > 0:
        sel = chips.sample(min(TEST_N, n_all), random_state=0)
        print(f"ТЕСТ: {len(sel)} случайных фрагментов из {n_all} "
              f"({int(sel.n_points.sum())} точек)")

    cache = TileCache(os.path.join(out_dir, f"_tile_cache_{PROVIDER}.sqlite"))
    ext = {"gtiff": ".tif", "jpg": ".jpg", "png": ".png", "webp": ".webp"}[OUT_FORMAT]

    rows = []
    pts_total, pts_done = int(sel["n_points"].sum()), 0
    build_sec, built_km2, n_new = 0.0, 0.0, 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        for k, c in enumerate(sel.itertuples(), 1):
            mem = pts.iloc[c.idx_pts]
            geom = chip_geom(mem.geometry.x, mem.geometry.y, RADIUS_M, epsg_t)
            fn = os.path.join(out_dir, f"{c.chip_id}{ext}")
            status, miss = "ok", 0.0
            t1 = time.time()
            if SKIP_EXISTING and _chip_is_current(fn, c, old_keys, geom):
                status = "exists"
            else:
                try:
                    img, tr, epsg, g2, miss = build_chip(PROVIDER, cache, geom, ZOOM, pool)
                    img, tr, epsg, g2 = maybe_reproject(img, tr, epsg, c.lon, c.lat, g2)
                    write_chip(fn, img, tr, epsg, g2)
                    if miss > 0:
                        status = f"partial ({miss:.0%} тайлов нет)"
                    build_sec += time.time() - t1
                    built_km2 += c.area_km2
                    n_new += 1
                except Exception as e:
                    status = f"error: {e}"
            size_px, res_m, bnd, f_epsg = (_file_info(fn, c.lat) if not status.startswith("error")
                                           else (None, None, None, None))
            foot = (_footprint(bnd, f_epsg) if bnd is not None
                    else _footprint(chip_window(geom, ZOOM)["bounds"], epsg_t))
            rows.append(dict(chip_id=c.chip_id, n_points=c.n_points, members=c.members,
                             src_file=c.src_file, lon=c.lon, lat=c.lat,
                             width_m=c.width_m, height_m=c.height_m,
                             file=os.path.basename(fn) if bnd is not None else "",
                             provider=PROVIDER, zoom=ZOOM, radius_m=RADIUS_M,
                             pixel_m=res_m, size_px=size_px, missing_share=round(miss, 3),
                             status=status, key=c.key, geometry=foot))
            el = time.time() - t0
            pts_done += int(c.n_points)
            eta = el / k * (len(sel) - k)
            tag = f" ({c.n_points} тчк)" if c.n_points > 1 else ""
            print(f"[фрагм. {k}/{len(sel)} {k / len(sel):6.1%} | точки {pts_done}/{pts_total} "
                  f"{pts_done / pts_total:6.1%}] {c.chip_id}{tag}: {status} | "
                  f"{el / 60:.1f} мин, осталось ~{eta / 60:.0f} мин", flush=True)

    # ---- индекс фрагментов: текущий план + фрагменты других файлов точек ----
    idx = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)
    cur_files = {os.path.basename(f) for f in files}
    plan_key = dict(zip(chips.chip_id, chips.key))
    if old is not None and len(old):
        keep, rekey = [], {}
        plan_row = {c.chip_id: c for c in chips.itertuples()}
        for r in old.itertuples():
            if r.chip_id in set(idx.chip_id):
                continue
            has_file = isinstance(r.file, str) and r.file and \
                os.path.exists(os.path.join(out_dir, r.file))
            if not has_file:
                continue
            if r.chip_id in plan_key:              # фрагмент плана, не обработанный сейчас
                k_old = getattr(r, "key", None)
                if isinstance(k_old, str) and k_old:
                    ok = k_old == plan_key[r.chip_id]
                else:                              # индекс прежней версии: сверка охвата
                    c = plan_row[r.chip_id]
                    mem = pts.iloc[c.idx_pts]
                    g = chip_geom(mem.geometry.x, mem.geometry.y, RADIUS_M, epsg_t)
                    ok = _chip_is_current(os.path.join(out_dir, r.file), c, {}, g)
                    if ok:
                        rekey[r.Index] = plan_key[r.chip_id]
            else:                                  # от других файлов точек
                ok = not (set(str(r.src_file).split(";")) & cur_files)
            if ok:
                keep.append(r.Index)
        if "key" not in old:
            old["key"] = None
        for i, k in rekey.items():
            old.at[i, "key"] = k
        old = old.loc[keep].copy()
        if len(old):
            fix = old.geometry.geom_type != "Polygon"         # индекс прежних версий: точки
            for i in old.index[fix]:
                _, _, b, e = _file_info(os.path.join(out_dir, old.at[i, "file"]), old.at[i, "lat"])
                old.at[i, "geometry"] = _footprint(b, e) if b is not None else None
            old = old[old.geometry.notna()]
            idx = gpd.GeoDataFrame(pd.concat([old, idx], ignore_index=True),
                                   geometry="geometry", crs=4326)
    idx.to_file(ip, driver="GPKG")
    idx.drop(columns="geometry").to_csv(os.path.join(out_dir, "chips_index.csv"),
                                        index=False, encoding="utf-8-sig")

    # ---- точки с номером фрагмента -------------------------------------------
    pp = os.path.join(out_dir, "chips_points.gpkg")
    attrs = _clean_attrs(pts.drop(columns=["geometry", "point_id", "chip_id", "src_file"]))
    pout = gpd.GeoDataFrame(pd.concat([pts[["point_id", "chip_id", "src_file"]],
                                       pd.DataFrame({"lon": pts.geometry.x,
                                                     "lat": pts.geometry.y}), attrs], axis=1),
                            geometry=pts.geometry.values, crs=4326)
    if os.path.exists(pp):
        try:
            po = gpd.read_file(pp)
            po = po[~po.src_file.isin(cur_files)]
            if len(po):
                pout = gpd.GeoDataFrame(pd.concat([po, pout], ignore_index=True),
                                        geometry="geometry", crs=4326)
        except Exception:
            pass
    try:
        pout.to_file(pp, driver="GPKG")
    except Exception as e:
        print(f"[!] Атрибуты точек не записались в GPKG ({e}); записаны только поля фрагментов.")
        pout = pout[["point_id", "chip_id", "src_file", "lon", "lat", "geometry"]]
        pout.to_file(pp, driver="GPKG")
    pout.drop(columns="geometry").to_csv(os.path.join(out_dir, "chips_points.csv"),
                                         index=False, encoding="utf-8-sig")

    # ---- итоги и прогнозы ------------------------------------------------------
    el = time.time() - t0
    st = pd.Series([r["status"] for r in rows])
    print(f"\nИтого: {len(rows)} фрагментов за {el / 60:.1f} мин; новых {n_new}")
    print("Статусы: " + "; ".join(f"{k}: {v}" for k, v in
                                  st.str.split(" ").str[0].value_counts().items()))
    done = [(os.path.getsize(os.path.join(out_dir, r["file"])) / 2**20,
             r["width_m"] * r["height_m"] / 1e6) for r in rows
            if r["file"] and os.path.exists(os.path.join(out_dir, r["file"]))]
    if done:
        mb = np.array(done)
        mb_km2 = mb[:, 0].sum() / mb[:, 1].sum()
        print(f"Размер фрагмента: медиана {np.median(mb[:, 0]):.1f} МБ, всего {mb[:, 0].sum():.0f} МБ "
              f"({mb_km2:.1f} МБ/км²); прогноз на все {n_all} фрагментов ({plan_km2:.0f} км²): "
              f"~{mb_km2 * plan_km2 / 1024:.2f} ГБ")
    rest = n_all - len(rows)
    if rest > 0:
        rest_km2 = plan_km2 - sum(r["width_m"] * r["height_m"] / 1e6 for r in rows)
        if n_new and built_km2 > 0:
            print(f"Прогноз загрузки остальных {rest} фрагментов ({rest_km2:.0f} км²): "
                  f"~{build_sec / built_km2 * rest_km2 / 3600:.1f} ч "
                  f"(по {n_new} новым фрагментам этого запуска)")
        else:
            print("Новых фрагментов в этом запуске нет (все уже были скачаны), поэтому прогноз "
                  "времени не рассчитан.")
    known = set(chips.chip_id) | set(idx.chip_id)      # весь план (не только тест) + другие файлы
    stale = sorted(f for f in os.listdir(out_dir) if f.endswith(ext)
                   and os.path.splitext(f)[0] not in known)
    if stale:
        sp = os.path.join(out_dir, "_stale_chips.txt")
        with open(sp, "w", encoding="utf-8") as f:
            f.write("\n".join(stale))
        print(f"[i] {len(stale)} файлов фрагментов не входят в текущий план (остались от прежних "
              f"настроек группировки), 11_ их не обрабатывает. Список: {sp}")
    print(f"Кэш тайлов: {os.path.getsize(cache.path) / 2**20:.0f} МБ")
    print(f"Точки с номером фрагмента: {pp}")
    print(f"Лог: {log_path}")
    print(f"\nГотово: {out_dir}")
    print("Источник подложки: " + {"google": "© Google", "bing": "© Microsoft Bing",
                                   "yandex": "© Яндекс", "esri": "© Esri, Maxar, Earthstar Geographics"}[PROVIDER])


if __name__ == "__main__":
    try:
        main()
    finally:
        stop_log()
