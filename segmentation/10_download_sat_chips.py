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
  3. Для каждой точки: круг R м (в локальной азимутальной проекции -> точный
     радиус в метрах), подбор XYZ-тайлов на уровне ZOOM, загрузка с кэшем
     (SQLite, повторно не качаются — удобно при перекрывающихся буферах),
     сшивка, обрезка по квадрату, описанному вокруг круга, (опц.) маска
     вне круга, (опц.) перепроецирование в UTM.
  4. Пишет фрагмент + индекс chips_index.gpkg / .csv (id, координаты, файл,
     провайдер, zoom, размер пикселя, доля пропущенных тайлов).

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

CIRCLE_MASK = False        # True -> пиксели вне круга = 0 (+ маска/альфа). Для SAM обычно
                           # лучше False (квадрат без "чёрных углов"), а круг применять к маскам SAM.
REPROJECT_EPSG = None      # None -> оставить Web Mercator (3857, для Yandex — 3395);
                           # "utm" -> авто-UTM по точке; или код, напр. 32637

# AOI-фильтр (None, "gee" или путь к файлу GPKG/SHP/GeoJSON)
AOI_SOURCE = None
GEE_PROJECT = "steel-capsule-438113-h8"
GEE_AOI_ASSET = "projects/steel-capsule-438113-h8/assets/MO_AOI26"

SHEET_NAME = None          # для XLSX: None -> первый лист с координатами (у вас это "Class_all")
ID_FIELD = None            # None -> авто (id / point_id / NAME... ), иначе имя столбца
DEDUP_M = 25.0             # точки ближе DEDUP_M м друг к другу -> один общий фрагмент
                           # (в индексе перечислены все их имена); 0 -> без объединения
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
        bad = df[cx].isna() | df[cy].isna()
        if VALID_BBOX is not None:
            x0, y0, x1, y1 = VALID_BBOX
            bad |= ~df[cx].between(x0, x1) | ~df[cy].between(y0, y1)
        if bad.any():
            print(f"  [!] {os.path.basename(path)}: {bad.sum()} строк без координат или вне VALID_BBOX "
                  f"-> rejected_points.csv")
            df[bad].assign(src_file=os.path.basename(path)).to_csv(
                os.path.join(os.path.dirname(os.path.abspath(path)), "rejected_points.csv"),
                mode="a", index=False, encoding="utf-8-sig",
                header=not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(path)),
                                                       "rejected_points.csv")))
        df = df[~bad]
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[cx], df[cy]), crs=4326)
    else:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        gdf = gdf.to_crs(4326)
        gdf["geometry"] = gdf.geometry.representative_point()  # полигоны -> точка
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
    gdf["chip_id"] = ids.values
    return gdf


def dedup_points(gdf):
    """Точки не дальше DEDUP_M м от центра группы -> один фрагмент (центр = первая точка
    группы; цепочек нет, поэтому сдвиг любой точки от центра фрагмента <= DEDUP_M)."""
    gdf = gdf.copy()
    if not DEDUP_M or DEDUP_M <= 0:
        gdf["members"] = gdf["chip_id"]
        gdf["n_points"] = 1
        return gdf
    m = gdf.to_crs(gdf.estimate_utm_crs())
    xy = np.c_[m.geometry.x.values, m.geometry.y.values]
    leaders, grid = [], {}                 # grid: ячейка -> индексы лидеров
    group = np.empty(len(xy), dtype=int)
    for i, p in enumerate(xy):
        cx, cy = (p // DEDUP_M).astype(int)
        best, bd = None, DEDUP_M
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
        group[i] = best
    gdf["_g"] = group
    members = gdf.groupby("_g")["chip_id"].agg(lambda v: ";".join(v))
    sizes = gdf.groupby("_g").size()
    first = gdf.iloc[leaders].copy()
    first["members"] = first["_g"].map(members)
    first["n_points"] = first["_g"].map(sizes)
    return first.drop(columns="_g")


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


def build_chip(provider, cache, lon, lat, z, r, pool):
    epsg = PROVIDERS[provider]["epsg"]
    circ = circle_in_crs(lon, lat, r, epsg)
    minx, miny, maxx, maxy = circ.bounds
    tsize = 2 * HALF / (1 << z)          # размер тайла, м
    res = tsize / TILE                   # размер пикселя (в единицах проекции)

    tx0 = int(math.floor((minx + HALF) / tsize))
    tx1 = int(math.floor((maxx + HALF) / tsize))
    ty0 = int(math.floor((HALF - maxy) / tsize))
    ty1 = int(math.floor((HALF - miny) / tsize))

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

    # обрезка до квадрата, описанного вокруг круга
    ox, oy = -HALF + tx0 * tsize, HALF - ty0 * tsize
    c0 = int(math.floor((minx - ox) / res)); c1 = int(math.ceil((maxx - ox) / res))
    r0 = int(math.floor((oy - maxy) / res)); r1 = int(math.ceil((oy - miny) / res))
    img = mosaic[r0:r1, c0:c1]
    transform = from_origin(ox + c0 * res, oy - r0 * res, res, res)
    return img, transform, epsg, circ, missing / len(jobs)


def maybe_reproject(img, transform, src_epsg, lon, lat, circ):
    if REPROJECT_EPSG is None:
        return img, transform, src_epsg, circ
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
    return out, dst_t, dst_epsg, shp_transform(tr, circ)


# ============================================================================
# ЗАПИСЬ
# ============================================================================
def write_chip(path, img, transform, epsg, circ):
    h, w = img.shape[:2]
    inside = None
    if CIRCLE_MASK:
        inside = ~geometry_mask([circ], out_shape=(h, w), transform=transform)
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
                         "DEDUP_M", "VALID_BBOX", "TEST_N", "N_THREADS"]))
    print("Файлы точек: " + "; ".join(files))
    print(f"Лог: {log_path}")
    pts = pd.concat([read_points(f) for f in files], ignore_index=True)
    pts = gpd.GeoDataFrame(pts, geometry="geometry", crs=4326)
    pts = assign_ids(pts)
    print(f"Точек загружено: {len(pts)}")

    aoi = load_aoi()
    if aoi is not None:
        n0 = len(pts)
        pts = pts[pts.within(aoi)].copy()
        print(f"В пределах AOI: {len(pts)} из {n0}")

    n0 = len(pts)
    pts = dedup_points(pts)
    print(f"Уникальных мест (фрагментов): {len(pts)} (из {n0} точек, DEDUP_M = {DEDUP_M} м)")
    print(f"Распределение точек на фрагмент: {pts.n_points.value_counts().sort_index().to_dict()}")
    n_all = len(pts)
    os.makedirs(out_dir, exist_ok=True)
    import json
    with open(os.path.join(out_dir, "_plan.json"), "w", encoding="utf-8") as f:
        json.dump({"n_places": int(n_all), "n_points": int(pts.n_points.sum())}, f)
    if TEST_N and TEST_N > 0:
        pts = pts.sample(min(TEST_N, len(pts)), random_state=0)
        print(f"ТЕСТ: {len(pts)} случайных мест из {n_all}")

    os.makedirs(out_dir, exist_ok=True)
    cache = TileCache(os.path.join(out_dir, f"_tile_cache_{PROVIDER}.sqlite"))
    ext = {"gtiff": ".tif", "jpg": ".jpg", "png": ".png", "webp": ".webp"}[OUT_FORMAT]

    rows = []
    pts_total, pts_done = int(pts["n_points"].sum()), 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        for k, (_, p) in enumerate(pts.iterrows(), 1):
            lon, lat = p.geometry.x, p.geometry.y
            fn = os.path.join(out_dir, f"{p.chip_id}{ext}")
            status, miss, res_m, shape = "ok", 0.0, None, None
            if SKIP_EXISTING and os.path.exists(fn):
                status = "exists"
            else:
                try:
                    img, tr, epsg, circ, miss = build_chip(PROVIDER, cache, lon, lat,
                                                           ZOOM, RADIUS_M, pool)
                    img, tr, epsg, circ = maybe_reproject(img, tr, epsg, lon, lat, circ)
                    write_chip(fn, img, tr, epsg, circ)
                    shape = f"{img.shape[1]}x{img.shape[0]}"
                    # реальный размер пикселя на местности
                    k_scale = 1.0 if REPROJECT_EPSG else math.cos(math.radians(lat))
                    res_m = round(tr.a * k_scale, 3)
                    if miss > 0:
                        status = f"partial ({miss:.0%} тайлов нет)"
                except Exception as e:
                    status = f"error: {e}"
            rows.append(dict(chip_id=p.chip_id, n_points=p.n_points, members=p.members,
                             src_file=p.src_file, lon=lon, lat=lat,
                             file=os.path.basename(fn) if not status.startswith("error") else "",
                             provider=PROVIDER, zoom=ZOOM, radius_m=RADIUS_M,
                             pixel_m=res_m, size_px=shape, missing_share=round(miss, 3),
                             status=status, geometry=p.geometry))
            el = time.time() - t0
            pts_done += int(p.n_points)
            eta = el / k * (len(pts) - k)
            print(f"[фрагм. {k}/{len(pts)} {k / len(pts):6.1%} | точки {pts_done}/{pts_total} "
                  f"{pts_done / pts_total:6.1%}] {p.chip_id}: {status} | {el / 60:.1f} мин, "
                  f"осталось ~{eta / 60:.0f} мин", flush=True)

    idx = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)
    # индекс дополняется, а не перезаписывается (тест + полный запуск)
    ip = os.path.join(out_dir, "chips_index.gpkg")
    if os.path.exists(ip):
        old = gpd.read_file(ip)
        old = old[~old.chip_id.isin(idx.chip_id)]
        idx = gpd.GeoDataFrame(pd.concat([old, idx], ignore_index=True), geometry="geometry",
                               crs=4326)
    idx.to_file(ip, driver="GPKG")
    idx.drop(columns="geometry").to_csv(os.path.join(out_dir, "chips_index.csv"),
                                        index=False, encoding="utf-8-sig")
    el = time.time() - t0
    st = pd.Series([r["status"] for r in rows])
    sizes = [os.path.getsize(os.path.join(out_dir, r["file"])) / 2**20 for r in rows
             if r["file"] and os.path.exists(os.path.join(out_dir, r["file"]))]
    print(f"\nИтого: {len(rows)} фрагментов за {el / 60:.1f} мин "
          f"({el / max(len(rows), 1):.1f} с на фрагмент)")
    print("Статусы: " + "; ".join(f"{k}: {v}" for k, v in
                                  st.str.split(" ").str[0].value_counts().items()))
    if sizes:
        print(f"Размер фрагмента: медиана {np.median(sizes):.1f} МБ, всего {sum(sizes):.0f} МБ; "
              f"прогноз на все {n_all} мест: ~{np.median(sizes) * n_all / 1024:.2f} ГБ")
        if TEST_N:
            print(f"Прогноз времени загрузки всех {n_all} мест: "
                  f"~{el / max(len(rows), 1) * n_all / 3600:.1f} ч")
    print(f"Кэш тайлов: {os.path.getsize(cache.path) / 2**20:.0f} МБ")
    print(f"Лог: {log_path}")
    print(f"\nГотово: {out_dir}")
    print("Источник подложки: " + {"google": "© Google", "bing": "© Microsoft Bing",
                                   "yandex": "© Яндекс", "esri": "© Esri, Maxar, Earthstar Geographics"}[PROVIDER])


if __name__ == "__main__":
    try:
        main()
    finally:
        stop_log()
