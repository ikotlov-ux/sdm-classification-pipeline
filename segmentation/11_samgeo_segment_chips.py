# -*- coding: utf-8 -*-
"""
11_samgeo_segment_chips.py
==========================
Пакетная сегментация всех фрагментов, созданных 10_download_sat_chips.py,
через segment-geospatial (samgeo): SAM 2 (по умолчанию) или SAM 1.

Результат — ОДИН векторный слой segments.gpkg на все точки:
  chip_id, members (все точки фрагмента), n_points, seg_id, area_m2,
  cut_by_circle (сегмент обрезан границей кругов), points_in / n_pts_in (точки
  внутри сегмента), near_point / near_dist_m (ближайшая точка и расстояние от
  контура сегмента до неё, 0 — точка внутри), geometry.
Полигоны — в метрической проекции (OUT_EPSG, по умолчанию UTM 37N),
(опц.) обрезаны по кругам RADIUS_M вокруг всех точек фрагмента
(координаты точек — из chips_points.gpkg, который пишет 10_).

Общие фрагменты (несколько близких точек, до 2 км): чтобы детальность была
такой же, как у одиночного фрагмента 1 км, крупный фрагмент делится на
больше кадров SAM (ADAPTIVE_CROPS, кадр не больше CROP_MAX_PX). Время
обработки фрагмента растёт примерно пропорционально его площади; прогноз
считается по числу кадров. Маски хранятся в сжатом виде (RLE), а мелкие
островки убираются на растре меток (MIN_REGION_PX) — так расход памяти
почти не зависит от размера фрагмента.

Параллельная работа (N_WORKERS):
  * CPU: несколько процессов, у каждого своя копия модели и THREADS_PER_WORKER
    потоков (один процесс PyTorch плохо загружает 64 ядра; 8 x 8 — заметно
    быстрее). "auto" -> физические ядра / THREADS_PER_WORKER.
  * GPU NVIDIA: "auto" -> WORKERS_PER_GPU процессов на каждую видеокарту.
  Веса модели скачиваются один раз в главном процессе до запуска рабочих
  (одновременная запись в кэш Hugging Face из нескольких процессов на Windows
  даёт PermissionError); рабочие процессы читают кэш без обращения к сети.
  Фрагменты раздаются динамически (кто освободился — берёт следующий).
  Каждый процесс пишет свой файл _parts/segments_<pid>.gpkg; в конце все
  части сливаются в segments.gpkg (дубли после прерванных запусков убираются).

Прогресс: фрагменты и точки (шт. и %), сегменты, скорость, оставшееся время.

Пробный запуск: TEST_N = 16 -> обработать 16 случайных фрагментов, вывести
время на фрагмент и прогноз на все оставшиеся (в общий результат идут).

Повторный запуск продолжает с места остановки (_done.txt). Ошибки — в
_errors.csv, работа не прерывается.

Ускорение ценой мелких контуров: PRESET = "fast" (crop_n_layers=0, ~3x
быстрее) и/или фрагменты zoom 17 в скрипте 10.

Установка:
  pip install segment-geospatial sam2 geopandas rasterio      # + PyTorch (CUDA при наличии GPU)
  SAM 1: pip install segment-anything
Проверка видеокарты: nvidia-smi
"""

import csv
import gc
import glob
import math
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import shape, Point
from pyproj import CRS, Transformer
from shapely.ops import transform as shp_transform

# ============================================================================
# НАСТРОЙКИ
# ============================================================================
CHIPS_DIR = None            # None -> окно выбора папки с chips_index.gpkg
MODEL = "sam2"              # "sam2" | "sam1"
SAM2_MODEL_ID = "sam2-hiera-large"   # sam2-hiera-tiny / -small / -base-plus / -large
SAM1_MODEL_TYPE = "vit_h"            # vit_b / vit_l / vit_h
DEVICE = None               # None -> авто (cuda, если есть), "cpu", "cuda", "mps"

N_WORKERS = "auto"          # "auto" или число процессов (1 -> последовательно)
THREADS_PER_WORKER = 8      # потоков PyTorch на процесс
WORKERS_PER_GPU = 2         # процессов на видеокарту при N_WORKERS="auto": пока один процесс
                            # готовит маски и контуры на процессоре, другой занимает карту
MEM_PER_WORKER_GB = 16      # запас ОЗУ на процесс при N_WORKERS="auto" (large + quality);
                            # процессов не больше, чем свободная ОЗУ / MEM_PER_WORKER_GB
TEST_N = 20                 # ТЕСТОВЫЙ ЗАПУСК на 20 фрагментах; после проверки лога -> 0 (все)

PRESET = "quality"          # "quality" (crop_n_layers=1) | "fast" (crop_n_layers=0)

# Параметры автоматического генератора масок.
# SAM сжимает изображение до 1024 px по длинной стороне; фрагмент zoom 18
# (~3000 px) -> crop_n_layers=1 добавляет проход по 4 кадрам с полным
# разрешением, мелкие контуры (кроны, опушки, просеки) выделяются лучше.
AMG_KWARGS = dict(
    points_per_side=32,
    points_per_batch=64,
    pred_iou_thresh=0.70,
    stability_score_thresh=0.85,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=0,         # 0: мелкие "дырки" и островки убираются ниже
)                                   # (MIN_REGION_PX) — экономнее по памяти, чем в SAM
if PRESET == "fast":
    AMG_KWARGS["crop_n_layers"] = 0
MIN_REGION_PX = 100         # островки и дырки меньше (пикс.) сливаются с соседним сегментом
ADAPTIVE_CROPS = True       # крупные (общие) фрагменты -> больше кадров, детальность как у 1 км
CROP_MAX_PX = 2000          # наибольший размер кадра SAM, пикс. (у фрагмента 1 км z18 ~2000)

ENHANCE = True              # растяжка контраста 2–98 перцентиль по каналам перед SAM
MIN_AREA_M2 = 50.0          # отбросить сегменты меньше (м²)
SIMPLIFY_M = 0.5            # упрощение контуров, м (0 -> без упрощения)
CLIP_TO_CIRCLE = True       # обрезать сегменты по кругам RADIUS_M вокруг точек фрагмента
OUT_EPSG = 32637            # UTM 37N — вся Московская обл.; None -> CRS фрагмента
KEEP_MASK_TIF = False       # сохранять растр масок (LZW) рядом с фрагментом
OUT_NAME = "segments.gpkg"

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
                 "segment-geospatial", "sam2", "segment-anything"):
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


def _crop_grid(n_px):
    """Число кадров SAM по стороне фрагмента (не меньше 2, как в SAM)."""
    ov = int(512 / 1500 * CROP_MAX_PX)                     # перекрытие кадров, как в SAM
    return max(2, math.ceil((n_px - ov) / (CROP_MAX_PX - ov)))


def _adaptive_crop_boxes(orig):
    """Замена generate_crop_boxes SAM: слой 1 — сетка nx x ny кадров не больше
    CROP_MAX_PX. Для обычного фрагмента (2 x 2) — исходная функция SAM."""
    def fn(im_size, n_layers, overlap_ratio):
        h, w = im_size
        if n_layers != 1:
            return orig(im_size, n_layers, overlap_ratio)
        nx, ny = _crop_grid(w), _crop_grid(h)
        if nx == 2 and ny == 2:
            return orig(im_size, n_layers, overlap_ratio)
        ov = int(overlap_ratio * CROP_MAX_PX)
        cw = math.ceil((ov * (nx - 1) + w) / nx)
        ch = math.ceil((ov * (ny - 1) + h) / ny)
        boxes, layers = [[0, 0, w, h]], [0]
        for i in range(nx):
            for j in range(ny):
                x0, y0 = int((cw - ov) * i), int((ch - ov) * j)
                boxes.append([x0, y0, min(x0 + cw, w), min(y0 + ch, h)])
                layers.append(1)
        return boxes, layers
    fn._orig = orig
    return fn


def n_crops(w, h):
    """Сколько кадров SAM обработает для фрагмента w x h (для прогноза времени)."""
    L = int(AMG_KWARGS.get("crop_n_layers", 0))
    if L == 0:
        return 1
    if L == 1:
        return 1 + (_crop_grid(w) * _crop_grid(h) if ADAPTIVE_CROPS else 4)
    return sum(4 ** i for i in range(L + 1))


def load_model(device):
    if MODEL == "sam2":
        from samgeo import SamGeo2
        sam = SamGeo2(model_id=SAM2_MODEL_ID, automatic=True, device=device, **AMG_KWARGS)
    else:
        from samgeo import SamGeo
        sam = SamGeo(model_type=SAM1_MODEL_TYPE, automatic=True, device=device,
                     sam_kwargs=AMG_KWARGS)
    mg = sam.mask_generator
    mg.output_mode = "uncompressed_rle"      # маски в RLE: память не растёт с размером фрагмента
    if ADAPTIVE_CROPS:
        mod = sys.modules[type(mg).__module__]
        f = mod.generate_crop_boxes
        mod.generate_crop_boxes = _adaptive_crop_boxes(getattr(f, "_orig", f))
    return sam


def prefetch_weights():
    """Скачивает веса модели один раз в главном процессе (до запуска рабочих).
    Возвращает True, если веса есть в кэше и рабочие процессы могут работать без сети."""
    t = time.time()
    try:
        if MODEL == "sam2":
            from huggingface_hub import hf_hub_download
            from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES
            repo = SAM2_MODEL_ID if "/" in SAM2_MODEL_ID else f"facebook/{SAM2_MODEL_ID}"
            path = hf_hub_download(repo_id=repo, filename=HF_MODEL_ID_TO_FILENAMES[repo][1])
        else:
            import warnings
            warnings.filterwarnings("ignore")
            sam = load_model("cpu")                 # SamGeo сам скачивает чекпойнт, если его нет
            path = getattr(sam, "checkpoint", "") or "чекпойнт SAM 1"
            del sam
            gc.collect()
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


def _rle_to_mask(seg, h, w):
    if isinstance(seg, np.ndarray):
        return seg
    counts = np.asarray(seg["counts"], dtype=np.int64)
    vals = (np.arange(len(counts)) % 2).astype(bool)
    return np.repeat(vals, counts).reshape(w, h).T          # RLE SAM — по столбцам


def masks_to_labels(masks, h, w):
    """Список масок SAM -> растр меток (крупные снизу, мелкие поверх); маски
    раскодируются по одной. Мелкие островки и дырки -> соседний сегмент."""
    lab = np.zeros((h, w), dtype=np.int32)
    for k, m in enumerate(sorted(masks, key=lambda x: x["area"], reverse=True), 1):
        lab[_rle_to_mask(m["segmentation"], h, w)] = k
    if MIN_REGION_PX and MIN_REGION_PX > 1:
        lab = features.sieve(lab, size=int(MIN_REGION_PX), connectivity=4)
    return lab


def circle_geom(lon, lat, r, epsg):
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m")
    tr = Transformer.from_crs(aeqd, CRS.from_epsg(epsg), always_xy=True).transform
    return shp_transform(tr, Point(0, 0).buffer(r, 128))


def segment_chip(sam, path, row, out_epsg):
    row = pd.Series(row)
    with rasterio.open(path) as src:
        h, w, transform, crs = src.height, src.width, src.transform, src.crs
        img = np.moveaxis(src.read([1, 2, 3]), 0, 2)
    _W["meta"] = (w, h, n_crops(w, h))
    if ENHANCE:
        img = stretch(img)
    sam.generate(img, output=None)           # маски остаются в sam.masks
    masks = sam.masks or []
    if not masks:
        return gpd.GeoDataFrame(columns=["chip_id", "geometry"], geometry="geometry",
                                crs=out_epsg or crs)
    lab = masks_to_labels(masks, h, w)
    sam.masks = None
    del masks, img

    if KEEP_MASK_TIF:
        mp = os.path.splitext(path)[0] + "_mask.tif"
        with rasterio.open(mp, "w", driver="GTiff", height=h, width=w, count=1,
                           dtype="uint16" if lab.max() < 65535 else "uint32",
                           crs=crs, transform=transform, compress="LZW") as dst:
            dst.write(lab.astype(dst.dtypes[0]), 1)

    geoms, vals = [], []
    for g, v in features.shapes(lab, mask=lab > 0, transform=transform):
        geoms.append(shape(g)); vals.append(int(v))
    del lab
    gdf = gpd.GeoDataFrame({"seg_id": vals}, geometry=geoms, crs=crs)
    gdf = gdf.dissolve("seg_id", as_index=False)
    if out_epsg:
        gdf = gdf.to_crs(out_epsg)
    epsg = gdf.crs.to_epsg()

    # точки фрагмента (из chips_points.gpkg; для старых индексов — центр фрагмента)
    pts = row.get("pts") or [[row.chip_id, row.lon, row.lat]]
    pid = [str(p[0]) for p in pts]
    pg = gpd.GeoSeries(gpd.points_from_xy([p[1] for p in pts], [p[2] for p in pts]),
                       crs=4326).to_crs(gdf.crs)
    if CLIP_TO_CIRCLE:
        from shapely.ops import unary_union
        circ = unary_union([circle_geom(p[1], p[2], float(row.radius_m), epsg) for p in pts])
        inner = circ.buffer(-1.0)
        gdf["cut_by_circle"] = ~gdf.within(inner)
        gdf["geometry"] = gdf.intersection(circ)
        gdf = gdf[~gdf.is_empty]
    else:
        gdf["cut_by_circle"] = False
    if SIMPLIFY_M:
        gdf["geometry"] = gdf.simplify(SIMPLIFY_M, preserve_topology=True)
    gdf = gdf[gdf.geometry.is_valid & ~gdf.is_empty].copy()
    gdf["area_m2"] = gdf.area.round(1) if gdf.crs.is_projected else np.nan
    gdf = gdf[gdf["area_m2"].fillna(MIN_AREA_M2) >= MIN_AREA_M2].copy()

    # связь сегментов с точками
    if len(gdf):
        dist = np.array([[g.distance(p) for p in pg] for g in gdf.geometry])
        inside = dist <= 1e-6
        gdf["points_in"] = [";".join(np.array(pid)[m]) for m in inside]
        gdf["n_pts_in"] = inside.sum(1)
        j = dist.argmin(1)
        gdf["near_point"] = np.array(pid)[j]
        gdf["near_dist_m"] = dist[np.arange(len(gdf)), j].round(1)
    gdf.insert(0, "chip_id", row.chip_id)
    gdf.insert(1, "members", row.members if "members" in row else row.chip_id)
    return gdf


# ============================================================================
# РАБОЧИЙ ПРОЦЕСС
# ============================================================================
_W = {}


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
    warnings.filterwarnings("ignore")
    for attempt in range(3):            # файлы кэша на Windows бывают кратковременно заняты
        try:
            _W["sam"] = load_model(device)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(5 + 2 * attempt + wid)
    _W["dir"] = chips_dir
    os.makedirs(os.path.join(chips_dir, "_parts"), exist_ok=True)
    _W["part"] = os.path.join(chips_dir, "_parts", f"segments_{os.getpid()}.gpkg")


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


def process_chip(row):
    sam, d = _W["sam"], _W["dir"]
    t1 = time.time()
    _W["meta"] = (0, 0, 0)
    try:
        gdf = segment_chip(sam, os.path.join(d, row["file"]), row, OUT_EPSG)
        if len(gdf):
            gdf["n_points"] = int(row.get("n_points", 1) or 1)
            gdf["chip_key"] = _key(row.get("key"))
            gdf.to_file(_W["part"], layer="segments", driver="GPKG",
                        mode="a" if os.path.exists(_W["part"]) else "w")
        res = (row["chip_id"], len(gdf), time.time() - t1, None)
    except Exception as e:
        res = (row["chip_id"], 0, time.time() - t1,
               f"{e!r} | {traceback.format_exc(limit=2)}")
    rss = peak_ram_gb()
    sam.masks = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return res + (os.getpid(), rss) + tuple(_W["meta"])


def worker_loop(wid, task_q, res_q, chips_dir, n_gpu, threads):
    t = time.time()
    try:
        init_worker(wid, chips_dir, n_gpu, threads)
    except Exception as e:
        res_q.put(("fail", wid, f"{e!r} | {traceback.format_exc(limit=3)}"))
        return
    res_q.put(("ready", wid, (os.getpid(), time.time() - t)))
    while True:
        row = task_q.get()
        if row is None:
            break
        res_q.put(("start", wid, row["chip_id"]))
        res_q.put(("done", wid, process_chip(row)))


# ============================================================================
# СЛИЯНИЕ ЧАСТЕЙ
# ============================================================================
def merge_parts(d, done, valid=None):
    parts = sorted(glob.glob(os.path.join(d, "_parts", "segments_*.gpkg")))
    if not parts:
        print("Нет частей для слияния."); return
    chunks, seen = [], set()
    for p in parts:
        try:
            g = gpd.read_file(p, layer="segments")
        except Exception as e:
            print(f"  [!] часть {os.path.basename(p)} не читается: {e}"); continue
        tok = [_token(c, k) for c, k in zip(g.chip_id, g["chip_key"] if "chip_key" in g
                                            else [""] * len(g))]
        g = g.assign(_tok=tok)
        ok = g._tok.isin(done) & ~g._tok.isin(seen)      # один источник на фрагмент
        if valid is not None:
            ok &= g._tok.isin(valid)                     # только фрагменты текущего индекса
        g = g[ok].drop(columns="_tok")
        seen |= set(t for t, o in zip(tok, ok) if o)
        chunks.append(g)
    chunks = [c for c in chunks if len(c)]
    if not chunks:
        print("Нет готовых сегментов для слияния."); return
    out = pd.concat(chunks, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=chunks[0].crs)
    out.insert(0, "uid", np.arange(1, len(out) + 1))
    path = os.path.join(d, OUT_NAME)
    tmp = path + ".tmp.gpkg"
    out.to_file(tmp, layer="segments", driver="GPKG")
    if os.path.exists(path):
        os.remove(path)
    os.replace(tmp, path)
    print(f"Слито: {len(out)} сегментов из {len(seen)} фрагментов -> {path}")


# ============================================================================
# MAIN
# ============================================================================
def _key(v):
    return v if isinstance(v, str) and v and v.lower() != "nan" else ""


def _token(chip_id, key):
    """Метка готового фрагмента в _done.txt: chip_id|key (key меняется при смене состава точек)."""
    k = _key(key)
    return f"{chip_id}|{k}" if k else str(chip_id)


def _dims(d, r):
    """Размер фрагмента (w, h) в пикселях: из индекса или из файла."""
    v = r.get("size_px")
    if isinstance(v, str) and "x" in v:
        try:
            w, h = v.lower().split("x")
            return int(w), int(h)
        except ValueError:
            pass
    try:
        with rasterio.open(os.path.join(d, r["file"])) as src:
            return src.width, src.height
    except Exception:
        return 3000, 3000


def fmt_t(sec):
    sec = int(sec)
    return f"{sec // 3600} ч {sec % 3600 // 60:02d} мин" if sec >= 3600 else f"{sec // 60} мин {sec % 60:02d} с"


def main():
    d = pick_dir()
    if not d:
        print("Папка не выбрана."); return
    log_path, stamp = start_log(os.path.join(d, "_logs"), "11_segment")
    print("=" * 78)
    print("11_samgeo_segment_chips.py" + ("  [ТЕСТОВЫЙ ЗАПУСК]" if TEST_N else ""))
    print("=" * 78)
    print(system_info())
    print("Настройки:")
    print(settings_info(["MODEL", "SAM2_MODEL_ID", "SAM1_MODEL_TYPE", "DEVICE", "N_WORKERS",
                         "THREADS_PER_WORKER", "WORKERS_PER_GPU", "TEST_N", "PRESET",
                         "AMG_KWARGS", "ENHANCE", "MIN_REGION_PX", "ADAPTIVE_CROPS", "CROP_MAX_PX", "MIN_AREA_M2",
                         "SIMPLIFY_M", "CLIP_TO_CIRCLE", "OUT_EPSG"]))
    print(f"Папка фрагментов: {os.path.abspath(d)}")
    print(f"Лог: {log_path}")
    idx = gpd.read_file(os.path.join(d, "chips_index.gpkg"))
    idx = idx[idx["file"].fillna("") != ""].copy()
    for c in ("zoom", "pixel_m", "size_px"):
        if c in idx:
            print(f"  {c}: {idx[c].astype(str).value_counts().head(3).to_dict()}")
    if "n_points" not in idx:
        idx["n_points"] = 1
    idx["n_points"] = idx["n_points"].fillna(1).astype(int)
    if "key" not in idx:
        idx["key"] = ""
    idx["token"] = [_token(c, k) for c, k in zip(idx.chip_id, idx.key)]
    wh = [_dims(d, r) for r in idx.to_dict("records")]
    idx["w_px"], idx["h_px"] = [a for a, _ in wh], [b for _, b in wh]
    idx["cost"] = [n_crops(a, b) for a, b in wh]
    print(f"  кадров SAM на фрагмент: {idx.cost.value_counts().sort_index().to_dict()}")

    # координаты всех точек фрагментов (для обрезки по кругам и связи сегментов с точками)
    pts_by_chip = {}
    pp = os.path.join(d, "chips_points.gpkg")
    if os.path.exists(pp):
        cp = gpd.read_file(pp, columns=["point_id", "chip_id", "lon", "lat"])
        cp["point_id"] = cp["point_id"].astype(str)
        for cid, g in cp.groupby("chip_id"):
            pts_by_chip[cid] = g[["point_id", "lon", "lat"]].values.tolist()
        print(f"Точки фрагментов: {len(cp)} (chips_points.gpkg)")
    else:
        print("[i] chips_points.gpkg нет (индекс прежней версии 10_): круг строится вокруг "
              "центра фрагмента")

    done_f = os.path.join(d, "_done.txt")
    err_f = os.path.join(d, "_errors.csv")
    done = set(open(done_f, encoding="utf-8").read().split()) if os.path.exists(done_f) else set()
    todo = idx[~idx.token.isin(done)]
    if TEST_N and TEST_N > 0:
        todo = todo.sample(min(TEST_N, len(todo)), random_state=0)

    N_ch, N_pt = len(idx), int(idx.n_points.sum())
    is_done = idx.token.isin(done)
    done_ch, done_pt = int(is_done.sum()), int(idx.loc[is_done, "n_points"].sum())
    print(f"Фрагментов: {N_ch} (точек {N_pt}, кадров SAM {int(idx.cost.sum())}); готово ранее: "
          f"{done_ch} ({done_pt} точек); к обработке сейчас: {len(todo)} "
          f"(кадров {int(todo.cost.sum())}){' [ПРОБНЫЙ ЗАПУСК]' if TEST_N else ''}")
    valid = set(idx.token)
    if todo.empty:
        merge_parts(d, done, valid); return

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
    nw = max(1, min(nw, len(todo)))
    dev = (f"GPU x{n_gpu}, процессов на карту: {math.ceil(nw / n_gpu)}" if n_gpu
           else f"CPU, {THREADS_PER_WORKER} потоков на процесс")
    print(f"Процессов: {nw} ({dev}); модель {SAM2_MODEL_ID if MODEL == 'sam2' else SAM1_MODEL_TYPE}, "
          f"preset {PRESET}. Загрузка моделей...")

    keep = ["chip_id", "file", "lon", "lat", "radius_m", "n_points", "members", "key", "token",
            "cost"]
    rows = todo[[c for c in keep if c in todo.columns]].to_dict("records")
    for r in rows:
        r["key"] = _key(r.get("key"))
        if r["chip_id"] in pts_by_chip:
            r["pts"] = pts_by_chip[r["chip_id"]]
    npts = {r["chip_id"]: int(r["n_points"]) for r in rows}
    cost = {r["chip_id"]: int(r["cost"]) for r in rows}
    tokens = {r["chip_id"]: r["token"] for r in rows}
    cost_todo = sum(cost.values())

    if prefetch_weights():
        os.environ["HF_HUB_OFFLINE"] = "1"      # рабочие процессы: только кэш, без записи и сети
    ctx = mp.get_context("spawn")
    task_q, res_q = ctx.Queue(), ctx.Queue()
    for r in rows:
        task_q.put(r)                   # «стоп»-сигналы отправляются после завершения всех задач
    row_by_id = {r["chip_id"]: r for r in rows}
    retries = {}
    fails = {}

    def spawn(wid):
        p = ctx.Process(target=worker_loop, daemon=True,
                        args=(wid, task_q, res_q, d, n_gpu, THREADS_PER_WORKER))
        p.start()
        return p

    workers = {w: spawn(w) for w in range(nw)}
    inflight = {}                       # wid -> chip_id
    timing_f = os.path.join(d, "_logs", f"11_timing_{stamp}.csv")
    with open(timing_f, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(["chip_id", "n_points", "n_segments", "seconds", "width_px",
                                "height_px", "n_crops", "sec_per_crop", "worker_pid",
                                "worker_peak_ram_gb", "error"])
    print(f"Модели загружаются в {nw} процессах (первый фрагмент появится после загрузки)...",
          flush=True)
    t0 = time.time()
    k = segs = errs = 0
    pts_run = 0
    sec_sum = 0.0
    cost_done = 0
    n_ready = 0
    import queue as _queue

    def finish(chip_id, nseg, sec, err, pid, rss, w=0, h=0, nc=0):
        nonlocal k, segs, errs, pts_run, sec_sum, cost_done
        k += 1
        sec_sum += sec
        cost_done += cost[chip_id]
        with open(timing_f, "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow([chip_id, npts[chip_id], nseg, round(sec, 1), w, h, nc,
                                    round(sec / nc, 2) if nc else "", pid,
                                    round(rss, 2), (err or "")[:300]])
        if err:
            errs += 1
            with open(err_f, "a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow([chip_id, err])
            msg = f"ОШИБКА {err.splitlines()[0][:150]}"
        else:
            with open(done_f, "a", encoding="utf-8") as f:
                f.write(tokens[chip_id] + "\n")
            done.add(tokens[chip_id])
            segs += nseg
            pts_run += npts[chip_id]
            msg = f"{nseg} сегм."
        el = time.time() - t0
        rate = k / el
        eta = (cost_todo - cost_done) / (cost_done / el) if cost_done else 0
        ch_all = done_ch + k - errs
        pt_all = done_pt + pts_run
        print(f"[фрагм. {k}/{len(todo)} {k / len(todo):6.1%} | всего {ch_all}/{N_ch} "
              f"{ch_all / N_ch:6.1%} | точки {pt_all}/{N_pt} {pt_all / N_pt:6.1%}] "
              f"{chip_id}: {msg}, {sec:.0f} с | сегм. {segs} | ош. {errs} | "
              f"{rate * 3600:.0f} фр/ч | пик ОЗУ проц. {rss:.1f} ГБ | осталось ~{fmt_t(eta)}",
              flush=True)

    while k < len(todo):
        try:
            kind, wid, data = res_q.get(timeout=5)
        except _queue.Empty:
            # контроль аварийно завершившихся процессов (нехватка ОЗУ, сбой драйвера и т.п.)
            for wid, p in list(workers.items()):
                if p.is_alive() or (p.exitcode == 0 and wid not in inflight):
                    continue
                cid = inflight.pop(wid, None)
                print(f"[!] Процесс {wid} завершился аварийно (код {p.exitcode}) "
                      f"на фрагменте {cid} — вероятно, не хватило памяти. Перезапуск.",
                      flush=True)
                if cid is not None:
                    retries[cid] = retries.get(cid, 0) + 1
                    if retries[cid] <= 1:
                        task_q.put(row_by_id[cid])            # одна повторная попытка
                    else:
                        finish(cid, 0, 0.0, f"процесс аварийно завершён (код {p.exitcode}) "
                                            f"дважды — вероятно нехватка ОЗУ", -1, float("nan"))
                workers[wid] = spawn(wid)
            continue
        if kind == "ready":
            n_ready += 1
            tag = f"[{n_ready}/{nw}]" if n_ready <= nw else "[перезапуск]"
            print(f"  процесс {wid} (pid {data[0]}) готов, модель загружена за {data[1]:.0f} с "
                  f"{tag}", flush=True)
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
    print(f"\nОбработано {k} фрагментов за {fmt_t(el)}; ошибок {errs}.")
    if TEST_N and cost_done:
        # оставшиеся фрагменты: из плана 10_ (_plan.json, все места) или из индекса
        done_ids = {t.split("|")[0] for t in done}
        rest_cost = {r.chip_id: r.cost for r in idx.itertuples() if r.token not in done}
        plan = os.path.join(d, "_plan.json")
        if os.path.exists(plan):                     # тест после тестовой загрузки 10_
            import json
            pj = json.load(open(plan, encoding="utf-8"))
            for cid, (w, h) in (pj.get("chips") or {}).items():
                if cid not in done_ids and cid not in rest_cost:
                    rest_cost[cid] = n_crops(w, h)
        rc = sum(rest_cost.values())
        thr = cost_done / el                          # кадров SAM в секунду, все процессы
        print(f"Тестовый запуск: {sec_sum / cost_done:.1f} с на кадр SAM в одном процессе "
              f"(одиночный фрагмент 1 км = {n_crops(2976, 2976)} кадров), "
              f"{thr * 3600:.0f} кадров/ч при {nw} процессах (включая загрузку моделей — "
              f"реальная скорость будет немного выше).")
        print(f"Прогноз на оставшиеся {len(rest_cost)} фрагментов ({rc} кадров): ~{fmt_t(rc / thr)}")
    try:
        tdf = pd.read_csv(timing_f, encoding="utf-8-sig")
        ok = tdf[tdf.error.isna()]
        if len(ok):
            print(f"Время на фрагмент в одном процессе: медиана {ok.seconds.median():.0f} с, "
                  f"мин {ok.seconds.min():.0f}, макс {ok.seconds.max():.0f}; на кадр SAM: "
                  f"медиана {ok.sec_per_crop.median():.1f} с; сегментов на фрагмент: медиана "
                  f"{ok.n_segments.median():.0f}; пик ОЗУ процесса "
                  f"{tdf.worker_peak_ram_gb.max():.1f} ГБ")
    except Exception:
        pass
    merge_parts(d, done, valid)
    print(f"Таблица времени: {timing_f}")
    print(f"Лог: {log_path}")
    if TEST_N:
        print("Тест завершён. Пришлите лог; для полного запуска поставьте TEST_N = 0.")


def _in_jupyter():
    return "ipykernel" in sys.modules and os.environ.get("SEG11_CHILD") != "1"


def run_from_jupyter():
    """В Jupyter (%run) параллельные процессы Windows не могут импортировать функции
    из ячейки, поэтому скрипт перезапускает сам себя обычным python и выводит его
    строки в ячейку по мере работы."""
    import subprocess
    d = pick_dir()
    if not d:
        print("Папка не выбрана."); return
    env = dict(os.environ, SEG11_CHILD="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    print(f"Запуск в отдельном процессе: {sys.executable} {os.path.abspath(__file__)}", flush=True)
    p = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), d], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         encoding="utf-8", errors="replace", bufsize=1)
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
    except KeyboardInterrupt:
        p.terminate()
        print("\nОстановлено пользователем (готовые фрагменты сохранены).")
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
