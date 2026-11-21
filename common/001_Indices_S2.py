# =============================================================
# 001_Indices_S2.py  (v9: несколько файлов, блоки, все ядра, GPU)
# Расчет всех возможных спектральных индексов для Sentinel-2
# по формулам библиотеки spyndex; для каждого выбранного растра
# создаётся отдельный GeoTIFF <имя>_indices.tif
# =============================================================
# Адаптация R-скрипта (001_Indices_S2_v8.r) на Python.
# Sentinel-2 bands (9 каналов 10/20м):
#   B2  = Blue     (490 nm)
#   B3  = Green    (560 nm)
#   B4  = Red      (665 nm)
#   B5  = RE1      (705 nm)
#   B6  = RE2      (740 nm)
#   B7  = RE3      (783 nm)
#   B8  = NIR      (842 nm)
#   B11 = SWIR1    (1610 nm)
#   B12 = SWIR2    (2190 nm)
#
# Sentinel-2 full (13 каналов):
#   B1=Coastal, B2=Blue, B3=Green, B4=Red, B5=RE1, B6=RE2, B7=RE3,
#   B8=NIR, B8A=NIR-narrow, B9=WV, B10=Cirrus, B11=SWIR1, B12=SWIR2
#
# Что изменилось по сравнению с прежней версией:
#   * Окно Проводника с мультивыбором: можно выбрать несколько растров,
#     индексы считаются для каждого. Вопросы об именах каналов и scaleFactor
#     задаются один раз для всех файлов.
#   * Расчёт блоками BLOCK x BLOCK пикселей в N_WORKERS потоках (по умолчанию
#     все физические ядра). Память не зависит от размера растра.
#   * Результат пишется с INTERLEAVE=BAND и многопоточным сжатием. Прежняя
#     версия писала каналы по одному в файл с чередованием пикселей
#     (INTERLEAVE=PIXEL): GDAL при каждом новом канале заново распаковывал
#     и пережимал те же блоки, а файл получался в 2–3 раза больше несжатого
#     объёма. Если несжатый объём был меньше ~2 ГБ, GDAL создавал обычный
#     TIFF (предел 4 ГБ), файл упирался в этот предел, и последние каналы
#     молча оставались заполнены нулями.
#   * Формулы spyndex компилируются один раз (spyndex.computeIndex при каждом
#     вызове заново читает JSON со списком индексов). Результат тот же:
#     spyndex.computeIndex внутри делает eval(формула, {}, params).
#   * Необязательно: расчёт формул на видеокарте NVIDIA (PyTorch с CUDA).
#   Набор индексов, их порядок, имена каналов (S2_<индекс>, *_SKIPPED,
#   *_ERROR) и NoData = -9999 остались прежними.
# =============================================================

import os
import re
import sys
import glob
import time
import shutil
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as np

# =============================================================
# НАСТРОЙКИ
# =============================================================
N_WORKERS = "auto"     # потоков расчёта: "auto" = число физических ядер; или число, напр. 32
USE_GPU = "auto"       # "auto" = видеокарта NVIDIA, если её видит PyTorch с CUDA, иначе процессор;
                       # False = только процессор; True = только видеокарта (ошибка, если её нет)
BLOCK = 512            # сторона блока расчёта, пикс. (кратно 256). 1024 -> меньше накладных
                       # расходов, но ~1 ГБ памяти на поток при 230 индексах
COMPRESS = "DEFLATE"   # "DEFLATE" (уровень 1: быстрее LZW и файл меньше; читают ArcGIS, QGIS, GDAL)
                       # | "LZW" (как в прежней версии) | "ZSTD" (ещё быстрее, но старые версии
                       # ArcGIS его не читают) | "NONE"
PREDICTOR = 3          # 3 = предиктор для float (заметно уменьшает файл); 1 = без предиктора
BAND_NAMES = "ask"     # "ask" = спросить; "keep" = оставить имена каналов;
                       # "standard" = стандартные имена Sentinel-2 (для 9 и 13 каналов)
SCALE_FACTOR = "ask"   # "ask" = спросить; "auto" = рекомендованный для каждого файла;
                       # или число для всех файлов (10000 / 1)
OVERWRITE = True       # False -> файлы, у которых уже есть <имя>_indices.tif, пропускаются
INDEX_LIST = None      # None = все доступные индексы; или список, напр. ["NDVI", "EVI", "NDMI"]
NODATA_OUT = -9999.0

# --- Настройка путей и временной папки ---------------------------------------
#
# Все рабочие папки берутся рядом со скриптом, а не из текущей папки запуска.
# Временные файлы Python/GDAL/rasterio перенаправляются в ./TEMP рядом со
# скриптом, чтобы не забивать системный диск C: при обработке больших GeoTIFF.
# Результат пишется рядом с исходным растром.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

input_dir = os.path.join(SCRIPT_DIR, "INPUT")
temp_dir = os.path.join(SCRIPT_DIR, "TEMP")

for d in [input_dir, temp_dir]:
    os.makedirs(d, exist_ok=True)

for env_name in ["TMP", "TEMP", "TMPDIR"]:
    os.environ[env_name] = temp_dir

os.environ["CPL_TMPDIR"] = temp_dir
os.environ["GDAL_PAM_PROXY_DIR"] = temp_dir
os.environ.setdefault("GDAL_CACHEMAX", "2048")

import rasterio
from rasterio.windows import Window
import spyndex

# Предупреждения numpy о делении на 0 и т.п. в формулах ожидаемы (такие пиксели -> NoData)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Стандартные имена Sentinel-2 ---------------------------------------------
s2_standard_9 = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]
s2_desc_9 = [
    "Blue 490nm", "Green 560nm", "Red 665nm",
    "RE1 705nm", "RE2 740nm", "RE3 783nm",
    "NIR 842nm", "SWIR1 1610nm", "SWIR2 2190nm"
]
s2_standard_13 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7",
                  "B8", "B8A", "B9", "B10", "B11", "B12"]
s2_desc_13 = [
    "Coastal 443nm", "Blue 490nm", "Green 560nm", "Red 665nm",
    "RE1 705nm", "RE2 740nm", "RE3 783nm",
    "NIR 842nm", "NIR-narrow 865nm", "WV 945nm",
    "Cirrus 1375nm", "SWIR1 1610nm", "SWIR2 2190nm"
]

# --- Константы, используемые некоторыми индексами (как в прежней версии) -------
CONSTS = {
    "L": 0.5,          # Soil brightness correction (SAVI)
    "g": 2.5,          # Gain factor (EVI)
    "C1": 6.0,         # Coefficient 1 (EVI)
    "C2": 7.5,         # Coefficient 2 (EVI)
    "cexp": 1.16,      # Canopy background (MSAVI)
    "nexp": 2.0,       # Power exponent
    "alpha": 0.1,      # Weighting coefficient
    "sla": 1.0,        # Soil line slope
    "slb": 0.0,        # Soil line intercept
    "gamma": 1.0,      # Gamma for ARVI
    "omega": 2.0,      # Weighting coefficient
    "k": 0.0,          # Slope factor
    "fdelta": 0.581,   # FCVI delta
    "lambdaN": 0.8420,  # NIR wavelength (µm) — Sentinel-2 B8
    "lambdaR": 0.6650,  # Red wavelength (µm) — Sentinel-2 B4
    "lambdaG": 0.5600,  # Green wavelength (µm) — Sentinel-2 B3
}

# Параметры, которые учитываются при отборе индексов (как в прежней версии)
KNOWN_PARAMS = ("B", "G", "R", "N", "N2", "RE1", "RE2", "RE3", "RE4",
                "S1", "S2", "T1", "T2", "T", "A", "WV",
                "lambdaN", "lambdaR", "lambdaG",
                "L", "g", "C1", "C2", "cexp", "nexp", "alpha",
                "sla", "slb", "gamma", "omega", "k", "fdelta", "PAR")

LINE = "════════════════════════════════════════════════════════"


def header(title):
    print("\n" + LINE)
    print("  " + title)
    print(LINE + "\n")


def fmt_time(sec):
    sec = int(round(sec))
    if sec < 60:
        return f"{sec} с"
    if sec < 3600:
        return f"{sec // 60} мин {sec % 60:02d} с"
    return f"{sec // 3600} ч {sec % 3600 // 60:02d} мин"


def physical_cores():
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def free_ram_bytes():
    try:
        import psutil
        return psutil.virtual_memory().available
    except Exception:
        return 8 * 2**30


# =============================================================
# ШАГ 0: ВЫБОР ФАЙЛОВ
# =============================================================

def _parse_selection(text, n):
    """'1,3,5-7' / 'all' / '' -> список номеров (1-based)."""
    text = text.strip().lower()
    if text == "":
        return [1]
    if text in ("all", "a", "*", "все"):
        return list(range(1, n + 1))
    out = []
    for part in re.split(r"[,\s;]+", text):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [i for i in out if i < 1 or i > n]
    if bad:
        raise ValueError(f"Неверные номера файлов: {bad}")
    return list(dict.fromkeys(out))


def choose_files():
    """Окно Проводника с мультивыбором; если окно недоступно — выбор номерами в консоли."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        files = filedialog.askopenfilenames(
            title="Выберите растры Sentinel-2 (можно несколько: Ctrl / Shift)",
            initialdir=input_dir,
            filetypes=[("GeoTIFF", "*.tif *.tiff *.TIF *.TIFF"), ("Все файлы", "*.*")])
        root.destroy()
        return [os.path.normpath(f) for f in files]
    except Exception as e:
        print(f"  [!] Окно выбора недоступно ({e}). Выбор по списку из папки INPUT.\n")
    tif_files = sorted(set(glob.glob(os.path.join(input_dir, "*.tif")) +
                           glob.glob(os.path.join(input_dir, "*.TIF"))))
    tif_files = [f for f in tif_files if not f.lower().endswith("_indices.tif")]
    if not tif_files:
        raise FileNotFoundError("В папке INPUT не найдено .tif файлов!")
    print("Доступные растры в папке INPUT:\n")
    for i, f in enumerate(tif_files, 1):
        with rasterio.open(f) as src:
            n_lyr = src.count
        print(f"  [{i}] {os.path.basename(f)}  ({n_lyr} каналов)")
    sel = _parse_selection(input("\nНомера растров через запятую, диапазон 2-5, all = все "
                                 "(Enter = 1): "), len(tif_files))
    return [tif_files[i - 1] for i in sel]


def read_info(path):
    """Каналы, размер, NoData и выборка значений 1-го канала (для scaleFactor)."""
    with rasterio.open(path) as src:
        n = src.count
        desc = list(src.descriptions)
        names = [d if d else f"B{i}" for i, d in enumerate(desc, 1)]
        h, w = src.height, src.width
        # Прореженное чтение 1-го канала: для оценки диапазона хватает ~1 млн пикселей
        f = max(1, int(np.ceil(max(h, w) / 1024)))
        b1 = src.read(1, out_shape=(max(1, h // f), max(1, w // f))).astype(np.float64)
        nodata = src.nodata
    vals = b1[np.isfinite(b1)]
    if nodata is not None:
        vals = vals[vals != nodata]
    if len(vals) > 10000:
        vals = np.random.default_rng(0).choice(vals, 10000, replace=False)
    return dict(path=path, name=os.path.basename(path), n=n, names=names, h=h, w=w,
                nodata=nodata, vmin=float(vals.min()) if len(vals) else None,
                vmax=float(vals.max()) if len(vals) else None)


# =============================================================
# ШАГ 1–2: ИМЕНА КАНАЛОВ И ПЕРЕИМЕНОВАНИЕ
# =============================================================

def standard_names(n):
    if n == 9:
        return s2_standard_9, s2_desc_9
    if n == 13:
        return s2_standard_13, s2_desc_13
    return None, None


def rename_bands(infos):
    """Один вопрос для всех файлов; ручное переименование — для каждой группы файлов
    с одинаковым набором каналов."""
    groups = {}
    for inf in infos:
        groups.setdefault(tuple(inf["names"]), []).append(inf)

    header("ТЕКУЩИЕ ИМЕНА КАНАЛОВ")
    for names, members in groups.items():
        fl = ", ".join(m["name"] for m in members)
        print(f"  {fl}  ({len(names)} каналов):")
        for i, name in enumerate(names, 1):
            print(f"    [{i:2d}] {name}")
        print()

    mode = BAND_NAMES
    if mode == "ask":
        header("ПЕРЕИМЕНОВАНИЕ КАНАЛОВ")
        print("  Варианты (для всех выбранных файлов):")
        print("  [Enter] Оставить текущие имена")
        print("  [s]     Применить стандартные Sentinel-2 (к файлам с 9 или 13 каналами):")
        print(f"          9:  {', '.join(s2_standard_9)}")
        print(f"          13: {', '.join(s2_standard_13)}")
        print("  [m]     Переименовать каждый канал вручную\n")
        ch = input("  Выбор: ").strip().lower()
        mode = {"s": "standard", "m": "manual"}.get(ch, "keep")

    if mode == "standard":
        for names, members in groups.items():
            std, desc = standard_names(len(names))
            fl = ", ".join(m["name"] for m in members)
            if std is None:
                print(f"\n  ! {fl}: стандартные имена только для 9 или 13 каналов "
                      f"(здесь {len(names)}), имена оставлены.")
                continue
            print(f"\n  ✓ {fl}: стандартные имена ({len(names)} каналов):")
            for i in range(len(std)):
                print(f"    [{i + 1}] {std[i]}  ← {names[i]}  ({desc[i]})")
            for m in members:
                m["names"] = list(std)
    elif mode == "manual":
        for names, members in groups.items():
            fl = ", ".join(m["name"] for m in members)
            std, desc = standard_names(len(names))
            print(f"\n  {fl}: переименование каналов (Enter = оставить текущее):\n")
            new = []
            for i, name in enumerate(names):
                hint = f" [подсказка: {std[i]} — {desc[i]}]" if std else ""
                inp = input(f"    [{i + 1:2d}] '{name}'{hint}\n"
                            f"         Новое имя (Enter = '{name}'): ").strip()
                new.append(inp if inp else name)
            print("\n  ✓ Каналы переименованы:")
            for i, (o, nn) in enumerate(zip(names, new), 1):
                print(f"    [{i:2d}] {nn}{' ← ИЗМЕНЕНО' if o != nn else ''}")
            for m in members:
                m["names"] = list(new)
    else:
        print("\n  ✓ Имена оставлены без изменений.")
    print()


# =============================================================
# ШАГ 3: МАППИНГ КАНАЛОВ ДЛЯ spyndex
# =============================================================
# Sentinel-2 → spyndex параметры:
#   B2 → B, B3 → G, B4 → R, B5 → RE1, B6 → RE2, B7 → RE3,
#   B8 → N, B11 → S1, B12 → S2

def find_band(patterns, names):
    """Ищет имя канала по шаблону regex, возвращает индекс (1-based) или None."""
    for p in patterns:
        matches = [i for i, n in enumerate(names) if re.match(p, n, re.IGNORECASE)]
        if len(matches) == 1:
            return matches[0] + 1
    return None


BAND_PATTERNS = [
    ("B", "blue", "490nm", [r"^B0?2$", r"^B02_", r"_B0?2$", r"blue"]),
    ("G", "green", "560nm", [r"^B0?3$", r"^B03_", r"_B0?3$", r"green"]),
    ("R", "red", "665nm", [r"^B0?4$", r"^B04_", r"_B0?4$", r"red"]),
    ("RE1", "redEdge1", "705nm", [r"^B0?5$", r"^B05_", r"_B0?5$", r"rededge1", r"RE1"]),
    ("RE2", "redEdge2", "740nm", [r"^B0?6$", r"^B06_", r"_B0?6$", r"rededge2", r"RE2"]),
    ("RE3", "redEdge3", "783nm", [r"^B0?7$", r"^B07_", r"_B0?7$", r"rededge3", r"RE3"]),
    ("N", "nir", "842nm", [r"^B0?8$", r"^B08_", r"_B0?8$", r"^B8A$", r"^B08A", r"nir"]),
    ("S1", "swir1", "1610nm", [r"^B11$", r"^B11_", r"_B11$", r"swir1", r"SWIR_1"]),
    ("S2", "swir2", "2190nm", [r"^B12$", r"^B12_", r"_B12$", r"swir2", r"SWIR_2"]),
]


def map_bands(inf):
    """{параметр spyndex: номер канала (1-based)}; печатает итоговый маппинг."""
    names, n = inf["names"], inf["n"]
    idx = {p: find_band(pat, names) for p, _, _, pat in BAND_PATTERNS}
    print(f"  {inf['name']}:")
    if idx["B"] is None and n >= 9:
        print("    ! Автоопределение не удалось → порядковая нумерация:")
        print("      1=Blue, 2=Green, 3=Red, 4=RE1, 5=RE2, 6=RE3, 7=NIR, 8=SWIR1, 9=SWIR2")
        idx = {p: i for i, (p, _, _, _) in enumerate(BAND_PATTERNS, 1)}
    for p, label, wl, _ in BAND_PATTERNS:
        i = idx[p]
        st, bl = ("✓", names[i - 1]) if i is not None else ("✗", "—")
        print(f"    {st} {label:<10s} → {bl:<6s}  ({wl})")
    n_mapped = sum(v is not None for v in idx.values())
    print(f"    Найдено каналов: {n_mapped} из 9\n")
    return {p: i for p, i in idx.items() if i is not None}


def computable_indices(band_params):
    """Индексы spyndex, для которых есть все параметры (логика прежней версии)."""
    available = set(band_params) | set(CONSTS)
    out = []
    for name, obj in spyndex.indices.items():
        req = {p for p in obj.bands if p in KNOWN_PARAMS}
        if req and all(p in available for p in req):
            out.append(name)
    if INDEX_LIST:
        want = list(dict.fromkeys(INDEX_LIST))
        miss = [x for x in want if x not in out]
        if miss:
            print(f"  ! Из INDEX_LIST нельзя рассчитать (нет каналов или нет в spyndex): "
                  f"{', '.join(miss)}")
        out = [x for x in out if x in set(want)]
    return out


def probe_indices(names, band_params):
    """Проверка формул на маленьком массиве: ошибки (нет параметра) и скалярные
    результаты не зависят от данных, такие индексы дальше не считаются."""
    rng = np.random.default_rng(0)
    fake = {p: rng.uniform(0.02, 0.5, (2, 2)).astype(np.float32) for p in band_params}
    plan, failed = [], {}
    for j, name in enumerate(names):
        obj = spyndex.indices[name]
        need = [p for p in obj.bands if p in fake or p in CONSTS]
        try:
            code = compile(obj.formula, f"<{name}>", "eval")
        except SyntaxError as e:
            failed[j] = ("ERROR", str(e)[:80])
            continue
        params = {p: fake[p] if p in fake else CONSTS[p] for p in need}
        missing = [p for p in obj.bands if p not in params]
        if missing:
            failed[j] = ("ERROR", f"'{missing[0]}' is missing in the parameters for "
                                  f"{name} computation!")
            continue
        try:
            with np.errstate(all="ignore"):
                r = eval(code, {}, params)
        except Exception as e:
            failed[j] = ("ERROR", str(e)[:80])
            continue
        if not isinstance(r, np.ndarray):
            failed[j] = ("SKIPPED", "скалярный результат")
            continue
        plan.append((j, code, need))
    return plan, failed


# =============================================================
# ВИДЕОКАРТА
# =============================================================

def pick_device():
    """None -> процессор; иначе torch.device('cuda:0')."""
    if USE_GPU is False:
        print("  Расчёт на процессоре (USE_GPU = False).")
        return None
    smi = []
    if shutil.which("nvidia-smi"):
        try:
            import subprocess
            r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                "--format=csv,noheader"], capture_output=True, text=True,
                               timeout=20)
            smi = [s.strip() for s in r.stdout.splitlines() if s.strip()]
        except Exception:
            pass
    why = ""
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.device("cuda:0")
            p = torch.cuda.get_device_properties(0)
            print(f"  Видеокарта: {p.name}, {p.total_memory / 2**30:.0f} ГБ "
                  f"(PyTorch {torch.__version__}, CUDA {torch.version.cuda})")
            return dev
        why = (f"PyTorch {torch.__version__} без CUDA" if not torch.version.cuda
               else "PyTorch не видит видеокарту")
    except Exception as e:
        why = f"PyTorch не установлен ({type(e).__name__})"
    if smi:
        why += f"; nvidia-smi: {'; '.join(smi)}"
        hint = ("\n    Карта есть, но PyTorch установлен без CUDA. Установка с CUDA:\n"
                "    pip install torch --index-url https://download.pytorch.org/whl/cu126")
    else:
        why += "; видеокарт NVIDIA не найдено"
        hint = ""
    if USE_GPU is True:
        raise RuntimeError(f"USE_GPU = True, но видеокарта недоступна: {why}{hint}")
    print(f"  Видеокарта не используется: {why}.{hint}\n  Расчёт на процессоре.")
    return None


# =============================================================
# РАСЧЁТ ОДНОГО БЛОКА
# =============================================================

class Job:
    """Всё, что нужно потоку для расчёта блоков одного файла."""

    def __init__(self, path, band_params, plan, n_out, sf, nodata, device):
        self.path = path
        self.band0 = {p: i - 1 for p, i in band_params.items()}   # 0-based
        self.plan = plan
        self.n_out = n_out
        self.sf = sf
        self.nodata = nodata
        self.device = device
        self.errors = {}                    # j -> текст ошибки при расчёте блоков
        self._local = threading.local()
        self._opened = []
        self._lock = threading.Lock()

    def src(self):
        ds = getattr(self._local, "ds", None)
        if ds is None:                      # у каждого потока свой дескриптор файла
            ds = rasterio.open(self.path)
            self._local.ds = ds
            with self._lock:
                self._opened.append(ds)
        return ds

    def close(self):
        for ds in self._opened:
            try:
                ds.close()
            except Exception:
                pass
        self._opened = []

    def read(self, win):
        data = self.src().read(window=win, out_dtype="float32")
        invalid = ~np.isfinite(data)
        if self.nodata is not None:
            invalid |= data == np.float32(self.nodata)
        mask = np.any(invalid, axis=0)
        if mask.all():
            return None, mask
        data[invalid] = np.nan
        if self.sf != 1:
            data /= np.float32(self.sf)
        return data, mask


def compute_cpu(job, win):
    h, w = win.height, win.width
    out = np.full((job.n_out, h, w), NODATA_OUT, dtype=np.float32)
    cnt = np.zeros(job.n_out, dtype=np.int64)
    data, mask = job.read(win)
    if data is None:
        return win, out, cnt
    bands = {p: data[i] for p, i in job.band0.items()}
    for j, code, need in job.plan:
        params = {p: bands[p] if p in bands else CONSTS[p] for p in need}
        try:
            with np.errstate(all="ignore"):
                r = eval(code, {}, params)
        except Exception as e:
            job.errors.setdefault(j, str(e)[:80])
            continue
        o = out[j]
        np.copyto(o, r, casting="unsafe")
        o[mask | ~np.isfinite(o)] = NODATA_OUT
        cnt[j] = np.count_nonzero(o != NODATA_OUT)
    return win, out, cnt


def compute_gpu(job, win):
    import torch
    h, w = win.height, win.width
    data, mask = job.read(win)
    if data is None:
        return (win, np.full((job.n_out, h, w), NODATA_OUT, dtype=np.float32),
                np.zeros(job.n_out, dtype=np.int64))
    dev = job.device
    t = torch.from_numpy(data).to(dev)
    tm = torch.from_numpy(mask).to(dev)
    bands = {p: t[i] for p, i in job.band0.items()}
    out_t = torch.full((job.n_out, h, w), NODATA_OUT, dtype=torch.float32, device=dev)
    done = torch.zeros(job.n_out, dtype=torch.bool)
    cpu_bands = None
    for j, code, need in job.plan:
        params = {p: bands[p] if p in bands else CONSTS[p] for p in need}
        try:
            r = eval(code, {}, params)
            if not torch.is_tensor(r):
                raise TypeError("не тензор")
            out_t[j] = r
        except Exception:
            # формула не работает с тензорами -> этот индекс для блока считается numpy
            if cpu_bands is None:
                cpu_bands = {p: data[i] for p, i in job.band0.items()}
            params = {p: cpu_bands[p] if p in cpu_bands else CONSTS[p] for p in need}
            try:
                with np.errstate(all="ignore"):
                    r = np.asarray(eval(code, {}, params), dtype=np.float32)
                out_t[j] = torch.from_numpy(np.ascontiguousarray(r)).to(dev)
            except Exception as e:
                job.errors.setdefault(j, str(e)[:80])
                continue
        done[j] = True
    bad = tm.unsqueeze(0) | ~torch.isfinite(out_t)
    out_t.masked_fill_(bad, NODATA_OUT)
    out_t[~done.to(dev)] = NODATA_OUT
    cnt = (out_t != NODATA_OUT).sum(dim=(1, 2)).cpu().numpy().astype(np.int64)
    out = out_t.cpu().numpy()
    del t, tm, out_t, bad
    return win, out, cnt


# =============================================================
# ОДИН ФАЙЛ
# =============================================================

def process_file(inf, k, n_files, device, n_workers):
    path = inf["path"]
    t0 = time.time()
    header(f"ФАЙЛ {k}/{n_files}: {inf['name']}")
    band_params = inf["band_params"]
    names = computable_indices(band_params)
    if not names:
        print("  ! Нет индексов для расчёта — файл пропущен.")
        return dict(name=inf["name"], status="нет индексов")
    plan, failed = probe_indices(names, band_params)
    n_out = len(names)
    print(f"  Доступно для расчёта: {n_out} индексов "
          f"(из них не считаются по формуле: {len(failed)})")

    base = os.path.splitext(inf["name"])[0]
    out_dir = os.path.dirname(path)
    out_name = f"{base}_indices.tif"
    out_path = os.path.join(out_dir, out_name)
    part_path = os.path.join(out_dir, f"{base}_indices.part.tif")

    if os.path.exists(out_path):
        if not OVERWRITE:
            print(f"  {out_name} уже есть — пропуск (OVERWRITE = False).")
            return dict(name=inf["name"], status="уже есть", out=out_path)
        try:
            os.remove(out_path)
            print(f"  ⚠ Старый файл {out_name} удалён")
        except OSError:
            print(f"  ⚠ Не удалось удалить {out_name} — возможно, он открыт в другой программе.")
            print("    Закройте QGIS/ArcGIS и запустите этот файл ещё раз. Файл пропущен.")
            return dict(name=inf["name"], status="файл заблокирован")
    if os.path.exists(part_path):
        os.remove(part_path)

    h, w = inf["h"], inf["w"]
    raw_gb = n_out * h * w * 4 / 2**30
    free_gb = shutil.disk_usage(out_dir).free / 2**30
    print(f"  Растр {w} x {h} пикс., каналов {inf['n']}, scaleFactor = {inf['sf']}")
    print(f"  Объём индексов без сжатия: {raw_gb:.1f} ГБ; свободно на диске: {free_gb:.0f} ГБ")
    if free_gb < raw_gb * 0.8:
        print("  ! Места на диске может не хватить.")

    with rasterio.open(path) as src:
        profile = src.profile.copy()
    for key in ["photometric", "alpha", "mask", "interleave", "compress", "predictor",
                "blockxsize", "blockysize", "tiled", "zlevel", "zstd_level", "jpeg_quality"]:
        profile.pop(key, None)
    profile.update(driver="GTiff", count=n_out, dtype="float32", nodata=NODATA_OUT,
                   tiled=True, blockxsize=256, blockysize=256, interleave="band",
                   BIGTIFF="IF_SAFER", num_threads="ALL_CPUS")
    if COMPRESS and str(COMPRESS).upper() != "NONE":
        profile["compress"] = str(COMPRESS).lower()
        if profile["compress"] == "deflate":
            profile["zlevel"] = 1
        elif profile["compress"] == "zstd":
            profile["zstd_level"] = 1
        if PREDICTOR and int(PREDICTOR) != 1:
            profile["predictor"] = int(PREDICTOR)

    blk = max(256, int(BLOCK) // 256 * 256)
    wins = [Window(c, r, min(blk, w - c), min(blk, h - r))
            for r in range(0, h, blk) for c in range(0, w, blk)]

    # потоки и число блоков «в работе» с учётом свободной памяти
    per_blk = (n_out + 3 * inf["n"] + 8) * blk * blk * 4
    cap = max(2, int(free_ram_bytes() * 0.5 // per_blk))
    nw = max(1, min(n_workers, cap - 1, len(wins)))
    if device is not None:
        nw = max(1, min(3, nw))             # видеокарта: 3 потока (чтение + расчёт + выгрузка)
    inflight = max(1, min(nw + 2, cap))
    job = Job(path, band_params, plan, n_out, inf["sf"], inf["nodata"], device)
    fn = compute_gpu if device is not None else compute_cpu
    print(f"  Блоков: {len(wins)} по {blk} x {blk} пикс.; потоков расчёта: {nw} "
          f"({'видеокарта' if device is not None else 'процессор'}); "
          f"сжатие {profile.get('compress', 'none').upper()}"
          f"{', predictor ' + str(profile['predictor']) if 'predictor' in profile else ''}\n")

    counts = np.zeros(n_out, dtype=np.int64)
    done_px, total_px = 0, h * w
    t_last = time.time()
    p_last = 0.0
    ok = False
    try:
        with rasterio.open(part_path, "w", **profile) as dst, \
                ThreadPoolExecutor(max_workers=nw) as ex:
            it = iter(wins)
            pending = set()

            def submit():
                win = next(it, None)
                if win is not None:
                    pending.add(ex.submit(fn, job, win))

            for _ in range(inflight):
                submit()
            n_done = 0
            while pending:
                fin, pending = wait(pending, return_when=FIRST_COMPLETED)
                for f in fin:
                    win, out, cnt = f.result()
                    dst.write(out, window=win)
                    counts += cnt
                    del out
                    n_done += 1
                    done_px += win.height * win.width
                    submit()
                    frac = done_px / total_px
                    now = time.time()
                    if n_done == len(wins) or frac - p_last >= 0.05 or now - t_last >= 30:
                        el = now - t0
                        eta = el / frac * (1 - frac) if frac > 0 else 0
                        mpx = done_px / 1e6 / max(el, 1e-6)
                        print(f"    [файл {k}/{n_files} | блоки {n_done}/{len(wins)} "
                              f"{frac:6.1%}] {mpx:.2f} Мпикс/с | прошло {fmt_time(el)}, "
                              f"осталось ~{fmt_time(eta)}", flush=True)
                        t_last, p_last = now, frac

            # имена каналов
            failed_all = dict(failed)
            for j, msg in job.errors.items():
                failed_all.setdefault(j, ("ERROR", msg))
            written = 0
            for j, name in enumerate(names):
                layer = f"S2_{name}"
                if j in failed_all:
                    dst.set_band_description(j + 1, f"{layer}_{failed_all[j][0]}")
                elif counts[j] == 0:
                    dst.set_band_description(j + 1, f"{layer}_SKIPPED")
                    failed_all[j] = ("SKIPPED", "все значения NaN/Inf")
                else:
                    dst.set_band_description(j + 1, layer)
                    written += 1
            dst.update_tags(source=inf["name"], scale_factor=str(inf["sf"]),
                            spyndex_version=getattr(spyndex, "__version__", "?"))
        ok = True
    finally:
        job.close()
        if not ok and os.path.exists(part_path):
            try:
                os.remove(part_path)
            except OSError:
                pass
    os.replace(part_path, out_path)

    el = time.time() - t0
    print(f"\n  ✓ Записано индексов с данными: {written}")
    if failed_all:
        print(f"    Индексов без данных/с ошибками: {len(failed_all)}")
        print("    Первые проблемные индексы:")
        for j in sorted(failed_all)[:20]:
            print(f"      - {names[j]}: {failed_all[j][1]}")
        if len(failed_all) > 20:
            print(f"      ... ещё {len(failed_all) - 20}")
    if written == 0:
        print("  ! Не записано ни одного индекса с валидными пикселями. Проверьте, что выбран "
              "правильный растр и scaleFactor.")
    size_mb = os.path.getsize(out_path) / 2**20
    print(f"  ✓ Файл:    {out_name}")
    print(f"    Размер:  {size_mb:.2f} MB")
    print(f"    Каналов: {n_out}")
    print(f"    Время:   {fmt_time(el)}")
    print(f"    Путь:    {out_path}")
    return dict(name=inf["name"], status="ok" if written else "нет данных", out=out_path,
                written=written, n_out=n_out, size_mb=size_mb, sec=el)


# =============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================

def main():
    header("РАСЧЁТ СПЕКТРАЛЬНЫХ ИНДЕКСОВ (Sentinel-2)")
    files = choose_files()
    files = [f for f in files if f]
    skip = [f for f in files if f.lower().endswith("_indices.tif")]
    for f in skip:
        print(f"  ! {os.path.basename(f)} — это уже результат расчёта индексов, пропущен.")
    files = [f for f in files if f not in skip]
    if not files:
        print("  Файлы не выбраны — расчёт не запускался.")
        return

    print(f"  Выбрано файлов: {len(files)}\n")
    infos = []
    for f in files:
        try:
            inf = read_info(f)
        except Exception as e:
            print(f"  ! {os.path.basename(f)}: не удалось открыть ({e}) — пропущен.")
            continue
        infos.append(inf)
        print(f"  ✓ {inf['name']}  ({inf['n']} каналов, {inf['w']} x {inf['h']} пикс.)")
    if not infos:
        return

    # ШАГ 1–2: имена каналов
    rename_bands(infos)

    # ШАГ 3: маппинг
    header("МАППИНГ КАНАЛОВ ДЛЯ spyndex")
    ready = []
    for inf in infos:
        bp = map_bands(inf)
        if len(bp) < 3:
            print(f"  ! {inf['name']}: слишком мало каналов (нужно минимум blue+red+nir) "
                  f"— файл пропущен.\n")
            continue
        inf["band_params"] = bp
        ready.append(inf)
    if not ready:
        raise ValueError("Ни в одном файле не найдено нужных каналов.")

    # ШАГ 4: scaleFactor
    header("МАСШТАБНЫЙ КОЭФФИЦИЕНТ (scaleFactor)")
    for inf in ready:
        vmin, vmax = inf["vmin"], inf["vmax"]
        if vmax is None:
            inf["suggested_sf"] = None
            print(f"  {inf['name']}: в 1-м канале нет числовых значений (NaN/NoData)")
            continue
        if vmax > 100:
            sf, why = 10000, "похоже на DN (0–10000)"
        elif vmax <= 1.0 and vmin >= 0:
            sf, why = 1, "похоже на reflectance [0, 1]"
        else:
            sf, why = 10000, "неясный диапазон, по умолчанию"
        inf["suggested_sf"] = sf
        print(f"  {inf['name']}: диапазон 1-го канала [{vmin:.2f}, {vmax:.2f}] → {why} "
              f"→ рекомендуется {sf}")
    ready = [r for r in ready if r["suggested_sf"] is not None]
    if not ready:
        raise ValueError("В первом канале нет конечных числовых значений. Проверьте входные "
                         "растры: вероятно, они состоят из NaN/NoData.")

    sf_all = None
    if SCALE_FACTOR == "ask":
        s = input("\n  scaleFactor для всех файлов (Enter = рекомендованный для каждого): ").strip()
        if s:
            try:
                sf_all = int(s)
                if sf_all <= 0:
                    raise ValueError
            except ValueError:
                print("  ! Некорректный ввод, использую рекомендованные значения")
                sf_all = None
    elif SCALE_FACTOR not in (None, "auto"):
        sf_all = int(SCALE_FACTOR)
    for inf in ready:
        inf["sf"] = sf_all if sf_all else inf["suggested_sf"]
        print(f"  ✓ {inf['name']}: scaleFactor = {inf['sf']}")

    # Процессор / видеокарта
    header("РЕСУРСЫ")
    nw = physical_cores() if N_WORKERS in (None, "auto") else max(1, int(N_WORKERS))
    print(f"  Ядер: физических {physical_cores()}, логических {os.cpu_count()}; "
          f"потоков расчёта: {nw}; свободно ОЗУ {free_ram_bytes() / 2**30:.0f} ГБ")
    device = pick_device()

    # ШАГ 5: расчёт
    t_all = time.time()
    results = []
    for k, inf in enumerate(ready, 1):
        try:
            results.append(process_file(inf, k, len(ready), device, nw))
        except KeyboardInterrupt:
            print("\n  Остановлено пользователем.")
            results.append(dict(name=inf["name"], status="остановлено"))
            break
        except Exception as e:
            print(f"\n  ! {inf['name']}: ошибка — {e}")
            results.append(dict(name=inf["name"], status=f"ошибка: {str(e)[:80]}"))

    header("ИТОГ")
    for r in results:
        if r.get("status") == "ok":
            print(f"  ✓ {r['name']}: {r['written']}/{r['n_out']} индексов, "
                  f"{r['size_mb']:.0f} MB, {fmt_time(r['sec'])} → {os.path.basename(r['out'])}")
        else:
            print(f"  ✗ {r['name']}: {r['status']}")
    print(f"\n  Всего: {fmt_time(time.time() - t_all)}")
    print("  Готово.\n")


if __name__ == "__main__":
    main()
