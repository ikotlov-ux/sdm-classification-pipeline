# =============================================================
# 002_DEM_Indices_v7.py (v2)
# Расчёт морфометрических, мультимасштабных и гидрологических индексов из DEM
# WhiteboxTools + точные векторные производные Флоринского (5x5) и Хорна (3x3)
# ВЫВОД: отдельные многослойные GeoTIFF для каждой группы индексов
# =============================================================
# Исправления по сравнению с v1 (v6):
# 1. Автоматическая диагностика ступенчатых/целочисленных ЦМР (например, SRTM,
#    перепроецированный методом ближайшего соседа на сетку 10 м в int16) и
#    опциональное гауссово предсглаживание (de-stepping) без потери границ NoData.
# 2. Устранён баг WhiteboxTools.aspect (if fx != 0.0), из-за которого все
#    склоны строго северной (0 град.) и южной (180 град.) экспозиции (fx == 0,
#    fy != 0) получали код плоской поверхности -1.0.
# 3. Устранена ошибка расчёта DEM_eastness и DEM_northness на плоских участках
#    (aspect == -1.0): раньше считались sin(-1 град.) = -0.01745 и
#    cos(-1 град.) = +0.99985, из-за чего все плоские участки и северные/южные
#    склоны кодировались как крутые северные склоны (+0.99985). Теперь на
#    плоских участках eastness = 0.0 и northness = 0.0.
# 4. Устранён тот же баг (fx == 0) в DEM_hillshade, DEM_multihillshade и
#    DEM_circvar_aspect_w*, а также устранены 36% дыр NoData в
#    circular_variance_of_aspect на плоских участках.
# 5. Перед расчётом DEM_flowdir (D8 pointer), SCA, DEM_TWI, DEM_STI и DEM_SPI
#    выполняется гидрологическая коррекция микрозападин и плоских участков
#    (breach_depressions_least_cost + fill_depressions с fix_flats=True), что
#    устраняет обрыв стока (78.6% нулей в DEM_flowdir). В DEM_TWI на плоских
#    участках задаётся минимальный уклон tan(beta) >= 0.001 вместо записи NoData
#    (-32768), которая раньше вырезала 28.6% территории в композите.
# 6. В DEM_planc (плановая кривизна) добавлена регуляризация знаменателя при
#    околонулевых уклонах (< 0.1 град.), устраняющая сингулярные выбросы
#    1 / sin(slope) -> +-10^4 на седловинах и плоских террасах.
# 7. Оконные индексы TPI, TRI, DMV, DevFromMean, RTP, PercentElevRange,
#    MaxDifferenceFromMean переведены на точный расчёт в float64 (без
#    внутреннего округления высот до 1 см в целочисленном интегральном
#    изображении WhiteboxTools и без зажима min_scale >= 3, max_scale >= 5,
#    делавшего слои w3 и w5 идентичными).
# 8. Единое значение NoData (-32768.0) во всех каналах выходных файлов и
#    размещение временной папки вне облачных папок (Яндекс.Диск / OneDrive).
# =============================================================

import os
import sys
import glob
import math
import shutil
import tempfile
import platform
import multiprocessing
import numpy as np
import rasterio
from scipy.ndimage import (
    gaussian_filter,
    correlate,
    uniform_filter,
    minimum_filter,
    maximum_filter,
    distance_transform_edt,
)
from whitebox import WhiteboxTools

# --- Настройка путей ---
input_dir = "INPUT"
output_dir = "INPUT"
NODATA_VAL = -32768.0

# --- Инициализация WhiteboxTools ---
wbt = WhiteboxTools()
wbt.set_verbose_mode(False)
print(f"WhiteboxTools version: {wbt.version()}")

# =============================================================
# НАСТРОЙКА ВРЕМЕННОЙ ПАПКИ (ВНЕ ЯНДЕКС.ДИСКА)
# =============================================================


def _get_free_gb(path):
    try:
        os.makedirs(path, exist_ok=True)
        if platform.system() == "Windows":
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(path), None, None, ctypes.pointer(free_bytes)
            )
            return free_bytes.value / (1024 ** 3)
        else:
            st = os.statvfs(path)
            return (st.f_bavail * st.f_frsize) / (1024 ** 3)
    except Exception:
        return 0.0


def _choose_temp_dir():
    candidates = []
    sys_tmp = os.path.join(tempfile.gettempdir(), "TEMP_dem_indices")
    candidates.append(sys_tmp)
    if platform.system() == "Windows":
        for drv in ("C", "D", "E", "F", "G", "H", "I"):
            root = f"{drv}:\\"
            if os.path.exists(root):
                candidates.append(os.path.join(root, "TEMP_dem_indices"))
    best_path = sys_tmp
    best_free = -1.0
    for c in candidates:
        fgb = _get_free_gb(c)
        if fgb > best_free:
            best_free = fgb
            best_path = c
    os.makedirs(best_path, exist_ok=True)
    return best_path, max(0.0, best_free)


temp_dir, free_gb = _choose_temp_dir()
wbt.set_working_dir(temp_dir)
print(f"Временная папка: {temp_dir} (свободно: {free_gb:.1f} GB)\n")

_keep_files = {"dem_input.tif", "dem_hydro.tif", "tmp_slope.tif", "tmp_flowacc.tif"}


def clean_temp(silent=False, remove_all=False):
    """Удаляет временные файлы из temp_dir."""
    removed = 0
    for f in glob.glob(os.path.join(temp_dir, "*.tif")):
        if not remove_all and os.path.basename(f) in _keep_files:
            continue
        try:
            os.remove(f)
            removed += 1
        except OSError:
            pass
    if not silent and removed > 0:
        print(f"  Очищено временных файлов: {removed}")


# =============================================================
# ВЫБОР КОЛИЧЕСТВА ПОТОКОВ
# =============================================================
print("========================================================")
print("  КОЛИЧЕСТВО ПОТОКОВ (ПАРАЛЛЕЛИЗАЦИЯ)")
print("========================================================\n")

n_cores_logical = multiprocessing.cpu_count()
n_cores_physical = max(1, n_cores_logical // 2)
recommended = max(1, n_cores_physical - 1)

print(f"  Ядер (логических):  {n_cores_logical}")
print(f"  Ядер (физических):  ~{n_cores_physical}")
print(f"  Рекомендуемо:       {recommended}  (физические - 1)\n")
print("  Варианты:")
print("    1   - последовательный режим")
print(f"    {recommended}   - рекомендуемое")
print(f"    {n_cores_logical}  - все логические (макс.)\n")

n_threads_input = input(f"Введите число потоков (Enter = {recommended}): ").strip()
if n_threads_input == "":
    n_threads = recommended
else:
    try:
        n_threads = int(n_threads_input)
        if n_threads < 1:
            raise ValueError
    except ValueError:
        print("\nНекорректное значение. Используется рекомендуемое.")
        n_threads = recommended
    if n_threads > n_cores_logical:
        print(f"\nБольше доступных ({n_cores_logical}). Ограничиваю.")
        n_threads = n_cores_logical

print(f"\nПотоков: {n_threads}\n")
wbt.set_max_procs(n_threads)

# =============================================================
# ШАГ 1: ВЫБОР ФАЙЛА DEM И ДИАГНОСТИКА СТУПЕНЧАТОСТИ
# =============================================================

print("\n========================================================")
print("  РАСЧЁТ МОРФОМЕТРИЧЕСКИХ ИНДЕКСОВ ИЗ DEM")
print("========================================================\n")


def _unique_tif_list(directory):
    seen = set()
    result = []
    for pattern in ("*.tif", "*.TIF"):
        for path in sorted(glob.glob(os.path.join(directory, pattern))):
            key = os.path.normcase(os.path.abspath(path))
            if key not in seen:
                seen.add(key)
                result.append(path)
    return result


tif_files = _unique_tif_list(input_dir)
if len(tif_files) == 0:
    raise FileNotFoundError(f"В папке {input_dir} не найдено .tif файлов.")

# Определяем файл по умолчанию (первый растр, который не является уже посчитанным DEM_*_terrain/curvatures/multiscale/extra)
default_idx = 1
_derived_suffixes = ("_terrain.tif", "_curvatures.tif", "_multiscale.tif", "_extra.tif")
for idx, f in enumerate(tif_files, 1):
    bname = os.path.basename(f).lower()
    if not (bname.startswith("dem_") and bname.endswith(_derived_suffixes)):
        default_idx = idx
        break

print(f"Доступные растры в папке {input_dir}:\n")
for i, f in enumerate(tif_files, 1):
    with rasterio.open(f) as src:
        n_lyr = src.count
        dt = src.dtypes[0]
    mark = "  <-- по умолчанию" if i == default_idx else ""
    print(f"  [{i}] {os.path.basename(f)}  ({n_lyr} кан., {dt}){mark}")

choice = input(f"\nВыберите растр с DEM (Enter = {default_idx}): ").strip()
if choice == "":
    choice = default_idx
else:
    choice = int(choice)
if choice < 1 or choice > len(tif_files):
    raise ValueError("Неверный номер файла.")

input_path = tif_files[choice - 1]
base_name = os.path.splitext(os.path.basename(input_path))[0]
print(f"  Выбран: {os.path.basename(input_path)}\n")

# --- Загрузка DEM ---
with rasterio.open(input_path) as src_ds:
    n_bands = src_ds.count
    if n_bands > 1:
        print(f"  Растр содержит {n_bands} каналов:")
        descriptions = src_ds.descriptions
        for i in range(n_bands):
            desc = descriptions[i] if descriptions[i] else f"Band {i+1}"
            print(f"    [{i+1}] {desc}")
        lyr_choice = input("\n  Какой канал содержит высоту? (Enter = 1): ").strip()
        band_idx = 1 if lyr_choice == "" else int(lyr_choice)
    else:
        band_idx = 1

    raw_band = src_ds.read(band_idx)
    src_dtype = src_ds.dtypes[band_idx - 1]
    dem_raw = raw_band.astype(np.float64)
    profile = src_ds.profile.copy()
    src_nodata = src_ds.nodata
    src_crs = src_ds.crs
    src_bounds = src_ds.bounds
    res_x_raw, res_y_raw = abs(src_ds.res[0]), abs(src_ds.res[1])
    height, width = src_ds.height, src_ds.width

# Определяем физический размер пикселя в метрах (с учётом географических СК EPSG:4326)
is_geographic = bool(src_crs is not None and src_crs.is_geographic)
if is_geographic or (res_x_raw < 0.1 and res_y_raw < 0.1):
    lat_c = 0.5 * (src_bounds.bottom + src_bounds.top)
    res_x_m = res_x_raw * 111320.0 * max(0.05, math.cos(math.radians(lat_c)))
    res_y_m = res_y_raw * 110574.0
    zfactor = 1.0 / math.sqrt(res_x_m * res_y_m)
    print(f"  ВНИМАНИЕ: географическая СК ({src_crs}), шаг пересчитан в метры: {res_x_m:.2f} x {res_y_m:.2f} м (zfactor={zfactor:.6g})")
else:
    res_x_m, res_y_m = res_x_raw, res_y_raw
    zfactor = 1.0

print(f"  Размер:     {height} x {width} пикс. (тип: {src_dtype})")
print(f"  Разрешение: {res_x_m:.2f} x {res_y_m:.2f} м")

# Формируем точную маску NoData исходного DEM
dem_nodata_mask = ~np.isfinite(dem_raw)
if src_nodata is not None and not np.isnan(src_nodata):
    dem_nodata_mask |= (dem_raw == float(src_nodata))
# Дополнительная защита от стандартных кодов NoData (-32768, -9999), если они не прописаны в теге
finite_vals = dem_raw[~dem_nodata_mask]
if finite_vals.size > 0 and np.percentile(finite_vals, 5) > -500.0:
    for nd_cand in (-32768.0, -9999.0):
        if np.any(dem_raw == nd_cand):
            dem_nodata_mask |= (dem_raw == nd_cand)

valid_vals = dem_raw[~dem_nodata_mask]
if valid_vals.size == 0:
    raise RuntimeError("В выбранном канале DEM нет валидных пикселей.")

n_nodata_dem = int(np.sum(dem_nodata_mask))
print(f"  Диапазон высот: [{valid_vals.min():.2f}, {valid_vals.max():.2f}] м | NoData: {n_nodata_dem} пикс. ({100.0 * n_nodata_dem / dem_raw.size:.2f}%)")

# Заполняем NoData ближайшими валидными высотами для расчёта локальных окон без краевой эрозии
if n_nodata_dem > 0:
    _, nearest_idx = distance_transform_edt(dem_nodata_mask, return_indices=True)
    dem_filled_nd = dem_raw[tuple(nearest_idx)]
else:
    dem_filled_nd = dem_raw.copy()

# --- Диагностика ступенчатости (Nearest-Neighbor ресэмплинг грубой ЦМР / целочисленный int16) ---
dx_zero_frac = float(np.mean(np.diff(dem_filled_nd, axis=1) == 0.0))
dy_zero_frac = float(np.mean(np.diff(dem_filled_nd, axis=0) == 0.0))
sample_vals = valid_vals if valid_vals.size <= 200000 else valid_vals[:: max(1, valid_vals.size // 200000)]
is_integer_dem = bool(np.all(np.mod(sample_vals, 1.0) == 0.0))
step_x_px = 1.0 / max(1e-4, 1.0 - dx_zero_frac)
step_y_px = 1.0 / max(1e-4, 1.0 - dy_zero_frac)

# Доля полностью плоских окон 3x3
z_min_3 = minimum_filter(dem_filled_nd, size=3, mode="nearest")
z_max_3 = maximum_filter(dem_filled_nd, size=3, mode="nearest")
flat_3x3_frac = float(np.mean((z_max_3 - z_min_3)[~dem_nodata_mask] == 0.0))
del z_min_3, z_max_3

has_step_artifacts = (dx_zero_frac > 0.50) or (dy_zero_frac > 0.50) or (flat_3x3_frac > 0.20) or is_integer_dem
if has_step_artifacts:
    sigma_rec = round(max(1.0, min(8.0, 0.35 * max(step_x_px, step_y_px))), 1)
    print("\n  --------------------------------------------------------")
    print("  ДИАГНОСТИКА КАЧЕСТВА DEM: ОБНАРУЖЕНА СТУПЕНЧАТОСТЬ")
    print("  --------------------------------------------------------")
    print(f"  Целочисленные высоты (шаг 1 м):     {'ДА' if is_integer_dem else 'НЕТ'}")
    print(f"  Доля нулевых разностей по X / Y:    {100*dx_zero_frac:.1f}% / {100*dy_zero_frac:.1f}%")
    print(f"  Средний размер плоской ступеньки:   ~{step_x_px:.1f} x {step_y_px:.1f} пикс. (~{step_x_px*res_x_m:.0f} м З-В x {step_y_px*res_y_m:.0f} м С-Ю)")
    print(f"  Доля абсолютно плоских окон 3x3:    {100*flat_3x3_frac:.1f}%")
    print("  Причина: ЦМР более грубого разрешения (например, SRTM 30-90 м)")
    print("  была перепроецирована методом ближайшего соседа (Nearest Neighbor)")
    print("  и/или округлена до целых метров (int16).")
    print("  Без сглаживания производные 3x3 и 5x5 будут показывать")
    print("  прямоугольную сетку ступенек вместо реального рельефа.")
    print(f"\n  Рекомендуемое гауссово предсглаживание: sigma = {sigma_rec} пикс. ({sigma_rec * res_x_m:.0f} м)")
    print("  (Введите 0, если хотите считать по ступенчатому DEM без сглаживания)")
    s_in = input(f"  Параметр сглаживания sigma в пикселях (Enter = {sigma_rec}): ").strip()
    if s_in == "":
        smooth_sigma = sigma_rec
    else:
        try:
            smooth_sigma = max(0.0, float(s_in.replace(",", ".")))
        except ValueError:
            smooth_sigma = sigma_rec
else:
    print(f"\n  Качество DEM: гладкая поверхность с плавающей точкой (плоских окон 3x3: {100*flat_3x3_frac:.2f}%).")
    s_in = input("  Гауссово предсглаживание sigma в пикселях (Enter = 0, без сглаживания): ").strip()
    if s_in == "":
        smooth_sigma = 0.0
    else:
        try:
            smooth_sigma = max(0.0, float(s_in.replace(",", ".")))
        except ValueError:
            smooth_sigma = 0.0

if smooth_sigma > 0.0:
    # Нормированное гауссово сглаживание без проникновения NoData внутрь области
    if n_nodata_dem > 0:
        valid_w = (~dem_nodata_mask).astype(np.float64)
        num = gaussian_filter(np.where(dem_nodata_mask, 0.0, dem_raw), sigma=smooth_sigma, mode="nearest")
        den = gaussian_filter(valid_w, sigma=smooth_sigma, mode="nearest")
        dem_work = np.where(den > 1e-6, num / den, dem_filled_nd)
        del valid_w, num, den
    else:
        dem_work = gaussian_filter(dem_filled_nd, sigma=smooth_sigma, mode="nearest")
    diff_std = float(np.std((dem_work - dem_raw)[~dem_nodata_mask]))
    # Округляем до 1 мм (0.001 м), чтобы убрать субнанометровые хвосты (1e-9 м) после фильтра Гаусса,
    # из-за которых в WBT AverageNormalVectorAngularDeviation возникает acos(1 + 1e-16) -> NaN в интегральном изображении
    dem_work = np.round(dem_work, 3)
    print(f"  Выполнено сглаживание DEM (sigma={smooth_sigma} пикс. = {smooth_sigma*res_x_m:.1f} м, СКО коррекции ступенек: {diff_std:.2f} м)\n")
else:
    dem_work = np.round(dem_filled_nd, 3)
    print("  Предсглаживание отключено (sigma = 0).\n")

del dem_raw, dem_filled_nd
dem_mean_global = float(np.mean(dem_work[~dem_nodata_mask]))
dem_centered = dem_work - dem_mean_global

# Сохраняем подготовленный однослойный DEM во временную папку (несжатый/LZW с явным nodata=-32768.0)
dem_tmp_path = os.path.join(temp_dir, "dem_input.tif")
single_profile = {
    "driver": "GTiff",
    "height": height,
    "width": width,
    "count": 1,
    "dtype": "float64",
    "crs": src_crs,
    "transform": profile["transform"],
    "nodata": NODATA_VAL,
    "compress": "lzw",
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
}
if height * width * 8 > 3.5 * (1024 ** 3):
    single_profile["BIGTIFF"] = "YES"

dem_for_wbt = dem_work.copy()
if n_nodata_dem > 0:
    dem_for_wbt[dem_nodata_mask] = NODATA_VAL
with rasterio.open(dem_tmp_path, "w", **single_profile) as dst:
    dst.write(dem_for_wbt, 1)
del dem_for_wbt

# =============================================================
# ШАГ 2: ВЫБОР РАЗМЕРОВ ОКНА
# =============================================================

print("========================================================")
print("  РАЗМЕР СКОЛЬЗЯЩЕГО ОКНА")
print("========================================================\n")

print("  Варианты:")
print("  [3]    Окно 3x3 (стандарт)")
print("  [5]    Окно 5x5")
print("  [7]    Окно 7x7")
print("  [0]    Три окна: 3, 5, 7 (мультимасштабный анализ)\n")

w_input = input("  Выбор (Enter = 3): ").strip()
if w_input == "":
    w_input = "3"

if w_input == "0":
    win_sizes = [3, 5, 7]
    print("  Выбрано три окна: 3x3, 5x5, 7x7\n")
else:
    ws = int(w_input)
    if ws < 3 or ws % 2 == 0:
        print("  Некорректный размер окна, используется 3x3")
        ws = 3
    win_sizes = [ws]
    print(f"  Выбрано окно: {ws}x{ws}\n")


# =============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================


def _needs_bigtiff(h, w, count, dtype_bytes=4):
    return h * w * count * dtype_bytes > 3.5 * (1024 ** 3)


def save_temp_array(arr, out_name):
    """Сохраняет 2D float32-массив во временный GeoTIFF с единым NODATA_VAL."""
    out_tmp = os.path.join(temp_dir, out_name)
    out = np.asarray(arr, dtype=np.float32).copy()
    bad = ~np.isfinite(out)
    if n_nodata_dem > 0:
        bad |= dem_nodata_mask
    out[bad] = NODATA_VAL
    prof = single_profile.copy()
    prof.update(dtype="float32", nodata=NODATA_VAL)
    with rasterio.open(out_tmp, "w", **prof) as dst:
        dst.write(out, 1)
    return out_tmp


def stack_and_save(layer_paths, layer_names, out_path, fallback_zero_on_valid=True):
    """Собирает несколько однослойных TIF в один многослойный файл с единым NODATA_VAL."""
    valid = [(p, n) for p, n in zip(layer_paths, layer_names) if p and os.path.exists(p)]
    if not valid:
        return False

    out_profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": len(valid),
        "dtype": "float32",
        "crs": src_crs,
        "transform": profile["transform"],
        "nodata": NODATA_VAL,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    if _needs_bigtiff(height, width, len(valid), dtype_bytes=4):
        out_profile["BIGTIFF"] = "YES"

    if os.path.exists(out_path):
        try:
            os.remove(out_path)
        except OSError:
            pass

    with rasterio.open(out_path, "w", **out_profile) as dst:
        for i, (path, name) in enumerate(valid, 1):
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
                snd = src.nodata
            # Определяем пиксели, где инструмент вернул NoData / NaN / Inf
            is_nd = ~np.isfinite(arr) | (arr == NODATA_VAL) | (arr == -9999.0)
            if snd is not None and not np.isnan(snd):
                is_nd |= (arr == float(snd))
            # На валидных пикселях DEM заполняем краевые/внутренние дыры NoData ближайшим валидным соседом
            if fallback_zero_on_valid:
                internal_holes = is_nd & (~dem_nodata_mask)
                if np.any(internal_holes):
                    if np.all(internal_holes):
                        arr[internal_holes] = 0.0
                    else:
                        _, near_idx = distance_transform_edt(internal_holes, return_indices=True)
                        arr[internal_holes] = arr[tuple(near_idx)][internal_holes]
            # На истинных NoData исходного DEM ставим строго NODATA_VAL
            if n_nodata_dem > 0:
                arr[dem_nodata_mask] = NODATA_VAL
            dst.write(arr, i)
            dst.set_band_description(i, name)

    for path, _ in valid:
        if os.path.basename(path) in _keep_files:
            continue
        try:
            os.remove(path)
        except OSError:
            pass

    return True


def run_wbt(tool_func, out_name, **kwargs):
    """Запускает инструмент WhiteboxTools, возвращает путь к результату или None."""
    out_tmp = os.path.join(temp_dir, out_name)
    fname = getattr(tool_func, "__name__", out_name)
    try:
        if os.path.exists(out_tmp):
            try:
                os.remove(out_tmp)
            except OSError:
                pass
        ret = tool_func(output=out_tmp, **kwargs)
        if os.path.exists(out_tmp):
            return out_tmp
        msg = f"  Предупреждение ({fname}): файл не создан"
        if ret is not None and ret != 0:
            msg += f" (код возврата: {ret})"
        print(msg)
    except Exception as e:
        print(f"  Предупреждение ({fname}): {e}")
    return None


# --- Точные ядра производных 1-го порядка ---
# Полином 3-го порядка Флоринского (2009) 5x5 (совпадает с ядром WBT slope с точностью 1e-6,
# но позволяет вычислить aspect/eastness/northness/hillshade через atan2 без бага fx == 0)
_Y5, _X5 = np.mgrid[-2:3, -2:3]
_A5 = np.column_stack([
    np.ones(25), _X5.ravel(), _Y5.ravel(),
    _X5.ravel() ** 2, (_X5 * _Y5).ravel(), _Y5.ravel() ** 2,
    _X5.ravel() ** 3, (_X5 ** 2 * _Y5).ravel(), (_X5 * _Y5 ** 2).ravel(), _Y5.ravel() ** 3,
])
_PINV5 = np.linalg.pinv(_A5)
KERNEL_FX_5x5 = _PINV5[1].reshape(5, 5) / res_x_m
KERNEL_FY_5x5 = -_PINV5[2].reshape(5, 5) / res_y_m  # ось Y на север (-строка)

# Ядро Хорна (1981) 3x3 (используется в круговой дисперсии экспозиции)
KERNEL_FX_3x3 = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / (8.0 * res_x_m)
KERNEL_FY_3x3 = np.array([[ 1.0, 2.0, 1.0], [ 0.0, 0.0, 0.0], [-1.0,-2.0,-1.0]]) / (8.0 * res_y_m)

# =============================================================
# ШАГ 3: БАЗОВЫЕ ИНДЕКСЫ — DEM_{base_name}_terrain.tif
# =============================================================

print("========================================================")
print("  БАЗОВЫЕ ИНДЕКСЫ РЕЛЬЕФА")
print("========================================================\n")

terrain_layers = []
terrain_names = []

# Вычисляем точные производные 5x5 (Флоринский)
fx5 = correlate(dem_work, KERNEL_FX_5x5, mode="nearest")
fy5 = correlate(dem_work, KERNEL_FY_5x5, mode="nearest")
grad5 = np.hypot(fx5, fy5)
slope_rad = np.arctan(grad5)
slope_deg = np.degrees(slope_rad).astype(np.float32)

# Сохраняем tmp_slope.tif (нужен также в Шаге 6)
slope_path = save_temp_array(slope_deg, "tmp_slope.tif")
terrain_layers.append(slope_path)
terrain_names.append("DEM_slope")
print("  [OK] slope (угол наклона, 5x5 Флоринский)")

# Экспозиция (азимут по часовой стрелке от севера, 0..360 град., -1.0 на строго плоских)
# Направление быстрейшего спуска в (Восток, Север) = (-fx5, -fy5)
flat_mask = grad5 < 1e-12
aspect_deg = ((np.degrees(np.arctan2(-fx5, -fy5)) + 360.0) % 360.0).astype(np.float32)
aspect_deg[flat_mask] = -1.0
aspect_path = save_temp_array(aspect_deg, "tmp_aspect.tif")
terrain_layers.append(aspect_path)
terrain_names.append("DEM_aspect")
print("  [OK] aspect (экспозиция через atan2, исправлен баг fx=0 на С/Ю склонах)")

# Eastness = sin(aspect) = -fx / grad; Northness = cos(aspect) = -fy / grad
# На плоских участках (grad == 0) строго 0.0 (нейтральное значение вместо sin/cos(-1 град.))
safe_grad = np.maximum(grad5, 1e-15)
eastness = np.where(flat_mask, 0.0, -fx5 / safe_grad).astype(np.float32)
northness = np.where(flat_mask, 0.0, -fy5 / safe_grad).astype(np.float32)
del safe_grad

east_path = save_temp_array(eastness, "tmp_eastness.tif")
north_path = save_temp_array(northness, "tmp_northness.tif")
terrain_layers.extend([east_path, north_path])
terrain_names.extend(["DEM_eastness", "DEM_northness"])
del aspect_deg, eastness, northness
print("  [OK] eastness, northness (на плоских участках = 0.0)")

# --- TPI (3x3 базовый, точный float64 без 1-см квантования WBT) ---
mean_c_3x3 = uniform_filter(dem_centered, size=3, mode="nearest")
tpi_3x3 = (dem_centered - mean_c_3x3).astype(np.float32)
tpi_path = save_temp_array(tpi_3x3, "tmp_tpi.tif")
terrain_layers.append(tpi_path)
terrain_names.append("DEM_TPI")
del tpi_3x3
print("  [OK] TPI (разность со средним в окне 3x3, float64)")

# --- TRI (Ruggedness Index = СКВО разностей с центром в окне 3x3, центрированный float64) ---
# sum_{i=1..8} (z_i - z_0)^2 = 9 * (mean(z_c^2) - 2*z_c*mean(z_c) + z_c^2)
mean_sq_3x3 = uniform_filter(dem_centered * dem_centered, size=3, mode="nearest")
sum_diff_sq = np.maximum(0.0, 9.0 * (mean_sq_3x3 - 2.0 * dem_centered * mean_c_3x3 + dem_centered * dem_centered))
tri_3x3 = np.sqrt(sum_diff_sq / 8.0).astype(np.float32)
del mean_c_3x3, mean_sq_3x3, sum_diff_sq
tri_path = save_temp_array(tri_3x3, "tmp_tri.tif")
terrain_layers.append(tri_path)
terrain_names.append("DEM_TRI")
del tri_3x3
print("  [OK] TRI (индекс пересечённости рельефа 3x3, float64)")

# --- Гидрологическая подготовка DEM для корректного flowdir и водосборной площади ---
dem_hydro_path = os.path.join(temp_dir, "dem_hydro.tif")
p_hydro = run_wbt(
    wbt.breach_depressions_least_cost,
    "dem_hydro.tif",
    dem=dem_tmp_path,
    dist=50,
    fill=True,
)
if not p_hydro:
    p_hydro = run_wbt(
        wbt.fill_depressions,
        "dem_hydro.tif",
        dem=dem_tmp_path,
        fix_flats=True,
    )
if not p_hydro:
    dem_hydro_path = dem_tmp_path
else:
    print("  [OK] гидрологическая коррекция микрозападин (breach + fill + fix_flats)")

# --- flowdir (D8 pointer по гидрологически корректному DEM) ---
p_fdir = run_wbt(wbt.d8_pointer, "tmp_flowdir.tif", dem=dem_hydro_path)
if p_fdir:
    terrain_layers.append(p_fdir)
    terrain_names.append("DEM_flowdir")
    print("  [OK] flowdir (D8 направления стока)")
else:
    print("  [!] flowdir - пропущен")

# --- Сохраняем базовые индексы в один файл ---
if terrain_layers:
    out_terrain = os.path.join(output_dir, f"DEM_{base_name}_terrain.tif")
    stack_and_save(terrain_layers, terrain_names, out_terrain)
    print(f"\n  Сохранено: {os.path.basename(out_terrain)}  ({len(terrain_names)} слоёв)")
    clean_temp(silent=True)

# =============================================================
# ШАГ 4: КРИВИЗНЫ — DEM_{base_name}_curvatures.tif
# =============================================================

print("\n========================================================")
print("  КРИВИЗНЫ ПОВЕРХНОСТИ (5x5)")
print("========================================================\n")

curv_layers = []
curv_names = []

curvature_tools = [
    ("profile_curvature",    wbt.profile_curvature,    "DEM_profc"),
    ("plan_curvature",       wbt.plan_curvature,       "DEM_planc"),
    ("mean_curvature",       wbt.mean_curvature,       "DEM_meanc"),
    ("tangential_curvature", wbt.tangential_curvature, "DEM_tangc"),
    ("total_curvature",      wbt.total_curvature,      "DEM_totalc"),
    ("gaussian_curvature",   wbt.gaussian_curvature,   "DEM_gaussc"),
    ("minimal_curvature",    wbt.minimal_curvature,    "DEM_minc"),
]

tangc_tmp_path = None
planc_tmp_path = None

for tool_name, tool_func, layer_name in curvature_tools:
    kw = {"dem": dem_tmp_path}
    if zfactor != 1.0:
        kw["zfactor"] = zfactor
    p = run_wbt(tool_func, f"tmp_{tool_name}.tif", **kw)
    if p:
        if layer_name == "DEM_tangc":
            tangc_tmp_path = p
        elif layer_name == "DEM_planc":
            planc_tmp_path = p
        curv_layers.append(p)
        curv_names.append(layer_name)
        print(f"  [OK] {tool_name}")
    else:
        print(f"  [!] {tool_name} - пропущен")

# Регуляризация плановой кривизны (DEM_planc) при малых углах наклона (< 0.1 град.),
# чтобы устранить сингулярные выбросы k_plan = k_tang / sin(slope) -> +-10^4 на плоских участках
if planc_tmp_path and tangc_tmp_path and os.path.exists(tangc_tmp_path):
    with rasterio.open(tangc_tmp_path) as s_tc:
        tc_arr = s_tc.read(1).astype(np.float64)
    tc_arr[~np.isfinite(tc_arr) | (tc_arr == NODATA_VAL)] = 0.0
    sin_slp = np.sin(slope_rad)
    eps_sin = math.sin(math.radians(0.1))  # порог регуляризации 0.1 градуса
    planc_reg = (tc_arr * sin_slp / (sin_slp * sin_slp + eps_sin * eps_sin)).astype(np.float32)
    save_temp_array(planc_reg, os.path.basename(planc_tmp_path))
    del tc_arr, sin_slp, planc_reg
    print("  [OK] регуляризация plan_curvature на малых уклонах (< 0.1 град.)")

if curv_layers:
    out_curv = os.path.join(output_dir, f"DEM_{base_name}_curvatures.tif")
    stack_and_save(curv_layers, curv_names, out_curv)
    print(f"\n  Сохранено: {os.path.basename(out_curv)}  ({len(curv_names)} слоёв)")
    clean_temp(silent=True)

# =============================================================
# ШАГ 5: МУЛЬТИМАСШТАБНЫЕ ИНДЕКСЫ (по окнам)
# =============================================================

print("\n========================================================")
print("  МУЛЬТИМАСШТАБНЫЕ ИНДЕКСЫ (ПО ОКНАМ)")
print("========================================================\n")

# Предварительно вычисляем единичные векторы экспозиции по Хорну 3x3 для CircularVarianceOfAspect
fx3 = correlate(dem_work, KERNEL_FX_3x3, mode="nearest")
fy3 = correlate(dem_work, KERNEL_FY_3x3, mode="nearest")
grad3 = np.hypot(fx3, fy3)
sloped3 = (grad3 > 1e-12).astype(np.float64)
sin_asp3 = np.where(sloped3 > 0, -fx3 / np.maximum(grad3, 1e-15), 0.0)
cos_asp3 = np.where(sloped3 > 0, -fy3 / np.maximum(grad3, 1e-15), 0.0)
del fx3, fy3, grad3

for win_size in win_sizes:
    w_tag = f"w{win_size}"
    print(f"  ОКНО {win_size}x{win_size}")

    ms_layers = []
    ms_names = []

    # 1. DiffFromMeanElev (DMV / TPI мультимасштабный, точный float64)
    mean_c_w = uniform_filter(dem_centered, size=win_size, mode="nearest")
    diff_c_w = dem_centered - mean_c_w
    dmv_w = diff_c_w.astype(np.float32)
    p = save_temp_array(dmv_w, f"tmp_dmv_{w_tag}.tif")
    ms_layers.append(p)
    ms_names.append(f"DEM_dmv_{w_tag}")
    del dmv_w
    print("    [OK] DiffFromMeanElev (DMV/TPI, float64)")

    # 2. DevFromMeanElev (z-score в окне WxW, центрированный float64 с границей Самуэльсона sqrt(W^2-1))
    mean_sq_w = uniform_filter(dem_centered * dem_centered, size=win_size, mode="nearest")
    std_w = np.sqrt(np.maximum(0.0, mean_sq_w - mean_c_w * mean_c_w))
    z_bound = math.sqrt(float(win_size * win_size - 1))
    dev_w = np.where(
        std_w > 1e-4,
        np.clip(diff_c_w / np.maximum(std_w, 1e-4), -z_bound, z_bound),
        0.0,
    ).astype(np.float32)
    del mean_sq_w, std_w, diff_c_w
    p = save_temp_array(dev_w, f"tmp_devmean_{w_tag}.tif")
    ms_layers.append(p)
    ms_names.append(f"DEM_devmean_{w_tag}")
    del dev_w
    print("    [OK] DevFromMeanElev (z-score, float64)")

    # 3. RelativeTopographicPosition (RTP, точный float64)
    mean_w = mean_c_w + dem_mean_global
    del mean_c_w
    zmin_w = minimum_filter(dem_work, size=win_size, mode="nearest")
    zmax_w = maximum_filter(dem_work, size=win_size, mode="nearest")
    rng_w = zmax_w - zmin_w
    rtp_w = np.where(
        rng_w < 1e-4,
        0.0,
        np.where(
            dem_work < mean_w,
            (dem_work - mean_w) / np.maximum(mean_w - zmin_w, 1e-6),
            (dem_work - mean_w) / np.maximum(zmax_w - mean_w, 1e-6),
        ),
    )
    rtp_w = np.clip(rtp_w, -1.0, 1.0).astype(np.float32)
    del mean_w
    p = save_temp_array(rtp_w, f"tmp_rtp_{w_tag}.tif")
    ms_layers.append(p)
    ms_names.append(f"DEM_rtp_{w_tag}")
    del rtp_w
    print("    [OK] RelativeTopographicPosition (float64)")

    # 4. StandardDeviationOfSlope
    p = run_wbt(
        wbt.standard_deviation_of_slope,
        f"tmp_sdslope_{w_tag}.tif",
        i=dem_tmp_path,
        filterx=win_size,
        filtery=win_size,
    )
    if p:
        ms_layers.append(p)
        ms_names.append(f"DEM_sdslope_{w_tag}")
        print("    [OK] StandardDeviationOfSlope")
    else:
        print("    [!] StandardDeviationOfSlope - пропущен")

    # 5. CircularVarianceOfAspect (точный векторный расчёт без бага fx=0 и без дыр NoData)
    m_sin = uniform_filter(sin_asp3, size=win_size, mode="nearest")
    m_cos = uniform_filter(cos_asp3, size=win_size, mode="nearest")
    m_cnt = uniform_filter(sloped3, size=win_size, mode="nearest")
    r_bar = np.where(m_cnt > 1e-9, np.hypot(m_sin, m_cos) / np.maximum(m_cnt, 1e-9), 1.0)
    cva_w = np.clip(1.0 - r_bar, 0.0, 1.0).astype(np.float32)
    del m_sin, m_cos, m_cnt, r_bar
    p = save_temp_array(cva_w, f"tmp_cva_{w_tag}.tif")
    ms_layers.append(p)
    ms_names.append(f"DEM_circvar_aspect_{w_tag}")
    del cva_w
    print("    [OK] CircularVarianceOfAspect (без дыр NoData)")

    # 6. SphericalStdDevOfNormals
    p = run_wbt(
        wbt.spherical_std_dev_of_normals,
        f"tmp_ssdn_{w_tag}.tif",
        dem=dem_tmp_path,
        filter=win_size,
    )
    if p:
        ms_layers.append(p)
        ms_names.append(f"DEM_spherical_sdn_{w_tag}")
        print("    [OK] SphericalStdDevOfNormals")
    else:
        print("    [!] SphericalStdDevOfNormals - пропущен")

    # 7. AverageNormalVectorAngularDeviation
    p = run_wbt(
        wbt.average_normal_vector_angular_deviation,
        f"tmp_anvad_{w_tag}.tif",
        dem=dem_tmp_path,
        filter=win_size,
    )
    if p:
        # Убираем отрицательную погрешность округления (-1e-13) на плоских участках
        with rasterio.open(p) as s_anv:
            anv = s_anv.read(1).astype(np.float32)
        anv = np.where((anv != NODATA_VAL) & (anv < 0.0), 0.0, anv)
        save_temp_array(anv, os.path.basename(p))
        del anv
        ms_layers.append(p)
        ms_names.append(f"DEM_avg_norm_angdev_{w_tag}")
        print("    [OK] AvgNormalVectorAngularDeviation")
    else:
        print("    [!] AvgNormalVectorAngularDeviation - пропущен")

    # 8. SurfaceAreaRatio (фиксированное окно 3x3)
    if win_size == 3:
        p = run_wbt(wbt.surface_area_ratio, f"tmp_sar_{w_tag}.tif", dem=dem_tmp_path)
        if p:
            ms_layers.append(p)
            ms_names.append(f"DEM_surface_area_ratio_{w_tag}")
            print("    [OK] SurfaceAreaRatio")
        else:
            print("    [!] SurfaceAreaRatio - пропущен")

    # 9. ElevPercentile (с нейтральным значением 50.0% на абсолютно плоских плато)
    p = run_wbt(
        wbt.elev_percentile,
        f"tmp_elevpct_{w_tag}.tif",
        dem=dem_tmp_path,
        filterx=win_size,
        filtery=win_size,
    )
    if p:
        with rasterio.open(p) as s_ep:
            ep = s_ep.read(1).astype(np.float32)
        ep = np.where(rng_w < 1e-4, 50.0, ep)
        save_temp_array(ep, os.path.basename(p))
        del ep
        ms_layers.append(p)
        ms_names.append(f"DEM_elev_percentile_{w_tag}")
        print("    [OK] ElevPercentile")
    else:
        print("    [!] ElevPercentile - пропущен")

    # 10. PercentElevRange (точный float64, 50.0% на абсолютно плоских плато)
    pct_rng = np.where(
        rng_w < 1e-4,
        50.0,
        (dem_work - zmin_w) / np.maximum(rng_w, 1e-6) * 100.0,
    )
    pct_rng = np.clip(pct_rng, 0.0, 100.0).astype(np.float32)
    del zmin_w, zmax_w, rng_w
    p = save_temp_array(pct_rng, f"tmp_pctrange_{w_tag}.tif")
    ms_layers.append(p)
    ms_names.append(f"DEM_pct_elev_range_{w_tag}")
    del pct_rng
    print("    [OK] PercentElevRange (float64)")

    # 11. MaxDifferenceFromMean (по радиусам r = 1..win_size в центрированном float64,
    #     без зажима WBT min_scale>=3, max_scale>=5 и без 1-см квантования)
    best_mag = np.zeros((height, width), dtype=np.float32)
    best_abs = np.full((height, width), -1.0, dtype=np.float32)
    best_scl = np.ones((height, width), dtype=np.float32)
    for r_scale in range(1, win_size + 1):
        k_size = 2 * r_scale + 1
        m_r = uniform_filter(dem_centered, size=k_size, mode="nearest")
        msq_r = uniform_filter(dem_centered * dem_centered, size=k_size, mode="nearest")
        std_r = np.sqrt(np.maximum(0.0, msq_r - m_r * m_r))
        z_b_r = math.sqrt(float(k_size * k_size - 1))
        dev_r = np.where(
            std_r > 1e-4,
            np.clip((dem_centered - m_r) / np.maximum(std_r, 1e-4), -z_b_r, z_b_r),
            0.0,
        ).astype(np.float32)
        abs_r = np.abs(dev_r)
        upd = abs_r > best_abs
        best_abs[upd] = abs_r[upd]
        best_mag[upd] = dev_r[upd]
        best_scl[upd] = float(r_scale)
        del m_r, msq_r, std_r, dev_r, abs_r, upd
    del best_abs
    mag_path = save_temp_array(best_mag, f"tmp_maxdiffmean_mag_{w_tag}.tif")
    scale_path = save_temp_array(best_scl, f"tmp_maxdiffmean_scale_{w_tag}.tif")
    del best_mag, best_scl
    ms_layers.extend([mag_path, scale_path])
    ms_names.extend([f"DEM_max_diff_mean_{w_tag}", f"DEM_max_diff_mean_scale_{w_tag}"])
    print("    [OK] MaxDifferenceFromMean (magnitude + scale, радиусы 1.." + str(win_size) + ")")

    # Сохраняем все мультимасштабные индексы окна в один файл
    if ms_layers:
        out_ms = os.path.join(output_dir, f"DEM_{base_name}_{w_tag}_multiscale.tif")
        stack_and_save(ms_layers, ms_names, out_ms)
        print(f"\n    Сохранено: {os.path.basename(out_ms)}  ({len(ms_names)} слоёв)")
    clean_temp(silent=True)
    print()

del sin_asp3, cos_asp3, sloped3, dem_centered

# =============================================================
# ШАГ 6: ДОПОЛНИТЕЛЬНЫЕ И ГИДРОЛОГИЧЕСКИЕ ИНДЕКСЫ — DEM_{base_name}_extra.tif
# =============================================================

print("========================================================")
print("  ДОПОЛНИТЕЛЬНЫЕ И ГИДРОЛОГИЧЕСКИЕ ИНДЕКСЫ")
print("========================================================\n")

extra_layers = []
extra_names = []

# --- 1. Hillshade (азимут 315 град., высота 45 град., точный векторный расчёт без бага fx=0) ---
alt_rad = math.radians(45.0)
sin_alt = math.sin(alt_rad)
cos_alt = math.cos(alt_rad)
cos_slp = np.cos(slope_rad)
sin_slp = np.sin(slope_rad)
# Азимут склона alpha: sin(alpha) = -fx5/grad, cos(alpha) = -fy5/grad
# cos(az_sun - alpha) = cos(az_sun)*cos(alpha) + sin(az_sun)*sin(alpha)
safe_grad5 = np.maximum(grad5, 1e-15)
sin_a5 = np.where(flat_mask, 0.0, -fx5 / safe_grad5)
cos_a5 = np.where(flat_mask, 0.0, -fy5 / safe_grad5)

az315 = math.radians(315.0)
cos_diff_315 = math.cos(az315) * cos_a5 + math.sin(az315) * sin_a5
hillshade = np.clip(sin_alt * cos_slp + cos_alt * sin_slp * cos_diff_315, 0.0, 1.0).astype(np.float32)
p_hill = save_temp_array(hillshade, "tmp_hillshade.tif")
extra_layers.append(p_hill)
extra_names.append("DEM_hillshade")
del hillshade, cos_diff_315
print("  [OK] hillshade (азимут 315, высота 45, исправлен баг fx=0)")

# --- 2. MultidirectionalHillshade (Mark 1992: 225, 270, 315, 360 град., без бага fx=0) ---
# Веса по Mark (1992): w_i = sin^2(alpha - az_i), для плоских весов поровну 0.25
az_list = [math.radians(225.0), math.radians(270.0), math.radians(315.0), math.radians(360.0)]
mhill = np.zeros((height, width), dtype=np.float64)
w_sum = np.zeros((height, width), dtype=np.float64)
for az_i in az_list:
    sin_d = sin_a5 * math.cos(az_i) - cos_a5 * math.sin(az_i)
    cos_d = cos_a5 * math.cos(az_i) + sin_a5 * math.sin(az_i)
    w_i = np.where(flat_mask, 0.25, np.maximum(sin_d * sin_d, 1e-4))
    hs_i = np.clip(sin_alt * cos_slp + cos_alt * sin_slp * cos_d, 0.0, 1.0)
    mhill += w_i * hs_i
    w_sum += w_i
mhill = (mhill / np.maximum(w_sum, 1e-12)).astype(np.float32)
del w_sum, sin_a5, cos_a5, cos_slp, safe_grad5, fx5, fy5, grad5, flat_mask
p_mhill = save_temp_array(mhill, "tmp_multihill.tif")
extra_layers.append(p_mhill)
extra_names.append("DEM_multihillshade")
del mhill
print("  [OK] multidirectional_hillshade (4 азимута, исправлен баг fx=0)")

# --- 3. Водосборная площадь SCA по гидрологически скорректированному DEM ---
flow_acc_path = os.path.join(temp_dir, "tmp_flowacc.tif")
p_fa = run_wbt(
    wbt.d8_flow_accumulation,
    "tmp_flowacc.tif",
    i=dem_hydro_path,
    out_type="sca",
)
if p_fa and os.path.exists(flow_acc_path):
    print("  [OK] flow accumulation (SCA по гидрологически корректному DEM)")
    with rasterio.open(flow_acc_path) as s_fa:
        sca_arr = s_fa.read(1).astype(np.float64)
    sca_arr[~np.isfinite(sca_arr) | (sca_arr <= 0.0)] = res_x_m

    # Задаём минимальный физический уклон tan(beta) >= 0.001 (~0.057 град.) на плоских местах,
    # чтобы TWI не обращался в NoData (-32768) на 28.6% территории
    tan_slp_safe = np.maximum(np.tan(slope_rad), 0.001)
    sin_slp_safe = np.maximum(sin_slp, 0.0001)

    # WetnessIndex (TWI) = ln(SCA / tan(slope))
    twi_arr = np.log(sca_arr / tan_slp_safe).astype(np.float32)
    p_twi = save_temp_array(twi_arr, "tmp_twi.tif")
    extra_layers.append(p_twi)
    extra_names.append("DEM_TWI")
    del twi_arr
    print("  [OK] wetness_index (TWI без дыр NoData на плоских участках)")

    # SedimentTransportIndex (STI) = (m+1)*(SCA/22.13)^m * (sin(beta)/0.0896)^n, m=0.4, n=1.3
    sti_arr = (1.4 * np.power(sca_arr / 22.13, 0.4) * np.power(np.maximum(sin_slp, 0.0) / 0.0896, 1.3)).astype(np.float32)
    p_sti = save_temp_array(sti_arr, "tmp_sti.tif")
    extra_layers.append(p_sti)
    extra_names.append("DEM_STI")
    del sti_arr
    print("  [OK] sediment_transport_index (STI)")

    # StreamPowerIndex (SPI) = SCA * tan(slope)
    spi_arr = (sca_arr * np.tan(slope_rad)).astype(np.float32)
    p_spi = save_temp_array(spi_arr, "tmp_spi.tif")
    extra_layers.append(p_spi)
    extra_names.append("DEM_SPI")
    del spi_arr, sca_arr, tan_slp_safe, sin_slp_safe
    print("  [OK] stream_power_index (SPI)")
else:
    print("  [!] flow accumulation (SCA) - пропущен, TWI/STI/SPI не рассчитаны")

del slope_rad, sin_slp, dem_work

# --- 4. Geomorphons ---
p_geo = run_wbt(
    wbt.geomorphons,
    "tmp_geomorphons.tif",
    dem=dem_tmp_path,
    search=50,
    threshold=0,
    forms=True,
)
if p_geo:
    extra_layers.append(p_geo)
    extra_names.append("DEM_geomorphons")
    print("  [OK] geomorphons")
else:
    print("  [!] geomorphons - пропущен")

if extra_layers:
    out_extra = os.path.join(output_dir, f"DEM_{base_name}_extra.tif")
    stack_and_save(extra_layers, extra_names, out_extra)
    print(f"\n  Сохранено: {os.path.basename(out_extra)}  ({len(extra_names)} слоёв)")
    clean_temp(silent=True)

# =============================================================
# ИТОГ
# =============================================================

print("\n========================================================")
print("  ИТОГ")
print("========================================================\n")

out_files = sorted(glob.glob(os.path.join(output_dir, f"DEM_{base_name}_*.tif")))
total_layers = 0
for f in out_files:
    with rasterio.open(f) as src:
        n = src.count
        total_layers += n
    size_mb = os.path.getsize(f) / (1024 ** 2)
    print(f"  {os.path.basename(f):50s}  {n:2d} слоёв  {size_mb:6.1f} MB")

print(f"\n  Всего файлов: {len(out_files)}")
print(f"  Всего слоёв:  {total_layers}")

clean_temp(silent=True, remove_all=True)
try:
    os.rmdir(temp_dir)
except OSError:
    pass

print("\nГотово. Все индексы сохранены как отдельные файлы рядом с исходным DEM.\n")
