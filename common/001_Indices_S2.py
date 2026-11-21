# =============================================================
# 001_Indices_S2_v8.py
# Расчет всех возможных спектральных индексов для Sentinel-2
# с помощью библиотеки spyndex и сохранение в один GeoTIFF
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
# =============================================================

import os
import re
import glob
import warnings
import numpy as np

# --- Настройка путей и временной папки ---------------------------------------
#
# Все рабочие папки берутся рядом со скриптом, а не из текущей папки запуска.
# Временные файлы Python/GDAL/rasterio перенаправляются в ./TEMP рядом со
# скриптом, чтобы не забивать системный диск C: при обработке больших GeoTIFF.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

input_dir  = os.path.join(SCRIPT_DIR, "INPUT")
output_dir = os.path.join(SCRIPT_DIR, "INPUT")  # результат рядом с исходным растром
temp_dir   = os.path.join(SCRIPT_DIR, "TEMP")

for d in [input_dir, output_dir, temp_dir]:
    os.makedirs(d, exist_ok=True)

for env_name in ["TMP", "TEMP", "TMPDIR"]:
    os.environ[env_name] = temp_dir

os.environ["CPL_TMPDIR"] = temp_dir
os.environ["GDAL_PAM_PROXY_DIR"] = temp_dir
os.environ.setdefault("GDAL_CACHEMAX", "512")

import rasterio
import spyndex

# =============================================================
# ШАГ 0: ВЫБОР ФАЙЛА
# =============================================================

tif_files = sorted(glob.glob(os.path.join(input_dir, "*.tif"), recursive=False))
tif_files += [f for f in sorted(glob.glob(os.path.join(input_dir, "*.TIF"), recursive=False))
              if f not in tif_files]

if len(tif_files) == 0:
    raise FileNotFoundError("В папке INPUT не найдено .tif файлов!")

print("\n════════════════════════════════════════════════════════")
print("  РАСЧЁТ СПЕКТРАЛЬНЫХ ИНДЕКСОВ (Sentinel-2)")
print("════════════════════════════════════════════════════════\n")

print("Доступные растры в папке INPUT:\n")
for i, f in enumerate(tif_files, 1):
    with rasterio.open(f) as src:
        n_lyr = src.count
    print(f"  [{i}] {os.path.basename(f)}  ({n_lyr} каналов)")

choice = input(f"\nВыберите растр (Enter = 1): ").strip()
if choice == "":
    choice = 1
else:
    choice = int(choice)
if choice < 1 or choice > len(tif_files):
    raise ValueError("Неверный номер файла!")

input_path = tif_files[choice - 1]
print(f"  ✓ Выбран: {os.path.basename(input_path)}\n")

# --- Загрузка растра ---
src_ds = rasterio.open(input_path)
n_bands = src_ds.count
band_names = list(src_ds.descriptions) if src_ds.descriptions[0] is not None else [f"B{i}" for i in range(1, n_bands + 1)]
band_names = [name if name else f"B{i}" for i, name in enumerate(band_names, 1)]

# =============================================================
# ШАГ 1: ПРОВЕРКА ИМЁН КАНАЛОВ
# =============================================================

print("════════════════════════════════════════════════════════")
print("  ТЕКУЩИЕ ИМЕНА КАНАЛОВ")
print("════════════════════════════════════════════════════════\n")

print(f"  Каналов: {n_bands}\n")
for i, name in enumerate(band_names, 1):
    print(f"  [{i:2d}] {name}")

# --- Стандартные имена Sentinel-2 (9 каналов 10/20м) ---
s2_standard_9 = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]
s2_desc_9 = [
    "Blue 490nm", "Green 560nm", "Red 665nm",
    "RE1 705nm", "RE2 740nm", "RE3 783nm",
    "NIR 842nm", "SWIR1 1610nm", "SWIR2 2190nm"
]

# --- Стандартные имена (13 каналов полный S2) ---
s2_standard_13 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7",
                   "B8", "B8A", "B9", "B10", "B11", "B12"]
s2_desc_13 = [
    "Coastal 443nm", "Blue 490nm", "Green 560nm", "Red 665nm",
    "RE1 705nm", "RE2 740nm", "RE3 783nm",
    "NIR 842nm", "NIR-narrow 865nm", "WV 945nm",
    "Cirrus 1375nm", "SWIR1 1610nm", "SWIR2 2190nm"
]

# =============================================================
# ШАГ 2: ПРЕДЛОЖИТЬ ПЕРЕИМЕНОВАНИЕ
# =============================================================

print("\n════════════════════════════════════════════════════════")
print("  ПЕРЕИМЕНОВАНИЕ КАНАЛОВ")
print("════════════════════════════════════════════════════════\n")

print("  Варианты:")
print("  [Enter] Оставить текущие имена")

if n_bands == 9:
    print("  [s]     Применить стандартные Sentinel-2 (9 каналов):")
    print(f"          {', '.join(s2_standard_9)}")
elif n_bands == 13:
    print("  [s]     Применить стандартные Sentinel-2 (13 каналов):")
    print(f"          {', '.join(s2_standard_13)}")
else:
    print(f"  [s]     Недоступно (у вас {n_bands} каналов, стандарт: 9 или 13)")
print("  [m]     Переименовать каждый канал вручную\n")

rename_choice = input("  Выбор: ").strip().lower()

if rename_choice == "s":
    if n_bands == 9:
        old_names = band_names[:]
        band_names = s2_standard_9[:]
        print("\n  ✓ Применены стандартные имена (9 каналов):")
        for i in range(9):
            print(f"    [{i+1}] {s2_standard_9[i]}  ← {old_names[i]}  ({s2_desc_9[i]})")
    elif n_bands == 13:
        old_names = band_names[:]
        band_names = s2_standard_13[:]
        print("\n  ✓ Применены стандартные имена (13 каналов):")
        for i in range(13):
            print(f"    [{i+1}] {s2_standard_13[i]}  ← {old_names[i]}  ({s2_desc_13[i]})")
    else:
        print(f"\n  ! Стандартные имена доступны только для 9 или 13 каналов (у вас {n_bands}).")
        print("    Оставляю текущие имена.")

elif rename_choice == "m":
    print("\n  Переименование каналов (Enter = оставить текущее):\n")
    new_names = []
    for i, name in enumerate(band_names):
        hint = ""
        if n_bands == 9 and i < 9:
            hint = f" [подсказка: {s2_standard_9[i]} — {s2_desc_9[i]}]"
        elif n_bands == 13 and i < 13:
            hint = f" [подсказка: {s2_standard_13[i]} — {s2_desc_13[i]}]"
        inp = input(f"    [{i+1:2d}] '{name}'{hint}\n         Новое имя (Enter = '{name}'): ").strip()
        new_names.append(inp if inp else name)
    old_names = band_names[:]
    band_names = new_names
    print("\n  ✓ Каналы переименованы:")
    for i, (old, new) in enumerate(zip(old_names, band_names), 1):
        changed = " ← ИЗМЕНЕНО" if old != new else ""
        print(f"    [{i:2d}] {new}{changed}")
else:
    print("\n  ✓ Имена оставлены без изменений.")

print()

# =============================================================
# ШАГ 3: МАППИНГ КАНАЛОВ ДЛЯ spyndex
# =============================================================
# Sentinel-2 → spyndex параметры:
#   B2  (Blue 490nm)     → B
#   B3  (Green 560nm)    → G
#   B4  (Red 665nm)      → R
#   B5  (RE1 705nm)      → RE1
#   B6  (RE2 740nm)      → RE2
#   B7  (RE3 783nm)      → RE3
#   B8  (NIR 842nm)      → N
#   B11 (SWIR1 1610nm)   → S1
#   B12 (SWIR2 2190nm)   → S2
# =============================================================

print("════════════════════════════════════════════════════════")
print("  МАППИНГ КАНАЛОВ ДЛЯ spyndex")
print("════════════════════════════════════════════════════════\n")


def find_band(patterns, names):
    """Ищет имя канала по шаблону regex, возвращает индекс (1-based) или None."""
    for p in patterns:
        matches = [i for i, n in enumerate(names) if re.match(p, n, re.IGNORECASE)]
        if len(matches) == 1:
            return matches[0] + 1
    return None


blue_idx  = find_band([r"^B0?2$", r"^B02_", r"_B0?2$", r"blue"],  band_names)
green_idx = find_band([r"^B0?3$", r"^B03_", r"_B0?3$", r"green"], band_names)
red_idx   = find_band([r"^B0?4$", r"^B04_", r"_B0?4$", r"red"],   band_names)
re1_idx   = find_band([r"^B0?5$", r"^B05_", r"_B0?5$", r"rededge1", r"RE1"], band_names)
re2_idx   = find_band([r"^B0?6$", r"^B06_", r"_B0?6$", r"rededge2", r"RE2"], band_names)
re3_idx   = find_band([r"^B0?7$", r"^B07_", r"_B0?7$", r"rededge3", r"RE3"], band_names)
nir_idx   = find_band([r"^B0?8$", r"^B08_", r"_B0?8$", r"^B8A$", r"^B08A", r"nir"], band_names)
swir1_idx = find_band([r"^B11$",  r"^B11_", r"_B11$",  r"swir1", r"SWIR_1"], band_names)
swir2_idx = find_band([r"^B12$",  r"^B12_", r"_B12$",  r"swir2", r"SWIR_2"], band_names)

# Если автоматика не сработала — откат на порядковые номера (9 каналов)
if blue_idx is None and n_bands >= 9:
    print("  ! Автоопределение не удалось → порядковая нумерация:")
    print("    1=Blue, 2=Green, 3=Red, 4=RE1, 5=RE2, 6=RE3, 7=NIR, 8=SWIR1, 9=SWIR2\n")
    blue_idx  = 1; green_idx = 2; red_idx   = 3
    re1_idx   = 4; re2_idx   = 5; re3_idx   = 6
    nir_idx   = 7; swir1_idx = 8; swir2_idx = 9

# Показываем итоговый маппинг
mapping = [
    ("blue",     blue_idx,  "490nm"),
    ("green",    green_idx, "560nm"),
    ("red",      red_idx,   "665nm"),
    ("redEdge1", re1_idx,   "705nm"),
    ("redEdge2", re2_idx,   "740nm"),
    ("redEdge3", re3_idx,   "783nm"),
    ("nir",      nir_idx,   "842nm"),
    ("swir1",    swir1_idx, "1610nm"),
    ("swir2",    swir2_idx, "2190nm"),
]

print("  Итоговый маппинг:\n")

n_mapped = 0
for param, idx, wavelength in mapping:
    if idx is not None:
        status = "✓"
        band_label = band_names[idx - 1]
        n_mapped += 1
    else:
        status = "✗"
        band_label = "—"
    print(f"    {status} {param:<10s} → {band_label:<6s}  ({wavelength})")

print(f"\n  Найдено каналов: {n_mapped} из 9")

if n_mapped < 3:
    raise ValueError("Слишком мало каналов для расчёта индексов (нужно минимум blue+red+nir).")

# =============================================================
# ШАГ 4: ВЫБОР scaleFactor
# =============================================================

print("\n════════════════════════════════════════════════════════")
print("  МАСШТАБНЫЙ КОЭФФИЦИЕНТ (scaleFactor)")
print("════════════════════════════════════════════════════════\n")

band1_data = src_ds.read(1).astype(np.float64)
sample_vals = band1_data[np.isfinite(band1_data)]
if len(sample_vals) > 10000:
    sample_vals = np.random.choice(sample_vals, 10000, replace=False)

if len(sample_vals) == 0:
    raise ValueError(
        "В первом канале нет конечных числовых значений. "
        "Проверьте входной растр: вероятно, он состоит из NaN/NoData."
    )

val_min = float(np.nanmin(sample_vals))
val_max = float(np.nanmax(sample_vals))

print(f"  Диапазон значений 1-го канала: [{val_min:.2f}, {val_max:.2f}]\n")

if val_max > 100:
    suggested_sf = 10000
    print(f"  Похоже на DN (0–10000) → рекомендуется scaleFactor = {suggested_sf}")
elif val_max <= 1.0 and val_min >= 0:
    suggested_sf = 1
    print(f"  Похоже на reflectance [0, 1] → рекомендуется scaleFactor = {suggested_sf}")
else:
    suggested_sf = 10000
    print(f"  Неясный диапазон → по умолчанию scaleFactor = {suggested_sf}")

sf_input = input(f"\n  scaleFactor (Enter = {suggested_sf}): ").strip()
if sf_input == "":
    scale_factor = suggested_sf
else:
    try:
        scale_factor = int(sf_input)
        if scale_factor <= 0:
            raise ValueError
    except ValueError:
        print(f"  ! Некорректный ввод, использую {suggested_sf}")
        scale_factor = suggested_sf

print(f"  ✓ scaleFactor = {scale_factor}\n")

# =============================================================
# ШАГ 5: РАСЧЁТ ИНДЕКСОВ
# =============================================================

print("════════════════════════════════════════════════════════")
print("  РАСЧЁТ СПЕКТРАЛЬНЫХ ИНДЕКСОВ")
print("════════════════════════════════════════════════════════\n")

print("  Загружаю каналы и нормализую в reflectance [0, 1]...")

# Читаем все каналы.
# float32 достаточно для спектральных индексов и требует в 2 раза меньше RAM,
# чем float64. Для больших мозаик это критично.
all_data = src_ds.read().astype(np.float32)
nodata_val = src_ds.nodata

# Маска nodata / NaN / inf.
# В GEE-экспортах часто бывает nodata=None, а пустые пиксели записаны как NaN.
# Для spyndex такие пиксели нужно исключить ДО расчёта формул, иначе NaN/inf
# проходят через деление и степени и затем весь слой выглядит как NoData.
invalid_data_mask = ~np.isfinite(all_data)
if nodata_val is not None:
    invalid_data_mask |= all_data == nodata_val

nodata_mask = np.any(invalid_data_mask, axis=0)
valid_input_pixels = int(np.count_nonzero(~nodata_mask))
total_input_pixels = int(nodata_mask.size)
print(f"  Валидных пикселей входного растра: {valid_input_pixels}/{total_input_pixels}")

if valid_input_pixels == 0:
    raise ValueError(
        "Во входном растре нет пикселей, где все каналы имеют конечные значения. "
        "Проверьте предыдущий шаг мозаики или входной файл."
    )

# Внутри формул spyndex пустые пиксели должны быть NaN, а не случайными числами.
all_data[invalid_data_mask] = np.nan
del invalid_data_mask

# Применяем scaleFactor → reflectance in-place, без создания второй копии массива
if scale_factor != 1:
    all_data /= np.float32(scale_factor)


def get_band_data(band_idx):
    """Возвращает 2D массив канала (0-based из 1-based индекса)."""
    if band_idx is None:
        return None
    return all_data[band_idx - 1]


B   = get_band_data(blue_idx)   # Blue
G   = get_band_data(green_idx)  # Green
R   = get_band_data(red_idx)    # Red
RE1 = get_band_data(re1_idx)    # RedEdge1
RE2 = get_band_data(re2_idx)    # RedEdge2
RE3 = get_band_data(re3_idx)    # RedEdge3
N   = get_band_data(nir_idx)    # NIR
S1  = get_band_data(swir1_idx)  # SWIR1 (~1.6µm)
S2  = get_band_data(swir2_idx)  # SWIR2 (~2.1µm)

print("  Запускаю расчёт индексов через spyndex...\n")

# --- Доступные параметры для spyndex ---
available_params = {}
if B   is not None: available_params["B"]   = B
if G   is not None: available_params["G"]   = G
if R   is not None: available_params["R"]   = R
if RE1 is not None: available_params["RE1"] = RE1
if RE2 is not None: available_params["RE2"] = RE2
if RE3 is not None: available_params["RE3"] = RE3
if N   is not None: available_params["N"]   = N
if S1  is not None: available_params["S1"]  = S1
if S2  is not None: available_params["S2"]  = S2

# Дополнительные константы, используемые некоторыми индексами
available_params["L"]       = 0.5        # Soil brightness correction (SAVI)
available_params["g"]       = 2.5        # Gain factor (EVI)
available_params["C1"]      = 6.0        # Coefficient 1 (EVI)
available_params["C2"]      = 7.5        # Coefficient 2 (EVI)
available_params["cexp"]    = 1.16       # Canopy background (MSAVI)
available_params["nexp"]    = 2.0        # Power exponent
available_params["alpha"]   = 0.1        # Weighting coefficient
available_params["sla"]     = 1.0        # Soil line slope
available_params["slb"]     = 0.0        # Soil line intercept
available_params["gamma"]   = 1.0        # Gamma for ARVI
available_params["omega"]   = 2.0        # Weighting coefficient
available_params["k"]       = 0.0        # Slope factor
available_params["fdelta"]  = 0.581      # FCVI delta
available_params["lambdaN"] = 0.8420     # NIR wavelength (µm) — Sentinel-2 B8
available_params["lambdaR"] = 0.6650     # Red wavelength (µm) — Sentinel-2 B4
available_params["lambdaG"] = 0.5600     # Green wavelength (µm) — Sentinel-2 B3

# Фильтруем индексы: берём только те, для которых все параметры доступны
spectral_indices = spyndex.indices

computable_indices = []
for idx_name, idx_obj in spectral_indices.items():
    required_bands = set()
    for param in idx_obj.bands:
        if param in ("B", "G", "R", "N", "N2", "RE1", "RE2", "RE3", "RE4",
                      "S1", "S2", "T1", "T2", "T", "A", "WV",
                      "lambdaN", "lambdaR", "lambdaG",
                      "L", "g", "C1", "C2", "cexp", "nexp", "alpha",
                      "sla", "slb", "gamma", "omega", "k", "fdelta", "PAR"):
            required_bands.add(param)

    all_available = all(p in available_params for p in required_bands)
    if all_available and len(required_bands) > 0:
        computable_indices.append(idx_name)

print(f"  Доступно для расчёта: {len(computable_indices)} индексов\n")

base_name = os.path.splitext(os.path.basename(input_path))[0]
out_name = f"{base_name}_indices.tif"
out_path = os.path.join(output_dir, out_name)

# Удаляем старый файл если существует
if os.path.exists(out_path):
    try:
        os.remove(out_path)
        print(f"  ⚠ Старый файл {out_name} удалён")
    except OSError:
        print(f"  ⚠ Не удалось удалить {out_name} — возможно открыт в другой программе!")
        print("    Закройте QGIS/ArcGIS и попробуйте снова.")
        raise RuntimeError("Файл заблокирован.")

nodata_output = -9999.0
height, width = all_data.shape[1], all_data.shape[2]
n_indices = len(computable_indices)

# Записываем GeoTIFF
profile = src_ds.profile.copy()
for key in ["photometric", "alpha", "mask", "interleave"]:
    profile.pop(key, None)

profile.update(
    count=n_indices,
    dtype="float32",
    nodata=nodata_output,
    compress="lzw",
    tiled=True,
    blockxsize=256,
    blockysize=256,
    BIGTIFF="IF_SAFER",
)

print("  Индексы будут рассчитываться и сразу записываться на диск.")
print("  Это медленнее, но не требует собирать массив на десятки ГБ в RAM.\n")

failed = []
written = 0

with rasterio.open(out_path, "w", **profile) as dst:
    empty_band = np.full((height, width), nodata_output, dtype=np.float32)

    for band_i, idx_name in enumerate(computable_indices, 1):
        idx_obj = spectral_indices[idx_name]
        params = {}
        for param in idx_obj.bands:
            if param in available_params:
                params[param] = available_params[param]

        layer_name = f"S2_{idx_name}"

        try:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    result = spyndex.computeIndex(idx_name, params=params)

            if not isinstance(result, np.ndarray):
                dst.write(empty_band, band_i)
                dst.set_band_description(band_i, f"{layer_name}_SKIPPED")
                failed.append((idx_name, "скалярный результат"))
                continue

            band = np.asarray(result, dtype=np.float32)
            band[nodata_mask] = nodata_output
            band[~np.isfinite(band)] = nodata_output

            finite_count = int(np.count_nonzero((band != nodata_output) & np.isfinite(band)))
            if finite_count == 0:
                dst.write(empty_band, band_i)
                dst.set_band_description(band_i, f"{layer_name}_SKIPPED")
                failed.append((idx_name, "все значения NaN/Inf"))
            else:
                dst.write(band, band_i)
                dst.set_band_description(band_i, layer_name)
                written += 1
                print(f"    [{band_i:3d}/{n_indices}] {layer_name}: записан, valid={finite_count}")

            del result
            del band

        except MemoryError:
            dst.write(empty_band, band_i)
            dst.set_band_description(band_i, f"{layer_name}_MEMORY_ERROR")
            failed.append((idx_name, "MemoryError"))
            print(f"    [{band_i:3d}/{n_indices}] {layer_name}: MemoryError, записан NoData-band")
        except Exception as e:
            dst.write(empty_band, band_i)
            dst.set_band_description(band_i, f"{layer_name}_ERROR")
            failed.append((idx_name, str(e)[:80]))
            print(f"    [{band_i:3d}/{n_indices}] {layer_name}: ошибка, записан NoData-band")

        if band_i % 25 == 0:
            print(f"    ... обработано {band_i}/{n_indices}")

print(f"\n  ✓ Записано индексов с данными: {written}")
if failed:
    print(f"    Индексов без данных/с ошибками: {len(failed)}")
    print("    Первые проблемные индексы:")
    for idx_name, reason in failed[:20]:
        print(f"      - {idx_name}: {reason}")
    if len(failed) > 20:
        print(f"      ... ещё {len(failed) - 20}")

if written == 0:
    raise ValueError(
        "Не удалось записать ни одного индекса с валидными пикселями. "
        "Проверьте, что выбран правильный растр и правильный scaleFactor."
    )

src_ds.close()

file_size_mb = os.path.getsize(out_path) / (1024 ** 2)
print(f"  ✓ Файл:    {out_name}")
print(f"    Размер:  {file_size_mb:.2f} MB")
print(f"    Каналов: {n_indices}")
print(f"    Путь:    {out_path}\n")
print("  Готово!\n")
