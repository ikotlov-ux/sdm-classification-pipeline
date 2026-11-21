# ==============================================================================
# 003_SAR_indices_Palsar_v3.py
# Расчёт производных SAR-индексов из двухполяризационных данных ALOS PALSAR
# (HH, HV — два отдельных файла)
# Сохранение каждого индекса как отдельный однослойный TIF: SAR_*.tif
# + Автоудаление ранее созданных SAR-файлов перед расчётом
# ==============================================================================

import os
import glob
import numpy as np
import rasterio
from tqdm import tqdm

# --- Директории ---------------------------------------------------------------

input_dir  = "INPUT"
output_dir = input_dir

if not os.path.isdir(input_dir):
    raise FileNotFoundError(f"Папка INPUT не найдена: {input_dir}")
os.makedirs(output_dir, exist_ok=True)

# --- Очистка ранее созданных SAR-файлов ---------------------------------------

print("\n========================================")
print("  Очистка ранее созданных SAR-файлов")
print("========================================\n")

old_sar_files = sorted(glob.glob(os.path.join(input_dir, "SAR_*.tif")))
old_sar_files += [f for f in sorted(glob.glob(os.path.join(input_dir, "SAR_*.TIF")))
                  if f not in old_sar_files]

if old_sar_files:
    print(f"Найдено {len(old_sar_files)} ранее созданных SAR-файлов:")
    n_deleted = 0
    for f in old_sar_files:
        print(f"  ✗ {os.path.basename(f)}")
        try:
            os.remove(f)
            n_deleted += 1
        except OSError as e:
            print(f"  ⚠ Не удалось удалить {os.path.basename(f)}: {e}")
    print(f"\n✓ Удалено {n_deleted} файлов\n")
else:
    print("Ранее созданных SAR-файлов не обнаружено.\n")

# --- Список растровых файлов в INPUT ------------------------------------------

raster_ext = (".tif", ".tiff", ".img", ".dat", ".sdat")
all_files = sorted(os.listdir(input_dir))
raster_files = [os.path.join(input_dir, f) for f in all_files
                if f.lower().endswith(raster_ext)]

if len(raster_files) == 0:
    raise FileNotFoundError("В папке INPUT не найдено растровых файлов.")

print("\n========================================")
print("  SAR Indices Calculator for PALSAR HH/HV")
print("========================================\n")
print("Найденные растровые файлы в INPUT:")
for i, f in enumerate(raster_files, 1):
    print(f"  [{i}] {os.path.basename(f)}")

# --- Выбор слоя HH -----------------------------------------------------------

hh_idx = input("\nВведите номер файла для слоя HH: ").strip()
hh_idx = int(hh_idx)
if hh_idx < 1 or hh_idx > len(raster_files):
    raise ValueError("Некорректный номер файла для HH.")

# --- Выбор слоя HV -----------------------------------------------------------

hv_idx = input("Введите номер файла для слоя HV: ").strip()
hv_idx = int(hv_idx)
if hv_idx < 1 or hv_idx > len(raster_files):
    raise ValueError("Некорректный номер файла для HV.")

print(f"\nHH: {os.path.basename(raster_files[hh_idx - 1])}")
print(f"HV: {os.path.basename(raster_files[hv_idx - 1])}")

# --- Чтение данных -----------------------------------------------------------

print("\nЧтение растров...")
hh_ds = rasterio.open(raster_files[hh_idx - 1])
hv_ds = rasterio.open(raster_files[hv_idx - 1])

# Проверка совпадения размеров
if (hh_ds.width != hv_ds.width or hh_ds.height != hv_ds.height or
        hh_ds.crs != hv_ds.crs or hh_ds.res != hv_ds.res):
    raise ValueError("Растры HH и HV имеют разную геометрию (extent, CRS или разрешение).")

HH = hh_ds.read(1).astype(np.float64)
HV = hv_ds.read(1).astype(np.float64)

# Профиль для записи результатов
profile = hh_ds.profile.copy()
profile.update(count=1, dtype="float32", compress="lzw", bigtiff="yes",
               tiled=True, blockxsize=256, blockysize=256, nodata=-9999.0)

hh_nodata = hh_ds.nodata
hv_nodata = hv_ds.nodata
hh_ds.close()
hv_ds.close()

# --- Проверка: дБ или линейные значения ---------------------------------------

hh_valid = HH[~np.isnan(HH)]
if hh_nodata is not None:
    hh_valid = hh_valid[hh_valid != hh_nodata]
hv_valid = HV[~np.isnan(HV)]
if hv_nodata is not None:
    hv_valid = hv_valid[hv_valid != hv_nodata]

hh_mean = np.mean(hh_valid) if len(hh_valid) > 0 else 0
hv_mean = np.mean(hv_valid) if len(hv_valid) > 0 else 0

is_db = False
if hh_mean < 0 or hv_mean < 0:
    print("ВНИМАНИЕ: Обнаружены отрицательные средние значения.")
    print("Данные интерпретируются как дБ и будут конвертированы в линейные единицы.")
    is_db = True
    HH = np.power(10.0, HH / 10.0)
    HV = np.power(10.0, HV / 10.0)
    print("Конвертация из дБ в linear (power) выполнена.\n")
else:
    print("Данные интерпретируются как линейные (power) единицы.\n")

# Маска nodata (объединённая)
nodata_mask = np.isnan(HH) | np.isnan(HV)
if hh_nodata is not None and not is_db:
    nodata_mask |= (HH == hh_nodata)
if hv_nodata is not None and not is_db:
    nodata_mask |= (HV == hv_nodata)

# --- Определение SAR-индексов -------------------------------------------------

indices_list = [
    {
        "name": "RVI",
        "desc": "Radar Vegetation Index (dual-pol): 4*HV/(HH+HV)",
        "fun": lambda hh, hv: 4 * hv / (hh + hv),
    },
    {
        "name": "RFDI",
        "desc": "Radar Forest Degradation Index: (HH-HV)/(HH+HV)",
        "fun": lambda hh, hv: (hh - hv) / (hh + hv),
    },
    {
        "name": "HH_HV_Ratio",
        "desc": "Polarization Ratio: HV/HH",
        "fun": lambda hh, hv: hv / hh,
    },
    {
        "name": "HH_HV_Sum",
        "desc": "Total backscatter: HH+HV",
        "fun": lambda hh, hv: hh + hv,
    },
    {
        "name": "HH_HV_Diff",
        "desc": "Backscatter difference: HH-HV",
        "fun": lambda hh, hv: hh - hv,
    },
    {
        "name": "HH_HV_Mean",
        "desc": "Mean backscatter: (HH+HV)/2",
        "fun": lambda hh, hv: (hh + hv) / 2,
    },
    {
        "name": "NDI",
        "desc": "Normalized Difference Index: (HH-HV)/(HH+HV) (=RFDI)",
        "fun": lambda hh, hv: (hh - hv) / (hh + hv),
    },
    {
        "name": "HV_dB",
        "desc": "HV in dB: 10*log10(HV)",
        "fun": lambda hh, hv: 10 * np.log10(np.maximum(hv, 1e-10)),
    },
    {
        "name": "HH_dB",
        "desc": "HH in dB: 10*log10(HH)",
        "fun": lambda hh, hv: 10 * np.log10(np.maximum(hh, 1e-10)),
    },
    {
        "name": "Ratio_dB",
        "desc": "Polarization ratio in dB: 10*log10(HV/HH)",
        "fun": lambda hh, hv: 10 * np.log10(np.maximum(hv / np.maximum(hh, 1e-10), 1e-10)),
    },
    {
        "name": "BMI",
        "desc": "Biomass Index (proxy): (HH+HV)/2",
        "fun": lambda hh, hv: (hh + hv) / 2,
    },
    {
        "name": "VSI",
        "desc": "Volume Scattering Index (proxy): HV / (HH + HV) * 100",
        "fun": lambda hh, hv: hv / (hh + hv) * 100,
    },
    {
        "name": "Span",
        "desc": "Span (total power, dual-pol approx): HH + 2*HV",
        "fun": lambda hh, hv: hh + 2 * hv,
    },
    {
        "name": "PDI",
        "desc": "Polarimetric Discrimination Index: sqrt(HH^2 + HV^2)",
        "fun": lambda hh, hv: np.sqrt(hh**2 + hv**2),
    },
]

n_indices = len(indices_list)
print(f"Будет рассчитано {n_indices} индексов:")
for i, idx in enumerate(indices_list, 1):
    print(f"  [{i:02d}] {idx['name']:<20s} - {idx['desc']}")
print()

# --- Расчёт индексов ---------------------------------------------------------

print("Начинаю расчёт индексов...\n")

saved_files = []

for idx_info in tqdm(indices_list, desc="Расчёт индексов", ncols=70):
    idx_name = idx_info["name"]
    idx_fun  = idx_info["fun"]

    try:
        result = idx_fun(HH, HV).astype(np.float32)

        # Применяем маску nodata и заменяем Inf
        result[nodata_mask] = -9999.0
        result[~np.isfinite(result)] = -9999.0

        # Сохраняем
        out_name = f"SAR_{idx_name}.tif"
        out_path = os.path.join(output_dir, out_name)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(result, 1)
            dst.set_band_description(1, f"SAR_{idx_name}")

        saved_files.append(out_path)

    except Exception as e:
        print(f"\n  ОШИБКА при расчёте {idx_name}: {e}")

# --- Итоговый отчёт -----------------------------------------------------------

print(f"\n\n========================================")
print(f"  РАСЧЁТ ЗАВЕРШЁН")
print(f"========================================")
print(f"Рассчитано индексов: {len(saved_files)}")
print(f"Результаты сохранены в: {output_dir}\n")

print("Список сохранённых файлов (каждый — отдельный однослойный TIF):")
for f in saved_files:
    size_mb = os.path.getsize(f) / (1024 ** 2)
    print(f"  - {os.path.basename(f)} ({size_mb:.1f} MB)")

print("\n--- Описание индексов ---")
for idx in indices_list:
    print(f"  {idx['name']:<20s} : {idx['desc']}")

print("\nГотово!")
