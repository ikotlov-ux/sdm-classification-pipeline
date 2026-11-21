# segmentation/: фрагменты подложек вокруг точек и сегментация в samgeo

Для каждой точки скачивается свой фрагмент 1×1 км (круг радиусом 500 м) со спутниковой подложки высокого разрешения. Затем все фрагменты сегментируются в [samgeo](https://samgeo.gishub.org/) (SAM 2 / SAM 1). Один огромный GeoTIFF на всю территорию не нужен: хранятся только фрагменты вокруг точек.

| Скрипт | Назначение |
|---|---|
| `10_download_sat_chips.py` | Фрагменты Google / Bing / Yandex / ESRI вокруг точек (CSV / XLSX / GPKG / SHP / GeoJSON / KML). Результат: GeoTIFF с JPEG-сжатием или JPG/PNG/WebP с файлом привязки. Кэш тайлов в SQLite, отсев ошибочных координат (`VALID_BBOX`), объединение точек ближе `DEDUP_M`, необязательный фильтр по AOI (ассет GEE или локальный файл), таблица `chips_index.gpkg` |
| `11_samgeo_segment_chips.py` | Сегментация всех фрагментов: определение видеокарты (PyTorch и `nvidia-smi`), параллельные процессы на CPU или GPU с перезапуском после сбоя, усиление контраста, обрезка по кругу, перевод в UTM 37N. Результат: один слой `segments.gpkg` |

## Порядок работы

1. `pip install segment-geospatial sam2 requests pillow psutil` и PyTorch под вашу CUDA ([pytorch.org](https://pytorch.org/get-started/locally/)).
2. **Тестовый запуск**: в обоих скриптах по умолчанию `TEST_N = 20`.
   - `python segmentation/10_download_sat_chips.py`: выбрать файлы точек в окне Проводника. Скрипт скачает 20 мест и даст прогноз объёма и времени загрузки для всех.
   - `python segmentation/11_samgeo_segment_chips.py`: выбрать папку `sat_chips_<provider>_z<zoom>`. Скрипт обработает 20 фрагментов и даст прогноз времени для всех.
   - Логи и таблица времени по фрагментам сохраняются в `<папка фрагментов>/_logs/`.
3. **Полный запуск**: `TEST_N = 0` в обоих скриптах. Уже готовые фрагменты и сегменты не пересоздаются, прерванный запуск продолжается с места остановки.

## Основные параметры

- **10**: `PROVIDER` (`google` / `bing` / `yandex` / `esri`), `ZOOM` (17 ≈ 0,67 м; 18 ≈ 0,34 м; 19 ≈ 0,17 м на 55° с. ш.), `RADIUS_M`, `OUT_FORMAT`, `DEDUP_M`, `VALID_BBOX`, `SHEET_NAME`, `AOI_SOURCE` (`"gee"` использует ассет `GEE_AOI_ASSET`).
- **11**: `MODEL`, `SAM2_MODEL_ID`, `PRESET` (`quality` / `fast`), `N_WORKERS`, `THREADS_PER_WORKER`, `MEM_PER_WORKER_GB`, `AMG_KWARGS`, `ENHANCE`, `CLIP_TO_CIRCLE`, `OUT_EPSG`.
  - `N_WORKERS = "auto"` на CPU: число физических ядер / `THREADS_PER_WORKER`, но не больше, чем позволяет свободная память.
  - `N_WORKERS = "auto"` на GPU: по одному процессу на карту.

## Поля `segments.gpkg`

`uid`, `chip_id`, `members` (все точки фрагмента), `n_points`, `seg_id`, `area_m2`, `cut_by_circle` (сегмент обрезан окружностью), `center_dist_m` (расстояние от центра сегмента до точки), `geometry`.

## Примечания

- Yandex отдаёт тайлы в проекции EPSG:3395, остальные источники в EPSG:3857. Скрипт учитывает это, привязка фрагментов совпадает.
- Условия использования Google, Bing и Yandex ограничивают массовую загрузку тайлов. Для публикаций безопаснее ESRI World Imagery с атрибуцией «© Esri, Maxar, Earthstar Geographics».
- Дата съёмки у подложек неизвестна и может различаться между соседними фрагментами.
