# Схема развилки пайплайнов

## Полная схема (текстовая)

```
                          ╔═══════════════════════════════════════════════╗
                          ║              ОБЩАЯ ЧАСТЬ (common/)             ║
                          ╠═══════════════════════════════════════════════╣
                          ║                                                ║
                          ║  Источники: Sentinel-2, Landsat, MODIS,        ║
                          ║             Sentinel-1, PALSAR, ЦМР            ║
                          ║                       │                        ║
                          ║                       ▼                        ║
                          ║      00_Prepare_Layers.py                      ║
                          ║  ─ выравнивание на общую сетку                 ║
                          ║  ─ индексы (spyndex, WhiteboxTools)            ║
                          ║  ─ фильтр ZV/NZV                               ║
                          ║                       │                        ║
                          ║                       ▼                        ║
                          ║  Composite_filtered.tif + predictor_names.pkl  ║
                          ║                                                ║
                          ╚════════════════════╤══════════════════════════╝
                                               │
                                               ▼
                          ┌────────────── FORK ─────────────┐
                          │                                  │
                          ▼                                  ▼
╔═══════════════════════════════════════╗   ╔═══════════════════════════════════════╗
║  classification/                       ║   ║  sdm/                                  ║
╠═══════════════════════════════════════╣   ╠═══════════════════════════════════════╣
║  Задача: мультиклассовая классификация ║   ║  Задача: presence-only SDM             ║
║          растительного покрова         ║   ║          (пригодность местообитаний)   ║
║                                        ║   ║                                        ║
║  01_prepare_data.py                    ║   ║  01s_prepare_sdm.py                    ║
║    Вход: полигоны обучающих участков  ║   ║    Вход: CSV точек присутствия        ║
║    → samples.gpkg (пиксели полигонов)  ║   ║    → bias-grid Gaussian KDE            ║
║    Служ. поля: polygon_id, class_name  ║   ║    → background (10 000 точек)         ║
║                       │                ║   ║    → samples.gpkg (pa = 1/0)           ║
║                       ▼                ║   ║                       │                ║
║  02_classification.py                  ║   ║                       ▼                ║
║    Корр. анализ, Kruskal–Wallis        ║   ║  02s_select_sdm.py                     ║
║    Отбор предикторов                   ║   ║    Отбор алгоритмов                    ║
║                       │                ║   ║    Стратификация p/bg                  ║
║                       ▼                ║   ║                       │                ║
║  03_tune_models.py                     ║   ║                       ▼                ║
║    RF / CatBoost / LightGBM /          ║   ║  03s_tune_sdm.py                       ║
║    XGBoost / Ridge / LDA / QDA         ║   ║    MaxEnt (jar) / LightGBM / GLM      ║
║    StratifiedGroupKFold(polygon_id)    ║   ║    Spatial Block CV                    ║
║    + Spatial Block CV                  ║   ║    Метрики: AUC, AUC-PR, Boyce, TSS,  ║
║    Метрики: OA, Kappa, TSS, F1-macro  ║   ║             omission@10pct             ║
║                       │                ║   ║                       │                ║
║                       ▼                ║   ║                       ▼                ║
║  04_predict_map.py                     ║   ║  04s_predict_suitability.py            ║
║    Растр 0..n-1, nodata=-9999          ║   ║    Float32 suitability                 ║
║    + QGIS .qgs + .qml                  ║   ║    + бинаризация (maxSSS, 10pct)       ║
║                       │                ║   ║                       │                ║
║                       ▼                ║   ║                       ▼                ║
║  05_classwise_importance.py            ║   ║  05s_response_curves.py                ║
║    Per-class permutation importance    ║   ║    Permutation importance              ║
║    GAM-отклики P(class|x_j)            ║   ║    Response curves                     ║
║                       │                ║   ║                       │                ║
║                       ▼                ║   ║                       ▼                ║
║  06_sieve_marching_squares.py          ║   ║  06s_model_dynamic.py                  ║
║    Sieve + сглаженный вектор           ║   ║    Разность, тренд, группы,           ║
║    QgsCategorizedSymbolRenderer        ║   ║    матрица переходов между годами     ║
║                                        ║   ║                                        ║
╚═══════════════════════════════════════╝   ╚═══════════════════════════════════════╝
```

## Что реально общее

**Общее (physically):**
- Файл `common/00_Prepare_Layers.py` — код используется как есть обеими ветками.
- Выходы этого шага (`Composite_filtered.tif`, `predictor_names.pkl`) — читаются обеими.

**Общее (конвенции):**
- Единый набор служебных полей исключается из предикторов (различается частично: классификация — `polygon_id`, `n_pixels`, `class_name`, `ID`, `x`, `y`; SDM — `pa`, `x`, `y`, `ID`).
- Spatial Block CV на одном и том же наборе координат.
- Единый список имён предикторов.

**Различное:**
- Целевая переменная (мультикласс vs. presence/background).
- Тип выборки (пиксели полигонов vs. точки + background).
- Алгоритмы моделирования.
- Метрики оценки.
- Формат итоговой карты (категориальный растр vs. непрерывная suitability + бинарные пороги).

## Развитие

При дальнейшем расширении держать общие утилиты, если появятся, в `common/utils/` (например, единый Spatial Block CV, общая обёртка над rasterio-чтением) — тогда обе ветки будут импортировать одну реализацию.
