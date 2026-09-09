# classification/ — пайплайн классификации растительности

Многоклассовая классификация растительного покрова по многослойному композиту `Composite_filtered.tif`, подготовленному в `common/00_Prepare_Layers.py`.

## Обзор пайплайна

```
common/00_Prepare_Layers.py   →   Composite_filtered.tif + predictor_names.pkl
                                          │
                                          ▼
      01_prepare_data.py         обучающая выборка из полигонов (samples.gpkg)
                                          │
                                          ▼
      02_classification.py       отбор предикторов, диагностика, разбиения
                                          │
                                          ▼
      03_tune_models.py          обучение и оценка RF / CatBoost / LightGBM /
                                 XGBoost / Ridge / LDA / QDA (Spatial Block CV)
                                          │
                                          ▼
      04_predict_map.py          финальная растровая карта + QGIS-проект + QML
                                          │
                                          ▼
      05_classwise_importance.py per-class permutation importance + GAM-отклики
                                          │
                                          ▼
      06_sieve_marching_squares.py sieve + сглаженный векторный слой
```

## Скрипты

### 01_prepare_data.py
Собирает обучающую выборку. Читает полигоны + `Composite_filtered.tif`. Извлекает значения предикторов для каждого пикселя внутри полигона; для точечных эталонов использует `rasterio.sample()`. Сохраняет `samples.gpkg` со слоем `sample_pixels` и служебными полями `polygon_id`, `n_pixels`, `class_name`. Регистронезависимые коллизии имён слоёв (SQLite) устраняются суффиксами `_2`, `_3`; исходные имена сохраняются в `raw_layer_names`/`band_name_map`.

### 02_classification.py
Отбор предикторов и диагностика перед обучением: корреляционный анализ, тест Краскела–Уоллиса по классам, отсечение малоинформативных слоёв, диагностические графики (с автоматическим понижением DPI при риске превышения предела Agg в 65 535 пикселей).

### 03_tune_models.py
Обучение и оценка семи алгоритмов: Random Forest, CatBoost, LightGBM, XGBoost, Ridge, LDA, QDA. Разбиения: **StratifiedGroupKFold по `polygon_id`** (пиксели одного полигона не попадают одновременно в train и test) плюс пространственная блочная кросс-валидация на K-Means по координатам. Метрики: OA, каппа Коэна, TSS, F1-macro, AIC (где применимо). Сохраняет `final_models.pkl`.

### 04_predict_map.py
Строит финальный категориальный растр. Значения классов сохраняются **0-based** (`0..n-1`) с `nodata=-9999`; таблица классов, фильтр статистики и per-class циклы согласованы. Дополнительно выпускает переносимый `.qgs`-проект и сайдкар `.qml` с paletted/unique-values раскраской (палитра `1..24`, значение `v` → цвет `v+1`).

### 05_classwise_importance.py
Пермутационная важность **по классам** (что важно именно для класса `k`) и одномерные GAM-зависимости вероятности принадлежности к классу от значений признаков.

### 06_sieve_marching_squares.py
Читает `04_FINAL/*.tif`, применяет sieve (присоединение мелких групп к соседу с максимальной общей границей), строит сглаженный вектор тайлами. `resume` переиспользует готовые части, промежуточный формат — Parquet (с fallback на Pickle при отсутствии PyArrow). Палитра растра переносится в категории вектора (`QgsCategorizedSymbolRenderer` по полю `class_value`).

## Служебные поля (не предикторы)

Из выборки на всех этапах исключаются:

```
polygon_id  n_pixels  class_name  ID  x  y
```

## Входы и выходы

| Файл | Тип | Где создан | Кем читается |
|---|---|---|---|
| `Composite_filtered.tif` | растр | `common/00_Prepare_Layers.py` | 01, 04 |
| `predictor_names.pkl` | pickle | `common/00_Prepare_Layers.py` | 02, 03, 04, 05 |
| `samples.gpkg` (слой `sample_pixels`) | GeoPackage | 01 | 02, 03 |
| `data_info.pkl` | pickle | 02/03 | 03, 04, 05 |
| `final_models.pkl` | pickle | 03 | 04, 05 |
| `04_FINAL/*.tif` | растр | 04 | 06 |
| `*.qgs`, `*.qml` | QGIS | 04 | — |
| `vec.gpkg` | вектор | 06 | — |
