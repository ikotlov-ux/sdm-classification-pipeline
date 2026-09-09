# sdm-classification-pipeline

Единый репозиторий двух связанных пайплайнов дистанционного зондирования и экологического моделирования:

1. **Классификация растительного покрова** по мультисенсорным спутниковым и рельефным данным (Sentinel-2, Landsat, MODIS, Sentinel-1, PALSAR, ЦМР).
2. **Моделирование пригодности местообитаний (SDM)** по данным только о присутствии (presence-only / presence-background) с использованием MaxEnt.

Оба пайплайна разделяют общий этап подготовки предикторов, а затем расходятся по разным задачам и метрикам оценки.

## Развилка пайплайнов

```
                  ┌───────────────────────────────────┐
                  │  common/00_Prepare_Layers.py       │
                  │  Подготовка мультислойного растра  │
                  │  Composite_filtered.tif            │
                  └────────────────┬──────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
        ┌───────────▼───────────┐   ┌─────────────▼──────────────┐
        │  classification/       │   │  sdm/                       │
        │  01 → 02 → 03 → 04    │   │  01s → 02s → 03s → 04s     │
        │  → 05 → 06            │   │  → 05s → 06s                │
        │                        │   │                             │
        │  Полигоны обучающих   │   │  Presence-only точки        │
        │  участков             │   │  + background с bias-grid   │
        │                        │   │                             │
        │  Multi-class output    │   │  Continuous suitability     │
        │  + сглаженный вектор  │   │  + бинарные пороги          │
        └───────────────────────┘   └────────────────────────────┘
```

Подробная схема — в `docs/pipeline_diagram.md`.

## Структура репозитория

```
sdm-classification-pipeline/
├── common/                 общая часть (шаг 00)
│   └── 00_Prepare_Layers.py
├── classification/         пайплайн классификации (шаги 01–06)
├── sdm/                    пайплайн SDM (шаги 01s–06s)
├── docs/                   схема развилки и заметки
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Быстрый старт

### Установка

```bash
git clone https://github.com/ikotlov-ux/sdm-classification-pipeline.git
cd sdm-classification-pipeline
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Для SDM-ветки дополнительно нужен `maxent.jar` (Phillips et al., 2021) и установленный JRE (Java 8+). Скачать: <https://biodiversityinformatics.amnh.org/open_source/maxent/>.

### Общий шаг

```bash
python common/00_Prepare_Layers.py
```

Готовит выровненный на общую сетку многослойный композит `Composite.tif` и его отфильтрованную по ZV/NZV версию `Composite_filtered.tif`, а также `predictor_names.pkl`.

### Ветвь классификации

```bash
python classification/01_prepare_data.py           # обучающая выборка из полигонов
python classification/02_classification.py         # выбор предикторов, диагностика
python classification/03_tune_models.py            # обучение RF / CatBoost / LightGBM / XGBoost / Ridge / LDA / QDA
python classification/04_predict_map.py            # финальная растровая карта + QGIS-проект
python classification/05_classwise_importance.py   # per-class permutation importance + GAM-отклики
python classification/06_sieve_marching_squares.py # sieve + сглаженный вектор
```

**Метрики оценки:** OA, каппа Коэна, TSS, F1-macro. Пространственная блочная кросс-валидация (K-Means по координатам).

### Ветвь SDM

```bash
python sdm/01s_prepare_sdm.py            # точки присутствия + bias-grid + background
python sdm/02s_select_sdm.py             # отбор алгоритмов, spatial block CV
python sdm/03s_tune_sdm.py               # MaxEnt (jar), LightGBM, GLM
python sdm/04s_predict_suitability.py    # непрерывная suitability + бинаризация (maxSSS, 10pct)
python sdm/05s_response_curves.py        # permutation importance + response curves
python sdm/06s_model_dynamic.py          # межгодовая динамика suitability
```

**Метрики оценки:** AUC, AUC-PR, Continuous Boyce Index, TSS, omission@10pct. Только Spatial Block CV.

## Что общего и в чём различие

| Шаг | Классификация | SDM |
|---|---|---|
| **Целевая переменная** | `class_name` (мультикласс) | `pa` (1/0, presence-background) |
| **Выборка** | пиксели полигонов | точки присутствия + background из bias-grid |
| **Служебные поля** | `polygon_id`, `n_pixels`, `class_name`, `ID`, `x`, `y` | `pa`, `x`, `y`, `ID` |
| **Валидация** | StratifiedGroupKFold по `polygon_id` + Spatial Block CV | только Spatial Block CV |
| **Основные модели** | RF, CatBoost, LightGBM, XGBoost, Ridge, LDA, QDA | MaxEnt (jar), LightGBM, GLM |
| **Пороги** | argmax класса | maxSSS, 10-й процентиль |
| **Выход** | категориальный растр + вектор | непрерывная suitability + бинарные карты |

## Данные и артефакты

Крупные файлы (растры, векторы, обученные модели) в репозитории **не хранятся** — см. `.gitignore`. Хранение:
- Локально: рекомендуемая структура `data/`, `outputs/`, `predictions/`.
- Публичные ссылки — в `docs/` (при необходимости).

## Ссылки

- Phillips, S. J.; Anderson, R. P.; Schapire, R. E. Maximum Entropy Modeling of Species Geographic Distributions. Ecol. Modell. 2006, 190, 231–259.
- Boyce, M. S. et al. Evaluating Resource Selection Functions. Ecol. Modell. 2002, 157, 281–300.
- Roberts, D. R. et al. Cross-Validation Strategies for Data with Spatial Structure. Ecography 2017, 40, 913–929.
- Brown, J. L. et al. SDMtoolbox 2.0. PeerJ 2017, 5, e4095.

## Лицензия

MIT. См. `LICENSE`.

## Автор

Иван Котлов ([@ikotlov-ux](https://github.com/ikotlov-ux)).
