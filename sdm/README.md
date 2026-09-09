# sdm/ — пайплайн Species Distribution Modeling (presence-only)

Моделирование пригодности местообитаний по данным только о присутствии (presence-only / presence-background). Использует тот же композит предикторов `Composite_filtered.tif`, что и ветвь классификации.

## Обзор пайплайна

```
common/00_Prepare_Layers.py   →   Composite_filtered.tif + predictor_names.pkl
                                          │
                                          ▼
      01s_prepare_sdm.py         точки присутствия (lon/lat CSV) →
                                 извлечение предикторов + Gaussian KDE bias-grid +
                                 10 000 background → samples.gpkg
                                          │
                                          ▼
      02s_select_sdm.py          отбор алгоритмов и предикторов,
                                 стратификация p/bg, Spatial Block CV
                                          │
                                          ▼
      03s_tune_sdm.py            MaxEnt (jar) + LightGBM + GLM
                                 → final_models.pkl (MaxentJarModel на уровне модуля)
                                          │
                                          ▼
      04s_predict_suitability.py непрерывная suitability (Float32) +
                                 бинарные карты по maxSSS и 10-му процентилю
                                          │
                                          ▼
      05s_response_curves.py     permutation importance + response curves
                                          │
                                          ▼
      06s_model_dynamic.py       межгодовая динамика: разность, тренд,
                                 группировка по порогам, матрица переходов
```

## Скрипты

### 01s_prepare_sdm.py
Вход: CSV с колонками `species`, `longitude`, `latitude` (соглашение MaxEnt). Читает `Composite_filtered.tif`, извлекает значения предикторов в точках присутствия, строит **bias-grid методом Gaussian KDE** (подход SDMToolbox), генерирует ~10 000 фоновых точек с учётом bias, сохраняет `samples.gpkg` со столбцом `pa` (1 — присутствие, 0 — фон).

### 02s_select_sdm.py
Наследует Spatial Block CV из ветви классификации. Может читать либо результат `01s_prepare_sdm.py` (со столбцом `pa`), либо результат классификации `01_prepare_data.py` (только присутствия) — тогда назначает существующие точки как присутствия и генерирует фон Gaussian KDE. Стратификация по presence/background. Выбор алгоритмов: MaxEnt (jar), LightGBM, GLM.

### 03s_tune_sdm.py
Обучение и оценка. **MaxEnt (maxent.jar)** вызывается через subprocess (SWD, samples-with-data). LightGBM и GLM — без Java. Метрики: AUC, AUC-PR, **Continuous Boyce Index**, TSS, omission@10pct. `MaxentJarModel` определён на уровне модуля во всех скриптах, читающих `final_models.pkl`.

### 04s_predict_suitability.py
Строит непрерывную карту suitability (Float32) и бинарные карты по двум порогам:
- **maxSSS** — максимизация суммы чувствительности и специфичности;
- **10pct** — 10-й процентиль пригодности в точках присутствия.

### 05s_response_curves.py
Permutation importance (10 повторов, падение AUC при перемешивании значений предиктора) и кривые отклика (response curves) при фиксации остальных предикторов на медианном значении в точках присутствия.

### 06s_model_dynamic.py
Анализирует серию годовых карт suitability. Считает:
1. попиксельную разность между крайними годами;
2. попиксельный линейный тренд (наклон и R²);
3. площади по классам пригодности (заданное число порогов, произвольные значения);
4. карту переходов между классами с полной матрицей переходов.

## Служебные поля (не предикторы)

```
pa  x  y  ID
```

## Входные данные

**CSV точек присутствия** (формат MaxEnt):

```
species,longitude,latitude
Panthera_tigris,137.061,49.089
Panthera_tigris,137.244,49.132
...
```

## Установка MaxEnt

MaxEnt (Java) в репозитории не хранится. Скачать `maxent.jar` (v3.4.x): <https://biodiversityinformatics.amnh.org/open_source/maxent/>. Путь к `maxent.jar` задаётся в 03s через переменную окружения или параметр скрипта. Требуется JRE 8+.

## Особенности presence-only при малых выборках

При 14–52 точках присутствия в год (как в погодичном анализе амурского тигра):
- ограничивать сложность MaxEnt линейными и квадратичными признаками (threshold-признаки отключить);
- повторять отбор предикторов **для каждого года**;
- оценивать AUC-CV, Continuous Boyce Index, TSS, maxSSS;
- прогноз интерпретировать как **относительную пригодность**, а не абсолютную вероятность присутствия.

## Входы и выходы

| Файл | Тип | Где создан | Кем читается |
|---|---|---|---|
| `Composite_filtered.tif` | растр | `common/00_Prepare_Layers.py` | 01s, 04s |
| `predictor_names.pkl` | pickle | `common/00_Prepare_Layers.py` | все |
| Presence CSV | csv | пользовательский | 01s |
| `samples.gpkg` (со `pa`) | GeoPackage | 01s | 02s, 03s |
| `bias.tif` | растр | 01s | 01s (внутренне) |
| `final_models.pkl` | pickle | 03s | 04s, 05s |
| `suitability_<year>.tif` | растр | 04s | 06s |
| `thresholds.json` | JSON | 04s | 06s |
| `importance_maxent.csv/jpg` | CSV/JPG | 05s | — |
| `response_curves_*.pkl/jpg` | pickle/JPG | 05s | — |
| `dynamic_summary.csv`, `transition_matrix.csv`, `groups_*.tif`, `trend_*.tif`, `change_status_*.jpg` | различные | 06s | — |
