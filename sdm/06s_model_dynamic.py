# ============================================================================
# 06s_model_dynamic_v3.py
# СКРИПТ 06s: ДИНАМИКА МОДЕЛЕЙ ПРИГОДНОСТИ МЕСТООБИТАНИЙ (HABITAT DYNAMICS)
#
# ИЗМЕНЕНИЯ В v3:
#   - ДОБАВЛЕН ПОЛЬЗОВАТЕЛЬСКИЙ ВАРИАНТ ПОРОГОВ [3]: пользователь задаёт
#     КОЛИЧЕСТВО порогов (например, 2) и их ЗНАЧЕНИЯ (например,
#     0.5 и 0.75). N порогов дают N+1 ГРУППУ пригодности
#     (например, 2 порога → 3 группы: [<0.5], [0.5–0.75), [≥0.75]).
#   - Расчёт площадей, карты классов пригодности и change map теперь
#     работают с произвольным числом групп. Для вариантов [1]/[2]
#     (один порог) поведение прежнее: 2 группы (непригодно/пригодно),
#     change map gain/loss/стабильно.
#
# ИЗМЕНЕНИЯ В v2:
#   - ГОД СРЕЗА ВСЕГДА ЗАДАЁТ ПОЛЬЗОВАТЕЛЬ вручную для каждой
#     выбранной папки (режим 'runs'). Раньше год ошибочно брался из
#     штампа даты запуска DDMMYY (например, '200626' → ложно 2006).
#   - Штамп даты запуска (DDMMYY_HHMM) БОЛЬШЕ НЕ интерпретируется
#     как год среза.
#   - В режиме 'single' (один каталог) год по-прежнему берётся из
#     имени ФАЙЛА suitability_<model>_<год>.tif (явный год в имени),
#     а не из имени папки.
#
# Шаг ПОСЛЕ 04s_predict_suitability.py. Берёт растровые карты пригодности
# suitability_*.tif (Float32, [0,1], nodata=-1.0) из нескольких ВРЕМЕННЫХ
# СРЕЗОВ (по годам / датам) и анализирует их динамику во времени.
#
# ЧТО ДЕЛАЕТ:
#   1) ИЗМЕНЕНИЯ МЕЖДУ ГОДАМИ/ДАТАМИ:
#      - попарные разности suitability (последний минус первый, год к году);
#      - попиксельный линейный тренд (наклон регрессии suitability ~ год),
#        в единицах suitability/год;
#   2) СТАТИСТИКА ПЛОЩАДЕЙ ПО ПОРОГАМ (бинаризация maxSSS / p10):
#      - площадь пригодных местообитаний по каждому срезу (га и км²),
#        таблица + графики динамики площадей;
#   3) КАРТЫ ИЗМЕНЕНИЙ (change maps):
#      - растр разности suitability (Float32);
#      - карты классов/групп пригодности по срезам (groups_*);
#      - один порог: бинарная карта статуса: появление (gain) /
#        исчезновение (loss) / стабильно пригодно / стабильно непригодно;
#      - несколько порогов: карта переходов групп (вверх/вниз/стаб.) +
#        полная матрица переходов групп (transition_matrix_*.csv);
#   4) СВОДНЫЕ ГРАФИКИ И ОТЧЁТ:
#      - PNG-панели (срезы, тренд, change map, динамика площадей);
#      - CSV-сводка по всем срезам/моделям.
#
# ИСТОЧНИК ДАННЫХ:
#   По умолчанию читает несколько запусков из 04_FINAL_SDM (каждый запуск =
#   один временной срез). Каждому срезу нужно сопоставить ГОД (или индекс
#   времени). Год определяется автоматически из имени папки/файла (4 цифры
#   19xx/20xx), иначе запрашивается у пользователя.
#   Альтернатива: одна папка с набором suitability_<model>_<год>.tif.
#
# СОГЛАШЕНИЯ (как в остальном SDM-пайплайне):
#   - suitability: Float32, [0,1], nodata = -1.0;
#   - бинарные карты: 1 = пригодно, 0 = непригодно, 255 = nodata;
#   - выбор подпапки и суффикс _vNNN наследуются из 04s.
#
# Автор пайплайна: I.P. Kotlov. Скрипт под JupyterLab / CLI.
# ============================================================================

import os
import re
import glob
import pickle
import warnings
from datetime import datetime

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as RIOResampling

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

warnings.filterwarnings("ignore")

# Служебное значение nodata для suitability (как в 04s)
SUIT_NODATA = -1.0
# Год из имени (4 цифры 1900-2099)
_YEAR_PAT = re.compile(r"(19\d{2}|20\d{2})")


# ============================================================================
# ВЫБОР ВРЕМЕННЫХ СРЕЗОВ
# ============================================================================

def _detect_year(text):
    """Извлечь год (int) из строки или None."""
    m = _YEAR_PAT.search(str(text))
    return int(m.group(1)) if m else None


def _list_run_subdirs(base_dir):
    """Список подпапок-запусков (с суффиксом _vNNN) внутри base_dir."""
    if not os.path.isdir(base_dir):
        return []
    return sorted(
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and re.search(r"_v\d{3}$", d)
    )


def _select_time_slices(root_dir):
    """Интерактивный выбор временных срезов карт пригодности.

    Возвращает:
      slices : список dict {label, year, dir, suffix}
      mode   : 'runs' (несколько запусков 04_FINAL_SDM) или
               'single' (один каталог с suitability_*_<год>.tif)
    """
    base = os.path.join(root_dir, "04_FINAL_SDM")
    runs = _list_run_subdirs(base)

    print("Выберите способ задания временных срезов:\n")
    print("  [1] Несколько запусков из 04_FINAL_SDM (каждый запуск = один срез)")
    print("  [2] Один каталог с файлами suitability_<model>_<год>.tif")
    raw = input("\nВаш выбор (Enter = 1): ").strip() or "1"

    slices = []
    if raw == "2":
        # Один каталог с годами в именах файлов
        if runs:
            print("\nДоступные каталоги в 04_FINAL_SDM:\n")
            for i, sd in enumerate(runs, 1):
                print(f" [{i}] {sd}")
            print(f" [{len(runs)+1}] Указать путь вручную")
            rr = input("\nВыберите каталог (Enter = 1): ").strip() or "1"
            idx = int(rr) - 1
            if idx < len(runs):
                cat_dir = os.path.join(base, runs[idx])
            else:
                cat_dir = input("Путь к каталогу: ").strip()
        else:
            cat_dir = input("Путь к каталогу с suitability_*.tif: ").strip()
        m = re.search(r"(_v\d{3})$", os.path.basename(cat_dir.rstrip("/")))
        suffix = m.group(1) if m else "_v001"
        return {"mode": "single", "dir": cat_dir, "suffix": suffix}

    # Режим запусков
    if not runs:
        raise FileNotFoundError(
            "В 04_FINAL_SDM нет результатов. Сначала запустите "
            "04s_predict_suitability.py для каждого временного среза."
        )
    print("\nДоступные запуски в 04_FINAL_SDM (отметьте срезы):\n")
    # ВАЖНО: имя папки начинается со штампа даты запуска DDMMYY_HHMM —
    # это НЕ год среза. Год задаёт пользователь ниже.
    for i, sd in enumerate(runs, 1):
        print(f" [{i}] {sd}")
    raw = input(
        "\nНомера запусков через запятую (Enter = все, по порядку): "
    ).strip()
    if raw:
        chosen = [runs[int(x) - 1] for x in raw.split(",") if x.strip()]
    else:
        chosen = runs

    print("\nУкажите ГОД (или метку времени) для каждой выбранной папки:\n")
    for sd in chosen:
        d = os.path.join(base, sd)
        # Всегда спрашиваем год у пользователя (повтор при пустом/неверном вводе).
        yr = None
        while yr is None:
            ans = input(f"  Год для '{sd}': ").strip()
            if not ans:
                print("    ! Год обязателен. Введите четырёхзначный год, например 2006.")
                continue
            try:
                yr = int(ans)
            except ValueError:
                print("    ! Нужно целое число (год), например 2006.")
                yr = None
        m = re.search(r"(_v\d{3})$", sd)
        slices.append({
            "label": str(yr), "year": yr, "dir": d,
            "suffix": (m.group(1) if m else "_v001"),
        })

    # Сортируем по году (все годы теперь известны)
    slices.sort(key=lambda s: s["year"])

    # Предупреждение о дублях годов
    yrs = [s["year"] for s in slices]
    dup = {y for y in yrs if yrs.count(y) > 1}
    if dup:
        print(f"\n  ⚠ Одинаковый год у нескольких срезов: {sorted(dup)}. "
              "Проверьте корректность ввода.")
    return {"mode": "runs", "slices": slices,
            "suffix": (slices[0]["suffix"] if slices else "_v001")}


# ============================================================================
# СБОР КАРТ ПРИГОДНОСТИ ПО МОДЕЛЯМ И СРЕЗАМ
# ============================================================================

def _collect_suitability(plan, root_dir):
    """Собрать пути suitability по структуре {model: [(year, path), ...]}.

    Также возвращает порядок моделей и список (label, year) срезов.
    """
    by_model = {}
    slice_meta = []  # (label, year)

    if plan["mode"] == "single":
        cat = plan["dir"]
        tifs = sorted(glob.glob(os.path.join(cat, "suitability_*.tif")))
        if not tifs:
            raise FileNotFoundError(
                f"В {cat} нет файлов suitability_*.tif"
            )
        # имя: suitability_<model>_<год>.tif  или suitability_<model>.tif
        for tf in tifs:
            base = os.path.basename(tf)[len("suitability_"):-len(".tif")]
            yr = _detect_year(base)
            # модель = всё без года
            model = re.sub(r"_?(19\d{2}|20\d{2})", "", base).strip("_") or "model"
            by_model.setdefault(model, []).append((yr, tf))
        # метки срезов = уникальные годы
        years = sorted({y for lst in by_model.values() for (y, _) in lst
                        if y is not None})
        slice_meta = [(str(y), y) for y in years]
    else:
        slices = plan["slices"]
        for s in slices:
            slice_meta.append((s["label"], s["year"]))
            tifs = sorted(glob.glob(os.path.join(s["dir"], "suitability_*.tif")))
            for tf in tifs:
                base = os.path.basename(tf)[len("suitability_"):-len(".tif")]
                model = base  # имя модели в этом запуске
                by_model.setdefault(model, []).append((s["year"], tf))

    # сортировка внутри модели по году (None в конец)
    for model in by_model:
        by_model[model].sort(key=lambda t: (t[0] is None, t[0]))
    return by_model, slice_meta


# ============================================================================
# ВЫРАВНИВАНИЕ РАСТРОВ (на сетку первого среза)
# ============================================================================

def _read_aligned(paths, ref_path):
    """Прочитать список растров, выровняв их на сетку ref_path.

    Возвращает (stack, profile, valid_mask):
      stack       : (n_time, H, W) Float32, nodata → np.nan
      profile     : профиль опорного растра
      valid_mask  : (H, W) bool — где во ВСЕ срезы есть валидные данные
    """
    with rasterio.open(ref_path) as ref:
        ref_prof = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs
        H, W = ref.height, ref.width

    layers = []
    for p in paths:
        with rasterio.open(p) as src:
            same_grid = (
                src.crs == ref_crs and src.width == W and src.height == H
                and src.transform.almost_equals(ref_transform)
            )
            if same_grid:
                a = src.read(1).astype(np.float32)
                nodata = src.nodata
            else:
                # перепроецируем/ресемплируем на опорную сетку
                a = np.full((H, W), SUIT_NODATA, dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=a,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=ref_transform, dst_crs=ref_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata, dst_nodata=SUIT_NODATA,
                )
                nodata = SUIT_NODATA
        # nodata → nan; отрицательные (служебный nodata) → nan
        if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
            a = np.where(a == nodata, np.nan, a)
        a = np.where(a < 0, np.nan, a)
        layers.append(a)

    stack = np.stack(layers, axis=0)
    valid_mask = np.all(np.isfinite(stack), axis=0)
    return stack, ref_prof, valid_mask


# ============================================================================
# МЕТРИКИ ДИНАМИКИ
# ============================================================================

def _pixel_trend(stack, years, valid_mask):
    """Попиксельный линейный тренд suitability ~ год (наклон, ед./год).

    Возвращает (slope, r2) — массивы (H, W), вне valid_mask = nan.
    """
    n_t, H, W = stack.shape
    x = np.asarray(years, dtype=np.float64)
    x = x - x.mean()
    Sxx = np.sum(x * x)

    flat = stack.reshape(n_t, -1)             # (n_t, P)
    vmask = valid_mask.ravel()
    slope = np.full(flat.shape[1], np.nan, dtype=np.float32)
    r2 = np.full(flat.shape[1], np.nan, dtype=np.float32)

    if Sxx == 0:
        return slope.reshape(H, W), r2.reshape(H, W)

    Y = flat[:, vmask]                        # (n_t, Pv)
    ybar = Y.mean(axis=0)
    Sxy = np.sum(x[:, None] * (Y - ybar[None, :]), axis=0)
    b = Sxy / Sxx                             # наклон
    yhat = ybar[None, :] + b[None, :] * x[:, None]
    ss_res = np.sum((Y - yhat) ** 2, axis=0)
    ss_tot = np.sum((Y - ybar[None, :]) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_v = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)

    slope[vmask] = b.astype(np.float32)
    r2[vmask] = r2_v.astype(np.float32)
    return slope.reshape(H, W), r2.reshape(H, W)


def _pixel_area_ha(profile):
    """Площадь одного пикселя в гектарах (по transform; учёт градусов)."""
    t = profile["transform"]
    px_w = abs(t.a)
    px_h = abs(t.e)
    crs = profile.get("crs")
    if crs is not None and getattr(crs, "is_geographic", False):
        # приблизительный перевод градусов в метры на средней широте
        # (для точной площади используйте проецированную CRS)
        m_per_deg = 111320.0
        area_m2 = (px_w * m_per_deg) * (px_h * m_per_deg)
    else:
        area_m2 = px_w * px_h  # предполагаются метры
    return area_m2 / 10000.0


def _area_by_threshold(stack, valid_mask, threshold, area_ha):
    """Площадь пригодных пикселей (>= threshold) по каждому срезу, га."""
    areas = []
    for t in range(stack.shape[0]):
        a = stack[t]
        suitable = valid_mask & np.isfinite(a) & (a >= threshold)
        areas.append(float(suitable.sum()) * area_ha)
    return areas


def _change_status(first, last, threshold, valid_mask):
    """Карта статуса изменений между первым и последним срезом.

    Коды: 0 = стабильно непригодно, 1 = появление (gain),
          2 = исчезновение (loss), 3 = стабильно пригодно, 255 = nodata.
    """
    H, W = first.shape
    status = np.full((H, W), 255, dtype=np.uint8)
    vm = valid_mask & np.isfinite(first) & np.isfinite(last)
    f = first >= threshold
    l = last >= threshold
    status[vm & (~f) & (~l)] = 0
    status[vm & (~f) & (l)] = 1   # gain
    status[vm & (f) & (~l)] = 2   # loss
    status[vm & (f) & (l)] = 3
    return status


# ============================================================================
# СОХРАНЕНИЕ И ВИЗУАЛИЗАЦИЯ
# ============================================================================

def _save_float(arr, profile, out_path, nodata=SUIT_NODATA):
    prof = profile.copy()
    prof.update(dtype="float32", count=1, nodata=nodata, compress="lzw")
    out = np.where(np.isfinite(arr), arr, nodata).astype(np.float32)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(out, 1)


def _save_byte(arr, profile, out_path, nodata=255):
    prof = profile.copy()
    prof.update(dtype="uint8", count=1, nodata=nodata, compress="lzw")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(arr.astype(np.uint8), 1)


def _plot_change_status(status, out_png, title):
    cmap = ListedColormap(["#d9d9d9", "#2c7fb8", "#d73027", "#1a9850"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    disp = np.where(status == 255, np.nan, status).astype(float)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(disp, cmap=cmap, norm=norm)
    ax.set_title(title, fontweight="bold")
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels([
        "стабильно непригодно", "появление (gain)",
        "исчезновение (loss)", "стабильно пригодно",
    ])
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_trend(slope, out_png, title):
    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = np.nanmax(np.abs(slope)) if np.isfinite(slope).any() else 1.0
    vmax = vmax if vmax > 0 else 1.0
    im = ax.imshow(slope, cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Δ suitability / год")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_area_dynamics(years, areas_by_model, threshold_name, out_png):
    """Динамика «пригодной» площади (выше первого порога) по моделям."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model, areas in areas_by_model.items():
        ax.plot(years, areas, marker="o", linewidth=2, label=model)
    ax.set_xlabel("Год")
    ax.set_ylabel("Площадь пригодных местообитаний, га")
    ax.set_title(
        f"Динамика площади пригодных местообитаний (пороги {threshold_name})",
        fontweight="bold",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_group_dynamics(years, group_areas_by_model, group_names, out_png):
    """Динамика площадей КАЖДОЙ группы по годам.

    group_areas_by_model: {model: [[g0,g1,...], ...]}  (по срезам -> по группам)
    Отдельная панель (subplot) на каждую модель; линия на каждую группу.
    """
    models = list(group_areas_by_model.keys())
    n = len(models)
    if n == 0:
        return
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.8 * nrows))
    axes = np.atleast_1d(axes).flatten()
    n_groups = len(group_names)
    for ax, model in zip(axes, models):
        rows = group_areas_by_model[model]  # [[g0,g1,...], ...]
        for g in range(n_groups):
            series = [rows[t][g] for t in range(len(rows))]
            ax.plot(years, series, marker="o", linewidth=2, label=group_names[g])
        ax.set_title(model, fontweight="bold")
        ax.set_xlabel("Год")
        ax.set_ylabel("Площадь, га")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Динамика площадей по группам пригодности", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_slices(stack, slice_labels, out_png, model):
    n = stack.shape[0]
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, t in zip(axes, range(n)):
        im = ax.imshow(stack[t], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"{model}: {slice_labels[t]}", fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, shrink=0.7, label="пригодность")
    for ax in axes[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================================
# ГРУППОВАЯ ЛОГИКА ПОРОГОВ (v3): N порогов -> N+1 групп
# ============================================================================

def _make_group_names(thr_list):
    """Человекочитаемые названия N+1 группы по списку порогов thr_list.

    Пример: thr_list=[0.5, 0.75] ->
      ['<0.5', '0.5–0.75', '≥0.75']
    Группа i содержит значения [thr[i-1], thr[i]) (полуинтервалы),
    последняя группа: [thr[-1], +inf).
    """
    thr = sorted(float(t) for t in thr_list)
    if not thr:
        return ["непригодно", "пригодно"]
    names = [f"<{thr[0]:g}"]
    for i in range(1, len(thr)):
        names.append(f"{thr[i-1]:g}–{thr[i]:g}")
    names.append(f"≥{thr[-1]:g}")
    return names


def _classify_groups(arr, thr_list, valid_mask):
    """Классифицировать массив пригодности на группы 0..N по порогам.

    Группа i = число порогов, которые значение >= (np.digitize, right=False
    с учётом полуинтервалов [thr[i-1], thr[i])).
    Возвращает uint8-массив: 0..N в valid_mask, 255 вне (nodata).
    """
    thr = np.asarray(sorted(float(t) for t in thr_list), dtype=np.float64)
    H, W = arr.shape
    grp = np.full((H, W), 255, dtype=np.uint8)
    vm = valid_mask & np.isfinite(arr)
    if thr.size == 0:
        grp[vm] = 0
        return grp
    # digitize: возвращает индекс интервала; значение>=thr[k] увеличивает класс
    codes = np.digitize(arr[vm], thr, right=False)  # 0..N
    grp[vm] = codes.astype(np.uint8)
    return grp


def _area_by_groups(stack, valid_mask, thr_list, area_ha):
    """Площади (га) каждой из N+1 групп по каждому срезу.

    Возвращает список (по срезам) списков площадей [g0, g1, ..., gN].
    """
    n_groups = len(thr_list) + 1
    out = []
    for t in range(stack.shape[0]):
        grp = _classify_groups(stack[t], thr_list, valid_mask)
        row = []
        for g in range(n_groups):
            row.append(float((grp == g).sum()) * area_ha)
        out.append(row)
    return out


def _change_status_multi(first, last, thr_list, valid_mask):
    """Карта переходов групп между первым и последним срезом.

    Код = grp_first * (N+1) + grp_last  (0 .. (N+1)^2-1), 255 = nodata.
    Диагональ (grp_first == grp_last) = стабильно в группе.
    Возвращает (status_uint, n_groups).
    """
    n_groups = len(thr_list) + 1
    gf = _classify_groups(first, thr_list, valid_mask)
    gl = _classify_groups(last, thr_list, valid_mask)
    H, W = first.shape
    status = np.full((H, W), 255, dtype=np.uint8)
    vm = (gf != 255) & (gl != 255)
    code = gf.astype(np.int32) * n_groups + gl.astype(np.int32)
    status[vm] = code[vm].astype(np.uint8)
    return status, n_groups


def _plot_groups(grp, group_names, out_png, title):
    """Карта классов пригодности (групп) для одного среза."""
    n = len(group_names)
    # палитра от серого к зелёному
    base_colors = [
        "#d9d9d9", "#fee08b", "#a6d96a", "#1a9850",
        "#006837", "#00441b", "#08306b", "#2171b5",
    ]
    colors = [base_colors[i % len(base_colors)] for i in range(n)]
    cmap = ListedColormap(colors)
    bounds = [i - 0.5 for i in range(n + 1)]
    norm = BoundaryNorm(bounds, cmap.N)
    disp = np.where(grp == 255, np.nan, grp).astype(float)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(disp, cmap=cmap, norm=norm)
    ax.set_title(title, fontweight="bold")
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, ticks=list(range(n)))
    cbar.set_ticklabels(group_names)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_change_multi(status, n_groups, group_names, out_png, title):
    """Карта переходов групп: упрощённо красим по знаку изменения группы.

    Зелёный = переход вверх (улучшение), красный = вниз (ухудшение),
    серый = стабильно. Подробная матрица переходов — в CSV.
    """
    H, W = status.shape
    direction = np.full((H, W), 255, dtype=np.uint8)
    vm = status != 255
    gf = (status[vm].astype(np.int32)) // n_groups
    gl = (status[vm].astype(np.int32)) % n_groups
    dcode = np.full(gf.shape, 0, dtype=np.uint8)  # 0=стабильно
    dcode[gl > gf] = 1   # вверх
    dcode[gl < gf] = 2   # вниз
    direction[vm] = dcode
    cmap = ListedColormap(["#d9d9d9", "#1a9850", "#d73027"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    disp = np.where(direction == 255, np.nan, direction).astype(float)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(disp, cmap=cmap, norm=norm)
    ax.set_title(title, fontweight="bold")
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, ticks=[0, 1, 2])
    cbar.set_ticklabels([
        "стабильно (та же группа)",
        "переход вверх (улучшение)",
        "переход вниз (ухудшение)",
    ])
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n============================================================")
    print(" SDM — ДИНАМИКА МОДЕЛЕЙ ПРИГОДНОСТИ МЕСТООБИТАНИЙ")
    print("============================================================\n")

    root_dir = os.getcwd()
    plan = _select_time_slices(root_dir)
    suffix = plan.get("suffix", "_v001")

    by_model, slice_meta = _collect_suitability(plan, root_dir)
    if not by_model:
        raise FileNotFoundError("Не найдено ни одной карты suitability_*.tif")

    # Папка вывода
    base_out = os.path.join(root_dir, "06_DYNAMIC_SDM")
    os.makedirs(base_out, exist_ok=True)
    run_stamp = datetime.now().strftime("%d%m%y_%H%M")
    OUTPUT_DIR = os.path.join(base_out, f"{run_stamp}{suffix}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n✓ Моделей: {', '.join(by_model.keys())}")
    print(f"✓ Выход : {OUTPUT_DIR}\n")

    # Порог бинаризации / группы
    print("Пороги пригодности для расчёта площадей и change map:")
    print("  [1] Фиксированный один порог (например 0.5) -> 2 группы")
    print("  [2] Из thresholds.pkl (maxSSS / p10) выбранного запуска -> 2 группы")
    print("  [3] Пользовательские: задать КОЛИЧЕСТВО порогов и их ЗНАЧЕНИЯ")
    print("      (N порогов -> N+1 группа, напр. 0.5 и 0.75 -> 3 группы)")
    thr_raw = input("Ваш выбор (Enter = 1): ").strip() or "1"

    # Единый внутренний механизм: всегда работаем со списком порогов thr_list.
    # Варианты [1]/[2] -> один порог (2 группы, обратная совместимость).
    thr_list = [0.5]
    threshold_name = "0.5"
    if thr_raw == "2":
        threshold = 0.5
        # пробуем найти thresholds.pkl рядом в 03_TUNE_SDM
        tdir = os.path.join(root_dir, "03_TUNE_SDM")
        tps = glob.glob(os.path.join(tdir, "*", "thresholds.pkl"))
        if tps:
            with open(sorted(tps)[-1], "rb") as f:
                thr_data = pickle.load(f)
            which = input("  Порог maxSSS или p10? (Enter = maxSSS): ").strip() or "maxSSS"
            tkey = "max_sss" if which.lower().startswith("max") else "p10"
            # усредняем по моделям
            vals = [v.get(tkey) for v in thr_data.values()
                    if isinstance(v, dict) and v.get(tkey) is not None]
            if vals:
                threshold = float(np.mean(vals))
                threshold_name = f"{which} (~{threshold:.3f})"
                print(f"  ✓ Порог {which} ≈ {threshold:.3f}")
            else:
                print("  ! Порог не найден, использую 0.5")
        else:
            print("  ! thresholds.pkl не найден, использую 0.5")
        thr_list = [threshold]
    elif thr_raw == "3":
        # Пользовательские пороги: сначала количество, затем значения.
        n_thr = None
        while n_thr is None:
            ans = input("  Сколько порогов (целое >= 1, напр. 2): ").strip()
            try:
                n_thr = int(ans)
                if n_thr < 1:
                    print("    ! Нужно целое число >= 1.")
                    n_thr = None
            except (ValueError, TypeError):
                print("    ! Нужно целое число, например 2.")
                n_thr = None
        vals = []
        for k in range(n_thr):
            v = None
            while v is None:
                ans = input(f"  Значение порога #{k + 1} (0..1, напр. 0.5): ").strip()
                try:
                    fv = float(ans.replace(",", "."))
                    if not (0.0 <= fv <= 1.0):
                        print("    ! Значение должно быть в диапазоне [0, 1].")
                        continue
                    v = fv
                except (ValueError, TypeError):
                    print("    ! Нужно число, например 0.75.")
            vals.append(v)
        # сортируем по возрастанию и убираем дубли
        thr_list = sorted(set(vals))
        if len(thr_list) != len(vals):
            print(f"  ⚠ Убраны дублирующиеся пороги; осталось {len(thr_list)}.")
        threshold_name = ", ".join(f"{t:g}" for t in thr_list)
        print(f"  ✓ Пороги: {threshold_name}  ->  {len(thr_list) + 1} групп")
    else:
        v = input("  Значение порога (Enter = 0.5): ").strip()
        if v:
            thr_list = [float(v.replace(",", "."))]
            threshold_name = f"{thr_list[0]:g}"

    # Названия групп и признак однопорогового (backward-compat) режима
    group_names = _make_group_names(thr_list)
    n_groups = len(thr_list) + 1
    single_thr = (len(thr_list) == 1)
    # Для однопорогового случая — старый gain/loss; пригодность >= порога
    threshold = thr_list[0] if single_thr else None
    print(f"✓ Группы ({n_groups}): {', '.join(group_names)}\n")

    # CSV-сводка — заголовок с погрупповыми площадями
    summary_rows = []
    grp_cols = ";".join(f"area_grp{g}_ha[{group_names[g]}]" for g in range(n_groups))
    summary_rows.append("model;year;mean_suit;" + grp_cols + ";area_suitable_ha;area_suitable_km2")

    # Динамика площадей для общего графика (по моделям)
    area_dynamics = {}
    common_years = None

    for model, entries in by_model.items():
        years = [y for (y, _) in entries]
        paths = [p for (_, p) in entries]
        labels = [str(y) if y is not None else os.path.basename(os.path.dirname(p))
                  for (y, p) in entries]
        if len(paths) < 2:
            print(f"--- Модель {model}: < 2 срезов, пропуск динамики ---\n")
            continue
        print(f"--- Модель {model}: {len(paths)} срезов ({labels}) ---")

        stack, profile, valid_mask = _read_aligned(paths, paths[0])
        area_ha = _pixel_area_ha(profile)

        # --- погрупповые площади по срезам (N+1 групп) ---
        group_areas = _area_by_groups(stack, valid_mask, thr_list, area_ha)
        # «Площадь пригодных» = сумма групп выше ПЕРВОГО порога
        # (для однопорогового — это обычная площадь >= порога).
        areas_suitable = [sum(row[1:]) for row in group_areas]
        for t in range(stack.shape[0]):
            a = stack[t]
            mean_suit = float(np.nanmean(np.where(valid_mask, a, np.nan)))
            yr = years[t] if years[t] is not None else labels[t]
            grp_str = ";".join(f"{group_areas[t][g]:.1f}" for g in range(n_groups))
            summary_rows.append(
                f"{model};{yr};{mean_suit:.4f};{grp_str};"
                f"{areas_suitable[t]:.1f};{areas_suitable[t]/100.0:.3f}"
            )
        # для общего графика — только если годы известны
        if all(y is not None for y in years):
            # сохраняем погрупповые площади (список по срезам -> по группам)
            area_dynamics[model] = group_areas
            common_years = years

        # --- карты классов пригодности (группы) по срезам ---
        # Сохраняем карту групп для первого и последнего среза.
        for idx, tag in ((0, labels[0]), (-1, labels[-1])):
            grp_map = _classify_groups(stack[idx], thr_list, valid_mask)
            grp_path = os.path.join(OUTPUT_DIR, f"groups_{model}_{tag}.tif")
            _save_byte(grp_map, profile, grp_path)
            _plot_groups(
                grp_map, group_names,
                os.path.join(OUTPUT_DIR, f"groups_{model}_{tag}.png"),
                f"Классы пригодности {model}: {tag}",
            )
        print(f"  ✓ groups_{model}_{labels[0]}.tif/.png и groups_{model}_{labels[-1]}.tif/.png")

        # --- карта разности (последний минус первый) ---
        diff = stack[-1] - stack[0]
        diff_path = os.path.join(OUTPUT_DIR, f"diff_{model}_{labels[0]}_{labels[-1]}.tif")
        _save_float(diff, profile, diff_path)
        print(f"  ✓ {os.path.basename(diff_path)}")

        # --- попиксельный тренд (если >= 3 срезов и годы известны) ---
        if len(paths) >= 3 and all(y is not None for y in years):
            slope, r2 = _pixel_trend(stack, years, valid_mask)
            slope_path = os.path.join(OUTPUT_DIR, f"trend_slope_{model}.tif")
            r2_path = os.path.join(OUTPUT_DIR, f"trend_r2_{model}.tif")
            _save_float(slope, profile, slope_path)
            _save_float(r2, profile, r2_path)
            _plot_trend(
                slope, os.path.join(OUTPUT_DIR, f"trend_{model}.png"),
                f"Тренд пригодности {model} ({years[0]}–{years[-1]})",
            )
            print(f"  ✓ trend_slope_{model}.tif / trend_r2_{model}.tif / trend_{model}.png")

        # --- change map ---
        if single_thr:
            # Один порог: старая логика gain/loss/стабильно (backward-compat).
            status = _change_status(stack[0], stack[-1], threshold, valid_mask)
            status_path = os.path.join(OUTPUT_DIR, f"change_status_{model}.tif")
            _save_byte(status, profile, status_path)
            _plot_change_status(
                status, os.path.join(OUTPUT_DIR, f"change_status_{model}.png"),
                f"Изменения пригодности {model}: {labels[0]} → {labels[-1]} "
                f"(порог {threshold_name})",
            )
            gain_ha = float((status == 1).sum()) * area_ha
            loss_ha = float((status == 2).sum()) * area_ha
            stable_ha = float((status == 3).sum()) * area_ha
            print(f"  ✓ change_status_{model}.tif / .png")
            print(f"    Появление: {gain_ha:.0f} га | Исчезновение: {loss_ha:.0f} га "
                  f"| Стабильно: {stable_ha:.0f} га")
        else:
            # Несколько порогов: карта переходов групп (код gf*(N+1)+gl).
            status, ng = _change_status_multi(stack[0], stack[-1], thr_list, valid_mask)
            status_path = os.path.join(OUTPUT_DIR, f"change_status_{model}.tif")
            _save_byte(status, profile, status_path)
            _plot_change_multi(
                status, ng, group_names,
                os.path.join(OUTPUT_DIR, f"change_status_{model}.png"),
                f"Переходы групп пригодности {model}: {labels[0]} → {labels[-1]}",
            )
            # сводка по направлениям и полная матрица переходов в CSV
            vm = status != 255
            gf_all = (status[vm].astype(np.int32)) // ng
            gl_all = (status[vm].astype(np.int32)) % ng
            up_ha = float((gl_all > gf_all).sum()) * area_ha
            down_ha = float((gl_all < gf_all).sum()) * area_ha
            stab_ha = float((gl_all == gf_all).sum()) * area_ha
            print(f"  ✓ change_status_{model}.tif / .png  (переходы групп)")
            print(f"    Вверх: {up_ha:.0f} га | Вниз: {down_ha:.0f} га "
                  f"| Стабильно: {stab_ha:.0f} га")
            # полная матрица переходов (из группы -> в группу) — отдельный CSV
            tm_path = os.path.join(OUTPUT_DIR, f"transition_matrix_{model}.csv")
            with open(tm_path, "w", encoding="utf-8") as f:
                f.write("from_group\\to_group;" + ";".join(group_names) + "\n")
                for gf in range(ng):
                    cells = []
                    for gl in range(ng):
                        code = gf * ng + gl
                        ha = float((status == code).sum()) * area_ha
                        cells.append(f"{ha:.1f}")
                    f.write(group_names[gf] + ";" + ";".join(cells) + "\n")
            print(f"  ✓ transition_matrix_{model}.csv")

        # --- панель срезов ---
        _plot_slices(stack, labels, os.path.join(OUTPUT_DIR, f"slices_{model}.png"), model)
        print(f"  ✓ slices_{model}.png\n")

    # --- общие графики динамики площадей ---
    if area_dynamics and common_years:
        # 1) «Пригодная» площадь (выше первого порога) по моделям
        suitable_by_model = {
            m: [sum(row[1:]) for row in rows]
            for m, rows in area_dynamics.items()
        }
        _plot_area_dynamics(
            common_years, suitable_by_model, threshold_name,
            os.path.join(OUTPUT_DIR, "area_dynamics.png"),
        )
        print("✓ area_dynamics.png (динамика пригодной площади по моделям)")
        # 2) Погрупповая динамика (всегда, особенно полезна при N+1 > 2)
        _plot_group_dynamics(
            common_years, area_dynamics, group_names,
            os.path.join(OUTPUT_DIR, "group_area_dynamics.png"),
        )
        print("✓ group_area_dynamics.png (динамика площадей по группам)")

    # --- CSV-сводка ---
    csv_path = os.path.join(OUTPUT_DIR, "dynamic_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_rows) + "\n")
    print(f"✓ dynamic_summary.csv")

    print("\n============================================================")
    print("  АНАЛИЗ ДИНАМИКИ ЗАВЕРШЁН")
    print("============================================================")
    print(f"Папка: {OUTPUT_DIR}")
    print(f"Группы пригодности ({n_groups}): {', '.join(group_names)}")
    print("  - diff_<model>_<год0>_<год1>.tif      (разность suitability)")
    print("  - trend_slope_<model>.tif / trend_r2_<model>.tif (тренд)")
    print("  - groups_<model>_<год>.tif / .png    (классы/группы пригодности)")
    if single_thr:
        print("  - change_status_<model>.tif / .png    (gain/loss/стабильно)")
    else:
        print("  - change_status_<model>.tif / .png    (переходы групп: вверх/вниз/стаб.)")
        print("  - transition_matrix_<model>.csv       (матрица переходов групп)")
    print("  - slices_<model>.png                  (срезы по годам)")
    print("  - area_dynamics.png                   (динамика пригодной площади)")
    print("  - group_area_dynamics.png             (динамика площадей по группам)")
    print("  - dynamic_summary.csv                 (сводка с погрупповыми площадями)")
    print("============================================================\n")

    return {"output_dir": OUTPUT_DIR}


if __name__ == "__main__":
    main()
