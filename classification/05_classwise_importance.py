# =============================================================================
# 05_classwise_importance_v7.py
#
# СКРИПТ 5: CLASS-WISE PERMUTATION IMPORTANCE + SCATTERPLOTS (ТОП-ПАРЫ)
#
# Читает данные из папки 04_FINAL/<stamp>/:
#   final_model.pkl, samples_df.pkl, class_table.pkl, run_meta.pkl
#
# Выходы (в 05_CLASSWISE/<stamp>/):
#   1. *_cwimp_overall.png   — общая permutation importance (macro-F1 drop)
#   2. *_cwimp_<CLASS>.png   — importance по каждому классу (class-F1 drop)
#   3. *_cwimp_heatmap.png   — тепловая карта importance (предиктор x класс)
#   4. *_scatter_<CLS>_<FEAT>.png — scatter: lm/poly2 (AIC) + CI (топ-пары)
#   5. *_gam_<CLS>_<FEAT>.png     — scatter: GAM + CI (топ-пары)
#   6. *_cwimp_table.csv     — таблица importance
#
# v7: scatter/GAM строятся только для топ-пар (предиктор, класс)
#     с наибольшей permutation importance (≥ 10 пар, мин. 1 на класс).
#     Каждый график — отдельный PNG. GAM через pygam.
# =============================================================================

import os
import sys
import re
import time
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from sklearn.metrics import f1_score, confusion_matrix

# pygam — для GAM-сплайнов (аналог mgcv::gam в R)
try:
    from pygam import LinearGAM, s
    PYGAM_AVAILABLE = True
except ImportError:
    PYGAM_AVAILABLE = False

# statsmodels — для AIC сравнения lm vs poly2
try:
    import statsmodels.api as sm_api
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

print("=" * 60)
print("  CLASS-WISE PERMUTATION IMPORTANCE  (v7)")
print("=" * 60)
print()

# --- Рабочая директория ---
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
root_dir = os.getcwd()
print(f"Рабочая директория: {root_dir}\n")


# ============================================================================
# Вспомогательные функции
# ============================================================================

def class_f1(y_true, y_pred, cls):
    """F1 для одного класса (бинарная: cls vs rest)."""
    tp = np.sum((y_pred == cls) & (y_true == cls))
    fp = np.sum((y_pred == cls) & (y_true != cls))
    fn = np.sum((y_pred != cls) & (y_true == cls))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def macro_f1(y_true, y_pred, classes):
    """Macro-F1: среднее class-wise F1."""
    f1s = [class_f1(y_true, y_pred, cls) for cls in classes]
    return np.nanmean(f1s)


def parse_stamp(stamp):
    """Парсит метку времени DDMMYY_HHMM из имени папки."""
    parts = stamp.split("_")
    if len(parts) < 2:
        return None
    try:
        return datetime.strptime(parts[0] + parts[1], "%d%m%y%H%M")
    except ValueError:
        return None


def safe_filename(name):
    """Заменяет небезопасные символы в имени файла на '_'."""
    return re.sub(r"[^\w\-]", "_", name)


def choose_smooth(x, y):
    """
    Выбирает лучший параметрический фит: lm vs poly2 по AIC.
    Возвращает ('linear', degree=1) или ('quad', degree=2).
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 5:
        return "linear", 1

    if not STATSMODELS_AVAILABLE:
        # Без statsmodels — используем numpy polyfit + RSS → приближённый AIC
        try:
            # Linear
            c1 = np.polyfit(x_clean, y_clean, 1)
            res1 = y_clean - np.polyval(c1, x_clean)
            rss1 = np.sum(res1 ** 2)
            n = len(y_clean)
            k1 = 2  # slope + intercept
            aic1 = n * np.log(rss1 / n + 1e-30) + 2 * k1

            # Quadratic
            c2 = np.polyfit(x_clean, y_clean, 2)
            res2 = y_clean - np.polyval(c2, x_clean)
            rss2 = np.sum(res2 ** 2)
            k2 = 3
            aic2 = n * np.log(rss2 / n + 1e-30) + 2 * k2

            if aic2 + 2 < aic1:
                return "quad", 2
            else:
                return "linear", 1
        except Exception:
            return "linear", 1
    else:
        try:
            X1 = sm_api.add_constant(x_clean)
            m1 = sm_api.OLS(y_clean, X1).fit()
            aic1 = m1.aic
        except Exception:
            aic1 = np.inf

        try:
            X2 = np.column_stack([np.ones(len(x_clean)), x_clean, x_clean ** 2])
            m2 = sm_api.OLS(y_clean, X2).fit()
            aic2 = m2.aic
        except Exception:
            aic2 = np.inf

        if not np.isfinite(aic1) and not np.isfinite(aic2):
            return "linear", 1
        if aic2 + 2 < aic1:
            return "quad", 2
        return "linear", 1


def predict_model(model, X):
    """Предсказание классов для произвольной модели (sklearn-совместимой или catboost)."""
    try:
        return model.predict(X)
    except Exception:
        # catboost.CatBoostClassifier.predict может требовать Pool
        try:
            from catboost import Pool
            pool = Pool(X)
            return model.predict(pool).flatten()
        except Exception:
            return model.predict(X.values)


def predict_proba_model(model, X):
    """Вероятности классов."""
    try:
        return model.predict_proba(X)
    except Exception:
        try:
            from catboost import Pool
            pool = Pool(X)
            return model.predict_proba(pool)
        except Exception:
            return None


# ============================================================================
# ШАГ 1: ВЫБОР ПАПКИ 04_FINAL
# ============================================================================
print("\n" + "=" * 60)
print("  ВЫБОР ДАННЫХ ИЗ 04_FINAL")
print("=" * 60 + "\n")

final_base = os.path.join(root_dir, "04_FINAL")
if not os.path.isdir(final_base):
    sys.exit("Папка 04_FINAL не найдена! Сначала запустите 04_predict_map.py")

# Подпапки формата DDMMYY_HHMM_vNNN
stamp_pattern = re.compile(r"^\d{6}_\d{4}_v\d{3}$")
final_subfolders = [
    d for d in os.listdir(final_base)
    if os.path.isdir(os.path.join(final_base, d)) and stamp_pattern.match(d)
]

if not final_subfolders:
    sys.exit("В папке 04_FINAL нет подпапок с результатами!")

# Сортировка по дате
final_times = [(d, parse_stamp(d)) for d in final_subfolders]
final_times.sort(key=lambda x: x[1] if x[1] is not None else datetime.min, reverse=True)
final_subfolders = [d for d, _ in final_times]

print("Доступные результаты 04:\n")
for i, folder_name in enumerate(final_subfolders, 1):
    folder_path = os.path.join(final_base, folder_name)
    summary_txt = ""
    meta_path = os.path.join(folder_path, "run_meta.pkl")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            summary_txt = (
                f" | {meta.get('model_name', '?')} "
                f"(OA={meta.get('accuracy', 0):.3f}, "
                f"K={meta.get('kappa', 0):.3f}, "
                f"{len(meta.get('predictors', []))} пред.)"
            )
        except Exception:
            pass

    marker = " <-- последняя" if i == 1 else ""
    ts = parse_stamp(folder_name)
    ts_fmt = ts.strftime("%d.%m.%Y %H:%M") if ts else folder_name
    print(f"  [{i}] {folder_name} ({ts_fmt}){summary_txt}{marker}")

folder_choice = input(f"\nВыберите номер папки (Enter = 1, последняя): ").strip()
if folder_choice == "":
    folder_choice = "1"
try:
    folder_idx = int(folder_choice)
except ValueError:
    folder_idx = 1

if folder_idx < 1 or folder_idx > len(final_subfolders):
    folder_idx = 1
    print("! Некорректный ввод, выбрана последняя.")

selected_dir = os.path.join(final_base, final_subfolders[folder_idx - 1])
print(f"\nV Выбрана папка: {selected_dir}\n")

# Наследуем суффикс версии из имени папки 04_FINAL
final_basename = os.path.basename(selected_dir)
version_suffix = ""
m = re.match(r"^(.*)(_v\d{3})$", final_basename)
if m:
    version_suffix = m.group(2)  # _v001


# ============================================================================
# ШАГ 2: ЗАГРУЗКА ДАННЫХ
# ============================================================================
print("[ШАГ 2] Загрузка данных...")

model_path = os.path.join(selected_dir, "final_model.pkl")
samples_path = os.path.join(selected_dir, "samples_df.pkl")
meta_path = os.path.join(selected_dir, "run_meta.pkl")

if not os.path.isfile(model_path):
    sys.exit(f"final_model.pkl не найден в {selected_dir}")
if not os.path.isfile(samples_path):
    sys.exit(f"samples_df.pkl не найден в {selected_dir}")

with open(model_path, "rb") as f:
    final_model = pickle.load(f)

with open(samples_path, "rb") as f:
    samples_df = pickle.load(f)

model_name = "model"
base_name = "model"
predictors = None

if os.path.isfile(meta_path):
    with open(meta_path, "rb") as f:
        run_meta = pickle.load(f)
    model_name = run_meta.get("model_name", "model")
    base_name = run_meta.get("base_name", "model")
    predictors = run_meta.get("predictors", None)
    print(
        f"  V Модель: {model_name} | "
        f"OA={run_meta.get('accuracy', 0):.3f} | "
        f"Kappa={run_meta.get('kappa', 0):.3f}"
    )
else:
    # Попробуем извлечь имя модели
    if hasattr(final_model, "named_steps"):
        # Pipeline — имя последнего шага
        last_step_name = list(final_model.named_steps.keys())[-1]
        model_name = type(final_model.named_steps[last_step_name]).__name__
    elif hasattr(final_model, "__class__"):
        model_name = type(final_model).__name__

# Определяем целевой столбец
if isinstance(samples_df, pd.DataFrame):
    if "class" in samples_df.columns:
        y_col = "class"
    elif ".outcome" in samples_df.columns:
        y_col = ".outcome"
    else:
        sys.exit("Не найден столбец 'class' или '.outcome' в samples_df")
else:
    sys.exit("samples_df не является DataFrame")

if predictors is None:
    exclude = {y_col, ".weights", ".modelWeights"}
    exclude.update(c for c in samples_df.columns if c.startswith("."))
    predictors = [c for c in samples_df.columns if c not in exclude]

X_data = samples_df[predictors].copy()
y_data = samples_df[y_col].astype(str).values
class_levels = sorted(np.unique(y_data))
n_classes = len(class_levels)
n_feats = len(predictors)

print(f"  V Наблюдений: {len(X_data)} | Предикторов: {n_feats} | Классов: {n_classes}")
print(f"    Классы: {', '.join(class_levels)}")


# ============================================================================
# ШАГ 3: НАСТРОЙКА ВЫХОДА
# ============================================================================
print("\n[ШАГ 3] Папка результатов...")

output_base_05 = os.path.join(root_dir, "05_CLASSWISE")
os.makedirs(output_base_05, exist_ok=True)

stamp_05 = datetime.now().strftime("%d%m%y_%H%M")
if version_suffix:
    run_folder_05 = stamp_05 + version_suffix
else:
    run_folder_05 = stamp_05

output_dir = os.path.join(output_base_05, run_folder_05)
os.makedirs(output_dir, exist_ok=True)
print(f"  V {output_dir}")


# ============================================================================
# ШАГ 4: НАСТРОЙКА PERMUTATION IMPORTANCE
# ============================================================================
print("\n[ШАГ 4] Настройка permutation importance...")
print("  Число повторов перестановки каждого предиктора:")
print("  [1] 10   -- быстро (рекомендуется)")
print("  [2] 25   -- средне")
print("  [3] 50   -- точнее, но дольше")
print("  [4] Ввести число")

n_perm_choice = input("  Выбор (Enter = 10): ").strip()

n_perm = 10
if n_perm_choice == "2":
    n_perm = 25
elif n_perm_choice == "3":
    n_perm = 50
elif n_perm_choice == "4" or re.match(r"^\d+$", n_perm_choice or ""):
    try:
        val = int(n_perm_choice)
        if 1 <= val <= 200:
            n_perm = val
    except ValueError:
        pass

print(f"  V Повторов: {n_perm}")


# ============================================================================
# ШАГ 5: ВЫЧИСЛЕНИЕ CLASS-WISE PERMUTATION IMPORTANCE
# ============================================================================
print(f"\n[ШАГ 5] Вычисление class-wise permutation importance...")
print(f"  {n_feats} предикторов x {n_classes} классов x {n_perm} повторов = "
      f"{n_feats * n_perm} перестановок")

# Базовое предсказание
X_arr = X_data.values if hasattr(X_data, "values") else X_data
base_pred = predict_model(final_model, X_data)
if hasattr(base_pred, "values"):
    base_pred = base_pred.values
base_pred = np.asarray(base_pred).flatten().astype(str)

base_f1_overall = macro_f1(y_data, base_pred, class_levels)
base_f1_class = {cls: class_f1(y_data, base_pred, cls) for cls in class_levels}

print(f"  Базовый macro-F1: {base_f1_overall:.4f}")
for cls in class_levels:
    print(f"    F1({cls}) = {base_f1_class[cls]:.4f}")

# Матрица importance
imp_matrix = np.zeros((n_feats, n_classes))
imp_overall = np.zeros(n_feats)

print("\n  Перестановки:")
t_start = time.time()

for f_idx, feat in enumerate(predictors):
    f1_drops_overall = np.zeros(n_perm)
    f1_drops_class = np.zeros((n_perm, n_classes))

    for p in range(n_perm):
        X_perm = X_data.copy()
        X_perm[feat] = np.random.permutation(X_perm[feat].values)
        perm_pred = predict_model(final_model, X_perm)
        if hasattr(perm_pred, "values"):
            perm_pred = perm_pred.values
        perm_pred = np.asarray(perm_pred).flatten().astype(str)

        f1_drops_overall[p] = base_f1_overall - macro_f1(y_data, perm_pred, class_levels)
        for c_idx, cls in enumerate(class_levels):
            f1_drops_class[p, c_idx] = (
                base_f1_class[cls] - class_f1(y_data, perm_pred, cls)
            )

    imp_overall[f_idx] = np.nanmean(f1_drops_overall)
    for c_idx in range(n_classes):
        imp_matrix[f_idx, c_idx] = np.nanmean(f1_drops_class[:, c_idx])

    # Прогресс-бар
    pct = (f_idx + 1) / n_feats
    bar_width = 40
    done = int(round(pct * bar_width))
    bar = "[" + "=" * done + " " * (bar_width - done) + "]"
    elapsed = time.time() - t_start
    eta = elapsed / pct * (1 - pct) if pct > 0 else 0
    print(f"\r  {bar} {pct * 100:3.0f}%  {feat}  ETA: {eta:.0f}s   ", end="", flush=True)

print("\n")
elapsed_total = (time.time() - t_start) / 60
print(f"  V Готово за {elapsed_total:.1f} мин")


# ============================================================================
# ШАГ 6: ТАБЛИЦА IMPORTANCE (CSV)
# ============================================================================
print("\n[ШАГ 6] Сохранение таблицы...")

imp_df = pd.DataFrame({
    "Predictor": predictors,
    "Overall": np.round(imp_overall, 6),
})
for c_idx, cls in enumerate(class_levels):
    imp_df[cls] = np.round(imp_matrix[:, c_idx], 6)

imp_df = imp_df.sort_values("Overall", ascending=False).reset_index(drop=True)

file_csv = os.path.join(output_dir, f"{base_name}_cwimp_table.csv")
imp_df.to_csv(file_csv, index=False)
print(f"  V {os.path.basename(file_csv)}")

print(f"\n  Permutation Importance (macro-F1 drop):")
print("  " + "-" * 60)
for _, row in imp_df.iterrows():
    parts = f"  {row['Predictor']:<25s}  Overall: {row['Overall']:+.4f}"
    for cls in class_levels:
        parts += f"  | {cls}: {row[cls]:+.4f}"
    print(parts)
print("  " + "-" * 60)


# ============================================================================
# ШАГ 7: BARPLOTS — OVERALL + ПО КЛАССАМ
# ============================================================================
print("\n[ШАГ 7] Barplots importance...")


def plot_importance_bar(imp_values, feat_names, title, xlab, color, filepath):
    """Горизонтальный barplot importance → PNG."""
    ord_idx = np.argsort(imp_values)  # ascending → самый важный внизу
    sorted_vals = imp_values[ord_idx]
    sorted_names = [feat_names[i] for i in ord_idx]
    colors = [color if v >= 0 else "gray" for v in sorted_vals]

    fig_h = max(5, len(feat_names) * 0.35 + 1.5)
    fig, ax = plt.subplots(figsize=(7, fig_h))
    ax.barh(range(len(sorted_vals)), sorted_vals, color=colors, edgecolor="none")
    ax.set_yticks(range(len(sorted_vals)))
    ax.set_yticklabels(sorted_names, fontsize=max(6, 10 - len(feat_names) * 0.15))
    ax.axvline(0, linestyle="--", color="red", linewidth=0.8)
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Overall
file_overall = os.path.join(output_dir, f"{base_name}_cwimp_overall.png")
plot_importance_bar(
    imp_values=imp_overall[imp_df.index.values] if False else np.array([
        imp_overall[predictors.index(p)] for p in imp_df["Predictor"]
    ]),
    feat_names=imp_df["Predictor"].tolist(),
    title=f"Permutation Importance (macro-F1 drop)\n{model_name}",
    xlab="macro-F1 drop (higher = more important)",
    color="steelblue",
    filepath=file_overall,
)
print(f"  V {os.path.basename(file_overall)}")

# По классам
class_colors = {}
cmap = plt.cm.get_cmap("hsv", n_classes + 1)
for c_idx, cls in enumerate(class_levels):
    rgba = cmap(c_idx / n_classes)
    # Desaturate: s=0.7, v=0.85
    import colorsys
    r, g, b = rgba[0], rgba[1], rgba[2]
    h, s_val, v_val = colorsys.rgb_to_hsv(r, g, b)
    r2, g2, b2 = colorsys.hsv_to_rgb(h, min(s_val, 0.7), min(v_val, 0.85))
    class_colors[cls] = (r2, g2, b2)

for c_idx, cls in enumerate(class_levels):
    try:
        cls_imp = imp_matrix[:, c_idx]
        cls_ord = np.argsort(cls_imp)[::-1]
        ordered_names = [predictors[i] for i in cls_ord]

        cls_safe = safe_filename(cls)
        file_cls = os.path.join(output_dir, f"{base_name}_cwimp_{cls_safe}.png")

        plot_importance_bar(
            imp_values=cls_imp,
            feat_names=predictors,
            title=(
                f"Permutation Importance: {cls}\n"
                f"{model_name} | base F1({cls})={base_f1_class[cls]:.3f}"
            ),
            xlab=f"F1({cls}) drop",
            color=class_colors[cls],
            filepath=file_cls,
        )
        print(f"  V {os.path.basename(file_cls)}")
    except Exception as e:
        print(f"  ! Класс {cls}: {e}")


# ============================================================================
# ШАГ 8: HEATMAP IMPORTANCE (предиктор x класс)
# ============================================================================
print("\n[ШАГ 8] Heatmap importance...")

try:
    # Порядок предикторов: по Overall importance (сверху вниз)
    ordered_predictors = imp_df["Predictor"].tolist()

    # Формируем матрицу для heatmap
    hm_data = np.zeros((len(ordered_predictors), n_classes))
    for i, pred in enumerate(ordered_predictors):
        pred_idx = predictors.index(pred)
        for j in range(n_classes):
            hm_data[i, j] = imp_matrix[pred_idx, j]

    # Размеры
    cell_w = max(1.2, 2.5 - n_classes * 0.05)
    cell_h = max(0.3, 0.55 - len(ordered_predictors) * 0.005)
    fig_w = max(8, n_classes * cell_w + 4)
    fig_h = max(6, len(ordered_predictors) * cell_h + 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap_hm = mcolors.LinearSegmentedColormap.from_list(
        "importance",
        ["white", "#FFF7BC", "#FEC44F", "#D95F0E", "#7F0000"],
        N=100,
    )

    im = ax.imshow(hm_data, cmap=cmap_hm, aspect="auto", interpolation="nearest")

    # Подписи
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_levels, rotation=45, ha="right",
                       fontsize=max(6, 10 - n_classes * 0.3))
    ax.set_yticks(range(len(ordered_predictors)))
    ax.set_yticklabels(ordered_predictors,
                       fontsize=max(5.5, 9 - len(ordered_predictors) * 0.15))

    # Значения в ячейках
    q70 = np.nanquantile(hm_data, 0.7)
    font_size_cell = max(5, 8 - max(len(ordered_predictors), n_classes) * 0.12)
    for i in range(len(ordered_predictors)):
        for j in range(n_classes):
            val = hm_data[i, j]
            txt_col = "white" if val > q70 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=font_size_cell, color=txt_col)

    # Сетка
    for i in range(len(ordered_predictors) + 1):
        ax.axhline(i - 0.5, color="gray", linewidth=0.3, alpha=0.5)
    for j in range(n_classes + 1):
        ax.axvline(j - 0.5, color="gray", linewidth=0.3, alpha=0.5)

    ax.set_title(f"Class-wise Permutation Importance\n{model_name}",
                 fontsize=11, fontweight="bold", pad=12)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()

    file_heatmap = os.path.join(output_dir, f"{base_name}_cwimp_heatmap.png")
    fig.savefig(file_heatmap, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  V {os.path.basename(file_heatmap)}")

except Exception as e:
    print(f"  ! Heatmap: {e}")


# ============================================================================
# ШАГ 9: SCATTERPLOTS — предиктор vs P(класс)
#         Только для пар (предиктор, класс) с наибольшей permutation importance.
#         Минимум 1 пара на каждый класс, всего не менее 10.
#         9a: lm / poly2 (AIC) — красная линия, красный CI — отдельный PNG
#         9b: GAM (pygam)      — красная линия, красный CI — отдельный PNG
# ============================================================================
print("\n[ШАГ 9] Scatterplots: предиктор vs P(класс) (топ-пары по importance)...")

# --- Выбор сложности GAM (параметр n_splines) ---
print("\n  Сложность GAM-сплайна (параметр n_splines — число базисных функций):")
print("  Чем больше n_splines, тем гибче кривая (может уловить узкие пики),")
print("  но при малом числе наблюдений возможно переобучение.\n")
print("  [1] n_splines = 4   -- плавная кривая (широкий колокол, мало изгибов)")
print("  [2] n_splines = 6   -- умеренная гибкость (рекомендуется)")
print("  [3] n_splines = 10  -- гибкая кривая (узкие пики, сложные формы)")
print("  [4] n_splines = 20  -- очень гибкая (только при >1000 наблюдений)")
print("  [5] Ввести число (3..30)")

gam_k_choice = input("  Выбор (Enter = 6): ").strip()

gam_k = 6
if gam_k_choice == "1":
    gam_k = 4
elif gam_k_choice == "3":
    gam_k = 10
elif gam_k_choice == "4":
    gam_k = 20
elif gam_k_choice == "5" or re.match(r"^\d+$", gam_k_choice or ""):
    try:
        val = int(gam_k_choice)
        if 3 <= val <= 30:
            gam_k = val
    except ValueError:
        pass

print(f"  V GAM n_splines = {gam_k}")

# --- Отбор топ-пар ---
MIN_PAIRS = 10

all_pairs = []
for p_idx, pred_name in enumerate(predictors):
    for c_idx, cls in enumerate(class_levels):
        all_pairs.append({
            "Predictor": pred_name,
            "Class": cls,
            "imp": imp_matrix[p_idx, c_idx],
        })

all_pairs_df = pd.DataFrame(all_pairs)

# 1) Для каждого класса — предиктор с максимальной importance
must_have = (
    all_pairs_df
    .loc[all_pairs_df.groupby("Class")["imp"].idxmax()]
    .copy()
)

# 2) Дополняем до MIN_PAIRS
already = set(zip(must_have["Predictor"], must_have["Class"]))
rest = all_pairs_df[
    ~all_pairs_df.apply(lambda r: (r["Predictor"], r["Class"]) in already, axis=1)
].sort_values("imp", ascending=False)

n_extra = max(0, MIN_PAIRS - len(must_have))
if n_extra > 0 and len(rest) > 0:
    extra = rest.head(n_extra)
    selected_pairs = pd.concat([must_have, extra], ignore_index=True)
else:
    selected_pairs = must_have.copy()

selected_pairs = selected_pairs.sort_values("imp", ascending=False).reset_index(drop=True)

print(f"  Отобрано {len(selected_pairs)} пар (предиктор, класс) для scatter/GAM:")
for r_idx, row in selected_pairs.iterrows():
    print(f"    {r_idx + 1:2d}. {row['Predictor']:<25s}  класс {row['Class']:<5s}  "
          f"imp = {row['imp']:+.4f}")
print()

# --- Вычисление вероятностей ---
print("  Предсказание вероятностей на обучающих данных...")
prob_train = predict_proba_model(final_model, X_data)

if prob_train is not None:
    prob_train = np.asarray(prob_train)

    # Определяем порядок классов в prob_train
    if hasattr(final_model, "classes_"):
        class_names_prob = [str(c) for c in final_model.classes_]
    elif hasattr(final_model, "named_steps"):
        # Pipeline — ищем classes_ у последнего estimator
        last_step = list(final_model.named_steps.values())[-1]
        if hasattr(last_step, "classes_"):
            class_names_prob = [str(c) for c in last_step.classes_]
        else:
            class_names_prob = class_levels
    else:
        class_names_prob = class_levels

    print(f"  V {len(class_names_prob)} классов x {prob_train.shape[0]} наблюдений\n")

    # Словарь: класс → индекс столбца в prob_train
    cls_to_prob_col = {}
    for i, cn in enumerate(class_names_prob):
        cls_to_prob_col[cn] = i

    # ---- Цикл по отобранным парам ----
    for _, pair_row in selected_pairs.iterrows():
        feat = pair_row["Predictor"]
        cls = pair_row["Class"]
        cls_safe = safe_filename(cls)
        feat_safe = safe_filename(feat)

        # Проверяем наличие класса
        if cls not in cls_to_prob_col:
            print(f"  ! Класс '{cls}' не найден в prob — пропуск")
            continue

        x_vals = X_data[feat].values.astype(float)
        y_vals = prob_train[:, cls_to_prob_col[cls]]

        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        x_clean = x_vals[mask]
        y_clean = y_vals[mask]

        if len(x_clean) < 5:
            print(f"  ! Мало данных для {feat} / {cls} — пропуск")
            continue

        # ---- 9a: параметрический scatter (отдельный PNG) ----
        try:
            fit_label, degree = choose_smooth(x_clean, y_clean)
            coeffs = np.polyfit(x_clean, y_clean, degree)

            # Кривая и CI
            x_sorted = np.sort(x_clean)
            y_fit = np.polyval(coeffs, x_sorted)

            # Bootstrap-CI (упрощённый)
            residuals = y_clean - np.polyval(coeffs, x_clean)
            se = np.std(residuals)

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(x_clean, y_clean, alpha=0.15, s=3, color="black",
                       edgecolors="none", rasterized=True)

            # CI 95%
            ax.fill_between(x_sorted, y_fit - 1.96 * se, y_fit + 1.96 * se,
                            color="#CC0000", alpha=0.15, linewidth=0)
            # CI 80%
            ax.fill_between(x_sorted, y_fit - 1.28 * se, y_fit + 1.28 * se,
                            color="#CC0000", alpha=0.30, linewidth=0)
            # Линия
            ax.plot(x_sorted, y_fit, color="#CC0000", linewidth=1.2)

            ax.set_xlabel(feat, fontsize=9)
            ax.set_ylabel(f"P({cls})", fontsize=9)
            ax.set_title(f"{feat} [{fit_label}]", fontsize=10, fontweight="bold")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            file_scatter = os.path.join(
                output_dir,
                f"{base_name}_scatter_{cls_safe}_{feat_safe}.png"
            )
            fig.savefig(file_scatter, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  V {os.path.basename(file_scatter)}")

        except Exception as e:
            print(f"  ! Scatter {feat}/{cls}: {e}")

        # ---- 9b: GAM scatter (отдельный PNG) ----
        try:
            if not PYGAM_AVAILABLE:
                raise ImportError("pygam не установлен — GAM-графики пропущены")

            n_unique_x = len(np.unique(x_clean))
            k_use = min(gam_k, n_unique_x - 1)
            k_use = max(k_use, 3)

            gam_model = LinearGAM(s(0, n_splines=k_use)).fit(
                x_clean.reshape(-1, 1), y_clean
            )

            # Предсказание + CI
            x_grid = np.linspace(x_clean.min(), x_clean.max(), 200)
            y_gam = gam_model.predict(x_grid.reshape(-1, 1))
            ci = gam_model.prediction_intervals(x_grid.reshape(-1, 1), width=0.95)
            ci80 = gam_model.prediction_intervals(x_grid.reshape(-1, 1), width=0.80)

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(x_clean, y_clean, alpha=0.15, s=3, color="black",
                       edgecolors="none", rasterized=True)

            # CI 95%
            ax.fill_between(x_grid, ci[:, 0], ci[:, 1],
                            color="#CC0000", alpha=0.15, linewidth=0)
            # CI 80%
            ax.fill_between(x_grid, ci80[:, 0], ci80[:, 1],
                            color="#CC0000", alpha=0.30, linewidth=0)
            # Линия GAM
            ax.plot(x_grid, y_gam, color="#CC0000", linewidth=1.2)

            ax.set_xlabel(feat, fontsize=9)
            ax.set_ylabel(f"P({cls})", fontsize=9)
            ax.set_title(f"{feat} [GAM, n_splines={k_use}]",
                         fontsize=10, fontweight="bold")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            file_gam = os.path.join(
                output_dir,
                f"{base_name}_gam_{cls_safe}_{feat_safe}.png"
            )
            fig.savefig(file_gam, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  V {os.path.basename(file_gam)}")

        except ImportError as ie:
            print(f"  ! GAM {feat}/{cls}: {ie}")
        except Exception as e:
            print(f"  ! GAM {feat}/{cls}: {e}")

else:
    print("  ! predict_proba не поддерживается — scatter/GAM пропущены")


# ============================================================================
# ИТОГО
# ============================================================================
print("\n" + "=" * 60)
print("  ЗАВЕРШЕНО")
print("=" * 60 + "\n")

print(f"  Модель:     {model_name}")
print(f"  Источник:   {selected_dir}")
print(f"  Результаты: {output_dir}\n")

out_files = sorted(os.listdir(output_dir))
print("  Файлы:")
for i, fn in enumerate(out_files, 1):
    print(f"  {i}. {fn}")

print("\n  Macro-F1 drop (top-5 предикторов):")
top_n = min(5, len(imp_df))
for i in range(top_n):
    row = imp_df.iloc[i]
    print(f"    {i + 1}. {row['Predictor']:<25s}  {row['Overall']:+.4f}")

print("\n" + "=" * 60)
