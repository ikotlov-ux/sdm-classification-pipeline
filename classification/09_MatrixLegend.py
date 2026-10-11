#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_MatrixLegend.py — матричная легенда для полигонального слоя.

Класс полигона описывается двумя векторами (например, тип растительности ×
стадия сукцессии). Скрипт строит матрицу «вектор по вертикали × вектор по
горизонтали», раскрашивает её и сохраняет в XLSX (цвета — HEX).

Порядок работы в окне
---------------------
1. Слой полигонов: файл и слой.
2. Где записаны векторы:
   • в двух полях — выбрать поле по горизонтали и поле по вертикали;
   • в одном поле — выбрать поле и способ разбора кода:
       - «первые k цифр | остальные»: 3125 при k = 1 → 3 и 125;
       - «остальные | последние k цифр»: 12125 при k = 3 → 12 и 125;
       - «разделитель»: 3_125, 3-125, 3.125 → 3 и 125;
     и указать, по какой оси идёт первая часть кода.
3. Названия классов: CSV с двумя столбцами «номер; название» для каждого
   вектора (разделитель и кодировка UTF-8 / Windows-1251 определяются сами,
   строка заголовка допускается). Можно один файл на оба вектора.
4. «Построить матрицу». Затем:
   • какая ось задаётся цветом (по умолчанию — ось с бо́льшим числом классов),
     другая — тоном;
   • цвет каждой категории цветовой оси (автопалитра + выбор вручную);
   • способ изменения тона (по умолчанию для всех категорий):
       - к бледному (смешение с белым);
       - к тёмному (смешение с чёрным);
       - к серому (снижение насыщенности);
       - от бледного через базовый к тёмному;
       - светлота HLS (оттенок и насыщенность сохраняются);
       - случайная в пределах цвета (оттенок ± немного, разная светлота);
     с силой, обратным порядком и зерном для случайной шкалы;
   • для любой категории можно задать свой градиент (кнопка в списке);
   • любую ячейку можно перекрасить вручную.
5. «Сохранить XLSX».

Матрица на экране
-----------------
    ЛКМ по ячейке           — изменить цвет ячейки
    ЛКМ по заголовку цвета  — изменить базовый цвет категории
    ПКМ                     — меню (HEX, сброс, градиент категории)
    перетаскивание ЛКМ/СКМ  — сдвиг; стрелки — сдвиг
    колесо мыши             — масштаб к точке под курсором
    Shift + колесо          — сдвиг по горизонтали
    Home                    — вписать матрицу в окно
Заголовки строк и столбцов закреплены.

Выход: 09_MATRIX_LEGEND/<ддммгг_ччмм><_vXXX источника>/<слой>_legend.xlsx
    лист «Матрица»   — раскрашенная матрица, в ячейках HEX
    лист «Классы»    — все сочетания из слоя: код, значения, названия, HEX, RGB,
                       число объектов
    лист «Цвета»     — базовые цвета и градиенты категорий цветовой оси
    лист «Параметры» — исходные данные и настройки (по ним раскраску можно
                       загрузить обратно кнопкой «Загрузить раскраску»)

Запуск из папки проекта:
    %run 09_MatrixLegend.py      # Jupyter — откроется окно
    python 09_MatrixLegend.py    # терминал — откроется окно

CLI (раскраска по умолчанию, без окна):
    python 09_MatrixLegend.py map.gpkg --layer classes --field code --parse first:1 \\
        --first-axis v --names-v veg.csv --names-h stage.csv --out legend.xlsx
    python 09_MatrixLegend.py map.gpkg --fields STAGE VEG --tone tint

Зависимости:
    pip install numpy pandas geopandas pyogrio openpyxl
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import io
import json
import math
import os
import random
import re
import sys
from datetime import datetime

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

OUT_BASE_NAME = "09_MATRIX_LEGEND"
VECTOR_TYPES = [("Векторные слои", "*.gpkg *.shp *.geojson *.json *.fgb *.gml *.kml *.sqlite *.tab"),
                ("Все файлы", "*.*")]
CSV_TYPES = [("CSV", "*.csv *.txt"), ("Все файлы", "*.*")]

PARSE_METHODS = [
    ("first", "Первые k цифр | остальные (3125 → 3 | 125)"),
    ("last", "Остальные | последние k цифр (12125 → 12 | 125)"),
    ("sep", "Разделитель (3_125 → 3 | 125)"),
]

TONE_MODES = [
    ("tint", "К бледному (смешение с белым)"),
    ("shade", "К тёмному (смешение с чёрным)"),
    ("grey", "К серому (снижение насыщенности)"),
    ("tint_shade", "От бледного через базовый к тёмному"),
    ("lightness", "Светлота HLS (насыщенность сохраняется)"),
    ("random", "Случайная в пределах цвета"),
]
TONE_LABEL = dict(TONE_MODES)
TONE_SHORT = {"tint": "к бледному", "shade": "к тёмному", "grey": "к серому",
              "tint_shade": "бледн.→тёмн.", "lightness": "светлота", "random": "случайная"}

PALETTES = {
    "Контрастная качественная": [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
        "#f032e6", "#bfef45", "#469990", "#9a6324", "#800000", "#808000",
        "#000075", "#ffe119", "#a9a9a9"],
    "Радуга (равные шаги оттенка)": None,
    "Золотое сечение (оттенок)": None,
    "Природная (зелёные, бурые, синие)": [
        "#1b7837", "#5aae61", "#a6611a", "#dfc27d", "#2166ac", "#67a9cf",
        "#762a83", "#c51b7d", "#8c510a", "#35978f", "#b2182b", "#7f3b08"],
}
DEFAULT_PALETTE = "Контрастная качественная"


def log(msg=""):
    print(msg, flush=True)


# ===========================================================================
# 1. ДАННЫЕ: ключи, разбор кодов, названия
# ===========================================================================

def norm_key(v):
    """Значение класса → строковый ключ: 3, 3.0, '3', '003' → '3'."""
    if v is None:
        return None
    try:
        import numpy as np
        if isinstance(v, np.generic):
            v = v.item()
    except Exception:
        pass
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return None
        return str(int(v)) if v.is_integer() else repr(v)
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None
    if re.fullmatch(r"[+-]?\d+(\.0*)?", s):
        return str(int(float(s)))
    return s


def raw_code(v):
    """Код как строка для разбора (3125.0 → '3125')."""
    k = norm_key(v)
    if k is None:
        return None
    # '003125' → разбирать как записано, если строка из цифр с ведущими нулями
    s = str(v).strip() if not isinstance(v, (int, float)) else k
    if re.fullmatch(r"\d+", s):
        return s
    return k


def parse_code(v, method, k=1, sep="_"):
    """Код из одного поля → (часть 1, часть 2) или None."""
    s = raw_code(v)
    if s is None:
        return None
    if method == "first":
        if len(s) <= k:
            return None
        a, b = s[:k], s[k:]
    elif method == "last":
        if len(s) <= k:
            return None
        a, b = s[:-k], s[-k:]
    elif method == "sep":
        if not sep or sep not in s:
            return None
        a, b = s.split(sep, 1)
    else:
        raise ValueError(f"Неизвестный способ разбора: {method}")
    a, b = norm_key(a), norm_key(b)
    if a is None or b is None:
        return None
    return a, b


def sort_keys(keys):
    def k(x):
        try:
            return (0, float(x), "")
        except ValueError:
            return (1, 0.0, x.lower())
    return sorted(keys, key=k)


def read_names_csv(path):
    """CSV «номер; название» → {ключ: название}."""
    raw = open(path, "rb").read()
    text = None
    for enc in ("utf-8-sig", "cp1251", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {}
    sample = "\n".join(lines[:20])
    counts = {d: sample.count(d) for d in (";", "\t", ",", "|")}
    delim = max(counts, key=counts.get) if max(counts.values()) > 0 else ";"
    rows = list(csv.reader(io.StringIO("\n".join(lines)), delimiter=delim))
    out = {}
    for i, r in enumerate(rows):
        if len(r) < 2:
            continue
        key = norm_key(r[0])
        name = r[1].strip()
        if key is None:
            continue
        if i == 0 and not re.fullmatch(r"[+-]?\d+(\.\d+)?", r[0].strip()) and len(rows) > 1 \
                and re.fullmatch(r"[+-]?\d+(\.\d+)?", rows[1][0].strip() if rows[1] else ""):
            continue    # строка заголовка
        out[key] = name
    return out


def list_layers(path):
    try:
        import pyogrio
        return [str(x[0]) for x in pyogrio.list_layers(path)]
    except Exception:
        pass
    try:
        import fiona
        return list(fiona.listlayers(path))
    except Exception:
        return [os.path.splitext(os.path.basename(path))[0]]


def read_attrs(path, layer=None):
    import geopandas as gpd
    kw = {"ignore_geometry": True}
    if layer and len(list_layers(path)) > 1:
        kw["layer"] = layer
    elif layer and path.lower().endswith((".gpkg", ".sqlite", ".gdb")):
        kw["layer"] = layer
    df = gpd.read_file(path, **kw)
    return df.drop(columns=[c for c in df.columns if c == "geometry"], errors="ignore")


# ===========================================================================
# 2. МОДЕЛЬ МАТРИЦЫ
# ===========================================================================

class MatrixModel:
    """Матрица: строки — вертикальная ось (v), столбцы — горизонтальная (h)."""

    def __init__(self):
        self.h_keys, self.v_keys = [], []
        self.h_names, self.v_names = {}, {}
        self.h_title, self.v_title = "по горизонтали", "по вертикали"
        self.counts = {}        # (hk, vk) -> число объектов
        self.codes = {}         # (hk, vk) -> исходные коды (одно поле)
        self.skipped = {}       # нераспознанные значения -> число
        self.source = {}
        self.hue_side = "h"
        self.base = {}          # категория цветовой оси -> HEX
        self.tone = {"mode": "tint", "strength": 0.75, "reverse": False, "seed": 1}
        self.override = {}      # категория цветовой оси -> {mode, strength, reverse}
        self.manual = {}        # (hk, vk) -> HEX
        self.hide_absent = True
        self.palette = DEFAULT_PALETTE

    # ---------- построение ----------
    @classmethod
    def from_table(cls, df, cfg):
        """cfg: mode two|one; h_field, v_field | field, method, k, sep, first_axis."""
        m = cls()
        m.source = dict(cfg)
        if cfg["mode"] == "two":
            hf, vf = cfg["h_field"], cfg["v_field"]
            m.h_title, m.v_title = hf, vf
            for (hv, vv), n in df.groupby([hf, vf], dropna=False).size().items():
                hk, vk = norm_key(hv), norm_key(vv)
                if hk is None or vk is None:
                    m.skipped[f"{hv} | {vv}"] = m.skipped.get(f"{hv} | {vv}", 0) + int(n)
                    continue
                m.counts[(hk, vk)] = m.counts.get((hk, vk), 0) + int(n)
        else:
            f = cfg["field"]
            m.h_title = m.v_title = f
            first_v = cfg.get("first_axis", "v") == "v"
            m.v_title = f"{f}: {'1-я' if first_v else '2-я'} часть"
            m.h_title = f"{f}: {'2-я' if first_v else '1-я'} часть"
            for val, n in df[f].value_counts(dropna=False).items():
                pr = parse_code(val, cfg["method"], int(cfg.get("k", 1)), cfg.get("sep", "_"))
                if pr is None:
                    kk = str(val)
                    m.skipped[kk] = m.skipped.get(kk, 0) + int(n)
                    continue
                a, b = pr
                vk, hk = (a, b) if first_v else (b, a)
                m.counts[(hk, vk)] = m.counts.get((hk, vk), 0) + int(n)
                m.codes.setdefault((hk, vk), set()).add(raw_code(val))
        m.h_keys = sort_keys({h for h, _ in m.counts})
        m.v_keys = sort_keys({v for _, v in m.counts})
        if not m.h_keys or not m.v_keys:
            raise ValueError("Не найдено ни одного сочетания классов — проверьте поля и способ разбора.")
        m.hue_side = "h" if len(m.h_keys) >= len(m.v_keys) else "v"
        m.apply_palette(m.palette)
        return m

    def set_names(self, h_names=None, v_names=None):
        if h_names is not None:
            self.h_names = dict(h_names)
        if v_names is not None:
            self.v_names = dict(v_names)

    # ---------- оси ----------
    @property
    def hue_keys(self):
        return self.h_keys if self.hue_side == "h" else self.v_keys

    @property
    def tone_keys(self):
        return self.v_keys if self.hue_side == "h" else self.h_keys

    def name(self, side, key):
        names = self.h_names if side == "h" else self.v_names
        return names.get(key, "")

    def label(self, side, key):
        n = self.name(side, key)
        return f"{key} — {n}" if n else str(key)

    def set_hue_side(self, side):
        if side != self.hue_side:
            self.hue_side = side
            self.override.clear()
            self.apply_palette(self.palette)

    def apply_palette(self, name):
        self.palette = name
        cols = palette_colors(name, len(self.hue_keys))
        self.base = {k: c for k, c in zip(self.hue_keys, cols)}

    # ---------- цвета ----------
    def tone_settings(self, hue_key):
        s = dict(self.tone)
        s.update(self.override.get(hue_key, {}))
        return s

    def auto_color(self, hk, vk):
        hue_key, tone_key = (hk, vk) if self.hue_side == "h" else (vk, hk)
        tk_ = self.tone_keys
        s = self.tone_settings(hue_key)
        return tone_color(self.base[hue_key], tk_.index(tone_key), len(tk_), s["mode"],
                          float(s["strength"]), bool(s["reverse"]), int(self.tone.get("seed", 1)),
                          self.hue_keys.index(hue_key))

    def color(self, hk, vk):
        return self.manual.get((hk, vk)) or self.auto_color(hk, vk)

    def present(self, hk, vk):
        return (hk, vk) in self.counts

    # ---------- состояние ----------
    def state(self):
        return {"hue_side": self.hue_side, "base": self.base, "tone": self.tone,
                "override": self.override, "palette": self.palette,
                "manual": [[h, v, c] for (h, v), c in self.manual.items()],
                "hide_absent": self.hide_absent}

    def load_state(self, st):
        """Применить сохранённую раскраску; категории, которых нет, пропускаются."""
        if st.get("hue_side") in ("h", "v"):
            self.hue_side = st["hue_side"]
        self.apply_palette(st.get("palette", self.palette))
        for k, c in st.get("base", {}).items():
            if k in self.base and is_hex(c):
                self.base[k] = norm_hex(c)
        self.tone.update({k: v for k, v in st.get("tone", {}).items() if k in self.tone})
        self.override = {k: v for k, v in st.get("override", {}).items() if k in self.base}
        hs, vs = set(self.h_keys), set(self.v_keys)
        self.manual = {(h, v): norm_hex(c) for h, v, c in st.get("manual", [])
                       if h in hs and v in vs and is_hex(c)}
        self.hide_absent = bool(st.get("hide_absent", self.hide_absent))


# ===========================================================================
# 3. ЦВЕТ
# ===========================================================================

def is_hex(s):
    return isinstance(s, str) and re.fullmatch(r"#?[0-9a-fA-F]{6}", s.strip()) is not None


def norm_hex(s):
    s = s.strip().lstrip("#")
    return "#" + s.lower()


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def rgb2hex(rgb):
    return "#" + "".join(f"{max(0, min(255, int(round(c * 255)))):02x}" for c in rgb)


def mix(c1, c2, t):
    return tuple(a + (b - a) * t for a, b in zip(c1, c2))


def luminance(h):
    r, g, b = hex2rgb(h)
    return 0.299 * r + 0.587 * g + 0.114 * b


def text_color(h):
    return "#000000" if luminance(h) > 0.55 else "#ffffff"


def palette_colors(name, n):
    base = PALETTES.get(name)
    if base:
        if n <= len(base):
            return base[:n]
        # не хватает — продолжаем по золотому сечению
        return base + palette_colors("Золотое сечение (оттенок)", n - len(base))
    if name.startswith("Радуга"):
        return [rgb2hex(colorsys.hsv_to_rgb((i / max(n, 1)) * 0.83, 0.78, 0.88)) for i in range(n)]
    out = []
    for i in range(n):
        h = (0.07 + i * 0.6180339887) % 1.0
        v = (0.90, 0.72, 0.82)[i % 3]
        out.append(rgb2hex(colorsys.hsv_to_rgb(h, 0.75, v)))
    return out


_RANDOM_CACHE = {}


def _random_scale(base_hex, n, strength, seed, hue_index):
    key = (base_hex, n, round(strength, 4), seed, hue_index)
    if key in _RANDOM_CACHE:
        return _RANDOM_CACHE[key]
    rng = random.Random(seed * 100003 + hue_index * 7919 + n)
    h, l, s = colorsys.rgb_to_hls(*hex2rgb(base_hex))
    span = 0.32 * strength
    lo, hi = max(0.12, l - span), min(0.90, l + span)
    ls = [lo + (hi - lo) * i / max(n - 1, 1) for i in range(n)]
    rng.shuffle(ls)
    out = []
    for i in range(n):
        hh = (h + rng.uniform(-1, 1) * (12.0 / 360.0) * strength) % 1.0
        ss = min(1.0, max(0.15, s * rng.uniform(1 - 0.35 * strength, 1.0)))
        out.append(rgb2hex(colorsys.hls_to_rgb(hh, ls[i], ss)))
    if n >= 1:
        out[0] = base_hex if n == 1 else out[0]
    _RANDOM_CACHE[key] = out
    return out


def tone_color(base_hex, i, n, mode, strength=0.75, reverse=False, seed=1, hue_index=0):
    """Цвет i-й категории тоновой оси (из n) для базового цвета."""
    if n <= 1:
        return base_hex
    t = i / (n - 1)
    if reverse:
        t = 1.0 - t
    s = max(0.0, min(1.0, strength))
    c = hex2rgb(base_hex)
    if mode == "tint":
        return rgb2hex(mix(c, (1, 1, 1), s * t))
    if mode == "shade":
        return rgb2hex(mix(c, (0, 0, 0), s * t))
    if mode == "grey":
        g = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        return rgb2hex(mix(c, (g, g, g), s * t))
    if mode == "tint_shade":
        u = 2 * t - 1
        return rgb2hex(mix(c, (1, 1, 1), s * -u) if u < 0 else mix(c, (0, 0, 0), s * u * 0.85))
    if mode == "lightness":
        h, l, sat = colorsys.rgb_to_hls(*c)
        top, bot = 0.5 + 0.40 * s, 0.5 - 0.36 * s
        return rgb2hex(colorsys.hls_to_rgb(h, top + (bot - top) * t, sat))
    if mode == "random":
        idx = int(round(t * (n - 1)))
        return _random_scale(base_hex, n, s, seed, hue_index)[idx]
    raise ValueError(f"Неизвестный способ тона: {mode}")


# ===========================================================================
# 4. ЗАПИСЬ XLSX
# ===========================================================================

def save_xlsx(m, path):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    wb = Workbook()
    thin = Side(style="thin", color="999999")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    bold = Font(bold=True)

    def fill(h):
        return PatternFill("solid", fgColor=h.lstrip("#").upper())

    # ---------- Матрица ----------
    ws = wb.active
    ws.title = "Матрица"
    ws.cell(1, 1, f"{m.v_title} (строки) \\ {m.h_title} (столбцы)").font = bold
    ws.cell(1, 1).alignment = Alignment(wrap_text=True, vertical="bottom")
    for j, hk in enumerate(m.h_keys):
        c = ws.cell(1, j + 2, m.label("h", hk))
        c.alignment = Alignment(text_rotation=90, wrap_text=True, horizontal="center", vertical="bottom")
        c.font = bold
        c.border = border
        if m.hue_side == "h":
            c.fill = fill(m.base[hk])
            c.font = Font(bold=True, color=text_color(m.base[hk]).lstrip("#").upper())
        ws.column_dimensions[get_column_letter(j + 2)].width = 11
    for i, vk in enumerate(m.v_keys):
        c = ws.cell(i + 2, 1, m.label("v", vk))
        c.font = bold
        c.border = border
        c.alignment = Alignment(wrap_text=True, vertical="center")
        if m.hue_side == "v":
            c.fill = fill(m.base[vk])
            c.font = Font(bold=True, color=text_color(m.base[vk]).lstrip("#").upper())
        for j, hk in enumerate(m.h_keys):
            cell = ws.cell(i + 2, j + 2)
            cell.border = border
            cell.alignment = Alignment(horizontal="center", vertical="center")
            if m.hide_absent and not m.present(hk, vk):
                continue
            h = m.color(hk, vk)
            cell.value = h.upper()
            cell.fill = fill(h)
            cell.font = Font(size=9, color=text_color(h).lstrip("#").upper())
    ws.column_dimensions["A"].width = 42
    ws.row_dimensions[1].height = 150
    ws.freeze_panes = "B2"

    # ---------- Классы ----------
    wc = wb.create_sheet("Классы")
    one = m.source.get("mode") == "one"
    head = (["Код"] if one else []) + [
        f"Горизонталь ({m.h_title})", "Название (горизонталь)",
        f"Вертикаль ({m.v_title})", "Название (вертикаль)",
        "HEX", "R", "G", "B", "Объектов", "Ручная правка"]
    for j, t in enumerate(head):
        c = wc.cell(1, j + 1, t)
        c.font = bold
    r = 2
    for vk in m.v_keys:
        for hk in m.h_keys:
            if not m.present(hk, vk):
                continue
            h = m.color(hk, vk)
            rgb = [int(round(x * 255)) for x in hex2rgb(h)]
            row = ([", ".join(sorted(m.codes.get((hk, vk), [])))] if one else []) + [
                _num(hk), m.name("h", hk), _num(vk), m.name("v", vk),
                h.upper(), *rgb, m.counts[(hk, vk)], "да" if (hk, vk) in m.manual else ""]
            for j, v in enumerate(row):
                wc.cell(r, j + 1, v)
            hc = wc.cell(r, head.index("HEX") + 1)
            hc.fill = fill(h)
            hc.font = Font(color=text_color(h).lstrip("#").upper())
            r += 1
    for j, w in enumerate([10] * (1 if one else 0) + [14, 34, 14, 34, 10, 6, 6, 6, 10, 13]):
        wc.column_dimensions[get_column_letter(j + 1)].width = w
    wc.freeze_panes = "A2"

    # ---------- Цвета ----------
    wk = wb.create_sheet("Цвета")
    side = "h" if m.hue_side == "h" else "v"
    for j, t in enumerate(["Категория", "Название", "Базовый HEX", "Градиент", "Сила",
                           "Обратный порядок", "Свой градиент"]):
        wk.cell(1, j + 1, t).font = bold
    for i, k in enumerate(m.hue_keys):
        s = m.tone_settings(k)
        row = [_num(k), m.name(side, k), m.base[k].upper(), TONE_LABEL[s["mode"]],
               float(s["strength"]), "да" if s["reverse"] else "", "да" if k in m.override else ""]
        for j, v in enumerate(row):
            wk.cell(i + 2, j + 1, v)
        wk.cell(i + 2, 3).fill = fill(m.base[k])
        wk.cell(i + 2, 3).font = Font(color=text_color(m.base[k]).lstrip("#").upper())
    for j, w in enumerate([12, 36, 13, 38, 7, 16, 14]):
        wk.column_dimensions[get_column_letter(j + 1)].width = w

    # ---------- Параметры ----------
    wp = wb.create_sheet("Параметры")
    src = m.source
    meth = dict(PARSE_METHODS).get(src.get("method"), "")
    pars = [
        ("Дата", datetime.now().strftime("%d.%m.%Y %H:%M")),
        ("Слой", src.get("path", "")), ("Имя слоя", src.get("layer", "") or ""),
        ("Векторы", "два поля" if src.get("mode") == "two" else "одно поле"),
        ("Поле по горизонтали", src.get("h_field", "") if src.get("mode") == "two" else ""),
        ("Поле по вертикали", src.get("v_field", "") if src.get("mode") == "two" else ""),
        ("Поле кода", src.get("field", "") if src.get("mode") == "one" else ""),
        ("Разбор", meth), ("k", src.get("k", "") if src.get("mode") == "one" else ""),
        ("Разделитель", src.get("sep", "") if src.get("method") == "sep" else ""),
        ("Первая часть кода", ("по вертикали" if src.get("first_axis") == "v" else "по горизонтали")
         if src.get("mode") == "one" else ""),
        ("Названия по горизонтали (CSV)", src.get("names_h", "") or ""),
        ("Названия по вертикали (CSV)", src.get("names_v", "") or ""),
        ("Цвет задаёт ось", "горизонталь" if m.hue_side == "h" else "вертикаль"),
        ("Автопалитра", m.palette),
        ("Градиент по умолчанию", TONE_LABEL[m.tone["mode"]]),
        ("Сила", m.tone["strength"]), ("Обратный порядок", "да" if m.tone["reverse"] else "нет"),
        ("Зерно случайной шкалы", m.tone["seed"]),
        ("Сочетаний в слое", len(m.counts)),
        ("Размер матрицы", f"{len(m.v_keys)} × {len(m.h_keys)}"),
        ("Нераспознанных значений", sum(m.skipped.values())),
        ("Ручных правок ячеек", len(m.manual)),
        ("state_json", json.dumps(m.state(), ensure_ascii=False)),
    ]
    for i, (k, v) in enumerate(pars):
        wp.cell(i + 1, 1, k).font = bold
        wp.cell(i + 1, 2, v)
    wp.column_dimensions["A"].width = 32
    wp.column_dimensions["B"].width = 90
    wb.save(path)
    return path


def _num(k):
    try:
        f = float(k)
        return int(f) if f.is_integer() else f
    except ValueError:
        return k


def load_state_xlsx(path):
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True)
    if "Параметры" not in wb.sheetnames:
        raise ValueError("В файле нет листа «Параметры».")
    for row in wb["Параметры"].iter_rows(values_only=True):
        if row and row[0] == "state_json" and row[1]:
            return json.loads(row[1])
    raise ValueError("В листе «Параметры» нет сохранённой раскраски.")


def default_out_path(path, layer=None, root_dir=None):
    root_dir = root_dir or os.getcwd()
    stamp = datetime.now().strftime("%d%m%y_%H%M")
    mm = re.search(r"(_v\d{3})$", os.path.basename(os.path.dirname(os.path.abspath(path))))
    folder = os.path.join(root_dir, OUT_BASE_NAME, stamp + (mm.group(1) if mm else ""))
    base = os.path.splitext(os.path.basename(path))[0]
    if layer and layer != base:
        base = f"{base}_{layer}"
    return os.path.join(folder, f"{base}_legend.xlsx")


def _initial_dir(root_dir):
    for name in ("08_TOPOLOGY", "07_GENERALIZATION"):
        b = os.path.join(root_dir, name)
        if os.path.isdir(b):
            subs = [os.path.join(b, d) for d in os.listdir(b) if os.path.isdir(os.path.join(b, d))]
            return max(subs, key=os.path.getmtime) if subs else b
    return root_dir


# ===========================================================================
# 5. ОКНО
# ===========================================================================

class ScrollFrame:
    """Вертикально прокручиваемая панель (для левой панели настроек)."""

    def __init__(self, master, width=430):
        import tkinter as tk
        from tkinter import ttk
        self.outer = ttk.Frame(master)
        self.canvas = tk.Canvas(self.outer, width=width, highlightthickness=0)
        sb = ttk.Scrollbar(self.outer, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        wid = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(wid, width=e.width))
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.outer.bind("<Enter>", self._bind)
        self.outer.bind("<Leave>", self._unbind)

    def _wheel(self, e):
        if getattr(e, "num", None) == 4:
            d = -1
        elif getattr(e, "num", None) == 5:
            d = 1
        else:
            d = -1 if e.delta > 0 else 1
        self.canvas.yview_scroll(d * 2, "units")

    def _bind(self, _):
        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.bind_all(ev, self._wheel)

    def _unbind(self, _):
        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.unbind_all(ev)


class App:
    CELL_W, CELL_H = 74, 30      # размер ячейки при масштабе 1
    RH, CH = 230, 150            # ширина заголовков строк / высота заголовков столбцов (px)

    def __init__(self, root_dir):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.root_dir = root_dir
        self.last_dir = _initial_dir(root_dir)
        self.df = None
        self.m = None
        self.z, self.ox, self.oy = 1.0, 0.0, 0.0
        self._redraw_pending = False
        self._drag = None

        w = self.win = tk.Tk()
        w.title("09 — Матричная легенда")
        w.geometry("1360x860")
        w.minsize(900, 560)
        w.attributes("-topmost", True)
        w.after(500, lambda: w.attributes("-topmost", False))

        pw = ttk.PanedWindow(w, orient="horizontal")
        pw.pack(fill="both", expand=True)
        self.sf = ScrollFrame(pw, width=440)
        pw.add(self.sf.outer, weight=0)
        right = ttk.Frame(pw)
        pw.add(right, weight=1)
        self._build_left(self.sf.inner)
        self._build_canvas(right)
        self.status = tk.StringVar(value="Выберите слой полигонов.")
        ttk.Label(w, textvariable=self.status, anchor="w", relief="sunken").pack(fill="x", side="bottom")

    # ------------------------------------------------------------------
    # левая панель
    # ------------------------------------------------------------------
    def _build_left(self, p):
        tk, ttk = self.tk, self.ttk
        pad = {"padx": 6, "pady": 3}

        # 1. слой
        f1 = ttk.LabelFrame(p, text="1. Слой полигонов")
        f1.pack(fill="x", padx=6, pady=4)
        self.v_path = tk.StringVar()
        self.v_layer = tk.StringVar()
        e = ttk.Entry(f1, textvariable=self.v_path)
        e.grid(row=0, column=0, columnspan=2, sticky="we", **pad)
        ttk.Button(f1, text="Обзор…", command=self.browse_layer).grid(row=0, column=2, **pad)
        ttk.Label(f1, text="Слой").grid(row=1, column=0, sticky="w", **pad)
        self.cb_layer = ttk.Combobox(f1, textvariable=self.v_layer, state="readonly", width=28)
        self.cb_layer.grid(row=1, column=1, columnspan=2, sticky="we", **pad)
        self.cb_layer.bind("<<ComboboxSelected>>", lambda e: self.load_attrs())
        f1.columnconfigure(1, weight=1)

        # 2. векторы
        f2 = ttk.LabelFrame(p, text="2. Векторы матрицы")
        f2.pack(fill="x", padx=6, pady=4)
        self.v_mode = tk.StringVar(value="two")
        ttk.Radiobutton(f2, text="В двух полях", value="two", variable=self.v_mode,
                        command=self._mode_changed).grid(row=0, column=0, sticky="w", **pad)
        ttk.Radiobutton(f2, text="В одном поле (код)", value="one", variable=self.v_mode,
                        command=self._mode_changed).grid(row=0, column=1, sticky="w", **pad)
        self.f_two = ttk.Frame(f2)
        self.f_two.grid(row=1, column=0, columnspan=2, sticky="we")
        self.v_hf, self.v_vf = tk.StringVar(), tk.StringVar()
        ttk.Label(self.f_two, text="По горизонтали").grid(row=0, column=0, sticky="w", **pad)
        self.cb_hf = ttk.Combobox(self.f_two, textvariable=self.v_hf, state="readonly", width=24)
        self.cb_hf.grid(row=0, column=1, sticky="we", **pad)
        ttk.Label(self.f_two, text="По вертикали").grid(row=1, column=0, sticky="w", **pad)
        self.cb_vf = ttk.Combobox(self.f_two, textvariable=self.v_vf, state="readonly", width=24)
        self.cb_vf.grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(self.f_two, text="Поменять местами", command=self._swap_fields).grid(
            row=2, column=1, sticky="w", **pad)

        self.f_one = ttk.Frame(f2)
        self.v_f, self.v_meth = tk.StringVar(), tk.StringVar(value=PARSE_METHODS[0][1])
        self.v_k, self.v_sep = tk.StringVar(value="1"), tk.StringVar(value="_")
        self.v_first = tk.StringVar(value="по вертикали")
        ttk.Label(self.f_one, text="Поле кода").grid(row=0, column=0, sticky="w", **pad)
        self.cb_f = ttk.Combobox(self.f_one, textvariable=self.v_f, state="readonly", width=24)
        self.cb_f.grid(row=0, column=1, columnspan=3, sticky="we", **pad)
        ttk.Label(self.f_one, text="Разбор").grid(row=1, column=0, sticky="w", **pad)
        cb_m = ttk.Combobox(self.f_one, textvariable=self.v_meth, state="readonly", width=36,
                            values=[t for _, t in PARSE_METHODS])
        cb_m.grid(row=1, column=1, columnspan=3, sticky="we", **pad)
        ttk.Label(self.f_one, text="k цифр").grid(row=2, column=0, sticky="w", **pad)
        ttk.Spinbox(self.f_one, from_=1, to=8, textvariable=self.v_k, width=4,
                    command=self._preview).grid(row=2, column=1, sticky="w", **pad)
        ttk.Label(self.f_one, text="Разделитель").grid(row=2, column=2, sticky="e", **pad)
        ttk.Entry(self.f_one, textvariable=self.v_sep, width=4).grid(row=2, column=3, sticky="w", **pad)
        ttk.Label(self.f_one, text="1-я часть кода").grid(row=3, column=0, sticky="w", **pad)
        ttk.Combobox(self.f_one, textvariable=self.v_first, state="readonly", width=16,
                     values=["по вертикали", "по горизонтали"]).grid(row=3, column=1, columnspan=3,
                                                                     sticky="w", **pad)
        self.v_prev = tk.StringVar()
        ttk.Label(self.f_one, textvariable=self.v_prev, foreground="#555555",
                  wraplength=390, justify="left").grid(row=4, column=0, columnspan=4, sticky="w", **pad)
        for v in (self.v_f, self.v_meth, self.v_k, self.v_sep, self.v_first):
            v.trace_add("write", lambda *a: self._preview())
        self.f_one.columnconfigure(1, weight=1)

        # 3. названия
        f3 = ttk.LabelFrame(p, text="3. Названия классов (CSV: номер; название)")
        f3.pack(fill="x", padx=6, pady=4)
        self.v_nh, self.v_nv = tk.StringVar(), tk.StringVar()
        for r, (lab, var) in enumerate((("По горизонтали", self.v_nh), ("По вертикали", self.v_nv))):
            ttk.Label(f3, text=lab).grid(row=r, column=0, sticky="w", **pad)
            ttk.Entry(f3, textvariable=var, width=22).grid(row=r, column=1, sticky="we", **pad)
            ttk.Button(f3, text="Обзор…", width=7,
                       command=lambda v=var: self.browse_csv(v)).grid(row=r, column=2, **pad)
            ttk.Button(f3, text="×", width=2, command=lambda v=var: v.set("")).grid(row=r, column=3, **pad)
        f3.columnconfigure(1, weight=1)
        ttk.Button(p, text="Построить матрицу", command=self.build).pack(fill="x", padx=12, pady=6)

        # 4. цвет и тон
        f4 = ttk.LabelFrame(p, text="4. Цвет и тон")
        f4.pack(fill="x", padx=6, pady=4)
        self.v_hue = tk.StringVar(value="h")
        self.rb_h = ttk.Radiobutton(f4, text="Цвет — по горизонтали", value="h",
                                    variable=self.v_hue, command=self._hue_changed)
        self.rb_v = ttk.Radiobutton(f4, text="Цвет — по вертикали", value="v",
                                    variable=self.v_hue, command=self._hue_changed)
        self.rb_h.grid(row=0, column=0, columnspan=2, sticky="w", **pad)
        self.rb_v.grid(row=1, column=0, columnspan=2, sticky="w", **pad)
        ttk.Label(f4, text="Автопалитра").grid(row=2, column=0, sticky="w", **pad)
        self.v_pal = tk.StringVar(value=DEFAULT_PALETTE)
        ttk.Combobox(f4, textvariable=self.v_pal, state="readonly", width=30,
                     values=list(PALETTES)).grid(row=2, column=1, sticky="we", **pad)
        ttk.Button(f4, text="Применить автопалитру", command=self._apply_palette).grid(
            row=3, column=1, sticky="w", **pad)
        ttk.Separator(f4).grid(row=4, column=0, columnspan=2, sticky="we", pady=4)
        ttk.Label(f4, text="Тон по умолчанию").grid(row=5, column=0, sticky="w", **pad)
        self.v_tmode = tk.StringVar(value=TONE_LABEL["tint"])
        cbt = ttk.Combobox(f4, textvariable=self.v_tmode, state="readonly", width=34,
                           values=[t for _, t in TONE_MODES])
        cbt.grid(row=5, column=1, sticky="we", **pad)
        cbt.bind("<<ComboboxSelected>>", lambda e: self._tone_changed())
        ttk.Label(f4, text="Сила").grid(row=6, column=0, sticky="w", **pad)
        self.v_str = tk.DoubleVar(value=0.75)
        fs = ttk.Frame(f4)
        fs.grid(row=6, column=1, sticky="we", **pad)
        ttk.Scale(fs, from_=0.1, to=1.0, variable=self.v_str,
                  command=lambda e: self._tone_changed()).pack(side="left", fill="x", expand=True)
        self.v_str_lbl = tk.StringVar(value="0.75")
        ttk.Label(fs, textvariable=self.v_str_lbl, width=5).pack(side="left")
        self.v_rev = tk.BooleanVar(value=False)
        ttk.Checkbutton(f4, text="Обратный порядок тона", variable=self.v_rev,
                        command=self._tone_changed).grid(row=7, column=1, sticky="w", **pad)
        ttk.Label(f4, text="Зерно (случайная)").grid(row=8, column=0, sticky="w", **pad)
        self.v_seed = tk.StringVar(value="1")
        sp = ttk.Spinbox(f4, from_=1, to=9999, textvariable=self.v_seed, width=6,
                         command=self._tone_changed)
        sp.grid(row=8, column=1, sticky="w", **pad)
        sp.bind("<Return>", lambda e: self._tone_changed())
        f4.columnconfigure(1, weight=1)

        # 5. категории цветовой оси
        self.f5 = ttk.LabelFrame(p, text="5. Категории цветовой оси")
        self.f5.pack(fill="x", padx=6, pady=4)
        ttk.Label(self.f5, text="Щелчок по цветному квадрату — изменить цвет категории;\n"
                                "«Градиент» — свой способ тона для категории.",
                  foreground="#555555", justify="left").pack(anchor="w", padx=6)
        self.f_cats = ttk.Frame(self.f5)
        self.f_cats.pack(fill="x")

        # 6. ячейки и сохранение
        f6 = ttk.LabelFrame(p, text="6. Ячейки и сохранение")
        f6.pack(fill="x", padx=6, pady=4)
        self.v_hide = tk.BooleanVar(value=True)
        ttk.Checkbutton(f6, text="Не раскрашивать сочетания, которых нет в слое",
                        variable=self.v_hide, command=self._hide_changed).grid(
            row=0, column=0, columnspan=3, sticky="w", **pad)
        ttk.Button(f6, text="Сбросить ручные правки ячеек", command=self._reset_manual).grid(
            row=1, column=0, columnspan=3, sticky="w", **pad)
        ttk.Button(f6, text="Загрузить раскраску из XLSX…", command=self._load_state).grid(
            row=2, column=0, columnspan=3, sticky="w", **pad)
        ttk.Label(f6, text="Файл XLSX").grid(row=3, column=0, sticky="w", **pad)
        self.v_out = tk.StringVar()
        ttk.Entry(f6, textvariable=self.v_out, width=24).grid(row=3, column=1, sticky="we", **pad)
        ttk.Button(f6, text="Обзор…", width=7, command=self.browse_out).grid(row=3, column=2, **pad)
        f6.columnconfigure(1, weight=1)
        fb = ttk.Frame(p)
        fb.pack(fill="x", padx=6, pady=8)
        ttk.Button(fb, text="Сохранить XLSX", command=self.save).pack(side="left", padx=6)
        ttk.Button(fb, text="Закрыть", command=self.win.destroy).pack(side="right", padx=6)
        self._mode_changed()

    # ------------------------------------------------------------------
    # холст матрицы
    # ------------------------------------------------------------------
    def _build_canvas(self, p):
        tk, ttk = self.tk, self.ttk
        bar = ttk.Frame(p)
        bar.pack(fill="x")
        ttk.Button(bar, text="Вписать (Home)", command=self.fit).pack(side="left", padx=4, pady=3)
        ttk.Button(bar, text="Масштаб 1:1", command=self.zoom_1).pack(side="left", padx=4)
        ttk.Button(bar, text="+", width=3, command=lambda: self.zoom_center(1.25)).pack(side="left")
        ttk.Button(bar, text="−", width=3, command=lambda: self.zoom_center(0.8)).pack(side="left", padx=4)
        ttk.Label(bar, foreground="#555555",
                  text="ЛКМ — цвет ячейки, ПКМ — меню, перетаскивание — сдвиг, колесо — масштаб").pack(
            side="left", padx=10)
        self.cv = c = tk.Canvas(p, background="#f4f4f4", highlightthickness=0)
        c.pack(fill="both", expand=True)
        c.bind("<Configure>", lambda e: self.redraw())
        c.bind("<ButtonPress-1>", self._press)
        c.bind("<B1-Motion>", self._motion)
        c.bind("<ButtonRelease-1>", self._release)
        c.bind("<ButtonPress-2>", lambda e: self._press(e, pan_only=True))
        c.bind("<B2-Motion>", self._motion)
        c.bind("<ButtonRelease-2>", self._release)
        c.bind("<Button-3>", self._menu)
        c.bind("<MouseWheel>", self._wheel)
        c.bind("<Shift-MouseWheel>", lambda e: self._pan_wheel(e, horiz=True))
        c.bind("<Button-4>", self._wheel)
        c.bind("<Button-5>", self._wheel)
        c.bind("<Shift-Button-4>", lambda e: self._pan_wheel(e, horiz=True))
        c.bind("<Shift-Button-5>", lambda e: self._pan_wheel(e, horiz=True))
        c.bind("<Enter>", lambda e: c.focus_set())
        for key, dx, dy in (("<Left>", 60, 0), ("<Right>", -60, 0), ("<Up>", 0, 60), ("<Down>", 0, -60)):
            c.bind(key, lambda e, dx=dx, dy=dy: self.pan(dx, dy))
        c.bind("<Home>", lambda e: self.fit())

    # ---------- геометрия ----------
    def cw(self):
        return self.CELL_W * self.z

    def ch(self):
        return self.CELL_H * self.z

    def cell_at(self, x, y):
        """→ ('cell', j, i) | ('hhead', j) | ('vhead', i) | None."""
        if self.m is None:
            return None
        nj, ni = len(self.m.h_keys), len(self.m.v_keys)
        j = math.floor((x - self.RH - self.ox) / self.cw())
        i = math.floor((y - self.CH - self.oy) / self.ch())
        if x >= self.RH and y >= self.CH:
            if 0 <= j < nj and 0 <= i < ni:
                return ("cell", j, i)
        elif y < self.CH and x >= self.RH and 0 <= j < nj:
            return ("hhead", j)
        elif x < self.RH and y >= self.CH and 0 <= i < ni:
            return ("vhead", i)
        return None

    def redraw(self):
        if not self._redraw_pending:
            self._redraw_pending = True
            self.win.after_idle(self._draw)

    def _draw(self):
        self._redraw_pending = False
        c = self.cv
        c.delete("all")
        W, H = c.winfo_width(), c.winfo_height()
        if self.m is None:
            c.create_text(W / 2, H / 2, fill="#777777", font=("TkDefaultFont", 12),
                          text="Выберите слой и поля, затем нажмите «Построить матрицу».")
            return
        m = self.m
        cw, ch = self.cw(), self.ch()
        nj, ni = len(m.h_keys), len(m.v_keys)
        j0 = max(0, math.floor((-self.ox) / cw))
        j1 = min(nj, math.ceil((W - self.RH - self.ox) / cw))
        i0 = max(0, math.floor((-self.oy) / ch))
        i1 = min(ni, math.ceil((H - self.CH - self.oy) / ch))
        fsz = int(max(6, min(13, 8.5 * self.z)))
        show_txt = ch >= 15 and cw >= 44
        outline = "#9a9a9a" if cw > 8 else ""
        for i in range(i0, i1):
            vk = m.v_keys[i]
            y = self.CH + self.oy + i * ch
            for j in range(j0, j1):
                hk = m.h_keys[j]
                x = self.RH + self.ox + j * cw
                if m.hide_absent and not m.present(hk, vk):
                    c.create_rectangle(x, y, x + cw, y + ch, fill="#ffffff", outline=outline)
                    if cw > 10:
                        c.create_line(x + 3, y + 3, x + cw - 3, y + ch - 3, fill="#dddddd")
                    continue
                col = m.color(hk, vk)
                c.create_rectangle(x, y, x + cw, y + ch, fill=col, outline=outline)
                if (hk, vk) in m.manual and cw > 12:
                    s = min(10, cw / 4, ch / 2.5)
                    c.create_polygon(x + cw - s, y, x + cw, y, x + cw, y + s,
                                     fill=text_color(col), outline="")
                if show_txt:
                    c.create_text(x + cw / 2, y + ch / 2, text=col.upper(), fill=text_color(col),
                                  font=("TkDefaultFont", fsz))
        # закреплённые заголовки
        hfs = int(max(7, min(11, 9 * self.z)))
        c.create_rectangle(0, 0, W, self.CH, fill="#e8e8e8", outline="")
        c.create_rectangle(0, 0, self.RH, H, fill="#e8e8e8", outline="")
        for j in range(j0, j1):
            hk = m.h_keys[j]
            x = self.RH + self.ox + j * cw
            if m.hue_side == "h":
                b = m.base[hk]
                c.create_rectangle(x, self.CH - 16, x + cw, self.CH, fill=b, outline="#777777")
            c.create_line(x, 0, x, self.CH, fill="#cccccc")
            if cw >= 7:
                f_ = int(max(6, min(hfs, cw * 0.75)))
                txt = _clip(m.label("h", hk), int((self.CH - 24) / (f_ * 0.62)))
                c.create_text(x + cw / 2, self.CH - 20, text=txt, angle=90, anchor="w",
                              font=("TkDefaultFont", f_), fill="#222222")
        for i in range(i0, i1):
            vk = m.v_keys[i]
            y = self.CH + self.oy + i * ch
            if m.hue_side == "v":
                c.create_rectangle(self.RH - 16, y, self.RH, y + ch, fill=m.base[vk], outline="#777777")
            c.create_line(0, y, self.RH, y, fill="#cccccc")
            if ch >= 7:
                f_ = int(max(6, min(hfs, ch * 0.75)))
                txt = _clip(m.label("v", vk), int((self.RH - 26) / (f_ * 0.62)))
                c.create_text(self.RH - 20, y + ch / 2, text=txt, anchor="e",
                              font=("TkDefaultFont", f_), fill="#222222")
        c.create_rectangle(0, 0, self.RH, self.CH, fill="#dcdcdc", outline="#bbbbbb")
        c.create_text(8, 8, anchor="nw", width=self.RH - 14, font=("TkDefaultFont", 9),
                      fill="#333333",
                      text=f"Строки: {m.v_title}\nСтолбцы: {m.h_title}\n\n"
                           f"{ni} × {nj}, сочетаний в слое: {len(m.counts)}\n"
                           f"Цвет: {'столбцы' if m.hue_side == 'h' else 'строки'}; "
                           f"тон: {'строки' if m.hue_side == 'h' else 'столбцы'}")

    # ---------- навигация ----------
    def _clamp(self):
        """Матрица не уходит за край окна целиком."""
        if self.m is None:
            return
        W = self.cv.winfo_width() - self.RH
        H = self.cv.winfo_height() - self.CH
        mw, mh = len(self.m.h_keys) * self.cw(), len(self.m.v_keys) * self.ch()
        keep = 60
        self.ox = min(max(self.ox, min(0, W - mw) - keep if mw > W else -mw + keep), max(W - keep, 0))
        self.oy = min(max(self.oy, min(0, H - mh) - keep if mh > H else -mh + keep), max(H - keep, 0))

    def pan(self, dx, dy):
        self.ox += dx
        self.oy += dy
        self._clamp()
        self.redraw()

    def _zoom_at(self, f, x, y):
        nz = max(0.08, min(8.0, self.z * f))
        f = nz / self.z
        px, py = x - self.RH, y - self.CH
        self.ox = px - (px - self.ox) * f
        self.oy = py - (py - self.oy) * f
        self.z = nz
        self._clamp()
        self.redraw()

    def zoom_center(self, f):
        self._zoom_at(f, (self.cv.winfo_width() + self.RH) / 2, (self.cv.winfo_height() + self.CH) / 2)

    def zoom_1(self):
        self.z, self.ox, self.oy = 1.0, 0.0, 0.0
        self.redraw()

    def view_readable(self):
        """Начальный вид: вписать, но не мельче читаемого масштаба (дальше — сдвиг)."""
        self.fit()
        if self.z < 0.75:
            self.z, self.ox, self.oy = 0.75, 0.0, 0.0
            self.redraw()

    def fit(self):
        if self.m is None:
            return
        W, H = self.cv.winfo_width() - self.RH - 10, self.cv.winfo_height() - self.CH - 10
        nj, ni = len(self.m.h_keys), len(self.m.v_keys)
        self.z = max(0.08, min(3.0, W / (nj * self.CELL_W), H / (ni * self.CELL_H)))
        self.ox = self.oy = 0.0
        self.redraw()

    def _wheel(self, e):
        up = (getattr(e, "num", None) == 4) or (getattr(e, "delta", 0) > 0)
        self._zoom_at(1.15 if up else 1 / 1.15, e.x, e.y)

    def _pan_wheel(self, e, horiz=False):
        up = (getattr(e, "num", None) == 4) or (getattr(e, "delta", 0) > 0)
        self.pan(80 if up else -80, 0)

    def _press(self, e, pan_only=False):
        self._drag = {"x": e.x, "y": e.y, "moved": False, "pan_only": pan_only}

    def _motion(self, e):
        d = self._drag
        if not d:
            return
        dx, dy = e.x - d["x"], e.y - d["y"]
        if not d["moved"] and abs(dx) + abs(dy) < 4:
            return
        d["moved"] = True
        d["x"], d["y"] = e.x, e.y
        self.pan(dx, dy)

    def _release(self, e):
        d, self._drag = self._drag, None
        if not d or d["moved"] or d["pan_only"]:
            return
        hit = self.cell_at(e.x, e.y)
        if not hit:
            return
        if hit[0] == "cell":
            self.edit_cell(self.m.h_keys[hit[1]], self.m.v_keys[hit[2]])
        elif hit[0] == "hhead" and self.m.hue_side == "h":
            self.edit_base(self.m.h_keys[hit[1]])
        elif hit[0] == "vhead" and self.m.hue_side == "v":
            self.edit_base(self.m.v_keys[hit[1]])

    def _menu(self, e):
        hit = self.cell_at(e.x, e.y)
        if not hit:
            return
        tk = self.tk
        m = self.m
        menu = tk.Menu(self.win, tearoff=0)
        if hit[0] == "cell":
            hk, vk = m.h_keys[hit[1]], m.v_keys[hit[2]]
            hue_key = hk if m.hue_side == "h" else vk
            n = m.counts.get((hk, vk), 0)
            menu.add_command(label=f"{m.label('v', vk)}  ×  {m.label('h', hk)}  ({n} объектов)",
                             state="disabled")
            menu.add_separator()
            menu.add_command(label="Изменить цвет ячейки…", command=lambda: self.edit_cell(hk, vk))
            menu.add_command(label="Ввести HEX ячейки…", command=lambda: self.edit_cell(hk, vk, hex_=True))
            menu.add_command(label="Сбросить ячейку",
                             command=lambda: (m.manual.pop((hk, vk), None), self.redraw()))
        else:
            hue_key = None
            if hit[0] == "hhead" and m.hue_side == "h":
                hue_key = m.h_keys[hit[1]]
            if hit[0] == "vhead" and m.hue_side == "v":
                hue_key = m.v_keys[hit[1]]
            if hue_key is None:
                return
        side = m.hue_side
        menu.add_separator()
        menu.add_command(label=f"Базовый цвет «{_clip(m.label(side, hue_key), 40)}»…",
                         command=lambda: self.edit_base(hue_key))
        menu.add_command(label="Ввести HEX базового цвета…",
                         command=lambda: self.edit_base(hue_key, hex_=True))
        menu.add_command(label="Градиент этой категории…", command=lambda: self.edit_gradient(hue_key))
        menu.add_separator()
        menu.add_command(label="Сбросить все ручные правки ячеек", command=self._reset_manual)
        menu.tk_popup(e.x_root, e.y_root)

    # ------------------------------------------------------------------
    # действия
    # ------------------------------------------------------------------
    def browse_layer(self):
        from tkinter import filedialog
        p = filedialog.askopenfilename(parent=self.win, title="Слой полигонов",
                                       initialdir=self.last_dir, filetypes=VECTOR_TYPES)
        if not p:
            return
        self.last_dir = os.path.dirname(p)
        self.v_path.set(p)
        lays = list_layers(p)
        self.cb_layer["values"] = lays
        self.v_layer.set(lays[0] if lays else "")
        self.load_attrs()

    def load_attrs(self):
        from tkinter import messagebox
        p = self.v_path.get().strip()
        if not p:
            return
        try:
            self.df = read_attrs(p, self.v_layer.get() or None)
        except Exception as ex:
            messagebox.showerror("Ошибка чтения", str(ex), parent=self.win)
            return
        cols = [c for c in self.df.columns]
        for cb in (self.cb_hf, self.cb_vf, self.cb_f):
            cb["values"] = cols
        # угадываем поля
        low = {c.lower(): c for c in cols}
        guess = next((low[k] for k in ("code", "class_value", "class", "gridcode", "kod") if k in low),
                     cols[0] if cols else "")
        self.v_f.set(guess)
        if len(cols) >= 2:
            self.v_hf.set(cols[0])
            self.v_vf.set(cols[1])
        self.v_out.set(default_out_path(p, self.v_layer.get(), self.root_dir))
        self.status.set(f"Объектов: {len(self.df)}, полей: {len(cols)}.")
        self._preview()

    def browse_csv(self, var):
        from tkinter import filedialog
        p = filedialog.askopenfilename(parent=self.win, title="CSV с названиями классов",
                                       initialdir=self.last_dir, filetypes=CSV_TYPES)
        if p:
            self.last_dir = os.path.dirname(p)
            var.set(p)

    def browse_out(self):
        from tkinter import filedialog
        cur = self.v_out.get()
        p = filedialog.asksaveasfilename(parent=self.win, title="Файл легенды",
                                         defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")],
                                         initialdir=os.path.dirname(cur) if cur else self.last_dir,
                                         initialfile=os.path.basename(cur) if cur else "legend.xlsx")
        if p:
            self.v_out.set(p)

    def _mode_changed(self):
        if self.v_mode.get() == "two":
            self.f_one.grid_remove()
            self.f_two.grid(row=1, column=0, columnspan=2, sticky="we")
        else:
            self.f_two.grid_remove()
            self.f_one.grid(row=1, column=0, columnspan=2, sticky="we")
            self._preview()

    def _swap_fields(self):
        a, b = self.v_hf.get(), self.v_vf.get()
        self.v_hf.set(b)
        self.v_vf.set(a)

    def _parse_cfg(self):
        meth = next(k for k, t in PARSE_METHODS if t == self.v_meth.get())
        try:
            k = max(1, int(self.v_k.get()))
        except ValueError:
            k = 1
        return meth, k, self.v_sep.get()

    def _preview(self):
        if self.df is None or self.v_mode.get() != "one" or self.v_f.get() not in self.df.columns:
            self.v_prev.set("")
            return
        meth, k, sep = self._parse_cfg()
        vals = list(self.df[self.v_f.get()].dropna().unique()[:400])
        ok = [(v, parse_code(v, meth, k, sep)) for v in vals]
        good = [(v, pr) for v, pr in ok if pr]
        bad = [v for v, pr in ok if not pr]
        first_v = self.v_first.get() == "по вертикали"
        ex = "; ".join(f"{raw_code(v)} → {a} | {b}" for v, (a, b) in good[:4])
        t = f"Пример: {ex}" if ex else "Ни одно значение не разобрано."
        t += f"\n1-я часть → {'строки' if first_v else 'столбцы'}, 2-я → {'столбцы' if first_v else 'строки'}."
        if bad:
            t += f"\nНе разобрано (из первых {len(vals)}): {len(bad)}, напр. {', '.join(map(str, bad[:3]))}"
        self.v_prev.set(t)

    def build(self):
        from tkinter import messagebox
        if self.df is None:
            messagebox.showwarning("Нет данных", "Сначала выберите слой.", parent=self.win)
            return
        cfg = {"path": self.v_path.get(), "layer": self.v_layer.get(), "mode": self.v_mode.get(),
               "names_h": self.v_nh.get().strip(), "names_v": self.v_nv.get().strip()}
        if cfg["mode"] == "two":
            cfg.update(h_field=self.v_hf.get(), v_field=self.v_vf.get())
            if not cfg["h_field"] or not cfg["v_field"] or cfg["h_field"] == cfg["v_field"]:
                messagebox.showwarning("Поля", "Выберите два разных поля.", parent=self.win)
                return
        else:
            meth, k, sep = self._parse_cfg()
            cfg.update(field=self.v_f.get(), method=meth, k=k, sep=sep,
                       first_axis="v" if self.v_first.get() == "по вертикали" else "h")
        try:
            m = MatrixModel.from_table(self.df, cfg)
            m.set_names(read_names_csv(cfg["names_h"]) if cfg["names_h"] else {},
                        read_names_csv(cfg["names_v"]) if cfg["names_v"] else {})
        except Exception as ex:
            messagebox.showerror("Ошибка", str(ex), parent=self.win)
            return
        old = self.m
        m.palette = self.v_pal.get()
        m.apply_palette(m.palette)
        m.tone.update(self._tone_from_ui())
        m.hide_absent = self.v_hide.get()
        if old is not None and old.h_keys == m.h_keys and old.v_keys == m.v_keys:
            m.load_state(old.state())      # перестроение тех же осей — сохраняем раскраску
        self.m = m
        self.v_hue.set(m.hue_side)
        self._update_hue_labels()
        self._fill_cats()
        miss_h = [k for k in m.h_keys if k not in m.h_names] if m.h_names else []
        miss_v = [k for k in m.v_keys if k not in m.v_names] if m.v_names else []
        msg = (f"Матрица {len(m.v_keys)} × {len(m.h_keys)}, сочетаний в слое: {len(m.counts)}."
               + (f" Нераспознанных значений: {sum(m.skipped.values())}." if m.skipped else "")
               + (f" Нет названий: по горизонтали {len(miss_h)}" if miss_h else "")
               + (f", по вертикали {len(miss_v)}." if miss_v else ""))
        self.status.set(msg)
        log(msg)
        self.win.after(50, self.view_readable)

    def _update_hue_labels(self):
        if self.m:
            self.rb_h.configure(text=f"Цвет — по горизонтали ({len(self.m.h_keys)} кат.)")
            self.rb_v.configure(text=f"Цвет — по вертикали ({len(self.m.v_keys)} кат.)")

    def _tone_from_ui(self):
        mode = next(k for k, t in TONE_MODES if t == self.v_tmode.get())
        try:
            seed = int(self.v_seed.get())
        except ValueError:
            seed = 1
        s = round(float(self.v_str.get()), 2)
        self.v_str_lbl.set(f"{s:.2f}")
        return {"mode": mode, "strength": s, "reverse": bool(self.v_rev.get()), "seed": seed}

    def _tone_changed(self):
        t = self._tone_from_ui()
        if self.m:
            self.m.tone.update(t)
            self._fill_cats()
            self.redraw()

    def _hue_changed(self):
        if self.m:
            self.m.set_hue_side(self.v_hue.get())
            self._fill_cats()
            self.redraw()

    def _apply_palette(self):
        if self.m:
            self.m.apply_palette(self.v_pal.get())
            self._fill_cats()
            self.redraw()

    def _hide_changed(self):
        if self.m:
            self.m.hide_absent = self.v_hide.get()
            self.redraw()

    def _reset_manual(self):
        if self.m and self.m.manual:
            self.m.manual.clear()
            self.redraw()

    def _fill_cats(self):
        tk, ttk = self.tk, self.ttk
        for w in self.f_cats.winfo_children():
            w.destroy()
        if not self.m:
            return
        m = self.m
        side = m.hue_side
        self.f5.configure(text=f"5. Категории цветовой оси ({'горизонталь' if side == 'h' else 'вертикаль'}, "
                               f"{len(m.hue_keys)})")
        for r, k in enumerate(m.hue_keys):
            sw = tk.Label(self.f_cats, width=3, background=m.base[k], relief="raised", cursor="hand2")
            sw.grid(row=r, column=0, padx=(6, 4), pady=1, sticky="w")
            sw.bind("<Button-1>", lambda e, k=k: self.edit_base(k))
            ttk.Label(self.f_cats, text=_clip(m.label(side, k), 34)).grid(row=r, column=1, sticky="w")
            gl = (TONE_SHORT[m.override[k]["mode"]] if k in m.override else "по умолч.")
            ttk.Button(self.f_cats, text=f"Градиент: {gl}", width=20,
                       command=lambda k=k: self.edit_gradient(k)).grid(row=r, column=2, padx=4, pady=1,
                                                                        sticky="e")
        self.f_cats.columnconfigure(1, weight=1)

    def _ask_color(self, init, title):
        from tkinter import colorchooser
        res = colorchooser.askcolor(color=init, parent=self.win, title=title)
        return norm_hex(res[1]) if res and res[1] else None

    def _ask_hex(self, init, title):
        from tkinter import simpledialog, messagebox
        s = simpledialog.askstring(title, "HEX-код цвета (например, #3a7bd5):",
                                   initialvalue=init.upper(), parent=self.win)
        if s is None:
            return None
        if not is_hex(s):
            messagebox.showwarning("HEX", f"Неверный код цвета: {s}", parent=self.win)
            return None
        return norm_hex(s)

    def edit_cell(self, hk, vk, hex_=False):
        m = self.m
        cur = m.color(hk, vk)
        title = f"Ячейка {m.label('v', vk)} × {m.label('h', hk)}"
        new = self._ask_hex(cur, title) if hex_ else self._ask_color(cur, title)
        if new:
            if new == m.auto_color(hk, vk):
                m.manual.pop((hk, vk), None)
            else:
                m.manual[(hk, vk)] = new
            self.redraw()

    def edit_base(self, k, hex_=False):
        m = self.m
        title = f"Базовый цвет: {m.label(m.hue_side, k)}"
        new = self._ask_hex(m.base[k], title) if hex_ else self._ask_color(m.base[k], title)
        if new:
            m.base[k] = new
            self._fill_cats()
            self.redraw()

    def edit_gradient(self, k):
        tk, ttk = self.tk, self.ttk
        m = self.m
        s = m.tone_settings(k)
        top = tk.Toplevel(self.win)
        top.title(f"Градиент: {_clip(m.label(m.hue_side, k), 50)}")
        top.transient(self.win)
        top.resizable(False, False)
        pad = {"padx": 8, "pady": 4}
        v_own = tk.BooleanVar(value=k in m.override)
        v_mode = tk.StringVar(value=TONE_LABEL[s["mode"]])
        v_str = tk.DoubleVar(value=float(s["strength"]))
        v_rev = tk.BooleanVar(value=bool(s["reverse"]))
        ttk.Checkbutton(top, text="Свой градиент (иначе — по умолчанию)", variable=v_own).grid(
            row=0, column=0, columnspan=2, sticky="w", **pad)
        ttk.Label(top, text="Способ").grid(row=1, column=0, sticky="w", **pad)
        ttk.Combobox(top, textvariable=v_mode, state="readonly", width=36,
                     values=[t for _, t in TONE_MODES]).grid(row=1, column=1, **pad)
        ttk.Label(top, text="Сила").grid(row=2, column=0, sticky="w", **pad)
        ttk.Scale(top, from_=0.1, to=1.0, variable=v_str).grid(row=2, column=1, sticky="we", **pad)
        ttk.Checkbutton(top, text="Обратный порядок", variable=v_rev).grid(row=3, column=1, sticky="w", **pad)
        prev = tk.Canvas(top, height=26, highlightthickness=0, background="#ffffff")
        prev.grid(row=4, column=0, columnspan=2, sticky="we", **pad)

        def cur():
            return {"mode": next(a for a, t in TONE_MODES if t == v_mode.get()),
                    "strength": round(float(v_str.get()), 2), "reverse": bool(v_rev.get())}

        def show(*_):
            prev.delete("all")
            sset = cur() if v_own.get() else {kk: m.tone[kk] for kk in ("mode", "strength", "reverse")}
            n = len(m.tone_keys)
            w = max(prev.winfo_width(), 340) / max(n, 1)
            for i in range(n):
                col = tone_color(m.base[k], i, n, sset["mode"], sset["strength"], sset["reverse"],
                                 int(m.tone.get("seed", 1)), m.hue_keys.index(k))
                prev.create_rectangle(i * w, 0, (i + 1) * w, 26, fill=col, outline="")

        for v in (v_own, v_mode, v_str, v_rev):
            v.trace_add("write", show)

        def ok():
            if v_own.get():
                m.override[k] = cur()
            else:
                m.override.pop(k, None)
            top.destroy()
            self._fill_cats()
            self.redraw()

        fb = ttk.Frame(top)
        fb.grid(row=5, column=0, columnspan=2, sticky="e", **pad)
        ttk.Button(fb, text="Применить", command=ok).pack(side="left", padx=4)
        ttk.Button(fb, text="Отмена", command=top.destroy).pack(side="left")
        top.after(60, show)
        top.grab_set()

    def _load_state(self):
        from tkinter import filedialog, messagebox
        if not self.m:
            messagebox.showwarning("Нет матрицы", "Сначала постройте матрицу.", parent=self.win)
            return
        p = filedialog.askopenfilename(parent=self.win, title="Сохранённая легенда XLSX",
                                       initialdir=self.last_dir, filetypes=[("Excel", "*.xlsx")])
        if not p:
            return
        try:
            self.m.load_state(load_state_xlsx(p))
        except Exception as ex:
            messagebox.showerror("Ошибка", str(ex), parent=self.win)
            return
        t = self.m.tone
        self.v_hue.set(self.m.hue_side)
        self.v_tmode.set(TONE_LABEL[t["mode"]])
        self.v_str.set(t["strength"])
        self.v_rev.set(bool(t["reverse"]))
        self.v_seed.set(str(t["seed"]))
        self.v_pal.set(self.m.palette)
        self.v_hide.set(self.m.hide_absent)
        self._fill_cats()
        self.redraw()
        self.status.set(f"Раскраска загружена из {os.path.basename(p)}.")

    def save(self):
        from tkinter import messagebox
        if not self.m:
            messagebox.showwarning("Нет матрицы", "Сначала постройте матрицу.", parent=self.win)
            return
        out = self.v_out.get().strip() or default_out_path(self.v_path.get(), self.v_layer.get(),
                                                           self.root_dir)
        if not out.lower().endswith(".xlsx"):
            out += ".xlsx"
        try:
            save_xlsx(self.m, out)
        except PermissionError:
            messagebox.showerror("Ошибка записи", f"Файл занят (открыт в Excel?):\n{out}", parent=self.win)
            return
        except Exception as ex:
            messagebox.showerror("Ошибка записи", str(ex), parent=self.win)
            return
        self.v_out.set(out)
        self.status.set(f"Сохранено: {out}")
        log(f"Сохранено: {out}")

    def run(self):
        self.redraw()
        self.win.mainloop()


def _clip(s, n):
    s = str(s)
    return s if len(s) <= max(n, 4) else s[:max(n, 4) - 1] + "…"


# ===========================================================================
# 6. ЗАПУСК
# ===========================================================================

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    # Jupyter передаёт свои аргументы (-f kernel.json) — считаем их пустыми
    if argv and any(a.startswith("-f") or a.endswith(".json") for a in argv) and \
            not any(a.lower().endswith((".gpkg", ".shp", ".geojson", ".fgb", ".sqlite")) for a in argv):
        argv = []
    if not argv:
        log("09 - Матричная легенда. Рабочая папка: " + os.getcwd())
        try:
            app = App(os.getcwd())
        except Exception as e:
            log(f"Графический интерфейс недоступен ({e}). Используйте CLI: python 09_MatrixLegend.py --help")
            return
        app.run()
        return

    p = argparse.ArgumentParser(description="Матричная легенда (раскраска по умолчанию, без окна)")
    p.add_argument("path", help="слой полигонов")
    p.add_argument("--layer", default=None)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--fields", nargs=2, metavar=("ГОРИЗ", "ВЕРТ"), help="два поля: по горизонтали и по вертикали")
    g.add_argument("--field", help="одно поле с кодом")
    p.add_argument("--parse", default="first:1", help="first:k | last:k | sep:СИМВОЛ (для --field)")
    p.add_argument("--first-axis", choices=["v", "h"], default="v", help="ось 1-й части кода")
    p.add_argument("--names-h", default=None, help="CSV названий по горизонтали")
    p.add_argument("--names-v", default=None, help="CSV названий по вертикали")
    p.add_argument("--hue", choices=["h", "v"], default=None, help="ось цвета (по умолчанию — бо́льшая)")
    p.add_argument("--palette", default=DEFAULT_PALETTE, choices=list(PALETTES))
    p.add_argument("--tone", default="tint", choices=[k for k, _ in TONE_MODES])
    p.add_argument("--strength", type=float, default=0.75)
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--show-absent", action="store_true", help="раскрашивать и отсутствующие сочетания")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    cfg = {"path": a.path, "layer": a.layer, "names_h": a.names_h, "names_v": a.names_v}
    if a.fields:
        cfg.update(mode="two", h_field=a.fields[0], v_field=a.fields[1])
    else:
        meth, _, arg = a.parse.partition(":")
        if meth not in ("first", "last", "sep"):
            p.error("--parse: first:k, last:k или sep:СИМВОЛ")
        cfg.update(mode="one", field=a.field, method=meth, first_axis=a.first_axis,
                   k=int(arg) if meth != "sep" and arg else 1, sep=arg if meth == "sep" else "_")
    df = read_attrs(a.path, a.layer)
    m = MatrixModel.from_table(df, cfg)
    m.set_names(read_names_csv(a.names_h) if a.names_h else {},
                read_names_csv(a.names_v) if a.names_v else {})
    if a.hue:
        m.set_hue_side(a.hue)
    m.apply_palette(a.palette)
    m.tone.update(mode=a.tone, strength=a.strength, reverse=a.reverse, seed=a.seed)
    m.hide_absent = not a.show_absent
    out = a.out or default_out_path(a.path, a.layer)
    save_xlsx(m, out)
    log(f"Матрица {len(m.v_keys)} × {len(m.h_keys)}, сочетаний: {len(m.counts)}, "
        f"нераспознанных значений: {sum(m.skipped.values())}")
    log(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
