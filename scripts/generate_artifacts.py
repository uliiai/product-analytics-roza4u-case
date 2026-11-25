"""
Продуктовая аналитика кейса roza4u.ru («Собери свой букет»):
1. Модель юнит-экономики и анализ чувствительности (3 сценария).
2. Расчёт sample size, MDE и длительности A/B-теста для CR и ARPU.
3. Предэкспериментальная валидация дизайна A/B-теста (A/A-симуляция и оценка мощности).
4. Генерация SVG-графиков для README.md и исполненного Jupyter-ноутбука.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE_DIR / "images"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

NORM = NormalDist(0.0, 1.0)

# --- 1. Базовые параметры бизнеса (допущения на 1 месяц) ---
MONTHLY_VISITORS = 100_000
WEEKLY_VISITORS = MONTHLY_VISITORS * 7 / 30  # ~23 333 посетителя в неделю
BASELINE_CR = 0.03
BASELINE_AOV = 4_500.0
GROSS_MARGIN_RATE = 0.45  # валовая маржа до доп. затрат на сборку конструктора
EXTRA_ASSEMBLY_COST = 120.0  # доп. затраты на индивидуальную сборку/упаковку 1 кастомного букета
MVP_DEV_COST = 450_000.0  # разовая стоимость разработки MVP конструктора

BASELINE_ORDERS = MONTHLY_VISITORS * BASELINE_CR
BASELINE_REVENUE = BASELINE_ORDERS * BASELINE_AOV
BASELINE_GROSS_PROFIT = BASELINE_REVENUE * GROSS_MARGIN_RATE
BASELINE_ARPU = BASELINE_REVENUE / MONTHLY_VISITORS


def calc_scenarios() -> pd.DataFrame:
    """Расчёт юнит-экономики в 3 сценариях через долю пользователей конструктора (adoption)."""
    configs = [
        {
            "scenario": "Пессимистичный",
            "adoption": 0.15,
            "cr_builder": 0.038,
            "aov_builder": 4_900.0,
        },
        {
            "scenario": "Базовый",
            "adoption": 0.20,
            "cr_builder": 0.045,
            "aov_builder": 5_200.0,
        },
        {
            "scenario": "Оптимистичный",
            "adoption": 0.25,
            "cr_builder": 0.046,
            "aov_builder": 5_500.0,
        },
    ]

    rows = []
    for cfg in configs:
        a = cfg["adoption"]
        non_builder_users = MONTHLY_VISITORS * (1.0 - a)
        builder_users = MONTHLY_VISITORS * a

        orders_regular = non_builder_users * BASELINE_CR
        orders_builder = builder_users * cfg["cr_builder"]
        total_orders = orders_regular + orders_builder

        rev_regular = orders_regular * BASELINE_AOV
        rev_builder = orders_builder * cfg["aov_builder"]
        total_revenue = rev_regular + rev_builder

        gp_regular = rev_regular * GROSS_MARGIN_RATE
        gp_builder = rev_builder * GROSS_MARGIN_RATE - orders_builder * EXTRA_ASSEMBLY_COST
        total_gp = gp_regular + gp_builder

        overall_cr = total_orders / MONTHLY_VISITORS
        overall_aov = total_revenue / total_orders
        overall_arpu = total_revenue / MONTHLY_VISITORS

        delta_rev = total_revenue - BASELINE_REVENUE
        delta_gp = total_gp - BASELINE_GROSS_PROFIT
        payback_months = MVP_DEV_COST / delta_gp if delta_gp > 0 else float("inf")
        roi_6m = (delta_gp * 6 - MVP_DEV_COST) / MVP_DEV_COST

        rows.append(
            {
                "Сценарий": cfg["scenario"],
                "Adoption (%)": round(a * 100, 1),
                "CR в конструкторе (%)": round(cfg["cr_builder"] * 100, 2),
                "Общий CR (%)": round(overall_cr * 100, 2),
                "Δ CR (п.п.)": round((overall_cr - BASELINE_CR) * 100, 2),
                "Чек конструктора (₽)": int(cfg["aov_builder"]),
                "Общий AOV (₽)": round(overall_aov),
                "ARPU (₽)": round(overall_arpu, 2),
                "Выручка/мес (₽)": round(total_revenue),
                "Δ Выручки/мес (₽)": round(delta_rev),
                "Δ Валовой прибыли/мес (₽)": round(delta_gp),
                "Окупаемость MVP (мес.)": round(payback_months, 2),
                "ROI за 6 мес. (%)": round(roi_6m * 100),
            }
        )
    return pd.DataFrame(rows)


def sample_size_proportion(p1: float, p2: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Точный расчёт размера выборки на 1 группу для двухстороннего z-теста пропорций."""
    z_alpha = NORM.inv_cdf(1.0 - alpha / 2.0)
    z_beta = NORM.inv_cdf(power)
    p_bar = (p1 + p2) / 2.0
    num = (
        z_alpha * math.sqrt(2.0 * p_bar * (1.0 - p_bar))
        + z_beta * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    ) ** 2
    return math.ceil(num / ((p2 - p1) ** 2))


def mde_for_n(p1: float, n_per_group: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """Минимальный детектируемый эффект (абсолютный, в долях) при заданном размере группы."""
    lo, hi = 1e-5, 0.20
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if sample_size_proportion(p1, p1 + mid, alpha, power) > n_per_group:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def calc_mde_schedule() -> pd.DataFrame:
    """Таблица зависимости MDE (CR) от длительности A/B-теста (1–6 недель)."""
    rows = []
    for weeks in range(1, 7):
        total_users = round(WEEKLY_VISITORS * weeks)
        n_group = total_users // 2
        mde_abs = mde_for_n(BASELINE_CR, n_group)
        mde_rel = mde_abs / BASELINE_CR
        rows.append(
            {
                "Недели": weeks,
                "Всего пользователей": total_users,
                "В каждой группе": n_group,
                "MDE (п.п.)": round(mde_abs * 100, 2),
                "MDE (% отн.)": round(mde_rel * 100, 1),
                "Целевой CR (%)": round((BASELINE_CR + mde_abs) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def simulate_ab_validation(
    n_per_group: int = 30_400,
    p_ctrl: float = 0.03,
    p_test: float = 0.034,
    n_sims: int = 2_000,
    seed: int = 42,
) -> dict:
    """Монте-Карло валидация дизайна A/B-теста до запуска: A/A-тест (FPR) и оценка мощности (Power)."""
    rng = np.random.default_rng(seed)

    def run_z_test_batch(p_a: float, p_b: float) -> np.ndarray:
        x_a = rng.binomial(n_per_group, p_a, size=n_sims)
        x_b = rng.binomial(n_per_group, p_b, size=n_sims)
        hat_a = x_a / n_per_group
        hat_b = x_b / n_per_group
        p_pool = (x_a + x_b) / (2.0 * n_per_group)
        se = np.sqrt(p_pool * (1.0 - p_pool) * (2.0 / n_per_group))
        z_stat = np.abs(hat_b - hat_a) / np.where(se == 0, 1.0, se)
        # Двусторонний p-value через erfc
        p_vals = np.array([math.erfc(z / math.sqrt(2.0)) for z in z_stat])
        return p_vals

    aa_pvals = run_z_test_batch(p_ctrl, p_ctrl)
    ab_pvals = run_z_test_batch(p_ctrl, p_test)

    return {
        "n_per_group": n_per_group,
        "n_sims": n_sims,
        "aa_fpr": float(np.mean(aa_pvals < 0.05)),
        "ab_power": float(np.mean(ab_pvals < 0.05)),
        "aa_pvals": aa_pvals,
        "ab_pvals": ab_pvals,
    }


def save_svg_unit_economics(df: pd.DataFrame, path: Path) -> None:
    """Чистый SVG-график: прирост выручки и валовой прибыли в 3 сценариях."""
    width, height = 820, 420
    margin_l, margin_r, margin_t, margin_b = 85, 40, 72, 78
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b

    max_val = 3.2  # млн руб.
    scenarios = df["Сценарий"].tolist()
    rev_m = [v / 1e6 for v in df["Δ Выручки/мес (₽)"]]
    gp_m = [v / 1e6 for v in df["Δ Валовой прибыли/мес (₽)"]]
    paybacks = df["Окупаемость MVP (мес.)"].tolist()

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF" rx="12"/>',
        '<style>'
        'text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; }'
        '.title { font-size: 18px; font-weight: 700; fill: #222222; }'
        '.subtitle { font-size: 12px; fill: #666666; }'
        '.axis { font-size: 12px; fill: #555555; }'
        '.val { font-size: 12px; font-weight: 600; fill: #222222; }'
        '.badge { font-size: 11px; font-weight: 600; fill: #7A2E4D; }'
        '</style>',
        f'<text x="{margin_l}" y="30" class="title">Оценка месячного эффекта конструктора «Собери свой букет»</text>',
        f'<text x="{margin_l}" y="50" class="subtitle">Сравнение 3 сценариев (база: 100 000 чел./мес, CR 3.0%, AOV 4 500 ₽, маржа 45%, MVP 450 тыс. ₽)</text>',
    ]

    # Gridlines
    for tick in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        y = margin_t + chart_h - (tick / max_val) * chart_h
        parts.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width - margin_r}" y2="{y:.1f}" stroke="#EEEEEE" stroke-width="1"/>')
        parts.append(f'<text x="{margin_l - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{tick:.1f} млн</text>')

    # Legend
    leg_x = width - margin_r - 290
    parts.append(f'<rect x="{leg_x}" y="56" width="12" height="12" fill="#E397B4" rx="2"/>')
    parts.append(f'<text x="{leg_x + 18}" y="66" class="axis">Δ Выручки / мес</text>')
    parts.append(f'<rect x="{leg_x + 135}" y="56" width="12" height="12" fill="#A63A66" rx="2"/>')
    parts.append(f'<text x="{leg_x + 153}" y="66" class="axis">Δ Валовой прибыли / мес</text>')

    group_w = chart_w / len(scenarios)
    bar_w = 56
    gap = 12

    for i, sc in enumerate(scenarios):
        cx = margin_l + group_w * (i + 0.5)
        x1 = cx - bar_w - gap / 2
        x2 = cx + gap / 2

        h1 = (rev_m[i] / max_val) * chart_h
        y1 = margin_t + chart_h - h1
        h2 = (gp_m[i] / max_val) * chart_h
        y2 = margin_t + chart_h - h2

        parts.append(f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{bar_w}" height="{h1:.1f}" fill="#E397B4" rx="4"/>')
        parts.append(f'<text x="{x1 + bar_w/2:.1f}" y="{y1 - 6:.1f}" text-anchor="middle" class="val">+{rev_m[i]:.2f} млн</text>')

        parts.append(f'<rect x="{x2:.1f}" y="{y2:.1f}" width="{bar_w}" height="{h2:.1f}" fill="#A63A66" rx="4"/>')
        parts.append(f'<text x="{x2 + bar_w/2:.1f}" y="{y2 - 6:.1f}" text-anchor="middle" class="val">+{gp_m[i]:.2f} млн</text>')

        parts.append(f'<text x="{cx:.1f}" y="{margin_t + chart_h + 22}" text-anchor="middle" class="val">{sc}</text>')
        parts.append(
            f'<rect x="{cx - 66:.1f}" y="{margin_t + chart_h + 32}" width="132" height="22" fill="#FCEBF1" rx="11"/>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{margin_t + chart_h + 47}" text-anchor="middle" class="badge">Окупаемость: {paybacks[i]:.1f} мес.</text>'
        )

    # Bottom axis line
    y0 = margin_t + chart_h
    parts.append(f'<line x1="{margin_l}" y1="{y0}" x2="{width - margin_r}" y2="{y0}" stroke="#CCCCCC" stroke-width="1.5"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def save_svg_mde_curve(df_mde: pd.DataFrame, path: Path) -> None:
    """SVG-график: зависимость MDE по CR от длительности A/B-теста (1–6 недель)."""
    width, height = 820, 400
    margin_l, margin_r, margin_t, margin_b = 80, 40, 72, 65
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b

    weeks = df_mde["Недели"].tolist()
    mdes = df_mde["MDE (п.п.)"].tolist()
    max_y = 0.80

    def x_pos(w: float) -> float:
        return margin_l + ((w - 1) / 5.0) * chart_w

    def y_pos(m: float) -> float:
        return margin_t + chart_h - (m / max_y) * chart_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF" rx="12"/>',
        '<style>'
        'text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; }'
        '.title { font-size: 18px; font-weight: 700; fill: #222222; }'
        '.subtitle { font-size: 12px; fill: #666666; }'
        '.axis { font-size: 12px; fill: #555555; }'
        '.val { font-size: 12px; font-weight: 700; fill: #A63A66; }'
        '.note { font-size: 11px; fill: #444444; }'
        '</style>',
        f'<text x="{margin_l}" y="30" class="title">Зависимость MDE (конверсия в покупку) от длительности A/B-теста</text>',
        f'<text x="{margin_l}" y="50" class="subtitle">Трафик 100 000 чел./мес (~23 333/нед), базовый CR = 3.0%, α = 5%, мощность 80% (разделение 50/50)</text>',
    ]

    for tick in [0.0, 0.2, 0.4, 0.6, 0.8]:
        y = y_pos(tick)
        parts.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width - margin_r}" y2="{y:.1f}" stroke="#EEEEEE" stroke-width="1"/>')
        parts.append(f'<text x="{margin_l - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">+{tick:.1f} п.п.</text>')

    # Reference dashed lines for Base (+0.30 pp) and Optimistic (+0.40 pp)
    for ref_val, label, color in [
        (0.40, "MDE +0.40 п.п. (~3 нед., ~30 400 / гр.)", "#C25982"),
        (0.30, "Базовый сценарий +0.30 п.п. (~5 нед., ~53 200 / гр.)", "#2E7D6F"),
    ]:
        yr = y_pos(ref_val)
        parts.append(
            f'<line x1="{margin_l}" y1="{yr:.1f}" x2="{width - margin_r}" y2="{yr:.1f}" '
            f'stroke="{color}" stroke-width="1.5" stroke-dasharray="6,4"/>'
        )
        parts.append(f'<text x="{width - margin_r - 4}" y="{yr - 6:.1f}" text-anchor="end" class="note" fill="{color}">{label}</text>')

    # Polyline
    pts = " ".join(f"{x_pos(w):.1f},{y_pos(m):.1f}" for w, m in zip(weeks, mdes))
    parts.append(f'<polyline fill="none" stroke="#A63A66" stroke-width="3" points="{pts}"/>')

    for w, m in zip(weeks, mdes):
        x, y = x_pos(w), y_pos(m)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="#A63A66" stroke="#FFFFFF" stroke-width="2"/>')
        dy = -12 if w != 3 else -14
        parts.append(f'<text x="{x:.1f}" y="{y + dy:.1f}" text-anchor="middle" class="val">+{m:.2f} п.п.</text>')
        parts.append(f'<text x="{x:.1f}" y="{margin_t + chart_h + 24}" text-anchor="middle" class="axis">{w} нед.</text>')

    parts.append(f'<line x1="{margin_l}" y1="{margin_t + chart_h}" x2="{width - margin_r}" y2="{margin_t + chart_h}" stroke="#CCCCCC" stroke-width="1.5"/>')
    parts.append(f'<text x="{margin_l + chart_w / 2:.1f}" y="{height - 14}" text-anchor="middle" class="axis">Длительность эксперимента (полные недельные циклы)</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def save_svg_aa_validation(sim: dict, path: Path) -> None:
    """SVG-график: распределение p-value в A/A-симуляции (равномерность) и A/B-мощность."""
    width, height = 820, 360
    margin_l, margin_r, margin_t, margin_b = 60, 30, 68, 52
    panel_gap = 50
    panel_w = (width - margin_l - margin_r - panel_gap) / 2
    panel_h = height - margin_t - margin_b

    bins = np.linspace(0, 1, 11)
    aa_counts, _ = np.histogram(sim["aa_pvals"], bins=bins)
    aa_pct = aa_counts / sim["n_sims"] * 100.0
    ab_counts, _ = np.histogram(sim["ab_pvals"], bins=bins)
    ab_pct = ab_counts / sim["n_sims"] * 100.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF" rx="12"/>',
        '<style>'
        'text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; }'
        '.title { font-size: 17px; font-weight: 700; fill: #222222; }'
        '.subtitle { font-size: 12px; fill: #666666; }'
        '.panel { font-size: 13px; font-weight: 700; fill: #333333; }'
        '.axis { font-size: 11px; fill: #555555; }'
        '</style>',
        f'<text x="{margin_l}" y="28" class="title">Предэкспериментальная валидация дизайна A/B-теста (Монте-Карло, {sim["n_sims"]} симуляций)</text>',
        f'<text x="{margin_l}" y="47" class="subtitle">n = {sim["n_per_group"]:,} чел./группу · A/A-тест контролирует α = 5% (FPR = {sim["aa_fpr"]*100:.2f}%), A/B-симуляция подтверждает мощность ({sim["ab_power"]*100:.1f}%)</text>',
    ]

    # Left panel: A/A p-value distribution (max y = 20%)
    lx = margin_l
    parts.append(f'<text x="{lx}" y="{margin_t - 6}" class="panel">A/A-симуляция (H0 верна: CR 3.0% vs 3.0%)</text>')
    bar_w = panel_w / 10.0 - 4
    y_exp = margin_t + panel_h - (10.0 / 20.0) * panel_h
    parts.append(f'<line x1="{lx}" y1="{y_exp:.1f}" x2="{lx + panel_w:.1f}" y2="{y_exp:.1f}" stroke="#2E7D6F" stroke-width="1.5" stroke-dasharray="5,4"/>')
    parts.append(f'<text x="{lx + panel_w:.1f}" y="{y_exp - 5:.1f}" text-anchor="end" class="axis" fill="#2E7D6F">Равномерное U(0,1): 10%</text>')

    for i, pct in enumerate(aa_pct):
        bx = lx + i * (panel_w / 10.0) + 2
        bh = min(pct / 20.0, 1.0) * panel_h
        by = margin_t + panel_h - bh
        parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="#E397B4" rx="2"/>')
    parts.append(f'<line x1="{lx}" y1="{margin_t + panel_h}" x2="{lx + panel_w:.1f}" y2="{margin_t + panel_h}" stroke="#CCCCCC" stroke-width="1.5"/>')
    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        tx = lx + tick * panel_w
        parts.append(f'<text x="{tx:.1f}" y="{margin_t + panel_h + 16}" text-anchor="middle" class="axis">{tick:.1f}</text>')
    parts.append(f'<text x="{lx + panel_w / 2:.1f}" y="{height - 12}" text-anchor="middle" class="axis">p-value (FPR при α=0.05: {sim["aa_fpr"]*100:.2f}%)</text>')

    # Right panel: A/B p-value distribution under H1 (max y = 100%)
    rx = margin_l + panel_w + panel_gap
    parts.append(f'<text x="{rx}" y="{margin_t - 6}" class="panel">A/B-симуляция (H1: CR 3.0% → 3.4%)</text>')
    for i, pct in enumerate(ab_pct):
        bx = rx + i * (panel_w / 10.0) + 2
        bh = min(pct / 100.0, 1.0) * panel_h
        by = margin_t + panel_h - bh
        fill = "#A63A66" if i == 0 else "#E397B4"
        parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{fill}" rx="2"/>')
        if i == 0:
            parts.append(f'<text x="{bx + bar_w/2:.1f}" y="{by - 5:.1f}" text-anchor="middle" class="axis" font-weight="700" fill="#A63A66">{pct:.1f}%</text>')
    parts.append(f'<line x1="{rx}" y1="{margin_t + panel_h}" x2="{rx + panel_w:.1f}" y2="{margin_t + panel_h}" stroke="#CCCCCC" stroke-width="1.5"/>')
    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        tx = rx + tick * panel_w
        parts.append(f'<text x="{tx:.1f}" y="{margin_t + panel_h + 16}" text-anchor="middle" class="axis">{tick:.1f}</text>')
    parts.append(f'<text x="{rx + panel_w / 2:.1f}" y="{height - 12}" text-anchor="middle" class="axis">p-value (Мощность при α=0.05: {sim["ab_power"]*100:.1f}%)</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def build_executed_notebook(df_sc: pd.DataFrame, df_mde: pd.DataFrame, sim: dict, path: Path) -> None:
    """Создаёт воспроизводимый Jupyter Notebook с предвыполненными таблицами и графиками Plotly."""

    def md_cell(source: str) -> dict:
        return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}

    def code_cell(source: str, stdout: str = "", html_table: str | None = None, exec_count: int = 1) -> dict:
        outputs = []
        if stdout:
            outputs.append({"output_type": "stream", "name": "stdout", "text": stdout.splitlines(keepends=True)})
        if html_table:
            outputs.append(
                {
                    "output_type": "execute_result",
                    "execution_count": exec_count,
                    "metadata": {},
                    "data": {
                        "text/plain": ["DataFrame"],
                        "text/html": html_table.splitlines(keepends=True),
                    },
                }
            )
        return {
            "cell_type": "code",
            "execution_count": exec_count,
            "metadata": {},
            "outputs": outputs,
            "source": source.splitlines(keepends=True),
        }

    n_opt = sample_size_proportion(0.03, 0.034)
    n_base = sample_size_proportion(0.03, 0.033)
    n_pess = sample_size_proportion(0.03, 0.0312)

    cells = [
        md_cell(
            "# Продуктовый кейс `roza4u.ru`: юнит-экономика и дизайн A/B-теста конструктора букетов\n\n"
            "**Автор:** Юлия Богодяж  \n"
            "**Цель ноутбука:** провести количественную оценку внедрения функции «Собери свой букет» "
            "(модель юнит-экономики в 3 сценариях), рассчитать параметры A/B-теста (`sample size`, `MDE`, длительность) "
            "и выполнить предэкспериментальную валидацию статистического дизайна (A/A-симуляция и оценка мощности).\n\n"
            "> **Важно:** все симуляции в разделе 3 носят характер *предэкспериментальной проверки дизайна* "
            "(до запуска реального A/B-теста в продакшене)."
        ),
        code_cell(
            "import math\n"
            "from statistics import NormalDist\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import plotly.graph_objects as go\n"
            "from plotly.subplots import make_subplots\n\n"
            "NORM = NormalDist(0.0, 1.0)\n\n"
            "# Базовые параметры бизнеса (допущения на 1 месяц)\n"
            "MONTHLY_VISITORS = 100_000\n"
            "WEEKLY_VISITORS = MONTHLY_VISITORS * 7 / 30  # ~23 333 пользователя/нед\n"
            "BASELINE_CR = 0.03\n"
            "BASELINE_AOV = 4_500.0\n"
            "GROSS_MARGIN_RATE = 0.45\n"
            "EXTRA_ASSEMBLY_COST = 120.0   # доп. затраты на индивидуальную сборку/упаковку\n"
            "MVP_DEV_COST = 450_000.0      # оценка разовой разработки MVP\n\n"
            "baseline_orders = MONTHLY_VISITORS * BASELINE_CR\n"
            "baseline_revenue = baseline_orders * BASELINE_AOV\n"
            "baseline_gp = baseline_revenue * GROSS_MARGIN_RATE\n"
            "print(f'Базовая выручка: {baseline_revenue:,.0f} ₽/мес | Валовая прибыль: {baseline_gp:,.0f} ₽/мес')\n",
            stdout=f"Базовая выручка: {BASELINE_REVENUE:,.0f} ₽/мес | Валовая прибыль: {BASELINE_GROSS_PROFIT:,.0f} ₽/мес\n",
            exec_count=1,
        ),
        md_cell(
            "## 1. Модель юнит-экономики и анализ чувствительности (3 сценария)\n\n"
            "Вместо допущения, что конструктор букетов мгновенно изменит метрики у 100% посетителей сайта, "
            "эффект декомпозируется через **долю пользователей фичи (`adoption`)**:\n\n"
            "- Посетители, не открывшие конструктор ($1 - a$), сохраняют базовые `CR = 3.0%` и `AOV = 4 500 ₽`.\n"
            "- Посетители, открывшие конструктор ($a$), формируют отдельный поток заказов с целевыми `CR_builder` и `AOV_builder`, "
            "причём на каждый кастомный букет закладываются дополнительные операционные затраты `120 ₽` (время флориста и индивидуальная упаковка)."
        ),
        code_cell(
            "configs = [\n"
            "    {'scenario': 'Пессимистичный', 'adoption': 0.15, 'cr_builder': 0.038, 'aov_builder': 4_900.0},\n"
            "    {'scenario': 'Базовый',        'adoption': 0.20, 'cr_builder': 0.045, 'aov_builder': 5_200.0},\n"
            "    {'scenario': 'Оптимистичный',  'adoption': 0.25, 'cr_builder': 0.046, 'aov_builder': 5_500.0},\n"
            "]\n\n"
            "rows = []\n"
            "for cfg in configs:\n"
            "    a = cfg['adoption']\n"
            "    orders_reg = MONTHLY_VISITORS * (1 - a) * BASELINE_CR\n"
            "    orders_bld = MONTHLY_VISITORS * a * cfg['cr_builder']\n"
            "    tot_orders = orders_reg + orders_bld\n"
            "    tot_rev = orders_reg * BASELINE_AOV + orders_bld * cfg['aov_builder']\n"
            "    tot_gp = tot_rev * GROSS_MARGIN_RATE - orders_bld * EXTRA_ASSEMBLY_COST\n"
            "    delta_rev = tot_rev - baseline_revenue\n"
            "    delta_gp = tot_gp - baseline_gp\n"
            "    rows.append({\n"
            "        'Сценарий': cfg['scenario'],\n"
            "        'Adoption (%)': round(a * 100, 1),\n"
            "        'CR в конструкторе (%)': round(cfg['cr_builder'] * 100, 2),\n"
            "        'Общий CR (%)': round(tot_orders / MONTHLY_VISITORS * 100, 2),\n"
            "        'Δ CR (п.п.)': round((tot_orders / MONTHLY_VISITORS - BASELINE_CR) * 100, 2),\n"
            "        'Чек конструктора (₽)': int(cfg['aov_builder']),\n"
            "        'Общий AOV (₽)': round(tot_rev / tot_orders),\n"
            "        'ARPU (₽)': round(tot_rev / MONTHLY_VISITORS, 2),\n"
            "        'Выручка/мес (₽)': round(tot_rev),\n"
            "        'Δ Выручки/мес (₽)': round(delta_rev),\n"
            "        'Δ Валовой прибыли/мес (₽)': round(delta_gp),\n"
            "        'Окупаемость MVP (мес.)': round(MVP_DEV_COST / delta_gp, 2),\n"
            "        'ROI за 6 мес. (%)': round((delta_gp * 6 - MVP_DEV_COST) / MVP_DEV_COST * 100),\n"
            "    })\n\n"
            "df_scenarios = pd.DataFrame(rows)\n"
            "df_scenarios\n",
            html_table=df_sc.to_html(index=False),
            exec_count=2,
        ),
        code_cell(
            "fig_ue = go.Figure()\n"
            "fig_ue.add_bar(name='Δ Выручки / мес (млн ₽)', x=df_scenarios['Сценарий'], y=df_scenarios['Δ Выручки/мес (₽)'] / 1e6, marker_color='#E397B4')\n"
            "fig_ue.add_bar(name='Δ Валовой прибыли / мес (млн ₽)', x=df_scenarios['Сценарий'], y=df_scenarios['Δ Валовой прибыли/мес (₽)'] / 1e6, marker_color='#A63A66')\n"
            "fig_ue.update_layout(\n"
            "    title='Прирост выручки и валовой прибыли по сценариям',\n"
            "    yaxis_title='Млн ₽ / мес',\n"
            "    barmode='group',\n"
            "    template='plotly_white',\n"
            ")\n"
            "fig_ue.show()\n",
            exec_count=3,
        ),
        md_cell(
            "## 2. Расчёт размера выборки (Sample Size), MDE и длительности A/B-теста\n\n"
            "Для ключевой бинарной метрики **CR в оплаченный заказ** (двусторонний z-тест пропорций, $\\alpha = 5\\%$, мощность $80\\%$):"
        ),
        code_cell(
            "def sample_size_proportion(p1: float, p2: float, alpha: float = 0.05, power: float = 0.80) -> int:\n"
            "    z_alpha = NORM.inv_cdf(1.0 - alpha / 2.0)\n"
            "    z_beta = NORM.inv_cdf(power)\n"
            "    p_bar = (p1 + p2) / 2.0\n"
            "    num = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2\n"
            "    return math.ceil(num / ((p2 - p1) ** 2))\n\n"
            "for name, target_cr in [('Оптимистичный (+0.40 п.п.)', 0.034), ('Базовый (+0.30 п.п.)', 0.033), ('Пессимистичный (+0.12 п.п.)', 0.0312)]:\n"
            "    n = sample_size_proportion(BASELINE_CR, target_cr)\n"
            "    weeks = (2 * n) / WEEKLY_VISITORS\n"
            "    print(f'{name:28s}: n = {n:>7,} чел./группу | всего {2*n:>7,} | длительность ≈ {weeks:.1f} нед.')\n",
            stdout=(
                f"{'Оптимистичный (+0.40 п.п.)':28s}: n = {n_opt:>7,} чел./группу | всего {2*n_opt:>7,} | длительность ≈ {(2*n_opt)/WEEKLY_VISITORS:.1f} нед.\n"
                f"{'Базовый (+0.30 п.п.)':28s}: n = {n_base:>7,} чел./группу | всего {2*n_base:>7,} | длительность ≈ {(2*n_base)/WEEKLY_VISITORS:.1f} нед.\n"
                f"{'Пессимистичный (+0.12 п.п.)':28s}: n = {n_pess:>7,} чел./группу | всего {2*n_pess:>7,} | длительность ≈ {(2*n_pess)/WEEKLY_VISITORS:.1f} нед.\n"
            ),
            exec_count=4,
        ),
        code_cell(
            "def mde_for_n(p1: float, n_per_group: float, alpha: float = 0.05, power: float = 0.80) -> float:\n"
            "    lo, hi = 1e-5, 0.20\n"
            "    for _ in range(60):\n"
            "        mid = (lo + hi) / 2.0\n"
            "        if sample_size_proportion(p1, p1 + mid, alpha, power) > n_per_group:\n"
            "            lo = mid\n"
            "        else:\n"
            "            hi = mid\n"
            "    return (lo + hi) / 2.0\n\n"
            "mde_rows = []\n"
            "for w in range(1, 7):\n"
            "    tot = round(WEEKLY_VISITORS * w)\n"
            "    n_g = tot // 2\n"
            "    mde_abs = mde_for_n(BASELINE_CR, n_g)\n"
            "    mde_rows.append({\n"
            "        'Недели': w,\n"
            "        'Всего пользователей': tot,\n"
            "        'В каждой группе': n_g,\n"
            "        'MDE (п.п.)': round(mde_abs * 100, 2),\n"
            "        'MDE (% отн.)': round(mde_abs / BASELINE_CR * 100, 1),\n"
            "        'Целевой CR (%)': round((BASELINE_CR + mde_abs) * 100, 2),\n"
            "    })\n"
            "df_mde = pd.DataFrame(mde_rows)\n"
            "df_mde\n",
            html_table=df_mde.to_html(index=False),
            exec_count=5,
        ),
        code_cell(
            "fig_mde = go.Figure()\n"
            "fig_mde.add_scatter(x=df_mde['Недели'], y=df_mde['MDE (п.п.)'], mode='lines+markers+text',\n"
            "                    text=[f'+{v:.2f}' for v in df_mde['MDE (п.п.)']], textposition='top center',\n"
            "                    line=dict(color='#A63A66', width=3), name='MDE (п.п.)')\n"
            "fig_mde.add_hline(y=0.40, line_dash='dash', line_color='#C25982', annotation_text='MDE +0.40 п.п. (~3 нед.)')\n"
            "fig_mde.add_hline(y=0.30, line_dash='dash', line_color='#2E7D6F', annotation_text='Базовый эффект +0.30 п.п. (~5 нед.)')\n"
            "fig_mde.update_layout(title='MDE по конверсии в зависимости от длительности теста',\n"
            "                      xaxis_title='Длительность (недели)', yaxis_title='MDE (п.п.)', template='plotly_white')\n"
            "fig_mde.show()\n",
            exec_count=6,
        ),
        md_cell(
            "## 3. Предэкспериментальная валидация дизайна (A/A-симуляция и оценка мощности)\n\n"
            "Перед запуском эксперимента проверяем корректность статистического критерия методом Монте-Карло:\n"
            "1. **A/A-симуляция ($H_0$ верна):** распределение p-value должно быть равномерным $U(0, 1)$, а доля ложноположительных срабатываний (FPR) — близка к $\\alpha = 5\\%$.\n"
            "2. **A/B-симуляция ($H_1$ верна: CR 3.0% → 3.4% при $n = 30\\,400$):** эмпирическая мощность должна подтверждать расчётные $\\approx 80\\%$."
        ),
        code_cell(
            "rng = np.random.default_rng(42)\n"
            "N_GROUP = 30_400\n"
            "N_SIMS = 2_000\n\n"
            "def run_z_batch(p_a: float, p_b: float) -> np.ndarray:\n"
            "    x_a = rng.binomial(N_GROUP, p_a, size=N_SIMS)\n"
            "    x_b = rng.binomial(N_GROUP, p_b, size=N_SIMS)\n"
            "    p_pool = (x_a + x_b) / (2 * N_GROUP)\n"
            "    se = np.sqrt(p_pool * (1 - p_pool) * (2 / N_GROUP))\n"
            "    z = np.abs(x_b / N_GROUP - x_a / N_GROUP) / se\n"
            "    return np.array([math.erfc(val / math.sqrt(2)) for val in z])\n\n"
            "aa_pvals = run_z_batch(0.03, 0.03)\n"
            "ab_pvals = run_z_batch(0.03, 0.034)\n\n"
            "print(f'A/A-симуляция — эмпирический FPR (α=5%): {np.mean(aa_pvals < 0.05)*100:.2f}%')\n"
            "print(f'A/B-симуляция — эмпирическая мощность (цель 80%): {np.mean(ab_pvals < 0.05)*100:.1f}%')\n",
            stdout=(
                f"A/A-симуляция — эмпирический FPR (α=5%): {sim['aa_fpr']*100:.2f}%\n"
                f"A/B-симуляция — эмпирическая мощность (цель 80%): {sim['ab_power']*100:.1f}%\n"
            ),
            exec_count=7,
        ),
        code_cell(
            "fig_val = make_subplots(rows=1, cols=2, subplot_titles=('A/A-симуляция (p-value, H0)', 'A/B-симуляция (p-value, H1: +0.40 п.п.)'))\n"
            "fig_val.add_histogram(x=aa_pvals, nbinsx=10, histnorm='percent', marker_color='#E397B4', row=1, col=1, showlegend=False)\n"
            "fig_val.add_histogram(x=ab_pvals, nbinsx=10, histnorm='percent', marker_color='#A63A66', row=1, col=2, showlegend=False)\n"
            "fig_val.update_layout(title='Валидация статистического критерия до запуска эксперимента', template='plotly_white')\n"
            "fig_val.show()\n",
            exec_count=8,
        ),
        md_cell(
            "## Выводы\n\n"
            "1. **Экономика фичи устойчива даже в пессимистичном сценарии:** при adoption 15% и умеренном росте CR до 3.12% "
            "дополнительная валовая прибыль составляет **+277 тыс. ₽/мес**, а MVP за 450 тыс. ₽ окупается за **~1.6 месяца**. "
            "В базовом сценарии (adoption 20%, общий CR 3.30%) прирост валовой прибыли достигает **+783 тыс. ₽/мес** (окупаемость ~0.6 мес.).\n"
            "2. **Длительность A/B-теста:** при трафике 100 000 пользователей/мес тест на ~30 400 пользователей в группе (MDE +0.40 п.п.) "
            "требует **3 полных недели** (а не 2). Чтобы уверенно поймать эффект базового сценария (+0.30 п.п.), понадобится **5 недель** "
            "либо снижение дисперсии через **CUPED** по ключевой денежной метрике `ARPU` (с контролем выбросов и поправкой Холма при проверке нескольких метрик).\n"
            "3. **Критерий валидирован:** A/A-симуляция подтвердила контроль ошибки I рода на уровне ~5%, а A/B-симуляция — расчётную мощность ~80%."
        ),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    df_sc = calc_scenarios()
    df_mde = calc_mde_schedule()
    sim = simulate_ab_validation()

    save_svg_unit_economics(df_sc, IMAGES_DIR / "unit_economics_scenarios.svg")
    save_svg_mde_curve(df_mde, IMAGES_DIR / "ab_mde_duration.svg")
    save_svg_aa_validation(sim, IMAGES_DIR / "aa_power_validation.svg")
    build_executed_notebook(df_sc, df_mde, sim, NOTEBOOKS_DIR / "ab_test_and_unit_economics.ipynb")

    print("=== UNIT ECONOMICS (3 SCENARIOS) ===")
    print(df_sc.to_string(index=False))
    print("\n=== SAMPLE SIZE BY SCENARIO ===")
    for name, target in [("Optimistic +0.40pp", 0.034), ("Base +0.30pp", 0.033), ("Pessimistic +0.12pp", 0.0312)]:
        n = sample_size_proportion(BASELINE_CR, target)
        print(f"{name:22s}: n={n:,}/group, total={2*n:,}, weeks={(2*n)/WEEKLY_VISITORS:.2f}")
    print("\n=== MDE SCHEDULE (1-6 WEEKS) ===")
    print(df_mde.to_string(index=False))
    print(f"\n=== MONTE CARLO VALIDATION (n={sim['n_per_group']:,}, sims={sim['n_sims']}) ===")
    print(f"AA FPR: {sim['aa_fpr']*100:.2f}% | AB Power: {sim['ab_power']*100:.1f}%")


if __name__ == "__main__":
    main()
