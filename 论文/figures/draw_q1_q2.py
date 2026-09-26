from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(__file__).resolve().parent
RESULTS = ROOT / "结果"
sys.path.insert(0, str(ROOT / "代码"))
from core import load_aircraft_types  # noqa: E402


COLORS = {"A": "#0072B2", "B": "#D55E00", "C": "#008A5B"}
MARKERS = {"A": "o", "B": "s", "C": "^"}
LINESTYLES = {"A": "-", "B": "--", "C": "-."}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 9.6,
            "axes.labelsize": 9.6,
            "axes.titlesize": 10,
            "xtick.labelsize": 9.6,
            "ytick.labelsize": 9.6,
            "legend.fontsize": 9.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def read_result(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name, encoding="utf-8-sig")


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf", metadata={"Creator": "draw_q1_q2.py"})
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, metadata={"Software": "draw_q1_q2.py"})
    plt.close(fig)


def draw_q1_payload() -> None:
    data = read_result("Q1_最大安全载荷.csv").sort_values("服务区编号")
    aircraft = load_aircraft_types()
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(6.55, 3.45))

    for model in "ABC":
        values = data[f"{model}最大安全载荷kg"].astype(float)
        capacity = aircraft[model].max_payload_kg
        ax.plot(
            x,
            values,
            color=COLORS[model],
            marker=MARKERS[model],
            linestyle=LINESTYLES[model],
            linewidth=1.4,
            markersize=4.2,
            label=f"{model}型最大安全载荷",
        )
        ax.axhline(capacity, color=COLORS[model], linewidth=0.75, linestyle=":", alpha=0.7)

    ax.set_xticks(x, data["服务区编号"], rotation=45, ha="right")
    ax.set_ylabel("最大安全载荷（kg）")
    ax.set_xlabel("服务区")
    ax.set_ylim(0, max(aircraft[m].max_payload_kg for m in "ABC") * 1.12)
    ax.grid(axis="y", color="#D7DCE0", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#555555", linestyle=":", linewidth=1.0,
                          label="机体载荷上限"))
    labels.append("机体载荷上限")
    ax.legend(handles, labels, ncol=2, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.29))
    save_figure(fig, "q1_payload_by_service")


def draw_q1_sensitivity() -> None:
    data = read_result("Q1_全局方案灵敏度.csv")
    feasible = data["可行性"].eq("可行")
    rho = data["返航安全余量rho"].astype(float) * 100
    fig, axes = plt.subplots(3, 1, figsize=(6.55, 4.75), sharex=True)
    columns = [
        ("架次数", "架次", "#0072B2", "o", "-"),
        ("总能耗kWh", "总能耗（kWh）", "#D55E00", "s", "--"),
        ("累计作业时间s", "累计作业时间（s）", "#008A5B", "^", "-."),
    ]
    for ax, (column, label, color, marker, linestyle) in zip(axes, columns):
        y = pd.to_numeric(data[column], errors="coerce")
        ax.plot(rho[feasible], y[feasible], color=color, marker=marker,
                linestyle=linestyle, linewidth=1.5, markersize=4.5)
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#D7DCE0", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axvline(40, color="#555555", linewidth=0.9, linestyle=":")
    axes[-1].set_xlabel("返航安全余量（%）")
    axes[-1].set_xticks(rho, [f"{v:g}" for v in rho])
    axes[0].annotate("40%：不可行", xy=(40, 0.78), xycoords=("data", "axes fraction"),
                     xytext=(-8, 0), textcoords="offset points", ha="right", va="center")
    fig.subplots_adjust(hspace=0.28)
    save_figure(fig, "q1_global_reserve_sensitivity")


def draw_q2_gantt() -> None:
    trips = read_result("Q2_运输架次.csv")
    batteries = read_result("Q2_电池资源台账.csv")
    makespan = pd.to_numeric(trips["返回O01时刻s"]).max()
    x_max = max(pd.to_numeric(batteries["充电完成s"]).max(), makespan) + 250

    drone_rows = []
    for model in "ABC":
        names = sorted(trips.loc[trips["机型编号"].eq(model), "无人机编号"].unique())
        drone_rows.extend((model, name) for name in names)
    battery_rows = []
    for model in "ABC":
        names = sorted(batteries.loc[batteries["机型编号"].eq(model), "电池编号"].unique())
        battery_rows.extend((model, name) for name in names)

    fig, (ax_drone, ax_battery) = plt.subplots(
        2, 1, figsize=(6.55, 7.25), sharex=True,
        gridspec_kw={"height_ratios": [len(drone_rows), len(battery_rows)]},
    )
    lane_h = 0.64

    for y, (model, name) in enumerate(drone_rows):
        rows = trips.loc[trips["无人机编号"].eq(name)]
        for _, row in rows.iterrows():
            start = float(row["开始时刻s"])
            end = float(row["返回O01时刻s"])
            ax_drone.barh(y, end - start, left=start, height=lane_h,
                          color=COLORS[model], edgecolor="#252525", linewidth=0.45)
    ax_drone.set_yticks(range(len(drone_rows)), [f"{name}（{model}型）" for model, name in drone_rows])
    ax_drone.invert_yaxis()
    ax_drone.set_title("实体无人机任务占用", loc="left", pad=5)
    ax_drone.set_ylabel("无人机")

    for y, (model, name) in enumerate(battery_rows):
        rows = batteries.loc[batteries["电池编号"].eq(name)]
        for _, row in rows.iterrows():
            start = float(row["占用开始s"])
            returned = float(row["任务返回s"])
            charged = float(row["充电完成s"])
            ax_battery.barh(y, returned - start, left=start, height=lane_h,
                            color=COLORS[model], edgecolor="#252525", linewidth=0.45)
            ax_battery.barh(y, charged - returned, left=returned, height=lane_h,
                            color="white", edgecolor=COLORS[model], linewidth=0.8, hatch="////")
    ax_battery.set_yticks(range(len(battery_rows)), [name for _, name in battery_rows])
    ax_battery.invert_yaxis()
    ax_battery.set_title("共享电池任务占用与充电", loc="left", pad=5)
    ax_battery.set_ylabel("电池")
    ax_battery.set_xlabel("时间（s）")

    for ax in (ax_drone, ax_battery):
        ax.set_xlim(0, x_max)
        ax.axvline(makespan, color="#222222", linestyle="--", linewidth=1.0)
        ax.grid(axis="x", color="#D7DCE0", linewidth=0.55)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=3)
    legend = [
        Patch(facecolor=COLORS[m], edgecolor="#252525", label=f"{m}型") for m in "ABC"
    ] + [
        Patch(facecolor="white", edgecolor="#555555", hatch="////", label="返航后充电"),
        Line2D([0], [0], color="#222222", linestyle="--", label=f"最晚返场 {makespan:.1f} s"),
    ]
    fig.legend(handles=legend, ncol=5, frameon=False, loc="lower center",
               bbox_to_anchor=(0.52, 0.005), columnspacing=1.0, handlelength=1.5)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.97, bottom=0.10, hspace=0.44)
    save_figure(fig, "q2_drone_battery_gantt")


def draw_q2_delivery() -> None:
    data = read_result("Q2_逐箱交付.csv")
    data["交付完成时刻s"] = pd.to_numeric(data["交付完成时刻s"])
    data["期望送达时刻s"] = pd.to_numeric(data["期望送达时刻s"])
    data["货箱类别"] = data["货箱编号"].str.extract(r"-([A-Z]+)-")[0]
    labels = {"MED": "医疗物资", "WAT": "饮用水", "FOD": "应急食品", "HYG": "生活卫生用品"}
    marker_map = {"MED": "o", "WAT": "s", "FOD": "^", "HYG": "D"}
    colors = {"MED": "#0072B2", "WAT": "#D55E00", "FOD": "#008A5B", "HYG": "#8A5A9E"}
    fig, ax = plt.subplots(figsize=(6.55, 4.35))
    for kind in ("MED", "WAT", "FOD", "HYG"):
        subset = data.loc[data["货箱类别"].eq(kind)]
        hard = subset["硬时限s"].notna()
        ax.scatter(subset.loc[~hard, "期望送达时刻s"], subset.loc[~hard, "交付完成时刻s"],
                   marker=marker_map[kind], color=colors[kind], s=27, alpha=0.78,
                   edgecolors="white", linewidths=0.35, label=labels[kind])
        if hard.any():
            ax.scatter(subset.loc[hard, "期望送达时刻s"], subset.loc[hard, "交付完成时刻s"],
                       marker=marker_map[kind], facecolors="none", edgecolors=colors[kind],
                       s=51, linewidths=1.0)
    limit = max(data["期望送达时刻s"].max(), data["交付完成时刻s"].max()) * 1.04
    ax.plot([0, limit], [0, limit], color="#333333", linestyle="--", linewidth=1.0,
            label="按时期望时刻（$y=x$）")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("期望送达时刻（s）")
    ax.set_ylabel("实际交付完成时刻（s）")
    ax.grid(color="#D7DCE0", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    hard_late = ((data["硬时限s"].notna()) &
                 (data["交付完成时刻s"] > pd.to_numeric(data["硬时限s"], errors="coerce"))).sum()
    late = (data["交付完成时刻s"] > data["期望送达时刻s"]).sum()
    save_figure(fig, "q2_expected_vs_actual_delivery")


def main() -> None:
    setup_style()
    draw_q1_payload()
    draw_q1_sensitivity()
    draw_q2_gantt()
    draw_q2_delivery()


if __name__ == "__main__":
    main()
