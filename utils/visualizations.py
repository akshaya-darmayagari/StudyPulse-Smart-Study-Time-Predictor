"""
StudyPulse – Visualization Module
All matplotlib/seaborn charts for the dashboard.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List

# ── Theme ─────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":       "#0D1117",
    "card":     "#161B22",
    "accent1":  "#58A6FF",
    "accent2":  "#3FB950",
    "accent3":  "#F78166",
    "accent4":  "#D29922",
    "text":     "#C9D1D9",
    "muted":    "#8B949E",
}

def apply_theme(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PALETTE["card"])
    ax.tick_params(colors=PALETTE["text"], labelsize=9)
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["muted"])
    if title:  ax.set_title(title, color=PALETTE["text"], fontsize=11, fontweight="bold", pad=10)
    if xlabel: ax.set_xlabel(xlabel, color=PALETTE["muted"], fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=PALETTE["muted"], fontsize=9)


# ── 1. Hourly Productivity Heatmap ────────────────────────────────────────────

def plot_hourly_productivity(hourly_scores: Dict[int, float], save_path: str = None):
    hours = list(range(6, 24))
    scores = [hourly_scores.get(h, 0) for h in hours]

    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor(PALETTE["bg"])

    # Bar chart with gradient coloring
    colors = [plt.cm.RdYlGn(s / 10) for s in scores]
    bars = ax.bar(hours, scores, color=colors, width=0.75, edgecolor=PALETTE["bg"], linewidth=0.5)

    # Mark top 3 hours
    top3 = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    for idx in top3:
        ax.bar(hours[idx], scores[idx], color=PALETTE["accent2"],
               width=0.75, edgecolor=PALETTE["bg"], linewidth=0.5)
        ax.text(hours[idx], scores[idx] + 0.15, "★", ha="center", color=PALETTE["accent2"], fontsize=8)

    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, ha="right")
    ax.set_ylim(0, 11)
    apply_theme(ax, "⚡ Your Hourly Productivity Forecast", "Hour of Day", "Focus Score (0-10)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=PALETTE["bg"], bbox_inches="tight")
    return fig


# ── 2. Weekly Focus Trend ─────────────────────────────────────────────────────

def plot_weekly_trend(df: pd.DataFrame, save_path: str = None):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date").agg(
        focus=("focus_level", "mean"),
        sleep=("sleep_hours", "mean"),
        stress=("stress_level", "mean"),
    ).reset_index().tail(30)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax.plot(daily["date"], daily["focus"], color=PALETTE["accent1"],
            linewidth=2, label="Focus Level", marker="o", markersize=3)
    ax.fill_between(daily["date"], daily["focus"], alpha=0.15, color=PALETTE["accent1"])

    ax.plot(daily["date"], daily["sleep"], color=PALETTE["accent2"],
            linewidth=1.5, linestyle="--", label="Sleep Hours", marker=".", markersize=3)

    ax.plot(daily["date"], daily["stress"], color=PALETTE["accent3"],
            linewidth=1.5, linestyle=":", label="Stress Level", marker=".", markersize=3)

    ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["muted"],
              labelcolor=PALETTE["text"], fontsize=8)
    apply_theme(ax, "📈 30-Day Focus / Sleep / Stress Trend", "Date", "Score")

    plt.xticks(rotation=30)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=PALETTE["bg"], bbox_inches="tight")
    return fig


# ── 3. Subject Performance Radar ─────────────────────────────────────────────

def plot_subject_radar(df: pd.DataFrame, save_path: str = None):
    subjects = df["subject"].unique().tolist()
    avg_focus  = [df[df["subject"] == s]["focus_level"].mean() for s in subjects]
    avg_score  = [df[df["subject"] == s]["previous_score"].mean() / 10 for s in subjects]  # normalise to 10

    N = len(subjects)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    focus_vals  = avg_focus + avg_focus[:1]
    score_vals  = avg_score + avg_score[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])

    ax.plot(angles, focus_vals, color=PALETTE["accent1"], linewidth=2, label="Avg Focus")
    ax.fill(angles, focus_vals, alpha=0.2, color=PALETTE["accent1"])

    ax.plot(angles, score_vals, color=PALETTE["accent2"], linewidth=2, label="Avg Score /10")
    ax.fill(angles, score_vals, alpha=0.2, color=PALETTE["accent2"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(subjects, color=PALETTE["text"], fontsize=8)
    ax.set_yticklabels([], color=PALETTE["muted"])
    ax.spines["polar"].set_color(PALETTE["muted"])
    ax.grid(color=PALETTE["muted"], alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              facecolor=PALETTE["card"], edgecolor=PALETTE["muted"],
              labelcolor=PALETTE["text"], fontsize=8)
    ax.set_title("🎯 Subject Performance Radar",
                 color=PALETTE["text"], fontsize=11, fontweight="bold", pad=20)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=PALETTE["bg"], bbox_inches="tight")
    return fig


# ── 4. Time-of-Day Focus Distribution ────────────────────────────────────────

def plot_tod_distribution(df: pd.DataFrame, save_path: str = None):
    tod_order = ["Morning", "Afternoon", "Evening", "Night"]
    tod_avg = df.groupby("time_of_day")["focus_level"].mean().reindex(tod_order).fillna(0)
    colors = [PALETTE["accent4"], PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent3"]]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    bars = ax.bar(tod_avg.index, tod_avg.values, color=colors, width=0.5, edgecolor=PALETTE["bg"])

    for bar, val in zip(bars, tod_avg.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1,
                f"{val:.1f}", ha="center", va="bottom", color=PALETTE["text"], fontsize=9)

    apply_theme(ax, "🌅 Focus by Time of Day", "Time of Day", "Avg Focus (0-10)")
    ax.set_ylim(0, 11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=PALETTE["bg"], bbox_inches="tight")
    return fig


# ── 5. Sleep vs Focus Scatter ─────────────────────────────────────────────────

def plot_sleep_focus(df: pd.DataFrame, save_path: str = None):
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(PALETTE["bg"])

    scatter = ax.scatter(df["sleep_hours"], df["focus_level"],
                         c=df["stress_level"], cmap="RdYlGn_r",
                         alpha=0.6, s=30, edgecolors="none")

    # Trend line
    z = np.polyfit(df["sleep_hours"], df["focus_level"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["sleep_hours"].min(), df["sleep_hours"].max(), 100)
    ax.plot(x_line, p(x_line), color=PALETTE["accent1"], linewidth=2, linestyle="--", label="Trend")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Stress Level", color=PALETTE["muted"], fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=PALETTE["muted"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["text"])

    apply_theme(ax, "😴 Sleep vs Focus (coloured by Stress)", "Sleep Hours", "Focus Level")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=PALETTE["bg"], bbox_inches="tight")
    return fig


# ── 6. Burnout Risk Timeline ──────────────────────────────────────────────────

def plot_burnout_timeline(df: pd.DataFrame, save_path: str = None):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    daily_burnout = df.groupby("date")["burnout_risk"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 2.8))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax.fill_between(daily_burnout["date"], daily_burnout["burnout_risk"],
                    where=daily_burnout["burnout_risk"] >= 0.6,
                    color=PALETTE["accent3"], alpha=0.5, label="High Risk Zone")
    ax.fill_between(daily_burnout["date"], daily_burnout["burnout_risk"],
                    where=daily_burnout["burnout_risk"] < 0.6,
                    color=PALETTE["accent2"], alpha=0.3)

    ax.plot(daily_burnout["date"], daily_burnout["burnout_risk"],
            color=PALETTE["accent3"], linewidth=1.5)
    ax.axhline(0.6, color=PALETTE["accent4"], linestyle="--", linewidth=1, label="Risk Threshold")

    ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["muted"],
              labelcolor=PALETTE["text"], fontsize=8)
    apply_theme(ax, "🔥 Burnout Risk Over Time", "Date", "Risk Score (0-1)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=PALETTE["bg"], bbox_inches="tight")
    return fig


# ── 7. Study Schedule Bar Chart ───────────────────────────────────────────────

def plot_daily_schedule(schedule: List[Dict], save_path: str = None):
    if not schedule:
        return None

    subjects = list({b["subject"] for b in schedule})
    subject_colors = dict(zip(subjects, [
        PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent3"],
        PALETTE["accent4"], "#A371F7", "#EC775C"
    ]))

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor(PALETTE["bg"])

    for block in schedule:
        h = block["hour"]
        dur = block["duration_min"] / 60  # convert to hours
        color = subject_colors.get(block["subject"], PALETTE["accent1"])
        ax.barh(0, dur, left=h, height=0.5, color=color, edgecolor=PALETTE["bg"], linewidth=1)
        if dur > 0.4:
            ax.text(h + dur / 2, 0, block["subject"][:4],
                    ha="center", va="center", color="white", fontsize=7, fontweight="bold")

    # Break markers
    for block in schedule:
        brk_start = block["hour"] + block["duration_min"] / 60
        brk_dur   = block["break_after_min"] / 60
        ax.barh(0, brk_dur, left=brk_start, height=0.5,
                color=PALETTE["muted"], alpha=0.4, edgecolor=PALETTE["bg"])

    ax.set_xlim(6, 24)
    ax.set_xticks(range(6, 24))
    ax.set_xticklabels([f"{h}h" for h in range(6, 24)], fontsize=8, color=PALETTE["text"])
    ax.set_yticks([])
    ax.set_xlabel("Hour of Day", color=PALETTE["muted"], fontsize=9)
    ax.set_title("📅 Your Personalised Daily Schedule", color=PALETTE["text"],
                 fontsize=11, fontweight="bold", pad=10)

    # Legend
    patches = [mpatches.Patch(color=c, label=s) for s, c in subject_colors.items() if any(b["subject"] == s for b in schedule)]
    patches.append(mpatches.Patch(color=PALETTE["muted"], alpha=0.5, label="Break"))
    ax.legend(handles=patches, loc="upper right", facecolor=PALETTE["card"],
              edgecolor=PALETTE["muted"], labelcolor=PALETTE["text"], fontsize=7, ncol=3)

    apply_theme(ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=PALETTE["bg"], bbox_inches="tight")
    return fig


# ── 8. Feature Importance ─────────────────────────────────────────────────────

def plot_feature_importance(feat_imp: pd.Series, save_path: str = None):
    top = feat_imp.head(10).sort_values()

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(PALETTE["bg"])

    colors = [PALETTE["accent1"] if v >= top.quantile(0.7) else PALETTE["muted"] for v in top.values]
    ax.barh(top.index, top.values, color=colors, edgecolor=PALETTE["bg"])
    apply_theme(ax, "🧠 Top Features Driving Focus Prediction", "Importance", "Feature")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=PALETTE["bg"], bbox_inches="tight")
    return fig
