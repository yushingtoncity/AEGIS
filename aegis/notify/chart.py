"""The picture attached to the friends' feed recap: the paper account's
change over the last few recaps, next to the benchmark's.

Everything it draws comes from the ``feed_recap`` events the feed already
writes (one per New York day): each one carries the account's change on
the day in percent and, since this chart, the benchmark's. Chaining them
gives a running change from the first day shown. Percent only, whatever
``feed.show_amounts`` says: the chart never shows dollars, equity or
positions.

matplotlib is imported only inside ``render_png`` and drawn with its Agg
backend (no display), so nothing else in the feed needs it, and a chart
that cannot be drawn leaves the recap as text (see ``aegis.notify.feed``).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date

MIN_POINTS = 2
"""Days needed before a chart is drawn: one point is not a line."""
WIDTH_IN, HEIGHT_IN, DPI = 8.0, 4.0, 100
"""800 x 400 pixels: readable on a phone, a few tens of kilobytes."""
ACCOUNT_COLOR = "#b14aed"
BENCHMARK_COLOR = "#8a8f98"
BACKGROUND = "#1e1f24"
FOREGROUND = "#e6e6e6"
GRID = "#3a3c44"


@dataclass(frozen=True)
class DayChange:
    """One recap day: the account's and the benchmark's change in percent
    (None when that figure was not available that day)."""

    day: date
    account_pct: float | None
    benchmark_pct: float | None


def cumulative(changes: list[float | None]) -> list[float | None]:
    """Running change in percent from before the first day, compounding each
    day's change. A missing day shows as a gap and carries the level on."""
    level = 1.0
    out: list[float | None] = []
    for pct in changes:
        if pct is None:
            out.append(None)
            continue
        level *= 1 + pct / 100
        out.append((level - 1) * 100)
    return out


def drawable(days: list[DayChange]) -> bool:
    """True when the account line has at least ``MIN_POINTS`` points."""
    return sum(1 for d in days if d.account_pct is not None) >= MIN_POINTS


def render_png(days: list[DayChange], benchmark: str, disclaimer: str) -> bytes:
    """The chart as PNG bytes, oldest day on the left. Raises whatever
    matplotlib raises; the caller decides what a failed chart means."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter

    labels = [f"{d.day.strftime('%b')} {d.day.day}" for d in days]
    account = cumulative([d.account_pct for d in days])
    bench = cumulative([d.benchmark_pct for d in days])
    fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN), dpi=DPI)
    try:
        fig.patch.set_facecolor(BACKGROUND)
        ax.set_facecolor(BACKGROUND)
        x = list(range(len(days)))
        ax.axhline(0, color=GRID, linewidth=1)
        if any(v is not None for v in bench):
            ax.plot(x, _nan(bench), color=BENCHMARK_COLOR, linewidth=1.5, label=benchmark)
        ax.plot(x, _nan(account), color=ACCOUNT_COLOR, linewidth=2.5, marker="o",
                markersize=3, label="AEGIS paper account")
        step = max(1, len(days) // 8)
        ax.set_xticks(x[::step], labels[::step])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.1f}%"))
        ax.set_title(f"Running change, {labels[0]} to {labels[-1]}", color=FOREGROUND, loc="left")
        ax.tick_params(colors=FOREGROUND)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        legend = ax.legend(loc="upper left", frameon=False)
        for text in legend.get_texts():
            text.set_color(FOREGROUND)
        fig.text(0.99, 0.01, disclaimer, ha="right", va="bottom", fontsize=8, color=BENCHMARK_COLOR)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", facecolor=BACKGROUND)
        return buffer.getvalue()
    finally:
        plt.close(fig)


def _nan(values: list[float | None]) -> list[float]:
    return [float("nan") if v is None else v for v in values]
