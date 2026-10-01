"""DOC PLOT of the cache-size interference curve (CPU only, matplotlib; retroinfer-eval tests/test_cache_curve.py renders it from
a synthetic CSV). Reads cache_curve.py's ccurve_points.csv and writes one figure as PNG, SVG and PDF, next to a copy of the CSV
and of this script:

  rows     panel 1 = resident-decode slowdown % beside the transfer; panel 2 = useful delivered GB/s (H2D bytes / the request-busy
           union within the tick-window hull), plus, in the decode-paced regime, each method's OWN offered rate (dashed, in that
           method's colour: decode-paced releases follow each method's own decode ticks, so the offered loads are method-dependent
           and NOT matched; no averaged line is drawn)
  columns  one facet per (batch, regime): the primary batch (B192 or its registered fallback) first, then B64; saturation, then
           decode-paced (CSV regime key 'paced'; CC1's 'arrival-paced')
  lines    one per method: CPU8 (categorical slot 1, circles) and W8 (slot 2, squares); error bars = 95% bootstrap CI over trace
           steps; a cell whose status is not OK is not drawn and is marked MISSING under the axis. Rows of other methods / regimes
           (the host-pack control, ext-paced) are not drawn in this figure (ccurve_hostpack.csv / the table carry them)

    python cache_curve_plot.py --csv <dir>/ccurve_points.csv [--csv <other job>/ccurve_points.csv ...] --out-dir <dir>/plot

SEVERAL CSVs (job CC2 passes CC1's points next to its own, so ONE figure carries the primary batch and B64): the rows are merged
per (batch, C, method, regime); an OK row replaces a MISSING one (a leftover cell measured by CC2), two OK rows for the same key
are REFUSED (never a silent pick).
"""
import argparse
import csv
import math
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CAPACITIES = (63, 73, 81, 96, 113, 128)
REGIMES = (("saturation", "saturation"), ("paced", "decode-paced"))
METHODS = (("cpu8", "CPU8 (CPU pack + bulk DMA + GPU scatter)", "#2a78d6", "o"),      # reference palette slots 1 and 2,
           ("w8", "W8 (NOSI GPU gather, 8 CTAs)", "#eb6834", "s"))                   # validated all-pairs (first three slots)
SHORT = {"cpu8": "CPU8", "w8": "W8"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
CAPTION = ("NOSA-8B, PG-19, L = 16128, A100. C = 63 attended + P victim-pool groups (64 tokens) per layer, KV head and request. "
           "Slowdown: resident decode tick beside the transfer vs decode alone (every tick includes the decode-state restore), mean over "
           "trace steps, bars = 95% bootstrap CI over steps. Useful GB/s: natural H2D miss bytes / the request-busy union within the "
           "tick-window hull (includes restore and receipt gaps; not a per-byte cost). Saturation: a finite train released at once. "
           "Decode-paced: one trace step of plans released at each tick start of THAT method's own decode, so the arrival schedule is "
           "method-dependent and NOT a matched load; dashed = each method's own offered rate. MISSING = not certified or not run.")


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def read_points(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["batch"] = int(float(r["batch"]))
        r["C"] = int(float(r["C"]))
    return rows


def merge_points(row_lists):
    """Rows of several points CSVs, one per (batch, C, method, regime) (module docstring)."""
    out, order = {}, []
    for rows in row_lists:
        for r in rows:
            k = (r["batch"], r["C"], r.get("method"), r.get("regime"))
            if k not in out:
                out[k] = r
                order.append(k)
            elif r.get("status") == "OK":
                if out[k].get("status") == "OK":
                    raise ValueError("two OK points for batch %s C %s %s %s: refusing to pick one" % k)
                out[k] = r
    return [out[k] for k in order]


def facets(rows):
    batches = sorted({r["batch"] for r in rows}, key=lambda b: (b == 64, -b))       # the primary batch first, B64 last
    return [(b, rg, label) for b in batches for rg, label in REGIMES]


def _series(rows, b, rg, method, y, lo, hi):
    pts = sorted((r for r in rows if r["batch"] == b and r["regime"] == rg and r["method"] == method), key=lambda r: r["C"])
    xs, ys, el, eh = [], [], [], []
    for r in pts:
        v = _f(r.get(y))
        if r.get("status") != "OK" or v != v:
            continue
        a, c = _f(r.get(lo)), _f(r.get(hi))
        xs.append(r["C"])
        ys.append(v)
        el.append(max(0.0, v - a) if a == a else 0.0)
        eh.append(max(0.0, c - v) if c == c else 0.0)
    return xs, ys, [el, eh]


def offered_series(rows, b, rg, method):
    """One method's OWN offered rate per C (decode-paced: from its own release intervals). Never averaged across methods."""
    pts = sorted((r for r in rows if r["batch"] == b and r["regime"] == rg and r["method"] == method and r.get("status") == "OK"), key=lambda r: r["C"])
    xs, ys = [], []
    for r in pts:
        v = _f(r.get("offered_gbps"))
        if v == v:
            xs.append(r["C"])
            ys.append(v)
    return xs, ys


def _missing(rows, b, rg):
    st = {}
    for r in rows:
        if r["batch"] == b and r["regime"] == rg:
            st.setdefault(r["C"], []).append(r.get("status") == "OK")
    return [C for C in CAPACITIES if not any(st.get(C, []))]


def render(rows, out_dir, stem="ccurve_plot", title=None):
    fc = facets(rows)
    if not fc:
        raise ValueError("no points to plot")
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.titlesize": 10, "svg.fonttype": "none", "pdf.fonttype": 42})
    n = len(fc)
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n + 0.6, 6.4), sharex=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for j, (b, rg, label) in enumerate(fc):
        top, bot = axes[0][j], axes[1][j]
        for ax in (top, bot):
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.set_xticks(CAPACITIES)
            ax.set_xlim(58, 133)
        top.axhline(0.0, color=INK2, linewidth=0.8)
        top.set_title("B%d, %s" % (b, label), color=INK)
        if rg == "paced":
            bot.set_title("each method's own release schedule: offered loads NOT matched", fontsize=7, color=INK2)
        for m, name, col, mk in METHODS:
            xs, ys, err = _series(rows, b, rg, m, "slowdown_pct", "slowdown_ci_lo", "slowdown_ci_hi")
            if xs:
                top.errorbar(xs, ys, yerr=err, color=col, marker=mk, markersize=5, linewidth=1.6, capsize=2.5, elinewidth=1.0, label=name,
                             markeredgecolor=SURFACE, markeredgewidth=0.8)
            xs, ys, err = _series(rows, b, rg, m, "useful_gbps", "useful_gbps_ci_lo", "useful_gbps_ci_hi")
            if xs:
                bot.errorbar(xs, ys, yerr=err, color=col, marker=mk, markersize=5, linewidth=1.6, capsize=2.5, elinewidth=1.0, label=name,
                             markeredgecolor=SURFACE, markeredgewidth=0.8)
        if rg == "paced":
            for m, name, col, mk in METHODS:                      # one dashed line PER METHOD, in its colour (never averaged)
                xs, ys = offered_series(rows, b, rg, m)
                if xs:
                    bot.plot(xs, ys, color=col, linestyle="--", linewidth=1.1, marker=mk, markersize=3, markerfacecolor="none",
                             label="offered, %s (its own decode ticks)" % SHORT[m])
        miss = _missing(rows, b, rg)
        for C in miss:
            for ax in (top, bot):
                ax.annotate("MISSING", (C, 0.0), xycoords=("data", "axes fraction"), xytext=(0, 3), textcoords="offset points", ha="center",
                            va="bottom", fontsize=6.5, color=INK2, rotation=90)
        bot.set_xlabel("cache capacity C (groups per stream)")
        if j == 0:
            top.set_ylabel("resident-decode slowdown (%)")
            bot.set_ylabel("useful delivered GB/s")
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    ncol = len(labels) if n >= 4 else min(len(labels), 2)        # job 2179948: 3 long labels on 2 facets ran off both edges
    fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.995))
    if title:
        fig.suptitle(title, y=1.04, color=INK)
    fig.text(0.01, 0.005, CAPTION, ha="left", va="bottom", fontsize=7, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95 if ncol == len(labels) else 0.92))
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for ext in ("png", "svg", "pdf"):
        p = os.path.join(out_dir, "%s.%s" % (stem, ext))
        fig.savefig(p, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.12)   # nothing outside the canvas is cut
        out[ext] = p
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", required=True, action="append", help="points CSV; repeat to merge several jobs' points")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stem", default="ccurve_plot")
    ap.add_argument("--title", default=None)
    a = ap.parse_args(argv)
    rows = merge_points([read_points(c) for c in a.csv])
    out = render(rows, a.out_dir, a.stem, a.title)
    srcs = [(c, os.path.basename(c) if i == 0 else "input%d_%s" % (i + 1, os.path.basename(c))) for i, c in enumerate(a.csv)]
    for src, name in srcs + [(os.path.abspath(__file__), os.path.basename(__file__))]:
        dst = os.path.join(a.out_dir, name)
        if os.path.abspath(src) != os.path.abspath(dst) and not os.path.exists(dst):
            shutil.copyfile(src, dst)
    print("[ccurve-plot] %s" % " ".join(sorted(out.values())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
