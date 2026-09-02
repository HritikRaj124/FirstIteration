"""Comparison plots: nominal (unfiltered) vs. CBF-filtered rollout.

Pure plotting library -- no simulation logic, no driver code. Called from
nom_controller_with_CBF.py's main() with two logs (baseline, filtered).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _contiguous_spans(t, labels):
    """Contiguous (label, t_start, t_end) spans from a per-tick label list."""
    spans = []
    if not len(t):
        return spans
    start = t[0]
    cur = labels[0]
    for i in range(1, len(t)):
        if labels[i] != cur:
            spans.append((cur, start, t[i]))
            start = t[i]
            cur = labels[i]
    spans.append((cur, start, t[-1]))
    return spans


def plot_filter_comparison(baseline_log: list[dict], filtered_log: list[dict],
                            out_dir: Path) -> dict[str, Path]:
    """Compare an unfiltered (nominal) rollout against a CBF-filtered rollout:
    obstacle margins over time, tracking error, and violation counts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    t_b = np.array([e["t"] for e in baseline_log])
    t_f = np.array([e["t"] for e in filtered_log])
    phase_b = [e["phase"] for e in baseline_log]

    # --- 1: partition + table margins over time, baseline vs filtered ----- #
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axes[0].plot(t_b, [e["h_forearm"] for e in baseline_log], color="crimson", lw=1.2, label="nominal (unfiltered)")
    axes[0].plot(t_f, [e["h_forearm"] for e in filtered_log], color="navy", lw=1.4, label="CBF-filtered")
    axes[0].axhline(0.0, color="black", lw=0.9)
    axes[0].set_ylabel("h(q) — partition [m]")
    axes[0].set_title("Forearm-segment barrier value: nominal vs. safety-filtered")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(t_b, [e["h_forearm_table"] for e in baseline_log], color="crimson", lw=1.2, label="nominal (unfiltered)")
    axes[1].plot(t_f, [e["h_forearm_table"] for e in filtered_log], color="navy", lw=1.4, label="CBF-filtered")
    axes[1].axhline(0.0, color="black", lw=0.9)
    axes[1].set_ylabel("h(q) — table [m]")
    axes[1].set_xlabel("time [s]")
    axes[1].legend(loc="upper right", fontsize=8)

    for ax in axes:
        for name, t0, t1 in _contiguous_spans(t_b, phase_b):
            ax.axvspan(t0, t1, alpha=0.04, color="black")

    p = out_dir / "filter_comparison_margins.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written["filter_comparison_margins"] = p

    # --- 2: tracking error, baseline vs filtered --------------------------- #
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t_b, [e["err"] for e in baseline_log], color="crimson", lw=1.2, label="nominal (unfiltered)")
    ax.plot(t_f, [e["err"] for e in filtered_log], color="navy", lw=1.4, label="CBF-filtered")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("‖qpos − q_ref‖ [rad]")
    ax.set_title("Tracking deviation from the nominal reference")
    ax.legend(loc="upper right", fontsize=8)
    p = out_dir / "filter_comparison_tracking_error.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written["filter_comparison_tracking_error"] = p

    # --- 3: violation-count summary bar chart ------------------------------ #
    def _viol(log, key):
        return sum(1 for e in log if e[key] < 0)

    labels = ["partition\n(palm margin)", "partition\n(forearm h)", "table\n(forearm h)"]
    baseline_counts = [_viol(baseline_log, "margin"), _viol(baseline_log, "h_forearm"), _viol(baseline_log, "h_forearm_table")]
    filtered_counts = [_viol(filtered_log, "margin"), _viol(filtered_log, "h_forearm"), _viol(filtered_log, "h_forearm_table")]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, baseline_counts, w, color="crimson", label="nominal (unfiltered)")
    ax.bar(x + w/2, filtered_counts, w, color="navy", label="CBF-filtered")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"violating ticks (of {len(baseline_log)})")
    ax.set_title("Safety violations: nominal vs. CBF-filtered")
    ax.legend(loc="upper right", fontsize=8)
    p = out_dir / "filter_comparison_violations.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written["filter_comparison_violations"] = p

    return written