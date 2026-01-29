import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def read_rows(csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value):
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def aggregate(rows, metric):
    groups = defaultdict(list)
    for row in rows:
        value = as_float(row.get(metric))
        if value is None:
            continue
        key = (int(row["N"]), row["method"], row["scenario"])
        groups[key].append(value)

    stats = {}
    for key, values in groups.items():
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        std_val = math.sqrt(variance)
        stats[key] = (mean_val, std_val)
    return stats


def plot_metric(stats, title, ylabel, out_path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plot generation")
        return

    methods = sorted({key[1] for key in stats})
    scenarios = sorted({key[2] for key in stats})
    ns = sorted({key[0] for key in stats})

    fig, axes = plt.subplots(len(scenarios), 1, figsize=(8, 4 * len(scenarios)))
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        for method in methods:
            means = []
            stds = []
            for n in ns:
                values = stats.get((n, method, scenario))
                if values is None:
                    means.append(None)
                    stds.append(None)
                else:
                    means.append(values[0])
                    stds.append(values[1])
            xs = [n for n, m in zip(ns, means) if m is not None]
            ys = [m for m in means if m is not None]
            es = [s for s in stds if s is not None]
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", linestyle="-", label=method)
        ax.set_xlabel("N")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} ({scenario})")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path / f"{title}.png", dpi=300)
    plt.close(fig)


def plot_mechanism(stats_map, out_path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plot generation")
        return

    scenarios = sorted({key[2] for stats in stats_map.values() for key in stats})
    methods = sorted({key[1] for stats in stats_map.values() for key in stats})
    ns = sorted({key[0] for stats in stats_map.values() for key in stats})

    fig, axes = plt.subplots(len(scenarios), 3, figsize=(14, 4 * len(scenarios)))
    if len(scenarios) == 1:
        axes = [axes]
    for row_idx, scenario in enumerate(scenarios):
        row_axes = axes[row_idx]
        for col_idx, (metric, stats) in enumerate(stats_map.items()):
            ax = row_axes[col_idx]
            for method in methods:
                means = []
                stds = []
                for n in ns:
                    values = stats.get((n, method, scenario))
                    if values is None:
                        means.append(None)
                        stds.append(None)
                    else:
                        means.append(values[0])
                        stds.append(values[1])
                xs = [n for n, m in zip(ns, means) if m is not None]
                ys = [m for m in means if m is not None]
                es = [s for s in stds if s is not None]
                if xs:
                    ax.errorbar(xs, ys, yerr=es, marker="o", linestyle="-", label=method)
            ax.set_xlabel("N")
            ax.set_title(f"{metric} ({scenario})")
            ax.grid(True, linestyle="--", alpha=0.5)
        row_axes[0].set_ylabel("Value")
    axes[-1][2].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path / "Fig_Mechanism.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    out_path = Path(args.outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.input_path)

    runtime_stats = aggregate(rows, "runtime_total")
    obj_stats = aggregate(rows, "obj")
    pricing_share_stats = aggregate(rows, "pricing_time_share")
    pricing_calls_stats = aggregate(rows, "num_pricing_calls")
    num_iters_stats = aggregate(rows, "num_iters")

    plot_metric(runtime_stats, "Fig_Runtime", "runtime_total", out_path)
    plot_metric(obj_stats, "Fig_Obj", "obj", out_path)
    plot_mechanism(
        {
            "pricing_time_share": pricing_share_stats,
            "num_pricing_calls": pricing_calls_stats,
            "num_iters": num_iters_stats,
        },
        out_path,
    )


if __name__ == "__main__":
    main()
