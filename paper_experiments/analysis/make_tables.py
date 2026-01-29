import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


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


def aggregate_status(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (int(row["N"]), row["method"], row["scenario"])
        status = row.get("status")
        if status is not None:
            groups[key].append(str(status))
    return groups


def summarize_status(statuses):
    if any("error" in s.lower() for s in statuses):
        return "error"
    if any(s in {"9", "TIME_LIMIT"} or "time" in s.lower() for s in statuses):
        return "time limit"
    return "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    args = parser.parse_args()

    with open(args.input_path, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    obj_stats = aggregate(rows, "obj")
    runtime_stats = aggregate(rows, "runtime_total")
    status_stats = aggregate_status(rows)

    ns = sorted({key[0] for key in obj_stats})
    methods = sorted({key[1] for key in obj_stats})
    scenarios = sorted({key[2] for key in obj_stats})

    lines = [
        "\\begin{tabular}{l l l l l}",
        "\\hline",
        "N & Scenario & Method & Obj (mean$\\pm$std) & Status \\\\",
        "\\hline",
    ]
    for n in ns:
        for scenario in scenarios:
            for method in methods:
                obj = obj_stats.get((n, method, scenario))
                runtime = runtime_stats.get((n, method, scenario))
                status = status_stats.get((n, method, scenario), [])
                if obj is None:
                    continue
                obj_text = f"{obj[0]:.3f} $\\pm$ {obj[1]:.3f}"
                runtime_text = f"{runtime[0]:.3f} $\\pm$ {runtime[1]:.3f}" if runtime else "-"
                status_text = summarize_status(status)
                lines.append(
                    f"{n} & {scenario} & {method} & {obj_text} ({runtime_text}) & {status_text} \\\\"
                )
    lines.append("\\hline")
    lines.append("\\end{tabular}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
