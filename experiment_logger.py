import csv
from pathlib import Path


SCHEMA = [
    "instance_id",
    "N",
    "seed",
    "scenario",
    "method",
    "obj",
    "runtime_total",
    "runtime_rmp",
    "runtime_pricing",
    "status",
    "gap",
    "num_iters",
    "num_pricing_calls",
    "num_fallback_calls",
    "num_columns_added",
    "min_reduced_cost_last",
    "pricing_time_share",
]


class ExperimentLogger:
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self._ensure_parent()

    def _ensure_parent(self):
        if self.csv_path.parent != Path("."):
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def _format_row(self, row):
        return {key: row.get(key) for key in SCHEMA}

    def log_row(self, row):
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SCHEMA)
            if write_header:
                writer.writeheader()
            writer.writerow(self._format_row(row))
            handle.flush()
