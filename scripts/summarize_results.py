"""Summarize the transcribed competition report."""
import csv
from pathlib import Path


def main():
    report = Path(__file__).resolve().parents[1] / "results" / "tasks.csv"
    with report.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if sorted(int(row["id"]) for row in rows) != list(range(1, 51)):
        raise ValueError("Expected 50 unique task IDs")
    total = sum(int(row["episodes_per_condition"]) for row in rows)
    for scenario in ("clean", "random"):
        counts = [int(row[f"{scenario}_success"]) for row in rows]
        if any(not 0 <= n <= int(row["episodes_per_condition"])
               for n, row in zip(counts, rows)):
            raise ValueError(f"Invalid success count: {scenario}")
        success = sum(counts)
        print(f"{scenario}: {success}/{total} ({success / total:.2%})")


if __name__ == "__main__":
    main()
