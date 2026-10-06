"""Aggregate dInfer evaluation runs into the paper's results table.

    python scripts/aggregate.py outputs/eval/math/DMax-Math-MMD/seed*

Each run directory holds the `threshold_<t>/<task>/` cells written by scripts/eval.sh.
For every task and threshold, accuracy and TPF are averaged over runs with equal weight, with
the sample standard deviation across runs. For every task, the reported threshold is the one
with the highest mean TPF among thresholds whose mean accuracy is within `--tolerance` points
(default 0.5) of that task's best mean accuracy.
"""

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

TASKS = {
    "gsm8k_llada_mini": "GSM8K",
    "minerva_math500": "MATH500",
    "minerva_math_algebra": "Minerva-Algebra",
    "asdiv_llada_mini": "ASDIV",
    "humaneval_instruct": "HumanEval-Instruct",
    "mbpp_sanitized_llada_mini": "MBPP-Instruct",
}


def read_cell(cell):
    """Return (accuracy %, TPF) for one evaluated task and threshold, or None if incomplete."""
    answers = cell / "rank_0.jsonl"
    if not answers.is_file():
        return None
    # TPF is computed per example and then averaged over examples.
    tpfs = [json.loads(line)["tpf"] for line in answers.read_text().splitlines() if line]
    if not tpfs:
        return None
    tpf = sum(tpfs) / len(tpfs)
    # Math: the original DMax answer checker; code: lm-eval pass@1.
    checker = cell / "postprocess.log"
    if checker.is_file():
        found = re.findall(r"Accuracy:\s*([0-9.]+)%", checker.read_text(errors="replace"))
        return (float(found[-1]), tpf) if found else None
    results = sorted(cell.glob("*/results_*.json"))
    if results:
        scores = json.loads(results[-1].read_text())["results"][cell.name]
        return 100 * scores["pass_at_1,none"], tpf
    return None


def collect(runs):
    """Map (task, threshold) to one (accuracy, TPF) pair per run; drop cells missing in any run."""
    cells = {}
    for run in runs:
        for cell in run.glob("threshold_*/*"):
            if cell.name not in TASKS:
                continue
            threshold = float(cell.parent.name.removeprefix("threshold_"))
            cells.setdefault((cell.name, threshold), {})[run] = read_cell(cell)
    complete = {}
    order = list(TASKS)
    for key, values in sorted(
        cells.items(), key=lambda item: (order.index(item[0][0]), item[0][1])
    ):
        missing = [str(run) for run in runs if values.get(run) is None]
        if missing:
            print(
                f"Skipping {key[0]} @ {key[1]}: incomplete in {', '.join(missing)}", file=sys.stderr
            )
        else:
            complete[key] = [values[run] for run in runs]
    return complete


def fmt(mean, std):
    """Mean with the across-run standard deviation; a single run has no spread."""
    return f"{mean:.2f}" if std != std else f"{mean:.2f} ± {std:.2f}"


def summarize(values):
    def mean_std(xs):
        return statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else float("nan")

    accuracy, tpf = zip(*values)
    return (*mean_std(accuracy), *mean_std(tpf))


def select(rows, tolerance):
    """Per task: the highest-TPF threshold within `tolerance` points of the best mean accuracy."""
    selected = {}
    for task in TASKS:
        candidates = {t: row for (name, t), row in rows.items() if name == task}
        if not candidates:
            continue
        best = max(row[0] for row in candidates.values())
        eligible = {t: row for t, row in candidates.items() if row[0] >= best - tolerance}
        threshold = max(eligible, key=lambda t: eligible[t][2])
        selected[task] = threshold, candidates[threshold]
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+", type=Path, help="one output directory per seed")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="accuracy window in points below the best mean accuracy",
    )
    parser.add_argument("--all", action="store_true", help="also print every threshold")
    parser.add_argument("--compare", type=Path,
                        help="CSV of published points: task,threshold,accuracy_percent,tpf")
    args = parser.parse_args()

    rows = {key: summarize(values) for key, values in collect(args.runs).items()}
    if not rows:
        sys.exit("No complete evaluation cells found")
    print(f"{len(args.runs)} run{'s' if len(args.runs) > 1 else ''}: {', '.join(map(str, args.runs))}\n")
    if args.all:
        print("| Task | Threshold | Accuracy (%) | TPF |\n| --- | --- | --- | --- |")
        for (task, threshold), (acc, acc_std, tpf, tpf_std) in rows.items():
            print(
                f"| {TASKS[task]} | {threshold:g} | {fmt(acc, acc_std)} | {fmt(tpf, tpf_std)} |"
            )
        print()

    published = {}
    if args.compare:
        with args.compare.open() as f:
            published = {row["task"]: row for row in csv.DictReader(f)}
    header = "| Task | Threshold | Accuracy (%) | TPF |"
    if published:
        header += " Paper threshold | Paper accuracy | Paper TPF |"
    print(header)
    print("|" + " --- |" * (header.count("|") - 1))
    for task, (threshold, (acc, acc_std, tpf, tpf_std)) in select(rows, args.tolerance).items():
        line = (
            f"| {TASKS[task]} | {threshold:g} | {fmt(acc, acc_std)} | {fmt(tpf, tpf_std)} |"
        )
        if published:
            paper = published.get(task)
            line += (
                f" {float(paper['threshold']):g} | {float(paper['accuracy_percent']):.2f} "
                f"| {float(paper['tpf']):.2f} |"
                if paper
                else " | | |"
            )
        print(line)


if __name__ == "__main__":
    main()
