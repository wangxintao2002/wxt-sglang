#!/usr/bin/env python3
"""Analyze DLLM trace JSONL files and print per-round overhead breakdown.

Usage:
    python3 scripts/analyze_dllm_trace.py trace1.jsonl [trace2.jsonl ...]

Each file is a separate algo run (e.g., FDFO vs SP). The script prints a
side-by-side comparison of round-level and sub-operation timing.
"""

import json
import sys
from pathlib import Path


def load_trace(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    data = sorted(data)
    idx = (len(data) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (idx - lo)


def analyze(records: list[dict]) -> dict:
    rounds = [r for r in records if r["event"] == "dllm_round"]
    run_batches = [r for r in records if r["event"] == "dllm_run_batch"]
    algo_runs = [r for r in records if r["event"] == "dllm_algo_run"]

    algo = rounds[0]["algo"] if rounds else "unknown"

    def stats(vals):
        if not vals:
            return {"mean": 0, "p50": 0, "p99": 0, "total": 0, "count": 0}
        return {
            "mean": sum(vals) / len(vals),
            "p50": percentile(vals, 50),
            "p99": percentile(vals, 99),
            "total": sum(vals),
            "count": len(vals),
        }

    return {
        "algo": algo,
        "round": stats([r["round_ms"] for r in rounds]),
        "schedule": stats([r["schedule_ms"] for r in rounds]),
        "run_batch": stats([r["run_batch_ms"] for r in rounds]),
        "process_result": stats([r["process_result_ms"] for r in rounds]),
        "prepare": stats([r.get("prepare_ms", 0) for r in run_batches]),
        "forward": stats([r.get("forward_ms", 0) for r in run_batches]),
        "model_forward": stats([r.get("model_forward_ms", 0) for r in algo_runs]),
        "postprocess": stats([r.get("postprocess_ms", 0) for r in algo_runs]),
    }


def print_report(results: list[dict]):
    def row(label, key, subkey, unit="ms", pct_of=None):
        cols = []
        for r in results:
            s = r[key]
            val = s[subkey]
            if pct_of and r[pct_of]["mean"] > 0:
                pct = val / r[pct_of]["mean"] * 100
                cols.append(f"{val:8.2f}  ({pct:4.1f}%)")
            else:
                cols.append(f"{val:8.2f}")
        print(f"  {label:<36} " + "   ".join(cols))

    def header(title):
        print(f"\n{'─'*80}")
        print(f"  {title}")
        print(f"{'─'*80}")

    algos = [r["algo"] for r in results]
    print("\n" + "="*80)
    print("  DLLM Per-Round Overhead Breakdown")
    print("="*80)
    print(f"  {'Metric':<36} " + "   ".join(f"{a:<20}" for a in algos))

    header("Round counts and total time")
    for r in results:
        s = r["round"]
        print(f"  {'algo':<36} {r['algo']}")
        print(f"  {'total rounds':<36} {s['count']:>8}")
        print(f"  {'total round time (s)':<36} {s['total']/1000:>8.2f}")
        print()

    header("Per-round breakdown (ms)  [mean / p50 / p99]")
    cols_header = "   ".join(f"{'mean':>8}  {'p50':>6}  {'p99':>6}" for _ in results)
    print(f"  {'':36} {cols_header}")

    def ms_row(label, key):
        cols = []
        for r in results:
            s = r[key]
            pct = s["mean"] / r["round"]["mean"] * 100 if r["round"]["mean"] > 0 else 0
            cols.append(f"{s['mean']:8.2f}  {s['p50']:6.2f}  {s['p99']:6.2f}  ({pct:4.1f}%)")
        print(f"  {label:<36} " + "   ".join(cols))

    ms_row("TOTAL round", "round")
    ms_row("  schedule (recv+batch)", "schedule")
    ms_row("  run_batch", "run_batch")
    ms_row("    ├─ prepare", "prepare")
    ms_row("    └─ forward (algo+model)", "forward")
    ms_row("       ├─ model_forward", "model_forward")
    ms_row("       └─ postprocess", "postprocess")
    ms_row("  process_result", "process_result")

    header("Fixed overhead summary")
    for r in results:
        fixed = r["schedule"]["mean"] + r["prepare"]["mean"] + r["process_result"]["mean"]
        compute = r["model_forward"]["mean"]
        total = r["round"]["mean"]
        rounds = r["round"]["count"]
        print(f"  {r['algo']}")
        print(f"    Rounds:              {rounds}")
        print(f"    Fixed/round:         {fixed:.2f} ms  ({fixed/total*100:.1f}%)")
        print(f"    Compute/round:       {compute:.2f} ms  ({compute/total*100:.1f}%)")
        print(f"    Total fixed:         {fixed * rounds / 1000:.2f} s")
        print(f"    Total compute:       {compute * rounds / 1000:.2f} s")
        print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    results = []
    for path in sys.argv[1:]:
        records = load_trace(path)
        r = analyze(records)
        r["file"] = path
        results.append(r)
        print(f"Loaded {len(records)} records from {path} (algo={r['algo']}, rounds={r['round']['count']})")

    print_report(results)


if __name__ == "__main__":
    main()
