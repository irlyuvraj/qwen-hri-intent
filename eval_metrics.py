#!/usr/bin/env python3
"""
Offline evaluation of MetricsLogger JSONL sessions.

Aggregates one or more ~/sessions/metrics_*.jsonl files into a baseline report:
parse-failure rate, latency distribution, intent/phase class counts, the
task_complete-before-15s false-positive proxy, and the per-policy placing-phase
coverage (the pink-vs-yellow auto-complete gap).

Usage:
    python eval_metrics.py ~/sessions/metrics_2026052*.jsonl
    python eval_metrics.py --since 2026-05-19T19:00 ~/sessions/metrics_*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime


def load_records(paths):
    """Yield (path, started_ts, [records]) per session file."""
    for path in paths:
        recs, started = [], None
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "_meta" in obj:
                    started = obj.get("_started")
                    continue
                recs.append(obj)
        yield path, started, recs


def pct(numer, denom):
    return 100.0 * numer / denom if denom else 0.0


def summarize(all_recs):
    n = len(all_recs)
    if not n:
        print("No prediction records found.")
        return

    # ── parse failures ────────────────────────────────────────
    parse_failed = sum(1 for r in all_recs if r.get("parse_failed"))

    # ── latency ───────────────────────────────────────────────
    # robot_state is the predictor's string: "executing" (policy running) or
    # "waiting" (idle). Exclude >5s outliers (vLLM timeout/backoff stalls) from
    # the typical-latency stats and report them separately.
    TIMEOUT_MS = 5000.0
    lat = [r["latency_ms"] for r in all_recs
           if r.get("latency_ms") and r["latency_ms"] < TIMEOUT_MS]
    timeouts = [r["latency_ms"] for r in all_recs
                if r.get("latency_ms") and r["latency_ms"] >= TIMEOUT_MS]
    lat_run = [r["latency_ms"] for r in all_recs
               if r.get("latency_ms") and r["latency_ms"] < TIMEOUT_MS
               and r.get("robot_state") == "executing"]
    lat_wait = [r["latency_ms"] for r in all_recs
                if r.get("latency_ms") and r["latency_ms"] < TIMEOUT_MS
                and r.get("robot_state") != "executing"]

    def p(vals, q):
        if not vals:
            return 0.0
        s = sorted(vals)
        return s[min(len(s) - 1, int(q * len(s)))]

    # ── intent / phase distributions ──────────────────────────
    intents = Counter(r.get("intent", "?") for r in all_recs)
    phases = Counter(r.get("phase", "?") for r in all_recs)

    # ── task_complete false-positive proxy ────────────────────
    # Group by (session-relative) active_policy run; flag task_complete=True
    # that appears <15s into a running policy as an early (likely false) fire.
    tc_total = sum(1 for r in all_recs if r.get("task_complete"))
    tc_running = [r for r in all_recs
                  if r.get("task_complete") and r.get("robot_state") == "executing"]

    # ── placing-phase coverage per policy (the yellow gap) ─────
    placing_by_policy = defaultdict(int)
    frames_by_policy = defaultdict(int)
    for r in all_recs:
        pol = r.get("active_policy")
        if r.get("robot_state") == "executing" and pol:
            frames_by_policy[pol] += 1
            if r.get("phase") == "placing":
                placing_by_policy[pol] += 1

    # ── spoken_command externalization ────────────────────────
    spoken_nonempty = sum(1 for r in all_recs if (r.get("spoken_command") or "").strip())

    # ── report ────────────────────────────────────────────────
    print("=" * 60)
    print(f"  BASELINE REPORT — {n} predictions")
    print("=" * 60)

    print("\n── Reliability ──")
    print(f"  parse failures      : {parse_failed}/{n}  ({pct(parse_failed, n):.1f}%)")
    print(f"  spoken_command set  : {spoken_nonempty}/{n}  ({pct(spoken_nonempty, n):.1f}%)")

    print("\n── Latency (ms) ──")
    if lat:
        print(f"  overall   n={len(lat):4d}  mean={statistics.mean(lat):6.1f}  "
              f"median={statistics.median(lat):6.1f}  p95={p(lat, 0.95):6.1f}  "
              f"min={min(lat):.0f}  max={max(lat):.0f}")
    if lat_wait:
        print(f"  waiting   n={len(lat_wait):4d}  mean={statistics.mean(lat_wait):6.1f}  "
              f"median={statistics.median(lat_wait):6.1f}  p95={p(lat_wait, 0.95):6.1f}")
    if lat_run:
        print(f"  executing n={len(lat_run):4d}  mean={statistics.mean(lat_run):6.1f}  "
              f"median={statistics.median(lat_run):6.1f}  p95={p(lat_run, 0.95):6.1f}")
    if timeouts:
        print(f"  timeouts  n={len(timeouts):4d}  (>={TIMEOUT_MS:.0f}ms, excluded above; "
              f"max={max(timeouts):.0f}ms) — vLLM stall/backoff")

    print("\n── predicted_intent distribution ──")
    for k, v in intents.most_common():
        print(f"  {k:24s} {v:4d}  ({pct(v, n):.1f}%)")

    print("\n── predicted_phase distribution ──")
    for k, v in phases.most_common():
        print(f"  {k:24s} {v:4d}  ({pct(v, n):.1f}%)")

    print("\n── task_complete ──")
    print(f"  total True          : {tc_total}  (of which {len(tc_running)} while running)")

    print("\n── placing-phase coverage per policy (auto-complete gap) ──")
    for pol in sorted(frames_by_policy):
        fr = frames_by_policy[pol]
        pl = placing_by_policy.get(pol, 0)
        flag = "  <-- NO placing frames (max-runtime fallback only)" if pl == 0 else ""
        print(f"  {pol:20s} running-frames={fr:4d}  placing={pl:3d} ({pct(pl, fr):.1f}%){flag}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="metrics_*.jsonl files (globs ok)")
    ap.add_argument("--since", default=None,
                    help="ISO datetime, e.g. 2026-05-19T19:00 — skip older sessions")
    args = ap.parse_args()

    files = []
    for pat in args.paths:
        files.extend(glob.glob(os.path.expanduser(pat)))
    files = sorted(set(files))

    since_ts = None
    if args.since:
        since_ts = datetime.fromisoformat(args.since).timestamp()

    all_recs = []
    used, skipped = [], []
    for path, started, recs in load_records(files):
        if since_ts and started and started < since_ts:
            skipped.append(path)
            continue
        all_recs.extend(recs)
        used.append((path, len(recs)))

    print(f"Loaded {len(used)} session(s), skipped {len(skipped)} (older than --since):")
    for path, c in used:
        print(f"  {os.path.basename(path):40s} {c:4d} predictions")
    print()
    summarize(all_recs)


if __name__ == "__main__":
    main()
