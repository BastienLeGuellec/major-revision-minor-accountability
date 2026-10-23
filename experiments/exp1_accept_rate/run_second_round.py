#!/usr/bin/env python3
"""
Revision-round framing (manuscript section 3.2): re-review every corpus
manuscript under a system prompt identical to Exp 1's except for its opening
sentence, which says the manuscript was revised after a first round of review
(review_prompt.SECOND_ROUND_SYSTEM_TEMPLATE).

Usage:
    python3 run_second_round.py [--models m1,m2,...] [--workers N]

Idempotent: (doi, model) pairs already present in results/exp1_secondround.csv
are skipped.
"""

import argparse
import csv
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import llm_clients, review_prompt, parse_decision  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT / "papers"
EXP1_CSV = ROOT / "results" / "exp1_decisions.csv"
RESULTS_CSV = ROOT / "results" / "exp1_secondround.csv"

_write_lock = threading.Lock()


def review_one(doi, paper_text, model):
    prompt = review_prompt.build_prompt(paper_text)
    system = review_prompt.build_second_round_system(review_prompt.journal_for_doi(doi))
    raw = llm_clients.review(model, system, prompt)
    decision, method = parse_decision.parse_decision(raw)
    return [doi, model, decision, method, len(raw.split()), raw]


def already_done():
    if not RESULTS_CSV.exists():
        return set()
    with RESULTS_CSV.open(newline="", encoding="utf-8") as f:
        return {(row["doi"], row["model"]) for row in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default=",".join(llm_clients.PANEL_MODELS),
                    help="comma-separated model names; default = the four-model reviewer panel")
    ap.add_argument("--workers", type=int, default=8, help="concurrent review calls")
    args = ap.parse_args()

    csv.field_size_limit(sys.maxsize)
    models = args.models.split(",")
    with EXP1_CSV.open(newline="", encoding="utf-8") as f:
        dois = sorted({row["doi"] for row in csv.DictReader(f)})
    done_pairs = already_done()

    jobs = []
    for doi in dois:
        paper_path = PAPERS_DIR / f"{doi.replace('/', '_')}.txt"
        if not paper_path.exists():
            print(f"missing paper file for {doi}", file=sys.stderr)
            continue
        jobs.extend((doi, paper_path, m) for m in models if (doi, m) not in done_pairs)

    print(f"{len(jobs)} new (doi, model) pairs to review; {len(done_pairs)} already done.", file=sys.stderr)
    if not jobs:
        return

    write_header = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["doi", "model", "decision", "parse_method", "word_count", "raw_review"])

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(review_one, doi, path.read_text(encoding="utf-8"), model): (doi, model)
                for doi, path, model in jobs
            }
            completed = 0
            for future in as_completed(futures):
                doi, model = futures[future]
                completed += 1
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"[{completed}/{len(jobs)}] {doi} x {model}: ERROR {exc}", file=sys.stderr)
                    continue
                print(f"[{completed}/{len(jobs)}] {doi} x {model}: {row[2]}", file=sys.stderr)
                with _write_lock:
                    writer.writerow(row)
                    f.flush()

    print(f"Done. Results appended to {RESULTS_CSV}", file=sys.stderr)


if __name__ == "__main__":
    main()
