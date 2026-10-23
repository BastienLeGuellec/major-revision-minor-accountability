#!/usr/bin/env python3
"""
Exp 1: what decision does an LLM reviewer hand back on a paper that was
actually published?

Usage:
    python3 run.py [--n N] [--models m1,m2,...] [--workers N]

Requires papers/*.txt to already exist -- run
experiments/common/get_fulltext.py or get_recent_papers.py first.

Idempotent: (doi, model) pairs already present in results/exp1_decisions.csv
are skipped, so re-running after fetching more papers only reviews the new
ones.
"""

import argparse
import csv
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import llm_clients, review_prompt, parse_decision, journal_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT / "papers"
RESULTS_CSV = ROOT / "results" / "exp1_decisions.csv"

_write_lock = threading.Lock()


def review_one(doi, paper_text, model):
    prompt = review_prompt.build_prompt(paper_text)
    system = review_prompt.build_system(review_prompt.journal_for_doi(doi))
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
    ap.add_argument("--n", type=int, default=None, help="number of papers to sample (default: all in papers/)")
    ap.add_argument("--models", type=str, default=",".join(llm_clients.PANEL_MODELS),
                    help="comma-separated model names; default = the four-model reviewer panel")
    ap.add_argument("--workers", type=int, default=8, help="concurrent review calls")
    args = ap.parse_args()

    models = args.models.split(",")
    unavailable = [m for m in models if m not in llm_clients.available_models()]
    if unavailable:
        print(f"No API key configured for: {', '.join(unavailable)} -- check .env.", file=sys.stderr)
        sys.exit(1)

    papers = sorted(PAPERS_DIR.glob("*.txt"))
    if args.n is not None:
        papers = papers[: args.n]
    if not papers:
        print(f"No papers found in {PAPERS_DIR}. Run common/get_fulltext.py or get_recent_papers.py first.", file=sys.stderr)
        sys.exit(1)

    done_pairs = already_done()
    RESULTS_CSV.parent.mkdir(exist_ok=True)
    write_header = not RESULTS_CSV.exists()

    jobs = []
    skipped = 0
    for paper_path in papers:
        doi = journal_manifest.doi_for_stem(paper_path.stem)
        for model in models:
            if (doi, model) in done_pairs:
                skipped += 1
                continue
            jobs.append((doi, paper_path, model))

    print(f"{len(jobs)} new (doi, model) pairs to review; {skipped} already done, skipped.", file=sys.stderr)
    if not jobs:
        return

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
