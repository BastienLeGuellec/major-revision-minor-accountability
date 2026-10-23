#!/usr/bin/env python3
"""
Intra-rater reliability test: re-review the same manuscripts a second time
under the identical protocol (temperature=0.7, so genuine stochasticity),
and save to a separate results file (not merged into exp1_decisions.csv) so
Cohen's/weighted kappa can be computed between pass 1 and pass 2 per model.

Usage:
    python3 run_intra_rater.py --dois-file results/intra_rater_sample.json --workers 20
"""
import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from common import llm_clients, review_prompt, parse_decision, journal_manifest  # noqa: E402

PAPERS_DIR = ROOT / "papers"
RESULTS_CSV = ROOT / "results" / "exp1_intra_rater_retest.csv"

MODELS = ["z-ai/glm-5.3-flash", "openai/gpt-5.6-sol", "google/gemini-3.7-flash", "deepseek/deepseek-v4-flash-0731"]

_print_lock = threading.Lock()
_write_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, file=sys.stderr)


def review_one(doi, paper_text, model):
    t0 = time.time()
    prompt = review_prompt.build_prompt(paper_text)
    system = review_prompt.build_system(review_prompt.journal_for_doi(doi))
    raw = llm_clients.review(model, system, prompt)
    decision, method = parse_decision.parse_decision(raw)
    dt = time.time() - t0
    return [doi, model, decision, method, len(raw.split()), len(paper_text), dt, raw]


def already_done():
    if not RESULTS_CSV.exists():
        return set()
    with RESULTS_CSV.open(newline="", encoding="utf-8") as f:
        return {(row["doi"], row["model"]) for row in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dois-file", type=str, default=str(ROOT / "results" / "intra_rater_sample.json"))
    ap.add_argument("--models", type=str, default=",".join(MODELS))
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    dois = json.loads(Path(args.dois_file).read_text())
    models = args.models.split(",")
    done_pairs = already_done()

    jobs = []
    for doi in dois:
        paper_path = PAPERS_DIR / (doi.replace("/", "_") + ".txt")
        text = paper_path.read_text(encoding="utf-8")
        for model in models:
            if (doi, model) in done_pairs:
                continue
            jobs.append((doi, text, model))

    log(f"{len(dois)} papers x {len(models)} models; {len(done_pairs)} already done; {len(jobs)} new jobs.")
    if not jobs:
        return

    write_header = not RESULTS_CSV.exists()
    RESULTS_CSV.parent.mkdir(exist_ok=True)
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        if write_header:
            writer.writerow(["doi", "model", "decision", "parse_method", "word_count", "paper_chars", "call_seconds", "raw_review"])

        completed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(review_one, doi, text, model): (doi, model) for doi, text, model in jobs}
            for future in as_completed(futures):
                doi, model = futures[future]
                completed += 1
                try:
                    row = future.result()
                except Exception as exc:
                    log(f"[{completed}/{len(jobs)}] {doi} x {model}: ERROR {exc}")
                    continue
                with _write_lock:
                    writer.writerow(row)
                    out.flush()
                log(f"[{completed}/{len(jobs)}] {doi} x {model}: {row[2]} ({row[6]:.1f}s)")

    log("Done.")


if __name__ == "__main__":
    main()
