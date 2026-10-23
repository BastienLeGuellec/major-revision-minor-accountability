#!/usr/bin/env python3
"""
Add a reviewer model to Exp2 for manuscripts that already have a planted flaw
(from a prior run for other models), without re-running injection. Reuses the
existing injected text in papers/_exp2_injected/ and the planted-flaw record
already stored in exp2_injections.jsonl for the same doi.

Usage:
    python3 add_model.py --model deepseek/deepseek-v4-flash-0731 [--workers 6]
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import review_and_judge, RESULTS_JSONL, INJECTED_DIR, DEFAULT_JUDGE_MODEL  # noqa: E402

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    with RESULTS_JSONL.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    planted_by_doi = {}
    for r in rows:
        planted_by_doi.setdefault(r["doi"], {
            "category": r["category"],
            "central_conclusion": r["central_conclusion"],
            "flaw_summary": r["flaw_summary"],
            "flaw_location": r["flaw_location"],
        })

    already = {r["doi"] for r in rows if r["reviewer_model"] == args.model}
    todo = [doi for doi in planted_by_doi if doi not in already]
    log(f"{len(planted_by_doi)} manuscripts with a planted flaw; {len(already)} already done for {args.model}; {len(todo)} to do.")
    if not todo:
        return

    def one(doi):
        planted = planted_by_doi[doi]
        paper_path = INJECTED_DIR / (doi.replace("/", "_") + ".txt")
        modified_text = paper_path.read_text(encoding="utf-8")
        return review_and_judge(doi, planted, modified_text, args.model, args.judge_model)

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, doi): doi for doi in todo}
        for future in as_completed(futures):
            doi = futures[future]
            completed += 1
            try:
                record = future.result()
            except Exception as exc:
                log(f"[{completed}/{len(todo)}] {doi}: ERROR {exc}")
                continue
            with RESULTS_JSONL.open("a", encoding="utf-8") as out:
                out.write(json.dumps(record) + "\n")
            log(f"[{completed}/{len(todo)}] {doi} x {args.model}: detected={record['detected']}")

    log("Done.")


if __name__ == "__main__":
    main()
