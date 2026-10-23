#!/usr/bin/env python3
"""
Exp 2: plant one critical, hard-to-spot flaw into each of N published papers
(via inject_errors.py, using an off-panel injector model), then have each
reviewer model review the doctored version and check whether the review
actually catches the planted flaw.

Usage:
    python3 run.py [--n N] [--models m1,m2,...] [--workers N] [--injector-model M] [--judge-model M]

Injection and review+judge are each parallelized across papers/models -- the
injector call alone is slow (tens of seconds to a few minutes per paper) and
retries on failure, so running papers sequentially doesn't scale.
"""

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import llm_clients, review_prompt, journal_manifest  # noqa: E402
from inject_errors import CATEGORIES, INJECTOR_MODEL, inject  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT / "papers"
RESULTS_JSONL = ROOT / "results" / "exp2_injections.jsonl"
INJECTED_DIR = ROOT / "papers" / "_exp2_injected"

DEFAULT_MODELS = llm_clients.PANEL_MODELS
DEFAULT_JUDGE_MODEL = "anthropic/claude-opus-5"  # off-panel judge; not one of the models under test.
# Cost note: the judge call is a ~10-token yes/no/partial classification, not a
# full review -- negligible cost regardless of model, unlike the reviewer panel.

JUDGE_SYSTEM = """You are checking whether a peer review caught a specific planted flaw \
in a manuscript. You will be given: the paper's real central conclusion, a description of \
the one critical flaw that was planted, and the full text of a review someone wrote of the \
doctored manuscript.

Decide whether the review explicitly identifies this specific flaw -- the actual \
methodological/logical problem described, in the same location -- as opposed to just \
raising generic or unrelated criticisms that happen to be in the neighborhood.

Respond with ONLY one word: yes, no, or partial."""

_write_lock = threading.Lock()
_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, file=sys.stderr)


def already_done_dois():
    if not RESULTS_JSONL.exists():
        return set()
    with RESULTS_JSONL.open(encoding="utf-8") as f:
        return {json.loads(line)["doi"] for line in f if line.strip()}


def judge_detection(central_conclusion, flaw_summary, flaw_location, review_text, model=DEFAULT_JUDGE_MODEL):
    prompt = (
        f"Central conclusion: {central_conclusion}\n"
        f"Planted flaw: {flaw_summary}\n"
        f"Flaw location: {flaw_location}\n\n"
        f"Review:\n{review_text}\n\n"
        "Did the review catch this specific flaw?"
    )
    raw = llm_clients.review(
        model, JUDGE_SYSTEM, prompt, max_tokens=200, temperature=0, reasoning_effort="low"
    ).strip().lower()
    if raw.startswith("yes"):
        return True
    if raw.startswith("partial"):
        return "partial"
    return False


def inject_one(paper_path, category, injector_model):
    doi = journal_manifest.doi_for_stem(paper_path.stem)
    paper_text = paper_path.read_text(encoding="utf-8")

    def on_attempt(i, error):
        if error:
            log(f"  {doi} [{category}]: injection attempt {i+1} failed ({error})")
        else:
            log(f"  {doi} [{category}]: injection succeeded on attempt {i+1}")

    result = inject(paper_text, category, model=injector_model, on_attempt=on_attempt)
    if result is None:
        log(f"{doi} [{category}]: injection failed after all attempts, skipping")
        return None
    modified_text, planted = result
    (INJECTED_DIR / paper_path.name).write_text(modified_text, encoding="utf-8")
    log(f"{doi} [{category}]: flaw -- {planted['flaw_summary'][:120]}")
    return doi, modified_text, planted


def review_and_judge(doi, planted, modified_text, model, judge_model):
    system = review_prompt.build_system(review_prompt.journal_for_doi(doi))
    prompt = review_prompt.build_prompt(modified_text)
    raw_review = llm_clients.review(model, system, prompt)
    detected = judge_detection(
        planted["central_conclusion"], planted["flaw_summary"], planted["flaw_location"], raw_review, model=judge_model
    )
    return {
        "doi": doi,
        "reviewer_model": model,
        "injector_model": INJECTOR_MODEL,
        "category": planted["category"],
        "central_conclusion": planted["central_conclusion"],
        "flaw_summary": planted["flaw_summary"],
        "flaw_location": planted["flaw_location"],
        "raw_review": raw_review,
        "detected": detected,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="number of papers to sample (default: all new papers)")
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS), help="comma-separated reviewer models")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--injector-model", type=str, default=INJECTOR_MODEL)
    ap.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--first-category", type=str, choices=CATEGORIES, default=CATEGORIES[0],
                    help="category the round-robin starts on, to keep a top-up batch balanced with the existing corpus")
    args = ap.parse_args()

    models = args.models.split(",")
    done_dois = already_done_dois()
    all_papers = sorted(PAPERS_DIR.glob("*.txt"))
    papers = [p for p in all_papers if journal_manifest.doi_for_stem(p.stem) not in done_dois]
    log(f"{len(all_papers)} papers total, {len(done_dois)} already done, {len(papers)} new.")
    if args.n is not None:
        papers = papers[: args.n]
    if not papers:
        log("Nothing new to inject.")
        return

    INJECTED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSONL.parent.mkdir(exist_ok=True)

    # round-robin category assignment so the batch gets balanced statistical/methodological coverage
    offset = CATEGORIES.index(args.first_category)
    paper_categories = [(p, CATEGORIES[(i + offset) % len(CATEGORIES)]) for i, p in enumerate(papers)]

    log(f"Injecting flaws into {len(papers)} papers (model={args.injector_model}, {args.workers} parallel workers) ...")
    injected = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(inject_one, p, category, args.injector_model): p for p, category in paper_categories}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                injected.append(result)

    log(f"Injection done: {len(injected)}/{len(papers)} papers succeeded.")
    if not injected:
        return

    jobs = [(doi, planted, modified_text, model) for doi, modified_text, planted in injected for model in models]
    log(f"Reviewing {len(jobs)} (paper, model) pairs ({args.workers} parallel workers) ...")

    records = []
    with RESULTS_JSONL.open("a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(review_and_judge, doi, planted, modified_text, model, args.judge_model): (doi, model)
                for doi, planted, modified_text, model in jobs
            }
            completed = 0
            for future in as_completed(futures):
                doi, model = futures[future]
                completed += 1
                try:
                    record = future.result()
                except Exception as exc:  # noqa: BLE001
                    log(f"[{completed}/{len(jobs)}] {doi} x {model}: ERROR {exc}")
                    continue
                log(f"[{completed}/{len(jobs)}] {doi} x {model}: detected={record['detected']}")
                records.append(record)
                with _write_lock:
                    out.write(json.dumps(record) + "\n")
                    out.flush()

    log(f"Done. Results appended to {RESULTS_JSONL}")
    for category in CATEGORIES:
        cat_records = [r for r in records if r["category"] == category]
        if not cat_records:
            continue
        n_yes = sum(1 for r in cat_records if r["detected"] is True)
        n_partial = sum(1 for r in cat_records if r["detected"] == "partial")
        n = len(cat_records)
        log(f"  {category}: {n_yes}/{n} detected ({n_partial} partial)")


if __name__ == "__main__":
    main()
