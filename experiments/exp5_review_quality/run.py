#!/usr/bin/env python3
"""
Exp 5: blind review-quality judging. For each manuscript, take the four
panel models' Experiment-1 reviews (of the real, unmodified manuscript),
relabel them "Review A/B/C/D" in a random order per manuscript (order not
correlated with model identity across the sample), and ask an off-panel
judge -- with no indication these are AI-written reviews at all -- to:

  1. cluster every major concern raised across the four reviews, noting
     which review(s) raised each one (gives per-review counts AND the
     cross-review overlap in a single pass);
  2. classify each concern cluster as relevant or nitpicking;
  3. rate each review's constructiveness 1-5;
  4. pick the single best review and say why.

Usage:
    python3 run.py [--judge-model x-ai/grok-4.6] [--n N] [--workers 6] [--seed 42]
"""
import argparse
import csv
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from common import llm_clients, journal_manifest  # noqa: E402

PAPERS_DIR = ROOT / "papers"
EXP1_CSV = ROOT / "results" / "exp1_decisions.csv"
RESULTS_JSONL = ROOT / "results" / "exp5_review_quality.jsonl"

MODELS = ["z-ai/glm-5.3-flash", "openai/gpt-5.6-sol", "google/gemini-3.7-flash", "deepseek/deepseek-v4-flash-0731"]
LETTERS = ["A", "B", "C", "D"]

JUDGE_SYSTEM = """You are an experienced journal editor evaluating four peer reviews of the same \
submitted manuscript, to assess review quality. You do not know who wrote each review -- treat all \
four as anonymous submissions labeled Review A, Review B, Review C, and Review D, in no particular \
order, and judge them purely on their content.

Read the manuscript and all four reviews carefully, then:

1. Identify every major concern raised across all four reviews -- a substantive methodological, \
statistical, or interpretive objection, not a typo, formatting note, or citation request. Group \
concerns that are substantively the same underlying issue, even if phrased differently or focused on \
a different symptom of it, into a single cluster, and list every review letter that raised it.

2. For each concern cluster, classify it as "relevant" (a real, well-founded issue that could \
legitimately affect the paper's conclusions or its acceptance) or "nitpicking" (technically a \
criticism, but minor, pedantic, or unlikely to matter -- e.g. requesting an additional citation, \
quibbling over word choice, or a point the manuscript already addresses elsewhere).

3. Rate each review's overall constructiveness on a 1-5 scale: 1 = purely critical with no \
actionable guidance for the authors; 5 = highly constructive, specific, and clearly aimed at helping \
the authors improve the paper (independent of how harsh or lenient its overall recommendation is).

4. Pick the single best review of the four -- the one you would most want an author to receive -- \
and say why in one sentence.

Respond with ONLY a JSON object (no markdown fences, no commentary):
{
  "concern_clusters": [
    {"summary": "<short phrase describing the concern>", "raised_by": ["A", "C"], "category": "relevant"},
    {"summary": "<short phrase describing the concern>", "raised_by": ["B"], "category": "nitpicking"}
  ],
  "constructiveness": {"A": <1-5>, "B": <1-5>, "C": <1-5>, "D": <1-5>},
  "best_review": "A",
  "best_review_reason": "<one sentence>"
}"""

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, file=sys.stderr)


def already_done_dois(path, judge_model):
    """DOIs already judged *by this judge* in `path`.

    Keyed on (doi, judge_model) so a second-judge arm re-judges the full
    sample instead of skipping everything the first judge covered.
    """
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return {r["doi"] for r in records if r.get("judge_model") == judge_model}


def load_reviews_by_doi():
    with EXP1_CSV.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["model"] in MODELS]
    by_doi = {}
    for r in rows:
        by_doi.setdefault(r["doi"], {})[r["model"]] = r["raw_review"]
    return {doi: cells for doi, cells in by_doi.items() if len(cells) == len(MODELS)}


def judge_one(doi, reviews_by_model, paper_text, judge_model, rng, max_tokens=4000):
    shuffled_models = list(reviews_by_model.keys())
    rng.shuffle(shuffled_models)
    letter_to_model = dict(zip(LETTERS, shuffled_models))
    model_to_letter = {m: l for l, m in letter_to_model.items()}

    review_block = "\n\n".join(
        f"REVIEW {letter}:\n{reviews_by_model[letter_to_model[letter]]}" for letter in LETTERS
    )
    prompt = f"MANUSCRIPT:\n{paper_text}\n\n{review_block}"

    raw = llm_clients.review(judge_model, JUDGE_SYSTEM, prompt, max_tokens=max_tokens, temperature=0, reasoning_effort="low")
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    data = json.loads(raw)  # let malformed JSON raise -- caller retries

    return {
        "doi": doi,
        "judge_model": judge_model,
        "letter_to_model": letter_to_model,
        "concern_clusters": data["concern_clusters"],
        "constructiveness": data["constructiveness"],
        "best_review_letter": data["best_review"],
        "best_review_model": letter_to_model.get(data["best_review"]),
        "best_review_reason": data["best_review_reason"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", type=str, default="x-ai/grok-4.6")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=4000,
                    help="judge completion budget; reasoning models may need more than the default")
    ap.add_argument("--out", type=Path, default=RESULTS_JSONL,
                    help="output JSONL (default results/exp5_review_quality.jsonl)")
    args = ap.parse_args()
    out_path = args.out

    reviews_by_doi = load_reviews_by_doi()
    done = already_done_dois(out_path, args.judge_model)
    todo = [doi for doi in reviews_by_doi if doi not in done]
    log(f"{len(reviews_by_doi)} manuscripts with complete 4-model coverage; {len(done)} already judged by {args.judge_model}; {len(todo)} to do.")
    if args.n is not None:
        todo = todo[: args.n]
    if not todo:
        return

    rng_master = random.Random(args.seed)
    out_path.parent.mkdir(exist_ok=True)

    def one(doi):
        paper_path = PAPERS_DIR / (doi.replace("/", "_") + ".txt")
        paper_text = paper_path.read_text(encoding="utf-8")
        rng = random.Random(rng_master.randint(0, 2**31))
        last_err = None
        for attempt in range(args.attempts):
            try:
                return judge_one(doi, reviews_by_doi[doi], paper_text, args.judge_model, rng, args.max_tokens)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        raise RuntimeError(f"failed after {args.attempts} attempts: {last_err}")

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
            with out_path.open("a", encoding="utf-8") as out:
                out.write(json.dumps(record) + "\n")
            log(f"[{completed}/{len(todo)}] {doi}: best={record['best_review_model']}")

    log("Done.")


if __name__ == "__main__":
    main()
