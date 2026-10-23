#!/usr/bin/env python3
"""
Re-run Exp 2 reviews whose text carries no parseable decision -- truncated
outputs (reasoning-token exhaustion), off-task replies, runaway reasoning --
against the same injected manuscript and planted flaw, then re-judge them.
Each such record is replaced in place in exp2_injections.jsonl; the replaced
originals are appended to results/_replaced_unparsed_exp2.jsonl for the record.

Run only when nothing else is appending to exp2_injections.jsonl.

Usage:
    python3 rerun_unparsed.py [--attempts 3] [--workers 6]
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import parse_decision  # noqa: E402
from run import review_and_judge, RESULTS_JSONL, INJECTED_DIR, DEFAULT_JUDGE_MODEL, log  # noqa: E402

ARCHIVE = RESULTS_JSONL.parent / "_replaced_unparsed_exp2.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    records = [json.loads(line) for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    bad = [i for i, r in enumerate(records) if parse_decision.parse_decision(r["raw_review"])[0] is None]
    log(f"{len(bad)} of {len(records)} reviews have no parseable decision.")
    if not bad:
        return

    def one(i):
        r = records[i]
        text = (INJECTED_DIR / (r["doi"].replace("/", "_") + ".txt")).read_text(encoding="utf-8")
        for attempt in range(args.attempts):
            new = review_and_judge(r["doi"], r, text, r["reviewer_model"], args.judge_model)
            if parse_decision.parse_decision(new["raw_review"])[0] is not None:
                return i, new
            log(f"  {r['doi']} x {r['reviewer_model']}: attempt {attempt + 1} still unparseable")
        return i, None

    replaced = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(one, i) for i in bad]):
            i, new = future.result()
            old = records[i]
            if new is None:
                log(f"{old['doi']} x {old['reviewer_model']}: gave up, record left as is")
                continue
            log(f"{old['doi']} x {old['reviewer_model']}: detected {old['detected']} -> {new['detected']}")
            replaced.append(old)
            records[i] = new

    with ARCHIVE.open("a", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in replaced)
    RESULTS_JSONL.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    log(f"Replaced {len(replaced)} records; originals appended to {ARCHIVE}")


if __name__ == "__main__":
    main()
