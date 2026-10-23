#!/usr/bin/env python3
"""
Recover the true original body text for papers whose papers/*.txt got
overwritten by a buggy re-fetch, by reversing the Exp2 injection recorded in
flaw_location ('<repr original_span> -> <repr replacement_span>') against the
still-intact papers/_exp2_injected/*.txt file for that doi.

Only touches the dois passed on argv (or results/table_fig_recheck_changed.json
if none given). Prints a diff-length report; does not overwrite anything if
the reversed text doesn't exactly reconstruct (original_span not found).
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT / "papers"
INJECTED_DIR = PAPERS_DIR / "_exp2_injected"


def parse_flaw_location(s):
    quote = s[0]
    i = 1
    while True:
        if s[i] == "\\":
            i += 2
            continue
        if s[i] == quote:
            break
        i += 1
    first_repr = s[: i + 1]
    rest = s[i + 1 :]
    assert rest.startswith(" -> "), rest[:20]
    second_repr = rest[4:]
    return ast.literal_eval(first_repr), ast.literal_eval(second_repr)


def main():
    dois = sys.argv[1:]
    if not dois:
        dois = json.loads((ROOT / "results" / "table_fig_recheck_changed.json").read_text())

    with open(ROOT / "results" / "exp2_injections.jsonl") as f:
        by_doi = {}
        for line in f:
            r = json.loads(line)
            by_doi.setdefault(r["doi"], r)  # any one model's row has the same flaw_location

    restored, failed = [], []
    for doi in dois:
        rec = by_doi.get(doi)
        if rec is None:
            failed.append((doi, "no exp2 record"))
            continue
        injected_path = INJECTED_DIR / (doi.replace("/", "_") + ".txt")
        if not injected_path.exists():
            failed.append((doi, "no injected file"))
            continue
        try:
            original_span, replacement_span = parse_flaw_location(rec["flaw_location"])
        except Exception as exc:
            failed.append((doi, f"parse error: {exc}"))
            continue

        injected_text = injected_path.read_text(encoding="utf-8")
        if replacement_span:
            if injected_text.count(replacement_span) != 1:
                failed.append((doi, f"replacement_span not uniquely found ({injected_text.count(replacement_span)}x)"))
                continue
            reconstructed = injected_text.replace(replacement_span, original_span, 1)
        else:
            # replacement was an empty-string deletion -- can't uniquely reverse a deletion,
            # skip these (rare) and flag for manual handling
            failed.append((doi, "replacement_span was empty (deletion, not reversible)"))
            continue

        dest = PAPERS_DIR / (doi.replace("/", "_") + ".txt")
        current = dest.read_text(encoding="utf-8") if dest.exists() else ""
        dest.write_text(reconstructed, encoding="utf-8")
        restored.append((doi, len(current), len(reconstructed)))

    print(f"Restored {len(restored)}, failed {len(failed)}.", file=sys.stderr)
    for doi, before, after in restored:
        print(f"  RESTORED {doi}: {before} -> {after} chars", file=sys.stderr)
    for doi, reason in failed:
        print(f"  FAILED {doi}: {reason}", file=sys.stderr)


if __name__ == "__main__":
    main()
