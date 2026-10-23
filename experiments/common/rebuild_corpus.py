#!/usr/bin/env python3
"""
Rebuild the corpus texts (papers/*.txt) and the Experiment-2 flaw-injected
texts (papers/_exp2_injected/*.txt) from the DOIs in papers/corpus.csv.

The texts are not redistributed with the repository. Each article's JATS XML
is re-fetched from the Springer Nature Open Access API (Nature, Nature
Medicine, Nature Communications, Nature Genetics, Scientific Reports) or from
Europe PMC (every other journal) and converted with the same extractors that
built the corpus. Each injected text is the original with the passage
recorded in results/exp2_injections.jsonl (flaw_location) replaced.

Every file is checked against its SHA-256 in papers/corpus.csv. Some corpus
texts were extracted by earlier versions of the code, and publishers can revise
an article's XML, so a re-fetched text may differ from the one the models
reviewed (53 of 200 when tested on 23 October 2026). Such files are still
written and are listed as mismatches. Existing files are never overwritten.

Usage:
    python3 rebuild_corpus.py [--out DIR] [DOI ...]
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from common import get_fulltext  # noqa: E402
from common.check_licenses import fetch_epmc_xml, is_springer  # noqa: E402
from common.fetch_ccby_topup import extract_text_epmc  # noqa: E402
from common.restore_from_injected import parse_flaw_location  # noqa: E402


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_text(doi):
    if is_springer(doi):
        return get_fulltext.extract_text(get_fulltext.fetch_jats(doi))
    xml_text, _ = fetch_epmc_xml(doi)
    return extract_text_epmc(xml_text)[0] if xml_text else None


def write_new(path, text):
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "papers", help="output folder (default: papers/)")
    ap.add_argument("dois", nargs="*", help="rebuild only these DOIs")
    args = ap.parse_args()

    rows = list(csv.DictReader((ROOT / "papers" / "corpus.csv").open(encoding="utf-8")))
    if args.dois:
        rows = [r for r in rows if r["doi"] in args.dois]
    flaws = {}
    for line in (ROOT / "results" / "exp2_injections.jsonl").open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            flaws.setdefault(r["doi"], r["flaw_location"])

    report = {"identical": [], "different": [], "failed": [], "skipped (file exists)": []}
    for i, row in enumerate(rows, 1):
        doi, stem = row["doi"], row["doi"].replace("/", "_")
        try:
            text = fetch_text(doi)
        except Exception as exc:  # noqa: BLE001
            text, err = None, str(exc)
        else:
            err = "no full text returned"
        if not text:
            report["failed"].append(f"{doi}: {err}")
            print(f"[{i}/{len(rows)}] {doi}: FAILED ({err})", file=sys.stderr)
            continue

        original, replacement = parse_flaw_location(flaws[doi])
        injected = text.replace(original, replacement, 1) if original in text else None
        same = sha256(text) == row["text_sha256"]
        same_inj = injected is not None and sha256(injected) == row["injected_sha256"]

        wrote = write_new(args.out / f"{stem}.txt", text)
        if injected is not None:
            wrote = write_new(args.out / "_exp2_injected" / f"{stem}.txt", injected) or wrote
        if not wrote:
            report["skipped (file exists)"].append(doi)
        (report["identical"] if same and same_inj else report["different"]).append(
            doi if same and same_inj else
            f"{doi}: original {'identical' if same else 'differs'}, injected "
            f"{'identical' if same_inj else 'differs' if injected else 'not built (altered passage not found)'}")
        print(f"[{i}/{len(rows)}] {doi}: {'identical' if same and same_inj else 'DIFFERS'}", file=sys.stderr)
        time.sleep(0.3)

    for k, v in report.items():
        print(f"{k}: {len(v)}", file=sys.stderr)
        for line in v if k != "identical" else []:
            print(f"  {line}", file=sys.stderr)


if __name__ == "__main__":
    main()
