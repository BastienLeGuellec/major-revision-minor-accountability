#!/usr/bin/env python3
"""
Fetch full text for the N most recent open-access articles from Nature and
Nature Medicine (or any journal id), via the Springer Nature Open Access API.
The API returns results newest-first, so we just take the top N per journal.

Usage:
    python3 get_recent_papers.py                 # 10 most recent from Nature + Nature Medicine
    python3 get_recent_papers.py --n 20
    python3 get_recent_papers.py --journals 41586 # Nature only
"""

import argparse
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from common.env import load_env  # noqa: E402
from common import get_fulltext as ft  # noqa: E402

load_env(ROOT / ".env")

JOURNALS = {
    "41586": "Nature",
    "41591": "Nature Medicine",
    "41467": "Nature Communications",
    "41588": "Nature Genetics",
    "41598": "Scientific Reports",
}

BASE = "https://api.springernature.com/openaccess/json"
REQUEST_DELAY = 0.5


def list_dois(journal_id, start, page_size, session):
    resp = session.get(
        BASE,
        params={"q": f"journalid:{journal_id}", "p": page_size, "s": start, "api_key": ft.API_KEY},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return [rec["doi"] for rec in data.get("records", [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="usable (full-text-available) papers wanted per journal")
    ap.add_argument("--journals", type=str, default=",".join(JOURNALS), help="comma-separated journal ids")
    ap.add_argument("--max-scan", type=int, default=100, help="give up after scanning this many candidates per journal")
    args = ap.parse_args()

    journal_ids = args.journals.split(",")
    session = requests.Session()
    ft.PAPERS_DIR.mkdir(exist_ok=True)

    for jid in journal_ids:
        name = JOURNALS.get(jid, jid)
        print(f"{name} ({jid}): scanning most recent open-access DOIs for {args.n} with usable full text...", file=sys.stderr)

        saved = 0
        start = 1
        page_size = min(25, args.max_scan)
        scanned = 0

        while saved < args.n and scanned < args.max_scan:
            dois = list_dois(jid, start, page_size, session)
            time.sleep(REQUEST_DELAY)
            if not dois:
                break

            for doi in dois:
                if saved >= args.n or scanned >= args.max_scan:
                    break
                scanned += 1
                result = ft.fetch_one(doi)
                print(f"  [{scanned}] {doi}: {result}", file=sys.stderr)
                if result in ("saved", "skipped"):
                    saved += 1
                if result != "skipped":
                    time.sleep(REQUEST_DELAY)

            start += page_size

        print(f"{name}: saved {saved}/{args.n} after scanning {scanned} candidates.", file=sys.stderr)


if __name__ == "__main__":
    main()
