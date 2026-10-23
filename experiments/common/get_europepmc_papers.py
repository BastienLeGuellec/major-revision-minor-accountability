#!/usr/bin/env python3
"""
Fetch the N most recent (>= a given year) open-access, full-text-in-EPMC
papers from non-Springer-Nature top journals, via the Europe PMC REST API
(free, no key). Unlike the Springer Nature family, these journals mostly
require the NIH/funder public-access deposit before a paper shows up here,
so "most recent" lags real publication by weeks to months.

Usage:
    python3 get_europepmc_papers.py                    # 10 most recent >=2026 from each default journal
    python3 get_europepmc_papers.py --n 10 --min-year 2026
    python3 get_europepmc_papers.py --journals "Science,Cell"
"""

import argparse
import re
import sys
import time
import urllib.parse
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from common import journal_manifest  # noqa: E402
from common.article_filter import is_non_research, extract_floating_content  # noqa: E402

PAPERS_DIR = ROOT / "papers"
SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
REQUEST_DELAY = 0.4

# display name -> Europe PMC JOURNAL query value
JOURNALS = {
    "Science": "Science",
    "Cell": "Cell",
    "New England Journal of Medicine": "N Engl J Med",
    "The Lancet": "Lancet",
    "PNAS": "PNAS",
    "eLife": "eLife",
    "PLoS Biology": "PLoS Biol",
    "Circulation": "Circulation",
    "Journal of Clinical Investigation": "J Clin Invest",
    "JAMA Internal Medicine": "JAMA Intern Med",
    "Lancet Global Health": "Lancet Glob Health",
    "Lancet Microbe": "Lancet Microbe",
    "JAMA Neurology": "JAMA Neurol",
    "JAMA Oncology": "JAMA Oncol",
    "JAMA Psychiatry": "JAMA Psychiatry",
    "Nature Materials": "Nat Mater",
    "Nature Biotechnology": "Nat Biotechnol",
    "Blood": "Blood",
    "European Heart Journal": "Eur Heart J",
}

session = requests.Session()
session.headers.update({"User-Agent": "top-journal-fulltext-fetcher/1.0"})


def search(journal_query, min_year, page_size, cursor="*"):
    q = f'JOURNAL:"{journal_query}" AND OPEN_ACCESS:Y AND IN_EPMC:Y AND PUB_YEAR:[{min_year} TO 3000]'
    resp = session.get(
        SEARCH_URL,
        params={"query": q, "format": "json", "pageSize": page_size, "sort": "P_PDATE_D desc", "cursorMark": cursor},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_fulltext_xml(pmcid):
    resp = session.get(FULLTEXT_URL.format(pmcid=pmcid), timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.text


def extract_text(xml_text):
    root = ET.fromstring(xml_text)
    article = root if root.tag == "article" else root.find(".//article")
    if article is None:
        return None

    title_el = article.find(".//article-title")
    title = "".join(title_el.itertext()).strip() if title_el is not None else ""

    abstract_el = article.find(".//abstract")
    abstract = "".join(abstract_el.itertext()).strip() if abstract_el is not None else ""

    body_el = article.find(".//body")
    body = "".join(body_el.itertext()).strip() if body_el is not None else ""

    if not body:
        return None
    if is_non_research(title):
        return None

    floating = extract_floating_content(article, body_el)

    parts = []
    if title:
        parts.append(f"TITLE: {title}\n")
    if abstract:
        parts.append(f"ABSTRACT:\n{abstract}\n")
    parts.append(f"BODY:\n{body}")
    if floating:
        parts.append(f"\n\nTABLES AND FIGURES NOT SHOWN IN MAIN TEXT ABOVE:\n{floating}")
    return "\n".join(parts)


def safe_filename(doi):
    return doi.replace("/", "_") + ".txt"


def fetch_one(record, display_journal):
    doi = record.get("doi")
    pmcid = record.get("pmcid")
    if not doi or not pmcid:
        return None, "missing_doi_or_pmcid"

    dest = PAPERS_DIR / safe_filename(doi)
    if dest.exists():
        journal_manifest.set_journal(doi, display_journal)
        return doi, "skipped"

    xml_text = fetch_fulltext_xml(pmcid)
    if xml_text is None:
        return doi, "no_fulltext"

    text = extract_text(xml_text)
    if not text:
        return doi, "no_body"

    dest.write_text(text, encoding="utf-8")
    journal_manifest.set_journal(doi, display_journal)
    return doi, "saved"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="usable papers wanted per journal")
    ap.add_argument("--min-year", type=int, default=2026)
    ap.add_argument("--journals", type=str, default=None, help="comma-separated display names from JOURNALS")
    ap.add_argument("--max-scan", type=int, default=60)
    args = ap.parse_args()

    journals = args.journals.split(",") if args.journals else list(JOURNALS)
    PAPERS_DIR.mkdir(exist_ok=True)

    for display_name in journals:
        query_name = JOURNALS.get(display_name, display_name)
        print(f"{display_name}: scanning most recent OA/full-text papers since {args.min_year}...", file=sys.stderr)

        saved = 0
        scanned = 0
        cursor = "*"
        page_size = min(25, args.max_scan)

        while saved < args.n and scanned < args.max_scan:
            data = search(query_name, args.min_year, page_size, cursor)
            records = data.get("resultList", {}).get("result", [])
            if not records:
                break

            for rec in records:
                if saved >= args.n or scanned >= args.max_scan:
                    break
                scanned += 1
                doi, result = fetch_one(rec, display_name)
                print(f"  [{scanned}] {doi or rec.get('id')}: {result}", file=sys.stderr)
                if result in ("saved", "skipped"):
                    saved += 1
                if result not in ("skipped",):
                    time.sleep(REQUEST_DELAY)

            cursor = data.get("nextCursorMark")
            if not cursor:
                break

        print(f"{display_name}: saved {saved}/{args.n} after scanning {scanned} candidates.", file=sys.stderr)


if __name__ == "__main__":
    main()
