#!/usr/bin/env python3
"""
Fetch additional CC-BY-only papers to top up the 20 kept journals (dropping
Blood, JAMA Neurology, NEJM, which have too little CC-BY coverage) to 10
CC-BY papers each, for a clean 200-paper / 20-journal all-CC-BY corpus.

Saves to papers/_ccby_topup/ (NOT papers/ directly) so nothing touches the
live corpus until the results are reviewed. Skips any DOI already present in
papers/ (existing corpus, including the ones being dropped) to avoid dupes.

Usage:
    python3 fetch_ccby_topup.py
"""
import json
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from common.env import load_env  # noqa: E402
from common.article_filter import is_non_research, extract_floating_content  # noqa: E402
from common import get_fulltext as ft  # noqa: E402
from common.check_licenses import extract_license  # noqa: E402

load_env(ROOT / ".env")
PAPERS_DIR = ROOT / "papers"
TOPUP_DIR = PAPERS_DIR / "_ccby_topup"

EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

# journal display name -> (Europe PMC JOURNAL query value, how many more CC-BY needed)
EPMC_NEEDED = {
    "Science": ("Science", 1),
    "Nature Materials": ("Nat Mater", 2),
    "PNAS": ("PNAS", 2),
    "JAMA Internal Medicine": ("JAMA Intern Med", 3),
    "JAMA Psychiatry": ("JAMA Psychiatry", 3),
    "Lancet Global Health": ("Lancet Glob Health", 3),
    "Lancet Microbe": ("Lancet Microbe", 4),
    "Nature Biotechnology": ("Nat Biotechnol", 5),
    "Cell": ("Cell", 5),
    "Circulation": ("Circulation", 5),
    "JAMA Oncology": ("JAMA Oncol", 6),
}

# Springer journal id -> (display name, how many more CC-BY needed)
SPRINGER_NEEDED = {
    "41586": ("Nature", 2),
    "41591": ("Nature Medicine", 3),
    "41467": ("Nature Communications", 3),
    "41588": ("Nature Genetics", 5),
    "41598": ("Scientific Reports", 4),
}

session = requests.Session()
session.headers.update({"User-Agent": "ccby-topup-fetcher/1.0"})


def existing_dois():
    """DOIs already anywhere in the live corpus (kept or dropped), to avoid dupes."""
    from common import journal_manifest
    stems = [p.stem for p in PAPERS_DIR.glob("*.txt")]
    return {journal_manifest.doi_for_stem(s) for s in stems}


def epmc_search(journal_query, cursor, page_size=25, min_year=None):
    q = f'JOURNAL:"{journal_query}" AND OPEN_ACCESS:Y AND IN_EPMC:Y AND LICENSE:"cc by"'
    if min_year:
        q += f' AND PUB_YEAR:[{min_year} TO 3000]'
    r = session.get(EPMC_SEARCH, params={
        "query": q, "format": "json", "pageSize": page_size,
        "sort": "P_PDATE_D desc", "cursorMark": cursor,
    }, timeout=30)
    r.raise_for_status()
    return r.json()


def epmc_fulltext(pmcid):
    resp = session.get(EPMC_FULLTEXT.format(pmcid=pmcid), timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.text


def extract_text_epmc(xml_text):
    root = ET.fromstring(xml_text)
    article = root if root.tag == "article" else root.find(".//article")
    if article is None:
        return None, None
    title_el = article.find(".//article-title")
    title = "".join(title_el.itertext()).strip() if title_el is not None else ""
    abstract_el = article.find(".//abstract")
    abstract = "".join(abstract_el.itertext()).strip() if abstract_el is not None else ""
    body_el = article.find(".//body")
    body = "".join(body_el.itertext()).strip() if body_el is not None else ""
    if not body or is_non_research(title):
        return None, None
    floating = extract_floating_content(article, body_el)
    parts = [f"TITLE: {title}\n"]
    if abstract:
        parts.append(f"ABSTRACT:\n{abstract}\n")
    parts.append(f"BODY:\n{body}")
    if floating:
        parts.append(f"\n\nTABLES AND FIGURES NOT SHOWN IN MAIN TEXT ABOVE:\n{floating}")
    lic = extract_license(xml_text)
    return "\n".join(parts), lic


def safe_filename(doi):
    return doi.replace("/", "_") + ".txt"


def fetch_epmc_journal(display_name, query_name, n_needed, skip_dois, max_scan=80, min_year=None):
    saved = []
    cursor = "*"
    scanned = 0
    while len(saved) < n_needed and scanned < max_scan:
        data = epmc_search(query_name, cursor, min_year=min_year)
        records = data.get("resultList", {}).get("result", [])
        if not records:
            break
        for rec in records:
            if len(saved) >= n_needed or scanned >= max_scan:
                break
            scanned += 1
            doi, pmcid = rec.get("doi"), rec.get("pmcid")
            if not doi or not pmcid or doi in skip_dois:
                continue
            xml = epmc_fulltext(pmcid)
            time.sleep(0.3)
            if xml is None:
                continue
            try:
                text, lic = extract_text_epmc(xml)
            except ET.ParseError:
                continue
            if not text or lic.get("license_type") != "CC-BY":
                continue
            if len(text.split()) < 2000:
                continue  # research letter / brief report, not a full-length manuscript
            dest = TOPUP_DIR / safe_filename(doi)
            dest.write_text(text, encoding="utf-8")
            saved.append(doi)
            print(f"  [{display_name}] {doi}: saved ({len(saved)}/{n_needed})", file=sys.stderr)
        cursor = data.get("nextCursorMark")
        if not cursor:
            break
    if len(saved) < n_needed:
        print(f"  [{display_name}] WARNING: only found {len(saved)}/{n_needed} after scanning {scanned}", file=sys.stderr)
    return saved


def springer_list_dois(journal_id, start, page_size):
    resp = session.get(
        "https://api.springernature.com/openaccess/json",
        params={"q": f"journalid:{journal_id}", "p": page_size, "s": start, "api_key": ft.API_KEY},
        timeout=30,
    )
    resp.raise_for_status()
    return [rec["doi"] for rec in resp.json().get("records", [])]


def fetch_springer_journal(display_name, journal_id, n_needed, skip_dois, max_scan=80):
    saved = []
    start = 1
    scanned = 0
    page_size = 25
    while len(saved) < n_needed and scanned < max_scan:
        dois = springer_list_dois(journal_id, start, page_size)
        if not dois:
            break
        for doi in dois:
            if len(saved) >= n_needed or scanned >= max_scan:
                break
            scanned += 1
            if doi in skip_dois:
                continue
            try:
                xml_text = ft.fetch_jats(doi)
                text = ft.extract_text(xml_text)
            except (requests.RequestException, ET.ParseError):
                continue
            time.sleep(0.3)
            if not text:
                continue
            lic = extract_license(xml_text)
            if lic.get("license_type") != "CC-BY":
                continue
            dest = TOPUP_DIR / safe_filename(doi)
            dest.write_text(text, encoding="utf-8")
            saved.append(doi)
            print(f"  [{display_name}] {doi}: saved ({len(saved)}/{n_needed})", file=sys.stderr)
        start += page_size
    if len(saved) < n_needed:
        print(f"  [{display_name}] WARNING: only found {len(saved)}/{n_needed} after scanning {scanned}", file=sys.stderr)
    return saved


def main():
    TOPUP_DIR.mkdir(exist_ok=True)
    skip = existing_dois()
    print(f"{len(skip)} existing corpus DOIs will be skipped as dupes.", file=sys.stderr)

    all_saved = {}
    for display_name, (query_name, n_needed) in EPMC_NEEDED.items():
        all_saved[display_name] = fetch_epmc_journal(display_name, query_name, n_needed, skip)

    for journal_id, (display_name, n_needed) in SPRINGER_NEEDED.items():
        all_saved[display_name] = fetch_springer_journal(display_name, journal_id, n_needed, skip)

    total = sum(len(v) for v in all_saved.values())
    print(f"\nDone. {total} new CC-BY papers saved to {TOPUP_DIR}", file=sys.stderr)
    for j, dois in all_saved.items():
        print(f"  {j}: {len(dois)}", file=sys.stderr)

    (ROOT / "results" / "ccby_topup_manifest.json").write_text(json.dumps(all_saved, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
