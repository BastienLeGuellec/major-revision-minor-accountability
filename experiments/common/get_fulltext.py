#!/usr/bin/env python3
"""
Fetch the full-text body (not the peer review file) of a Nature Medicine
article by DOI, via the same Springer Nature Open Access JATS endpoint used
by get_peer_reviews.py, and save plain text to papers/<doi-with-slashes-as-underscores>.txt

Usage:
    python3 get_fulltext.py                      # fetch every DOI in peer_review_index.csv
    python3 get_fulltext.py 10.1038/s41591-...    # fetch one DOI
"""

import csv
import os
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from common.env import load_env  # noqa: E402
from common.article_filter import is_non_research, extract_floating_content  # noqa: E402
load_env(ROOT / ".env")
API_KEY = os.environ["SPRINGER_NATURE_API_KEY"]
BASE = "https://api.springernature.com/openaccess/jats"
INDEX_CSV = ROOT / "peer_review_index.csv"
PAPERS_DIR = ROOT / "papers"
REQUEST_DELAY = 0.5

session = requests.Session()
session.headers.update({"User-Agent": "nature-medicine-fulltext-fetcher/1.0"})


def fetch_jats(doi):
    resp = session.get(BASE, params={"q": f"doi:{doi}", "api_key": API_KEY}, timeout=30)
    resp.raise_for_status()
    return resp.text


def extract_text(xml_text):
    root = ET.fromstring(xml_text)
    article = root.find(".//article")
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


def fetch_one(doi, overwrite=False):
    dest = PAPERS_DIR / safe_filename(doi)
    if dest.exists() and not overwrite:
        return "skipped"
    try:
        xml_text = fetch_jats(doi)
        text = extract_text(xml_text)
    except (requests.RequestException, ET.ParseError) as exc:
        print(f"  {doi}: ERROR {exc}", file=sys.stderr)
        return "error"
    if not text:
        print(f"  {doi}: no usable body (paywalled/malformed record, or a correction/erratum notice)", file=sys.stderr)
        return "no_body"
    dest.write_text(text, encoding="utf-8")
    return "saved"


def main():
    PAPERS_DIR.mkdir(exist_ok=True)

    if len(sys.argv) > 1:
        dois = sys.argv[1:]
    else:
        with INDEX_CSV.open(newline="", encoding="utf-8") as f:
            dois = [row["doi"] for row in csv.DictReader(f)]

    counts = {}
    for i, doi in enumerate(dois, 1):
        result = fetch_one(doi)
        counts[result] = counts.get(result, 0) + 1
        print(f"[{i}/{len(dois)}] {doi}: {result}", file=sys.stderr)
        if result != "skipped":
            time.sleep(REQUEST_DELAY)

    print(f"\nDone. {counts}", file=sys.stderr)


if __name__ == "__main__":
    main()
