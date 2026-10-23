#!/usr/bin/env python3
"""
Re-query the source XML for every paper in the corpus and extract the actual
JATS <permissions>/<license> block, which the extraction pipeline (get_fulltext.py,
get_europepmc_papers.py) discards when converting to plain text. The "open access"
filter used at fetch time (Springer's openaccess endpoint, Europe PMC's
OPEN_ACCESS:Y) only guarantees free-to-read, not a specific reuse license -- this
recovers the real per-article license so it can be reported accurately.

Usage:
    python3 check_licenses.py                 # all 230 papers
    python3 check_licenses.py --n 5           # quick sample
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from common.env import load_env  # noqa: E402
from common import journal_manifest  # noqa: E402

load_env(ROOT / ".env")
PAPERS_DIR = ROOT / "papers"
RESULTS_PATH = ROOT / "results" / "license_check.jsonl"

SPRINGER_API_KEY = os.environ.get("SPRINGER_NATURE_API_KEY")
SPRINGER_BASE = "https://api.springernature.com/openaccess/jats"
EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

SPRINGER_PREFIXES = ("10.1038/s41586", "10.1038/s41591", "10.1038/s41467", "10.1038/s41588", "10.1038/s41598")

session = requests.Session()
session.headers.update({"User-Agent": "license-check/1.0"})


def infer_license_type(text):
    t = text.lower()
    m = re.search(r"cc[- ]?(by(?:[- ](?:nc|nd|sa))*|0)\b", t)
    if m:
        return "CC-" + m.group(1).upper().replace(" ", "-")
    if "public domain" in t or "cc0" in t:
        return "CC0"
    if "aaas" in t or "science.org/about/science-licenses" in t:
        return "AAAS custom (non-CC)"
    has_attr = "attribution" in t or re.search(r"\bby\s*4\.0\b", t)
    nc = "noncommercial" in t or "non-commercial" in t
    nd = "noderivatives" in t or "no-derivatives" in t or "does not permit alteration" in t
    sa = "sharealike" in t or "share-alike" in t
    if has_attr:
        suffix = "".join(x for cond, x in [(nc, "-NC"), (nd, "-ND"), (sa, "-SA")] if cond)
        return "CC-BY" + suffix
    return None


def is_springer(doi):
    return doi.startswith(SPRINGER_PREFIXES)


def fetch_springer_xml(doi):
    resp = session.get(SPRINGER_BASE, params={"q": f"doi:{doi}", "api_key": SPRINGER_API_KEY}, timeout=30)
    resp.raise_for_status()
    return resp.text


def fetch_epmc_xml(doi):
    r = session.get(EPMC_SEARCH, params={"query": f'DOI:"{doi}"', "format": "json"}, timeout=30)
    r.raise_for_status()
    results = r.json().get("resultList", {}).get("result", [])
    if not results or not results[0].get("pmcid"):
        return None, None
    pmcid = results[0]["pmcid"]
    resp = session.get(EPMC_FULLTEXT.format(pmcid=pmcid), timeout=30)
    if resp.status_code == 404:
        return None, pmcid
    resp.raise_for_status()
    return resp.text, pmcid


def extract_license(xml_text):
    root = ET.fromstring(xml_text)
    article = root if root.tag == "article" else root.find(".//article")
    if article is None:
        return {}

    perms = article.find(".//permissions")
    out = {}
    if perms is not None:
        # a single <permissions> block can contain more than one <license> element
        # (e.g. a boilerplate "reprints and permissions" one plus the real CC one) --
        # examine all of them and keep whichever actually names a license.
        best_type, best_url, best_text, best_attr_type = None, None, None, None
        for lic in perms.findall("license"):
            attr_type = lic.get("license-type")
            url = None
            for k, v in lic.attrib.items():
                if k.endswith("href"):
                    url = v
            # Springer Nature / EMBO Press style: no href on <license>, the URL sits in an
            # <ali:license_ref> child, and the license-p text also names a CC0 *data* waiver
            # that text inference would otherwise pick up as the article's license.
            if url is None:
                for child in lic:
                    if child.tag.endswith("license_ref") and child.text and "creativecommons.org" in child.text:
                        url = child.text.strip()
                        break
            p_el = lic.find("license-p")
            license_text = "".join(p_el.itertext()).strip() if p_el is not None else ""
            if url and "creativecommons.org" in url:
                m = re.search(r"/licenses/([a-z-]+)/", url)
                inferred = "CC-" + m.group(1).upper() if m else infer_license_type(license_text)
            else:
                inferred = infer_license_type(license_text) if license_text else None
            if inferred and not best_type:
                best_type, best_url, best_text, best_attr_type = inferred, url, license_text, attr_type
            elif not best_text and license_text:
                best_text = license_text
        out["license_type"] = best_type
        if best_url:
            out["license_url"] = best_url
        if best_text:
            out["license_text"] = best_text
        if best_attr_type and best_attr_type != "open-access":
            out["springer_license_type_attr"] = best_attr_type
        cs = perms.find("copyright-statement")
        if cs is not None:
            out["copyright_statement"] = "".join(cs.itertext()).strip()
        holder = perms.find("copyright-holder")
        if holder is not None:
            out["copyright_holder"] = "".join(holder.itertext()).strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="limit to first n papers (else all)")
    args = ap.parse_args()

    stems = sorted(p.stem for p in PAPERS_DIR.glob("*.txt"))
    dois = [journal_manifest.doi_for_stem(s) for s in stems]
    if args.n:
        dois = dois[: args.n]

    manifest = journal_manifest.load()
    results = []
    for i, doi in enumerate(dois, 1):
        journal = manifest.get(doi, "?")
        try:
            if is_springer(doi):
                xml_text = fetch_springer_xml(doi)
                source = "springer"
            else:
                xml_text, pmcid = fetch_epmc_xml(doi)
                source = f"epmc:{pmcid}"
                if xml_text is None:
                    results.append({"doi": doi, "journal": journal, "source": source, "error": "no_fulltext"})
                    print(f"[{i}/{len(dois)}] {doi} ({journal}): no_fulltext", file=sys.stderr)
                    time.sleep(0.3)
                    continue
            lic = extract_license(xml_text)
            row = {"doi": doi, "journal": journal, "source": source, **lic}
            results.append(row)
            print(f"[{i}/{len(dois)}] {doi} ({journal}): {lic.get('license_type', 'NONE FOUND')}", file=sys.stderr)
        except Exception as exc:
            results.append({"doi": doi, "journal": journal, "error": str(exc)})
            print(f"[{i}/{len(dois)}] {doi} ({journal}): ERROR {exc}", file=sys.stderr)
        time.sleep(0.25)

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(results)} rows to {RESULTS_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
