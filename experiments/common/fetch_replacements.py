#!/usr/bin/env python3
"""
Fetch replacement candidates for corpus items that fail the inclusion criteria
(commentaries, editorials, perspectives, primers, research letters, brief
reports under 2,000 words, items first published before 2026), plus a full
10-paper set for each candidate journal to stand in for the Journal of Clinical
Investigation, 8 of whose 10 corpus items were not full research articles.

Every candidate must be: CC-BY (per the JATS <license>), JATS article-type
research-article with no short-format/non-research heading, at least 2,000
words of core body text (back matter excluded), and first published in 2026.
The most recent qualifying items are taken first. Two spares per journal are
kept so an odd pick can be swapped without re-running.

Saves to papers/_replacements/ (NOT papers/ directly) with a _manifest.json
recording what each candidate would replace; nothing touches the live corpus
or results/ until the candidates are reviewed.

Usage:
    python3 fetch_replacements.py
    python3 fetch_replacements.py --only "JAMA Oncology"   # top up one entry of the manifest
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from common.article_filter import is_full_research_article, jats_article_info, core_body_words  # noqa: E402
from common.check_licenses import extract_license  # noqa: E402
from common.fetch_ccby_topup import epmc_search, epmc_fulltext, extract_text_epmc, safe_filename  # noqa: E402

PAPERS_DIR = ROOT / "papers"
OUT_DIR = PAPERS_DIR / "_replacements"
YEAR = 2026
SPARES = 2

# journal display name -> (Europe PMC JOURNAL query value, DOI pattern, corpus DOIs to replace)
SAME_JOURNAL = {
    "Circulation": ("Circulation", r"^10\.1161/circulationaha\.", [
        "10.1161/circulationaha.125.076330", "10.1161/circulationaha.126.080380"]),
    "European Heart Journal": ("Eur Heart J", r"^10\.1093/eurheartj/", [
        "10.1093/eurheartj/ehaf724", "10.1093/eurheartj/ehag054",
        "10.1093/eurheartj/ehag129", "10.1093/eurheartj/ehag215"]),
    "JAMA Internal Medicine": ("JAMA Intern Med", r"^10\.1001/jamainternmed\.", [
        "10.1001/jamainternmed.2026.0493", "10.1001/jamainternmed.2026.1370"]),
    "JAMA Oncology": ("JAMA Oncol", r"^10\.1001/jamaoncol\.", [
        "10.1001/jamaoncol.2025.5511", "10.1001/jamaoncol.2025.5785",
        "10.1001/jamaoncol.2025.6430", "10.1001/jamaoncol.2026.0136"]),
    "PLoS Biology": ("PLoS Biol", r"^10\.1371/journal\.pbio\.", [
        "10.1371/journal.pbio.3003825", "10.1371/journal.pbio.3003839",
        "10.1371/journal.pbio.3003846", "10.1371/journal.pbio.3003853"]),
    "PLoS Medicine": ("PLoS Med", r"^10\.1371/journal\.pmed\.", [
        "10.1371/journal.pmed.1005179"]),
    "eLife": ("eLife", r"^10\.7554/elife\.", [
        "10.7554/elife.112100"]),
}

# candidate journals to replace the Journal of Clinical Investigation wholesale
NEW_JOURNAL_CANDIDATES = {
    "Journal of Experimental Medicine": ("J Exp Med", r"^10\.1084/jem\."),
    "EMBO Molecular Medicine": ("EMBO Mol Med", r"^10\.(1038/s44321|15252/emmm)"),
    "JCI Insight": ("JCI Insight", r"^10\.1172/jci\.insight\."),
}
JCI_DOIS_TO_REPLACE = 10

# journals too thin in 2026 CC-BY full research articles to fill every slot;
# here a brief report clearing the same 2,000-word floor is accepted too
BRIEF_REPORTS_OK = {"JAMA Oncology"}


def known_stems():
    """Filename stems of every paper anywhere under papers/ (live corpus, archives,
    earlier top-ups), so no candidate duplicates something already handled."""
    return {p.stem for p in PAPERS_DIR.rglob("*.txt")}


def fetch_journal(display_name, query_name, doi_pattern, n_needed, skip_stems, max_scan=300):
    saved = []
    cursor = "*"
    scanned = 0
    pattern = re.compile(doi_pattern, re.IGNORECASE)
    while len(saved) < n_needed and scanned < max_scan:
        data = epmc_search(query_name, cursor, min_year=YEAR)
        records = data.get("resultList", {}).get("result", [])
        if not records:
            break
        for rec in records:
            if len(saved) >= n_needed or scanned >= max_scan:
                break
            scanned += 1
            doi, pmcid = (rec.get("doi") or "").lower(), rec.get("pmcid")
            if not pmcid or (doi and (not pattern.match(doi) or safe_filename(doi)[:-4] in skip_stems)):
                continue
            xml = epmc_fulltext(pmcid)
            time.sleep(0.3)
            if xml is None:
                continue
            try:
                root = ET.fromstring(xml)
                text, lic = extract_text_epmc(xml)
            except ET.ParseError:
                continue
            article = root if root.tag == "article" else root.find(".//article")
            if not doi:  # search records sometimes omit it; the JATS always carries one
                doi = (article.findtext(".//article-meta/article-id[@pub-id-type='doi']") or "").strip().lower()
                if not doi or not pattern.match(doi) or safe_filename(doi)[:-4] in skip_stems:
                    continue
            if not text or lic.get("license_type") != "CC-BY" or not is_full_research_article(
                    article, year=YEAR, allow_brief_reports=display_name in BRIEF_REPORTS_OK):
                continue
            article_type, headings, years = jats_article_info(article)
            (OUT_DIR / safe_filename(doi)).write_text(text, encoding="utf-8")
            skip_stems.add(safe_filename(doi)[:-4])
            saved.append({
                "doi": doi, "journal": display_name, "pmcid": pmcid,
                "title": text.split("\n", 1)[0].removeprefix("TITLE: "),
                "first_pub_date": rec.get("firstPublicationDate"),
                "article_type": article_type, "headings": headings,
                "core_body_words": core_body_words(article),
            })
            print(f"  [{display_name}] {doi}: saved ({len(saved)}/{n_needed})", file=sys.stderr)
        cursor = data.get("nextCursorMark")
        if not cursor:
            break
    if len(saved) < n_needed:
        print(f"  [{display_name}] WARNING: only found {len(saved)}/{n_needed} after scanning {scanned}", file=sys.stderr)
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated journal display names; tops up just those entries of an existing manifest")
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    OUT_DIR.mkdir(exist_ok=True)
    skip = known_stems()
    manifest_path = OUT_DIR / "_manifest.json"
    if only and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"criteria": {"license": "CC-BY", "article_type": "research-article",
                                 "min_core_body_words": 2000, "first_published": YEAR,
                                 "brief_reports_ok": sorted(BRIEF_REPORTS_OK)},
                    "same_journal": {}, "new_journal_candidates": {}}

    for display_name, (query_name, pattern, old_dois) in SAME_JOURNAL.items():
        if only and display_name not in only:
            continue
        entry = manifest["same_journal"].setdefault(display_name, {"replaces": old_dois, "candidates": []})
        n_needed = len(old_dois) + SPARES - len(entry["candidates"])
        entry["candidates"] += fetch_journal(display_name, query_name, pattern, n_needed, skip)

    for display_name, (query_name, pattern) in NEW_JOURNAL_CANDIDATES.items():
        if only and display_name not in only:
            continue
        entry = manifest["new_journal_candidates"].setdefault(
            display_name, {"replaces": "Journal of Clinical Investigation", "candidates": []})
        n_needed = JCI_DOIS_TO_REPLACE + SPARES - len(entry["candidates"])
        entry["candidates"] += fetch_journal(display_name, query_name, pattern, n_needed, skip)
    manifest["criteria"]["brief_reports_ok"] = sorted(BRIEF_REPORTS_OK)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = sum(len(v["candidates"]) for k in ("same_journal", "new_journal_candidates") for v in manifest[k].values())
    print(f"\nDone. {total} candidates saved to {OUT_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
