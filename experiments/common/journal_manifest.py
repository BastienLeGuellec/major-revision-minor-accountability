"""
doi -> journal name lookup, persisted alongside papers/. Needed because DOI
prefix alone doesn't reliably identify the journal across publishers (e.g.
Cell and The Lancet are both Elsevier, both 10.1016/...).
"""

import json
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "papers" / "_journal_manifest.json"


def load():
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save(manifest):
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=0, sort_keys=True), encoding="utf-8")


def set_journal(doi, journal):
    manifest = load()
    manifest[doi] = journal
    save(manifest)


def set_many(doi_journal_pairs):
    manifest = load()
    manifest.update(dict(doi_journal_pairs))
    save(manifest)


def doi_for_stem(stem):
    """Recover a doi from a papers/<stem>.txt filename stem. Filenames are
    built as doi.replace('/', '_'), which is ambiguous to reverse for DOIs
    with more than one '/' (e.g. European Heart Journal:
    10.1093/eurheartj/ehXXXXX). Resolve unambiguously via the manifest first;
    fall back to replacing only the first underscore, which is correct for
    every single-slash DOI (Nature family, most Europe PMC journals)."""
    manifest = load()
    for doi in manifest:
        if doi.replace("/", "_") == stem:
            return doi
    return stem.replace("_", "/", 1)
