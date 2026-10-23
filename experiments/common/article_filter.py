"""Shared filter to exclude non-research items (corrections, errata, etc.)
from the corpus at fetch time, so downstream experiments never need a post
hoc cleanup pass."""

import re

NON_RESEARCH_PATTERN = re.compile(
    r"^\s*(author correction|publisher correction|correction|corrigendum|erratum|"
    r"retraction(?: note)?|expression of concern|editorial note)\b",
    re.IGNORECASE,
)


def is_non_research(title):
    """True if the title marks this as a correction/erratum/retraction notice
    rather than an original research article."""
    return bool(title) and bool(NON_RESEARCH_PATTERN.match(title.strip()))


def extract_floating_content(article, body_el):
    """Text for every <table-wrap> or <fig> in the article that is NOT already
    a descendant of body_el (JATS sometimes places floats in a separate
    floats-group, or elsewhere, outside the main body flow -- itertext() on
    body_el alone silently drops these). Returns label + caption + any table
    cell text, one float per paragraph, or "" if nothing is missing."""
    body_ids = set(id(e) for e in body_el.iter()) if body_el is not None else set()
    pieces = []
    for tag in ("table-wrap", "fig"):
        for el in article.findall(f".//{tag}"):
            if id(el) in body_ids:
                continue
            label_el = el.find("label")
            label = "".join(label_el.itertext()).strip() if label_el is not None else ""
            caption_el = el.find("caption")
            caption = "".join(caption_el.itertext()).strip() if caption_el is not None else ""
            table_el = el.find(".//table")
            table_text = " ".join("".join(table_el.itertext()).split()) if table_el is not None else ""
            piece = " ".join(x for x in (label, caption, table_text) if x)
            if piece:
                pieces.append(piece)
    return "\n\n".join(pieces)


# JATS subject headings that mark a short-format or non-research item even when
# article-type says "research-article" (e.g. JAMA research letters, PLOS primers).
SHORT_OR_NON_RESEARCH_HEADINGS = {
    "research letter", "letter", "brief report", "brief communication", "short report",
    "short reports", "commentary", "perspective", "primer", "essay", "insight", "insights",
    "editorial", "viewpoint", "correspondence",
}

# Section titles that are back matter, not argument. Europe PMC's JATS often nests
# these inside <body>, which inflates a naive body word count by thousands of words.
_BACK_MATTER_TITLE = re.compile(
    r"^(references?|acknowledge?ments?|contributor information|declarations?|associated data|"
    r"supplementary (material|data|information)s?|funding( statement| information)?|footnotes?|"
    r"data,? (code,? )?(and materials )?availability.*|data availability.*|additional information|"
    r"author contributions?|competing interests?|conflicts? of interest|ethics.*|appendix.*|abbreviations)$",
    re.IGNORECASE,
)


def jats_article_info(article):
    """(article-type attribute, lower-cased subject headings, sorted pub-date years)
    for a JATS <article> element."""
    headings = [
        "".join(s.itertext()).strip().lower()
        for g in article.findall(".//article-categories//subj-group")
        if g.get("subj-group-type") in ("heading", "display-channel", "article-type")
        for s in g.findall("subject")
    ]
    years = sorted({int(y) for d in article.findall(".//pub-date") if (y := d.findtext("year"))})
    return article.get("article-type"), headings, years


def core_body_words(article):
    """Words in <body>, excluding back-matter sections (references, declarations,
    contributor lists, supplementary-material stubs) that some sources nest there."""
    body = article.find(".//body")
    if body is None:
        return 0
    drop = set()
    for sec in body.iter("sec"):
        title = sec.find("title")
        if title is not None and _BACK_MATTER_TITLE.match("".join(title.itertext()).strip()):
            drop.update(id(e) for e in sec.iter())
    for tag in ("ref-list", "supplementary-material", "ack"):
        for el in body.iter(tag):
            drop.update(id(e) for e in el.iter())
    words = 0
    for el in body.iter():
        if id(el) in drop:
            continue
        words += len((el.text or "").split())
        words += sum(len((ch.tail or "").split()) for ch in el if id(ch) not in drop)
    return words


def is_full_research_article(article, min_words=2000, year=None, allow_brief_reports=False):
    """True for a full-length original research article: JATS article-type
    research-article, no short-format/non-research heading, at least min_words of
    core body text, and (if year is given) first published in that year. With
    allow_brief_reports, a brief report clearing the same word floor also counts."""
    article_type, headings, years = jats_article_info(article)
    allowed_types = {"research-article", "brief-report"} if allow_brief_reports else {"research-article"}
    excluded = SHORT_OR_NON_RESEARCH_HEADINGS - ({"brief report"} if allow_brief_reports else set())
    if article_type not in allowed_types:
        return False
    if excluded.intersection(headings):
        return False
    if year is not None and (not years or years[0] != year):
        return False
    return core_body_words(article) >= min_words
