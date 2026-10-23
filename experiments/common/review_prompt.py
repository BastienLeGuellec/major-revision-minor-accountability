"""Reviewer prompt template shared by Exp 1, 2 and 3."""

from common import journal_manifest

SYSTEM_TEMPLATE = """You are a reviewer for {journal}. You will be shown the \
full text of a submitted manuscript. Write the review you would hand to the \
Editor.

Structure your review as:

SUMMARY: <2-3 sentence summary of what the paper claims to show>

MAJOR CONCERNS:
- <substantive issue that would need to be resolved before publication, or "None.">

MINOR CONCERNS:
- <smaller issues, wording, presentation>

Then end your response with exactly one line in this format, with no other \
text on that line:

DECISION: <accept|minor_revision|major_revision|reject>
"""

# Revision-round framing (manuscript section 3.2): identical to SYSTEM_TEMPLATE
# except for the opening sentence.
SECOND_ROUND_SYSTEM_TEMPLATE = """You are a reviewer for {journal}, conducting \
the second round of peer review. You will be shown the full text of a \
submitted manuscript, revised after an initial round of review. Write the \
review you would hand to the Editor.

Structure your review as:

SUMMARY: <2-3 sentence summary of what the paper claims to show>

MAJOR CONCERNS:
- <substantive issue that would need to be resolved before publication, or "None.">

MINOR CONCERNS:
- <smaller issues, wording, presentation>

Then end your response with exactly one line in this format, with no other \
text on that line:

DECISION: <accept|minor_revision|major_revision|reject>
"""

USER_TEMPLATE = """Below is the full text of a submitted manuscript. Review it.

{paper_text}
"""

JOURNAL_BY_DOI_PREFIX = {
    "10.1038/s41586": "Nature",
    "10.1038/s41591": "Nature Medicine",
    "10.1038/s41467": "Nature Communications",
    "10.1038/s41588": "Nature Genetics",
    "10.1038/s41598": "Scientific Reports",
}

DEFAULT_JOURNAL = "Nature Medicine"


def journal_for_doi(doi):
    manifest = journal_manifest.load()
    if doi in manifest:
        return manifest[doi]
    for prefix, journal in JOURNAL_BY_DOI_PREFIX.items():
        if doi.startswith(prefix):
            return journal
    return DEFAULT_JOURNAL


def build_system(journal=DEFAULT_JOURNAL):
    return SYSTEM_TEMPLATE.format(journal=journal)


def build_second_round_system(journal=DEFAULT_JOURNAL):
    return SECOND_ROUND_SYSTEM_TEMPLATE.format(journal=journal)


def build_prompt(paper_text):
    return USER_TEMPLATE.format(paper_text=paper_text)
