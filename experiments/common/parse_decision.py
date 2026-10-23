"""Extract a normalized decision label from a raw LLM review."""

import re

VALID = {"accept", "minor_revision", "major_revision", "reject"}

# Tolerates markdown emphasis around the label/colon, e.g. "**DECISION:** major_revision"
# or "DECISION**: major_revision" -- models don't reliably emit a bare "DECISION: x" line.
_DECISION_LINE = re.compile(r"DECISION\**:\**\s*\**([a-z_]+)", re.IGNORECASE)

# Fallback keyword matching, checked in order (most specific / most negative first)
# so a review that says both "not ready" and "interesting" still lands on reject.
_KEYWORD_RULES = [
    ("reject", [r"\breject\b", r"cannot recommend publication", r"not suitable for publication"]),
    ("major_revision", [r"major revision", r"major concerns? (?:must|need to) be addressed"]),
    ("minor_revision", [r"minor revision", r"minor concerns? (?:only|remain)"]),
    ("accept", [r"\baccept\b(?! with major)", r"no major concerns", r"ready for publication"]),
]


def parse_decision(raw_text):
    """Return (decision, method) where method is 'explicit' or 'keyword_fallback' or 'unparsed'."""
    match = _DECISION_LINE.search(raw_text)
    if match:
        label = match.group(1).strip().lower()
        if label in VALID:
            return label, "explicit"

    lowered = raw_text.lower()
    for label, patterns in _KEYWORD_RULES:
        if any(re.search(p, lowered) for p in patterns):
            return label, "keyword_fallback"

    return None, "unparsed"
