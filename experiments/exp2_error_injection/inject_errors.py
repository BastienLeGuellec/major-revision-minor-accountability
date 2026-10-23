"""
Error injection for Exp 2: plant ONE critical, hard-to-spot flaw per paper,
from one of two categories.

The injector model should NOT be one of the models under test in Exp 1/3, so
its own review-writing tendencies don't leak into what gets planted.

The injector returns only the flaw description plus a short verbatim span to
replace (or delete) -- NOT the whole rewritten paper. Asking a model to
regenerate an entire 5-15k word paper inside a JSON string is slow, expensive,
and prone to breaking on JSON-escaping of a huge blob; a small, targeted edit
applied programmatically in Python is fast and robust.
"""

import json
import re

from common import llm_clients

INJECTOR_MODEL = "x-ai/grok-4.6"

CATEGORIES = ["statistical", "methodological"]

CATEGORY_GUIDANCE = {
    "statistical": """Category: STATISTICAL. Broadly, the analysis applied doesn't really
match the structure of the data or the design -- the kind of thing a careful statistical
reviewer would flag. Use your own judgment about what fits this paper; a few loose
examples of the shape this can take (don't feel bound by them, and don't reuse them
verbatim): an analysis that assumes independent observations when the data are clustered
or repeated; multiple testing with no correction or disclosure; ignoring censoring;
unmodeled confounding or stratification. Pick whatever's most natural and well-hidden for
this specific paper's actual data and analysis.""",
    "methodological": """Category: METHODOLOGICAL. Broadly, either the outcome was produced
by someone or something with undisclosed access to information that could bias the
judgment, or the instrument/reagent/measurement doesn't quite isolate what's claimed. Use
your own judgment about what fits this paper; a few loose examples of the shape this can
take (don't feel bound by them, and don't reuse them verbatim): an unblinded assessor
scoring a subjective outcome; ground truth produced by someone who'd already seen the
result being evaluated; a reagent or instrument with a known specificity/validity
limitation used without disclosing it. Pick whatever's most natural and well-hidden for
this specific paper's actual method.""",
}

INJECTOR_SYSTEM_TEMPLATE = """You are helping construct a dataset for evaluating peer reviewers' \
ability to catch critical flaws. You will be given the full text of a published, \
presumably sound paper.

Your task:
1. Identify the paper's central conclusion -- the one claim the abstract and title are
   built around.
2. Design one critical flaw to introduce into the paper that, if a reader noticed it,
   would undermine that central conclusion. It should fit this general category:

{category_guidance}

   General guidance, not a rigid checklist -- use judgment:
   - It should be a real, substantive problem with the study, specific to this paper's
     actual field, data and method -- not a generic template applied on autopilot.
   - It should be serious enough that a competent reviewer who noticed it would have to
     recommend major revision or rejection, not just note it as a minor concern. If the
     flaw you're considering would only prompt a passing comment, it's not the one --
     look for something that actually threatens the central conclusion.
   - It should be hard to spot: don't flag, hedge, or apologize for it anywhere in the
     text, and don't tip it off with suspicious phrasing. It should just read as a normal,
     confidently-written paper.
   - Introduce it with as small an edit as reasonably possible -- typically replacing or
     removing a short existing passage (roughly a sentence or few) is enough; you don't
     need to touch anything else.

3. Quote the passage you're changing or removing exactly as it appears in the manuscript
   below, character for character -- this will be used for an automated find-and-replace,
   so it has to match verbatim. Then give the replacement text (or leave it empty if the
   passage should just be deleted).

Respond with ONLY a JSON object (no markdown fences, no commentary):
{{
  "central_conclusion": "<one sentence: the paper's main claim>",
  "flaw_summary": "<one or two sentences: what the flaw is and why it undermines the central conclusion>",
  "original_span": "<the exact verbatim passage from the manuscript to change or remove>",
  "replacement_span": "<the replacement text, or an empty string if the original_span should simply be deleted>"
}}
"""


def _inject_once(paper_text, model, category, timeout):
    system = INJECTOR_SYSTEM_TEMPLATE.format(category_guidance=CATEGORY_GUIDANCE[category])
    prompt = f"Manuscript:\n\n{paper_text}"
    try:
        raw = llm_clients.review(
            model, system, prompt, max_tokens=16000, temperature=0.9,
            reasoning_effort="low", timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 -- provider/route flakiness (empty content, timeouts, etc.)
        return None, f"call failed: {exc}"

    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, "invalid JSON"

    required = {"central_conclusion", "flaw_summary", "original_span", "replacement_span"}
    if not required.issubset(data):
        return None, f"missing keys (got {sorted(data.keys())})"

    original_span = data["original_span"]
    if original_span not in paper_text:
        # model paraphrased instead of quoting verbatim -- can't safely apply the edit
        return None, "original_span not verbatim in paper"

    modified_text = paper_text.replace(original_span, data["replacement_span"], 1)
    data["flaw_location"] = f"{data['original_span']!r} -> {data['replacement_span']!r}"
    data["category"] = category
    return (modified_text, data), None


def inject(paper_text, category, model=INJECTOR_MODEL, attempts=3, timeout=90, on_attempt=None):
    """Try up to `attempts` times (fresh sample each time -- temperature=0.9) since
    injector calls are inherently flaky: providers can return empty content, drift
    from the requested JSON schema, or paraphrase instead of quoting verbatim.
    `category` must be one of CATEGORIES. `on_attempt(attempt_index, error_or_None)`
    is called after each try, for logging.
    """
    if category not in CATEGORIES:
        raise ValueError(f"category must be one of {CATEGORIES}, got {category!r}")
    for i in range(attempts):
        result, error = _inject_once(paper_text, model, category, timeout)
        if on_attempt:
            on_attempt(i, error)
        if result is not None:
            return result
    return None
