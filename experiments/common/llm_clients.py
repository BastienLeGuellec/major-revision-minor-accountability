"""
Uniform wrapper around whichever LLM provider SDKs are installed and
configured. Add an entry to MODELS to make a model available to every
experiment script.

Reads API keys from the environment:
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY

None of these are required to *load* this module -- only to actually call
that provider. Call available_models() to see what you can currently run.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from common.env import load_env

load_env(ROOT / ".env")

MODELS = {
    "claude-opus-5": "anthropic",
    "claude-sonnet-5": "anthropic",
    "gpt-5.1": "openai",
    "gemini-3.1-pro-preview": "google",
    "z-ai/glm-5.3-flash": "openrouter",
    "google/gemini-3.7-flash": "openrouter",
    "openai/gpt-5.6-sol": "openrouter",
    "anthropic/claude-opus-5": "openrouter",
    "deepseek/deepseek-v4-flash-0731": "openrouter",
    "x-ai/grok-4.6": "openrouter",
    "minimax/minimax-m3": "openrouter",
}

# The four-model reviewer panel used in every experiment of the manuscript.
PANEL_MODELS = [
    "z-ai/glm-5.3-flash",
    "openai/gpt-5.6-sol",
    "google/gemini-3.7-flash",
    "deepseek/deepseek-v4-flash-0731",
]

_PROVIDER_ENV_VAR = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def available_models():
    return [m for m, provider in MODELS.items() if os.environ.get(_PROVIDER_ENV_VAR[provider])]


def review(model, system, prompt, max_tokens=24000, temperature=0.7, reasoning_effort=None, timeout=120):
    """Call `model` with a system prompt + user prompt, return the raw text response.

    reasoning_effort (openrouter only): "low" | "medium" | "high". Pass "low" for
    short, mechanical tasks (classification, targeted extraction) where letting the
    model reason at full depth just adds minutes of latency for no quality gain.
    Ignored for non-openrouter providers.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}. Known: {sorted(MODELS)}")
    provider = MODELS[model]
    if provider == "anthropic":
        return _call_anthropic(model, system, prompt, max_tokens, temperature)
    if provider == "openai":
        return _call_openai(model, system, prompt, max_tokens, temperature)
    if provider == "google":
        return _call_google(model, system, prompt, max_tokens, temperature)
    if provider == "openrouter":
        return _call_openrouter(model, system, prompt, max_tokens, temperature, reasoning_effort, timeout)
    raise ValueError(f"Unhandled provider {provider!r}")


def _call_anthropic(model, system, prompt, max_tokens, temperature):
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in msg.content if block.type == "text")


def _call_openai(model, system, prompt, max_tokens, temperature):
    import openai

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content or ""


def _call_google(model, system, prompt, max_tokens, temperature):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    return resp.text


def _call_openrouter(model, system, prompt, max_tokens, temperature, reasoning_effort=None, timeout=120):
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1", timeout=timeout)
    if reasoning_effort:
        reasoning_config = {"effort": reasoning_effort}
    else:
        # Reasoning models (GLM, some Gemini/GPT routes) can burn the entire
        # max_tokens budget on their internal reasoning trace and get cut off
        # (finish_reason="length") before ever writing `content`. Capping
        # reasoning tokens guarantees room is left for the actual review.
        reasoning_config = {"max_tokens": max_tokens // 3}
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        extra_body={"reasoning": reasoning_config},
    )
    choice = resp.choices[0]
    content = choice.message.content or ""
    if not content:
        reasoning = getattr(choice.message, "reasoning", None) or ""
        raise RuntimeError(
            f"{model}: empty content (finish_reason={choice.finish_reason}, "
            f"reasoning_chars={len(reasoning)})"
        )
    return content
