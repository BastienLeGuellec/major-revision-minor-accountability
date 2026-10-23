"""
Minimal .env loader. Parses KEY=value lines (quotes stripped) and sets
os.environ, without overwriting anything already set in the real
environment. Also maps this project's .env key names to the names the
provider SDKs expect (ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY).
"""

import os

_ALIASES = {
    "CLAUDE_API_KEY": "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY": "GOOGLE_API_KEY",
}


def load_env(path):
    if not path.exists():
        return
    try:
        text = path.read_text()
    except OSError:
        # Unreadable or not a regular file (e.g. bind-mounted to /dev/null in a
        # sandbox): fall back to whatever is already in the real environment.
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
        if key in _ALIASES:
            os.environ.setdefault(_ALIASES[key], value)
