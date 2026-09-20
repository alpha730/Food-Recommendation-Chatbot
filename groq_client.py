"""
The single place where the Groq API key and models are configured.

Nothing in this project hardcodes a key. This module reads configuration from
the environment, and - for convenience - from a `.env` file sitting next to it,
so you only ever edit ONE file:

    C:\\Res\\.env

        GROQ_API_KEY=gsk_your_key_here
        GROQ_MODEL=qwen/qwen3.8-27b             (optional)
        GROQ_CHAT_MODEL=openai/gpt-oss-20b      (optional)

A real environment variable always wins over the `.env` file, so CI or a
terminal `set GROQ_API_KEY=...` still overrides it.

What this module exports (the notebooks were written against OpenAI, so these
mirror that surface exactly):

    client      Groq client - client.chat.completions.create(...) is
                call-compatible with the OpenAI one used in M3L1/M3L2
    MODEL       model for the six specialist agents
    CHAT_MODEL  smaller/faster model for the short chat-side calls
    ChatGroq    stand-in for ChatOpenAI, with .invoke(messages) -> .content
    SystemMessage / HumanMessage
                stand-ins for the langchain_core message classes
    get_env     helper so other scripts can read .env values too
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from groq import Groq


# The lesson code prints ✓, → and emoji. Windows consoles default to cp1252 and
# raise UnicodeEncodeError on those, so switch stdout/stderr to UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# .env loading (no third-party dependency needed)
# ---------------------------------------------------------------------------

ENV_FILE = Path(__file__).resolve().parent / ".env"


def load_env_file(path: Path = ENV_FILE) -> dict:
    """Parse a simple KEY=value file. Blank lines and # comments are ignored.

    Values already present in the real environment are NOT overwritten.
    """
    values = {}
    if not path.exists():
        return values

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
            os.environ.setdefault(key, value)
    return values


_ENV = load_env_file()


def get_env(name: str, default: str = None) -> str:
    """Read a setting from the environment, falling back to the .env file."""
    return os.environ.get(name) or _ENV.get(name) or default


# ---------------------------------------------------------------------------
# Groq client and models
# ---------------------------------------------------------------------------

GROQ_API_KEY = get_env("GROQ_API_KEY")

if not GROQ_API_KEY:
    print(f"Warning: no GROQ_API_KEY found - every model call will fail.\n"
          f"         Put  GROQ_API_KEY=gsk_your_key_here  in  {ENV_FILE}\n"
          f"         (or set it as an environment variable).")

# Groq() refuses to construct without a key, so use a placeholder rather than
# crashing on import - the warning above points at the real fix.
client = Groq(api_key=GROQ_API_KEY or "missing-api-key")

MODEL = get_env("GROQ_MODEL", "qwen/qwen3.8-27b")
CHAT_MODEL = get_env("GROQ_CHAT_MODEL", "openai/gpt-oss-20b")


# ---------------------------------------------------------------------------
# Minimal stand-ins for the LangChain pieces the notebooks used
# ---------------------------------------------------------------------------
# The Lesson 3 functions call llm.invoke([SystemMessage(...), HumanMessage(...)])
# and read response.content. These classes provide exactly that against Groq.

@dataclass
class SystemMessage:
    content: str
    role: str = "system"


@dataclass
class HumanMessage:
    content: str
    role: str = "user"


@dataclass
class _Reply:
    content: str


class ChatGroq:
    """Drop-in replacement for ChatOpenAI, backed by the Groq API."""

    def __init__(self, model: str = None, temperature: float = 0.7):
        self.model = model or CHAT_MODEL
        self.temperature = temperature

    def invoke(self, messages) -> _Reply:
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return _Reply(content=response.choices[0].message.content)
