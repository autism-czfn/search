from __future__ import annotations
"""
Query transformation: converts a raw user question into focused search keywords
using `claude -p`. This dramatically improves recall for live search by replacing
verbose natural-language questions with concise autism-domain keyword phrases.

Example:
  IN:  "how to deal with food pattern in my autistic child?"
  OUT: "autism food selectivity eating mealtime behavior"

Falls back to the original query on any failure (CLI not found, timeout, etc.)
so the search pipeline always continues.
"""

import asyncio
import logging
import os
import re

log = logging.getLogger(__name__)

_TRANSFORM_TIMEOUT = 10.0  # seconds — fast prompt, no tools needed

_SYSTEM_PROMPT = """\
You are a search query optimizer for an autism information service.
Convert the user's question into 5-8 focused search keywords. Rules:
- Output ONLY the keywords, no explanation, no punctuation, no numbering
- Always include "autism" or "ASD" as the first keyword
- Use specific clinical/behavioral terms (e.g. food selectivity, mealtime, stimming, ABA)
- Omit filler words (how, what, why, my, child, help, deal, etc.)
- Output a single line of space-separated keywords
"""

_KEYWORD_LINE_RE = re.compile(r"^[a-zA-Z0-9 \-]+$")


def _clean_output(raw: str) -> str:
    """Extract the keyword line from claude output, stripping markdown/prose."""
    for line in raw.splitlines():
        line = line.strip().strip("`").strip("*").strip()
        if not line:
            continue
        # Skip lines that look like explanations (contain punctuation / are very long)
        if len(line) > 150:
            continue
        if ":" in line and len(line) < 60:
            # Could be "Keywords: autism food selectivity" — strip prefix
            line = line.split(":", 1)[1].strip()
        if _KEYWORD_LINE_RE.match(line):
            return line
    return ""


async def transform_query(query: str) -> str:
    """
    Transform a natural-language question into focused search keywords via
    `claude -p`. Returns the original query on any failure.
    """
    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"User question: {query}\n"
        "Keywords:"
    )

    env = os.environ.copy()
    # Remove all Claude Code session variables to allow a nested `claude -p` subprocess.
    # Without this the child process detects it's inside an active Claude Code session
    # and may block, re-enter, or refuse to start.
    for key in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_EXECPATH",
                "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"):
        env.pop(key, None)

    try:
        proc = await asyncio.create_subprocess_exec(
            # Mirror agent.py's invocation pattern exactly — `--tools Read,Bash`
            # is required to keep claude in non-interactive -p mode; omitting it
            # causes the subprocess to hang waiting for interactive input.
            "claude", "--disable-slash-commands",
            "--dangerously-skip-permissions",
            "--tools", "Read,Bash",
            "-p", prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_TRANSFORM_TIMEOUT
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            log.warning("query_transform TIMEOUT after %.1fs — using original query", _TRANSFORM_TIMEOUT)
            return query

        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()[:200]
            log.warning("query_transform FAIL exit=%d err=%r — using original query",
                        proc.returncode, err)
            return query

        raw = stdout.decode(errors="replace").strip()
        transformed = _clean_output(raw)
        if transformed and len(transformed) > 3:
            log.info("query_transform OK %r → %r", query, transformed)
            return transformed

        log.warning("query_transform: no usable keyword line in %r — using original", raw[:100])
        return query

    except FileNotFoundError:
        log.warning("query_transform SKIP — 'claude' CLI not on PATH")
        return query
    except Exception as exc:
        log.warning("query_transform UNEXPECTED %s — using original query", exc)
        return query
