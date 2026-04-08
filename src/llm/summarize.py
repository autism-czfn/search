from __future__ import annotations
"""
Summarize top search results using `claude -p` subprocess.
Returns a plain-text summary string, or None on any failure.
None signals the caller to return results without a summary.
"""

import logging
import os
import subprocess

log = logging.getLogger(__name__)

SUMMARY_TIMEOUT = 30        # seconds before giving up on claude -p
DESCRIPTION_MAX = 200       # chars per result description in the prompt
TOP_N = 5                   # max results to include in prompt


def _build_prompt(query: str, results: list[dict]) -> str:
    sources = []
    for i, r in enumerate(results[:TOP_N], 1):
        title = r.get("title") or ""
        source = r.get("source") or ""
        desc = r.get("description") or ""
        if desc:
            desc = desc[:DESCRIPTION_MAX]
            sources.append(f"[{i}] {title} ({source})\n    {desc}")
        else:
            sources.append(f"[{i}] {title} ({source})")

    sources_text = "\n\n".join(sources)

    return (
        "You are a helpful assistant. Answer the user's question concisely "
        "using only the provided sources. Cite sources by number [1], [2], etc.\n\n"
        f"Question: {query}\n\n"
        f"Sources:\n{sources_text}\n\n"
        "Answer in 2–4 sentences."
    )


async def summarize(query: str, results: list[dict]) -> str | None:
    """
    Call `claude -p` with the query and top results.
    Returns the summary string, or None on any failure.
    """
    if not results:
        return None

    prompt = _build_prompt(query, results)

    try:
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)   # allow claude -p outside a nested session

        proc = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=SUMMARY_TIMEOUT,
            env=env,
        )
        if proc.returncode != 0:
            log.warning(
                "claude -p exited with code %d: %s",
                proc.returncode,
                proc.stderr.strip()[:200],
            )
            return None
        output = proc.stdout.strip()
        if not output:
            log.warning("claude -p returned empty output")
            return None
        return output

    except FileNotFoundError:
        log.warning("claude CLI not found — summary unavailable")
    except subprocess.TimeoutExpired:
        log.warning("claude -p timed out after %ds — summary unavailable", SUMMARY_TIMEOUT)
    except Exception as e:
        log.warning("claude -p unexpected error: %s — summary unavailable", e)

    return None
