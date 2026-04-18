from __future__ import annotations
"""
Summarize top search results using `claude -p` subprocess.
Returns a plain-text summary string, or None on any failure.
None signals the caller to return results without a summary.
"""

import asyncio
import logging
import os
import time

log = logging.getLogger(__name__)

SUMMARY_TIMEOUT = 30        # seconds before giving up on claude -p
DESCRIPTION_MAX = 200       # chars per result description in the prompt
TOP_N = 5                   # max results to include in prompt


def _build_prompt(
    query: str,
    results: list[dict],
    log_context: str | None = None,
) -> str:
    from ..safety import SAFETY_PROMPT

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

    prompt = (
        "You are a medical and scientific information assistant. "
        "Answer the user's question accurately and concisely using only the provided sources. "
        "Strongly prefer information from official or scientific sources "
        "(e.g. CDC, NIH, PubMed, WHO, medical journals, government health agencies) "
        "over anecdotal or community posts (e.g. Reddit, forums, personal blogs). "
        "If only community sources are available, mention that official sources were not found. "
        f"{SAFETY_PROMPT} "
        "Cite sources by number [1], [2], etc.\n\n"
        f"Question: {query}\n\n"
    )

    if log_context:
        prompt += (
            f"User's recent log context (last 30 days):\n{log_context}\n"
            "Where relevant, reference this data in your answer using phrases like "
            "'in your child's case' or 'your logs show'.\n\n"
        )

    prompt += f"Sources:\n{sources_text}\n\nAnswer in 3–5 sentences based on the most authoritative sources available."
    return prompt


async def summarize(query: str, results: list[dict], log_context: str | None = None) -> str | None:
    """
    Call `claude -p` with the query and top results.
    Returns the summary string, or None on any failure.
    """
    if not results:
        return None

    prompt = _build_prompt(query, results, log_context)

    try:
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)   # allow claude -p outside a nested session

        log.info("summarize LAUNCH claude -p (timeout=%ds prompt_chars=%d)", SUMMARY_TIMEOUT, len(prompt))
        t0 = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            "claude", "--disable-slash-commands", "-p", prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        log.info("summarize WAITING pid=%s", proc.pid)
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=SUMMARY_TIMEOUT
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()   # drain pipes to avoid zombie
            log.warning(
                "summarize TIMEOUT claude -p exceeded %ds after %dms — summary unavailable",
                SUMMARY_TIMEOUT, int((time.monotonic() - t0) * 1000),
            )
            return None

        elapsed = int((time.monotonic() - t0) * 1000)
        if proc.returncode != 0:
            log.warning(
                "summarize FAIL claude -p exit=%d elapsed=%dms stderr=%r",
                proc.returncode,
                elapsed,
                stderr.decode(errors="replace").strip()[:200],
            )
            return None
        output = stdout.decode(errors="replace").strip()
        if not output:
            log.warning("summarize FAIL claude -p empty output elapsed=%dms", elapsed)
            return None
        log.info("summarize OK chars=%d elapsed=%dms", len(output), elapsed)
        return output

    except FileNotFoundError:
        log.warning("summarize FAIL claude CLI not found — is 'claude' on PATH?")
    except Exception as e:
        log.warning("summarize FAIL unexpected error: %s — summary unavailable", e)

    return None
