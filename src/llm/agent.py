from __future__ import annotations
"""
Loop agent: enhanced claude -p subprocess that gives Claude a temp JSON file
of pre-fetched results and two CLI search tools it can call via Bash.

Claude's internal agentic loop (Read → reason → Bash → repeat) handles
iterative search autonomously. From this module's perspective the agent ran
or it didn't — agent_iterations is 1 on success, 0 on any failure.

Falls back transparently to summarize() if the claude CLI is not on PATH
or if any failure occurs.
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

log = logging.getLogger(__name__)

AGENT_TIMEOUT = 60    # seconds — longer than summarize (30s) to allow tool calls
TOP_N_INITIAL =  5    # results written to temp file


def _write_temp_file(results: list[dict]) -> Path:
    """
    Serialise initial_results to a temp JSON file so Claude can read it
    with its built-in Read tool.

    Strips embedding vectors (too large / not useful) and converts
    datetime fields to ISO strings for JSON serialisation.
    """
    tmp_path = Path(f"/tmp/autism_agent_{uuid.uuid4().hex}.json")
    serialisable = []
    for r in results[:TOP_N_INITIAL]:
        item = {}
        for k, v in r.items():
            if k == "embedding":
                continue                    # skip raw embedding vectors
            if hasattr(v, "isoformat"):
                item[k] = v.isoformat()     # datetime → ISO string
            else:
                item[k] = v
        serialisable.append(item)
    tmp_path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
    return tmp_path


def _build_prompt(
    query: str,
    tmp_path: Path,
    python_exe: str,
    log_context: str | None = None,
) -> str:
    """
    Build the claude -p prompt. Instructs Claude to:
      1. Read the temp JSON file using its built-in Read tool
      2. Optionally call CLI search tools via Bash if results are weak
      3. Produce a 3–6 sentence answer citing sources by number
    """
    from ..safety import SAFETY_PROMPT

    prompt = (
        "You are a medical and scientific information assistant specialising in autism (ASD).\n\n"
        f"User question: {query}\n\n"
        f"Initial search results are in the file: {tmp_path}\n"
        "Read that file first using your Read tool.\n\n"
        "If those results do not fully answer the question, you may call these CLI tools via Bash:\n"
        f"  {python_exe} -m src.tools.search \"<query>\"   — hybrid local DB search\n"
        f"  {python_exe} -m src.tools.pubmed \"<query>\"   — live PubMed search\n\n"
    )

    if log_context:
        prompt += (
            f"User's recent log context (last 30 days):\n{log_context}\n"
            "Where relevant, reference this data in your answer using phrases like "
            "'in your child's case' or 'your logs show'.\n\n"
        )

    prompt += (
        f"{SAFETY_PROMPT}\n\n"
        "Guidelines:\n"
        "  - Strongly prefer authoritative sources (PubMed, CDC, NIH, WHO, medical journals)\n"
        "  - Cite sources by number [1], [2], etc.\n"
        "  - Answer in 3–6 sentences\n"
        "  - Never hallucinate citations\n"
        "  - If only community sources are available, note that official sources were not found\n\n"
        "Produce your final answer now."
    )
    return prompt


async def run_agent(
    query: str,
    initial_results: list[dict],
    pool,               # not used directly — CLI tools create their own pool
    fetch_limit: int = 10,
    log_context: str | None = None,
) -> tuple[str | None, int]:
    """
    Run the enhanced claude -p agent loop.

    Writes initial_results to a temp JSON file, then invokes `claude -p`
    with Read + Bash tools enabled. Claude autonomously decides whether to
    call the CLI search wrappers before producing a final answer.

    Returns:
        (summary_text, agent_iterations)
        agent_iterations = 1 when agent ran successfully
        agent_iterations = 0 on any failure (caller should fall back to summarize())
    """
    tmp_path: Path | None = None
    try:
        tmp_path = _write_temp_file(initial_results)
        prompt   = _build_prompt(query, tmp_path, sys.executable, log_context)

        env = os.environ.copy()
        # Claude Code sets CLAUDECODE=1 to detect when it is already running
        # inside a Claude Code session. Unsetting it allows a nested `claude -p`
        # subprocess to start without refusing due to re-entrancy detection.
        env.pop("CLAUDECODE", None)

        log.info("agent LAUNCH claude -p (timeout=%ds)", AGENT_TIMEOUT)
        t0 = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            # --dangerously-skip-permissions: suppresses all permission prompts
            #   so the subprocess never blocks waiting for interactive input.
            # --tools Read,Bash: restricts the available toolset to ONLY Read and
            #   Bash. Unlike --allowedTools (which only pre-approves), --tools
            #   removes Edit/Write/etc. entirely so Claude cannot touch server files.
            "claude", "--dangerously-skip-permissions",
            "--tools", "Read,Bash",
            "-p", prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        log.info("agent WAITING pid=%s", proc.pid)
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=AGENT_TIMEOUT
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()   # drain pipes — avoid zombie process
            log.warning(
                "agent TIMEOUT claude -p exceeded %ds after %dms",
                AGENT_TIMEOUT, int((time.monotonic() - t0) * 1000),
            )
            return None, 0

        elapsed = int((time.monotonic() - t0) * 1000)
        if proc.returncode != 0:
            log.warning(
                "agent FAIL claude -p exit=%d elapsed=%dms stderr=%r",
                proc.returncode, elapsed,
                stderr.decode(errors="replace").strip()[:200],
            )
            return None, 0

        output = stdout.decode(errors="replace").strip()
        if not output:
            log.warning("agent FAIL claude -p empty output elapsed=%dms", elapsed)
            return None, 0

        log.info("agent OK chars=%d elapsed=%dms", len(output), elapsed)
        # agent_iterations = 1 signals "agent ran"; the internal loop count is
        # opaque when using claude -p and is not accessible from the subprocess.
        return output, 1

    except FileNotFoundError:
        log.warning("agent SKIP claude CLI not found — is 'claude' on PATH?")
        return None, 0
    except Exception as e:
        log.warning("agent UNEXPECTED %s — falling back to summarize()", e)
        return None, 0
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)   # always clean up temp file
