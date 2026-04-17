from __future__ import annotations
"""
Streaming agent: runs claude -p with --output-format stream-json and yields
(event_type, payload) tuples as the agent works, so callers can forward
progress to the browser via SSE before the final answer is ready.

Yields:
  ("agent_activity", {"type": str, "message": str, "detail": str | None})
      — once per tool invocation (Read or Bash)
  ("summary",        {"text": str})
      — when the final answer is ready
  ("done_agent",     {"agent_iterations": int, "llm_ms": int})
      — always emitted after a successful summary
  ("error",          {"message": str})
      — on any failure; caller should fall back to summarize()

The claude --output-format stream-json format emits one JSON object per line:
  {"type":"system",    "subtype":"init", ...}           → ignore
  {"type":"assistant", "message":{"content":[...]}}     → inspect content blocks
      content block: {"type":"tool_use","id":"...",
                      "name":"Bash"|"Read","input":{...}}
  {"type":"user",      "message":{"content":[...]}}     → tool results; ignore
  {"type":"result",    "subtype":"success","result":"..."} → final answer
  {"type":"result",    "subtype":"error",  "error":"..."}  → agent failed

Falls back gracefully: if anything goes wrong, yields ("error", ...) and
stream.py falls back to summarize() directly.
"""

import asyncio
import json
import logging
import os
import shlex
import sys
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

log = logging.getLogger(__name__)

AGENT_TIMEOUT    = 60       # seconds
TOP_N_INITIAL    = 5        # results written to temp file


# ── Temp file + prompt helpers (inline — agent.py may not exist yet) ───────

def _write_temp_file(results: list[dict]) -> Path:
    """Serialise initial results to a temp JSON file for Claude to Read."""
    tmp_path = Path(f"/tmp/autism_agent_{uuid.uuid4().hex}.json")
    serialisable = []
    for r in results[:TOP_N_INITIAL]:
        item = {}
        for k, v in r.items():
            if k == "embedding":
                continue
            if hasattr(v, "isoformat"):
                item[k] = v.isoformat()
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
    """Build the claude -p prompt instructing the agent to read results and answer."""
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
        "  - Answer in 3-6 sentences\n"
        "  - Never hallucinate citations\n"
        "  - If only community sources are available, note that official sources were not found\n\n"
        "Produce your final answer now."
    )
    return prompt


STREAM_JSON_FLAG = "--output-format"
STREAM_JSON_VAL  = "stream-json"


# ── Tool-call classifier ───────────────────────────────────────────────────

def _classify_tool(name: str, inp: dict) -> tuple[str, str, str | None]:
    """
    Map a tool invocation to a human-readable activity event.

    Returns (activity_type, message, detail).
      activity_type: "read" | "search" | "pubmed" | "other_tool"
      message:       shown verbatim in the UI agent log
      detail:        raw command / file path (for debugging; not shown to user)
    """
    if name == "Read":
        fp = inp.get("file_path", "")
        if "autism_agent_" in fp:
            return "read", "Agent read initial search results", fp
        return "read", f"Agent read file: {fp}", fp

    if name == "Bash":
        cmd = inp.get("command", "")

        # Extract the query string from the CLI call.
        # Pattern: python -m <module> "<query>" [--flags ...]
        # The query is the token immediately after the module name — NOT parts[-1]
        # which would be the last flag value (e.g. "5" from "--max 5").
        def _extract_query(cmd: str, module_names: set[str]) -> str:
            try:
                parts = shlex.split(cmd)
                for i, tok in enumerate(parts):
                    if tok in module_names and i + 1 < len(parts):
                        return parts[i + 1]
            except ValueError:
                pass
            return cmd   # fallback: show raw command

        if "src.tools.pubmed" in cmd or "tools.pubmed" in cmd:
            q = _extract_query(cmd, {"src.tools.pubmed", "tools.pubmed"})
            return "pubmed", f'Agent searched PubMed for "{q}"', cmd

        if "src.tools.search" in cmd or "tools.search" in cmd:
            q = _extract_query(cmd, {"src.tools.search", "tools.search"})
            return "search", f'Agent ran hybrid search for "{q}"', cmd

        return "other_tool", f"Agent ran: {cmd[:120]}", cmd

    return "other_tool", f"Agent called tool: {name}", None


# ── Main streaming generator ───────────────────────────────────────────────

async def run_agent_stream(
    query: str,
    initial_results: list[dict],
    pool,                   # unused directly — CLI tools create their own pools
    fetch_limit: int = 10,
    log_context: str | None = None,
) -> AsyncGenerator[tuple[str, dict], None]:
    """
    Async generator that streams agent activity as (event_type, payload) tuples.

    The caller (stream.py) converts these to SSE events and forwards them to
    the browser.  On any error this generator yields ("error", ...) and exits;
    the caller falls back to summarize() directly so the user always gets
    an answer.
    """
    tmp_path: Path | None = None
    t_start = time.monotonic()
    tool_call_count = 0

    try:
        tmp_path = _write_temp_file(initial_results)
        prompt   = _build_prompt(query, tmp_path, sys.executable, log_context)

        env = os.environ.copy()
        # Unset CLAUDECODE so the nested claude -p subprocess starts cleanly
        # without triggering re-entrancy detection.
        env.pop("CLAUDECODE", None)

        log.info("agent_stream LAUNCH claude -p stream-json (timeout=%ds)", AGENT_TIMEOUT)
        proc = await asyncio.create_subprocess_exec(
            "claude",
            "--disable-slash-commands",
            "--dangerously-skip-permissions",
            "--allowedTools", "Read,Bash",
            STREAM_JSON_FLAG, STREAM_JSON_VAL,
            "-p", prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        log.info("agent_stream WAITING pid=%s", proc.pid)

        final_result: str | None = None
        deadline = time.monotonic() + AGENT_TIMEOUT

        # Read stdout line-by-line.  Each line is a JSONL event from claude.
        # We give each readline() its own wait_for so a stalled subprocess
        # cannot block the FastAPI event loop beyond the deadline.
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                proc.kill()
                await proc.communicate()
                log.warning("agent_stream TIMEOUT after %ds", AGENT_TIMEOUT)
                yield ("error", {"message": "Agent timed out"})
                return

            try:
                raw = await asyncio.wait_for(
                    proc.stdout.readline(),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                log.warning("agent_stream TIMEOUT (readline) after %ds", AGENT_TIMEOUT)
                yield ("error", {"message": "Agent timed out"})
                return

            if not raw:
                # EOF — subprocess exited
                break

            line = raw.decode(errors="replace").strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue   # non-JSON line (e.g. a progress bar from claude); skip

            obj_type = obj.get("type")

            # ── Tool invocations live inside assistant content blocks ──────
            if obj_type == "assistant":
                content = obj.get("message", {}).get("content", [])
                for block in content:
                    if block.get("type") == "tool_use":
                        tool_name  = block.get("name", "")
                        tool_input = block.get("input", {})
                        act_type, message, detail = _classify_tool(tool_name, tool_input)
                        tool_call_count += 1
                        log.info(
                            "agent_stream TOOL #%d %s: %s",
                            tool_call_count, tool_name, message,
                        )
                        yield ("agent_activity", {
                            "type":    act_type,
                            "message": message,
                            "detail":  detail,
                        })

            # ── Final result ───────────────────────────────────────────────
            elif obj_type == "result":
                if obj.get("subtype") == "success":
                    final_result = obj.get("result", "").strip()
                else:
                    err_msg = obj.get("error", "Unknown agent error")
                    log.warning("agent_stream FAIL subtype=%s err=%r", obj.get("subtype"), err_msg)
                    proc.kill()
                    await proc.communicate()
                    yield ("error", {"message": err_msg})
                    return

        # Drain stderr and reap the process.  Using communicate() (not wait())
        # avoids a deadlock if the subprocess filled the stderr pipe buffer.
        await proc.communicate()

        elapsed = int((time.monotonic() - t_start) * 1000)

        if not final_result:
            log.warning("agent_stream FAIL no result in stream elapsed=%dms", elapsed)
            yield ("error", {"message": "Agent produced no output"})
            return

        log.info(
            "agent_stream OK chars=%d tools=%d elapsed=%dms",
            len(final_result), tool_call_count, elapsed,
        )
        yield ("summary", {"text": final_result})
        yield ("done_agent", {
            # tool_call_count is the real number of tool invocations observed.
            # Fall back to 1 if the agent answered without calling any tools
            # (it still "ran" — it just didn't need extra searches).
            "agent_iterations": tool_call_count if tool_call_count > 0 else 1,
            "llm_ms": elapsed,
        })

    except FileNotFoundError:
        log.warning("agent_stream SKIP claude CLI not found — is 'claude' on PATH?")
        yield ("error", {"message": "claude CLI not found"})
    except Exception as e:
        log.warning("agent_stream UNEXPECTED %s", e)
        yield ("error", {"message": str(e)})
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
