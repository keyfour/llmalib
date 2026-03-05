"""
debug/inspector.py

Human-readable console output for pipeline runs and individual results.
Uses rich for colour output when available; falls back to plain text silently.
All output goes to stderr — never pollutes stdout.
"""

from __future__ import annotations

import sys
from llmalib.core.result import Attempt, Result
from llmalib.debug.tracer import Tracer

# Try to import rich; fall back to plain text
try:
    from rich.console import Console
    from rich.text import Text
    from rich.rule import Rule

    _console = Console(stderr=True, highlight=False)
    _RICH = True
except ImportError:
    _RICH = False
    _console = None


def print_result(result: Result) -> None:
    """Print a single task result immediately after completion (live debug mode)."""
    icon = "✓" if result.ok else "✗"
    attempts_str = f"[attempt {result.attempts}/{result.attempts}]"

    if _RICH:
        color = "green" if result.ok else "red"
        line = Text()
        line.append(f"{icon}  {result.task_name:<30}", style=f"bold {color}")
        line.append(f"  {attempts_str}  ", style="dim")
        if result.ok and result.value:
            preview = str(result.value.model_dump())[:80]
            line.append(preview, style="cyan")
        else:
            line.append(result.error or "unknown error", style="red")
        _console.print(line)
    else:
        status = "OK" if result.ok else "FAIL"
        detail = (
            str(result.value.model_dump())[:80]
            if result.ok and result.value
            else result.error
        )
        print(
            f"[{status}] {result.task_name} {attempts_str}  {detail}", file=sys.stderr
        )


def print_trace(tracer: Tracer) -> None:
    """Print a full run summary after all tasks complete."""
    total = len(tracer.task_traces)
    succeeded = sum(1 for t in tracer.task_traces if t.ok)
    elapsed = (
        f"{tracer.task_traces[-1].attempts[-1].duration_ms / 1000:.1f}s"
        if tracer.task_traces
        else "0s"
    )

    _rule(f"Pipeline Run: {tracer.run_id}")
    _print(f" Tasks: {total}  |  Success: {succeeded}  |  Failed: {total - succeeded}")
    _rule()

    for tt in tracer.task_traces:
        icon = "✓" if tt.ok else "✗"
        total_ms = sum(a.duration_ms for a in tt.attempts)
        _print(
            f" {icon}  {tt.task_name:<30}  {len(tt.attempts)} attempt(s)  {total_ms:.0f}ms"
        )
        if not tt.ok:
            for a in tt.attempts:
                if a.parse_error:
                    _print(
                        f"     └─ Attempt {a.attempt_number}: parse error — {a.parse_error[:80]}",
                        dim=True,
                    )
                for ge in a.guard_errors:
                    _print(
                        f"     └─ Attempt {a.attempt_number}: guard error — {ge[:80]}",
                        dim=True,
                    )

    _rule()


def print_attempt(attempt: Attempt) -> None:
    """Print a single attempt's full prompt and response (deep inspection)."""
    _rule(f"Attempt {attempt.attempt_number}")

    _print("PROMPT:")
    _print(attempt.rendered_prompt[:1000])

    _print("\nRESPONSE:")
    _print(attempt.raw_response[:500])

    parse_status = "✗  " + attempt.parse_error if attempt.parse_error else "✓"
    _print(f"\nPARSE: {parse_status}")

    if attempt.guard_errors:
        _print("GUARDS:")
        for ge in attempt.guard_errors:
            _print(f"  ✗ {ge}")
    else:
        _print("GUARDS: ✓")

    _rule()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _print(text: str, dim: bool = False) -> None:
    if _RICH:
        style = "dim" if dim else ""
        _console.print(text, style=style)
    else:
        print(text, file=sys.stderr)


def _rule(title: str = "") -> None:
    if _RICH:
        _console.print(Rule(title))
    else:
        line = f"─── {title} " if title else "─" * 60
        print(line + "─" * max(0, 60 - len(line)), file=sys.stderr)
