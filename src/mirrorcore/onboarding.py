"""
Guided onboarding and unified entry flow (Phase 30).

Deterministic inspection of stored progress and a simple CLI menu that routes
into existing command handlers without duplicating their logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

from .db.store import DatabaseStore

# One typical interview session stores ~4 scenarios; style calibration ~4 prompts.
_DECISION_USABLE = 4
_STYLE_USABLE = 4
_DECISION_STARTED_MAX = _DECISION_USABLE - 1
_STYLE_STARTED_MAX = _STYLE_USABLE - 1


@dataclass(frozen=True)
class OnboardingSnapshot:
    """Counts and flags derived from SQLite (no LLM)."""

    persona_exists: bool
    decision_count: int
    style_count: int
    terminal_incidents: int
    has_open_analysis_session: bool


def load_onboarding_snapshot(store: DatabaseStore) -> OnboardingSnapshot:
    stats = store.get_database_stats()
    open_sess = store.get_latest_analysis_session()
    return OnboardingSnapshot(
        persona_exists=store.persona_exists(),
        decision_count=store.count_decision_memory_rows(),
        style_count=store.count_style_memory_rows(),
        terminal_incidents=int(stats.get("terminal_incidents") or 0),
        has_open_analysis_session=open_sess is not None,
    )


def decision_model_label(count: int) -> str:
    if count <= 0:
        return "not started"
    if count < _DECISION_USABLE:
        return "started"
    return "usable"


def style_model_label(count: int) -> str:
    if count <= 0:
        return "not started"
    if count < _STYLE_USABLE:
        return "started"
    return "usable"


def personal_response_label(decision_count: int, style_count: int) -> str:
    """Conservative: user can always run the command; this is about usefulness."""
    if decision_count < 2 and style_count < 2:
        return "not ready"
    return "ready to try"


def troubleshooting_label(incident_count: int) -> str:
    return "available" if incident_count > 0 else "not yet used"


def format_progress_summary(snapshot: OnboardingSnapshot) -> List[str]:
    lines = [
        "Where things stand:",
        f"  • Decision habits: {decision_model_label(snapshot.decision_count)}",
        f"  • How you sound: {style_model_label(snapshot.style_count)}",
        f"  • Answer like you: {personal_response_label(snapshot.decision_count, snapshot.style_count)}",
        f"  • Debug memory: {troubleshooting_label(snapshot.terminal_incidents)}",
    ]
    if not snapshot.persona_exists:
        lines.append(
            "  • Baseline profile: not saved yet (run mirrorcore init when you want that)"
        )
    return lines


def continuation_hints(snapshot: OnboardingSnapshot) -> List[str]:
    """Short, honest nudges for “continue where I left off”."""
    hints: List[str] = []
    if snapshot.has_open_analysis_session:
        hints.append(
            "You have an open log investigation — follow-up can pick up where that left off."
        )
    if 0 < snapshot.decision_count < _DECISION_USABLE:
        hints.append(
            "You’ve answered some decision questions — another short session can add more."
        )
    if snapshot.decision_count >= _DECISION_USABLE and snapshot.style_count < _STYLE_USABLE:
        hints.append(
            "Your decision side is in decent shape — a little style calibration goes a long way."
        )
    if snapshot.style_count >= _STYLE_USABLE and snapshot.decision_count >= _DECISION_USABLE:
        hints.append(
            "You’ve already built solid style memory — you can refine it or move on anytime."
        )
    elif snapshot.style_count >= _STYLE_USABLE and snapshot.decision_count >= 2:
        hints.append(
            "You’ve got enough saved style and decisions to try an “answer like me” pass."
        )
    return hints


def primary_next_suggestion(snapshot: OnboardingSnapshot) -> Optional[str]:
    """Single deterministic hint for the welcome line (optional)."""
    if snapshot.has_open_analysis_session:
        return "Continue your log investigation when you’re ready."
    if snapshot.decision_count <= 0:
        return "If you’re new here, “Learn how I decide” is a friendly first step."
    if snapshot.style_count < _STYLE_USABLE:
        return "Next natural step: learn how you talk, then try answering like you."
    if personal_response_label(snapshot.decision_count, snapshot.style_count) == "ready to try":
        return "Worth trying: answer like you, or help with a log when something breaks."
    return None


def run_start_menu(
    store: DatabaseStore,
    *,
    handle_interview: Callable[[Any], None],
    handle_calibrate_style: Callable[[Any], None],
    handle_respond_like_me: Callable[[Any], None],
    handle_analyze_log: Callable[[Any], None],
    handle_analyze_followup: Callable[[Any], None],
    args_namespace: Any,
    read_choice: Callable[[str], str],
) -> None:
    """Interactive menu; routes to existing handlers."""
    from types import SimpleNamespace

    empty_args = SimpleNamespace()

    while True:
        snap = load_onboarding_snapshot(store)
        print()
        print("MirrorCore")
        print("-" * 40)
        for line in format_progress_summary(snap):
            print(line)
        sug = primary_next_suggestion(snap)
        if sug:
            print()
            print(sug)
        chints = continuation_hints(snap)
        if chints:
            print()
            for h in chints:
                print(f"  → {h}")

        print()
        print("What would you like to do?")
        print("  1) Learn how I decide")
        print("  2) Learn how I talk")
        print("  3) Answer like me")
        print("  4) Help me debug something")
        print("  5) Continue where I left off")
        print("  6) Exit")

        choice = (read_choice("Enter a number (1–6): ").strip() or "").lower()
        if choice in ("6", "q", "quit", "exit"):
            print("Okay — see you next time.")
            return

        if choice == "1":
            handle_interview(args_namespace)
            continue
        if choice == "2":
            handle_calibrate_style(args_namespace)
            continue
        if choice == "3":
            handle_respond_like_me(empty_args)
            continue
        if choice == "4":
            handle_analyze_log(args_namespace)
            continue

        if choice == "5":
            _run_continue_submenu(
                snap,
                handle_interview=handle_interview,
                handle_calibrate_style=handle_calibrate_style,
                handle_analyze_followup=handle_analyze_followup,
                args_namespace=args_namespace,
                read_choice=read_choice,
            )
            continue

        print("Please pick 1–6.")


def _run_continue_submenu(
    snap: OnboardingSnapshot,
    *,
    handle_interview: Callable[[Any], None],
    handle_calibrate_style: Callable[[Any], None],
    handle_analyze_followup: Callable[[Any], None],
    args_namespace: Any,
    read_choice: Callable[[str], str],
) -> None:
    from types import SimpleNamespace

    options: List[Tuple[str, str, Callable[[], None]]] = []

    if snap.has_open_analysis_session:
        options.append(
            (
                "1",
                "Continue log investigation (follow-up)",
                lambda: handle_analyze_followup(args_namespace),
            )
        )
    if snap.decision_count > 0:
        options.append(
            (
                str(len(options) + 1),
                "Add more decision questions",
                lambda: handle_interview(args_namespace),
            )
        )
    if snap.style_count > 0:
        options.append(
            (
                str(len(options) + 1),
                "Refine how I talk",
                lambda: handle_calibrate_style(args_namespace),
            )
        )

    if not options:
        print()
        print("Nothing obvious to resume yet — try option 1 or 2 from the main menu.")
        read_choice("Press Enter to go back… ")
        return

    print()
    print("Pick up from:")
    for key, label, _ in options:
        print(f"  {key}) {label}")
    print(f"  0) Back")

    sub = read_choice("Enter a number: ").strip()
    if sub == "0":
        return
    for key, _, fn in options:
        if sub == key:
            fn()
            return
    print("That number wasn’t on the list.")


def default_read_choice(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return "6"
