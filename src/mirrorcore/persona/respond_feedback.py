"""
Phase 38: short interactive feedback after respond-like-me (deterministic, no LLM).
"""

from __future__ import annotations

import sys
from typing import Callable, Optional

from ..db.store import DatabaseStore
from .respond import PersonalResponse


def _read_line(read_fn: Callable[[str], str], prompt_text: str) -> str:
    try:
        return (read_fn(prompt_text) or "").strip()
    except (EOFError, KeyboardInterrupt):
        return "\x03"


def maybe_prompt_respond_feedback(
    *,
    scenario_text: str,
    pr: PersonalResponse,
    store: DatabaseStore,
    read_line: Callable[[str], str],
    is_interactive: bool,
) -> None:
    """Optional one-pass rating; skips when non-interactive or empty path."""
    if not is_interactive:
        return
    if not pr.evidence_path.route_keys and not pr.evidence_path.decision_ids:
        return

    print(
        "How close was that? 1 = right  2 = partly  3 = wrong  "
        "(enter = skip)"
    )
    choice = _read_line(read_line, "> ").lower()
    if choice in ("\x03",):
        print("Skipped.")
        return
    if not choice:
        return

    rating: Optional[str] = None
    if choice in ("1", "r", "right", "y", "yes"):
        rating = "right"
    elif choice in ("2", "p", "partly", "partial"):
        rating = "partly"
    elif choice in ("3", "w", "wrong", "n", "no"):
        rating = "wrong"
    else:
        return

    partial_aspect: Optional[str] = None
    if rating == "partly":
        print(
            "What was off? 1 = action felt right, wording off  "
            "2 = wording felt right, action off  "
            "3 = same direction, different phrasing  "
            "(enter = skip detail)"
        )
        p2 = _read_line(read_line, "> ").lower()
        if p2 in ("\x03",):
            print("Skipped detail.")
        elif p2 in ("1", "a", "action"):
            partial_aspect = "action_ok_word_bad"
        elif p2 in ("2", "w", "word", "wording"):
            partial_aspect = "word_ok_action_bad"
        elif p2 in ("3", "s", "same"):
            partial_aspect = "same_direction_phrase"

    feedback_target: Optional[str] = None
    if rating in ("wrong", "right") and not partial_aspect:
        print(
            "Was this mainly about your move, your wording, or both?  "
            "1 = move  2 = wording  3 = both  (enter = both)"
        )
        t3 = _read_line(read_line, "> ").lower()
        if t3 in ("\x03",):
            print("Using both for weighting.")
        elif t3 in ("1", "a", "action", "move"):
            feedback_target = "action"
        elif t3 in ("2", "w", "word", "wording"):
            feedback_target = "wording"
        elif t3 in ("3", "b", "both"):
            feedback_target = "both"

    replacement: Optional[str] = None
    if rating in ("wrong", "partly"):
        print("Optional: one line you'd say instead (enter to skip)")
        rep = _read_line(read_line, "> ")
        if rep and rep != "\x03" and len(rep) < 2000:
            replacement = rep.strip() or None

    ft_store = feedback_target
    if (
        ft_store is None
        and rating != "partly"
        and getattr(pr, "answer_focus", None)
    ):
        af = str(getattr(pr, "answer_focus", "") or "").strip().lower()
        if af in ("action", "wording", "both"):
            ft_store = af

    store.record_personal_response_feedback(
        scenario_snippet=scenario_text,
        prompt_norm_hash=pr.prompt_norm_hash,
        rating=rating,
        partial_aspect=partial_aspect,
        replacement_text=replacement,
        confidence_shown=float(pr.confidence),
        effective_family=pr.effective_family,
        evidence_path=pr.evidence_path.to_storage_dict(),
        likely_answer_snippet=pr.likely_answer,
        feedback_target=ft_store,
    )
    print("Noted — I'll weight that path accordingly next time.")


def stdin_is_interactive() -> bool:
    try:
        return sys.stdin.isatty()
    except Exception:
        return False
