import os

from agentq.config.config import USER_PREFERENCES_PATH
from agentq.utils.logger import logger


def get_user_ltm(store=None):
    # agent-q Piece-4 (memory swap): when a langgraph ``BaseStore`` is injected,
    # read the user preference from the shared CW BigQueryMemoryStore by a
    # DETERMINISTIC exact key (``store.get(ns, "current")``) and extract the bare
    # string from the semantic ``Fact`` shape (content_field is ``"fact"``; a bare
    # string is stored as ``{"content": v}``). The ``str | None`` contract guards
    # the system-prompt sink (agentq_actor.py). When no store is given, keep the
    # original file fallback so existing callers are unaffected.
    if store is not None:
        item = store.get(("agentq", "semantic", "user_preferences"), "current")
        if item is None:
            return None
        raw = item.value.get("fact")
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump()
        if isinstance(raw, dict):
            raw = raw.get("content")
        return raw if isinstance(raw, str) else None

    user_preference_file_name = "user_preferences.txt"
    user_preference_file = os.path.join(
        USER_PREFERENCES_PATH, user_preference_file_name
    )
    try:
        with open(user_preference_file) as file:
            user_pref = file.read()
        return user_pref
    except FileNotFoundError:
        logger.warning(f"User preference file not found: {user_preference_file}")

    return None


def seed_user_preference(store, text: str, *, key: str = "current") -> bool:
    """agent-q Piece-4: WRITE the user preference into the shared CW semantic store
    — the seam that closes the read-without-writer gap (get_user_ltm reads what this
    writes). Uses the store's ``{"content": <text>}`` envelope; the BigQueryMemoryStore
    normalizes a bare string into the semantic ``Fact`` shape under the "fact"
    content-field, which get_user_ltm reads back. Fail-soft: returns False (no write)
    when no store is injected. Returns True on a successful write."""
    if store is None:
        return False
    store.put(("agentq", "semantic", "user_preferences"), key, {"content": text})
    return True
