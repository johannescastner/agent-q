"""agent-q Piece-4 (memory swap) — episodic write contract + helper.

Isolated from the agent stack (pydantic + a langgraph ``BaseStore`` only, no
``instructor``) so the typed write contract is unit-testable without the full
orchestrator. ``persist_episode`` maps a completed-command ``Memory`` onto the
typed ``AgentQEpisode`` — EXACTLY the 4 episode-STRUCT ``str`` fields
(``observation, thoughts, action, result``) — and writes it under namespace
``("agentq", "episodic")`` keyed ``session_id:sha1(objective)`` with the
``{"content": ...}`` envelope the shared CW ``BigQueryMemoryStore`` requires
(``memory.py:1397`` drops a row whose value lacks ``"content"``).

The kind-discriminator is the NAMESPACE (NS_ROOT ``"agentq"``), not an in-row
field — a reader separates agent-q task-execution episodes from baby-NICER
conversational episodes by the namespace prefix.
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional


from pydantic import BaseModel


class AgentQEpisode(BaseModel):
    """The episodic write contract — non-Optional ``str`` fields, so a non-string
    value raises ``ValidationError`` at construction (before the BQ write), and the
    dict matches the 4-field BQ ``episode`` STRUCT exactly."""

    observation: str
    thoughts: str
    action: str
    result: str


def persist_episode(store, memory, *, session_id: str, key: Optional[str] = None) -> None:
    """Write ONE completed-command episode to the shared CW store.

    Callers (``Orchestrator.execute_command``) wrap this in a NARROW
    ``try/except`` so a memory-write failure never suppresses the command's own
    eval-mode result. ``memory`` is duck-typed: ``objective`` (str), ``thought``
    (str), ``completed_tasks`` (optional list of ``.model_dump()``-ables),
    ``final_response`` (optional str)."""
    ep = AgentQEpisode(
        observation=memory.objective,
        thoughts=memory.thought,
        action=json.dumps(
            [t.model_dump(mode="json") for t in (memory.completed_tasks or [])]
        ),
        result=memory.final_response or "",
    )
    if key is None:
        digest = hashlib.sha1(memory.objective.encode("utf-8")).hexdigest()
        key = f"{session_id}:{digest}"
    store.put(("agentq", "episodic"), key, {"content": ep.model_dump()})
