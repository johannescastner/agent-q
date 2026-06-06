"""agent-q Piece-4 (memory swap) — episodic-write RED tests.

The write path is isolated into ``agentq/core/memory/episode.py`` (pydantic +
langgraph BaseStore only, no instructor) so it is testable without the full agent
stack. ``persist_episode`` maps a completed-command ``Memory`` onto the typed
``AgentQEpisode`` (exactly the 4 episode-STRUCT str fields) and writes it under
namespace ``("agentq","episodic")`` keyed ``session_id:sha1(objective)`` with the
``{"content": ...}`` envelope the BigQueryMemoryStore requires.
"""
import json

import pytest
from langgraph.store.memory import InMemoryStore

from agentq.core.memory.episode import AgentQEpisode, persist_episode


class _Task:
    def __init__(self, d):
        self._d = d

    def model_dump(self, mode=None):
        return self._d


class _Mem:
    objective = "buy milk"
    thought = "make a plan"
    final_response = "purchased"
    completed_tasks = [_Task({"id": 1, "description": "find item"})]


def test_persist_episode_writes_content_enveloped_4field_struct():
    store = InMemoryStore()
    persist_episode(store, _Mem(), session_id="sess-1")
    items = store.search(("agentq", "episodic"))
    assert len(items) == 1
    value = items[0].value
    assert "content" in value          # envelope (memory.py:1397 drops the row without it)
    c = value["content"]
    assert set(c) == {"observation", "thoughts", "action", "result"}   # EXACTLY the STRUCT
    assert all(isinstance(c[k], str) for k in c)                       # all 4 are str
    assert c["observation"] == "buy milk"
    assert c["thoughts"] == "make a plan"
    assert c["result"] == "purchased"
    assert json.loads(c["action"]) == [{"id": 1, "description": "find item"}]


def test_persist_episode_key_is_session_and_objective_hash():
    store = InMemoryStore()
    persist_episode(store, _Mem(), session_id="sess-1")
    item = store.search(("agentq", "episodic"))[0]
    assert item.key.startswith("sess-1:")


def test_agentq_episode_rejects_non_str_field():
    from pydantic import ValidationError
    # the typed str-field guard must raise ValidationError specifically (the
    # invariant the spec claims), not just any Exception
    with pytest.raises(ValidationError):
        AgentQEpisode(observation=123, thoughts="", action="", result="")


def test_persist_episode_none_completed_tasks_is_empty_json_list():
    class _Empty:
        objective = "x"
        thought = ""
        final_response = None
        completed_tasks = None

    store = InMemoryStore()
    persist_episode(store, _Empty(), session_id="s")
    c = store.search(("agentq", "episodic"))[0].value["content"]
    assert c["action"] == "[]" and c["result"] == ""
