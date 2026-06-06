"""agent-q Piece-4 — integration test for the episodic-write WIRING.

Verifies the live completion path: Orchestrator(store=...).execute_command(...)
persists exactly one episode to the injected store after the state loop completes,
and that the absence of a store is a no-op (fail-soft, no crash). _handle_state is
mocked to complete in one step (no browser/LLM), and PlaywrightManager is patched
so construction needs no browser.
"""
import asyncio
from unittest.mock import patch

from langgraph.store.memory import InMemoryStore

from agentq.core.models.models import State


def _make_orchestrator(store):
    # patch PlaywrightManager so Orchestrator() constructs without a browser
    with patch("agentq.core.orchestrator.orchestrator.PlaywrightManager"):
        from agentq.core.orchestrator.orchestrator import Orchestrator
        return Orchestrator(state_to_agent_map={}, eval_mode=True, store=store)


def _drive(orch, final_response):
    async def fake_handle_state():
        orch.memory.current_state = State.COMPLETED
        orch.memory.final_response = final_response
    orch._handle_state = fake_handle_state
    orch._print_final_response = lambda: None
    return asyncio.run(orch.execute_command("buy milk"))


def test_execute_command_persists_one_episode():
    store = InMemoryStore()
    orch = _make_orchestrator(store)
    result = _drive(orch, "purchased the milk")
    assert result == "purchased the milk"            # eval-mode return preserved
    items = store.search(("agentq", "episodic"))
    assert len(items) == 1                            # exactly one episode
    c = items[0].value["content"]
    assert c["observation"] == "buy milk"
    assert c["result"] == "purchased the milk"
    assert items[0].key.startswith(orch.session_id + ":")


def test_execute_command_without_store_is_noop_no_crash():
    orch = _make_orchestrator(None)                  # store=None
    assert _drive(orch, "ok") == "ok"                # returns normally, no persist


def test_persist_failure_does_not_suppress_result():
    class _BoomStore(InMemoryStore):
        def put(self, *a, **k):
            raise RuntimeError("store down")
    orch = _make_orchestrator(_BoomStore())
    # fail-soft: the write blows up but the command's result is still returned
    assert _drive(orch, "still returned") == "still returned"
