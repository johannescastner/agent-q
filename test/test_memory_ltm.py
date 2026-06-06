"""agent-q Piece-4 (memory swap) — get_user_ltm store-read RED tests.

get_user_ltm gains an optional `store: BaseStore` parameter: when given, it reads
the user preference from the shared CW BigQueryMemoryStore by DETERMINISTIC exact
key (`store.get(ns, "current")`), extracting the bare string from the semantic
`Fact` shape; when absent it keeps the file fallback. The contract is `str | None`
(a dict must never leak to the system-prompt sink at agentq_actor.py:31-36).

Seeding note (Brittleness fresh-v14 finding): an InMemoryStore returns `item.value`
VERBATIM, so we seed the BQ-READ shape `{"fact": {"content": ...}}` (what the real
store returns after `content_field="fact"` rewriting), NOT the write-envelope
`{"content": ...}` — otherwise the test would pass for the wrong reason.
"""
from langgraph.store.memory import InMemoryStore

from agentq.core.memory.ltm import get_user_ltm

NS = ("agentq", "semantic", "user_preferences")


def test_get_user_ltm_reads_preference_from_store_as_str():
    store = InMemoryStore()
    store.put(NS, "current", {"fact": {"content": "prefers dark mode"}})
    result = get_user_ltm(store)
    assert isinstance(result, str)          # TYPE — no dict leaks to the prompt sink
    assert result == "prefers dark mode"    # content adequacy


def test_get_user_ltm_none_when_no_preference_row():
    store = InMemoryStore()
    assert get_user_ltm(store) is None


def test_get_user_ltm_str_or_none_when_no_store():
    # store=None -> file fallback. agent-q ships a default user_preferences.txt,
    # so the fallback returns its contents as a str (the contract is str|None).
    result = get_user_ltm()
    assert result is None or isinstance(result, str)


class _NormalizingSemanticStore:
    """Models BigQueryMemoryStore's semantic write→read: a put with the
    ``{"content": <str>}`` envelope is normalized to the Fact shape and read back
    under the per-table content-field key ``"fact"`` (so seed↔get round-trips)."""

    def __init__(self):
        self._d = {}

    def put(self, ns, key, value):
        raw = value["content"]
        fact = raw if isinstance(raw, dict) else {"content": raw}
        item = type("Item", (), {})()
        item.value = {"fact": fact}
        item.key = key
        self._d[(tuple(ns), key)] = item

    def get(self, ns, key):
        return self._d.get((tuple(ns), key))


def test_seed_then_get_user_ltm_round_trip():
    # closes the read-without-writer gap: seed_user_preference WRITES, get_user_ltm READS
    from agentq.core.memory.ltm import seed_user_preference
    store = _NormalizingSemanticStore()
    assert seed_user_preference(store, "prefers dark mode") is True
    assert get_user_ltm(store) == "prefers dark mode"


def test_seed_user_preference_no_store_returns_false():
    from agentq.core.memory.ltm import seed_user_preference
    assert seed_user_preference(None, "x") is False
