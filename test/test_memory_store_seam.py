"""agent-q Piece-4 — build_stores() activation-dispatch tests (no live BQ).

Covers the SINK the fidelity committee flagged as untested: build_stores()'s
dispatch logic — disabled→(None,None); EmbeddingDimMismatchError PROPAGATES (never
swallowed); a creds/connectivity error DEGRADES to (None,None); both stores
returned when construction succeeds. A fake `nicer_core.graphs.memory` module is
injected via sys.modules so the seam import resolves without the real package or
any BigQuery access.
"""
import sys
import types

import pytest


class _DimErr(RuntimeError):
    pass


def _raise_dim():
    raise _DimErr("embedding dim mismatch")


def _raise_creds():
    raise RuntimeError("creds/connectivity down")


def _install_fake_seam(monkeypatch, sem_fn, epi_fn):
    nc = types.ModuleType("nicer_core")
    ncg = types.ModuleType("nicer_core.graphs")
    m = types.ModuleType("nicer_core.graphs.memory")
    m.build_agentq_semantic_store = sem_fn
    m.build_agentq_episodic_store = epi_fn
    m.EmbeddingDimMismatchError = _DimErr
    monkeypatch.setitem(sys.modules, "nicer_core", nc)
    monkeypatch.setitem(sys.modules, "nicer_core.graphs", ncg)
    monkeypatch.setitem(sys.modules, "nicer_core.graphs.memory", m)


def _store():
    import agentq.core.memory.store as s
    return s


def test_build_stores_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AGENTQ_MEMORY_BACKEND", raising=False)
    assert _store().build_stores() == (None, None)


def test_build_stores_constructs_when_enabled_and_available(monkeypatch):
    monkeypatch.setenv("AGENTQ_MEMORY_BACKEND", "bigquery")
    sem, epi = object(), object()
    _install_fake_seam(monkeypatch, lambda: sem, lambda: epi)
    assert _store().build_stores() == (sem, epi)


def test_build_stores_propagates_dim_mismatch_not_swallowed(monkeypatch):
    monkeypatch.setenv("AGENTQ_MEMORY_BACKEND", "bigquery")
    _install_fake_seam(monkeypatch, _raise_dim, lambda: object())
    with pytest.raises(_DimErr):
        _store().build_stores()


def test_build_stores_degrades_to_none_on_creds_error(monkeypatch):
    monkeypatch.setenv("AGENTQ_MEMORY_BACKEND", "bigquery")
    _install_fake_seam(monkeypatch, _raise_creds, lambda: object())
    assert _store().build_stores() == (None, None)


def test_build_stores_falls_back_to_src_when_nicer_core_lacks_symbols(monkeypatch):
    # lagging-mirror case: nicer_core is PRESENT but lacks the seam symbols, so the
    # import raises a bare ImportError (not ModuleNotFoundError) — the src fallback
    # must still fire (regression for the except ModuleNotFoundError → ImportError fix).
    monkeypatch.setenv("AGENTQ_MEMORY_BACKEND", "bigquery")
    nc = types.ModuleType("nicer_core")
    ncg = types.ModuleType("nicer_core.graphs")
    nm = types.ModuleType("nicer_core.graphs.memory")  # NO build_agentq_* attrs
    monkeypatch.setitem(sys.modules, "nicer_core", nc)
    monkeypatch.setitem(sys.modules, "nicer_core.graphs", ncg)
    monkeypatch.setitem(sys.modules, "nicer_core.graphs.memory", nm)
    sem, epi = object(), object()
    sg = types.ModuleType("src.graphs.memory")
    sg.build_agentq_semantic_store = lambda: sem
    sg.build_agentq_episodic_store = lambda: epi
    sg.EmbeddingDimMismatchError = _DimErr
    monkeypatch.setitem(sys.modules, "src", types.ModuleType("src"))
    monkeypatch.setitem(sys.modules, "src.graphs", types.ModuleType("src.graphs"))
    monkeypatch.setitem(sys.modules, "src.graphs.memory", sg)
    assert _store().build_stores() == (sem, epi)


def test_build_stores_degrades_to_none_when_seam_absent(monkeypatch):
    monkeypatch.setenv("AGENTQ_MEMORY_BACKEND", "bigquery")
    # ensure neither nicer_core nor src.graphs.memory is importable
    monkeypatch.setitem(sys.modules, "nicer_core", None)
    monkeypatch.setitem(sys.modules, "nicer_core.graphs.memory", None)
    monkeypatch.setitem(sys.modules, "src.graphs.memory", None)
    assert _store().build_stores() == (None, None)
