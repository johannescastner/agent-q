"""agent-q Piece-4 — shared CW memory store construction (fail-soft).

agent-q reads the user preference from the SEMANTIC table (content_field "fact")
and writes episodes to the EPISODIC table (content_field "episode") of the shared
CW ``agent_system_memory`` dataset, via the PUBLIC NICER-core ``BigQueryMemoryStore``
(``src/graphs/memory.py``, mirrored to nicer-core). Because that store is keyed per
content-field, the two surfaces need TWO store instances.

FAIL-SOFT BY DESIGN: ``build_stores()`` returns ``(semantic_store, episodic_store)``
where either may be ``None``. When ``None`` the agent degrades gracefully — the
file-based LTM is used for the read and episodic persistence is skipped — so agent-q
runs unchanged without CW credentials. The real BigQueryMemoryStore construction is
ENABLED only when ``AGENTQ_MEMORY_BACKEND=bigquery`` AND NICER-core + CW creds are
importable; it also depends on the ``from_client(credentials=...)`` seam (named fix
b65ed6bf) so a non-tenant SA can be injected — until that lands, construction here
returns ``None`` rather than guessing an unverifiable wiring.
"""
import os
from typing import Optional, Tuple

from agentq.utils.logger import logger


def _enabled() -> bool:
    return os.environ.get("AGENTQ_MEMORY_BACKEND", "").strip().lower() == "bigquery"


def build_stores() -> Tuple[Optional[object], Optional[object]]:
    """Return (semantic_store, episodic_store); either may be None (fail-soft)."""
    if not _enabled():
        return None, None
    try:
        # The per-table builders + the dim-guard error live in the PUBLIC
        # NICER-core package; fall back to the in-repo path when running inside
        # baby-NICER (the mirror source). Import only when the backend is enabled.
        try:
            from nicer_core.graphs.memory import (  # type: ignore  # noqa: F401
                build_agentq_semantic_store,
                build_agentq_episodic_store,
                EmbeddingDimMismatchError,
            )
        except ImportError:
            # ImportError (not just ModuleNotFoundError) so a PRESENT-but-symbol-less
            # nicer_core (the lagging-mirror case: seam built in baby-NICER src/ but
            # not yet in the public package) still falls back to the in-repo path.
            from src.graphs.memory import (  # type: ignore  # noqa: F401
                build_agentq_semantic_store,
                build_agentq_episodic_store,
                EmbeddingDimMismatchError,
            )
    except ImportError as e:  # NICER-core not installed / seam not published
        logger.warning(
            "agent-q: NICER-core memory store seam unavailable (%s); "
            "using file LTM and skipping episodic persistence", e
        )
        return None, None
    try:
        return build_agentq_semantic_store(), build_agentq_episodic_store()
    except EmbeddingDimMismatchError:
        # An embedding-dim mismatch is an INVARIANT violation (would silently
        # pollute the shared vector space). Surface it loudly — never degrade.
        raise
    except Exception as e:  # creds / BQ connectivity — degrade to file LTM
        logger.warning(
            "agent-q: BigQueryMemoryStore construction failed (%s); "
            "using file LTM and skipping episodic persistence", e
        )
        return None, None
