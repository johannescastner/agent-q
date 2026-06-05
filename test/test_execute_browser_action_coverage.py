"""Piece-2 (DeepSeek): execute_browser_action must dispatch EVERY ActionType — no
silent no-op. Type-space invariant test, hardened per the committee round-1
Brittleness/AIMA finding: a coverage check that only asserts `ActionType.X`
*appears* is too weak — a branch body of `pass` would GREEN it, and it can't see
the else-raise. So this asserts three things via AST (no browser deps):
  (1) COVERAGE  — every ActionType enum member has a dispatch branch;
  (2) DISPATCH  — each branch body actually CALLS something (not a `pass`/no-op);
  (3) TOTALITY  — the if/elif chain ends in an `else` that RAISES (no silent
                  fall-through for an unknown/future ActionType).

RED today (token 1a5db28e): SOLVE_CAPTCHA has no branch and there is no else-raise.
"""
import ast
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.path.join(REPO, "agentq/core/models/models.py")
_DISPATCHER = os.path.join(REPO, "agentq/core/mcts/browser_mcts.py")


def _action_type_members() -> set[str]:
    src = open(_MODELS).read()
    members: set[str] = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.ClassDef) and n.name == "ActionType":
            for b in n.body:
                if isinstance(b, ast.Assign) and isinstance(b.targets[0], ast.Name):
                    members.add(b.targets[0].id)
    return members


def _execute_browser_action_func() -> ast.AsyncFunctionDef:
    for n in ast.walk(ast.parse(open(_DISPATCHER).read())):
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "execute_browser_action":
            return n
    raise AssertionError("execute_browser_action not found in browser_mcts.py")


def _dispatch_if(func: ast.AsyncFunctionDef) -> ast.If:
    """The top-level `if action.type == ActionType...:` chain."""
    for stmt in func.body:
        if isinstance(stmt, ast.If) and any(
            isinstance(a, ast.Attribute)
            and isinstance(a.value, ast.Name)
            and a.value.id == "ActionType"
            for a in ast.walk(stmt.test)
        ):
            return stmt
    raise AssertionError("no `if action.type == ActionType...` dispatch chain found")


def _walk_chain(if_node: ast.If):
    """Yield (member|None, body, is_else) for each branch of the if/elif/else chain."""
    node = if_node
    while True:
        member = next(
            (a.attr for a in ast.walk(node.test)
             if isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name)
             and a.value.id == "ActionType"),
            None,
        )
        yield (member, node.body, False)
        orelse = node.orelse
        if len(orelse) == 1 and isinstance(orelse[0], ast.If):
            node = orelse[0]
        else:
            yield (None, orelse, True)  # the terminal else body (may be empty)
            return


def _branches():
    return list(_walk_chain(_dispatch_if(_execute_browser_action_func())))


def test_coverage_every_action_type_has_a_branch():
    members = _action_type_members()
    handled = {m for m, _body, is_else in _branches() if m and not is_else}
    assert members, "no ActionType members parsed — test wiring broken"
    missing = members - handled
    assert not missing, (
        f"execute_browser_action has no dispatch branch for ActionType(s): {sorted(missing)}"
    )


def test_each_branch_actually_dispatches_not_a_noop():
    # Require an `await <call>` (the codebase's skill-dispatch pattern), NOT just any
    # call — else a `print("TODO")`-only branch would GREEN (print is a Call but is
    # never awaited). All real skill dispatches are `await openurl(...)` etc.
    for member, body, is_else in _branches():
        if is_else or member is None:
            continue
        has_real_dispatch = any(
            isinstance(n, ast.Await) and isinstance(n.value, ast.Call)
            for stmt in body
            for n in ast.walk(stmt)
        )
        assert has_real_dispatch, (
            f"ActionType.{member} branch has no `await <skill>(...)` dispatch — a "
            f"`pass`/`print`-only body silently drops the action"
        )


def test_chain_ends_in_else_that_raises():
    branches = _branches()
    member, else_body, is_else = branches[-1]
    assert is_else, "if/elif chain has no terminal `else` — an unknown ActionType silently no-ops"
    has_raise = any(isinstance(n, ast.Raise) for stmt in else_body for n in ast.walk(stmt))
    assert has_raise, "terminal `else` does not raise — unknown/future ActionType silently no-ops"
