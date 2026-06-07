"""Pluggable context typing.

By default :func:`muflow.executor.execute_task` runs every task with a plain
:class:`~muflow.context.TaskContext`. A domain package (e.g. sds-workflows)
can register a *context factory* that upgrades the base context to a
domain-specific subclass based on the task's declared context type
(``register_task(context="...")``).

This keeps muFlow domain-agnostic: the engine knows only the *name* of the
context a task wants; the domain package supplies the implementation (e.g. a
context that resolves a ``SurfaceTopography`` from a file/URL/ORM subject and
exposes it as ``context.topography``). Tasks then never load data themselves —
they read it off the context.

Usage
-----
>>> from muflow.context.factory import set_context_factory
>>> def my_factory(base, context_type, context_data):
...     if context_type == "topography":
...         return TopographyContext(
...             base.storage, base.kwargs,
...             base.dependency_storages, base.progress_reporter,
...         )
...     return base
>>> set_context_factory(my_factory)
"""

from __future__ import annotations

from typing import Any, Callable, Optional

# factory signature: (base_context, context_type: str, context_data: dict) -> context
_context_factory: Optional[Callable[[Any, str, dict], Any]] = None


def set_context_factory(factory: Optional[Callable[[Any, str, dict], Any]]) -> None:
    """Register the global context factory (or ``None`` to clear it).

    The factory receives the base :class:`~muflow.context.TaskContext`, the
    task's declared ``context_type``, and the node's ``context_data`` (the
    ``context.json`` payload), and returns the context to run the task with —
    typically a domain subclass sharing the base context's storage,
    dependencies and progress reporter.
    """
    global _context_factory
    _context_factory = factory


def get_context_factory() -> Optional[Callable[[Any, str, dict], Any]]:
    """Return the registered context factory, or ``None``."""
    return _context_factory


def build_typed_context(base_context, context_type: str, context_data: dict):
    """Return the context to run a task with.

    Returns *base_context* unchanged when the task uses the default ``"task"``
    context type or no factory is registered. Otherwise delegates to the
    registered factory.
    """
    if not context_type or context_type == "task" or _context_factory is None:
        return base_context
    return _context_factory(base_context, context_type, context_data or {})
