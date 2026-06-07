"""Task context.

This package provides the ``TaskContext`` class that wraps a
``StorageBackend`` (from ``muflow.storage``) and adds task-level
concerns: dependency access, progress reporting, and parameters.

Modules
-------
task
    ``TaskContext`` — unified context class for all backends.
    ``create_local_context`` — convenience function for local testing.
"""

from muflow.context.factory import (
    build_typed_context,
    get_context_factory,
    set_context_factory,
)
from muflow.context.task import TaskContext, create_local_context

__all__ = [
    "TaskContext",
    "create_local_context",
    "set_context_factory",
    "get_context_factory",
    "build_typed_context",
]
