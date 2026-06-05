"""Tests for plan completion callbacks (``muflow/backends/callbacks.py``).

The callbacks notify the calling application when a plan finishes. They are
wired into the async backends (which are exercised elsewhere), so the callback
classes themselves were only partially covered. These unit tests drive each
implementation directly.
"""

import logging

from muflow.backends.callbacks import (
    CeleryCompletionCallback,
    LoggingCompletionCallback,
    NoOpCompletionCallback,
)


class _FakeCeleryApp:
    """Records send_task calls instead of dispatching to a broker."""

    def __init__(self):
        self.calls = []

    def send_task(self, name, args=None, queue=None):
        self.calls.append({"name": name, "args": args, "queue": queue})


def test_celery_callback_dispatches_task_with_args_and_queue():
    app = _FakeCeleryApp()
    cb = CeleryCompletionCallback(
        celery_app=app, task_name="myapp.on_complete", queue="callbacks"
    )
    cb.notify("plan-123", True, None)

    assert len(app.calls) == 1
    call = app.calls[0]
    assert call["name"] == "myapp.on_complete"
    assert call["args"] == ["plan-123", True, None]
    assert call["queue"] == "callbacks"


def test_celery_callback_forwards_failure_error_message():
    app = _FakeCeleryApp()
    cb = CeleryCompletionCallback(app, "t", queue="q2")
    cb.notify("p9", False, "boom")
    assert app.calls[0]["args"] == ["p9", False, "boom"]
    assert app.calls[0]["queue"] == "q2"


def test_noop_callback_does_nothing():
    # Simply must not raise.
    NoOpCompletionCallback().notify("p", True, None)
    NoOpCompletionCallback().notify("p", False, "err")


def test_logging_callback_logs_success_and_failure(caplog):
    cb = LoggingCompletionCallback()  # uses module logger by default
    with caplog.at_level(logging.DEBUG):
        cb.notify("plan-ok", True, None)
        cb.notify("plan-bad", False, "kaput")

    messages = [r.getMessage() for r in caplog.records]
    assert any("Plan completed" in m and "plan-ok" in m for m in messages)
    failure = [r for r in caplog.records if "Plan failed" in r.getMessage()]
    assert failure and failure[0].levelno == logging.ERROR
    assert "kaput" in failure[0].getMessage()


def test_logging_callback_uses_supplied_logger():
    custom = logging.getLogger("muflow.test.custom_callback_logger")
    cb = LoggingCompletionCallback(logger=custom)
    assert cb._log is custom
