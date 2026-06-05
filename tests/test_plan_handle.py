"""Tests for PlanHandle state-query and cancellation (``backends/handle.py``).

``get_progress`` (local + S3) is covered in ``test_plan_handle_progress.py``;
this module covers the backend-specific ``get_state()`` / ``cancel()`` paths for
the async (Celery, Step Functions) backends, plus the local and
misconfiguration branches. Celery's ``AsyncResult`` and boto3's Step Functions
client are stubbed so no broker or AWS account is needed.
"""

import celery.result
import pytest

from muflow.backends.handle import PlanHandle


def _handle(backend, plan_id="pid", **kw):
    return PlanHandle(
        backend=backend,
        plan_id=plan_id,
        node_prefixes=kw.get("node_prefixes", {"root": "/tmp/root"}),
        storage_type=kw.get("storage_type", "local"),
        storage_config=kw.get("storage_config", {}),
    )


@pytest.fixture(autouse=True)
def _reset_celery_app():
    """Each test starts with no class-level Celery app, restored afterwards."""
    saved = PlanHandle._celery_app
    PlanHandle._celery_app = None
    yield
    PlanHandle._celery_app = saved


# ── local backend ─────────────────────────────────────────────────────────────


def test_local_state_is_success():
    assert _handle("local").get_state() == "success"


def test_local_cancel_not_supported():
    with pytest.raises(NotImplementedError):
        _handle("local").cancel()


# ── configure_celery + unknown backend guard ──────────────────────────────────


def test_configure_celery_sets_class_app():
    sentinel = object()
    PlanHandle.configure_celery(sentinel)
    assert PlanHandle._celery_app is sentinel


def test_unknown_backend_raises():
    h = _handle("local")
    # Bypass pydantic validation to reach the defensive branch.
    object.__setattr__(h, "backend", "bogus")
    with pytest.raises(ValueError, match="Unknown backend"):
        h.get_state()


# ── Celery state mapping (AsyncResult stubbed) ────────────────────────────────


class _FakeAsyncResult:
    state_to_return = "PENDING"

    def __init__(self, task_id, app=None):
        self.id = task_id
        self.app = app
        self.state = type(self).state_to_return


@pytest.mark.parametrize(
    "celery_state,expected",
    [
        ("PENDING", "pending"),
        ("STARTED", "running"),
        ("SUCCESS", "success"),
        ("FAILURE", "failure"),
        ("REVOKED", "failure"),
        ("WEIRD_UNKNOWN", "pending"),  # falls back to "pending"
    ],
)
def test_celery_state_mapping(monkeypatch, celery_state, expected):
    _FakeAsyncResult.state_to_return = celery_state
    monkeypatch.setattr(celery.result, "AsyncResult", _FakeAsyncResult)
    PlanHandle.configure_celery(object())  # any non-None app
    assert _handle("celery").get_state() == expected


def test_celery_state_groupresult_branch(monkeypatch):
    """A result object lacking a ``state`` attribute (GroupResult-like) is
    classified via ready()/failed()."""

    class _FakeGroupResult:
        def __init__(self, task_id, app=None):
            pass

        def ready(self):
            return True

        def failed(self):
            return True

    monkeypatch.setattr(celery.result, "AsyncResult", _FakeGroupResult)
    PlanHandle.configure_celery(object())
    assert _handle("celery").get_state() == "failure"


def test_celery_state_requires_configured_app():
    # _celery_app is None (autouse fixture) -> RuntimeError.
    with pytest.raises(RuntimeError, match="No Celery app configured"):
        _handle("celery").get_state()


# ── Celery cancel ─────────────────────────────────────────────────────────────


def test_celery_cancel_revokes_task():
    class _Control:
        def __init__(self):
            self.revoked = []

        def revoke(self, task_id, terminate=False):
            self.revoked.append((task_id, terminate))

    class _App:
        def __init__(self):
            self.control = _Control()

    app = _App()
    PlanHandle.configure_celery(app)
    _handle("celery", plan_id="task-7").cancel()
    assert app.control.revoked == [("task-7", True)]


def test_celery_cancel_requires_configured_app():
    with pytest.raises(RuntimeError, match="No Celery app configured"):
        _handle("celery").cancel()


# ── Step Functions state + cancel (boto3 stubbed) ─────────────────────────────


class _FakeSfnClient:
    def __init__(self):
        self.stopped = []
        self.status = "SUCCEEDED"

    def describe_execution(self, executionArn):
        return {"status": self.status}

    def stop_execution(self, executionArn, cause=None):
        self.stopped.append((executionArn, cause))


@pytest.mark.parametrize(
    "sfn_status,expected",
    [
        ("RUNNING", "running"),
        ("SUCCEEDED", "success"),
        ("FAILED", "failure"),
        ("TIMED_OUT", "failure"),
        ("ABORTED", "failure"),
        ("SOMETHING_ELSE", "pending"),
    ],
)
def test_sfn_state_mapping(monkeypatch, sfn_status, expected):
    fake = _FakeSfnClient()
    fake.status = sfn_status
    monkeypatch.setattr("boto3.client", lambda service: fake)
    assert _handle("step_functions", plan_id="arn:exec").get_state() == expected


def test_sfn_cancel_stops_execution(monkeypatch):
    fake = _FakeSfnClient()
    monkeypatch.setattr("boto3.client", lambda service: fake)
    _handle("step_functions", plan_id="arn:exec:9").cancel()
    assert fake.stopped and fake.stopped[0][0] == "arn:exec:9"
