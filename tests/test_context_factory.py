"""Tests for the pluggable context factory (muflow.context.factory)."""

import tempfile

import pytest

from muflow import create_local_context, set_context_factory
from muflow.context.factory import build_typed_context, get_context_factory


@pytest.fixture(autouse=True)
def _reset_factory():
    """Ensure the global factory never leaks between tests."""
    set_context_factory(None)
    yield
    set_context_factory(None)


def _base_context():
    tmp = tempfile.mkdtemp()
    return create_local_context(path=tmp, kwargs={"x": 1})


class TestSetGetFactory:
    def test_set_and_get_roundtrip(self):
        def factory(base, context_type, context_data):  # pragma: no cover
            return base

        set_context_factory(factory)
        assert get_context_factory() is factory

    def test_clear_factory(self):
        set_context_factory(lambda *a: None)
        set_context_factory(None)
        assert get_context_factory() is None


class TestBuildTypedContext:
    def test_default_task_type_bypasses_factory(self):
        called = []
        set_context_factory(lambda *a: called.append(a) or "WRONG")
        base = _base_context()
        # "task" is the default — factory must NOT be consulted.
        assert build_typed_context(base, "task", {}) is base
        assert called == []

    def test_empty_type_bypasses_factory(self):
        set_context_factory(lambda *a: "WRONG")
        base = _base_context()
        assert build_typed_context(base, "", {}) is base

    def test_no_factory_returns_base_unchanged(self):
        base = _base_context()
        assert build_typed_context(base, "topography", {"k": "v"}) is base

    def test_factory_invoked_for_custom_type(self):
        seen = {}

        def factory(base, context_type, context_data):
            seen["type"] = context_type
            seen["data"] = context_data
            return "DOMAIN_CONTEXT"

        set_context_factory(factory)
        base = _base_context()
        result = build_typed_context(base, "topography", {"subject": 7})

        assert result == "DOMAIN_CONTEXT"
        assert seen == {"type": "topography", "data": {"subject": 7}}

    def test_factory_receives_empty_dict_when_data_is_none(self):
        captured = {}
        set_context_factory(
            lambda base, t, data: captured.setdefault("data", data) or base
        )
        build_typed_context(_base_context(), "surface", None)
        assert captured["data"] == {}
