"""Tests for the task registry lookup/unregister paths."""

import pytest

from muflow import registry
from muflow.registry import (
    NotRegisteredException,
    get_by_display_name,
    register_task,
    unregister,
)


@pytest.fixture(autouse=True)
def _clean():
    registry.clear()
    yield
    registry.clear()


class TestDisplayNameLookup:
    def test_get_by_display_name_returns_entry(self):
        @register_task(name="app.a", display_name="Alpha")
        def a(context):  # pragma: no cover - body never executed here
            pass

        entry = get_by_display_name("Alpha")
        assert entry is not None
        assert entry.name == "app.a"

    def test_get_by_display_name_missing_returns_none(self):
        assert get_by_display_name("Nope") is None

    def test_no_display_name_not_indexed(self):
        @register_task(name="app.b")
        def b(context):  # pragma: no cover
            pass

        # Empty display_name must not create a "" -> entry mapping.
        assert get_by_display_name("") is None


class TestUnregister:
    def test_unregister_removes_from_name_index(self):
        @register_task(name="app.c")
        def c(context):  # pragma: no cover
            pass

        assert registry.get("app.c") is not None
        unregister("app.c")
        assert registry.get("app.c") is None

    def test_unregister_removes_display_name_index(self):
        @register_task(name="app.d", display_name="Delta")
        def d(context):  # pragma: no cover
            pass

        unregister("app.d")
        assert get_by_display_name("Delta") is None

    def test_unregister_unknown_raises(self):
        with pytest.raises(NotRegisteredException) as exc:
            unregister("app.missing")
        assert exc.value.name == "app.missing"
        assert "app.missing" in str(exc.value)

    def test_unregister_then_reregister_allowed(self):
        @register_task(name="app.e")
        def e(context):  # pragma: no cover
            pass

        unregister("app.e")

        # Name is free again — re-registration must not raise.
        @register_task(name="app.e")
        def e2(context):  # pragma: no cover
            pass

        assert registry.get("app.e") is not None
