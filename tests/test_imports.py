"""Tests for mchammer_pt.parallel._imports helpers."""

from __future__ import annotations

import pytest

from mchammer_pt.parallel._imports import _check_importable


def test_check_importable_accepts_callable_instance_of_top_level_class():
    """Instance of a top-level callable class passes the check via type(obj)."""

    class TopLevelCallable:
        def __call__(self, replica):
            return None

    # Simulate the class living at module top level by overriding the
    # attributes that would otherwise contain "<locals>".
    TopLevelCallable.__qualname__ = "TopLevelCallable"
    TopLevelCallable.__module__ = __name__

    instance = TopLevelCallable()
    # Should not raise.
    _check_importable(instance, kind="observer factory")


def test_check_importable_rejects_callable_instance_of_function_local_class():
    """Instance of a function-local callable class is rejected via type(obj)."""

    class LocalCallable:
        def __call__(self, replica):
            return None

    instance = LocalCallable()
    with pytest.raises(ValueError, match="<locals>"):
        _check_importable(instance, kind="observer factory")


def test_check_importable_accepts_top_level_class():
    """A class defined at module scope passes the check."""
    from tests._observer_fixtures import StatefulCounter

    _check_importable(StatefulCounter, kind="observer class")


def test_check_importable_rejects_function_local_class():
    """A class defined inside a function is rejected."""

    class LocalClass:
        pass

    with pytest.raises(ValueError, match="<locals>"):
        _check_importable(LocalClass, kind="ensemble_cls")


def test_check_importable_accepts_plain_function():
    """A module-level function passes the check."""

    def factory(replica):
        return None

    # Simulate module-level by patching attributes.
    factory.__qualname__ = "factory"
    factory.__module__ = __name__

    _check_importable(factory, kind="factory")


def test_check_importable_rejects_function_local_function():
    """A function defined inside another function is rejected."""

    def outer():
        def inner(replica):
            return None

        return inner

    with pytest.raises(ValueError, match="<locals>"):
        _check_importable(outer(), kind="factory")
