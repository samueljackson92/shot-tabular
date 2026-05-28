import pytest
import xarray as xr

from shot_tabular.backends import Backend, BackendRegistry, registry


class DummyBackend(Backend):
    def get(self, shot: int, signal: str) -> xr.Dataset:
        return xr.Dataset()


class NotABackend:
    pass


def test_register_then_get_returns_instance():
    reg = BackendRegistry()
    reg.register("dummy", DummyBackend)
    assert isinstance(reg.get("dummy"), DummyBackend)


def test_get_unknown_backend_raises_key_error():
    reg = BackendRegistry()
    with pytest.raises(KeyError, match="nonexistent"):
        reg.get("nonexistent")


def test_register_non_backend_subclass_raises_type_error():
    reg = BackendRegistry()
    with pytest.raises(TypeError):
        reg.register("bad", NotABackend)  # type: ignore


def test_list_backends_returns_registered_names():
    reg = BackendRegistry()
    reg.register("a", DummyBackend)
    reg.register("b", DummyBackend)
    assert set(reg.list_backends()) == {"a", "b"}


def test_get_returns_fresh_instance_each_call():
    reg = BackendRegistry()
    reg.register("dummy", DummyBackend)
    assert reg.get("dummy") is not reg.get("dummy")


def test_global_registry_has_uda_and_sal():
    assert "uda" in registry.list_backends()
    assert "sal" in registry.list_backends()
