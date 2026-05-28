"""Backend implementations for shot_tabular."""

from abc import ABC, abstractmethod

import xarray as xr


class Backend(ABC):
    """Abstract base class for backends."""

    @abstractmethod
    def get(self, shot: int, signal: str) -> xr.Dataset:
        """Retrieve dataset for a given shot and signal."""


class BackendRegistry:
    """Registry for backend implementations."""

    def __init__(self):
        self._backends: dict[str, type[Backend]] = {}

    def register(self, name: str, backend_class: type[Backend]) -> None:
        """Register a backend class with a given name."""
        if not issubclass(backend_class, Backend):
            raise TypeError(f"{backend_class} must be a subclass of Backend")
        self._backends[name] = backend_class

    def get(self, name: str) -> Backend:
        """Get a backend instance by name."""
        if name not in self._backends:
            raise KeyError(
                f"Backend '{name}' not found. Available: {list(self._backends.keys())}"
            )
        return self._backends[name]()

    def list_backends(self) -> list[str]:
        """List all registered backend names."""
        return list(self._backends.keys())


# Global registry instance
registry = BackendRegistry()


class UDABackend(Backend):
    """Backend implementation for UDA."""

    def get(self, shot: int, signal: str) -> xr.Dataset:
        dataset = xr.open_dataset(f"uda://{signal}:{shot}")
        return dataset


class SALBackend(Backend):
    """Backend implementation for SAL."""

    def get(self, shot: int, signal: str) -> xr.Dataset:
        dataset = xr.open_dataset(f"sal://pulse/{shot}/{signal}")
        return dataset


# Register backends
registry.register("uda", UDABackend)
registry.register("sal", SALBackend)
