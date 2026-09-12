"""Op registry: config references ops by stable name; unknown names fail loudly."""
from __future__ import annotations

from typing import Type

from mlpipeline.preprocessing.base import PreprocessOp


class PreprocessingRegistry:
    """Name -> op class. Populated via the ``register_op`` decorator."""

    def __init__(self) -> None:
        self._ops: dict[str, Type] = {}

    def register(self, op_cls: Type) -> Type:
        name = getattr(op_cls, "name", None)
        if not name or not isinstance(name, str):
            raise ValueError(f"{op_cls.__name__} must define a string `name` class attribute")
        if name in self._ops:
            raise ValueError(f"preprocessing op '{name}' already registered")
        self._ops[name] = op_cls
        return op_cls

    def get(self, name: str) -> Type:
        if name not in self._ops:
            raise KeyError(
                f"unknown preprocessing op '{name}'. known ops: {sorted(self._ops)}"
            )
        return self._ops[name]

    def known_names(self) -> list[str]:
        return sorted(self._ops)


_registry = PreprocessingRegistry()


def register_op(op_cls: Type) -> Type:
    """Class decorator: @register_op class FooOp: name = \"foo\"."""
    return _registry.register(op_cls)


def get_op(name: str) -> Type:
    return _registry.get(name)


def known_op_names() -> list[str]:
    return _registry.known_names()
