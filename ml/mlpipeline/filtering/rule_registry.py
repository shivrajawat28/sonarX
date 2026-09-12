"""Rule registry: config references filter rules by stable name."""
from __future__ import annotations

from typing import Type

from mlpipeline.filtering.base import FilterRule


class RuleRegistry:
    def __init__(self) -> None:
        self._rules: dict[str, Type] = {}

    def register(self, cls: Type) -> Type:
        name = getattr(cls, "name", None)
        if not name or not isinstance(name, str):
            raise ValueError(f"{cls.__name__} must define a string `name` attribute")
        if name in self._rules:
            raise ValueError(f"filter rule '{name}' already registered")
        self._rules[name] = cls
        return cls

    def get(self, name: str) -> Type:
        if name not in self._rules:
            raise KeyError(f"unknown filter rule '{name}'. known: {sorted(self._rules)}")
        return self._rules[name]

    def known_names(self) -> list[str]:
        return sorted(self._rules)


_rule_registry = RuleRegistry()


def register_rule(cls: Type) -> Type:
    return _rule_registry.register(cls)


def get_rule(name: str) -> Type:
    return _rule_registry.get(name)


def known_rule_names() -> list[str]:
    return _rule_registry.known_names()
