"""Detector registry: maps a `kind` string to an adapter factory.

Adding a new detector architecture = one adapter module + one registration.
Nothing else in the app changes (ADR-002).
"""
from __future__ import annotations

from typing import Callable, Type

from mlpipeline.detection.base import Detector


class DetectorRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], Detector]] = {}

    def register(self, kind: str, factory: Callable[[], Detector]) -> None:
        if kind in self._factories:
            raise ValueError(f"detector kind '{kind}' already registered")
        self._factories[kind] = factory

    def register_class(self, kind: str, cls: Type) -> None:
        self.register(kind, cls)

    def create(self, kind: str) -> Detector:
        if kind not in self._factories:
            raise KeyError(
                f"unknown detector kind '{kind}'. known: {sorted(self._factories)}"
            )
        return self._factories[kind]()

    def known_kinds(self) -> list[str]:
        return sorted(self._factories)


_registry = DetectorRegistry()


def register_detector(kind: str):
    """Class decorator: @register_detector("yolo") class YOLODetector."""

    def _wrap(cls: Type) -> Type:
        _registry.register_class(kind, cls)
        return cls

    return _wrap


def create_detector(kind: str) -> Detector:
    return _registry.create(kind)


def known_detector_kinds() -> list[str]:
    return _registry.known_kinds()
