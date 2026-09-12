"""NavigationProvider protocol + registry (Section 10.2)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from mlpipeline.datatypes.navigation import NavigationTrack


@runtime_checkable
class NavigationProvider(Protocol):
    """One parser per nav format; probed via can_parse in registration order."""

    name: str

    def can_parse(self, source: Path) -> bool: ...

    def parse(self, source: Path) -> NavigationTrack: ...


class NavigationParseError(Exception):
    """Nav data present but unusable — maps to geo_status='unparseable_metadata'."""


_providers: list[Callable[[], NavigationProvider]] = []


def register_navigation_provider(factory: Callable[[], NavigationProvider]) -> Callable[[], NavigationProvider]:
    """Register a zero-arg factory returning a provider instance."""
    _providers.append(factory)
    return factory


def known_providers() -> list[str]:
    return [f().name for f in _providers]


def parse_navigation(source: str | Path) -> NavigationTrack:
    """Probe registered providers in order; first can_parse wins.

    Raises NavigationParseError when no provider accepts the source.
    """
    p = Path(source)
    if not p.is_file():
        raise NavigationParseError(f"navigation source not found: {p}")
    errors = []
    for factory in _providers:
        provider = factory()
        try:
            if provider.can_parse(p):
                track = provider.parse(p)
                if track.source_ref is None:
                    track = track.model_copy(update={"source_ref": str(p)})
                return track
        except NavigationParseError as e:
            errors.append(f"{provider.name}: {e}")
    raise NavigationParseError(
        f"no navigation provider could parse {p.name}. attempts: {errors or 'none registered'}"
    )
