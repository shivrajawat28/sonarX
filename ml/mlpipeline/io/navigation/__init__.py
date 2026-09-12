"""Navigation metadata parsers (Section 10.2).

Parsers register via @register_navigation_provider and are probed in order with
can_parse(). Adding a vendor format = one new module + import here (OPEN #3).
"""
from mlpipeline.io.navigation.base import (
    NavigationProvider,
    NavigationParseError,
    register_navigation_provider,
    parse_navigation,
    known_providers,
)
from mlpipeline.io.navigation.generic_csv import GenericCSVParser

__all__ = [
    "NavigationProvider",
    "NavigationParseError",
    "register_navigation_provider",
    "parse_navigation",
    "known_providers",
    "GenericCSVParser",
]
