"""Secondary validator: INTERFACE STUB ONLY (Section 9.5, What-NOT-to-Build #4).

Exists so a verification model can be added later WITHOUT re-architecture.
Never registered as an enabled rule by any bundled config at MVP.
"""
from __future__ import annotations

from typing import Any, Protocol

from mlpipeline.datatypes.detection import Detection
from mlpipeline.filtering.base import FilterContext, Verdict
from mlpipeline.filtering.rule_registry import register_rule


class VerificationModel(Protocol):
    """Future seam: a lightweight second-stage classifier would implement this."""

    def verify(self, detection: Detection, pixels) -> tuple[bool, float]:
        """Returns (is_plausible, penalty_to_apply)."""
        ...


@register_rule
class SecondaryValidatorStub:
    name = "secondary_validator"

    def __init__(self, verifier: VerificationModel | None = None) -> None:
        self._verifier = verifier

    def evaluate(self, detection: Detection, ctx: FilterContext, params: dict[str, Any]) -> Verdict:
        if self._verifier is None:
            return Verdict(rule=self.name, penalty=0.0)
        plausible, penalty = self._verifier.verify(detection, ctx.pixels)
        if not plausible:
            return Verdict(
                rule=self.name, penalty=float(penalty), triggered=True,
                reason=f"secondary_validator: verifier rejected (penalty {penalty:.2f})",
            )
        return Verdict(rule=self.name, penalty=0.0)
