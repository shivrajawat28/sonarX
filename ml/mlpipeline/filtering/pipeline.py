"""FilterPipeline: config-driven rule chain -> final_confidence + status + reasons.

Annotates, never deletes (ADR-007). The filter config hash is recorded on every
detection it touches for reproducibility (Section 9.4).
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from mlpipeline.config.schemas import FilterConfig
from mlpipeline.datatypes.detection import Detection, FilterStatus
from mlpipeline.filtering.base import FilterContext, Verdict
from mlpipeline.filtering.scorer import score_detection, status_for


class FilteredResult(BaseModel):
    detections: list[Detection] = Field(default_factory=list)
    config_hash: str | None = None
    n_accepted: int = 0
    n_flagged: int = 0
    n_rejected: int = 0


class FilterPipeline:
    def __init__(self, config: FilterConfig, config_hash: str | None = None) -> None:
        import mlpipeline.filtering.rules  # noqa: F401 — register built-ins

        self.config = config
        self.config_hash = config_hash

    def _rule_class(self, name: str):
        from mlpipeline.filtering.rule_registry import get_rule

        return get_rule(name)  # shared registry — built-ins registered on import

    def apply(self, detections: list[Detection], ctx: FilterContext) -> FilteredResult:
        out: list[Detection] = []
        n_acc = n_flag = n_rej = 0
        for det in detections:
            verdicts: list[Verdict] = []
            for spec in self.config.enabled_rules:
                if not spec.enabled:
                    continue
                if spec.applies_to and det.class_name not in spec.applies_to:
                    continue
                rule = self._rule_class(spec.rule)()
                verdict = rule.evaluate(det, ctx, spec.params)
                # Honor the configured per-rule penalty cap (Section 9.4):
                # rules report their raw severity; the config decides how much
                # a rule may reduce final_confidence. A cap of 0.0 means the
                # rule's default severity applies unchanged (bugfix: the cap
                # in rules.yaml was previously parsed but never applied, so a
                # single edge_clip trigger zeroed out any confidence).
                if verdict.triggered and spec.penalty > 0:
                    verdict = Verdict(
                        rule=verdict.rule,
                        penalty=min(verdict.penalty, spec.penalty),
                        reason=verdict.reason,
                        triggered=True,
                    )
                verdicts.append(verdict)
            final, reasons = score_detection(det, verdicts)
            status: FilterStatus = status_for(
                final, self.config.policy.accept_threshold, self.config.policy.flag_threshold
            )  # type: ignore[assignment]
            updated = det.model_copy(
                update={
                    "final_confidence": final,
                    "filtering_status": status,
                    "filter_reasons": reasons,
                    "filter_config_hash": self.config_hash,
                }
            )
            out.append(updated)
            if status == "accepted":
                n_acc += 1
            elif status == "flagged":
                n_flag += 1
            else:
                n_rej += 1
        return FilteredResult(
            detections=out, config_hash=self.config_hash,
            n_accepted=n_acc, n_flagged=n_flag, n_rejected=n_rej,
        )
