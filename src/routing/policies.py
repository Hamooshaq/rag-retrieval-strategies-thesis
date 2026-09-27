from __future__ import annotations

import re
from dataclasses import dataclass

from storage.models import QuestionRecord, QuestionType


@dataclass(frozen=True)
class RouteDecision:
    question_type: QuestionType
    route: str
    confidence: float
    matched_features: tuple[str, ...]


def route_for_type(question_type: QuestionType) -> str:
    return {
        QuestionType.FACTUAL: "child",
        QuestionType.CONCEPTUAL: "parent",
        QuestionType.GLOBAL: "parent_neighbor",
        QuestionType.MULTI_HOP: "distinct_parents",
    }[question_type]


class RuleRouter:
    GLOBAL_PATTERNS = (
        r"\bsummar(?:y|ize|ise)\b",
        r"\boverview\b",
        r"\bmain themes?\b",
        r"\bacross (?:the )?(?:document|deck|module|section)\b",
        r"\bwhole (?:document|deck|module)\b",
    )
    MULTI_HOP_PATTERNS = (
        r"\bcombine\b",
        r"\busing both\b",
        r"\bcompare\b",
        r"\brelationship between\b",
        r"\btogether\b",
        r"\bjointly\b",
        r"\brelat(?:e|ed|ionship)\b",
        r"^\s*how (?:do|does|are|can)\b.{3,100}\band\b",
        r"\btwo (?:facts|sections|slides|pages)\b",
    )
    CONCEPTUAL_PATTERNS = (
        r"^\s*why\b",
        r"^\s*how\b",
        r"\bexplain\b",
        r"\bconcept\b",
        r"\breason\b",
        r"\bdifference between\b",
    )
    FACTUAL_PATTERNS = (
        r"^\s*(?:what|which|who|when|where|how many)\b",
        r"\bvalue\b",
        r"\bname\b",
        r"\blimit\b",
    )

    def predict(self, question: str) -> RouteDecision:
        text = question.lower().strip()
        for question_type, patterns in [
            (QuestionType.GLOBAL, self.GLOBAL_PATTERNS),
            (QuestionType.MULTI_HOP, self.MULTI_HOP_PATTERNS),
            (QuestionType.CONCEPTUAL, self.CONCEPTUAL_PATTERNS),
            (QuestionType.FACTUAL, self.FACTUAL_PATTERNS),
        ]:
            matched = tuple(pattern for pattern in patterns if re.search(pattern, text))
            if matched:
                confidence = min(0.95, 0.70 + 0.08 * len(matched))
                return RouteDecision(
                    question_type=question_type,
                    route=route_for_type(question_type),
                    confidence=confidence,
                    matched_features=matched,
                )
        # Conservative default: a coherent parent is safer than a narrow child.
        return RouteDecision(
            question_type=QuestionType.CONCEPTUAL,
            route=route_for_type(QuestionType.CONCEPTUAL),
            confidence=0.50,
            matched_features=("default_parent",),
        )


class OracleRouter:
    def predict(self, question: QuestionRecord) -> RouteDecision:
        return RouteDecision(
            question_type=question.question_type,
            route=route_for_type(question.question_type),
            confidence=1.0,
            matched_features=("gold_question_type",),
        )
