from __future__ import annotations

import math
import re
from collections import Counter

from storage.models import QuestionRecord, RunRecord

REFUSAL_TOKEN = "INSUFFICIENT_EVIDENCE"


def is_protocol_refusal(output: str) -> bool:
    """Accept only the stand-alone protocol marker, with harmless punctuation."""
    return output.strip().strip('`\"\' .!').upper() == REFUSAL_TOKEN


def _ordered_locator_set(record: RunRecord, top_k: int) -> list[str]:
    locators: list[str] = []
    for hit in record.ranking[:top_k]:
        locators.extend(hit.locators)
    return list(dict.fromkeys(locators))


def _valid_evidence_sets(question: QuestionRecord) -> list[list[str]]:
    candidates = [
        question.gold_evidence.locators,
        *question.gold_evidence.alternative_locator_sets,
    ]
    unique: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for candidate in candidates:
        normalized = tuple(dict.fromkeys(candidate))
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(list(normalized))
    return unique


def _locator_retrieval_bundle(
    record: RunRecord,
    top_k: int,
    gold: list[str],
) -> dict[str, float]:
    predicted = _ordered_locator_set(record, top_k)
    predicted_set = set(predicted)
    gold_set = set(gold)
    intersection = predicted_set & gold_set
    reciprocal_rank = 0.0
    relevance: list[int] = []
    covered_gold: set[str] = set()
    for rank, hit_record in enumerate(record.ranking[:top_k], 1):
        newly_covered = (set(hit_record.locators) & gold_set) - covered_gold
        is_relevant = int(bool(newly_covered))
        relevance.append(is_relevant)
        if is_relevant and reciprocal_rank == 0:
            reciprocal_rank = 1.0 / rank
        covered_gold.update(newly_covered)
    dcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(relevance))
    ideal = [1] * min(len(gold_set), top_k)
    idcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(ideal))
    context_locators = set(record.context.locators)
    context_intersection = context_locators & gold_set
    context_union = context_locators | gold_set
    return {
        "hit_rate_at_k": float(bool(intersection)),
        "recall_at_k": len(intersection) / len(gold_set),
        "precision_at_k": (
            len(intersection) / len(predicted_set) if predicted_set else 0.0
        ),
        "mrr": reciprocal_rank,
        "ndcg_at_k": dcg / idcg if idcg else 0.0,
        "source_location_accuracy": (
            len(context_intersection) / len(context_union) if context_union else 1.0
        ),
        "all_evidence_coverage": float(gold_set.issubset(context_locators)),
    }


def retrieval_metrics(
    record: RunRecord,
    question: QuestionRecord,
    top_k: int,
) -> dict[str, float | None]:
    if record.baseline == "B0":
        return {
            "hit_rate_at_k": None,
            "recall_at_k": None,
            "precision_at_k": None,
            "mrr": None,
            "ndcg_at_k": None,
            "source_location_accuracy": None,
            "all_evidence_coverage": None,
        }
    evidence_sets = _valid_evidence_sets(question)
    if not evidence_sets:
        return {
            "hit_rate_at_k": None,
            "recall_at_k": None,
            "precision_at_k": None,
            "mrr": None,
            "ndcg_at_k": None,
            "source_location_accuracy": None,
            "all_evidence_coverage": None,
        }
    bundles = [
        _locator_retrieval_bundle(record, top_k, gold)
        for gold in evidence_sets
    ]
    # Every support set is annotated before the run and is a valid way to answer
    # the question. Evaluate each metric against the best valid set, analogous
    # to multi-reference answer evaluation.
    return {
        metric: max(bundle[metric] for bundle in bundles)
        for metric in bundles[0]
    }


def extract_citations(output: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\[(?:page|slide):[1-9][0-9]*\]", output)))


def _normalize_locator(citation: str) -> str:
    return citation.strip("[]")


def answer_metrics(
    record: RunRecord,
    question: QuestionRecord,
) -> dict[str, float | None]:
    if bool(record.metadata.get("retrieval_only", False)):
        return {
            "refused": None,
            "correct_refusal": None,
            "false_refusal": None,
            "unsupported_answer": None,
            "reference_keyword_recall": None,
            "citation_precision": None,
            "citation_recall": None,
        }
    output_lower = record.output.lower()
    refused = is_protocol_refusal(record.output)
    correct_refusal = float((not question.answerable) and refused)
    false_refusal = float(question.answerable and refused)
    unsupported_answer = float((not question.answerable) and not refused)

    keywords = [keyword.lower() for keyword in question.expected_keywords]
    keyword_recall = (
        sum(keyword in output_lower for keyword in keywords) / len(keywords)
        if keywords and question.answerable and not refused
        else 0.0
    )
    cited = {_normalize_locator(value) for value in extract_citations(record.output)}
    evidence_sets = [set(item) for item in _valid_evidence_sets(question)]
    citation_scores = [
        (
            len(cited & gold) / len(cited) if cited else 0.0,
            len(cited & gold) / len(gold) if gold else 0.0,
        )
        for gold in evidence_sets
    ]
    citation_precision = max((score[0] for score in citation_scores), default=0.0)
    citation_recall = max((score[1] for score in citation_scores), default=0.0)
    return {
        "refused": float(refused),
        "correct_refusal": correct_refusal,
        "false_refusal": false_refusal,
        "unsupported_answer": unsupported_answer,
        "reference_keyword_recall": keyword_recall,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
    }


def evaluate_record(
    record: RunRecord, question: QuestionRecord, top_k: int
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "run_id": record.run_id,
        "question_id": record.question_id,
        "document_id": record.document_id,
        "baseline": record.baseline,
        "status": record.status,
        "question_type": question.question_type.value,
        "predicted_type": (
            record.question_type_predicted.value if record.question_type_predicted else None
        ),
        "answerable": question.answerable,
        "domain": question.domain,
        "document_format": question.document_format,
        "visual_required": question.visual_required,
        "repeat_index": record.repeat_index,
        "route": record.route,
        "context_tokens": record.context.token_count,
        "retrieval_latency_ms": record.retrieval_latency_ms,
        "generation_latency_ms": record.generation_latency_ms,
        "end_to_end_latency_ms": record.end_to_end_latency_ms,
        "prompt_tokens": record.prompt_tokens,
        "output_tokens": record.output_tokens,
        "generation_invoked": bool(record.metadata.get("generation_invoked", False)),
        "run_order_position": record.metadata.get("run_order_position"),
    }
    metrics.update(retrieval_metrics(record, question, top_k))
    metrics.update(answer_metrics(record, question))
    return metrics


def router_classification_metrics(records: list[RunRecord]) -> dict[str, object]:
    pairs = {
        record.question_id: (record.question_type_gold.value, record.question_type_predicted.value)
        for record in records
        if record.baseline == "B3"
        and record.status == "completed"
        and record.question_type_predicted is not None
    }
    if not pairs:
        return {"macro_f1": None, "accuracy": None, "classes": {}, "n": 0}
    classes = sorted({gold for gold, _ in pairs.values()} | {pred for _, pred in pairs.values()})
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    correct = 0
    for gold, pred in pairs.values():
        correct += int(gold == pred)
    for label in classes:
        tp = sum(gold == label and pred == label for gold, pred in pairs.values())
        fp = sum(gold != label and pred == label for gold, pred in pairs.values())
        fn = sum(gold == label and pred != label for gold, pred in pairs.values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[label] = {
            "support": sum(gold == label for gold, _ in pairs.values()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "macro_f1": sum(f1_values) / len(f1_values),
        "accuracy": correct / len(pairs),
        "classes": per_class,
        "n": len(pairs),
        "confusion": dict(Counter(f"{gold}->{pred}" for gold, pred in pairs.values())),
    }
