from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from storage.io import atomic_write_json, load_questions
from storage.models import RunRecord

from .metrics import evaluate_record, router_classification_metrics
from .statistics import (
    cluster_bootstrap_ci,
    clustered_paired_comparisons,
    interaction_permutation_test,
    paired_binary_tests,
    run_pilot_statistics,
)


def build_reports(
    records: list[RunRecord],
    questions_path: Path,
    output_dir: Path,
    top_k: int,
    seed: int,
    *,
    report_title: str = "Pilot Metrics",
    records_source: str = "raw/pilot_runs.jsonl",
    report_note: str = (
        "Pilot adalah feasibility study dan tidak menggantikan hasil penelitian final."
    ),
) -> dict[str, Path]:
    questions = {item.question_id: item for item in load_questions(questions_path)}
    error_records = [record for record in records if record.status == "error"]
    completed_records = [record for record in records if record.status == "completed"]
    rows = [
        evaluate_record(record, questions[record.question_id], top_k)
        for record in completed_records
        if record.question_id in questions
    ]
    if not rows:
        raise ValueError("No completed experiment records are available for evaluation")
    frame = pd.DataFrame(rows)
    metrics_dir = output_dir / "metrics"
    tables_dir = output_dir / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    per_run_path = metrics_dir / "pilot_per_run.csv"
    frame.to_csv(per_run_path, index=False)
    errors_path = metrics_dir / "pilot_errors.json"
    atomic_write_json(
        errors_path,
        [
            {
                "question_id": record.question_id,
                "baseline": record.baseline,
                "error_type": record.error_type,
                "error_message": record.error_message,
            }
            for record in error_records
        ],
    )

    numeric = [
        "hit_rate_at_k",
        "recall_at_k",
        "precision_at_k",
        "mrr",
        "ndcg_at_k",
        "source_location_accuracy",
        "all_evidence_coverage",
        "context_tokens",
        "retrieval_latency_ms",
        "end_to_end_latency_ms",
        "reference_keyword_recall",
        "citation_precision",
        "citation_recall",
        "correct_refusal",
        "false_refusal",
        "unsupported_answer",
    ]
    summary_rows: list[dict[str, object]] = []
    for baseline, subset in frame.groupby("baseline"):
        answerable = subset[subset["answerable"]]
        unanswerable = subset[~subset["answerable"]]
        row: dict[str, object] = {
            "baseline": baseline,
            "n": len(subset),
            "answerable_n": len(answerable),
            "unanswerable_n": len(unanswerable),
            "correct_refusal_rate_unanswerable": (
                float(unanswerable["correct_refusal"].mean())
                if len(unanswerable)
                and unanswerable["correct_refusal"].notna().any()
                else None
            ),
            "false_refusal_rate_answerable": (
                float(answerable["false_refusal"].mean())
                if len(answerable)
                and answerable["false_refusal"].notna().any()
                else None
            ),
            "unsupported_answer_rate_unanswerable": (
                float(unanswerable["unsupported_answer"].mean())
                if len(unanswerable)
                and unanswerable["unsupported_answer"].notna().any()
                else None
            ),
            "model_called_n": int(subset["generation_invoked"].sum()),
            "model_call_rate": float(subset["generation_invoked"].mean()),
        }
        model_called = subset[subset["generation_invoked"]]
        no_model_call = subset[~subset["generation_invoked"]]
        row["end_to_end_model_called_median"] = (
            float(model_called["end_to_end_latency_ms"].median())
            if len(model_called)
            else None
        )
        row["end_to_end_no_model_call_median"] = (
            float(no_model_call["end_to_end_latency_ms"].median())
            if len(no_model_call)
            else None
        )
        for metric in numeric:
            values = subset[metric].dropna().astype(float).to_numpy()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            row[f"{metric}_median"] = float(pd.Series(values).median()) if len(values) else None
            if len(values):
                low, high = cluster_bootstrap_ci(
                    subset, metric, cluster="document_id", seed=seed
                )
                row[f"{metric}_ci95_low"] = low
                row[f"{metric}_ci95_high"] = high
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("baseline")
    summary_path = metrics_dir / "pilot_summary.csv"
    summary.to_csv(summary_path, index=False)

    router = router_classification_metrics(completed_records)
    router_path = metrics_dir / "router_metrics.json"
    atomic_write_json(router_path, router)

    stats = {
        metric: run_pilot_statistics(frame, metric)
        for metric in ["recall_at_k", "source_location_accuracy", "context_tokens"]
    }
    stats["primary_document_clustered"] = {
        metric: clustered_paired_comparisons(frame, metric, seed=seed)
        for metric in (
            "recall_at_k",
            "source_location_accuracy",
            "context_tokens",
        )
    }
    stats["strategy_by_question_type"] = interaction_permutation_test(
        frame[frame["answerable"]],
        metric="recall_at_k",
        seed=seed,
    )
    stats["correct_refusal"] = paired_binary_tests(
        frame[~frame["answerable"]],
        outcome="correct_refusal",
    )
    stats["false_refusal"] = paired_binary_tests(
        frame[frame["answerable"]],
        outcome="false_refusal",
    )
    stats_path = metrics_dir / "pilot_statistics.json"
    atomic_write_json(stats_path, stats)

    failure = frame[
        (frame["answerable"] & ((frame["recall_at_k"].fillna(1) < 1) | (frame["false_refusal"] > 0)))
        | ((~frame["answerable"]) & (frame["unsupported_answer"] > 0))
    ]
    failure_path = metrics_dir / "failure_cases.csv"
    failure.to_csv(failure_path, index=False)

    markdown_path = tables_dir / "pilot_metrics.md"
    selected = summary[
        [
            "baseline",
            "n",
            "recall_at_k_mean",
            "hit_rate_at_k_mean",
            "precision_at_k_mean",
            "mrr_mean",
            "ndcg_at_k_mean",
            "source_location_accuracy_mean",
            "all_evidence_coverage_mean",
            "context_tokens_mean",
            "retrieval_latency_ms_median",
            "end_to_end_latency_ms_median",
            "model_call_rate",
            "end_to_end_model_called_median",
            "correct_refusal_rate_unanswerable",
            "false_refusal_rate_answerable",
        ]
    ].copy()
    selected.columns = [
        "Baseline",
        "N",
        "Recall@K",
        "Hit@K",
        "Precision@K",
        "MRR",
        "nDCG@K",
        "Source-location",
        "All-evidence coverage",
        "Context tokens",
        "Retrieval ms (median)",
        "End-to-end ms (median)",
        "Model-call rate",
        "E2E model-call ms (median)",
        "Correct refusal (unanswerable)",
        "False refusal (answerable)",
    ]
    selected = selected.astype(object).where(pd.notna(selected), None)
    markdown = (
        f"# {report_title}\n\n"
        f"Angka berasal dari `{records_source}`. {report_note}\n\n"
        + selected.to_markdown(index=False, floatfmt=".4f", missingval="N/A")
        + "\n\n## Router\n\n```json\n"
        + json.dumps(router, indent=2)
        + "\n```\n"
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    return {
        "per_run": per_run_path,
        "errors": errors_path,
        "summary": summary_path,
        "router": router_path,
        "statistics": stats_path,
        "failure_cases": failure_path,
        "markdown": markdown_path,
    }
