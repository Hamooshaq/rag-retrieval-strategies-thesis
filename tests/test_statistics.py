import pandas as pd

from evaluation.statistics import (
    cluster_bootstrap_ci,
    clustered_paired_comparisons,
    interaction_permutation_test,
    paired_binary_tests,
    run_pilot_statistics,
)


def _frame() -> pd.DataFrame:
    rows = []
    for question_index in range(12):
        question_type = "factual" if question_index % 2 == 0 else "global"
        document_id = f"d{question_index % 4}"
        for baseline, value in [("B1", 0.5), ("B2", 0.6), ("B3", 0.8)]:
            rows.append(
                {
                    "question_id": f"q{question_index}",
                    "document_id": document_id,
                    "question_type": question_type,
                    "baseline": baseline,
                    "metric": value + (0.1 if question_type == "global" and baseline == "B3" else 0),
                    "binary": int(baseline != "B1"),
                }
            )
    return pd.DataFrame(rows)


def test_effect_sizes_and_cluster_bootstrap_are_bounded():
    frame = _frame()
    result = run_pilot_statistics(frame, "metric")
    assert 0 <= result["friedman"]["kendalls_w"] <= 1
    assert all(
        -1 <= item["rank_biserial"] <= 1
        for item in result["pairwise_wilcoxon"].values()
    )
    low, high = cluster_bootstrap_ci(frame, "metric", "document_id", seed=7)
    assert low <= high


def test_interaction_and_binary_tests_return_auditable_outputs():
    frame = _frame()
    interaction = interaction_permutation_test(
        frame, "metric", repetitions=100, seed=7
    )
    assert interaction["comparisons"]["B3_minus_B1"]["n"] == 12
    binary = paired_binary_tests(frame, "binary")
    assert binary["n_complete_pairs"] == 12
    assert "cochrans_q" in binary


def test_clustered_strategy_comparison_uses_documents_as_units():
    result = clustered_paired_comparisons(
        _frame(),
        "metric",
        seed=7,
        bootstrap_repetitions=200,
    )
    comparison = result["comparisons"]["B1_vs_B3"]
    assert comparison["document_clusters"] == 4
    assert comparison["question_pairs"] == 12
    assert comparison["document_macro_mean_difference"] < 0
    assert len(comparison["document_bootstrap_ci95"]) == 2
