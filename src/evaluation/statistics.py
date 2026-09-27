from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, friedmanchisquare, rankdata, wilcoxon


def paired_bootstrap_ci(
    values: np.ndarray, seed: int, repetitions: int = 2_000, confidence: float = 0.95
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repetitions, values.size), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def cluster_bootstrap_ci(
    frame: pd.DataFrame,
    metric: str,
    cluster: str,
    seed: int,
    repetitions: int = 2_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    clean = frame[[cluster, metric]].dropna()
    clusters = clean[cluster].unique()
    if len(clusters) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(repetitions):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        values = [
            clean.loc[clean[cluster] == cluster_id, metric].to_numpy(dtype=float)
            for cluster_id in selected
        ]
        estimates.append(float(np.concatenate(values).mean()))
    alpha = (1 - confidence) / 2
    return (
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1 - alpha)),
    )


def _rank_biserial(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[~np.isclose(differences, 0)]
    if differences.size == 0:
        return 0.0
    ranks = rankdata(np.abs(differences))
    positive = float(ranks[differences > 0].sum())
    negative = float(ranks[differences < 0].sum())
    return (positive - negative) / (positive + negative)


def _holm_adjust(p_values: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (label, p_value) in enumerate(ordered):
        candidate = min(1.0, p_value * (count - index))
        running = max(running, candidate)
        adjusted[label] = running
    return adjusted


def run_pilot_statistics(
    frame: pd.DataFrame,
    metric: str,
    baselines: tuple[str, ...] = ("B1", "B2", "B3"),
) -> dict[str, object]:
    subset = frame[frame["baseline"].isin(baselines) & frame[metric].notna()]
    pivot = subset.pivot_table(index="question_id", columns="baseline", values=metric)
    pivot = pivot.dropna()
    result: dict[str, object] = {
        "metric": metric,
        "n_complete_pairs": len(pivot),
        "note": (
            "Question-level Friedman/Wilcoxon values are exploratory diagnostics, "
            "not the primary inference because questions are nested in documents."
        ),
    }
    if len(pivot) < 5 or any(name not in pivot for name in baselines):
        result["status"] = "insufficient_complete_pairs"
        return result
    try:
        statistic, p_value = friedmanchisquare(*(pivot[name].to_numpy() for name in baselines))
        kendalls_w = float(statistic) / (len(pivot) * (len(baselines) - 1))
        result["friedman"] = {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "kendalls_w": kendalls_w,
        }
    except ValueError as exc:
        result["friedman"] = {"error": str(exc)}
        return result
    pairwise: list[tuple[str, float]] = []
    raw: dict[str, object] = {}
    for left, right in combinations(baselines, 2):
        differences = pivot[left].to_numpy() - pivot[right].to_numpy()
        if np.allclose(differences, 0):
            p = 1.0
            statistic = 0.0
        else:
            test = wilcoxon(pivot[left], pivot[right], zero_method="pratt")
            statistic, p = float(test.statistic), float(test.pvalue)
        label = f"{left}_vs_{right}"
        pairwise.append((label, p))
        raw[label] = {
            "statistic": statistic,
            "p_value_raw": p,
            "mean_difference": float(differences.mean()),
            "rank_biserial": _rank_biserial(differences),
        }
    adjusted = _holm_adjust(pairwise)
    for label, value in raw.items():
        value["p_value_holm"] = adjusted[label]
    result["pairwise_wilcoxon"] = raw
    return result


def clustered_paired_comparisons(
    frame: pd.DataFrame,
    metric: str,
    baselines: tuple[str, ...] = ("B1", "B2", "B3"),
    cluster: str = "document_id",
    seed: int = 260226,
    bootstrap_repetitions: int = 10_000,
) -> dict[str, object]:
    """Compare paired strategies with documents as the inferential units."""

    metadata = (
        frame[["question_id", cluster]]
        .drop_duplicates("question_id")
        .set_index("question_id")
    )
    pivot = frame.pivot_table(index="question_id", columns="baseline", values=metric)
    result: dict[str, object] = {
        "metric": metric,
        "cluster": cluster,
        "method": (
            "document-macro paired mean difference with document bootstrap CI "
            "and two-sided document sign-flip permutation"
        ),
        "comparisons": {},
        "note": (
            "With only ten documents, intervals and p-values are reported as "
            "corpus-specific evidence."
        ),
    }
    rng = np.random.default_rng(seed)
    raw_p_values: list[tuple[str, float]] = []

    for left, right in combinations(baselines, 2):
        label = f"{left}_vs_{right}"
        if left not in pivot or right not in pivot:
            result["comparisons"][label] = {"status": "missing_baseline"}
            continue
        joined = pivot[[left, right]].dropna().join(metadata, how="inner")
        joined["difference"] = joined[left] - joined[right]
        document_means = joined.groupby(cluster)["difference"].mean().to_numpy(float)
        if len(document_means) < 4:
            result["comparisons"][label] = {
                "status": "insufficient_document_clusters",
                "document_clusters": len(document_means),
            }
            continue

        observed = float(document_means.mean())
        bootstrap = rng.choice(
            document_means,
            size=(bootstrap_repetitions, len(document_means)),
            replace=True,
        ).mean(axis=1)
        ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])

        if len(document_means) <= 16:
            permuted = np.asarray(
                [
                    float(np.mean(document_means * np.asarray(signs)))
                    for signs in product((-1.0, 1.0), repeat=len(document_means))
                ]
            )
            p_value = float(
                np.mean(np.abs(permuted) >= abs(observed) - 1e-12)
            )
            permutation_count = int(len(permuted))
        else:
            signs = rng.choice(
                (-1.0, 1.0),
                size=(10_000, len(document_means)),
            )
            permuted = (signs * document_means).mean(axis=1)
            p_value = float(
                (np.sum(np.abs(permuted) >= abs(observed) - 1e-12) + 1)
                / (len(permuted) + 1)
            )
            permutation_count = int(len(permuted))

        raw_p_values.append((label, p_value))
        result["comparisons"][label] = {
            "question_pairs": len(joined),
            "document_clusters": len(document_means),
            "document_macro_mean_difference": observed,
            "document_bootstrap_ci95": [float(ci_low), float(ci_high)],
            "document_rank_biserial": _rank_biserial(document_means),
            "p_value_raw": p_value,
            "sign_flip_permutations": permutation_count,
        }

    adjusted = _holm_adjust(raw_p_values)
    for label, p_value in adjusted.items():
        result["comparisons"][label]["p_value_holm"] = p_value
    return result


def interaction_permutation_test(
    frame: pd.DataFrame,
    metric: str,
    strategy_pairs: tuple[tuple[str, str], ...] = (("B3", "B1"), ("B3", "B2")),
    repetitions: int = 5_000,
    seed: int = 260226,
) -> dict[str, object]:
    """Test whether paired strategy differences vary by question type.

    Question-type labels are permuted within document so that document composition
    and the repeated-measures difference for each question remain intact.
    """

    metadata = (
        frame[["question_id", "document_id", "question_type"]]
        .drop_duplicates("question_id")
        .set_index("question_id")
    )
    pivot = frame.pivot_table(index="question_id", columns="baseline", values=metric)
    rng = np.random.default_rng(seed)
    result: dict[str, object] = {
        "metric": metric,
        "method": "document-stratified permutation of question-type labels",
        "repetitions": repetitions,
        "comparisons": {},
        "note": "Pilot interaction tests are exploratory.",
    }

    for left, right in strategy_pairs:
        if left not in pivot or right not in pivot:
            result["comparisons"][f"{left}_minus_{right}"] = {
                "status": "missing_baseline"
            }
            continue
        joined = pivot[[left, right]].dropna().join(metadata, how="inner")
        joined["difference"] = joined[left] - joined[right]
        if len(joined) < 8 or joined["question_type"].nunique() < 2:
            result["comparisons"][f"{left}_minus_{right}"] = {
                "status": "insufficient_data",
                "n": len(joined),
            }
            continue

        def statistic(labels: pd.Series, analysis_frame: pd.DataFrame = joined) -> float:
            values = analysis_frame.assign(_label=labels.to_numpy())
            overall = float(values["difference"].mean())
            return float(
                sum(
                    len(group) * (float(group["difference"].mean()) - overall) ** 2
                    for _, group in values.groupby("_label")
                )
            )

        observed = statistic(joined["question_type"])
        exceed = 0
        labels = joined["question_type"].copy()
        for _ in range(repetitions):
            permuted = labels.copy()
            for index in joined.groupby("document_id").groups.values():
                values = labels.loc[index].to_numpy(copy=True)
                rng.shuffle(values)
                permuted.loc[index] = values
            exceed += int(statistic(permuted) >= observed - 1e-12)
        by_type = {
            str(label): {
                "n": len(group),
                "mean_difference": float(group["difference"].mean()),
            }
            for label, group in joined.groupby("question_type")
        }
        result["comparisons"][f"{left}_minus_{right}"] = {
            "n": len(joined),
            "statistic": observed,
            "p_value": (exceed + 1) / (repetitions + 1),
            "mean_difference_by_type": by_type,
        }
    return result


def paired_binary_tests(
    frame: pd.DataFrame,
    outcome: str,
    baselines: tuple[str, ...] = ("B1", "B2", "B3"),
) -> dict[str, object]:
    pivot = frame.pivot_table(index="question_id", columns="baseline", values=outcome)
    pivot = pivot.dropna()
    result: dict[str, object] = {
        "outcome": outcome,
        "n_complete_pairs": len(pivot),
    }
    if len(pivot) < 5 or any(name not in pivot for name in baselines):
        result["status"] = "insufficient_complete_pairs"
        return result
    matrix = pivot[list(baselines)].to_numpy(dtype=int)
    column_totals = matrix.sum(axis=0)
    row_totals = matrix.sum(axis=1)
    total = matrix.sum()
    denominator = len(baselines) * total - np.square(row_totals).sum()
    if denominator == 0:
        q_statistic = 0.0
        q_p = 1.0
    else:
        numerator = (len(baselines) - 1) * (
            len(baselines) * np.square(column_totals).sum() - total**2
        )
        q_statistic = float(numerator / denominator)
        q_p = float(chi2.sf(q_statistic, df=len(baselines) - 1))
    result["cochrans_q"] = {"statistic": q_statistic, "p_value": q_p}

    pairwise: list[tuple[str, float]] = []
    details: dict[str, object] = {}
    for left, right in combinations(baselines, 2):
        left_values = pivot[left].astype(int)
        right_values = pivot[right].astype(int)
        left_only = int(((left_values == 1) & (right_values == 0)).sum())
        right_only = int(((left_values == 0) & (right_values == 1)).sum())
        discordant = left_only + right_only
        p_value = (
            float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue)
            if discordant
            else 1.0
        )
        label = f"{left}_vs_{right}"
        pairwise.append((label, p_value))
        details[label] = {
            "left_only": left_only,
            "right_only": right_only,
            "p_value_raw": p_value,
        }
    adjusted = _holm_adjust(pairwise)
    for label, detail in details.items():
        detail["p_value_holm"] = adjusted[label]
    result["pairwise_exact_mcnemar"] = details
    return result
