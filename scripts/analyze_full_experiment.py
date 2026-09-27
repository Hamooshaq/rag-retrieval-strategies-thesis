"""Auditable analysis of the frozen experiment, without altering its raw outputs."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from evaluation.metrics import evaluate_record, router_classification_metrics
from evaluation.statistics import clustered_paired_comparisons, interaction_permutation_test, _holm_adjust
from storage.io import atomic_write_json, load_questions, sha256_file
from storage.models import RunRecord


def reference_f1(answer: str, reference: str) -> float:
    import re
    def tokens(value):
        return re.findall(r"[a-z0-9]+", value.lower())
    left, right = tokens(answer), tokens(reference)
    overlap = sum((Counter(left) & Counter(right)).values())
    return 2 * overlap / (len(left) + len(right)) if left or right else 1.0


def locator_ndcg(record: RunRecord, gold: set[str], k: int = 4) -> float | None:
    """Binary nDCG over first k distinct locators in stable expanded rank order.

    Unlike the legacy novelty-hit proxy, multiple relevant locators from one
    parent contribute separately. Ties within a unit retain source order.
    """
    if record.baseline == 'B0' or not gold:
        return None
    locators = list(dict.fromkeys(loc for hit in record.ranking for loc in hit.locators))[:k]
    dcg = sum(int(loc in gold) / math.log2(i + 2) for i, loc in enumerate(locators))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return dcg / ideal


def analyze(raw: Path, questions_path: Path, out: Path):
    questions = {q.question_id: q for q in load_questions(questions_path)}
    records = [RunRecord.model_validate_json(line) for line in raw.read_text().splitlines() if line.strip()]
    if len(records) != 800 or len({r.key for r in records}) != 800:
        raise ValueError('Full analysis requires exactly 800 unique records, not a partial run')
    if any(r.status != 'completed' or r.metadata.get('retrieval_only') for r in records):
        raise ValueError('Full analysis requires successful end-to-end records')
    if len({r.configuration_hash for r in records}) != 1:
        raise ValueError('Mixed configurations must not be pooled')
    expected = {(qid, b, 1) for qid in questions for b in ['B0', 'B1', 'B2', 'B3', 'B4']}
    if {(r.question_id, r.baseline, r.repeat_index) for r in records} != expected:
        raise ValueError('Question/baseline coverage differs from the frozen plan')
    out.mkdir(parents=True, exist_ok=True)
    rows, failure_cases = [], []
    for r in records:
        q = questions[r.question_id]
        row = evaluate_record(r, q, 4)
        row['legacy_novel_hit_ndcg_proxy'] = row.pop('ndcg_at_k')
        row['distinct_locator_ndcg_at4'] = locator_ndcg(r, set(q.gold_evidence.locators))
        row['reference_token_f1'] = reference_f1(r.output, q.reference_answer) if q.answerable else None
        normalize = lambda text: re.sub(r'\s+', ' ', text).strip().lower()
        spans = q.gold_evidence.spans
        row['gold_span_exact_coverage'] = (
            sum(normalize(span) in normalize(r.context.text) for span in spans) / len(spans)
            if q.answerable and spans and r.baseline != 'B0' else None
        )
        row['reached_token_limit'] = bool(r.metadata.get('reached_token_limit'))
        row['primary_text_only'] = not q.visual_required
        rows.append(row)
        reasons = []
        if q.visual_required: reasons.append('visual_required_excluded_primary')
        if row['false_refusal']: reasons.append('false_refusal')
        if row['unsupported_answer']: reasons.append('answered_unanswerable')
        if q.answerable and r.baseline != 'B0' and row['recall_at_k'] < 1: reasons.append('incomplete_locator_retrieval')
        if q.answerable and not row['refused'] and row['citation_recall'] == 0 and r.baseline != 'B0': reasons.append('no_gold_locator_cited')
        if row['reached_token_limit']: reasons.append('output_limit_reached')
        if reasons:
            failure_cases.append({'question_id': r.question_id, 'baseline': r.baseline,
                                  'reasons': reasons, 'question': r.question,
                                  'output': r.output, 'reference_answer': q.reference_answer,
                                  'gold_locators': q.gold_evidence.locators,
                                  'context_locators': r.context.locators})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'all_800_metrics.csv', index=False)
    primary = frame[frame.primary_text_only]
    primary.to_csv(out / 'primary_795_metrics.csv', index=False)
    summaries = []
    for population, subset in [('primary_text_only', primary), ('all_160_sensitivity', frame)]:
        for baseline, group in subset.groupby('baseline'):
            answerable, unanswerable = group[group.answerable], group[~group.answerable]
            row = {'population': population, 'baseline': baseline, 'n': len(group),
                   'answerable_n': len(answerable), 'unanswerable_n': len(unanswerable),
                   'correct_refusal_rate': unanswerable.correct_refusal.mean(),
                   'false_refusal_rate': answerable.false_refusal.mean(),
                   'generator_calls': int(group.generation_invoked.sum()),
                   'output_limit_count': int(group.reached_token_limit.sum())}
            for metric in ['recall_at_k', 'precision_at_k', 'hit_rate_at_k', 'mrr',
                           'distinct_locator_ndcg_at4', 'source_location_accuracy',
                           'gold_span_exact_coverage',
                           'reference_keyword_recall', 'reference_token_f1',
                           'citation_precision', 'citation_recall']:
                row[metric] = answerable[metric].mean()
            for metric in ['context_tokens', 'retrieval_latency_ms', 'end_to_end_latency_ms']:
                row[metric + '_mean'] = group[metric].mean()
                row[metric + '_median'] = group[metric].median()
                row[metric + '_p95'] = group[metric].quantile(.95)
            summaries.append(row)
    pd.DataFrame(summaries).to_csv(out / 'summary.csv', index=False)
    for label in ['question_type', 'document_id', 'document_format', 'domain']:
        primary[primary.answerable].groupby(['baseline', label])[
            ['recall_at_k', 'source_location_accuracy', 'reference_token_f1', 'context_tokens']
        ].agg(['count', 'mean']).to_csv(out / f'by_{label}.csv')
    stats = {}
    for metric in ['recall_at_k', 'source_location_accuracy', 'reference_token_f1', 'context_tokens', 'correct_refusal', 'false_refusal']:
        subset = primary[~primary.answerable] if metric == 'correct_refusal' else primary[primary.answerable]
        stats[metric] = clustered_paired_comparisons(subset, metric)
    stats['interaction_recall'] = interaction_permutation_test(primary[primary.answerable], 'recall_at_k')
    interactions = stats['interaction_recall']['comparisons']
    adjusted = _holm_adjust([(k, v['p_value']) for k, v in interactions.items() if 'p_value' in v])
    for key, p in adjusted.items(): interactions[key]['p_value_holm'] = p
    atomic_write_json(out / 'statistics.json', stats)
    atomic_write_json(out / 'failure_cases.json', failure_cases)
    atomic_write_json(out / 'router.json', router_classification_metrics([r for r in records if not questions[r.question_id].visual_required]))
    atomic_write_json(out / 'verification.json', {
        'raw_sha256': sha256_file(raw), 'questions_sha256': sha256_file(questions_path),
        'configuration_hash': records[0].configuration_hash,
        'complete_records': len(records), 'primary_records': len(primary),
        'excluded_question_ids': [q.question_id for q in questions.values() if q.visual_required],
        'failure_reason_counts': dict(Counter(reason for case in failure_cases for reason in case['reasons'])),
        'semantics': 'All answer and citation overlap scores are automatic proxies, not human correctness or faithfulness.',
        'inference': 'Exploratory; document sign-flip assumes symmetric exchangeable paired document differences; ten purposively sampled documents limit generalization.',
        'multiplicity': 'Holm over three strategy pairs within each metric; two interaction contrasts corrected separately. No global confirmatory claim across metrics.',
    })
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=Path('artifacts/final_dataset_v1/full_experiment/raw/primary_runs.jsonl'))
    parser.add_argument('--questions', type=Path, default=Path('data/final_dataset/test_questions.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/final_dataset_v1/full_experiment/analysis'))
    args = parser.parse_args()
    analyze(args.raw, args.questions, args.output)
