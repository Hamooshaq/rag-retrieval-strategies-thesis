"""Summarize repeat timing without treating repeated answers as new samples."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from storage.io import atomic_write_json, sha256_file
from storage.models import RunRecord


def analyze(raw: Path, output: Path) -> None:
    records = [RunRecord.model_validate_json(line) for line in raw.read_text().splitlines() if line.strip()]
    if len(records) != 320 or len({record.key for record in records}) != 320:
        raise ValueError('Expected 320 unique repeated records')
    if any(record.status != 'completed' for record in records):
        raise ValueError('Timing summary requires completed records')
    if len({record.configuration_hash for record in records}) != 1:
        raise ValueError('Mixed repeated configurations')
    question_ids = {record.question_id for record in records}
    expected = {(qid, baseline, repeat) for qid in question_ids
                for baseline in ['B0', 'B1', 'B2', 'B3', 'B4'] for repeat in [2, 3]}
    if len(question_ids) != 32 or {(r.question_id, r.baseline, r.repeat_index) for r in records} != expected:
        raise ValueError('Repeated question/baseline coverage differs from protocol')
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{
        'question_id': r.question_id, 'baseline': r.baseline, 'repeat_index': r.repeat_index,
        'generation_invoked': bool(r.metadata.get('generation_invoked')),
        'retrieval_latency_ms': r.retrieval_latency_ms,
        'generation_latency_ms': r.generation_latency_ms,
        'end_to_end_latency_ms': r.end_to_end_latency_ms,
        'output': r.output,
    } for r in records])
    summaries = []
    for (baseline, repeat), group in frame.groupby(['baseline', 'repeat_index']):
        row = {'baseline': baseline, 'repeat_index': int(repeat), 'records': len(group),
               'generation_calls': int(group.generation_invoked.sum())}
        for metric in ['retrieval_latency_ms', 'end_to_end_latency_ms']:
            for label, quantile in [('q25', .25), ('median', .5), ('q75', .75), ('p95', .95)]:
                row[metric + '_' + label] = float(group[metric].quantile(quantile))
        generated = group[group.generation_invoked]
        row['generation_latency_ms_median'] = float(generated.generation_latency_ms.median())
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(output / 'timing_summary.csv', index=False)
    agreement = {}
    for baseline, group in frame.groupby('baseline'):
        pairs = group.pivot(index='question_id', columns='repeat_index', values='output')
        agreement[baseline] = {'pairs': len(pairs), 'exact_output_matches': int((pairs[2] == pairs[3]).sum())}
    atomic_write_json(output / 'verification.json', {
        'raw_sha256': sha256_file(raw), 'script_sha256': sha256_file(Path(__file__)),
        'records': 320, 'questions': 32, 'repeats': [2, 3],
        'configuration_hash': records[0].configuration_hash,
        'exact_output_repeatability': agreement,
        'interpretation': 'Timing diagnostics on the same 32 questions. Repeats are not independent quality samples. Device load and thermal state can change timings.',
    })
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=Path('artifacts/final_dataset_v1/repeatability/raw/repeatability_runs.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/final_dataset_v1/repeatability/analysis'))
    args = parser.parse_args()
    analyze(args.raw, args.output)
