from pathlib import Path

from evaluation.metrics import is_protocol_refusal
from experiments.runner import ExperimentRunner


def test_refusal_marker_must_not_hide_substantive_answer():
    assert is_protocol_refusal("INSUFFICIENT_EVIDENCE")
    assert is_protocol_refusal("  insufficient_evidence. ")
    assert not is_protocol_refusal("INSUFFICIENT_EVIDENCE but the answer is 42")
    assert not is_protocol_refusal("Do not return INSUFFICIENT_EVIDENCE")
    assert not is_protocol_refusal("")


def test_final_execution_plan_and_stratified_repeat_subset():
    root = Path(__file__).resolve().parents[1]
    runner = ExperimentRunner(root / "configs/full_experiment.yaml")
    assert len(runner.evaluation_questions) == 160
    assert runner.repeat_indices == [1]
    runner._validate_execution_mode(False)
    assert runner._result_path(False).name == "primary_runs.jsonl"
    selected = runner.select_timing_questions(runner.evaluation_questions, 32, runner.seed)
    selected_again = runner.select_timing_questions(runner.evaluation_questions, 32, runner.seed)
    assert len({q.question_id for q in selected}) == 32
    assert [q.question_id for q in selected] == [q.question_id for q in selected_again]
    assert {q.answerable for q in selected} == {True, False}
    assert len({q.domain for q in selected}) == 2
