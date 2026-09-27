import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_f1():
    module = load_script('analyze_full_experiment')
    assert module.reference_f1('a b', 'a b') == 1
    assert module.reference_f1('x y', 'a b') == 0
    assert module.reference_f1('a', 'a b c') == .5


def test_model_rubric_rejects_unparseable_or_invented_scores():
    module = load_script('evaluate_answer_rubric')
    valid = '{"correctness": 2, "completeness": 1, "relevance": 2, "faithfulness": null}'
    assert module.parse_scores(valid)['faithfulness'] is None
    for invalid in ['good answer', '{"correctness": 2}', valid.replace('"correctness": 2', '"correctness": 7')]:
        with pytest.raises(ValueError):
            module.parse_scores(invalid)


def test_locator_ndcg_deduplicates_and_preserves_source_order():
    module = load_script('analyze_full_experiment')
    record = SimpleNamespace(baseline='B3', ranking=[
        SimpleNamespace(locators=['page:1', 'page:2']),
        SimpleNamespace(locators=['page:2', 'page:3']),
    ])
    assert module.locator_ndcg(record, {'page:2'}) == pytest.approx(1 / 1.584962500721156)
    assert module.locator_ndcg(record, {'page:1', 'page:2'}) == 1.0
    assert module.locator_ndcg(record, {'page:9'}) == 0.0
    assert module.locator_ndcg(record, set()) is None
    record.baseline = 'B0'
    assert module.locator_ndcg(record, {'page:1'}) is None


def test_rubric_boolean_and_fraction_are_not_ordinal_scores():
    module = load_script('evaluate_answer_rubric')
    for score in ['true', '1.5', '-1']:
        value = '{"correctness": ' + score + ', "completeness": 1, "relevance": 2, "faithfulness": 0}'
        with pytest.raises(ValueError):
            module.parse_scores(value)
