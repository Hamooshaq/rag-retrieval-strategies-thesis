from pathlib import Path

import experiments.runner as runner_module
from chunking import chunk_fixed, chunk_parents
from generation import ContextBuilder, ExtractiveGenerator, PromptBuilder, WhitespaceTokenCounter
from ingestion import parse_document
from ingestion.parsers import _shape_sort_key
from retrieval import ExactCosineIndex, HashingEmbedder
from storage.io import JsonlResultStore, load_document_manifest
from structure import build_structure


def test_shape_sort_key_tolerates_missing_coordinates():
    class LegacyShape:
        top = None
        left = None

    assert _shape_sort_key(LegacyShape()) == (0, 0)


def test_generated_pdf_and_pptx_parse_and_retrieve():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data" / "manifests" / "pilot_documents.yaml"
    assert manifest_path.exists(), "Run scripts/create_pilot_corpus.py before tests"
    documents = load_document_manifest(manifest_path)
    base_dir = manifest_path.parent.parent
    formats = set()
    for document in documents:
        blocks = parse_document(document, base_dir)
        formats.add(document.format.value)
        structure = build_structure(blocks)
        children = chunk_parents(structure.parents, size=20, overlap=4)
        fixed = chunk_fixed(blocks, size=24, overlap=4)
        assert children and fixed
        index = ExactCosineIndex(children, HashingEmbedder())
        hits = index.search("locator provenance evidence", top_k=2, document_id=document.document_id)
        pack = ContextBuilder(WhitespaceTokenCounter(), max_tokens=60).build(
            hits, threshold=-1
        )
        prompt = PromptBuilder().build_rag_prompt("What evidence is available?", pack.text)
        result = ExtractiveGenerator().generate(prompt)
        assert result.text
        assert pack.token_count <= 60
    assert formats == {"pdf", "pptx"}


class FastGenerator(ExtractiveGenerator):
    def __init__(self, **_kwargs):
        pass

    def close(self):
        pass


def test_experiment_runner_executes_all_baselines_with_test_doubles(
    tmp_path: Path, monkeypatch
):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(
        runner_module,
        "SentenceTransformerEmbedder",
        lambda **_kwargs: HashingEmbedder(),
    )
    monkeypatch.setattr(
        runner_module,
        "HuggingFaceTokenCounter",
        lambda *_args, **_kwargs: WhitespaceTokenCounter(),
    )
    monkeypatch.setattr(runner_module, "TransformersGenerator", FastGenerator)
    monkeypatch.setattr(
        runner_module,
        "build_reports",
        lambda *_args, **_kwargs: {"status": tmp_path / "reports-built"},
    )

    runner = runner_module.ExperimentRunner(root / "configs" / "pilot.yaml")
    runner.output_dir = tmp_path / "artifacts"
    runner.raw_dir = runner.output_dir / "raw"
    runner.raw_dir.mkdir(parents=True)
    reports = runner.run()
    records = JsonlResultStore(runner.raw_dir / "pilot_runs.jsonl").records

    assert reports["status"].name == "reports-built"
    assert len(records) == 100
    assert {record.baseline for record in records} == {"B0", "B1", "B2", "B3", "B4"}
    assert all(record.status == "completed" for record in records)
    assert any(record.route == "distinct_parents" for record in records)
    assert len(list((runner.raw_dir / "retrieval_snapshots").glob("*.json"))) == 100
    positions = {
        baseline: sorted(
            record.metadata["run_order_position"]
            for record in records
            if record.baseline == baseline
        )
        for baseline in ("B0", "B1", "B2", "B3", "B4")
    }
    expected_positions = sorted([1, 2, 3, 4, 5] * 4)
    assert all(values == expected_positions for values in positions.values())
    assert all(record.metadata["generator_warmup_calls_completed"] == 1 for record in records)
