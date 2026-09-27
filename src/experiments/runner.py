from __future__ import annotations

import random
import hashlib
import time
import traceback
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any
from collections import defaultdict

import numpy as np

from chunking import chunk_fixed, chunk_parents
from evaluation.reporting import build_reports
from generation import (
    ContextBuilder,
    HuggingFaceTokenCounter,
    PromptBuilder,
    TransformersGenerator,
)
from ingestion import parse_document
from retrieval import ExactCosineIndex, RoutedRetriever, SentenceTransformerEmbedder
from routing import OracleRouter, RuleRouter
from storage.io import (
    JsonlResultStore,
    atomic_write_json,
    configuration_hash,
    environment_snapshot,
    load_document_manifest,
    load_questions,
    load_yaml,
    sha256_file,
)
from storage.models import ContextPack, QuestionRecord, RetrievalHit, RunRecord
from structure import build_structure


class ExperimentRunner:
    def __init__(self, config_path: Path):
        self.config_path = config_path.resolve()
        self.root = self.config_path.parent.parent
        self.config = load_yaml(self.config_path)
        self.run_id = str(self.config["run_id"])
        self.seed = int(self.config["seed"])
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.output_dir = (self.root / self.config["output_dir"]).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.output_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = (self.root / self.config["manifest_path"]).resolve()
        manifest_metadata = load_yaml(manifest_path)
        self.dataset_status = str(manifest_metadata.get("status", ""))
        self.dataset_version = str(manifest_metadata.get("dataset_version", ""))
        self.documents = load_document_manifest(manifest_path)
        self.document_by_id = {document.document_id: document for document in self.documents}
        self.base_data_dir = manifest_path.parent.parent
        self.development_questions_path = (
            self.root / self.config["development_questions_path"]
        ).resolve()
        self.evaluation_questions_path = (
            self.root / self.config["evaluation_questions_path"]
        ).resolve()
        self.development_questions = load_questions(self.development_questions_path)
        self.evaluation_questions = load_questions(self.evaluation_questions_path)
        self.all_evaluation_questions = list(self.evaluation_questions)
        experiment = self.config.get("experiment", {})
        self.experiment_role = str(experiment.get("role", "pilot"))
        if self.experiment_role not in {"pilot", "primary", "repeatability"}:
            raise ValueError("experiment.role must be pilot, primary, or repeatability")
        self.repeat_indices = [int(value) for value in experiment.get("repeat_indices", [1])]
        if not self.repeat_indices or len(set(self.repeat_indices)) != len(self.repeat_indices):
            raise ValueError("repeat_indices must be nonempty and unique")
        if self.experiment_role == "repeatability":
            if min(self.repeat_indices) < 2:
                raise ValueError("Repeatability records must use repeat_index >= 2")
            self.evaluation_questions = self.select_timing_questions(
                self.all_evaluation_questions,
                int(experiment.get("subset_size", 32)),
                self.seed,
            )
        elif self.repeat_indices != [1]:
            raise ValueError("Primary/pilot quality reports use only repeat_index 1")
        self._validate_question_documents()
        code_files = [
            self.root / "pyproject.toml",
            self.root / "uv.lock",
            *sorted((self.root / "src").rglob("*.py")),
        ]
        code_fingerprint = configuration_hash(
            {
                str(path.relative_to(self.root)): sha256_file(path)
                for path in code_files
            }
        )
        self.input_hashes = {
            "config": sha256_file(self.config_path),
            "manifest": sha256_file(manifest_path),
            "development_questions": sha256_file(self.development_questions_path),
            "evaluation_questions": sha256_file(self.evaluation_questions_path),
            "research_code": code_fingerprint,
        }
        self.config_hash = configuration_hash(
            {
                "config": self.config,
                "input_hashes": self.input_hashes,
            }
        )

        self.blocks_by_document: dict[str, list] = {}
        self.parent_by_id: dict = {}
        self.fixed_index: ExactCosineIndex | None = None
        self.child_index: ExactCosineIndex | None = None
        self.routed_retriever: RoutedRetriever | None = None
        self.query_vectors: dict[str, np.ndarray] = {}
        self.query_embedding_ms: dict[str, float] = {}
        default_threshold = float(self.config["retrieval"]["default_threshold"])
        self.thresholds: dict[str, float] = {
            "fixed": default_threshold,
            "child_family": default_threshold,
        }
        self._seeded_baseline_order = list(self.config["baselines"])
        random.Random(self.seed).shuffle(self._seeded_baseline_order)

    def _validate_question_documents(self) -> None:
        known = set(self.document_by_id)
        for question in self.development_questions + self.evaluation_questions:
            if question.document_id not in known:
                raise ValueError(
                    f"Question {question.question_id} references unknown document "
                    f"{question.document_id}"
                )

    @staticmethod
    def select_timing_questions(
        questions: list[QuestionRecord], size: int, seed: int,
    ) -> list[QuestionRecord]:
        """Proportional stratification independent of outputs and retrieval scores."""
        if size < 1 or size > len(questions):
            raise ValueError("Timing subset size must be between 1 and dataset size")
        strata: dict[str, list[QuestionRecord]] = defaultdict(list)
        for question in questions:
            label = question.question_type.value if question.answerable else "unanswerable"
            key = f"{question.domain}|{question.document_format}|{label}"
            strata[key].append(question)

        def stable_order(value: str) -> str:
            return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()

        allocations = {
            key: size * len(items) // len(questions)
            for key, items in strata.items()
        }
        remaining = size - sum(allocations.values())
        remainder_order = sorted(
            strata,
            key=lambda key: (
                -(size * len(strata[key]) % len(questions)), stable_order(key),
            ),
        )
        for key in remainder_order[:remaining]:
            allocations[key] += 1
        selected = {
            question.question_id
            for key, items in strata.items()
            for question in sorted(items, key=lambda q: stable_order(q.question_id))[
                :allocations[key]
            ]
        }
        return [question for question in questions if question.question_id in selected]

    def _report_metadata(self, retrieval_only: bool) -> tuple[str, str]:
        if not retrieval_only:
            if self.experiment_role == "primary":
                return (
                    "Full Experiment — Frozen Final Dataset",
                    "Hasil eksperimen lokal pada dataset final yang dibekukan. "
                    "Hanya repeat_index=1 digunakan dalam perbandingan kualitas; "
                    "ukuran otomatis tidak menggantikan penilaian manusia.",
                )
            return (
                "Pilot Metrics",
                "Pilot adalah feasibility study dan tidak menggantikan hasil "
                "penelitian final.",
            )
        if self.dataset_status == "author_accepted_final":
            version = f" v{self.dataset_version}" if self.dataset_version else ""
            return (
                f"Final Dataset{version} — Retrieval Dry-Run",
                "Dry-run ini menggunakan dataset final dan tidak menjalankan "
                "generator. Metrik kualitas jawaban, sitasi, dan refusal "
                "karenanya dilaporkan sebagai N/A; hasil ini hanya menjadi "
                "bukti teknis retrieval, bukan eksperimen lengkap.",
            )
        return (
            "Final-Dataset Draft Retrieval Dry-Run",
            "Dry-run ini hanya memeriksa kelayakan teknis retrieval. Dataset "
            "belum dibekukan dan angka ini bukan hasil penelitian final.",
        )

    def prepare(self) -> None:
        start = time.perf_counter()
        all_parents = []
        all_children = []
        all_fixed = []
        fixed_size = int(self.config["chunking"]["fixed_tokens"])
        fixed_overlap = int(self.config["chunking"]["fixed_overlap"])
        child_size = int(self.config["chunking"]["child_tokens"])
        child_overlap = int(self.config["chunking"]["child_overlap"])

        extraction_report: dict[str, Any] = {}
        for document in self.documents:
            blocks = parse_document(document, self.base_data_dir)
            self.blocks_by_document[document.document_id] = blocks
            structure = build_structure(blocks)
            children = chunk_parents(structure.parents, child_size, child_overlap)
            fixed = chunk_fixed(blocks, fixed_size, fixed_overlap)
            all_parents.extend(structure.parents)
            all_children.extend(children)
            all_fixed.extend(fixed)
            extraction_report[document.document_id] = {
                "blocks": len(blocks),
                "parents": len(structure.parents),
                "children": len(children),
                "fixed_chunks": len(fixed),
                "locators": sorted({block.locator for block in blocks}),
                "sha256": document.sha256,
                "license": document.license,
            }
        self.parent_by_id = {unit.unit_id: unit for unit in all_parents}

        embedding = self.config["embedding"]
        embedder = SentenceTransformerEmbedder(
            model_id=embedding["model_id"],
            revision=embedding["revision"],
            device=embedding.get("device", "cpu"),
            batch_size=int(embedding.get("batch_size", 32)),
            local_files_only=bool(embedding.get("local_files_only", True)),
        )
        fixed_index_start = time.perf_counter()
        self.fixed_index = ExactCosineIndex(all_fixed, embedder)
        fixed_index_ms = (time.perf_counter() - fixed_index_start) * 1000
        child_index_start = time.perf_counter()
        self.child_index = ExactCosineIndex(all_children, embedder)
        child_index_ms = (time.perf_counter() - child_index_start) * 1000
        self.routed_retriever = RoutedRetriever(self.child_index, self.parent_by_id)
        query_embedding_start = time.perf_counter()
        for question in self.evaluation_questions:
            item_start = time.perf_counter()
            self.query_vectors[question.question_id] = embedder.encode(
                [question.question]
            )[0]
            self.query_embedding_ms[question.question_id] = (
                time.perf_counter() - item_start
            ) * 1000
        query_embedding_total_ms = (time.perf_counter() - query_embedding_start) * 1000

        indexing_report = {
            "run_id": self.run_id,
            "configuration_hash": self.config_hash,
            "documents": extraction_report,
            "total_parents": len(all_parents),
            "total_children": len(all_children),
            "total_fixed_chunks": len(all_fixed),
            "fixed_index_ms": fixed_index_ms,
            "child_index_ms": child_index_ms,
            "query_embedding_total_ms": query_embedding_total_ms,
            "query_embedding_policy": (
                "Each evaluation query is embedded once, then the same frozen vector and "
                "measured embedding latency are reused for B1-B4."
            ),
            "prepare_total_ms": (time.perf_counter() - start) * 1000,
            "embedding_model": embedding,
            "input_hashes": self.input_hashes,
        }
        atomic_write_json(self.output_dir / "indexing.json", indexing_report)
        environment = environment_snapshot()
        environment.update(
            {
                "run_id": self.run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "configuration_hash": self.config_hash,
                "input_hashes": self.input_hashes,
            }
        )
        atomic_write_json(self.output_dir / "environment.json", environment)
        self.thresholds = self._calibrate_thresholds()

    @staticmethod
    def _select_balanced_threshold(
        observations: list[tuple[float, bool, str]],
        default_threshold: float,
    ) -> dict[str, object]:
        if not observations:
            return {
                "selected_threshold": default_threshold,
                "balanced_accuracy": None,
                "sensitivity": None,
                "specificity": None,
                "development_observations": [],
            }
        scores = sorted({score for score, *_ in observations})
        candidates = [scores[0] - 1e-6]
        candidates.extend((left + right) / 2 for left, right in pairwise(scores))
        candidates.append(scores[-1] + 1e-6)
        best = None
        for threshold in candidates:
            positives = [obs for obs in observations if obs[1]]
            negatives = [obs for obs in observations if not obs[1]]
            tpr = (
                sum(score >= threshold for score, *_ in positives) / len(positives)
                if positives
                else 1.0
            )
            tnr = (
                sum(score < threshold for score, *_ in negatives) / len(negatives)
                if negatives
                else 1.0
            )
            balanced = (tpr + tnr) / 2
            candidate = (balanced, threshold, tpr, tnr)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        assert best is not None
        return {
            "selected_threshold": best[1],
            "balanced_accuracy": best[0],
            "sensitivity": best[2],
            "specificity": best[3],
            "development_observations": [
                {
                    "score": score,
                    "answerable": answerable,
                    "question_id": question_id,
                }
                for score, answerable, question_id in observations
            ],
        }

    def _calibrate_thresholds(self) -> dict[str, float]:
        assert self.fixed_index is not None and self.child_index is not None
        candidate_k = int(self.config["retrieval"]["candidate_k"])
        observations: dict[str, list[tuple[float, bool, str]]] = {
            "fixed": [],
            "child_family": [],
        }
        for question in self.development_questions:
            for family, index in [
                ("fixed", self.fixed_index),
                ("child_family", self.child_index),
            ]:
                hits = index.search(
                    question.question, candidate_k, document_id=question.document_id
                )
                score = hits[0].score if hits else -1.0
                observations[family].append(
                    (score, question.answerable, question.question_id)
                )
        default_threshold = float(self.config["retrieval"]["default_threshold"])
        family_reports = {
            family: self._select_balanced_threshold(values, default_threshold)
            for family, values in observations.items()
        }
        report = {
            "strategy": self.config["retrieval"]["threshold_strategy"],
            "families": family_reports,
            "baseline_mapping": {
                "B1": "fixed",
                "B2": "child_family",
                "B3": "child_family",
                "B4": "child_family",
            },
            "note": (
                "Thresholds are selected only from development questions and "
                "calibrated separately for fixed and child-derived score families."
            ),
        }
        atomic_write_json(self.output_dir / "threshold_calibration.json", report)
        return {
            family: float(values["selected_threshold"])
            for family, values in family_reports.items()
        }

    def _threshold_for_baseline(self, baseline: str) -> float:
        family = "fixed" if baseline == "B1" else "child_family"
        return self.thresholds[family]


    def _validate_execution_mode(self, retrieval_only: bool) -> None:
        """Fail early when a run mode would mix pilot/dry-run and final outputs."""
        if self.experiment_role in {"primary", "repeatability"} and retrieval_only:
            raise ValueError(
                f"experiment.role={self.experiment_role!r} must run end-to-end; "
                "use a separate pilot/dry-run config for --retrieval-only."
            )
        if self.experiment_role in {"primary", "repeatability"}:
            if self.dataset_status != "author_accepted_final":
                raise ValueError(
                    "Primary/repeatability experiment requires a frozen final dataset "
                    f"(status='author_accepted_final'), got {self.dataset_status!r}."
                )
        if self.experiment_role == "primary" and self.repeat_indices != [1]:
            raise ValueError("Primary experiment must use repeat_indices: [1].")
        if self.experiment_role == "repeatability" and min(self.repeat_indices) < 2:
            raise ValueError("Repeatability experiment must use repeat_index >= 2.")

    def _result_path(self, retrieval_only: bool) -> Path:
        """Use separate raw files so pilot, primary, and repeatability never overwrite."""
        if retrieval_only:
            filename = "pilot_retrieval_runs.jsonl"
        elif self.experiment_role == "primary":
            filename = "primary_runs.jsonl"
        elif self.experiment_role == "repeatability":
            filename = "repeatability_runs.jsonl"
        else:
            filename = "pilot_runs.jsonl"
        return self.raw_dir / filename

    def _write_startup_error(self, exc: Exception, stage: str) -> None:
        atomic_write_json(
            self.output_dir / "startup_error.json",
            {
                "run_id": self.run_id,
                "configuration_hash": self.config_hash,
                "role": self.experiment_role,
                "stage": stage,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(limit=12),
                "created_at": datetime.now(UTC).isoformat(),
            },
        )

    def _current_records(
        self,
        records: list[RunRecord],
        retrieval_only: bool,
    ) -> list[RunRecord]:
        """Return only records belonging to the exact current execution plan."""
        allowed_questions = {q.question_id for q in self.evaluation_questions}
        allowed_baselines = set(self.config["baselines"])
        allowed_repeats = set(self.repeat_indices)
        current = [
            record
            for record in records
            if record.run_id == self.run_id
            and record.configuration_hash == self.config_hash
            and record.question_id in allowed_questions
            and record.baseline in allowed_baselines
            and record.repeat_index in allowed_repeats
            and bool(record.metadata.get("retrieval_only", False)) == retrieval_only
        ]
        current.sort(
            key=lambda r: (
                r.repeat_index,
                r.question_id,
                self.config["baselines"].index(r.baseline),
            )
        )
        return current

    def _write_coverage(
        self,
        records: list[RunRecord],
        retrieval_only: bool,
    ) -> None:
        current = self._current_records(records, retrieval_only)
        expected_keys = {
            (
                self.run_id,
                self.config_hash,
                question.question_id,
                baseline,
                repeat_index,
            )
            for repeat_index in self.repeat_indices
            for question in self.evaluation_questions
            for baseline in self.config["baselines"]
        }
        observed_keys = {record.key for record in current}
        completed = sum(record.status == "completed" for record in current)
        errors = sum(record.status == "error" for record in current)

        generation_invoked = sum(
            bool(record.metadata.get("generation_invoked", False))
            for record in current
        )
        pre_generation_refusals = sum(
            record.context.refused_before_generation
            for record in current
            if record.baseline != "B0"
        )

        atomic_write_json(
            self.output_dir / "coverage.json",
            {
                "run_id": self.run_id,
                "configuration_hash": self.config_hash,
                "role": self.experiment_role,
                "retrieval_only": retrieval_only,
                "expected_records": len(expected_keys),
                "observed_records": len(current),
                "unique_observed_keys": len(observed_keys),
                "completed_records": completed,
                "error_records": errors,
                "missing_records": len(expected_keys - observed_keys),
                "unexpected_records": len(observed_keys - expected_keys),
                "generation_invoked_records": generation_invoked,
                "pre_generation_refusal_records": pre_generation_refusals,
                "complete": observed_keys == expected_keys and errors == 0,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )

    def _write_repeatability_report(
        self,
        records: list[RunRecord],
    ) -> dict[str, Path]:
        """Write a conservative repeatability report for the frozen timing subset."""
        report_path = self.output_dir / "repeatability_report.json"
        completed = [record for record in records if record.status == "completed"]
        by_baseline: dict[str, dict[str, float | int | None]] = {}
        for baseline in self.config["baselines"]:
            rows = [record for record in completed if record.baseline == baseline]
            generation_rows = [
                record for record in rows
                if bool(record.metadata.get("generation_invoked", False))
            ]
            by_baseline[baseline] = {
                "records": len(rows),
                "generation_invoked": len(generation_rows),
                "retrieval_latency_ms_median": (
                    float(np.median([r.retrieval_latency_ms for r in rows]))
                    if rows else None
                ),
                "generation_latency_ms_median": (
                    float(np.median([r.generation_latency_ms for r in generation_rows]))
                    if generation_rows else None
                ),
                "end_to_end_latency_ms_median": (
                    float(np.median([r.end_to_end_latency_ms for r in rows]))
                    if rows else None
                ),
            }

        atomic_write_json(
            report_path,
            {
                "run_id": self.run_id,
                "configuration_hash": self.config_hash,
                "role": self.experiment_role,
                "repeat_indices": self.repeat_indices,
                "question_count": len(self.evaluation_questions),
                "completed_records": len(completed),
                "error_records": sum(record.status == "error" for record in records),
                "by_baseline": by_baseline,
                "note": (
                    "Repeatability records are kept separate from primary quality "
                    "statistics. This report summarizes execution/timing only."
                ),
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        return {"repeatability_report": report_path}


    def run(
        self,
        retrieval_only: bool = False,
        retry_errors: bool = False,
    ) -> dict[str, Path]:
        self._validate_execution_mode(retrieval_only)
        atomic_write_json(
            self.output_dir / "execution_plan.json",
            {
                "run_id": self.run_id,
                "configuration_hash": self.config_hash,
                "role": self.experiment_role,
                "retrieval_only": retrieval_only,
                "repeat_indices": self.repeat_indices,
                "question_ids": [q.question_id for q in self.evaluation_questions],
                "baselines": self.config["baselines"],
                "expected_records": len(self.evaluation_questions)
                * len(self.config["baselines"]) * len(self.repeat_indices),
                "subset_selection": (
                    "Proportional largest-remainder allocation by domain, format, "
                    "and answerable type/unanswerable; SHA-256 seeded ordering; "
                    "independent of model outputs."
                    if self.experiment_role == "repeatability" else "all evaluation questions"
                ),
            },
        )
        if self.fixed_index is None or self.child_index is None or self.routed_retriever is None:
            try:
                self.prepare()
            except Exception as exc:
                self._write_startup_error(exc, "prepare")
                raise
        generator_config = self.config["generator"]
        token_counter = HuggingFaceTokenCounter(
            generator_config["model_id"],
            generator_config["revision"],
            local_files_only=bool(generator_config.get("local_files_only", True)),
        )
        context_builder = ContextBuilder(
            token_counter,
            max_tokens=int(self.config["context"]["max_tokens"]),
            locator_prefix=bool(self.config["context"].get("locator_prefix", True)),
        )
        prompt_builder = PromptBuilder()
        generator = None
        warmup_calls = int(self.config.get("timing", {}).get("warmup_calls", 1))
        warmup_completed = 0
        if not retrieval_only:
            try:
                generator = TransformersGenerator(
                    model_id=generator_config["model_id"],
                    revision=generator_config["revision"],
                    device=generator_config.get("device", "auto"),
                    local_files_only=bool(generator_config.get("local_files_only", True)),
                    max_new_tokens=int(generator_config.get("max_new_tokens", 96)),
                    do_sample=bool(generator_config.get("do_sample", False)),
                    seed=self.seed,
                )
                self.generator_runtime = (
                    generator.runtime_metadata()
                    if hasattr(generator, "runtime_metadata") else {"test_double": True}
                )
                atomic_write_json(self.output_dir / "generator_runtime.json", self.generator_runtime)
                warmup_prompt = prompt_builder.build_no_rag_prompt(
                    "Warm-up call for deterministic runtime initialization."
                )
                for _ in range(warmup_calls):
                    generator.generate(warmup_prompt)
                    warmup_completed += 1
            except Exception as exc:
                self._write_startup_error(exc, "generator_initialization_or_warmup")
                if generator is not None:
                    generator.close()
                raise

        atomic_write_json(
            self.output_dir / "timing_protocol.json",
            {
                "run_id": self.run_id,
                "baseline_order_policy": "seeded cyclic counterbalancing per question",
                "seeded_base_order": self._seeded_baseline_order,
                "warmup_calls_requested": warmup_calls if not retrieval_only else 0,
                "warmup_calls_completed": warmup_completed,
                "warmup_excluded_from_metrics": True,
                "timed_repeats_per_question_baseline": len(self.repeat_indices),
                "repeat_indices": self.repeat_indices,
                "role": self.experiment_role,
                "quality_statistics_use_primary_repeat_only": True,
                "generation_invoked_recorded_per_run": True,
                "note": (
                    "Latency for threshold refusal without a model call is retained "
                    "separately from latency for records that invoke the generator."
                ),
            },
        )

        result_path = self._result_path(retrieval_only)
        store = JsonlResultStore(result_path)
        processed = 0
        expected = len(self.evaluation_questions) * len(self.config["baselines"]) * len(self.repeat_indices)
        progress_every = max(1, int(self.config.get("experiment", {}).get("progress_every", 20)))
        try:
            for repeat_index in self.repeat_indices:
                for question_position, question in enumerate(self.evaluation_questions):
                    baseline_order = self._baseline_order(question_position + repeat_index - 1)
                    for run_order_position, baseline in enumerate(baseline_order, start=1):
                        key = (
                            self.run_id, self.config_hash, question.question_id,
                            baseline, repeat_index,
                        )
                        processed += 1
                        if store.contains(key, retry_errors=retry_errors):
                            continue
                        record = self._run_one(
                            question=question,
                            baseline=baseline,
                            context_builder=context_builder,
                            prompt_builder=prompt_builder,
                            generator=generator,
                            retrieval_only=retrieval_only,
                            run_order_position=run_order_position,
                            baseline_order=baseline_order,
                            warmup_completed=warmup_completed,
                            repeat_index=repeat_index,
                        )
                        store.append(record)
                        if processed % progress_every == 0 or record.status == "error":
                            print(
                                f"[{self.run_id}] {processed}/{expected} "
                                f"repeat={repeat_index} last={record.status}", flush=True,
                            )
        finally:
            if generator is not None:
                generator.close()
            self._write_coverage(store.records, retrieval_only)
        current_records = self._current_records(store.records, retrieval_only)
        if self.experiment_role == "repeatability":
            return self._write_repeatability_report(current_records)
        report_title, report_note = self._report_metadata(retrieval_only)
        return build_reports(
            current_records,
            questions_path=self.evaluation_questions_path,
            output_dir=self.output_dir,
            top_k=int(self.config["retrieval"]["top_k"]),
            seed=self.seed,
            report_title=report_title,
            records_source=str(result_path.relative_to(self.output_dir)),
            report_note=report_note,
        )

    def _baseline_order(self, question_position: int) -> list[str]:
        order = self._seeded_baseline_order
        offset = question_position % len(order)
        return [*order[offset:], *order[:offset]]

    def _run_one(
        self,
        question: QuestionRecord,
        baseline: str,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        generator: TransformersGenerator | None,
        retrieval_only: bool,
        run_order_position: int,
        baseline_order: list[str],
        warmup_completed: int,
        repeat_index: int,
    ) -> RunRecord:
        start = time.perf_counter()
        retrieval_start = time.perf_counter()
        query_embedding_latency = (
            0.0
            if baseline == "B0"
            else self.query_embedding_ms.get(question.question_id, 0.0)
        )
        query_vector = self.query_vectors.get(question.question_id)
        predicted_type = None
        route = None
        hits: list[RetrievalHit] = []
        try:
            top_k = int(self.config["retrieval"]["top_k"])
            candidate_k = int(self.config["retrieval"]["candidate_k"])
            if baseline == "B0":
                context = ContextPack(text="", token_count=0)
            else:
                if baseline == "B1":
                    assert self.fixed_index is not None
                    assert query_vector is not None
                    route = "fixed"
                    hits = self.fixed_index.search_vector(
                        query_vector, top_k, document_id=question.document_id
                    )
                elif baseline == "B2":
                    assert self.child_index is not None
                    assert query_vector is not None
                    route = "child"
                    hits = self.child_index.search_vector(
                        query_vector, top_k, document_id=question.document_id
                    )
                elif baseline == "B3":
                    assert query_vector is not None
                    decision = RuleRouter().predict(question.question)
                    predicted_type = decision.question_type
                    route = decision.route
                    assert self.routed_retriever is not None
                    hits = self.routed_retriever.retrieve_vector(
                        query_vector,
                        question.document_id,
                        route,
                        top_k,
                        candidate_k,
                    )
                elif baseline == "B4":
                    assert query_vector is not None
                    decision = OracleRouter().predict(question)
                    predicted_type = decision.question_type
                    route = decision.route
                    assert self.routed_retriever is not None
                    hits = self.routed_retriever.retrieve_vector(
                        query_vector,
                        question.document_id,
                        route,
                        top_k,
                        candidate_k,
                    )
                else:
                    raise ValueError(f"Unsupported baseline: {baseline}")
                threshold = self._threshold_for_baseline(baseline)
                context = context_builder.build(hits, threshold, apply_threshold=True)
            retrieval_latency = (
                (time.perf_counter() - retrieval_start) * 1000
                + query_embedding_latency
            )
            self._write_retrieval_snapshot(
                question, baseline, route, hits, context, repeat_index
            )

            if baseline == "B0":
                prompt = prompt_builder.build_no_rag_prompt(question.question)
            else:
                prompt = prompt_builder.build_rag_prompt(question.question, context.text)

            generation_latency = 0.0
            generation_invoked = False
            serialized_prompt = ""
            serialized_prompt_sha256 = ""
            reached_token_limit = False
            if baseline != "B0" and context.refused_before_generation:
                output = PromptBuilder.REFUSAL_TOKEN
                prompt_tokens = context_builder.token_counter.count(prompt)
                output_tokens = context_builder.token_counter.count(output)
            elif retrieval_only:
                output = "RETRIEVAL_ONLY"
                prompt_tokens = context_builder.token_counter.count(prompt)
                output_tokens = context_builder.token_counter.count(output)
            else:
                if generator is None:
                    raise RuntimeError("Generator is required for an end-to-end run")
                generation_invoked = True
                generation = generator.generate(prompt)
                output = generation.text
                generation_latency = generation.latency_ms
                prompt_tokens = generation.prompt_tokens
                output_tokens = generation.output_tokens
                serialized_prompt = generation.serialized_prompt
                serialized_prompt_sha256 = generation.serialized_prompt_sha256
                reached_token_limit = generation.reached_token_limit

            from evaluation.metrics import extract_citations

            document = self.document_by_id[question.document_id]
            return RunRecord(
                run_id=self.run_id,
                configuration_hash=self.config_hash,
                question_id=question.question_id,
                document_id=question.document_id,
                baseline=baseline,
                repeat_index=repeat_index,
                status="completed",
                seed=self.seed,
                question=question.question,
                question_type_gold=question.question_type,
                question_type_predicted=predicted_type,
                route=route,
                answerable=question.answerable,
                gold_locators=question.gold_evidence.locators,
                ranking=hits,
                threshold=(
                    None
                    if baseline == "B0"
                    else self._threshold_for_baseline(baseline)
                ),
                context=context,
                prompt=prompt,
                output=output,
                citations=extract_citations(output),
                retrieval_latency_ms=retrieval_latency,
                generation_latency_ms=generation_latency,
                end_to_end_latency_ms=(
                    (time.perf_counter() - start) * 1000 + query_embedding_latency
                ),
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                embedding_model=self.config["embedding"]["model_id"],
                generator_model=(
                    None if retrieval_only else self.config["generator"]["model_id"]
                ),
                model_revisions={
                    "embedding": self.config["embedding"]["revision"],
                    "generator": self.config["generator"]["revision"],
                },
                document_sha256=document.sha256 or "",
                metadata={
                    "retrieval_only": retrieval_only,
                    "query_embedding_ms": query_embedding_latency,
                    "query_vector_reused_across_retrieval_baselines": baseline != "B0",
                    "run_order_position": run_order_position,
                    "baseline_order": baseline_order,
                    "baseline_order_policy": "seeded_cyclic_counterbalancing",
                    "generator_warmup_calls_completed": warmup_completed,
                    "warmup_excluded_from_metrics": True,
                    "generation_invoked": generation_invoked,
                    "serialized_prompt": serialized_prompt,
                    "serialized_prompt_sha256": serialized_prompt_sha256,
                    "reached_token_limit": reached_token_limit,
                    "timing_repeat": repeat_index,
                },
            )
        # Every per-question failure must become a resumable error record. Restricting
        # this boundary would allow an unanticipated provider/parser error to abort
        # the entire experiment and discard the remaining paired observations.
        except Exception as exc:  # noqa: BLE001
            document = self.document_by_id[question.document_id]
            return RunRecord(
                run_id=self.run_id,
                configuration_hash=self.config_hash,
                question_id=question.question_id,
                document_id=question.document_id,
                baseline=baseline,
                repeat_index=repeat_index,
                status="error",
                seed=self.seed,
                question=question.question,
                question_type_gold=question.question_type,
                question_type_predicted=predicted_type,
                route=route,
                answerable=question.answerable,
                gold_locators=question.gold_evidence.locators,
                ranking=hits,
                threshold=(
                    None
                    if baseline == "B0"
                    else self._threshold_for_baseline(baseline)
                ),
                context=ContextPack(text="", token_count=0),
                prompt="",
                output="",
                retrieval_latency_ms=max(
                    0.0,
                    (time.perf_counter() - retrieval_start) * 1000
                    + query_embedding_latency,
                ),
                generation_latency_ms=0,
                end_to_end_latency_ms=(
                    (time.perf_counter() - start) * 1000 + query_embedding_latency
                ),
                prompt_tokens=0,
                output_tokens=0,
                embedding_model=self.config["embedding"]["model_id"],
                generator_model=(
                    None if retrieval_only else self.config["generator"]["model_id"]
                ),
                model_revisions={
                    "embedding": self.config["embedding"]["revision"],
                    "generator": self.config["generator"]["revision"],
                },
                document_sha256=document.sha256 or "",
                error_type=type(exc).__name__,
                error_message=str(exc),
                metadata={
                    "traceback": traceback.format_exc(limit=8),
                    "run_order_position": run_order_position,
                    "baseline_order": baseline_order,
                    "generator_warmup_calls_completed": warmup_completed,
                    "generation_invoked": False,
                    "timing_repeat": repeat_index,
                },
            )

    def _write_retrieval_snapshot(
        self,
        question: QuestionRecord,
        baseline: str,
        route: str | None,
        hits: list[RetrievalHit],
        context: ContextPack,
        repeat_index: int,
    ) -> None:
        directory = self.raw_dir / "retrieval_snapshots"
        payload = {
            "run_id": self.run_id,
            "question_id": question.question_id,
            "baseline": baseline,
            "repeat_index": repeat_index,
            "route": route,
            "threshold": (
                None
                if baseline == "B0"
                else self._threshold_for_baseline(baseline)
            ),
            "ranking": [hit.model_dump(mode="json") for hit in hits],
            "context": context.model_dump(mode="json"),
            "saved_before_generation": True,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        atomic_write_json(
            directory / f"{question.question_id}_{baseline}_r{repeat_index}.json",
            payload,
        )

    def evaluate_existing(
        self,
        retrieval_only: bool = False,
    ) -> dict[str, Path]:
        result_path = self._result_path(retrieval_only)
        store = JsonlResultStore(result_path)
        current_records = self._current_records(store.records, retrieval_only)
        if self.experiment_role == "repeatability":
            return self._write_repeatability_report(current_records)
        report_title, report_note = self._report_metadata(retrieval_only)
        return build_reports(
            current_records,
            questions_path=self.evaluation_questions_path,
            output_dir=self.output_dir,
            top_k=int(self.config["retrieval"]["top_k"]),
            seed=self.seed,
            report_title=report_title,
            records_source=str(result_path.relative_to(self.output_dir)),
            report_note=report_note,
        )
