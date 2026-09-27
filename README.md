# Retrieval Strategies for Educational RAG

Research code for a comparative undergraduate study of retrieval strategies in Retrieval-Augmented Generation (RAG) over educational PDF and PowerPoint documents.

The project compares fixed-size retrieval, structure-aware multi-granular retrieval, and routed retrieval policies while keeping ingestion, generation, and evaluation behavior explicit and reproducible.

## Research Scope

The pipeline is designed to study how retrieval design affects document-grounded answering across different question types and document structures. The implementation separates:

- PDF and PowerPoint ingestion
- document-structure extraction
- fixed-size and structure-aware chunking
- embedding-based indexing and retrieval
- question-aware routing policies
- context assembly and local generation
- retrieval and answer evaluation
- experiment configuration, provenance, and failure logging

This repository is deliberately separate from the earlier Smart Study Assistant application. It contains the comparative research pipeline only.

## Architecture

```text
documents -> ingestion -> structure model -> chunking strategies
          -> embeddings/index -> retrieval or routing -> context
          -> local generator -> evaluation -> local artifacts
```

The experiment runner evaluates configured retrieval baselines through a common interface so comparisons use the same corpus, query set, ranking depth, and reporting path.

## Repository Structure

```text
configs/        experiment configuration templates
scripts/        dataset validation and analysis utilities
src/
  ingestion/    PDF/PPT parsing
  structure/    hierarchical document representation
  chunking/     fixed-size and structure-aware chunkers
  retrieval/    embeddings and vector search
  routing/      question-aware retrieval policies
  generation/   context construction and local generation
  evaluation/   metrics, statistics, and reporting
  experiments/  CLI and experiment orchestration
  storage/      typed records and safe persistence
tests/          unit and integration tests
```

Research documents, annotations, downloaded corpora, vector indexes, model caches, raw generations, and evaluation artifacts are not included. They remain local to protect unpublished work and third-party material.

## Environment

- Python 3.11 or 3.12
- `uv` for locked dependency installation
- local embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- optional local generator: `Qwen/Qwen2.5-1.5B-Instruct`

The pipeline does not require a hosted LLM API. Model downloads are subject to their respective licenses and are cached locally.

## Setup

```bash
uv sync --frozen --extra dev
```

Run the test suite:

```bash
uv run pytest
```

Lint the source and tests:

```bash
uv run --with ruff ruff check src tests
```

## Running an Experiment

Configuration files under `configs/` describe retrieval baselines, model identifiers, ranking depth, and local input/output paths. Before running, provide your own appropriately licensed educational documents and annotations, then update a configuration to point to those local files.

```bash
uv run edu-rag validate --config configs/pilot.yaml
uv run edu-rag run --config configs/pilot.yaml --retrieval-only
uv run edu-rag evaluate --config configs/pilot.yaml --retrieval-only
```

End-to-end generation is available when the configured local generator is installed and accessible:

```bash
uv run edu-rag run --config configs/pilot.yaml
```

Generated records and metrics are written locally and are ignored by Git.

## Evaluation

The evaluation modules support locator-level retrieval metrics, routing diagnostics, failure-case reporting, and statistical comparison. This public repository intentionally makes no thesis-result claim: final datasets, raw runs, metric tables, and unpublished findings are excluded. Any result should be reported together with its exact configuration, corpus version, annotation protocol, model revision, and random seed.

## Reproducibility and Safety

- Dependency resolution is recorded in `uv.lock`.
- Configuration fingerprints and document hashes are supported by the experiment runner.
- Partial failures are recorded instead of silently discarded.
- Secrets, credentials, personal documents, proposals, supervisor material, private notes, datasets, model weights, and generated research artifacts are excluded by `.gitignore`.

## Limitations

- Reproduction requires a compatible corpus and annotation set that are not distributed here.
- Local model behavior and latency vary by hardware and model revision.
- Automated answer-quality measures do not replace independent human evaluation.
- The implementation is a research prototype, not a production knowledge system.

## Project Status

Active undergraduate research. Public code is provided for engineering transparency; unpublished study materials and findings remain private.
