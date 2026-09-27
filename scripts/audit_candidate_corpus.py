from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from pptx import Presentation
from pypdf import PdfReader

from ingestion.parsers import parse_document
from storage.io import load_document_manifest
from storage.models import DocumentFormat


def _total_locators(document_path: Path, document_format: DocumentFormat) -> int:
    if document_format == DocumentFormat.PDF:
        return len(PdfReader(str(document_path)).pages)
    return len(Presentation(str(document_path)).slides)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def audit_corpus(manifest_path: Path, output_dir: Path) -> dict:
    documents = load_document_manifest(manifest_path)
    base_dir = manifest_path.parent.parent
    seen_hashes: Counter[str] = Counter(document.sha256 or "" for document in documents)
    summaries: list[dict] = []
    locator_records: list[dict] = []

    for document in documents:
        document_path = document.resolved_path(base_dir)
        blocks = parse_document(document, base_dir)
        blocks_by_locator: dict[str, list] = defaultdict(list)
        for block in blocks:
            blocks_by_locator[block.locator].append(block)

        total_locators = _total_locators(document_path, document.format)
        word_counts: list[int] = []
        for locator, locator_blocks in sorted(
            blocks_by_locator.items(),
            key=lambda item: min(block.order for block in item[1]),
        ):
            ordered_blocks = sorted(locator_blocks, key=lambda block: block.order)
            text = "\n".join(block.text for block in ordered_blocks)
            word_count = len(text.split())
            word_counts.append(word_count)
            locator_records.append(
                {
                    "document_id": document.document_id,
                    "domain": document.domain,
                    "format": document.format.value,
                    "title": document.title,
                    "locator": locator,
                    "parent_unit_id": f"{document.document_id}:parent:{locator}",
                    "word_count": word_count,
                    "block_ids": [block.block_id for block in ordered_blocks],
                    "text": text,
                }
            )

        extractable_locators = len(word_counts)
        empty_locators = total_locators - extractable_locators
        low_text_locators = sum(count < 12 for count in word_counts)
        overlong_locators = sum(count > 700 for count in word_counts)
        duplicate_hash = seen_hashes[document.sha256 or ""] > 1
        extractable_ratio = extractable_locators / total_locators if total_locators else 0.0
        warnings: list[str] = []
        if extractable_ratio < 0.8:
            warnings.append("More than 20% of page/slide locators have no extractable text.")
        if word_counts and low_text_locators / len(word_counts) > 0.35:
            warnings.append("More than 35% of extractable locators contain fewer than 12 words.")
        if overlong_locators:
            warnings.append(
                f"{overlong_locators} parent locator(s) exceed 700 whitespace tokens and require "
                "boundary-preserving splitting during indexing."
            )
        if duplicate_hash:
            warnings.append("Document bytes duplicate another manifest entry.")
        if not document.source_url:
            warnings.append("Primary source URL is missing.")

        passed = (
            total_locators > 0
            and extractable_ratio >= 0.8
            and sum(word_counts) >= 500
            and not duplicate_hash
            and bool(document.source_url)
        )
        summaries.append(
            {
                "document_id": document.document_id,
                "title": document.title,
                "domain": document.domain,
                "format": document.format.value,
                "license": document.license,
                "source_url": document.source_url,
                "sha256": document.sha256,
                "file_size_bytes": document_path.stat().st_size,
                "total_locators": total_locators,
                "extractable_locators": extractable_locators,
                "empty_locators": empty_locators,
                "extractable_ratio": round(extractable_ratio, 4),
                "block_count": len(blocks),
                "word_count": sum(word_counts),
                "median_words_per_extractable_locator": (
                    round(statistics.median(word_counts), 2) if word_counts else 0
                ),
                "low_text_locators_under_12_words": low_text_locators,
                "overlong_locators_over_700_words": overlong_locators,
                "automatic_acceptance_check": "pass" if passed else "review",
                "warnings": warnings,
            }
        )

    totals = {
        "documents": len(summaries),
        "pdf_documents": sum(item["format"] == "pdf" for item in summaries),
        "pptx_documents": sum(item["format"] == "pptx" for item in summaries),
        "artificial_intelligence_documents": sum(
            item["domain"] == "artificial_intelligence" for item in summaries
        ),
        "data_structures_algorithms_documents": sum(
            item["domain"] == "data_structures_algorithms" for item in summaries
        ),
        "locators": sum(item["total_locators"] for item in summaries),
        "extractable_locators": sum(item["extractable_locators"] for item in summaries),
        "words": sum(item["word_count"] for item in summaries),
        "automatic_pass": sum(
            item["automatic_acceptance_check"] == "pass" for item in summaries
        ),
        "requires_review": sum(
            item["automatic_acceptance_check"] == "review" for item in summaries
        ),
    }
    report = {
        "status": "candidate_corpus_pending_human_verification",
        "manifest": str(manifest_path),
        "acceptance_rule": {
            "minimum_extractable_locator_ratio": 0.8,
            "minimum_word_count": 500,
            "requires_unique_sha256": True,
            "requires_source_url": True,
            "note": (
                "Automatic checks only establish technical suitability. The two student "
                "researchers must still verify licence evidence, semantic quality, and visual risk."
            ),
        },
        "totals": totals,
        "documents": summaries,
    }
    _write_json(output_dir / "corpus_audit.json", report)
    _write_jsonl(output_dir / "locator_text.jsonl", locator_records)

    rows = [
        "# Audit Teknis Kandidat Korpus Final",
        "",
        "Status: **belum dibekukan; menunggu verifikasi manusia oleh kedua peneliti**.",
        "",
        "| Document ID | Format | Domain | Locator terambil/total | Kata | Status otomatis |",
        "|---|---:|---|---:|---:|---|",
    ]
    for item in summaries:
        rows.append(
            "| {document_id} | {format} | {domain} | {extractable_locators}/{total_locators} "
            "| {word_count} | {automatic_acceptance_check} |".format(**item)
        )
    rows.extend(
        [
            "",
            "## Peringatan",
            "",
        ]
    )
    for item in summaries:
        if item["warnings"]:
            for warning in item["warnings"]:
                rows.append(f"- `{item['document_id']}`: {warning}")
    if not any(item["warnings"] for item in summaries):
        rows.append("- Tidak ada peringatan otomatis.")
    rows.extend(
        [
            "",
            "## Batas audit otomatis",
            "",
            "Lulus otomatis tidak berarti isi, lisensi, atau gold evidence sudah disetujui. "
            "Kedua peninjau tetap perlu membuka setiap sumber, memeriksa sampling halaman/slide, "
            "dan menandatangani lembar penerimaan sebelum test set dibekukan.",
            "",
        ]
    )
    (output_dir / "corpus_audit.md").write_text("\n".join(rows), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit technical suitability of corpus candidates.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/final_candidate_documents.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/final_dataset_draft"),
    )
    args = parser.parse_args()
    report = audit_corpus(args.manifest, args.output_dir)
    print(json.dumps(report["totals"], indent=2))


if __name__ == "__main__":
    main()
