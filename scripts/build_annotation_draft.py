from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from storage.io import load_document_manifest
from storage.models import (
    DifficultyTag,
    DocumentSpec,
    GoldEvidence,
    QuestionRecord,
    QuestionType,
)

QUESTION_TYPES = (
    QuestionType.FACTUAL,
    QuestionType.CONCEPTUAL,
    QuestionType.GLOBAL,
    QuestionType.MULTI_HOP,
)
CELL_LABELS = (
    "artificial_intelligence:pdf",
    "artificial_intelligence:pptx",
    "data_structures_algorithms:pdf",
    "data_structures_algorithms:pptx",
)
STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "among",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "could",
    "document",
    "does",
    "each",
    "from",
    "have",
    "into",
    "more",
    "most",
    "other",
    "over",
    "same",
    "such",
    "than",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "through",
    "using",
    "very",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}
EXCLUDED_MARKERS = (
    "creative commons",
    "copyright",
    "http://",
    "https://",
    "www.",
    "all rights reserved",
    "table of contents",
    "acknowledgment",
    "bibliography",
    "references",
    "facilitator notes",
    "image source",
    "lecture notes",
    "available on github",
)
INCOMPLETE_ENDINGS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "because",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "so",
    "than",
    "that",
    "the",
    "then",
    "to",
    "when",
    "where",
    "which",
    "while",
    "with",
}


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_questions(path: Path, records: list[QuestionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record.model_dump(mode="json", exclude_none=True), ensure_ascii=False)
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" \t\r\n•-–—")


def _balanced(text: str, opening: str, closing: str) -> bool:
    return text.count(opening) == text.count(closing)


def _looks_complete(sentence: str) -> bool:
    words = sentence.split()
    if not words:
        return False
    ending = re.sub(r"[^A-Za-z]+", "", words[-1]).lower()
    if ending in INCOMPLETE_ENDINGS:
        return False
    if sentence.count('"') % 2 or sentence.count("“") != sentence.count("”"):
        return False
    if not _balanced(sentence, "(", ")") or not _balanced(sentence, "[", "]"):
        return False
    if re.search(r"\b(?:asO|fora|forb|range0)\b", sentence):
        return False
    return True


def _sentences(text: str) -> list[str]:
    # PDF parsers commonly preserve visual line wrapping. Joining lines before
    # splitting prevents a sentence from being treated as several fragments.
    normalized = _normalize_space(text.replace("\x0b", " "))
    candidates = [
        _normalize_space(piece)
        for piece in re.split(r"(?<=[.!?])\s+(?=[\"“A-Z0-9])", normalized)
    ]

    accepted: list[str] = []
    for sentence in candidates:
        sentence = re.sub(
            r"^(?:Facilitator notes|Trainer notes|Notes)\s*:\s*",
            "",
            sentence,
            flags=re.IGNORECASE,
        ).strip()
        lower = sentence.lower()
        words = sentence.split()
        alpha_ratio = sum(character.isalpha() for character in sentence) / max(
            1, len(sentence)
        )
        if not 10 <= len(words) <= 55:
            continue
        if alpha_ratio < 0.55:
            continue
        if any(marker in lower for marker in EXCLUDED_MARKERS):
            continue
        if sentence.count("=") > 2 or sentence.count("|") > 1:
            continue
        if not _looks_complete(sentence):
            continue
        if re.search(r"\b(?:chapter|figure|table)\s+\d+\s*$", lower):
            continue
        accepted.append(sentence)
    return list(dict.fromkeys(accepted))


def _keywords(text: str, limit: int = 5) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", text.lower())
    result: list[str] = []
    for word in words:
        if word in STOPWORDS or word in result:
            continue
        result.append(word)
        if len(result) == limit:
            break
    return result


def _topic(sentence: str) -> str:
    sentence = _normalize_space(sentence).strip("\"'“”")
    sentence = re.sub(
        r"^(?:According to the document|For example|In general|Therefore|However),?\s+",
        "",
        sentence,
        flags=re.IGNORECASE,
    )
    subject_match = re.match(
        r"^(?:An?|The)?\s*(.{3,90}?)\s+"
        r"(?:is|are|means|refers to|consists of|uses|requires|allows|provides|"
        r"can|helps|describes|represents)\b",
        sentence,
        flags=re.IGNORECASE,
    )
    if subject_match:
        subject = _normalize_space(subject_match.group(1))
        if 1 <= len(subject.split()) <= 10:
            return subject
    colon_match = re.match(r"^([A-Z][A-Za-z0-9 /-]{2,55}):", sentence)
    if colon_match:
        return _normalize_space(colon_match.group(1))
    keywords = _keywords(sentence, limit=3)
    return " ".join(keywords) if keywords else "the stated concept"


def _sentence_quality(sentence: str) -> tuple[int, int, str]:
    words = sentence.split()
    lower = sentence.lower()
    score = 0
    if sentence.endswith((".", "?", "!", '"', "”")):
        score += 3
    if re.search(
        r"\b(is|are|means|refers to|consists of|because|therefore|allows|"
        r"requires|can|helps|represents|defined as)\b",
        sentence,
        flags=re.IGNORECASE,
    ):
        score += 3
    if 14 <= len(words) <= 38:
        score += 2
    if sentence[:1].isupper() or sentence.startswith(("\"", "“")):
        score += 1
    if any(
        marker in lower
        for marker in (
            "we will",
            "let’s",
            "let us",
            "ask learners",
            "tell students",
            "in this course",
        )
    ):
        score -= 3
    return (-score, abs(len(words) - 24), sentence)


def _factual_question(sentence: str) -> str:
    patterns = (
        (r"^(?:The\s+)?(.{2,80}?)\s+is\s+(.+)$", "What is {subject}?"),
        (r"^(?:The\s+)?(.{2,80}?)\s+are\s+(.+)$", "What are {subject}?"),
        (r"^(?:The\s+)?(.{2,80}?)\s+means\s+(.+)$", "What does {subject} mean?"),
        (
            r"^(?:The\s+)?(.{2,80}?)\s+refers to\s+(.+)$",
            "What does {subject} refer to?",
        ),
        (
            r"^(?:The\s+)?(.{2,80}?)\s+consists of\s+(.+)$",
            "What does {subject} consist of?",
        ),
        (r"^(?:The\s+)?(.{2,80}?)\s+uses\s+(.+)$", "What does {subject} use?"),
        (
            r"^(?:The\s+)?(.{2,80}?)\s+requires\s+(.+)$",
            "What does {subject} require?",
        ),
        (r"^(?:The\s+)?(.{2,80}?)\s+can\s+(.+)$", "What can {subject} do?"),
        (
            r"^(?:The\s+)?(.{2,80}?)\s+allows\s+(.+)$",
            "What does {subject} allow?",
        ),
    )
    for pattern, template in patterns:
        match = re.match(pattern, sentence, flags=re.IGNORECASE)
        if match:
            subject = _normalize_space(match.group(1))
            if 1 <= len(subject.split()) <= 10:
                return template.format(subject=subject)
    return f"What information does the document provide about {_topic(sentence)}?"


def _difficulty(question_type: QuestionType) -> DifficultyTag:
    if question_type == QuestionType.FACTUAL:
        return DifficultyTag.EASY
    if question_type == QuestionType.MULTI_HOP:
        return DifficultyTag.HARD
    return DifficultyTag.MEDIUM


def _locator_number(locator: str) -> int:
    return int(locator.split(":", 1)[1])


def _cell(document: DocumentSpec) -> str:
    return f"{document.domain}:{document.format.value}"


def _eligible_locators(records: list[dict]) -> list[dict]:
    eligible = [record for record in records if _sentences(record["text"])]
    return sorted(
        eligible,
        key=lambda record: (
            _sentence_quality(min(_sentences(record["text"]), key=_sentence_quality)),
            _locator_number(record["locator"]),
        ),
    )


def _choose_single(records: list[dict], item_index: int, offset: int) -> dict:
    eligible = _eligible_locators(records)
    if not eligible:
        raise ValueError(f"No eligible locator for {records[0]['document_id']}")
    return eligible[(item_index * 7 + offset) % len(eligible)]


def _choose_pair(
    records: list[dict],
    item_index: int,
    offset: int,
    *,
    separated: bool,
) -> tuple[dict, dict]:
    eligible = _eligible_locators(records)
    if len(eligible) < 2:
        raise ValueError(f"Fewer than two eligible locators for {records[0]['document_id']}")
    first_index = (item_index * 5 + offset) % len(eligible)
    if separated:
        distance = max(2, len(eligible) // 3)
        second_index = (first_index + distance) % len(eligible)
    else:
        second_index = (first_index + 1) % len(eligible)
    if first_index == second_index:
        second_index = (second_index + 1) % len(eligible)
    return eligible[first_index], eligible[second_index]


def _best_sentence(locator_record: dict, salt: int) -> str:
    sentences = _sentences(locator_record["text"])
    if not sentences:
        raise ValueError(
            f"No eligible sentence in {locator_record['document_id']} "
            f"{locator_record['locator']}"
        )
    ranked = sorted(sentences, key=_sentence_quality)
    return ranked[salt % min(3, len(ranked))]


def _answerable_record(
    *,
    question_id: str,
    document: DocumentSpec,
    locator_records: list[dict],
    question_type: QuestionType,
    split: str,
    item_index: int,
    offset: int,
) -> QuestionRecord:
    if question_type in {QuestionType.FACTUAL, QuestionType.CONCEPTUAL}:
        evidence = [_choose_single(locator_records, item_index, offset)]
    elif question_type == QuestionType.GLOBAL:
        evidence = list(
            _choose_pair(locator_records, item_index, offset, separated=False)
        )
    else:
        evidence = list(_choose_pair(locator_records, item_index, offset, separated=True))

    sentences = [
        _best_sentence(record, salt=item_index + position)
        for position, record in enumerate(evidence)
    ]
    topics = [_topic(sentence) for sentence in sentences]
    if question_type == QuestionType.FACTUAL:
        question = _factual_question(sentences[0])
    elif question_type == QuestionType.CONCEPTUAL:
        question = f"How does the document describe {topics[0]}?"
    elif question_type == QuestionType.GLOBAL:
        question = (
            f"Based on the two cited locations, what overall explanation does "
            f"the document provide about {topics[0]} and {topics[1]}?"
        )
    else:
        question = (
            f"Using evidence from both cited locations, how are {topics[0]} "
            f"and {topics[1]} related in the document?"
        )

    locators = list(dict.fromkeys(record["locator"] for record in evidence))
    reference_answer = " ".join(sentences)
    return QuestionRecord(
        question_id=question_id,
        document_id=document.document_id,
        domain=document.domain,
        document_format=document.format,
        question=_normalize_space(question),
        question_type=question_type,
        difficulty_tag=_difficulty(question_type),
        answerable=True,
        reference_answer=reference_answer,
        expected_keywords=_keywords(reference_answer, limit=6),
        gold_evidence=GoldEvidence(
            locators=locators,
            unit_ids=[
                f"{document.document_id}:parent:{locator}" for locator in locators
            ],
            spans=sentences,
        ),
        provenance=(
            "machine-assisted draft from parser-extracted source text; "
            "pending independent verification by both student researchers"
        ),
        split=split,
        visual_required=False,
        annotation_status="draft_pending_human_verification",
    )


def _unanswerable_record(
    *,
    question_id: str,
    target_document: DocumentSpec,
    source_record: QuestionRecord,
    split: str,
    target_text: str,
) -> QuestionRecord:
    significant = _keywords(
        f"{source_record.question} {' '.join(source_record.expected_keywords)}",
        limit=10,
    )
    absent = [word for word in significant if word not in target_text.lower()]
    if len(absent) < 2:
        raise ValueError(
            f"Cannot establish absent anchors for {question_id} from "
            f"{source_record.question_id} to {target_document.document_id}"
        )
    scoped_question = (
        "Within this document's scope, "
        f"{source_record.question[0].lower()}{source_record.question[1:]}"
    )
    return QuestionRecord(
        question_id=question_id,
        document_id=target_document.document_id,
        domain=target_document.domain,
        document_format=target_document.format,
        question=scoped_question,
        question_type=source_record.question_type,
        difficulty_tag=_difficulty(source_record.question_type),
        answerable=False,
        reference_answer="",
        expected_keywords=[],
        gold_evidence=GoldEvidence(),
        provenance=(
            "cross-domain entity-swap draft; "
            f"source_question={source_record.question_id}; "
            f"verified_absent_anchors={','.join(absent[:5])}; "
            "pending independent verification by both student researchers"
        ),
        split=split,
        visual_required=False,
        annotation_status="draft_pending_human_verification",
    )


def _select_absent_source(
    sources: list[QuestionRecord],
    target_text: str,
    used_source_ids: set[str],
) -> QuestionRecord:
    for source in sources:
        if source.question_id in used_source_ids:
            continue
        significant = _keywords(
            f"{source.question} {' '.join(source.expected_keywords)}",
            limit=10,
        )
        absent = [word for word in significant if word not in target_text.lower()]
        if len(absent) >= 2:
            used_source_ids.add(source.question_id)
            return source
    raise ValueError("No unused cross-domain source with at least two absent anchors")


def build_draft(
    manifest_path: Path,
    locator_text_path: Path,
) -> tuple[list[QuestionRecord], list[QuestionRecord]]:
    documents = load_document_manifest(manifest_path)
    documents_by_id = {document.document_id: document for document in documents}
    documents_by_cell: dict[str, list[DocumentSpec]] = defaultdict(list)
    for document in documents:
        documents_by_cell[_cell(document)].append(document)
    if set(documents_by_cell) != set(CELL_LABELS):
        raise ValueError(f"Unexpected domain-format cells: {sorted(documents_by_cell)}")

    locator_records_by_document: dict[str, list[dict]] = defaultdict(list)
    for record in _load_jsonl(locator_text_path):
        locator_records_by_document[record["document_id"]].append(record)
    missing = set(documents_by_id) - set(locator_records_by_document)
    if missing:
        raise ValueError(f"Missing locator text for documents: {sorted(missing)}")

    full_text_by_document = {
        document_id: "\n".join(record["text"] for record in records).lower()
        for document_id, records in locator_records_by_document.items()
    }
    test_answerable: list[QuestionRecord] = []
    development_answerable: list[QuestionRecord] = []

    for cell_index, cell_label in enumerate(CELL_LABELS):
        cell_documents = sorted(
            documents_by_cell[cell_label], key=lambda document: document.document_id
        )
        short_cell = (
            "ai-pdf"
            if cell_label == "artificial_intelligence:pdf"
            else "ai-ppt"
            if cell_label == "artificial_intelligence:pptx"
            else "ds-pdf"
            if cell_label == "data_structures_algorithms:pdf"
            else "ds-ppt"
        )
        for type_index, question_type in enumerate(QUESTION_TYPES):
            type_code = {
                QuestionType.FACTUAL: "f",
                QuestionType.CONCEPTUAL: "c",
                QuestionType.GLOBAL: "g",
                QuestionType.MULTI_HOP: "m",
            }[question_type]
            for item_index in range(8):
                document = cell_documents[(item_index + type_index) % len(cell_documents)]
                test_answerable.append(
                    _answerable_record(
                        question_id=f"test-{short_cell}-{type_code}{item_index + 1:02d}",
                        document=document,
                        locator_records=locator_records_by_document[document.document_id],
                        question_type=question_type,
                        split="test",
                        item_index=item_index,
                        offset=cell_index * 11 + type_index * 3,
                    )
                )
            development_document = cell_documents[
                (type_index + cell_index) % len(cell_documents)
            ]
            development_answerable.append(
                _answerable_record(
                    question_id=f"dev-{short_cell}-{type_code}01",
                    document=development_document,
                    locator_records=locator_records_by_document[
                        development_document.document_id
                    ],
                    question_type=question_type,
                    split="development",
                    item_index=12 + type_index,
                    offset=cell_index * 13 + type_index * 5,
                )
            )

    test_records = list(test_answerable)
    development_records = list(development_answerable)
    used_test_sources: set[str] = set()
    used_development_sources: set[str] = set()

    for cell_index, cell_label in enumerate(CELL_LABELS):
        cell_documents = sorted(
            documents_by_cell[cell_label], key=lambda document: document.document_id
        )
        domain, document_format = cell_label.split(":")
        opposite_domain = (
            "data_structures_algorithms"
            if domain == "artificial_intelligence"
            else "artificial_intelligence"
        )
        source_cell = f"{opposite_domain}:{document_format}"
        source_test = [
            record
            for record in test_answerable
            if f"{record.domain}:{record.document_format.value}" == source_cell
        ]
        source_development = [
            record
            for record in development_answerable
            if f"{record.domain}:{record.document_format.value}" == source_cell
        ]
        short_cell = (
            "ai-pdf"
            if cell_label == "artificial_intelligence:pdf"
            else "ai-ppt"
            if cell_label == "artificial_intelligence:pptx"
            else "ds-pdf"
            if cell_label == "data_structures_algorithms:pdf"
            else "ds-ppt"
        )

        for item_index in range(8):
            question_type = QUESTION_TYPES[item_index % len(QUESTION_TYPES)]
            target_document = cell_documents[item_index % len(cell_documents)]
            target_text = full_text_by_document[target_document.document_id]
            matching_sources = [
                source
                for source in source_test
                if source.question_type == question_type
            ]
            source = _select_absent_source(
                matching_sources, target_text, used_test_sources
            )
            test_records.append(
                _unanswerable_record(
                    question_id=f"test-{short_cell}-u{item_index + 1:02d}",
                    target_document=target_document,
                    source_record=source,
                    split="test",
                    target_text=target_text,
                )
            )

        development_target = cell_documents[cell_index % len(cell_documents)]
        development_text = full_text_by_document[development_target.document_id]
        source = _select_absent_source(
            source_development,
            development_text,
            used_development_sources,
        )
        development_records.append(
            _unanswerable_record(
                question_id=f"dev-{short_cell}-u01",
                target_document=development_target,
                source_record=source,
                split="development",
                target_text=development_text,
            )
        )

    return (
        sorted(development_records, key=lambda record: record.question_id),
        sorted(test_records, key=lambda record: record.question_id),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a machine-assisted annotation draft from frozen locator text."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/final_candidate_documents.yaml"),
    )
    parser.add_argument(
        "--locator-text",
        type=Path,
        default=Path("artifacts/final_dataset_draft/locator_text.jsonl"),
    )
    parser.add_argument(
        "--development-output",
        type=Path,
        default=Path("data/annotations/final_development_questions.draft.jsonl"),
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path("data/annotations/final_test_questions.draft.jsonl"),
    )
    args = parser.parse_args()
    development, test = build_draft(args.manifest, args.locator_text)
    _write_questions(args.development_output, development)
    _write_questions(args.test_output, test)
    print(
        json.dumps(
            {
                "status": "draft_pending_human_verification",
                "development_questions": len(development),
                "test_questions": len(test),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
