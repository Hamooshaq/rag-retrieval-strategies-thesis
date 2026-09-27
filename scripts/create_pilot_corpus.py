from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_DIR = ROOT / "data" / "documents"
MANIFEST_DIR = ROOT / "data" / "manifests"
ANNOTATION_DIR = ROOT / "data" / "annotations"
LICENSE_NOTICE = (
    "Synthetic educational module created for technical validation. "
    "Licensed CC BY 4.0."
)


AI_PDF_PAGES = [
    (
        "Orion Protocol: Evidence-Centered Course Assistant",
        [
            "The Orion protocol is a closed-corpus workflow for a small course assistant. "
            "Its ordered stages are ingest, normalize, structure, embed, retrieve, build context, "
            "generate, evaluate, and persist.",
            "Every extracted block keeps a document identifier and a source locator. "
            "The locator is retained because a plausible answer is not auditable when the supporting "
            "page or slide cannot be reconstructed.",
            "Orion is a course-specific protocol name, not a general artificial intelligence model.",
        ],
    ),
    (
        "Retrieval Units and Frozen Sizes",
        [
            "A parent is one coherent page or slide in the technical pilot. A child is a token window "
            "that never crosses its parent boundary. A neighbor is the parent immediately before or "
            "after a selected parent.",
            "The fixed-size B1 baseline uses 48 whitespace tokens with an overlap of 8 tokens. "
            "The structure-aware child units use 44 tokens with an overlap of 6 tokens.",
            "Each child stores parent_id and locator. Parent expansion therefore changes context "
            "coverage without changing the child embedding model.",
        ],
    ),
    (
        "Question-Conditioned Routing",
        [
            "A factual question starts from child retrieval. A conceptual question maps the best child "
            "score to its parent. A global question selects a parent and its structural neighbor.",
            "A multi-hop question selects evidence from at least two distinct parents before expansion. "
            "The predicted B3 router uses transparent lexical rules, while B4 uses the gold question type.",
            "B4 is diagnostic: the gap between B3 and B4 estimates routing error and is not treated as a "
            "fair deployable baseline.",
        ],
    ),
    (
        "Evaluation, Budget, and Refusal",
        [
            "The pilot reports Recall at 4, mean reciprocal rank, normalized discounted cumulative gain, "
            "and source-location Jaccard accuracy.",
            "All retrieval configurations receive a maximum context budget of 220 generator tokens. "
            "The budget is checked after locator prefixes and deduplication are applied.",
            "If development-calibrated evidence similarity is below the frozen threshold, the system "
            "returns the literal token INSUFFICIENT_EVIDENCE without calling the generator.",
        ],
    ),
]

DATA_PDF_PAGES = [
    (
        "Kirana Module: Abstract Data Types",
        [
            "The Kirana module distinguishes an abstract data type from its implementation. "
            "An abstract data type defines permitted values and operations, while an implementation "
            "chooses a concrete memory representation.",
            "The module uses the decision sequence access pattern, update pattern, ordering requirement, "
            "and memory constraint. This sequence is local to the Kirana teaching module.",
        ],
    ),
    (
        "Arrays and Linked Lists",
        [
            "An array stores elements in a contiguous region and supports constant-time indexed access. "
            "Insertion in the middle may require shifting elements.",
            "A linked list stores nodes connected by references. It supports local insertion without "
            "shifting a contiguous suffix, but finding an arbitrary position requires traversal.",
            "The correct choice depends on the dominant operation rather than on a universal ranking.",
        ],
    ),
    (
        "Stacks and Queues",
        [
            "A stack follows last-in, first-out order. The Kirana example uses a stack to represent an "
            "undo history because the most recent action is reversed first.",
            "A queue follows first-in, first-out order. The Kirana example uses a queue for laboratory "
            "print jobs because requests are served in arrival order.",
            "Both can be implemented with arrays or linked nodes; the abstract ordering rule remains the same.",
        ],
    ),
    (
        "Trees, Graphs, and Traversal",
        [
            "A tree models hierarchical parent-child relations and has a distinguished root. "
            "A graph models more general relations and may contain multiple paths or cycles.",
            "Tree traversal can visit nodes in depth-first or breadth-first order. Graph traversal also "
            "requires a visited set when cycles are possible.",
            "The Kirana summary links representation choice to operations, structural constraints, and "
            "the cost of traversal.",
        ],
    ),
]

AI_SLIDES = [
    (
        "Nusa Retrieval Lab",
        ["Technical pilot for auditable educational RAG", "Closed corpus and local models"],
        "The deck uses course-specific names to reduce accidental parametric answers.",
    ),
    (
        "Provenance Contract",
        [
            "Every retrieved unit must expose unit_id, parent_id, and locator.",
            "PDF locators use page:n; PowerPoint locators use slide:n.",
            "Raw ranking is persisted before generation.",
        ],
        "A missing locator is a validation error, not a warning.",
    ),
    (
        "Threshold Calibration",
        [
            "Threshold is selected only on development questions.",
            "Balanced accuracy gives equal weight to supported and unsupported examples.",
            "The selected value is frozen before pilot evaluation.",
        ],
        "The evaluation split is never used to choose the threshold.",
    ),
    (
        "Failure and Resume",
        [
            "A completed key is run_id plus question_id plus baseline.",
            "Errors are stored with type and message.",
            "A repeated command skips completed keys and may retry error keys.",
        ],
        "Retrieval snapshots allow generation failures to be diagnosed without rerunning retrieval.",
    ),
    (
        "Five Configurations",
        [
            "B0 has no retrieved context.",
            "B1 uses fixed-size chunks; B2 uses structure-bounded child chunks.",
            "B3 uses predicted routing; B4 uses the gold type with the same policy.",
        ],
        "Confirmatory retrieval comparisons concern B1 to B3. B4 is diagnostic.",
    ),
    (
        "Pilot Boundary",
        [
            "Pilot metrics demonstrate technical feasibility only.",
            "Synthetic modules are not the frozen final benchmark.",
            "No claim about student learning is made.",
        ],
        "The final study requires independent annotation by both student researchers.",
    ),
]

DATA_SLIDES = [
    (
        "Kirana Selection Lab",
        ["Choosing structures from access and update needs", "Technical pilot deck"],
        "The deck is paired with, but not copied from, the Kirana PDF module.",
    ),
    (
        "Cost Vocabulary",
        [
            "A linear scan examines up to n elements and has O(n) time.",
            "Indexed array access has O(1) time under the module assumptions.",
            "Complexity describes growth, not measured wall-clock seconds.",
        ],
        "The symbols are treated as text in this pilot.",
    ),
    (
        "Selection Rules",
        [
            "Use a stack when the newest unresolved item must be processed first.",
            "Use a queue when arrival order must be preserved.",
            "Use a graph when relations are not restricted to one parent.",
        ],
        "An undo command is the stack example; a printer line is the queue example.",
    ),
    (
        "Trace Before Optimizing",
        [
            "Write the operation sequence.",
            "Record the state after each operation.",
            "Only then compare alternative implementations.",
        ],
        "This order prevents an asymptotic label from replacing a concrete trace.",
    ),
    (
        "Comparison Guardrail",
        [
            "The same workload must be used for both implementations.",
            "Input size, operation mix, and measurement unit must be recorded.",
            "A faster result on one workload is not a universal proof.",
        ],
        "The guardrail parallels budget matching in retrieval experiments.",
    ),
    (
        "Learning Check",
        [
            "Can the learner justify a structure from operations?",
            "Can the learner trace state changes?",
            "Can the learner state the limit of the comparison?",
        ],
        "The pilot evaluates retrieval, not actual learner outcomes.",
    ),
]


DEV_QUESTIONS = [
    {
        "question_id": "dev_ai_pdf_1",
        "document_id": "pilot_ai_pdf",
        "question": "What literal token is returned when the evidence score is below threshold?",
        "question_type": "factual",
        "answerable": True,
        "reference_answer": "INSUFFICIENT_EVIDENCE",
        "expected_keywords": ["INSUFFICIENT_EVIDENCE"],
        "gold_evidence": {"locators": ["page:4"], "spans": ["literal token"]},
        "provenance": "technical-pilot-author",
        "split": "development",
    },
    {
        "question_id": "dev_ai_ppt_1",
        "document_id": "pilot_ai_ppt",
        "question": "What three identifiers are required by the provenance contract?",
        "question_type": "factual",
        "answerable": True,
        "reference_answer": "unit_id, parent_id, and locator",
        "expected_keywords": ["unit_id", "parent_id", "locator"],
        "gold_evidence": {"locators": ["slide:2"], "spans": ["unit_id, parent_id, and locator"]},
        "provenance": "technical-pilot-author",
        "split": "development",
    },
    {
        "question_id": "dev_data_pdf_1",
        "document_id": "pilot_data_pdf",
        "question": "Which ordering rule does a stack follow?",
        "question_type": "factual",
        "answerable": True,
        "reference_answer": "last-in, first-out",
        "expected_keywords": ["last-in", "first-out"],
        "gold_evidence": {"locators": ["page:3"], "spans": ["last-in, first-out"]},
        "provenance": "technical-pilot-author",
        "split": "development",
    },
    {
        "question_id": "dev_data_ppt_1",
        "document_id": "pilot_data_ppt",
        "question": "What time complexity is assigned to indexed array access?",
        "question_type": "factual",
        "answerable": True,
        "reference_answer": "O(1)",
        "expected_keywords": ["O(1)"],
        "gold_evidence": {"locators": ["slide:2"], "spans": ["O(1)"]},
        "provenance": "technical-pilot-author",
        "split": "development",
    },
    {
        "question_id": "dev_ai_pdf_u1",
        "document_id": "pilot_ai_pdf",
        "question": "What is the Orion dashboard background color?",
        "question_type": "factual",
        "answerable": False,
        "reference_answer": "",
        "expected_keywords": [],
        "gold_evidence": {"locators": [], "spans": []},
        "provenance": "technical-pilot-author",
        "split": "development",
    },
    {
        "question_id": "dev_ai_ppt_u1",
        "document_id": "pilot_ai_ppt",
        "question": "Which cloud region hosts the Nusa service?",
        "question_type": "factual",
        "answerable": False,
        "reference_answer": "",
        "expected_keywords": [],
        "gold_evidence": {"locators": [], "spans": []},
        "provenance": "technical-pilot-author",
        "split": "development",
    },
    {
        "question_id": "dev_data_pdf_u1",
        "document_id": "pilot_data_pdf",
        "question": "What load factor does the Kirana hash table use?",
        "question_type": "factual",
        "answerable": False,
        "reference_answer": "",
        "expected_keywords": [],
        "gold_evidence": {"locators": [], "spans": []},
        "provenance": "technical-pilot-author",
        "split": "development",
    },
    {
        "question_id": "dev_data_ppt_u1",
        "document_id": "pilot_data_ppt",
        "question": "How many students completed the Kirana lab?",
        "question_type": "factual",
        "answerable": False,
        "reference_answer": "",
        "expected_keywords": [],
        "gold_evidence": {"locators": [], "spans": []},
        "provenance": "technical-pilot-author",
        "split": "development",
    },
]


PILOT_QUESTIONS = [
    {
        "question_id": "pilot_f01",
        "document_id": "pilot_ai_pdf",
        "question": "What token size and overlap are used by the fixed-size B1 baseline?",
        "question_type": "factual",
        "answerable": True,
        "reference_answer": "B1 uses 48 whitespace tokens with an overlap of 8 tokens.",
        "expected_keywords": ["48", "8"],
        "gold_evidence": {"locators": ["page:2"], "spans": ["48 whitespace tokens", "overlap of 8"]},
        "provenance": "technical-pilot-author",
        "split": "pilot",
    },
    {
        "question_id": "pilot_f02",
        "document_id": "pilot_ai_ppt",
        "question": "Which fields must every retrieved unit expose?",
        "question_type": "factual",
        "answerable": True,
        "reference_answer": "unit_id, parent_id, and locator",
        "expected_keywords": ["unit_id", "parent_id", "locator"],
        "gold_evidence": {"locators": ["slide:2"], "spans": ["unit_id, parent_id, and locator"]},
        "provenance": "technical-pilot-author",
        "split": "pilot",
    },
    {
        "question_id": "pilot_f03",
        "document_id": "pilot_data_pdf",
        "question": "Which removal order does a queue use in the Kirana module?",
        "question_type": "factual",
        "answerable": True,
        "reference_answer": "A queue uses first-in, first-out order.",
        "expected_keywords": ["first-in", "first-out"],
        "gold_evidence": {"locators": ["page:3"], "spans": ["first-in, first-out"]},
        "provenance": "technical-pilot-author",
        "split": "pilot",
    },
    {
        "question_id": "pilot_f04",
        "document_id": "pilot_data_ppt",
        "question": "What time complexity does the deck assign to a linear scan?",
        "question_type": "factual",
        "answerable": True,
        "reference_answer": "A linear scan has O(n) time.",
        "expected_keywords": ["O(n)"],
        "gold_evidence": {"locators": ["slide:2"], "spans": ["O(n)"]},
        "provenance": "technical-pilot-author",
        "split": "pilot",
    },
    {
        "question_id": "pilot_c01",
        "document_id": "pilot_ai_pdf",
        "question": "Why does the Orion protocol retain a page or slide locator?",
        "question_type": "conceptual",
        "answerable": True,
        "reference_answer": "It retains locators so supporting evidence can be reconstructed and audited.",
        "expected_keywords": ["audit", "supporting"],
        "gold_evidence": {"locators": ["page:1"], "spans": ["not auditable", "supporting page or slide"]},
        "provenance": "technical-pilot-author",
        "split": "pilot",
    },
    {
        "question_id": "pilot_c02",
        "document_id": "pilot_ai_ppt",
        "question": "Why is the threshold selected on development questions rather than the evaluation split?",
        "question_type": "conceptual",
        "answerable": True,
        "reference_answer": "Using development questions avoids tuning the threshold on the evaluation data.",
        "expected_keywords": ["development", "evaluation"],
        "gold_evidence": {"locators": ["slide:3"], "spans": ["selected only on development", "never used"]},
        "provenance": "technical-pilot-author",
        "split": "pilot",
    },
    {
        "question_id": "pilot_c03",
        "document_id": "pilot_data_pdf",
        "question": "Explain the difference between indexed access in an array and access in a linked list.",
        "question_type": "conceptual",
        "answerable": True,
        "reference_answer": "An array offers constant-time indexed access, whereas a linked list must traverse nodes.",
        "expected_keywords": ["constant-time", "traversal"],
        "gold_evidence": {"locators": ["page:2"], "spans": ["constant-time indexed access", "requires traversal"]},
        "provenance": "technical-pilot-author",
        "split": "pilot",
    },
    {
        "question_id": "pilot_c04",
        "document_id": "pilot_data_ppt",
        "question": "Explain why an undo history is represented by a stack in the selection rules.",
        "question_type": "conceptual",
        "answerable": True,
        "reference_answer": "Undo processes the newest unresolved action first, matching last-in, first-out behavior.",
        "expected_keywords": ["newest", "first"],
        "gold_evidence": {"locators": ["slide:3"], "spans": ["newest unresolved item"]},
        "provenance": "technical-pilot-author",
        "split": "pilot",
    },
    {
        "question_id": "pilot_g01",
        "document_id": "pilot_ai_pdf",
        "question": "Summarize the main themes across the Orion protocol document.",
        "question_type": "global",
        "answerable": True,
        "reference_answer": "The document covers an auditable RAG workflow, frozen retrieval units, question-conditioned routing, and budgeted evaluation with refusal.",
        "expected_keywords": ["workflow", "routing", "evaluation"],
        "gold_evidence": {"locators": ["page:1", "page:2", "page:3", "page:4"], "spans": []},
        "provenance": "technical-pilot-author",
        "split": "pilot",
    },
    {
        "question_id": "pilot_g02",
        "document_id": "pilot_ai_ppt",
        "question": "Provide an overview across the Nusa deck after the title slide.",
        "question_type": "global",
        "answerable": True,
        "reference_answer": "The deck covers provenance, development-only threshold calibration, failure resume, five baselines, and the limits of the pilot.",
        "expected_keywords": ["provenance", "threshold", "resume", "pilot"],
        "gold_evidence": {"locators": ["slide:2", "slide:3", "slide:4", "slide:5", "slide:6"], "spans": []},
        "provenance": "technical-pilot-author",
        "split": "pilot",
    },
    {
        "question_id": "pilot_g03",
        "document_id": "pilot_data_pdf",
        "question": "Summarize the main themes across the Kirana data structures module.",
        "question_type": "global",
        "answerable": True,
        "reference_answer": "The module relates abstract data types, arrays and lists, stacks and queues, and trees and graphs to operations and traversal costs.",
        "expected_keywords": ["abstract", "arrays", "stacks", "trees"],
        "gold_evidence": {"locators": ["page:1", "page:2", "page:3", "page:4"], "spans": []},
        "provenance": "technical-pilot-author",
        "split": "pilot",
    },
    {
        "question_id": "pilot_g04",
        "document_id": "pilot_data_ppt",
        "question": "Summarize the whole Kirana selection deck after its title.",
        "question_type": "global",
        "answerable": True,
        "reference_answer": "The deck explains cost vocabulary, selection rules, tracing, fair comparison, and learning checks.",
        "expected_keywords": ["cost", "selection", "trace", "comparison"],
        "gold_evidence": {"locators": ["slide:2", "slide:3", "slide:4", "slide:5", "slide:6"], "spans": []},
        "provenance": "technical-pilot-author",
        "split": "pilot",
    },
    {
        "question_id": "pilot_m01",
        "document_id": "pilot_ai_pdf",
        "question": "Combine the multi-hop routing rule with the context budget and refusal rule.",
        "question_type": "multi_hop",
        "answerable": True,
        "reference_answer": "Multi-hop retrieves distinct parents, then context is capped at 220 tokens and the system refuses if evidence remains below threshold.",
        "expected_keywords": ["distinct", "220", "refus"],
        "gold_evidence": {"locators": ["page:3", "page:4"], "spans": ["distinct parents", "220", "threshold"]},
        "provenance": "technical-pilot-author",
        "split": "pilot",
    },
    {
        "question_id": "pilot_m02",
        "document_id": "pilot_ai_ppt",
        "question": "Using both provenance and failure handling, explain how a run can be audited after a generator error.",
        "question_type": "multi_hop",
        "answerable": True,
        "reference_answer": "The stored raw ranking preserves unit IDs and locators, while error records and resume keys allow the failed generation to be retried.",
        "expected_keywords": ["ranking", "locator", "error", "retry"],
        "gold_evidence": {"locators": ["slide:2", "slide:4"], "spans": ["Raw ranking", "Errors are stored", "retry"]},
        "provenance": "technical-pilot-author",
        "split": "pilot",
    },
    {
        "question_id": "pilot_m03",
        "document_id": "pilot_data_pdf",
        "question": "Compare the ordering property of a stack with the structural property of a tree.",
        "question_type": "multi_hop",
        "answerable": True,
        "reference_answer": "A stack orders operations last-in first-out, while a tree represents hierarchical parent-child relations from a root.",
        "expected_keywords": ["last-in", "parent-child", "root"],
        "gold_evidence": {"locators": ["page:3", "page:4"], "spans": ["last-in", "parent-child", "root"]},
        "provenance": "technical-pilot-author",
        "split": "pilot",
    },
    {
        "question_id": "pilot_m04",
        "document_id": "pilot_data_ppt",
        "question": "Combine the cost vocabulary with the selection rules to justify array access versus a queue.",
        "question_type": "multi_hop",
        "answerable": True,
        "reference_answer": "Indexed array access is O(1), while a queue is selected when arrival order must be preserved.",
        "expected_keywords": ["O(1)", "arrival"],
        "gold_evidence": {"locators": ["slide:2", "slide:3"], "spans": ["O(1)", "arrival order"]},
        "provenance": "technical-pilot-author",
        "split": "pilot",
    },
    {
        "question_id": "pilot_u01",
        "document_id": "pilot_ai_pdf",
        "question": "What color is the Orion dashboard status light?",
        "question_type": "factual",
        "answerable": False,
        "reference_answer": "",
        "expected_keywords": [],
        "gold_evidence": {"locators": [], "spans": []},
        "provenance": "technical-pilot-author",
        "split": "pilot",
    },
    {
        "question_id": "pilot_u02",
        "document_id": "pilot_ai_ppt",
        "question": "Which cloud provider hosts the Nusa service?",
        "question_type": "factual",
        "answerable": False,
        "reference_answer": "",
        "expected_keywords": [],
        "gold_evidence": {"locators": [], "spans": []},
        "provenance": "technical-pilot-author",
        "split": "pilot",
    },
    {
        "question_id": "pilot_u03",
        "document_id": "pilot_data_pdf",
        "question": "What load factor is used by the Kirana hash table?",
        "question_type": "factual",
        "answerable": False,
        "reference_answer": "",
        "expected_keywords": [],
        "gold_evidence": {"locators": [], "spans": []},
        "provenance": "technical-pilot-author",
        "split": "pilot",
    },
    {
        "question_id": "pilot_u04",
        "document_id": "pilot_data_ppt",
        "question": "How many students passed the Kirana final examination?",
        "question_type": "factual",
        "answerable": False,
        "reference_answer": "",
        "expected_keywords": [],
        "gold_evidence": {"locators": [], "spans": []},
        "provenance": "technical-pilot-author",
        "split": "pilot",
    },
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pdf_footer(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(HexColor("#555555"))
    canvas.drawString(2 * cm, 1.2 * cm, LICENSE_NOTICE)
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {document.page}")
    canvas.restoreState()


def create_pdf(path: Path, title: str, pages: list[tuple[str, list[str]]]) -> None:
    styles = getSampleStyleSheet()
    heading = ParagraphStyle(
        "PilotHeading",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=HexColor("#17365D"),
        spaceAfter=18,
    )
    body = ParagraphStyle(
        "PilotBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=11,
        leading=16,
        spaceAfter=14,
    )
    story = []
    for page_index, (page_title, paragraphs) in enumerate(pages):
        if page_index == 0:
            story.append(Paragraph(title, ParagraphStyle(
                "DocTitle", parent=heading, fontSize=12, textColor=HexColor("#666666"), spaceAfter=10
            )))
        story.append(Paragraph(page_title, heading))
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph, body))
            story.append(Spacer(1, 4))
        if page_index + 1 < len(pages):
            story.append(PageBreak())
    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=title,
        author="Research Team",
        subject="Synthetic technical pilot educational module",
    )
    document.build(story, onFirstPage=_pdf_footer, onLaterPages=_pdf_footer)


def _add_license_footer(slide):
    box = slide.shapes.add_textbox(Inches(0.35), Inches(7.05), Inches(12.6), Inches(0.25))
    frame = box.text_frame
    frame.clear()
    paragraph = frame.paragraphs[0]
    paragraph.text = LICENSE_NOTICE
    paragraph.font.size = Pt(7)
    paragraph.font.color.rgb = RGBColor(90, 90, 90)
    paragraph.alignment = PP_ALIGN.CENTER


def create_pptx(path: Path, slides: list[tuple[str, list[str], str]]) -> None:
    presentation = Presentation()
    presentation.slide_width = Inches(13.333)
    presentation.slide_height = Inches(7.5)
    for slide_index, (title, bullets, notes) in enumerate(slides):
        layout = presentation.slide_layouts[1]
        slide = presentation.slides.add_slide(layout)
        slide.shapes.title.text = title
        title_frame = slide.shapes.title.text_frame
        title_frame.paragraphs[0].font.size = Pt(28)
        title_frame.paragraphs[0].font.bold = True
        body = slide.placeholders[1].text_frame
        body.clear()
        for index, bullet in enumerate(bullets):
            paragraph = body.paragraphs[0] if index == 0 else body.add_paragraph()
            paragraph.text = bullet
            paragraph.level = 0
            paragraph.font.size = Pt(21)
            paragraph.space_after = Pt(12)
        try:
            slide.notes_slide.notes_text_frame.text = notes
        except Exception:
            pass
        _add_license_footer(slide)
    presentation.save(path)


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False))
            stream.write("\n")


def main() -> None:
    DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    files = {
        "pilot_ai_pdf": DOCUMENT_DIR / "pilot_ai_foundations.pdf",
        "pilot_ai_ppt": DOCUMENT_DIR / "pilot_ai_pipeline.pptx",
        "pilot_data_pdf": DOCUMENT_DIR / "pilot_data_structures.pdf",
        "pilot_data_ppt": DOCUMENT_DIR / "pilot_data_selection.pptx",
    }
    create_pdf(files["pilot_ai_pdf"], "AI Foundations Pilot Module", AI_PDF_PAGES)
    create_pptx(files["pilot_ai_ppt"], AI_SLIDES)
    create_pdf(files["pilot_data_pdf"], "Data Structures Pilot Module", DATA_PDF_PAGES)
    create_pptx(files["pilot_data_ppt"], DATA_SLIDES)

    manifest = {
        "schema_version": "1.0",
        "created_for": "technical pilot only",
        "license_notice": LICENSE_NOTICE,
        "documents": [
            {
                "document_id": "pilot_ai_pdf",
                "path": "documents/pilot_ai_foundations.pdf",
                "format": "pdf",
                "domain": "artificial_intelligence",
                "title": "AI Foundations Pilot Module",
                "language": "en",
                "license": "CC BY 4.0",
                "source_url": None,
                "sha256": _sha256(files["pilot_ai_pdf"]),
                "pilot_only": True,
            },
            {
                "document_id": "pilot_ai_ppt",
                "path": "documents/pilot_ai_pipeline.pptx",
                "format": "pptx",
                "domain": "artificial_intelligence",
                "title": "Nusa Retrieval Lab",
                "language": "en",
                "license": "CC BY 4.0",
                "source_url": None,
                "sha256": _sha256(files["pilot_ai_ppt"]),
                "pilot_only": True,
            },
            {
                "document_id": "pilot_data_pdf",
                "path": "documents/pilot_data_structures.pdf",
                "format": "pdf",
                "domain": "data_structures",
                "title": "Kirana Module: Abstract Data Types",
                "language": "en",
                "license": "CC BY 4.0",
                "source_url": None,
                "sha256": _sha256(files["pilot_data_pdf"]),
                "pilot_only": True,
            },
            {
                "document_id": "pilot_data_ppt",
                "path": "documents/pilot_data_selection.pptx",
                "format": "pptx",
                "domain": "data_structures",
                "title": "Kirana Selection Lab",
                "language": "en",
                "license": "CC BY 4.0",
                "source_url": None,
                "sha256": _sha256(files["pilot_data_ppt"]),
                "pilot_only": True,
            },
        ],
    }
    with (MANIFEST_DIR / "pilot_documents.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False, allow_unicode=True)
    write_jsonl(ANNOTATION_DIR / "pilot_dev_questions.jsonl", DEV_QUESTIONS)
    write_jsonl(ANNOTATION_DIR / "pilot_questions.jsonl", PILOT_QUESTIONS)

    freeze = {
        "manifest_sha256": _sha256(MANIFEST_DIR / "pilot_documents.yaml"),
        "development_questions_sha256": _sha256(
            ANNOTATION_DIR / "pilot_dev_questions.jsonl"
        ),
        "evaluation_questions_sha256": _sha256(ANNOTATION_DIR / "pilot_questions.jsonl"),
        "document_sha256": {key: _sha256(value) for key, value in files.items()},
        "pilot_question_count": len(PILOT_QUESTIONS),
        "development_question_count": len(DEV_QUESTIONS),
        "license": "CC BY 4.0",
        "scope": "technical feasibility; not final research benchmark",
    }
    (MANIFEST_DIR / "pilot_freeze.json").write_text(
        json.dumps(freeze, indent=2) + "\n", encoding="utf-8"
    )
    (DOCUMENT_DIR / "LICENSE.md").write_text(
        "# Pilot Corpus License\n\n"
        + LICENSE_NOTICE
        + "\n\nThese synthetic documents may be shared and adapted with attribution under "
        "Creative Commons Attribution 4.0 International.\n",
        encoding="utf-8",
    )
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
