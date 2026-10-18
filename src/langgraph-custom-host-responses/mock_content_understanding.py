
"""Mock Content Understanding adapter for synthetic mortgage documents.

This adapter reads actual PDF text and extracts known fields using
deterministic rules. It does not call Azure AI Content Understanding.

Confidence values are illustrative placeholders, not model-generated scores.
"""

from pathlib import Path
import re
from typing import Any

from pypdf import PdfReader


def _money(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def extract_document(pdf_path: str | Path) -> dict[str, Any]:
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise ValueError(f"No extractable text found in PDF: {path.name}")

    if "DOC-APP-" in text or "Mortgage Application Summary" in text:
        document_type = "mortgage_application"
    elif "DOC-PAY-" in text or "Pay Statement" in text:
        document_type = "pay_stub"
    elif "DOC-EMP-" in text or "Employment Verification Letter" in text:
        document_type = "employment_letter"
    else:
        document_type = "unknown"

    fields: dict[str, Any] = {}

    patterns = {
        "annual_income_cad": r"(?:Declared annual income|Annualized income|Annual salary)\s*(?:CAD)?\s*:?\s*CAD?\s*([\d,]+)",
        "monthly_liabilities_cad": r"Declared monthly liabilities\s*:?\s*CAD?\s*([\d,]+)",
        "property_value_cad": r"Property value\s*:?\s*CAD?\s*([\d,]+)",
        "requested_mortgage_cad": r"Requested mortgage amount\s*:?\s*CAD?\s*([\d,]+)",
    }

    for field, pattern in patterns.items():
        value = _money(text, pattern)
        if value is not None:
            fields[field] = value

    if re.search(r"Permanent full-time", text, re.IGNORECASE):
        fields["employment_status"] = "permanent full-time"
    elif re.search(r"Employed", text, re.IGNORECASE):
        fields["employment_status"] = "employed"

    # Preserve extracted text as evidence for the test workflow.
    return {
        "case_id": _find_case_id(text),
        "document_id": _find_document_id(text),
        "file_name": path.name,
        "document_type": document_type,
        "extraction_mode": "mock_deterministic_pdf_text",
        "synthetic_only": True,
        "fields": {
            name: {
                "value": value,
                "confidence": None,
                "confidence_note": "Not scored by the mock adapter",
                "evidence": _evidence_for_field(text, name),
            }
            for name, value in fields.items()
        },
        "raw_text": text,
    }


def _find_case_id(text: str) -> str | None:
    match = re.search(r"\bMORT-\d+\b", text)
    return match.group(0) if match else None


def _find_document_id(text: str) -> str | None:
    match = re.search(r"\bDOC-[A-Z]+-\d+\b", text)
    return match.group(0) if match else None



def _evidence_for_field(text: str, field: str) -> str | None:
    labels = {
        "annual_income_cad": (
            "Declared annual income",
            "Annualized income",
            "Annual salary",
        ),
        "monthly_liabilities_cad": ("Declared monthly liabilities",),
        "property_value_cad": ("Property value",),
        "requested_mortgage_cad": ("Requested mortgage amount",),
        "employment_status": ("Employment status",),
    }

    for label in labels.get(field, ()):
        pattern = (
            rf"{re.escape(label)}\s*:?\s*"
            rf"(?:CAD\s*)?[\d,]+"
        ) if field != "employment_status" else (
            rf"{re.escape(label)}\s*:?\s*"
            rf"(?:Permanent full-time|Employed)"
        )

        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return " ".join(match.group(0).split())

        # In these PDFs, labels and values can be on separate lines.
        pattern = (
            rf"{re.escape(label)}\s*\n\s*(?:CAD\s*)?[\d,]+"
        ) if field != "employment_status" else (
            rf"{re.escape(label)}\s*\n\s*"
            rf"(?:Permanent full-time|Employed)"
        )

        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return " ".join(match.group(0).split())

    return None



def extract_case_folder(folder: str | Path) -> list[dict[str, Any]]:
    """Extract all PDFs in a case folder, sorted by filename."""
    path = Path(folder)
    if not path.is_dir():
        raise NotADirectoryError(f"Case folder not found: {path}")

    results = []
    for pdf in sorted(path.glob("*.pdf")):
        results.append(extract_document(pdf))
    return results
