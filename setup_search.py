"""Extract synthetic mortgage PDFs and index their fields in Azure AI Search.

Run from the repository root (using the app's dependency environment):
    uv run --project src/langgraph-custom-host-responses python setup_search.py

Authentication uses DefaultAzureCredential (for example, az login). The script
indexes only PDFs under sample-documents/MORT-1001 through MORT-1006. JSON fixtures
and manifests are deliberately excluded from the evidence index.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchField, SearchFieldDataType, SearchIndex
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "src" / "langgraph-custom-host-responses"
SAMPLE_ROOT = ROOT / "sample-documents"
sys.path.insert(0, str(APP_DIR))

from mock_content_understanding import extract_case_folder  # noqa: E402

CASE_IDS = [f"MORT-{number}" for number in range(1001, 1007)]
INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "mortgage-document-evidence")


def build_index() -> SearchIndex:
    """Define the schema consumed by main.py and its evidence search tool."""
    return SearchIndex(
        name=INDEX_NAME,
        fields=[
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(name="case_id", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SearchField(name="document_id", type=SearchFieldDataType.String, filterable=True),
            SearchField(name="file_name", type=SearchFieldDataType.String),
            SearchField(name="document_type", type=SearchFieldDataType.String, filterable=True),
            SearchField(name="field_name", type=SearchFieldDataType.String, filterable=True),
            SearchField(name="field_value", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="field_value_number", type=SearchFieldDataType.Double, filterable=True, sortable=True),
            SearchField(name="evidence", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="extraction_mode", type=SearchFieldDataType.String),
            SearchField(name="synthetic_only", type=SearchFieldDataType.Boolean, filterable=True),
        ],
    )


def _index_record(case_id: str, document: dict[str, Any]) -> list[dict[str, Any]]:
    if document.get("case_id") != case_id:
        raise ValueError(
            f"{document.get('file_name')}: expected case_id {case_id}, "
            f"found {document.get('case_id')!r}"
        )
    document_id = document.get("document_id")
    document_type = document.get("document_type")
    if not document_id:
        raise ValueError(f"{document.get('file_name')}: PDF has no recognizable document ID")
    if not document_type or document_type == "unknown":
        raise ValueError(f"{document.get('file_name')}: PDF type could not be identified")
    fields = document.get("fields") or {}
    if not fields:
        raise ValueError(f"{document.get('file_name')}: no fields extracted from PDF")

    records = []
    for field_name, item in fields.items():
        value = item.get("value")
        numeric = (
            float(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else None
        )
        records.append({
            "id": f"{document_id}-{field_name}",
            "case_id": case_id,
            "document_id": document_id,
            "file_name": document["file_name"],
            "document_type": document_type,
            "field_name": field_name,
            "field_value": "" if value is None else str(value),
            "field_value_number": numeric,
            "evidence": item.get("evidence") or "",
            "extraction_mode": document.get("extraction_mode", "mock_deterministic_pdf_text"),
            "synthetic_only": True,
        })
    return records


def main() -> int:
    load_dotenv(APP_DIR / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=CASE_IDS,
        action="append",
        help="Index only this case (repeat the option to select multiple cases). Default: all six.",
    )
    args = parser.parse_args()
    case_ids = args.case or CASE_IDS

    endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
    if not endpoint:
        parser.error("AZURE_SEARCH_ENDPOINT is required (set it in the environment or app .env file).")

    global INDEX_NAME
    INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "mortgage-document-evidence")
    credential = DefaultAzureCredential()
    index_client = SearchIndexClient(endpoint=endpoint.rstrip("/"), credential=credential)
    search_client = SearchClient(
        endpoint=endpoint.rstrip("/"),
        index_name=INDEX_NAME,
        credential=credential,
    )

    try:
        all_records: list[dict[str, Any]] = []
        expected_by_case: dict[str, int] = {}
        for case_id in case_ids:
            case_folder = SAMPLE_ROOT / case_id
            if not case_folder.is_dir():
                raise FileNotFoundError(f"Missing case folder: {case_folder}")
            pdfs = sorted(case_folder.glob("*.pdf"))
            if not pdfs:
                raise FileNotFoundError(f"No PDFs found in {case_folder}")

            extracted = extract_case_folder(case_folder)
            extracted_names = {item["file_name"] for item in extracted}
            if extracted_names != {pdf.name for pdf in pdfs}:
                raise RuntimeError(f"Not every PDF was extracted for {case_id}")

            case_records = [
                record
                for document in extracted
                for record in _index_record(case_id, document)
            ]
            if not case_records:
                raise ValueError(f"No indexable fields extracted for {case_id}")
            expected_by_case[case_id] = len(case_records)
            all_records.extend(case_records)
            print(f"{case_id}: extracted {len(pdfs)} PDFs into {len(case_records)} field records")

        index_client.create_or_update_index(build_index())

        # Remove old records for the selected synthetic cases first. This prevents
        # fields from a previous run (for example, a now-missing document) surviving.
        existing_ids: list[str] = []
        for case_id in case_ids:
            existing_ids.extend(
                item["id"]
                for item in search_client.search(
                    search_text="*",
                    filter=f"case_id eq '{case_id}'",
                    select=["id"],
                    top=1000,
                )
            )
        for offset in range(0, len(existing_ids), 500):
            deleted = search_client.delete_documents(
                documents=[{"id": item_id} for item_id in existing_ids[offset:offset + 500]]
            )
            failures = [result for result in deleted if not result.succeeded]
            if failures:
                details = "; ".join(f"{item.key}: {item.error_message}" for item in failures[:10])
                raise RuntimeError(f"Azure AI Search failed to remove {len(failures)} stale records: {details}")
        if existing_ids:
            print(f"Removed {len(existing_ids)} stale records for selected cases before reindexing.")

        batch_size = 500
        for offset in range(0, len(all_records), batch_size):
            results = search_client.upload_documents(documents=all_records[offset:offset + batch_size])
            failures = [result for result in results if not result.succeeded]
            if failures:
                details = "; ".join(f"{item.key}: {item.error_message}" for item in failures[:10])
                raise RuntimeError(f"Azure AI Search rejected {len(failures)} records: {details}")

        # Search indexing is near-real-time. Retry briefly before asserting visibility.
        for case_id, expected_count in expected_by_case.items():
            visible = []
            for attempt in range(5):
                visible = list(search_client.search(
                    search_text="*",
                    filter=f"case_id eq '{case_id}'",
                    select=["id", "case_id", "document_id", "field_name"],
                    top=1000,
                ))
                if len(visible) >= expected_count:
                    break
                time.sleep(2)
            if len(visible) < expected_count:
                raise RuntimeError(
                    f"{case_id}: uploaded {expected_count} records but only "
                    f"{len(visible)} are queryable. Check index configuration and permissions."
                )
            print(f"{case_id}: verified {len(visible)} searchable field records")

        print(
            f"Successfully indexed and verified {len(all_records)} PDF-derived field records "
            f"for {len(expected_by_case)} case(s) in '{INDEX_NAME}'."
        )
        return 0
    finally:
        credential.close()


if __name__ == "__main__":
    raise SystemExit(main())
