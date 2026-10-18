# Copyright (c) Microsoft. All rights reserved.
"""Synthetic mortgage-document validation and credit memo agent; no real banking actions."""
from __future__ import annotations

import json
import os
import logging
from azure.search.documents import SearchClient
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send
from typing_extensions import TypedDict
from langchain_azure_ai.agents.hosting import FoundryCheckpointSaver, ResponsesHostServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
_SCOPE = "https://cognitiveservices.azure.com/.default"
_LOCALE = ContextVar("request_locale", default="en-CA")
_CASE = "MORT-1001"
_OUT = Path(os.environ.get("CREDIT_MEMO_OUTPUT_DIR", "./artifacts")).resolve()
_MOCK_FILENET = _OUT / "mock-filenet"

_STORAGE_ACCOUNT_URL = "https://stdemo131233b.blob.core.windows.net"
_CREDIT_MEMO_CONTAINER = "mortgage-credit-memos"



class AssistantState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    locale: str


# Synthetic-only sample data. Replace only with approved test data in later phases.
APPLICATIONS: dict[str, dict[str, Any]] = {
    _CASE: {
        "case_id": _CASE, "applicant_name": "SYNTHETIC APPLICANT",
        "application_type": "Residential mortgage", "requested_mortgage_cad": 525000,
        "property_value_cad": 700000, "declared_annual_income_cad": 120000,
        "declared_monthly_liabilities_cad": 850,
        "documents": [
            {"document_id": "DOC-PAY-1001", "document_type": "Pay stub",
             "file_name": "synthetic_pay_stub.txt", "source": "Mock FileNet"},
            {"document_id": "DOC-EMP-1001", "document_type": "Employment letter",
             "file_name": "synthetic_employment_letter.txt", "source": "Mock FileNet"},
        ],
    }
}
DOCUMENT_FIELDS: dict[str, dict[str, Any]] = {
    "DOC-PAY-1001": {"document_type": "Pay stub", "employer": "Example North Ltd. (synthetic)",
                     "annual_income_cad": 108000, "pay_frequency": "biweekly",
                     "employment_status": "employed",
                     "evidence": "Synthetic pay stub: annualized income is CAD 108,000."},
    "DOC-EMP-1001": {"document_type": "Employment letter", "employer": "Example North Ltd. (synthetic)",
                     "annual_income_cad": 108000, "employment_status": "permanent full-time",
                     "evidence": "Synthetic employment letter: annual salary is CAD 108,000."},
}
INTERNAL_DATA: dict[str, dict[str, Any]] = {
    _CASE: {"customer_reference": "SYN-CUST-1001", "annual_income_cad": 120000,
            "monthly_liabilities_cad": 1450, "employment_status": "employed",
            "source_system": "Mock Internal Customer API"}
}
MOCK_FILENET_RECORDS: dict[str, dict[str, Any]] = {}


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _build_search_client() -> SearchClient:
    endpoint = os.environ["AZURE_SEARCH_ENDPOINT"].rstrip("/")
    index_name = os.environ.get(
        "AZURE_SEARCH_INDEX_NAME",
        "mortgage-document-evidence",
    )

    return SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=DefaultAzureCredential(),
    )


@tool
def search_mortgage_evidence(
    case_id: Annotated[str, "Synthetic mortgage case ID; use MORT-1001."]
) -> str:
    """Retrieve extracted mortgage document fields and evidence from Azure AI Search."""
    case_id = case_id.strip().upper()

    try:
        client = _build_search_client()
        results = client.search(
            search_text="*",
            filter=f"case_id eq '{case_id}'",
            select=[
                "case_id",
                "document_id",
                "file_name",
                "document_type",
                "field_name",
                "field_value",
                "field_value_number",
                "evidence",
                "extraction_mode",
                "synthetic_only",
            ],
            top=100,
        )

        records = [dict(record) for record in results]

        return _json({
            "case_id": case_id,
            "source": "Azure AI Search",
            "index_name": os.environ.get(
                "AZURE_SEARCH_INDEX_NAME",
                "mortgage-document-evidence",
            ),
            "record_count": len(records),
            "records": records,
        })
    except Exception:
        logger.exception("Azure AI Search retrieval failed for case %s", case_id)
        return _json({
            "case_id": case_id,
            "error": "Search retrieval failed. Check identity, RBAC, endpoint, and index configuration.",
        })



def _comparison(case_id: str) -> dict[str, Any]:
    case_id = case_id.strip().upper()

    app = APPLICATIONS.get(case_id)
    internal = INTERNAL_DATA.get(case_id)
    if not app or not internal:
        raise KeyError(f"No synthetic application and internal record for {case_id}")

    # Retrieve authoritative extracted fields from Azure AI Search.
    client = _build_search_client()
    results = list(client.search(
        search_text="*",
        filter=f"case_id eq '{case_id}'",
        select=[
            "case_id", "document_id", "document_type",
            "field_name", "field_value", "field_value_number",
            "evidence", "synthetic_only",
        ],
        top=100,
    ))

    if not results:
        raise ValueError(
            f"No indexed evidence found for {case_id}; comparison cannot proceed."
        )

    records = [dict(item) for item in results]

    if any(
        record.get("synthetic_only") is not True
        for record in records
    ):
        raise ValueError("Unexpected non-synthetic evidence in the POC index.")

    # Organize the indexed records by document and field.
    indexed = {}
    for record in records:
        document_id = record["document_id"]
        field_name = record["field_name"]
        indexed.setdefault(document_id, {})[field_name] = record

    application = indexed.get("DOC-APP-1001", {})
    pay_stub = indexed.get("DOC-PAY-1001", {})
    employment_letter = indexed.get("DOC-EMP-1001", {})

    def value(document: dict, field_name: str):
        record = document.get(field_name)
        if record is None:
            return None
        numeric = record.get("field_value_number")
        if numeric is not None:
            return numeric
        return record.get("field_value")

    def evidence(document: dict, field_name: str) -> str:
        record = document.get(field_name)
        if record is None:
            return "Not present in Azure AI Search."
        return (
            f"{record.get('document_id')}: "
            f"{record.get('evidence') or record.get('field_value')}"
        )

    declared_income = value(application, "annual_income_cad")
    pay_income = value(pay_stub, "annual_income_cad")
    letter_income = value(employment_letter, "annual_income_cad")
    internal_income = internal["annual_income_cad"]

    declared_liabilities = value(application, "monthly_liabilities_cad")
    internal_liabilities = internal["monthly_liabilities_cad"]

    app_employment = value(application, "employment_status")
    pay_employment = value(pay_stub, "employment_status")
    letter_employment = value(employment_letter, "employment_status")
    internal_employment = internal["employment_status"]

    income_values = [
        declared_income, pay_income, letter_income, internal_income
    ]
    income_complete = all(v is not None for v in income_values)
    income_status = (
        "MATCH"
        if income_complete and len(set(income_values)) == 1
        else "DISCREPANCY"
    )

    liability_status = (
        "MATCH"
        if declared_liabilities is not None
        and declared_liabilities == internal_liabilities
        else "DISCREPANCY"
    )

    employment_values = [
        app_employment, pay_employment,
        letter_employment, internal_employment,
    ]
    employment_status = (
        "REVIEW_REQUIRED"
        if any(v is None for v in employment_values)
        or len(set(employment_values)) > 1
        else "MATCH"
    )

    checks = [
        {
            "field": "annual_income_cad",
            "application_value": declared_income,
            "document_values": [pay_income, letter_income],
            "internal_api_value": internal_income,
            "status": income_status,
            "evidence": "; ".join([
                evidence(application, "annual_income_cad"),
                evidence(pay_stub, "annual_income_cad"),
                evidence(employment_letter, "annual_income_cad"),
                "Internal value from Mock Internal Customer API.",
            ]),
        },
        {
            "field": "monthly_liabilities_cad",
            "application_value": declared_liabilities,
            "document_values": [],
            "internal_api_value": internal_liabilities,
            "status": liability_status,
            "evidence": "; ".join([
                evidence(application, "monthly_liabilities_cad"),
                "Internal value from Mock Internal Customer API. "
                "Supporting liability documents are not included.",
            ]),
        },
        {
            "field": "employment_status",
            "application_value": app_employment,
            "document_values": [pay_employment, letter_employment],
            "internal_api_value": internal_employment,
            "status": employment_status,
            "evidence": "; ".join([
                evidence(application, "employment_status"),
                evidence(pay_stub, "employment_status"),
                evidence(employment_letter, "employment_status"),
                "Internal value from Mock Internal Customer API.",
            ]),
        },
    ]

    issues = [row for row in checks if row["status"] != "MATCH"]

    return {
        "case_id": case_id,
        "comparison_mode": "DETERMINISTIC_RULES_WITH_AZURE_AI_SEARCH",
        "evidence_source": "Azure AI Search",
        "indexed_record_count": len(records),
        "checks": checks,
        "discrepancy_count": len(issues),
        "discrepancies": issues,
        "overall_status": (
            "REVIEW_REQUIRED" if issues else "NO_DISCREPANCY_DETECTED"
        ),
        "human_review_required": True,
        "decision_boundary": (
            "This comparison does not approve or decline a mortgage."
        ),
    }


@tool
def get_mortgage_application(case_id: Annotated[str, "Synthetic case ID; use MORT-1001."]) -> str:
    """Get the synthetic mortgage application and document inventory."""
    record = APPLICATIONS.get(case_id.strip().upper())
    return _json(record if record else {"error": "Synthetic application not found.",
                                        "available_case_ids": list(APPLICATIONS)})


@tool
def extract_document_fields(case_id: Annotated[str, "Synthetic case ID; use MORT-1001."]) -> str:
    """Return pre-seeded synthetic OCR/extraction results; no real OCR service is called."""
    case_id = case_id.strip().upper()
    app = APPLICATIONS.get(case_id)
    if not app:
        return _json({"error": "Synthetic application not found.", "case_id": case_id})
    docs = []
    for doc in app["documents"]:
        fields = DOCUMENT_FIELDS.get(doc["document_id"])
        docs.append({**doc, "extraction_status": "SUCCESS" if fields else "NO_RESULT",
                     "extracted_fields": fields or {}})
    return _json({"case_id": case_id, "mode": "SYNTHETIC_OCR_RESULTS", "documents": docs})


@tool
def get_internal_customer_data(case_id: Annotated[str, "Synthetic case ID; use MORT-1001."]) -> str:
    """Get synthetic customer values from a mock internal operational API."""
    record = INTERNAL_DATA.get(case_id.strip().upper())
    return _json({"case_id": case_id.strip().upper(), **record, "mode": "MOCK_API"}
                  if record else {"error": "Mock internal record not found.", "case_id": case_id})


@tool
def compare_mortgage_data(case_id: Annotated[str, "Synthetic case ID; use MORT-1001."]) -> str:
    """Compare application, extracted document, and internal API values deterministically."""
    try:
        return _json(_comparison(case_id))
    except KeyError as exc:
        return _json({"error": str(exc), "case_id": case_id})


def _create_pdf(case_id: str, app: dict[str, Any], comparison: dict[str, Any]) -> Path:
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / f"{case_id.lower()}-credit-memo.pdf"
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Mortgage Document Validation — Credit Memo", styles["Title"]),
        Spacer(1, 0.12 * inch),
        Paragraph("<b>POC DATA — SYNTHETIC ONLY — NOT A CREDIT DECISION</b>", styles["Heading2"]),
        Paragraph(f"<b>Case:</b> {case_id}", styles["BodyText"]),
        Paragraph(f"<b>Applicant:</b> {app['applicant_name']}", styles["BodyText"]),
        Paragraph(f"<b>Generated (UTC):</b> {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}", styles["BodyText"]),
        Spacer(1, 0.12 * inch), Paragraph("Application summary", styles["Heading2"]),
        Paragraph(f"Requested mortgage: CAD {app['requested_mortgage_cad']:,}", styles["BodyText"]),
        Paragraph(f"Property value: CAD {app['property_value_cad']:,}", styles["BodyText"]),
        Paragraph(f"Declared annual income: CAD {app['declared_annual_income_cad']:,}", styles["BodyText"]),
        Paragraph(f"Declared monthly liabilities: CAD {app['declared_monthly_liabilities_cad']:,}", styles["BodyText"]),
        Spacer(1, 0.12 * inch), Paragraph("Validation results", styles["Heading2"]),
    ]
    
    # Use Paragraph cells so long values wrap within their columns.
    from xml.sax.saxutils import escape

    header_style = ParagraphStyle(
        "MemoTableHeader",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
        spaceAfter=0,
    )
    cell_style = ParagraphStyle(
        "MemoTableCell",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        spaceAfter=0,
        wordWrap="CJK",
    )

    def cell(value: Any, style: ParagraphStyle = cell_style) -> Paragraph:
        # Escape source values so characters such as < and & render safely.
        return Paragraph(escape(str(value)), style)

    headers = [
        "Field",
        "Application",
        "Document value(s)",
        "Internal API",
        "Status",
    ]
    rows = [[cell(value, header_style) for value in headers]]

    for item in comparison["checks"]:
        rows.append([
            cell(item["field"]),
            cell(item["application_value"]),
            cell(", ".join(map(str, item["document_values"])) or "Not supplied"),
            cell(item["internal_api_value"]),
            cell(item["status"]),
        ])

    # Letter page width minus the existing 0.55-inch left/right margins = 7.4 inches.
    table = Table(
        rows,
        repeatRows=1,
        colWidths=[1.35*inch, 1.15*inch, 1.85*inch, 1.25*inch, 1.8*inch],
        hAlign="LEFT",
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B53")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F1F5F9")]),
    ]))

    story.extend([table, Spacer(1, 0.12*inch), Paragraph("Discrepancies and evidence", styles["Heading2"])])
    for item in comparison["discrepancies"]:
        text = (f"<b>{item['field']} — {item['status']}</b><br/>"
                f"Application: {item['application_value']}<br/>"
                f"Document value(s): {', '.join(map(str, item['document_values'])) or 'Not supplied'}<br/>"
                f"Internal API: {item['internal_api_value']}<br/>Evidence: {item['evidence']}")
        story.extend([Paragraph(text, styles["BodyText"]), Spacer(1, 0.08*inch)])
    story.extend([
        Paragraph("Human adjudication required", styles["Heading2"]),
        Paragraph("Mortgage adjudication staff must review the source documents, internal data, and flagged differences. "
                  "This memo is an evidence summary only and does not make a final mortgage decision.", styles["BodyText"]),
        Spacer(1, 0.1*inch),
        Paragraph("POC limitation: OCR, operational data, and FileNet storage are simulated.", styles["Italic"]),
    ])
    SimpleDocTemplate(str(path), pagesize=letter, rightMargin=.55*inch, leftMargin=.55*inch,
                      topMargin=.55*inch, bottomMargin=.55*inch).build(story)
    return path


@tool
def generate_credit_memo(case_id: Annotated[str, "Synthetic case ID; use MORT-1001."]) -> str:
    """Generate a PDF credit memo from deterministic synthetic validation results."""
    case_id = case_id.strip().upper()
    app = APPLICATIONS.get(case_id)
    if not app:
        return _json({"error": "Synthetic application not found.", "case_id": case_id})
    try:
        comparison = _comparison(case_id)
    except KeyError as exc:
        return _json({"error": str(exc), "case_id": case_id})
    path = _create_pdf(case_id, app, comparison)
    return _json({"case_id": case_id, "status": "PDF_GENERATED", "file_path": str(path),
                  "file_name": path.name, "synthetic_data_only": True, "human_adjudication_required": True})



@tool
def save_credit_memo(case_id: Annotated[str, "Synthetic case ID; use MORT-1001."]) -> str:
    """Upload the generated PDF and metadata to Azure Blob Storage using managed identity."""
    from azure.storage.blob import BlobServiceClient, ContentSettings

    case_id = case_id.strip().upper()
    source = _OUT / f"{case_id.lower()}-credit-memo.pdf"

    if not source.exists():
        return _json({
            "error": "PDF not found; call generate_credit_memo first.",
            "case_id": case_id,
        })

    credential = DefaultAzureCredential()
    service = BlobServiceClient(
        account_url=_STORAGE_ACCOUNT_URL,
        credential=credential,
    )
    container = service.get_container_client(_CREDIT_MEMO_CONTAINER)

    prefix = case_id
    pdf_blob_name = f"{prefix}/{source.name}"
    metadata_blob_name = f"{prefix}/{case_id.lower()}-record.json"

    try:
        with source.open("rb") as pdf_file:
            container.upload_blob(
                name=pdf_blob_name,
                data=pdf_file,
                overwrite=True,
                content_settings=ContentSettings(content_type="application/pdf"),
            )

        record = {
            "case_id": case_id,
            "status": "BLOB_UPLOAD_SUCCEEDED",
            "document_id": f"CREDIT-MEMO-{case_id}",
            "pdf_blob": pdf_blob_name,
            "metadata_blob": metadata_blob_name,
            "storage_account": "stdemo131233b",
            "container": _CREDIT_MEMO_CONTAINER,
            "synthetic_data_only": True,
            "human_adjudication_required": True,
            "real_filenet_called": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        container.upload_blob(
            name=metadata_blob_name,
            data=json.dumps(record, indent=2, ensure_ascii=False).encode("utf-8"),
            overwrite=True,
            content_settings=ContentSettings(content_type="application/json"),
        )

        # Preserve the existing local mock record for compatibility.
        _MOCK_FILENET.mkdir(parents=True, exist_ok=True)
        record_path = _MOCK_FILENET / f"{case_id.lower()}-record.json"
        record_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        MOCK_FILENET_RECORDS[case_id] = record

        return _json(record)

    except Exception:
        logger.exception("Credit memo Blob upload failed for case %s", case_id)
        return _json({
            "error": "Credit memo upload to Blob Storage failed.",
            "case_id": case_id,
        })
    finally:
        service.close()
        credential.close()


@tool
def get_mock_filenet_record(
    case_id: Annotated[str, "Synthetic case ID; use MORT-1001."]
) -> str:
    """Retrieve the persisted mock FileNet record from local JSON metadata."""
    case_id = case_id.strip().upper()
    record_path = _MOCK_FILENET / f"{case_id.lower()}-record.json"

    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        MOCK_FILENET_RECORDS[case_id] = record
        return _json(record)

    return _json({
        "case_id": case_id,
        "status": "NOT_FOUND",
        "real_filenet_called": False,
    })

def _build_chat_model() -> ChatOpenAI:
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    deployment = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "").strip()

    if not deployment:
        raise RuntimeError(
            "Set AZURE_AI_MODEL_DEPLOYMENT_NAME to an Azure OpenAI deployment."
        )

    credential = DefaultAzureCredential()
    # Diagnostic: inspect the identity in the Azure access token.
    token = credential.get_token(_SCOPE)
    import base64
    import json

    token_parts = token.token.split(".")
    if len(token_parts) >= 2:
        payload = token_parts[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))

        logger.info(
            "Azure OpenAI authentication identity: oid=%s, appid=%s, "
            "identity_type=%s, audience=%s",
            claims.get("oid"),
            claims.get("appid") or claims.get("azp"),
            claims.get("idtyp"),
            claims.get("aud"),
        )
    token_provider = get_bearer_token_provider(credential, _SCOPE)

    return ChatOpenAI(
        model=deployment,
        base_url=f"{endpoint}/openai/v1/",
        api_key=token_provider,
        use_responses_api=True,
        output_version="responses/v1",
    )


def _system_prompt(state: AssistantState) -> str:
    return (
        f"You are a mortgage document validation POC assistant. Locale: {state.get('locale', 'en-CA')}. "
        "All records and integrations are synthetic/mocked. For a validation request, use tools as applicable: "
        "get_mortgage_application, extract_document_fields, get_internal_customer_data, search_mortgage_evidence, compare_mortgage_data, "
        "generate_credit_memo, save_credit_memo, get_mock_filenet_record. Do not claim tools ran unless results "
        "confirm it. Deterministic comparison results are authoritative; do not alter them. Explain discrepancies "
        "and evidence. Always say human adjudication is required. Never approve or decline a mortgage. Default case "
        "is MORT-1001. When asked to generate a memo, generate and simulate saving it unless asked not to."
    )


def _build_graph(model: ChatOpenAI):
    tools = [get_mortgage_application, extract_document_fields, get_internal_customer_data, search_mortgage_evidence,
             compare_mortgage_data, generate_credit_memo, save_credit_memo, get_mock_filenet_record]
    tool_model = model.bind_tools(tools)

    async def assistant(state: AssistantState) -> dict[str, list[BaseMessage]]:
        response = await tool_model.ainvoke([SystemMessage(content=_system_prompt(state)), *state["messages"]])
        return {"messages": [response]}

    def route(state: AssistantState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    graph = StateGraph(AssistantState)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", route)
    graph.add_edge("tools", "assistant")
    return graph.compile(checkpointer=FoundryCheckpointSaver())


class _LocaleMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        token = _LOCALE.set(headers.get("x-client-user-locale", "en-CA"))
        try:
            await self.app(scope, receive, send)
        finally:
            _LOCALE.reset(token)


class MortgageResponsesHostServer(ResponsesHostServer):
    """Responses host preserving the template's hosted-agent protocol."""
    def __init__(self, graph: Any) -> None:
        super().__init__(graph)
        self.app.add_middleware(_LocaleMiddleware)


def main() -> None:
    host = MortgageResponsesHostServer(_build_graph(_build_chat_model()))
    host.run(port=int(os.environ.get("PORT", "8088")))


if __name__ == "__main__":
    main()
