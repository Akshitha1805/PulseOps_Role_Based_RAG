
"""
PulseOps PDF ingestion.

Uses the existing authenticated website Admin session.
No terminal email/password login is required.

Safety:
- Dry-run is enabled by default.
- Dry-run never generates embeddings or inserts records.
- Content comparison uses .strip(), matching the original
  prepare_sql_ingestion.py behavior.
- Page number is excluded from comparison.
- Repeated chunks within the prepared PDFs are counted.
- Potential chunking differences are reported for review.
"""

from pathlib import Path
from typing import Any
from collections import defaultdict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import embeddings, verify_user_token


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


# ============================================================
# ROLE-SPECIFIC PDF DOCUMENTS
# ============================================================

ROLE_DOCUMENTS = {
    "Admin": {
        "path": "Admin/Admin_Knowledge_Guide.pdf",
        "role": "Admin",
    },
    "Inventory Manager": {
        "path": "Inventory_manager/Inventory_Manager_Knowledge.pdf",
        "role": "Inventory Manager",
    },
    "Procurement Manager": {
        "path": "Procurement_manager/Procurement_Manager_Knowledge.pdf",
        "role": "Procurement Manager",
    },
    "Supplier Manager": {
        "path": "Supplier_manager/Supplier_Manager_Knowledge.pdf",
        "role": "Supplier Manager",
    },
    "Delivery & Operations Manager": {
        "path": (
            "Delivery_operations_manager/"
            "Delivery_operations_Knowledge.pdf"
        ),
        "role": "Delivery & Operations Manager",
    },
    "Business Performance Manager": {
        "path": (
            "Business_performance_manager/"
            "Business_performance_Manager_Knowledge.pdf"
        ),
        "role": "Business Performance Manager",
    },
}

COMPANY_POLICY_PATH = (
    DATA_DIR / "Company_policy" / "Company_Policy.pdf"
)


# ============================================================
# TEXT SPLITTER
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)


# ============================================================
# PDF LOADING
# ============================================================

def load_pdf(pdf_path: Path):
    """Load all pages from a PDF."""

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    return PyPDFLoader(str(pdf_path)).load()


# ============================================================
# DOCUMENT PREPARATION
# ============================================================

def prepare_documents():
    """Load and split all six role PDFs and shared policy PDF."""

    all_chunks = []

    for role, details in ROLE_DOCUMENTS.items():

        pdf_path = DATA_DIR / details["path"]

        print(f"Loading {role} PDF...")

        pages = load_pdf(pdf_path)

        for page in pages:
            page.metadata.update({
                "role": details["role"],
                "access": "ROLE",
                "source": pdf_path.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
            })

        chunks = text_splitter.split_documents(pages)

        print(f"{role}: {len(chunks)} chunks")

        all_chunks.extend(chunks)

    print("Loading Company Policy PDF...")

    policy_pages = load_pdf(COMPANY_POLICY_PATH)

    for page in policy_pages:
        page.metadata.update({
            "access": "ALL",
            "source": COMPANY_POLICY_PATH.relative_to(
                PROJECT_ROOT
            ).as_posix(),
        })

    policy_chunks = text_splitter.split_documents(
        policy_pages
    )

    print(f"Company Policy: {len(policy_chunks)} chunks")

    all_chunks.extend(policy_chunks)

    if not all_chunks:
        raise ValueError(
            "No document chunks were generated."
        )

    print(f"Total chunks prepared: {len(all_chunks)}")

    return all_chunks


# ============================================================
# ADMIN AUTHENTICATION
# ============================================================

def get_verified_admin_client(access_token: str) -> Any:
    """
    Verify the existing website user's JWT and Admin permission.

    The role is verified by Supabase, not supplied by the frontend.
    """

    if not access_token:
        raise PermissionError(
            "An authenticated Admin session is required."
        )

    client, auth_user = verify_user_token(access_token)

    verification = client.rpc(
        "is_approved_admin",
        {},
    ).execute()

    if verification.data is not True:
        raise PermissionError(
            "Access denied. An approved Admin account is required."
        )

    print(f"Approved Admin verified: {auth_user.id}")

    return client


# ============================================================
# NORMALIZED DOCUMENT COMPARISON
# ============================================================


def normalize_content(content: str) -> str:
    """
    Normalize line endings before comparing document chunks.
    This treats Windows CRLF and Unix LF as equivalent.
    """
    return (
        (content or "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )



def make_document_key(content: str, metadata: dict) -> tuple:
    """
    Identify a document chunk using:
    source + access + role + normalized content.

    Page number is intentionally excluded.
    """

    metadata = metadata or {}

    return (
        metadata.get("source"),
        metadata.get("access"),
        metadata.get("role"),
        normalize_content(content),
    )


# ============================================================
# EXISTING SUPABASE DOCUMENT INVENTORY
# ============================================================

def get_existing_document_inventory(client: Any):
    """
    Read existing documents from Supabase using pagination.

    Returns:
        existing_keys: set of normalized document keys
        existing_by_source: record counts by PDF source
        existing_rows: total database rows read
    """

    existing_keys = set()
    existing_by_source = defaultdict(int)

    offset = 0
    page_size = 500
    existing_rows = 0

    print("\nReading existing Supabase document chunks...")

    while True:

        response = (
            client.table("documents")
            .select("content, metadata")
            .range(
                offset,
                offset + page_size - 1,
            )
            .execute()
        )

        rows = response.data or []

        existing_rows += len(rows)

        for row in rows:

            metadata = row.get("metadata") or {}

            key = make_document_key(
                content=row.get("content", ""),
                metadata=metadata,
            )

            existing_keys.add(key)

            source = metadata.get(
                "source",
                "Unknown source",
            )

            existing_by_source[source] += 1

        if len(rows) < page_size:
            break

        offset += page_size

    print(f"Existing records read: {existing_rows}")
    print(f"Existing unique comparison keys: {len(existing_keys)}")

    return (
        existing_keys,
        dict(existing_by_source),
        existing_rows,
    )


# ============================================================
# DOCUMENT COMPARISON
# ============================================================

def compare_documents(client: Any, documents: list):
    """
    Compare prepared PDF chunks with existing Supabase chunks.

    This function does not generate embeddings or insert records.
    """

    (
        existing_keys,
        existing_by_source,
        existing_rows,
    ) = get_existing_document_inventory(client)

    prepared_by_source = defaultdict(int)
    exact_matches_by_source = defaultdict(int)

    new_documents = []
    seen_prepared_keys = set()

    exact_matches = 0
    duplicate_in_prepared = 0

    for document in documents:

        metadata = document.metadata or {}

        source = metadata.get(
            "source",
            "Unknown source",
        )

        prepared_by_source[source] += 1

        key = make_document_key(
            content=document.page_content,
            metadata=metadata,
        )

        # Exact match in Supabase.
        if key in existing_keys:

            exact_matches += 1
            exact_matches_by_source[source] += 1

            continue

        # Duplicate among newly prepared chunks.
        if key in seen_prepared_keys:

            duplicate_in_prepared += 1

            continue

        seen_prepared_keys.add(key)
        new_documents.append(document)

    # Generate a source-by-source comparison report.
    source_report = []

    for source, prepared_count in prepared_by_source.items():

        existing_count = existing_by_source.get(
            source,
            0,
        )

        matched_count = exact_matches_by_source.get(
            source,
            0,
        )

        unmatched_count = prepared_count - matched_count

        source_report.append({
            "source": source,
            "prepared_chunks": prepared_count,
            "existing_chunks": existing_count,
            "exact_matches": matched_count,
            "unmatched_prepared_chunks": unmatched_count,
        })

    # Existing source records with unmatched prepared chunks
    # require further review. They are not confirmed duplicates.
    potential_chunking_differences = [
        item
        for item in source_report
        if (
            item["existing_chunks"] > 0
            and item["unmatched_prepared_chunks"] > 0
        )
    ]

    return {
        "prepared_chunks": len(documents),
        "existing_chunks": existing_rows,
        "existing_unique_keys": len(existing_keys),
        "exact_matches": exact_matches,
        "new_unique_chunks": len(new_documents),
        "duplicate_in_prepared": duplicate_in_prepared,
        "source_report": source_report,
        "potential_chunking_differences": (
            potential_chunking_differences
        ),
        "new_documents": new_documents,
    }


# ============================================================
# EMBEDDING AND DATABASE INSERTION
# ============================================================

def upload_documents(
    client: Any,
    documents: list,
    batch_size: int = 50,
    dry_run: bool = True,
):
    """
    Compare documents and optionally upload unmatched chunks.

    dry_run=True:
        Comparison only. No embeddings or database inserts.

    dry_run=False:
        Generates embeddings and inserts unmatched chunks.
    """

    report = compare_documents(
        client=client,
        documents=documents,
    )

    print("\n========== INGESTION COMPARISON ==========")

    print(f"Prepared chunks: {report['prepared_chunks']}")
    print(f"Existing Supabase records: {report['existing_chunks']}")
    print(f"Unique existing keys: {report['existing_unique_keys']}")
    print(f"Exact matches: {report['exact_matches']}")
    print(f"New unique chunks: {report['new_unique_chunks']}")
    print(
        "Repeated chunks within PDFs: "
        f"{report['duplicate_in_prepared']}"
    )

    print("\n========== SOURCE COMPARISON ==========")

    for item in report["source_report"]:

        print(
            f"{item['source']}: "
            f"prepared={item['prepared_chunks']}, "
            f"existing={item['existing_chunks']}, "
            f"exact_matches={item['exact_matches']}, "
            f"unmatched={item['unmatched_prepared_chunks']}"
        )

    print("\n========== CHUNKING REVIEW ==========")

    if report["potential_chunking_differences"]:

        for item in report["potential_chunking_differences"]:

            print(
                f"Review required: {item['source']} "
                f"({item['unmatched_prepared_chunks']} "
                f"unmatched prepared chunks)"
            )

        print(
            "\nWARNING: Unmatched chunks are not confirmed "
            "duplicates or missing documents."
        )

    else:

        print("No source-level chunking differences detected.")

    # Safety stop: dry-run never embeds or inserts.
    if dry_run:

        print("\n========== DRY RUN COMPLETE ==========")
        print("No embeddings were generated.")
        print("No Supabase records were inserted.")

        return {
            "dry_run": True,
            "uploaded": 0,
            "would_upload": report["new_unique_chunks"],
            "skipped_exact": report["exact_matches"],
            "duplicate_in_prepared": (
                report["duplicate_in_prepared"]
            ),
            "prepared_chunks": report["prepared_chunks"],
            "existing_chunks": report["existing_chunks"],
            "source_report": report["source_report"],
            "potential_chunking_differences": (
                report["potential_chunking_differences"]
            ),
        }

    # Actual insertion is possible only when dry_run=False.
    new_documents = report["new_documents"]

    total_uploaded = 0

    for start in range(
        0,
        len(new_documents),
        batch_size,
    ):

        batch = new_documents[
            start:start + batch_size
        ]

        # Store the same normalized content used in comparison.
        texts = [
            normalize_content(document.page_content)
            for document in batch
        ]

        vectors = embeddings.embed_documents(texts)

        if len(vectors) != len(batch):
            raise RuntimeError(
                "Embedding count does not match document count."
            )

        records = []

        for document, content, vector in zip(
            batch,
            texts,
            vectors,
        ):

            if len(vector) != 384:
                raise ValueError(
                    "Invalid embedding dimension. Expected 384."
                )

            vector_value = "[" + ",".join(
                str(float(value))
                for value in vector
            ) + "]"

            records.append({
                "content": content,
                "metadata": document.metadata,
                "embedding": vector_value,
            })

        response = (
            client.table("documents")
            .insert(records)
            .execute()
        )

        inserted_rows = response.data or []

        if len(inserted_rows) != len(records):
            raise RuntimeError(
                "Supabase did not confirm all document insertions."
            )

        total_uploaded += len(inserted_rows)

        print(
            f"Uploaded {total_uploaded}/"
            f"{len(new_documents)} new chunks..."
        )

    print("\nActual ingestion completed.")
    print(f"Newly uploaded: {total_uploaded}")
    print(f"Exact matches skipped: {report['exact_matches']}")

    return {
        "dry_run": False,
        "uploaded": total_uploaded,
        "would_upload": len(new_documents),
        "skipped_exact": report["exact_matches"],
        "duplicate_in_prepared": (
            report["duplicate_in_prepared"]
        ),
        "prepared_chunks": report["prepared_chunks"],
        "existing_chunks": report["existing_chunks"],
        "source_report": report["source_report"],
        "potential_chunking_differences": (
            report["potential_chunking_differences"]
        ),
    }


# ============================================================
# MAIN INGESTION FUNCTION
# ============================================================

def ingest_documents(
    access_token: str,
    dry_run: bool = True,
):
    """
    Process all seven PDFs.

    Dry-run is deliberately the default.
    The existing FastAPI endpoint can continue calling
    ingest_documents(access_token) safely.
    """

    print("\nStarting PulseOps PDF ingestion...")

    # 1. Verify existing website Admin session.
    admin_client = get_verified_admin_client(
        access_token
    )

    print("Approved Admin verification successful.")

    # 2. Load and split all seven PDFs.
    all_chunks = prepare_documents()

    # 3. Compare and optionally upload.
    upload_result = upload_documents(
        client=admin_client,
        documents=all_chunks,
        dry_run=dry_run,
    )

    print("\nPDF ingestion process completed.")

    return {
        "status": (
            "dry_run_complete"
            if dry_run
            else "success"
        ),
        "prepared_chunks": upload_result["prepared_chunks"],
        "existing_chunks": upload_result["existing_chunks"],
        "uploaded_chunks": upload_result["uploaded"],
        "would_upload_chunks": upload_result["would_upload"],
        "skipped_exact_chunks": upload_result["skipped_exact"],
        "duplicate_in_prepared_chunks": (
            upload_result["duplicate_in_prepared"]
        ),
        "source_report": upload_result["source_report"],
        "potential_chunking_differences": (
            upload_result["potential_chunking_differences"]
        ),
    }
