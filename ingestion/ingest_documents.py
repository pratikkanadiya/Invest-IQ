import os
from pathlib import Path
import json

from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg import connect

from ingestion.pdf_to_markdown import (
    PDFToMarkdownConverter
)

from ingestion.semantic_chunker import (
    chunk_markdown
)

from models.model import (
    langchain_embeddings
)


load_dotenv()

CONN_STRING = os.getenv(
    "DATABASE_URL"
)

if not CONN_STRING:

    raise RuntimeError(
        "DATABASE_URL is not set in .env"
    )


def validate_company(
    company: str
) -> str:

    if company is None:

        raise ValueError(
            "Company is required."
        )

    company = str(
        company
    ).strip()

    if not company:

        raise ValueError(
            "Company cannot be empty."
        )

    if len(company) > 200:

        raise ValueError(
            "Company name is too long."
        )

    return company


def validate_year(
    year
) -> int:

    if year is None:

        raise ValueError(
            "Report year is required."
        )

    year_text = str(
        year
    ).strip()

    if not year_text.isdigit():

        raise ValueError(
            "Report year must be a valid number."
        )

    year_int = int(
        year_text
    )

    if (
        year_int < 1900
        or year_int > 2100
    ):

        raise ValueError(
            "Report year must be between "
            "1900 and 2100."
        )

    return year_int


def insert_company(
    conn,
    company_name: str
):

    company_name = (
        validate_company(
            company_name
        )
    )

    with conn.cursor() as cur:

        cur.execute(
            """
            SELECT id
            FROM companies
            WHERE name = %s
            """,
            (
                company_name,
            )
        )

        row = cur.fetchone()

        if row:

            return row[0]

        cur.execute(
            """
            INSERT INTO companies
            (name)
            VALUES (%s)
            RETURNING id
            """,
            (
                company_name,
            )
        )

        company_id = (
            cur.fetchone()[0]
        )

    conn.commit()

    return company_id


def insert_document(
    conn,
    company_id,
    year,
    filename,
    file_path
):

    with conn.cursor() as cur:

        cur.execute(
            """
            INSERT INTO documents
            (
                company_id,
                filename,
                file_path,
                year
            )
            VALUES
            (
                %s,
                %s,
                %s,
                %s
            )
            RETURNING id
            """,
            (
                company_id,
                filename,
                file_path,
                year,
            )
        )

        document_id = (
            cur.fetchone()[0]
        )

    conn.commit()

    return document_id


def insert_chunk(
    conn,
    document_id,
    chunk_index,
    content,
    embedding,
    metadata
):

    with conn.cursor() as cur:

        cur.execute(
            """
            INSERT INTO document_chunks
            (
                document_id,
                chunk_index,
                content,
                embedding,
                metadata
            )
            VALUES
            (
                %s,
                %s,
                %s,
                %s,
                %s
            )
            """,
            (
                document_id,
                chunk_index,
                content,
                embedding,
                json.dumps(
                    metadata
                )
            )
        )

    conn.commit()


def ingest_document(
    pdf_path: str,
    embeddings,
    company: str,
    year
) -> None:


    company = validate_company(
        company
    )

    year = validate_year(
        year
    )


    pdf_file = Path(
        pdf_path
    )

    if not pdf_file.exists():

        raise FileNotFoundError(
            f"PDF file not found: "
            f"{pdf_path}"
        )

    if pdf_file.suffix.lower() != ".pdf":

        raise ValueError(
            "Only PDF files are supported."
        )


    print()
    print("=" * 80)
    print("[INGESTION]")
    print("=" * 80)

    print(
        f"File    : {pdf_file.name}"
    )

    print(
        f"Company : {company}"
    )

    print(
        f"Year    : {year}"
    )

    print("=" * 80)


    print(
        "[INGESTION] "
        "Converting PDF to Markdown..."
    )

    converter = (
        PDFToMarkdownConverter()
    )

    markdown_file = (
        converter.convert_pdf(
            pdf_path=pdf_path,
            output_dir="data/markdown"
        )
    )


    print(
        "[INGESTION] "
        "Creating semantic chunks..."
    )

    chunks = (
        chunk_markdown(
            markdown_file=markdown_file,
            embeddings=embeddings
        )
    )

    print(
        f"[INGESTION] "
        f"Generated {len(chunks)} chunks."
    )

    if not chunks:

        raise ValueError(
            "No chunks were generated "
            "from the PDF."
        )


    with connect(
        CONN_STRING
    ) as conn:

        register_vector(
            conn
        )


        company_id = (
            insert_company(
                conn=conn,
                company_name=company
            )
        )


        document_id = (
            insert_document(
                conn=conn,
                company_id=company_id,
                year=year,
                filename=pdf_file.name,
                file_path=str(
                    pdf_file
                ),
            )
        )


        for index, chunk in enumerate(
            chunks
        ):

            content = (
                chunk.page_content
            )

            if not content:
                continue


            embedding = (
                embeddings.embed_query(
                    content
                )
            )


            metadata_dict = {}

            if (
                hasattr(
                    chunk,
                    "metadata"
                )
                and chunk.metadata
            ):

                metadata_dict = (
                    chunk.metadata.copy()
                )


            metadata_dict.update(
                {
                    "company": company,

                    "year": year,

                    "source_file":
                        pdf_file.name,

                    "document_id":
                        document_id,

                    "chunk_index":
                        index,
                }
            )


            insert_chunk(
                conn=conn,
                document_id=document_id,
                chunk_index=index,
                content=content,
                embedding=embedding,
                metadata=metadata_dict,
            )

        print()
        print(
            "=" * 80
        )

        print(
            "[INGESTION] SUCCESS"
        )

        print(
            f"File    : {pdf_file.name}"
        )

        print(
            f"Company : {company}"
        )

        print(
            f"Year    : {year}"
        )

        print(
            f"Chunks  : {len(chunks)}"
        )

        print(
            "=" * 80
        )


def ingest_directory(
    input_dir: str
) -> None:

    pdf_files = list(
        Path(input_dir).glob(
            "*.pdf"
        )
    )

    print(
        f"Found {len(pdf_files)} PDF(s)"
    )

    print(
        "Automatic directory ingestion "
        "is disabled because company/year "
        "must be explicitly supplied."
    )

    for pdf_file in pdf_files:

        print(
            f"Skipping: {pdf_file.name}"
        )


if __name__ == "__main__":

    os.makedirs(
        "data/markdown",
        exist_ok=True
    )

    ingest_directory(
        "data/raw_pdfs"
    )