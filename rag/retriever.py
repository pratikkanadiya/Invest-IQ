import os
from typing import List, Optional

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from models.model import langchain_embeddings

load_dotenv()

CONN_STRING = os.getenv("DATABASE_URL")

if not CONN_STRING:
    raise RuntimeError(
        "DATABASE_URL is not set in .env"
    )


def detect_financial_intent(question: str) -> str:

    q = question.lower()

    if any(term in q for term in [
        "total revenue",
        "revenue",
        "total sales",
        "net sales",
        "sales revenue",
    ]):
        return "revenue"

    if any(term in q for term in [
        "net income",
        "net profit",
        "profit after tax",
        "profit",
        "net earnings",
    ]):
        return "net_income"

    if any(term in q for term in [
        "operating income",
        "income from operations",
        "operating profit",
    ]):
        return "operating_income"

    if any(term in q for term in [
        "cash flow",
        "operating cash flow",
        "cash generated from operating",
        "cash provided by operating",
        "cash provided by operating activities",
    ]):
        return "cash_flow"

    if any(term in q for term in [
        "total assets",
        "assets",
    ]):
        return "total_assets"

    if any(term in q for term in [
        "total liabilities",
        "liabilities",
    ]):
        return "total_liabilities"

    if any(term in q for term in [
        "risk",
        "risks",
        "risk factors",
        "risk factor",
    ]):
        return "risk"

    if any(term in q for term in [
        "growth",
        "growth drivers",
        "growth factors",
    ]):
        return "growth"

    return "general"


def build_search_query(
    question: str,
    company: Optional[str] = None,
    year: Optional[int] = None,
) -> str:

    intent = detect_financial_intent(question)

    company_text = company or ""
    year_text = (
        f"fiscal year {year}"
        if year
        else ""
    )


    if intent == "revenue":

        return f"""
{company_text}
{year_text}

Consolidated Statements of Operations
Income Statement
Total Revenue
Revenue
Total Net Sales
Net Sales

Find the exact financial statement row containing
the numerical revenue or total net sales value.

User question:
{question}
"""

    if intent == "net_income":

        return f"""
{company_text}
{year_text}

Consolidated Statements of Operations
Income Statement
Net Income
Net Profit
Net Earnings

Find the exact financial statement row containing
the numerical net income value.

User question:
{question}
"""

    if intent == "operating_income":

        return f"""
{company_text}
{year_text}

Consolidated Statements of Operations
Income Statement
Operating Income
Income from Operations
Operating Profit

Find the exact financial statement row containing
the numerical operating income value.

User question:
{question}
"""

    if intent == "cash_flow":

        return f"""
{company_text}
{year_text}

Consolidated Statements of Cash Flows
Cash Flow Statement
Operating Activities
Cash Provided by Operating Activities
Cash Generated from Operating Activities
Operating Cash Flow

Find the exact cash flow statement row
containing the requested numerical value.

User question:
{question}
"""

    if intent == "total_assets":

        return f"""
{company_text}
{year_text}

Consolidated Balance Sheets
Balance Sheet
Total Assets

Find the exact balance sheet row containing
the numerical total assets value.

User question:
{question}
"""

    if intent == "total_liabilities":

        return f"""
{company_text}
{year_text}

Consolidated Balance Sheets
Balance Sheet
Total Liabilities

Find the exact balance sheet row containing
the numerical total liabilities value.

User question:
{question}
"""

    if intent == "risk":

        return f"""
{company_text}
{year_text}

Annual Report
Risk Factors
Principal Risks
Business Risks
Financial Risks
Market Risks
Operational Risks

Find information discussing the company's
material risks.

User question:
{question}
"""

    if intent == "growth":

        return f"""
{company_text}
{year_text}

Annual Report
Growth Drivers
Revenue Growth
Business Growth
Growth Factors
Strategic Growth
Future Growth

Find information explaining the company's
growth drivers and growth factors.

User question:
{question}
"""

    return f"""
{company_text}
{year_text}

Annual Report
Financial Statements
Financial Performance
Business Operations
Management Discussion
Risk Factors
Strategy
Outlook

User question:
{question}
"""

def keyword_score(
    content: str,
    intent: str,
) -> float:

    text = content.lower()

    statement_keywords = {

        "revenue": [
            "consolidated statements of operations",
            "total net sales",
            "net sales",
            "total revenue",
            "revenue",
        ],

        "net_income": [
            "consolidated statements of operations",
            "net income",
            "net earnings",
            "net profit",
        ],

        "operating_income": [
            "consolidated statements of operations",
            "operating income",
            "income from operations",
            "operating profit",
        ],

        "cash_flow": [
            "consolidated statements of cash flows",
            "cash provided by operating activities",
            "cash generated from operating activities",
            "operating cash flow",
            "operating activities",
        ],

        "total_assets": [
            "consolidated balance sheets",
            "total assets",
        ],

        "total_liabilities": [
            "consolidated balance sheets",
            "total liabilities",
        ],

        "risk": [
            "risk factors",
            "market risk",
            "financial risk",
            "business risk",
            "operational risk",
            "principal risks",
        ],

        "growth": [
            "growth drivers",
            "revenue growth",
            "growth factors",
            "business growth",
            "strategic growth",
        ],
    }

    keywords = statement_keywords.get(
        intent,
        []
    )

    score = 0.0

    for keyword in keywords:

        if keyword in text:

            if keyword.startswith(
                "consolidated statements"
            ):
                score += 0.40

            elif keyword in [
                "total net sales",
                "total revenue",
                "net income",
                "operating income",
                "total assets",
                "total liabilities",
                "cash provided by operating activities",
            ]:
                score += 0.30

            else:
                score += 0.10

    return score


def retrieve_relevant_chunks(
    query_text: str,
    top_k: int = 5,
    company: Optional[str] = None,
    year: Optional[int] = None,
) -> List[dict]:

    intent = detect_financial_intent(
        query_text
    )

    search_query = build_search_query(
        query_text,
        company,
        year,
    )

    print("=" * 80)
    print("[RETRIEVER]")
    print("=" * 80)

    print("Original question:")
    print(query_text)

    print()

    print("Detected intent:")
    print(intent)

    print()

    print("Company:")
    print(company)

    print()

    print("Year:")
    print(year)

    print()

    print("Expanded search query:")
    print(search_query.strip())

    print("=" * 80)
    

    query_embedding = (
        langchain_embeddings.embed_query(
            search_query
        )
    )


    candidate_k = max(
        top_k * 4,
        20
    )

    results = []

    with psycopg.connect(
        CONN_STRING
    ) as conn:

        register_vector(conn)

        with conn.cursor() as cur:

            sql = """
                SELECT
                    dc.content,
                    dc.chunk_index,
                    dc.metadata,
                    d.filename,
                    c.name AS company_name,
                    c.ticker,
                    d.year,

                    1 - (
                        dc.embedding <=> %s::vector
                    ) AS similarity_score

                FROM document_chunks dc

                JOIN documents d
                    ON dc.document_id = d.id

                JOIN companies c
                    ON d.company_id = c.id

                WHERE

                    (
                        %s::text IS NULL
                        OR c.name ILIKE %s
                    )

                    AND

                    (
                        %s::integer IS NULL
                        OR d.year = %s::integer
                    )

                ORDER BY
                    dc.embedding <=> %s::vector

                LIMIT %s;
            """

            company_pattern = None

            if company:
                company_pattern = (
                    f"%{company}%"
                )

            cur.execute(
                sql,
                (
                    query_embedding,

                    company,
                    company_pattern,

                    year,
                    year,

                    query_embedding,

                    candidate_k,
                )
            )

            rows = cur.fetchall()

            for row in rows:

                content = row[0] or ""

                vector_similarity = float(
                    row[7]
                )

                financial_bonus = (
                    keyword_score(
                        content,
                        intent,
                    )
                )


                final_score = (
                    vector_similarity
                    + financial_bonus
                )

                results.append(
                    {
                        "content": content,

                        "chunk_index": row[1],

                        "metadata": row[2],

                        "filename": row[3],

                        "company": row[4],

                        "ticker": row[5],

                        "year": row[6],

                        "similarity": round(
                            vector_similarity,
                            4
                        ),

                        "financial_score": round(
                            financial_bonus,
                            4
                        ),

                        "final_score": round(
                            final_score,
                            4
                        ),
                    }
                )


    results.sort(
        key=lambda x: x.get(
            "final_score",
            0
        ),
        reverse=True,
    )


    unique_results = []

    seen = set()

    for result in results:

        key = (
            result.get("filename"),
            result.get("chunk_index"),
        )

        if key in seen:
            continue

        seen.add(key)

        unique_results.append(
            result
        )

    final_results = unique_results[:top_k]


    print()
    print("[RETRIEVER] SELECTED CHUNKS")

    for index, result in enumerate(
        final_results,
        start=1
    ):

        print(
            f"#{index} "
            f"{result.get('filename')} "
            f"chunk={result.get('chunk_index')} "
            f"vector={result.get('similarity')} "
            f"financial={result.get('financial_score')} "
            f"final={result.get('final_score')}"
        )

    print(
        f"[RETRIEVER] Retrieved "
        f"{len(final_results)} chunks"
    )

    return final_results


if __name__ == "__main__":

    test_query = (
        "What was Apple's total revenue "
        "in 2024?"
    )

    results = retrieve_relevant_chunks(
        query_text=test_query,
        top_k=2,
        company="Apple",
        year=2024,
    )

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    for result in results:

        print(
            result["filename"],
            result["chunk_index"],
            result["similarity"],
            result["financial_score"],
            result["final_score"],
        )