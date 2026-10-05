import os
import json
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from rag.retriever import retrieve_relevant_chunks
from llm.groq_structured import get_structured_completion

import re
import hashlib
from difflib import SequenceMatcher

load_dotenv()

class FinancialMetrics(BaseModel):
    revenue: float | None = Field(None, alias="Revenue")
    net_income: float | None = Field(None, alias="Net Income")
    operating_income: float | None = Field(None, alias="Operating Income")
    cash_flow: float | None = Field(
        None,
        alias="Cash Flow from Operating Activities"
    )
    total_assets: float | None = Field(None, alias="Total Assets")
    total_liabilities: float | None = Field(
        None,
        alias="Total Liabilities"
    )

    risk_factors: list[str] = Field(
        default_factory=list,
        alias="Top Risk Factors"
    )

    growth_drivers: list[str] = Field(
        default_factory=list,
        alias="Top Growth Drivers"
    )
    
MAX_CONTEXT_CHARS = 30000
MAX_CHUNK_CHARS = 8000

def normalize_text(text: str) -> str:
    """
    Normalize text so that small formatting differences
    do not prevent duplicate detection.
    """

    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove excessive punctuation spacing
    text = re.sub(r"\s*([,:;|])\s*", r"\1", text)

    return text.strip()


def deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """
    Remove:
    1. Exact duplicate chunks
    2. Same source/chunk duplicates
    3. Near-duplicate chunks caused by PDF/table overlap

    Keeps the chunk with the highest similarity score.
    """

    chunks = sorted(
        chunks,
        key=lambda x: float(x.get("similarity", 0)),
        reverse=True
    )

    unique_chunks = []

    seen_hashes = set()
    seen_source_chunks = set()

    for chunk in chunks:

        content = chunk.get("content", "").strip()

        if not content:
            continue


        normalized = normalize_text(content)


        content_hash = hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

        if content_hash in seen_hashes:
            print(
                f"[DEDUP] Exact duplicate removed: "
                f"{chunk.get('filename')} "
                f"chunk={chunk.get('chunk_index')}"
            )
            continue

        source_key = (
            chunk.get("filename"),
            chunk.get("chunk_index")
        )

        if source_key in seen_source_chunks:
            print(
                f"[DEDUP] Same source/chunk removed: "
                f"{source_key}"
            )
            continue

        seen_hashes.add(content_hash)
        seen_source_chunks.add(source_key)

        is_near_duplicate = False

        for existing in unique_chunks:

            existing_normalized = normalize_text(
                existing["content"]
            )

            similarity = SequenceMatcher(
                None,
                normalized,
                existing_normalized
            ).ratio()

            if similarity >= 0.85:

                print(
                    f"[DEDUP] Near duplicate removed "
                    f"(similarity={similarity:.2f}): "
                    f"{chunk.get('filename')} "
                    f"chunk={chunk.get('chunk_index')}"
                )

                is_near_duplicate = True
                break

        if not is_near_duplicate:
            unique_chunks.append(chunk)

    print(
        f"[DEDUP] Before: {len(chunks)} chunks | "
        f"After: {len(unique_chunks)} chunks"
    )

    return unique_chunks

def extract_relevant_window(
    text: str,
    keywords: list[str],
    window_lines: int = 8
) -> str:

    lines = text.splitlines()

    matched_indexes = []

    for i, line in enumerate(lines):

        line_lower = line.lower()

        for keyword in keywords:

            if keyword.lower() in line_lower:
                matched_indexes.append(i)
                break

    if not matched_indexes:
        # If nothing matches, keep a limited beginning
        # rather than sending the entire chunk.
        return text[:6000]

    selected_indexes = set()

    for index in matched_indexes:

        start = max(0, index - window_lines)
        end = min(
            len(lines),
            index + window_lines + 1
        )

        for j in range(start, end):
            selected_indexes.add(j)

    selected_lines = [
        lines[i]
        for i in sorted(selected_indexes)
    ]

    return "\n".join(selected_lines)

def retrieve_context_local(
    company: str,
    year: int
) -> str:

    income_query = f"""
    {company} fiscal year {year} annual report income statement.

    Find the exact financial statement table containing:
    - total revenue or net sales
    - operating income
    - net income

    Prioritize rows containing exact numerical values.
    """

    balance_query = f"""
    {company} fiscal year {year} annual report balance sheet.

    Find the exact balance sheet table containing:
    - total assets
    - total liabilities

    Prioritize rows containing exact numerical values.
    """

    cashflow_query = f"""
    {company} fiscal year {year} annual report cash flow statement.

    Find the exact cash flow statement table containing:
    - cash generated from operating activities
    - net cash provided by operating activities
    - operating cash flow

    Prioritize rows containing exact numerical values.
    """


    income_chunks = retrieve_relevant_chunks(
        income_query,
        top_k=2,
        company=company,
        year=year
    )

    balance_chunks = retrieve_relevant_chunks(
        balance_query,
        top_k=2,
        company=company,
        year=year
    )

    cashflow_chunks = retrieve_relevant_chunks(
        cashflow_query,
        top_k=2,
        company=company,
        year=year
    )

    all_chunks = (
        income_chunks
        + balance_chunks
        + cashflow_chunks
    )

    print("=" * 80)
    print("[RETRIEVAL] BEFORE DEDUPLICATION")
    print("=" * 80)
    print("Chunks:", len(all_chunks))

    all_chunks = deduplicate_chunks(all_chunks)

    print("=" * 80)
    print("[RETRIEVAL] AFTER DEDUPLICATION")
    print("=" * 80)
    print("Chunks:", len(all_chunks))


    all_chunks.sort(
        key=lambda x: float(
            x.get("similarity", 0)
        ),
        reverse=True
    )

    income_keywords = [
        "revenue",
        "net sales",
        "total net sales",
        "operating income",
        "income from operations",
        "net income"
    ]

    balance_keywords = [
        "total assets",
        "total liabilities",
        "assets",
        "liabilities"
    ]

    cashflow_keywords = [
        "cash generated",
        "cash provided",
        "operating activities",
        "operating cash flow",
        "net cash"
    ]

    context_parts = []

    MAX_CONTEXT_CHARS = 30000

    current_length = 0

    for chunk in all_chunks:

        content = chunk.get(
            "content",
            ""
        ).strip()

        if not content:
            continue

        combined_text = content.lower()

        if any(
            keyword in combined_text
            for keyword in income_keywords
        ):
            keywords = income_keywords

        elif any(
            keyword in combined_text
            for keyword in balance_keywords
        ):
            keywords = balance_keywords

        elif any(
            keyword in combined_text
            for keyword in cashflow_keywords
        ):
            keywords = cashflow_keywords

        else:
            keywords = []

        if keywords:

            relevant_content = extract_relevant_window(
                text=content,
                keywords=keywords,
                window_lines=8
            )

        else:

            relevant_content = content[:6000]


        block = f"""
SOURCE: {chunk.get("filename")}
CHUNK: {chunk.get("chunk_index")}
SIMILARITY: {chunk.get("similarity")}

{relevant_content}
"""


        remaining = (
            MAX_CONTEXT_CHARS
            - current_length
        )

        if remaining <= 0:
            break

        if len(block) > remaining:

            if remaining > 1000:

                block = block[:remaining]

                context_parts.append(block)

            break

        context_parts.append(block)

        current_length += len(block)

    context = "\n\n".join(context_parts)


    print("=" * 80)
    print("[RETRIEVAL] FINAL CONTEXT")
    print("=" * 80)

    print(
        "Final chunks:",
        len(context_parts)
    )

    print(
        "Characters:",
        len(context)
    )

    return context


def build_extraction_prompt(
    company: str,
    year: int,
    context: str
) -> str:

    return f"""
You are a financial data extraction system.

Company: {company}
Fiscal Year: {year}

Extract the requested financial metrics ONLY from
the supplied annual report context.

CRITICAL RULES:

1. Return ONLY valid JSON.

2. Do not return Markdown.

3. Do not return explanations.

4. Never guess.

5. Never invent financial values.

6. If a numerical value is not explicitly present,
   return null.

7. Financial statement labels are NOT values.

WRONG:
"Revenue": "Total net sales"

CORRECT:
"Revenue": 391035

8. Extract the numerical value from the same
   financial statement row as the requested metric.

9. Prefer consolidated financial statements.

10. Use fiscal year {year}.

11. Do not confuse the requested year with
    comparative prior-year columns.

12. Preserve the reported number.

13. Respect the reported unit.

14. Do not convert millions or billions.

15. Return risk factors as a list of actual risks.

16. Return growth drivers as a list of actual
    business growth factors.

17. If no supported risk factors exist, return [].

18. If no supported growth drivers exist, return [].

REQUIRED JSON:

{{
    "Revenue": null,
    "Net Income": null,
    "Operating Income": null,
    "Cash Flow from Operating Activities": null,
    "Total Assets": null,
    "Total Liabilities": null,
    "Top Risk Factors": [],
    "Top Growth Drivers": []
}}

ANNUAL REPORT CONTEXT:

{context}

Return JSON only.
"""

def extract_financial_metrics(company: str, year: int) -> dict:
    """Main operational workflow loop running entirely locally on your machine."""
    
    context = retrieve_context_local(company=company, year=year)

    prompt = build_extraction_prompt(company=company, year=year, context=context)

    print("\n" + "=" * 80)
    print("[DEBUG] PROMPT SENT TO GROQ")
    print("=" * 80)

    print(prompt)

    print("=" * 80)
    print("[DEBUG] PROMPT LENGTH")
    print("=" * 80)

    print("Characters:", len(prompt))
    
    metrics = get_structured_completion(
        prompt=prompt,
        response_model=FinancialMetrics
    )
    

    print("\n========== EXTRACTED METRICS ==========")
    print(metrics.model_dump())
    print("========================================")

    return metrics.model_dump()


def main() -> None:
    company = "Apple"
    year = 2024

    print(f"Starting Local KPI Extraction Pipeline for {company} ({year})...")
    results = extract_financial_metrics(company=company, year=year)

    print(f"\nSuccessfully Extracted Local KPIs for {company} {year}\n")

    for key, value in results.items():
        print(f"{key}:")
        print(value)
        print("-" * 80)

    try:
        from database.save_metrics import save_metrics
        save_metrics(company=company, year=year, metrics=results)
    except Exception as e:
        print(f"Error handling transactional database save sequence: {e}")


if __name__ == "__main__":
    main()