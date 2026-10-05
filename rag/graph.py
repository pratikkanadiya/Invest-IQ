import os
from typing import TypedDict, List, Optional

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from groq import Groq
from dotenv import load_dotenv

from rag.retriever import (
    retrieve_relevant_chunks,
    detect_financial_intent,
)


load_dotenv()

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set in .env"
    )

groq_client = Groq(
    api_key=GROQ_API_KEY
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


class RAGState(TypedDict, total=False):

    question: str

    company: Optional[str]

    year: Optional[int]

    context_chunks: List[dict]

    generation: str

    sources: List[dict]

    question_type: str


def classify_question(
    question: str
) -> str:

    q = question.lower().strip()


    complex_patterns = [
        "why",
        "explain",
        "analyze",
        "analysis",
        "impact",
        "effect",
        "trend",
        "risk",
        "risks",
        "risk factors",
        "growth drivers",
        "growth factors",
        "outlook",
        "strategy",
        "financial health",
        "performance analysis",
    ]

    if any(
        pattern in q
        for pattern in complex_patterns
    ):
        return "complex"


    comparison_patterns = [
        "compare",
        "comparison",
        "difference",
        "versus",
        " vs ",
        "between",
        "change from",
        "year over year",
        "year-on-year",
        "yoy",
    ]

    if any(
        pattern in q
        for pattern in comparison_patterns
    ):
        return "comparison"


    simple_patterns = [
        "what is",
        "what was",
        "how much",
        "give me",
        "tell me",
        "amount",
        "value",
        "how many",
        "which",
    ]

    if any(
        pattern in q
        for pattern in simple_patterns
    ):
        return "simple"


    return "medium"


def get_retrieval_k(
    question: str
) -> int:

    question_type = (
        classify_question(question)
    )

    if question_type == "simple":
        return 2

    if question_type == "comparison":
        return 4

    if question_type == "complex":
        return 6

    return 3


CONTEXT_BUDGETS = {

    "simple": 6000,

    "medium": 9000,

    "comparison": 12000,

    "complex": 15000,
}


def build_context(chunks, question_type, max_chars=None):

    if not chunks:
        print("[CONTEXT] No chunks received")
        return ""

    if max_chars is None:
        max_chars = CONTEXT_BUDGETS.get(question_type, 9000)

    print("=" * 80)
    print("[CONTEXT BUILDER]")
    print("=" * 80)
    print(f"Chunks received: {len(chunks)}")
    print(f"Maximum context characters: {max_chars}")

    selected_parts = []
    remaining_chars = max_chars

    for index, chunk in enumerate(chunks, start=1):

        content = chunk.get("content") or ""
        content = str(content).strip()

        print(
            f"[CONTEXT] Chunk {index}: "
            f"content length = {len(content)}"
        )

        if not content:
            print(
                f"[CONTEXT] Chunk {index} skipped: empty content"
            )
            continue

        filename = chunk.get("filename", "Unknown source")
        chunk_index = chunk.get("chunk_index", "Unknown")
        company = chunk.get("company", "Unknown")
        year = chunk.get("year", "Unknown")

        source_header = f"""
SOURCE {index}

FILE: {filename}
CHUNK: {chunk_index}
COMPANY: {company}
YEAR: {year}

CONTENT:
"""
        available = remaining_chars - len(source_header) - 100

        if available <= 0:
            print(
                f"[CONTEXT] Chunk {index} skipped: "
                f"no remaining context space"
            )
            break

        if len(content) > available:

            print(
                f"[CONTEXT] Chunk {index} is too large. "
                f"Trimming from {len(content)} "
                f"to {available} characters."
            )

            content = content[:available]

            last_space = content.rfind(" ")

            if last_space > int(available * 0.8):
                content = content[:last_space]

            content += "\n[END OF SELECTED CONTEXT]"

        source_text = source_header + content

        selected_parts.append(source_text.strip())

        remaining_chars -= len(source_text)

        print(
            f"[CONTEXT] Chunk {index} added. "
            f"Remaining characters: {remaining_chars}"
        )

        if remaining_chars <= 200:
            break

    final_context = (
        "\n\n------------------------------\n\n"
        .join(selected_parts)
    )

    print(
        f"[CONTEXT] Selected chunks: "
        f"{len(selected_parts)}"
    )

    print(
        f"[CONTEXT] Final context length: "
        f"{len(final_context)} characters"
    )

    if not final_context:
        print("[CONTEXT] WARNING: FINAL CONTEXT IS EMPTY")

    print("=" * 80)

    return final_context


def build_sources(
    chunks: List[dict]
) -> List[dict]:

    sources = []

    for chunk in chunks:

        source = {
            "filename": chunk.get(
                "filename",
                "Unknown"
            )
        }

        if "page" in chunk:

            source["page"] = (
                chunk["page"]
            )

        elif isinstance(
            chunk.get("metadata"),
            dict
        ):

            page = chunk[
                "metadata"
            ].get("page")

            if page is not None:

                source["page"] = page

        if "similarity" in chunk:

            source["similarity"] = (
                chunk["similarity"]
            )

        if "final_score" in chunk:

            source["final_score"] = (
                chunk["final_score"]
            )

        sources.append(source)

    return sources


def retrieve_node(
    state: RAGState
) -> dict:

    question = state[
        "question"
    ]

    company = state.get(
        "company"
    )

    year = state.get(
        "year"
    )

    question_type = (
        classify_question(question)
    )

    retrieval_k = (
        get_retrieval_k(question)
    )

    print()
    print("=" * 80)
    print("[NODE: RETRIEVE]")
    print("=" * 80)

    print(
        f"Question: {question}"
    )

    print(
        f"Company: {company}"
    )

    print(
        f"Year: {year}"
    )

    print(
        f"Question type: "
        f"{question_type}"
    )

    print(
        f"Retrieval K: "
        f"{retrieval_k}"
    )


    intent = detect_financial_intent(
        question
    )

    print(
        f"Financial intent: "
        f"{intent}"
    )


    matched_data = (
        retrieve_relevant_chunks(
            query_text=question,
            top_k=retrieval_k,
            company=company,
            year=year,
        )
    )


    if not matched_data:

        print(
            "[NODE: RETRIEVE] "
            "No relevant documents found"
        )

        return {
            "context_chunks": [],
            "sources": [],
            "question_type": question_type,
        }

    sources = build_sources(
        matched_data
    )

    print(
        f"[NODE: RETRIEVE] "
        f"Retrieved "
        f"{len(matched_data)} chunks"
    )

    return {

        "context_chunks":
            matched_data,

        "sources":
            sources,

        "question_type":
            question_type,
    }


def get_output_instruction(
    question: str,
    question_type: str,
) -> str:

    q = question.lower()


    only_amount = any(
        phrase in q
        for phrase in [
            "only amount",
            "only the amount",
            "just amount",
            "only number",
            "only the number",
            "amount only",
            "number only",
        ]
    )

    if only_amount:

        return """
The user explicitly requested ONLY the amount.

Return ONLY the requested financial value.

Do NOT return:
- explanation
- sentence
- source
- markdown
- bullet points
- analysis

Preserve the original number,
unit and currency.

If the requested value is not present
in the supplied context, return exactly:

NOT_FOUND
"""

    if question_type == "simple":

        return """
Answer the user's factual question directly.

Use the exact value or statement
supported by the supplied context.

Keep the answer concise.

Do not add unnecessary explanation.
"""

    if question_type == "comparison":

        return """
Answer the comparison clearly.

Identify the relevant years,
values and differences only when
supported by the supplied context.

Do not calculate a difference unless
the user explicitly asks for it.
"""

    if question_type == "complex":

        return """
Provide a concise but useful analysis.

Explain the answer using evidence
from the supplied annual-report context.

For trends, identify the relevant
years and direction of change.

For risks, identify risks explicitly
supported by the report.

For growth questions, identify
growth drivers supported by the report.

Do not invent causes or explanations.
"""

    return """
Answer the user's question clearly
and concisely.

Use only the supplied document context.
"""

def format_groq_error(
    error: Exception
) -> str:

    error_text = str(error)

    print()
    print(
        "[NODE: GENERATE] "
        f"Groq error: {error_text}"
    )

    if (
        "413" in error_text
        or "Request too large" in error_text
        or "TPM Limit" in error_text
    ):

        return (
            "The retrieved financial context "
            "was too large for the Groq request. "
            "Please try the question again."
        )

    if (
        "429" in error_text
        or "rate limit" in error_text.lower()
    ):

        return (
            "The Groq rate limit was reached. "
            "Please try again shortly."
        )

    if (
        "401" in error_text
        or "authentication" in error_text.lower()
        or "invalid api key" in error_text.lower()
    ):

        return (
            "The Groq API key is invalid "
            "or unavailable."
        )

    return (
        "I was unable to generate the "
        "financial answer."
    )


def generate_node(
    state: RAGState
) -> dict:

    print()
    print("=" * 80)
    print("[NODE: GENERATE]")
    print("=" * 80)

    context_chunks = state.get(
        "context_chunks",
        []
    )

    question = state[
        "question"
    ]

    company = state.get(
        "company"
    )

    year = state.get(
        "year"
    )

    question_type = state.get(
        "question_type"
    ) or classify_question(
        question
    )


    if not context_chunks:

        return {
            "generation": (
                "I cannot find that information "
                "in the uploaded documents."
            )
        }


    intent = detect_financial_intent(
        question
    )


    formatted_context = (
        build_context(
            chunks=context_chunks,
            question_type=question_type,
        )
    )
    
    print()
    print(
        "[NODE: GENERATE] "
        f"Formatted context length: "
        f"{len(formatted_context)}"
    )

    print(
        "[NODE: GENERATE] "
        f"Context preview:\n"
        f"{formatted_context[:1500]}"
    )

    if not formatted_context:

        return {
            "generation": (
                "I cannot find enough relevant "
                "information in the uploaded documents."
            )
        }

    metadata_lines = []

    if company:

        metadata_lines.append(
            f"Requested company: {company}"
        )

    if year:

        metadata_lines.append(
            f"Requested fiscal year: {year}"
        )

    metadata_context = (
        "\n".join(
            metadata_lines
        )
    )


    output_instruction = (
        get_output_instruction(
            question,
            question_type,
        )
    )

    system_prompt = f"""
You are an AI Financial Analyst for
an Investor Intelligence platform.

Your job is to answer questions using
ONLY the supplied company-report context.

IMPORTANT RULES:

1. Never use outside knowledge.

2. Never invent information.

3. Never guess missing information.

4. Respect the requested company.

5. Respect the requested fiscal year.

6. Do not confuse fiscal-year columns.

7. Preserve the original financial units.

8. Preserve the original currency.

9. Do not calculate unless the user
   explicitly requests a calculation.

10. If information is not supported
    by the supplied context, return
    NOT_FOUND.

11. Do not replace one financial metric
    with a similar metric.

12. For numerical questions, prefer
    exact reported values.

13. For comparison questions, clearly
    identify the relevant years.

14. For analysis questions, explain
    only what is supported by evidence.

15. Keep answers concise unless the
    user requests detailed analysis.

QUESTION TYPE:
{question_type}

FINANCIAL INTENT:
{intent}

OUTPUT INSTRUCTION:
{output_instruction}
"""


    user_prompt = f"""
{metadata_context}

DOCUMENT CONTEXT
================

{formatted_context}

USER QUESTION
=============

{question}

ANSWER
======
"""


    print(
        f"[NODE: GENERATE] "
        f"Question type: {question_type}"
    )

    print(
        f"[NODE: GENERATE] "
        f"Chunks available: "
        f"{len(context_chunks)}"
    )

    print(
        f"[NODE: GENERATE] "
        f"Context characters: "
        f"{len(formatted_context)}"
    )

    print(
        f"[NODE: GENERATE] "
        f"Groq model: {GROQ_MODEL}"
    )


    try:

        response = (
            groq_client.chat.completions.create(

                model=GROQ_MODEL,

                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],

                temperature=0,
            )
        )

        ai_response = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        if not ai_response:

            ai_response = (
                "I could not generate "
                "a valid answer."
            )

        print()
        print(
            "[NODE: GENERATE] "
            "Groq response received"
        )

        print(
            f"Answer: {ai_response}"
        )

    except Exception as e:

        ai_response = (
            format_groq_error(e)
        )

    return {
        "generation": ai_response
    }


builder = StateGraph(
    RAGState
)


builder.add_node(
    "retrieve_documents",
    retrieve_node,
)

builder.add_node(
    "generate_response",
    generate_node,
)


builder.add_edge(
    START,
    "retrieve_documents",
)

builder.add_edge(
    "retrieve_documents",
    "generate_response",
)

builder.add_edge(
    "generate_response",
    END,
)


rag_graph = builder.compile()


if __name__ == "__main__":

    print()
    print("=" * 80)
    print("INVESTOR INTELLIGENCE RAG")
    print("LANGGRAPH END-TO-END TEST")
    print("=" * 80)
    print()

    test_input: RAGState = {

        "question":
            "What is Apple's total revenue "
            "in 2024?",

        "company":
            "Apple",

        "year":
            2024,
    }

    print(
        "Executing LangGraph..."
    )

    try:

        final_state = (
            rag_graph.invoke(
                test_input
            )
        )

        print()
        print(
            "=" * 80
        )

        print(
            "ANSWER"
        )

        print(
            "=" * 80
        )

        print(
            final_state.get(
                "generation"
            )
        )

        print()

        if final_state.get(
            "sources"
        ):

            print(
                "SOURCES"
            )

            print(
                "-" * 80
            )

            for source in (
                final_state["sources"]
            ):

                print(
                    f"- "
                    f"{source.get('filename')}"
                )

        print(
            "=" * 80
        )

    except Exception as e:

        print(
            f"Graph execution failed: {e}"
        )