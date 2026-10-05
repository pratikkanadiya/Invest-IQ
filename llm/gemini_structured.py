import os
import re
import json

from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. "
        "Add GEMINI_API_KEY=your_api_key_here to your .env file."
    )


client = genai.Client(
    api_key=GEMINI_API_KEY
)


def get_structured_completion(
    prompt: str,
    response_model: type[BaseModel],
    model: str | None = None
) -> BaseModel:

    model_name = model or "gemini-3.8-flash"


    system_prompt = """
You are an expert financial analyst working inside an investor RAG system.

Your task is to extract financial information from the provided
retrieved context.

IMPORTANT RULES:

1. Return ONLY valid JSON.

2. Do NOT return Markdown.

3. Do NOT return ```json.

4. Do NOT add explanations before or after the JSON.

5. Use ONLY information present in the provided context.

6. Do NOT invent or guess financial values.

7. If a value cannot be found, use null when allowed by the schema.

8. Follow the requested JSON structure exactly.

9. Preserve the numerical value exactly as reported.

10. Pay attention to units such as millions or billions.

11. Do not convert currencies or units unless explicitly requested.

12. If the table states that amounts are in millions,
    return the reported number and do not silently multiply it.

13. Prefer values from financial statement tables over narrative text.

14. Make sure the value corresponds to the requested fiscal year.

15. NEVER return a financial statement LABEL as the value.

    Example of WRONG output:

    "Revenue": "Total net sales"

    Example of CORRECT output:

    "Revenue": 391035

16. The words such as "Revenue", "Total net sales",
    "Operating income", "Net income", etc. are labels.

    Extract the numerical value associated with the label.

17. If the exact numerical value is not present in the
    retrieved context, return null.

18. Do not return phrases such as:
    "not provided",
    "not found",
    "exact text value not provided",
    or similar placeholder text.

19. Prefer the financial statement row that belongs to
    the requested fiscal year.

20. Do not confuse prior-year comparative values with
    the requested fiscal year.
"""


    full_prompt = f"""
{system_prompt}

USER EXTRACTION REQUEST:

{prompt}
"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_model,
            ),
        )

    except Exception as e:
        raise RuntimeError(
            f"Gemini generation failed using model '{model_name}'.\n"
            f"Error: {type(e).__name__}: {e}"
        ) from e


    text_response = response.text

    if not text_response:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    print(
        "\n[debug] Raw Gemini output received:\n",
        text_response
    )


    clean_text = text_response.strip()

    # Remove accidental Markdown fences if present
    clean_text = (
        clean_text
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )


    try:
        parsed_json = json.loads(clean_text)

    except json.JSONDecodeError:

        match = re.search(
            r"\{.*\}",
            clean_text,
            re.DOTALL
        )

        if not match:
            raise RuntimeError(
                "Could not find valid JSON in Gemini response.\n"
                f"Raw response:\n{clean_text}"
            )

        json_text = match.group(0)

        try:
            parsed_json = json.loads(json_text)

        except json.JSONDecodeError as e:
            raise RuntimeError(
                "Gemini returned invalid JSON.\n"
                f"Raw response:\n{clean_text}"
            ) from e


    try:

        if hasattr(response_model, "model_validate"):

            # Pydantic v2
            result = response_model.model_validate(
                parsed_json
            )

        else:

            # Pydantic v1
            result = response_model.parse_obj(
                parsed_json
            )

    except Exception as e:

        raise RuntimeError(
            "Gemini JSON could not be validated against "
            "the requested Pydantic model.\n"
            f"Validation error: {e}\n"
            f"Parsed JSON: {parsed_json}"
        ) from e

    return result


if __name__ == "__main__":

    class SimpleTestModel(BaseModel):
        company_name: str
        founding_year: int
        ticker_symbol: str

    test_prompt = """
Extract the following information from this text:

Apple Inc was founded in 1976 and operates under ticker AAPL.

Return:

- company_name
- founding_year
- ticker_symbol
"""

    print(
        "Testing Gemini structured extraction..."
    )

    result = get_structured_completion(
        test_prompt,
        SimpleTestModel
    )

    print(
        "\n[debug] Final Verified Pydantic Object:"
    )

    print(
        result.model_dump()
    )