import re
import json

from pydantic import BaseModel
import ollama


def get_structured_completion(
    prompt: str,
    response_model: type[BaseModel],
    model: str | None = None
) -> BaseModel:
    
    model_name = model or "qwen2.5:3b"


    system_prompt = """
You are an expert financial analyst working inside an investor RAG system.

Your task is to extract financial information from the provided
retrieved context.

IMPORTANT RULES:

1. Return ONLY valid JSON.
2. Do NOT return Markdown.
3. Do NOT use ```json.
4. Do NOT add explanations before or after the JSON.
5. Use only information present in the provided context.
6. Do not invent or guess financial values.
7. If a value cannot be found, use null when allowed by the schema.
8. Follow the requested JSON structure exactly.
9. Preserve the numerical value exactly as reported.
10. Pay attention to units such as millions or billions.
11. Do not convert currencies or units unless explicitly requested.
12. If the table states that amounts are in millions, return the
    reported number and do not silently multiply it.
13. Prefer values from financial statement tables over narrative text.
14. Make sure the value corresponds to fiscal year {year}.
"""

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            format="json" 
        )

    except Exception as e:
        raise RuntimeError(
            f"Ollama local generation failed. Make sure Ollama app is running and '{model_name}' is downloaded.\nError: {e}"
        ) from e


    text_response = response['message']['content']

    if not text_response:
        raise RuntimeError(
            "Ollama returned an empty response."
        )

    print(
        "\n[debug] Raw Ollama local LLM output received:\n",
        text_response
    )


    clean_text = text_response.strip()

    # Remove accidental Markdown fences
    clean_text = clean_text.replace(
        "```json", ""
    ).replace(
        "```", ""
    ).strip()


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
                "Could not find valid JSON in Ollama response.\n"
                f"Raw response:\n{clean_text}"
            )

        json_text = match.group(0)

        try:
            parsed_json = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                "Ollama returned invalid JSON.\n"
                f"Raw response:\n{clean_text}"
            ) from e


    try:
        if hasattr(response_model, "model_validate"):
            # Pydantic v2
            result = response_model.model_validate(parsed_json)
        else:
            # Pydantic v1
            result = response_model.parse_obj(parsed_json)
    except Exception as e:
        raise RuntimeError(
            "Ollama JSON could not be validated against the requested Pydantic model.\n"
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

    print("Testing Ollama local structured extraction...")

    result = get_structured_completion(
        test_prompt,
        SimpleTestModel
    )

    print("\n[debug] Final Verified Pydantic Object:")
    print(result.model_dump())
