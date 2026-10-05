import os
import json

from dotenv import load_dotenv
from pydantic import BaseModel
from groq import Groq


load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set in .env"
    )


GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


client = Groq(
    api_key=GROQ_API_KEY
)


def get_structured_completion(
    prompt: str,
    response_model: type[BaseModel],
    model: str | None = None
) -> BaseModel:

    model_name = model or GROQ_MODEL

    print("=" * 80)
    print("[GROQ] MODEL")
    print("=" * 80)
    print(model_name)


    schema = response_model.model_json_schema()


    try:

        response = client.chat.completions.create(

            model=model_name,

            messages=[
                {
                    "role": "system",
                    "content": """
You are an expert financial data extraction system.

Extract information ONLY from the supplied context.

Rules:

1. Never invent financial values.
2. Never guess.
3. If a numerical value is not present, return null.
4. Financial statement labels are NOT values.
5. Preserve the reported numerical value.
6. Respect the reported units.
7. Use the requested fiscal year.
8. Prefer financial statement tables.
9. Return only the requested structured JSON.
"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "financial_metrics",
                    "strict": False,
                    "schema": schema
                }
            },

            temperature=0,

        )

    except Exception as e:

        raise RuntimeError(
            f"Groq generation failed using "
            f"model '{model_name}'.\n"
            f"{type(e).__name__}: {e}"
        ) from e


    content = (
        response.choices[0]
        .message
        .content
    )

    if not content:

        raise RuntimeError(
            "Groq returned an empty response."
        )


    print(
        "\n[DEBUG] Raw Groq output:\n",
        content
    )


    try:

        parsed_json = json.loads(
            content
        )

    except json.JSONDecodeError as e:

        raise RuntimeError(
            "Groq returned invalid JSON.\n"
            f"Raw response:\n{content}"
        ) from e


    try:

        return response_model.model_validate(
            parsed_json
        )

    except Exception as e:

        raise RuntimeError(
            "Groq JSON could not be validated "
            "against the Pydantic model.\n"
            f"Validation error: {e}\n"
            f"Parsed JSON: {parsed_json}"
        ) from e