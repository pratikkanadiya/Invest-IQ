import os
import re

import psycopg
from dotenv import load_dotenv

load_dotenv()

CONN_STRING = os.getenv("DATABASE_URL")


def clean_numeric_value(value):

    if value is None:
        return None

    if isinstance(value, (int, float)):
        return value

    value = str(value).strip()

    if not value:
        return None

    if value.lower() in {
        "n/a",
        "na",
        "none",
        "null",
        "not available",
        "not disclosed",
        "-"
    }:
        return None

    cleaned = value.replace("$", "")
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.strip()

    negative = False

    if cleaned.startswith("(") and cleaned.endswith(")"):
        negative = True
        cleaned = cleaned[1:-1].strip()

    multiplier = 1

    lower_value = cleaned.lower()

    if "billion" in lower_value:
        multiplier = 1_000_000_000
        cleaned = re.sub(
            r"\s*billion(s)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE
        )

    elif "million" in lower_value:
        multiplier = 1_000_000
        cleaned = re.sub(
            r"\s*million(s)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE
        )

    elif "thousand" in lower_value:
        multiplier = 1_000
        cleaned = re.sub(
            r"\s*thousand(s)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE
        )

    cleaned = cleaned.strip()

    try:
        number = float(cleaned) * multiplier

        if negative:
            number = -number

        return number

    except ValueError:
        return None


def save_metrics(
    company: str,
    year: int,
    metrics: dict
) -> None:


    query = """
    INSERT INTO financial_metrics (
        company,
        year,
        revenue,
        net_income,
        operating_income,
        cash_flow,
        total_assets,
        total_liabilities,
        risk_factors,
        growth_drivers
    )
    VALUES (
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s
    );
    """

    def clean_list_field(
        field_key_lower,
        field_key_upper
    ):

        data = (
            metrics.get(field_key_upper)
            or metrics.get(field_key_lower)
            or []
        )

        if isinstance(data, list):

            return "\n".join(
                str(item)
                for item in data
                if item
            )

        if data is None:
            return None

        return str(data)


    revenue = clean_numeric_value(
        metrics.get("Revenue")
        or metrics.get("revenue")
    )

    net_income = clean_numeric_value(
        metrics.get("Net Income")
        or metrics.get("net_income")
    )

    operating_income = clean_numeric_value(
        metrics.get("Operating Income")
        or metrics.get("operating_income")
    )

    cash_flow = clean_numeric_value(
        metrics.get(
            "Cash Flow from Operating Activities"
        )
        or metrics.get("cash_flow")
    )

    total_assets = clean_numeric_value(
        metrics.get("Total Assets")
        or metrics.get("total_assets")
    )

    total_liabilities = clean_numeric_value(
        metrics.get("Total Liabilities")
        or metrics.get("total_liabilities")
    )


    params = (
        company,
        int(year),

        revenue,
        net_income,
        operating_income,
        cash_flow,
        total_assets,
        total_liabilities,

        clean_list_field(
            "risk_factors",
            "Top Risk Factors"
        ),

        clean_list_field(
            "growth_drivers",
            "Top Growth Drivers"
        )
    )


    with psycopg.connect(CONN_STRING) as conn:

        with conn.cursor() as cur:

            cur.execute(
                query,
                params
            )

        conn.commit()


    print(
        f"Successfully saved metrics "
        f"for {company} {year} into PostgreSQL!"
    )


if __name__ == "__main__":

    sample_metrics = {

        "Revenue":
            "$391,035,000,000",

        "Net Income":
            "$93,736,000,000",

        "Operating Income":
            "$123,216,000,000",

        "Cash Flow from Operating Activities":
            "$118,254,000,000",

        "Total Assets":
            "$364,980,000,000",

        "Total Liabilities":
            "$308,030,000,000",

        "Top Risk Factors": [
            "Macroeconomic conditions including "
            "inflation and interest rates.",

            "High competition and rapid "
            "technological changes."
        ],

        "Top Growth Drivers": [
            "Increased Services revenue from "
            "ecosystem verticals.",

            "Continued strong global product "
            "distribution demand chains."
        ]
    }


    save_metrics(
        company="Apple",
        year=2024,
        metrics=sample_metrics
    )
