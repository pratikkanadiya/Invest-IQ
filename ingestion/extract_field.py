import re
from pathlib import Path


def extract_company_year(filename: str):

    if not filename:
        raise ValueError("Filename is required.")

    name = Path(filename).stem

    year_match = re.search(r"\b(19|20)\d{2}\b", name)

    if not year_match:
        raise ValueError(
            "Could not find a year in the PDF filename. "
            "Please include a 4-digit year, for example: Apple_2024.pdf"
        )

    year = year_match.group(0)

    company = name[:year_match.start()]

    company = re.sub(r"[_\-]+", " ", company)

    company = re.sub(
        r"\b(annual|report|financial|statement|statements|fy)\b",
        "",
        company,
        flags=re.IGNORECASE
    )

    company = re.sub(r"\s+", " ", company).strip()

    if not company:
        raise ValueError(
            "Could not determine the company name from the filename. "
            "Example: Apple_2024.pdf"
        )

    return company, year