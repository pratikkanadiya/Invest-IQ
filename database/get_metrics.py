import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

CONN_STRING = os.getenv("DATABASE_URL")

def get_metrics() -> list[dict]:
    query = """
    SELECT 
        id, company, year, revenue, net_income, 
        operating_income, cash_flow, total_assets, 
        total_liabilities, risk_factors, growth_drivers, created_at
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY company, year
                   ORDER BY created_at DESC
               ) AS rn
        FROM financial_metrics
    ) t
    WHERE rn = 1
    ORDER BY company;
    """

    rows = []
    
    with psycopg.connect(CONN_STRING) as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(query)
            rows = cur.fetchall()
            
    return rows

if __name__ == "__main__":
    print("Testing financial metrics data retrieval...")
    metrics_data = get_metrics()
    
    print(f"\nFound {len(metrics_data)} active company record(s) in database:\n")
    for company_record in metrics_data:
        print(f"=== {company_record['company']} ({company_record['year']}) ===")
        print(f"Revenue: {company_record['revenue']}")
        print(f"Net Income: {company_record['net_income']}")
        print("-" * 40)
