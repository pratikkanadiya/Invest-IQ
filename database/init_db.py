import os

import psycopg

from pgvector.psycopg import register_vector
from dotenv import load_dotenv

load_dotenv()

CONN_STRING = os.getenv("DATABASE_URL")

EMBEDDING_DIMENSION = 384


def setup_rag_table():

    with psycopg.connect(CONN_STRING) as conn:

        with conn.cursor() as cur:

            cur.execute(
                "CREATE EXTENSION IF NOT EXISTS vector;"
            )

            conn.commit()

            register_vector(conn)


            cur.execute("""
                CREATE TABLE IF NOT EXISTS companies (
                    id BIGSERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    ticker TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            

            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id BIGSERIAL PRIMARY KEY,
                    company_id BIGINT REFERENCES companies(id),
                    filename TEXT NOT NULL,
                    document_type TEXT,
                    year INT,
                    file_path TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)


            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    document_id BIGINT
                        REFERENCES documents(id)
                        ON DELETE CASCADE,

                    chunk_index INT NOT NULL,

                    content TEXT NOT NULL,

                    embedding VECTOR({EMBEDDING_DIMENSION})
                        NOT NULL,

                    metadata JSONB,

                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)


            cur.execute("""
                CREATE INDEX IF NOT EXISTS
                document_chunks_embedding_hnsw_idx

                ON document_chunks
                USING hnsw (embedding vector_cosine_ops);
            """)


            cur.execute("""
                CREATE TABLE IF NOT EXISTS financial_metrics (

                    id BIGSERIAL PRIMARY KEY,

                    company VARCHAR(255) NOT NULL,

                    year INT NOT NULL,

                    revenue NUMERIC(20, 2),

                    net_income NUMERIC(20, 2),

                    operating_income NUMERIC(20, 2),

                    cash_flow NUMERIC(20, 2),

                    total_assets NUMERIC(20, 2),

                    total_liabilities NUMERIC(20, 2),

                    risk_factors TEXT,

                    growth_drivers TEXT,

                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)


            conn.commit()

            print(
                "Successfully built the vector RAG tables!"
            )


if __name__ == "__main__":

    setup_rag_table()