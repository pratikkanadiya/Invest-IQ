from fastapi import APIRouter

from database.connection import get_connection


router = APIRouter(
    tags=["Health"]
)


@router.get("/health")
def health_check():

    database_status = "unhealthy"

    try:

        with get_connection() as conn:

            with conn.cursor() as cur:
                cur.execute("SELECT 1")

            database_status = "healthy"

    except Exception as e:

        return {
            "status": "unhealthy",
            "database": "unhealthy",
            "error": str(e)
        }

    return {
        "status": "healthy",
        "database": database_status
    }