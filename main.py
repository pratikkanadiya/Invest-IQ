from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from routes.health import router as health_router
from routes.chat import router as chat_router
from routes.ingestion import router as ingestion_router
from routes.dashboard import router as dashboard_router

from database.get_metrics import get_metrics


app = FastAPI(
    title="Investor Intelligence RAG",
    description="AI-powered investor intelligence platform using PostgreSQL, pgvector, LangGraph and Ollama.",
    version="1.0.0"
)


app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


templates = Jinja2Templates(
    directory="templates"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health_router)
app.include_router(chat_router)
app.include_router(ingestion_router)
app.include_router(dashboard_router)


@app.get("/")
async def dashboard(request: Request):

    # Get saved metrics from PostgreSQL
    metrics = get_metrics()

    # Count unique companies
    total_companies = len(
        set(row["company"] for row in metrics)
    )

    # Number of dashboard records
    total_reports = len(metrics)

    print(
        f"[Dashboard] Loaded {len(metrics)} metric record(s)"
    )

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "request": request,
            "metrics": metrics,
            "total_companies": total_companies,
            "total_reports": total_reports,
        },
    )