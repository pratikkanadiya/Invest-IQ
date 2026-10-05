from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from database.get_metrics import get_metrics

router = APIRouter(tags=["Dashboard"])

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@router.get("/")
async def dashboard(
    request: Request,
    company: str | None = None
):
    all_metrics = get_metrics()

    if company:
        metrics = [
            row for row in all_metrics
            if row["company"].lower() == company.lower()
        ]
    else:
        metrics = all_metrics

    total_companies = len(
        set(row["company"] for row in metrics)
    )

    total_reports = len(metrics)

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "request": request,
            "metrics": metrics,
            "total_companies": total_companies,
            "total_reports": total_reports,
            "selected_company": company,
        },
    )


@router.get("/api/metrics")
def metrics():
    data = get_metrics()

    return {
        "status": "success",
        "count": len(data),
        "data": data,
    }