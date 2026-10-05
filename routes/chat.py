from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from rag.graph import rag_graph


router = APIRouter(prefix="/api", tags=["Chat"])


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    company: str | None = None
    year: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict] = Field(default_factory=list)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    try:
        question = request.question.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )

        # Pass optional company/year information into LangGraph
        result = rag_graph.invoke(
            {
                "question": question,
                "company": request.company,
                "year": request.year,
            }
        )

        return ChatResponse(
            answer=result.get(
                "generation",
                "I could not generate an answer."
            ),
            sources=result.get(
                "sources",
                []
            )
        )

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"RAG processing failed: {str(e)}"
        )