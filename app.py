"""
FastAPI wrapper for the PageIndex vectorless RAG pipeline.

Run locally:
    uv run uvicorn app:app --reload

Then open:
    http://127.0.0.1:8000
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from single import (
    DEFAULT_ANSWER_MODEL,
    DEFAULT_PDF_PATH,
    DEFAULT_TREE_SEARCH_MODEL,
    count_nodes,
    create_clients,
    fetch_tree,
    load_api_keys,
    upload_document,
    vectorless_rag,
    wait_for_tree_index,
)


BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "templates" / "chat.html"


app = FastAPI(
    title="PageIndex Vectorless RAG API",
    description="FastAPI endpoints for uploading/indexing a PDF and asking questions over its PageIndex tree.",
    version="1.0.0",
)


class LoadDocumentRequest(BaseModel):
    """Payload for loading either a new PDF or an existing PageIndex document id."""

    pdf_path: str | None = Field(
        default=DEFAULT_PDF_PATH,
        description="Local PDF path to upload when doc_id is not provided.",
    )
    doc_id: str | None = Field(
        default=None,
        description="Existing PageIndex document id to reuse instead of uploading a PDF.",
    )
    tree_model: str = Field(
        default=DEFAULT_TREE_SEARCH_MODEL,
        description="Groq model used to reason over the PageIndex tree.",
    )


class AskRequest(BaseModel):
    """Payload for asking a question against the loaded document."""

    query: str = Field(..., min_length=1)
    answer_model: str = Field(default=DEFAULT_ANSWER_MODEL)


class AppState:
    """In-memory state for the currently loaded document."""

    def __init__(self) -> None:
        self.doc_id: str | None = None
        self.tree: list[dict[str, Any]] = []
        self.tree_search_llm: Any | None = None
        self.tree_model: str = DEFAULT_TREE_SEARCH_MODEL

    @property
    def is_ready(self) -> bool:
        return bool(self.doc_id and self.tree and self.tree_search_llm)


state = AppState()


@app.get("/")
def home() -> FileResponse:
    """Serve the Tailwind HTML UI."""
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=500, detail="templates/index.html is missing.")
    return FileResponse(INDEX_HTML)


@app.get("/api/health")
def health() -> dict[str, str]:
    """Simple health check for the frontend or monitoring."""
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict[str, Any]:
    """Return the currently loaded document state."""
    return {
        "ready": state.is_ready,
        "doc_id": state.doc_id,
        "tree_model": state.tree_model,
        "top_level_sections": len(state.tree),
        "total_nodes": count_nodes(state.tree) if state.tree else 0,
    }


@app.post("/api/load-document")
async def load_document(request: LoadDocumentRequest) -> dict[str, Any]:
    """
    Load a document into memory.

    If doc_id is provided, the API reuses an existing PageIndex document.
    Otherwise, it uploads pdf_path and waits until PageIndex finishes building the tree.
    """
    try:
        pageindex_api_key, _groq_api_key = load_api_keys()
        pageindex_client, tree_search_llm = create_clients(pageindex_api_key, request.tree_model)

        if request.doc_id:
            doc_id = request.doc_id
        else:
            if not request.pdf_path:
                raise HTTPException(status_code=400, detail="Provide either doc_id or pdf_path.")

            pdf_path = Path(request.pdf_path)
            if not pdf_path.is_absolute():
                pdf_path = BASE_DIR / pdf_path

            doc_id = await run_in_threadpool(upload_document, pageindex_client, pdf_path)
            await run_in_threadpool(wait_for_tree_index, pageindex_client, doc_id)

        tree = await run_in_threadpool(fetch_tree, pageindex_client, doc_id)

        state.doc_id = doc_id
        state.tree = tree
        state.tree_search_llm = tree_search_llm
        state.tree_model = request.tree_model

        return {
            "ready": state.is_ready,
            "doc_id": state.doc_id,
            "top_level_sections": len(state.tree),
            "total_nodes": count_nodes(state.tree),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/tree")
def get_tree() -> dict[str, Any]:
    """Return the loaded PageIndex tree."""
    if not state.is_ready:
        raise HTTPException(status_code=400, detail="No document is loaded yet.")

    return {
        "doc_id": state.doc_id,
        "top_level_sections": len(state.tree),
        "total_nodes": count_nodes(state.tree),
        "tree": state.tree,
    }


@app.post("/api/ask")
async def ask(request: AskRequest) -> dict[str, Any]:
    """Answer a question using the loaded document tree."""
    if not state.is_ready:
        raise HTTPException(status_code=400, detail="Load a document before asking questions.")

    try:
        answer = await run_in_threadpool(
            vectorless_rag,
            query=request.query,
            tree=state.tree,
            tree_search_llm=state.tree_search_llm, # type: ignore
            answer_model=request.answer_model,
            verbose=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "query": request.query,
        "answer": answer,
        "doc_id": state.doc_id,
    }
