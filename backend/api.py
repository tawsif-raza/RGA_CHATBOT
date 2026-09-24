"""FastAPI application for cited enterprise RAG answers."""
from __future__ import annotations

import base64
import json
import os
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import phoenix as px
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from phoenix.otel import register
from pydantic import BaseModel, Field

from backend.config import Settings
from backend.database import check_sql_connection, count_ingested_documents, create_sql_engine, get_document_metadata
from backend.guardrails import GREETING_RESPONSE, GUARDRAIL_MESSAGE, DomainRouter, check_guardrails, is_pure_greeting
from backend.model import EnterpriseGenerator
from backend.retriever import RERANKER_MODEL, EnterpriseRetriever, RetrievedChunk

FEEDBACK_LOG_PATH = Path("feedback.jsonl")

_tracer = trace.get_tracer("enterprise_rag.api")

# Phoenix and its OTel tracer provider are process-wide singletons: launching
# the local server or registering a provider twice (e.g. if create_app() is
# called more than once, as tests may do) would try to rebind the same port
# and re-register the global tracer provider, so this guards it to once.
_phoenix_session: Any = None
_tracer_provider: Any = None


def _ensure_observability() -> Any:
    """Register the OTel tracer provider once per process.

    Local dev (default): no PHOENIX_COLLECTOR_ENDPOINT set, so this launches
    an embedded Phoenix server in-process, exactly as before.

    Containerized deployment (docker-compose.prod.yml): PHOENIX_COLLECTOR_ENDPOINT
    points at the standalone `phoenix` service instead, so this registers
    against that external collector rather than also starting a second,
    redundant embedded Phoenix server inside the API container.
    """
    global _phoenix_session, _tracer_provider
    if _tracer_provider is None:
        collector_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
        if collector_endpoint is None:
            _phoenix_session = px.launch_app()
        # auto_instrument=True activates every installed OpenInference
        # instrumentor - here, openinference-instrumentation-langchain - so
        # the retriever's real LangChain retrieval and reranking steps
        # (ContextualCompressionRetriever, CrossEncoderReranker) are traced
        # automatically. The LLM generation step below is *not* a LangChain
        # call (it's raw transformers.generate()), so it gets an explicit span.
        _tracer_provider = register(project_name="enterprise-rag", auto_instrument=True)
    return _tracer_provider


class QueryRequest(BaseModel):
    """Validated public query input."""

    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=3, ge=1, le=10)
    department: str | None = Field(default=None, max_length=128)
    role: str | None = Field(default=None, max_length=128)


CITATION_PREVIEW_CHARACTERS = 500


class Citation(BaseModel):
    """Traceable source for a generated answer."""

    document_id: str
    chunk_id: str
    source_name: str
    score: float
    department: str | None
    role: str | None
    content: str


class QueryResponse(BaseModel):
    """Public RAG response. No longer returned directly by `/query` (streamed instead), but kept as the documented shape citations are encoded in."""

    answer: str
    citations: list[Citation]


class FeedbackRequest(BaseModel):
    """A thumbs up/down rating on one past answer, for the RLHF dataset."""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    rating: int = Field(ge=0, le=1)  # 0 = thumbs down, 1 = thumbs up


def generate_with_telemetry(generator: EnterpriseGenerator, question: str, chunks: list[RetrievedChunk]) -> str:
    """Generate an answer inside a manual OTel span carrying latency and token counts.

    `EnterpriseGenerator.generate()` calls raw `transformers` generation, not
    a LangChain `Runnable`, so it falls outside what the LangChain
    OpenInference auto-instrumentation can see - this span is what makes the
    LLM step show up in Phoenix at all. Kept alongside the streaming version
    below for callers that need the complete answer at once (`evaluate_model.py`).
    """
    with _tracer.start_as_current_span("llm.generate") as span:
        span.set_attribute("llm.chunks_provided", len(chunks))
        started = time.perf_counter()
        answer = generator.generate(question, chunks)
        span.set_attribute("llm.latency_ms", (time.perf_counter() - started) * 1000)
        if chunks and generator.ready and generator.tokenizer is not None:
            prompt_text = generator.tokenizer.apply_chat_template(generator.build_messages(question, chunks), tokenize=False, add_generation_prompt=True)
            span.set_attribute("llm.prompt_tokens", len(generator.tokenizer(prompt_text).input_ids))
            span.set_attribute("llm.completion_tokens", len(generator.tokenizer(answer).input_ids))
        return answer


def generate_with_telemetry_stream(generator: EnterpriseGenerator, question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
    """Stream an answer piece by piece inside a manual OTel span, recording latency and token counts once it completes."""
    with _tracer.start_as_current_span("llm.generate") as span:
        span.set_attribute("llm.chunks_provided", len(chunks))
        started = time.perf_counter()
        pieces: list[str] = []
        for piece in generator.generate_stream(question, chunks):
            pieces.append(piece)
            yield piece
        span.set_attribute("llm.latency_ms", (time.perf_counter() - started) * 1000)
        full_answer = "".join(pieces)
        if chunks and generator.ready and generator.tokenizer is not None:
            prompt_text = generator.tokenizer.apply_chat_template(generator.build_messages(question, chunks), tokenize=False, add_generation_prompt=True)
            span.set_attribute("llm.prompt_tokens", len(generator.tokenizer(prompt_text).input_ids))
            span.set_attribute("llm.completion_tokens", len(generator.tokenizer(full_answer).input_ids))


def encode_citations_header(citations: list[Citation]) -> str:
    """Base64-encode citations as JSON for an HTTP header.

    Headers must be ASCII/Latin-1 (RFC 7230); citation text pulled from real
    enterprise chat/email content can contain arbitrary Unicode, so raw JSON
    isn't safe to put in a header value directly.
    """
    payload = json.dumps([citation.model_dump() for citation in citations])
    return base64.b64encode(payload.encode("utf-8")).decode("ascii")


CITATIONS_STREAM_MARKER_PREFIX = "\n\n<!--CITATIONS:"
CITATIONS_STREAM_MARKER_SUFFIX = "-->"


def stream_with_citations_marker(pieces: Iterator[str], citations_header: str) -> Iterator[str]:
    """Append the same base64 citations payload as an in-band sentinel after the answer text.

    The Next.js frontend's `useChat` (Vercel AI SDK `@ai-sdk/react`) only
    hands the streamed response *body* to the caller - `HttpChatTransport`
    reads `response.body` and never surfaces `response.headers` to the React
    hook - so `X-Citations` (still sent as a header below, unchanged, for any
    other consumer) is unreachable from that hook. This appends the same
    payload in-band instead, after a delimiter the frontend strips before
    display.
    """
    yield from pieces
    yield f"{CITATIONS_STREAM_MARKER_PREFIX}{citations_header}{CITATIONS_STREAM_MARKER_SUFFIX}"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the application and defer external connections until startup."""
    configuration = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        app.state.settings = configuration
        app.state.sql_engine = create_sql_engine(configuration) if configuration.mssql_sa_password else None
        app.state.retriever = EnterpriseRetriever(configuration) if configuration.pinecone_api_key else None
        app.state.domain_router = DomainRouter(app.state.retriever.embeddings) if app.state.retriever else None
        app.state.generator = EnterpriseGenerator(configuration)
        app.state.generator.load()
        yield

    app = FastAPI(title="Enterprise RAG API", version="1.0.0", lifespan=lifespan)
    # The Next.js frontend (frontend-next, Vercel AI SDK) runs on a separate
    # origin (localhost:3000) in dev, unlike the deprecated Streamlit
    # frontend which talked to this API server-side with no browser CORS
    # involved at all. expose_headers is required for X-Citations to be
    # readable via `fetch`/`Response.headers` at all - browsers hide
    # non-CORS-safelisted response headers from JS by default otherwise.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Citations"],
    )
    tracer_provider = _ensure_observability()
    FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        """Report dependency readiness and live index metrics, without exposing configuration."""
        engine = request.app.state.sql_engine
        retriever: EnterpriseRetriever | None = request.app.state.retriever
        generator: EnterpriseGenerator = request.app.state.generator
        document_count = None
        vector_count = None
        if engine:
            try:
                document_count = count_ingested_documents(engine)
            except Exception:
                document_count = None
        if retriever:
            try:
                vector_count = retriever.index.describe_index_stats().total_vector_count
            except Exception:
                vector_count = None
        return {
            "api": True,
            "model": generator.ready,
            "mssql": bool(engine and check_sql_connection(engine)),
            "pinecone": bool(retriever and retriever.health()),
            "document_count": document_count,
            "vector_count": vector_count,
            "hybrid_search": bool(retriever and retriever.bm25_retriever is not None),
            "reranker_model": RERANKER_MODEL,
        }

    @app.post("/query")
    def query(payload: QueryRequest, request: Request) -> StreamingResponse:
        """Retrieve sources, then stream a generated, cited answer.

        Citations are known before generation starts (retrieval happens
        first), so they travel in the `X-Citations` header *and* as a
        trailing in-band sentinel (see `stream_with_citations_marker`) -
        the header for any plain HTTP consumer, the sentinel because the
        Next.js frontend's `useChat` hook only sees the response body.

        Kept as plain `text/plain` streaming, not `text/event-stream`/SSE:
        the installed `@ai-sdk/react`/`ai` versions (v4/v7) use a
        `ChatTransport` abstraction rather than the older Vercel Data
        Stream Protocol (`0:"<token>"\\n` framing). `TextStreamChatTransport`
        - the transport frontend-next/app/page.tsx uses - decodes the raw
        response body as plain text, so this stays unchanged from before.
        """
        retriever: EnterpriseRetriever | None = request.app.state.retriever
        generator: EnterpriseGenerator = request.app.state.generator
        if retriever is None:
            raise HTTPException(status_code=503, detail="Pinecone is not configured.")
        if is_pure_greeting(payload.question):
            # Short-circuits entirely before the domain-similarity check,
            # the hybrid EnsembleRetriever, and generation - a bare "hi" has
            # no document to retrieve and previously fell below the domain
            # threshold (built for corporate-document questions, not
            # greetings) and was incorrectly blocked as out-of-domain.
            citations_header = encode_citations_header([])
            return StreamingResponse(
                stream_with_citations_marker(iter([GREETING_RESPONSE]), citations_header),
                media_type="text/plain",
                headers={"X-Citations": citations_header},
            )
        domain_router: DomainRouter = request.app.state.domain_router
        with _tracer.start_as_current_span("guardrails.check") as span:
            violation = check_guardrails(payload.question, domain_router)
            span.set_attribute("guardrails.violation", violation or "none")
        if violation is not None:
            raise HTTPException(status_code=400, detail=GUARDRAIL_MESSAGE)
        try:
            chunks = retriever.search(payload.question, payload.top_k, payload.department, payload.role)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        metadata: dict[str, dict[str, Any]] = {}
        engine = request.app.state.sql_engine
        if engine and chunks:
            metadata = get_document_metadata(engine, list({chunk.document_id for chunk in chunks}))
        citations = [Citation(document_id=chunk.document_id, chunk_id=chunk.chunk_id, source_name=str(metadata.get(chunk.document_id, {}).get("source_name", chunk.source_name)), score=chunk.score, department=chunk.department, role=chunk.role, content=chunk.content[:CITATION_PREVIEW_CHARACTERS]) for chunk in chunks]
        citations_header = encode_citations_header(citations)
        return StreamingResponse(
            stream_with_citations_marker(generate_with_telemetry_stream(generator, payload.question, chunks), citations_header),
            media_type="text/plain",
            headers={"X-Citations": citations_header},
        )

    @app.post("/feedback")
    def feedback(payload: FeedbackRequest) -> dict[str, bool]:
        """Append a thumbs up/down rating on a past answer to the local RLHF dataset."""
        record = {**payload.model_dump(), "logged_at": datetime.now(UTC).isoformat()}
        with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return {"logged": True}

    return app


app = create_app()
