"""
DiffServe HTTP API Server.

Provides an OpenAI-compatible HTTP API for dLLM serving:
  - POST /v1/completions  — text completion
  - GET  /v1/models       — list available models
  - GET  /health          — server health check
  - GET  /metrics         — serving metrics

Uses FastAPI with uvicorn. The engine runs in the same event loop
as the HTTP server, using asyncio for concurrency.
"""

import asyncio
import logging
import time
import uuid
from typing import Optional

import torch

from .config import DiffServeConfig
from .engine import DiffServeEngine

logger = logging.getLogger(__name__)

# Lazy imports — only when server is actually started
app = None


def create_app(engine: DiffServeEngine, config: DiffServeConfig):
    """Create the FastAPI application with routes bound to the engine."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse, StreamingResponse
        from pydantic import BaseModel, Field
    except ImportError:
        raise ImportError(
            "FastAPI is required for the HTTP server. "
            "Install with: pip install fastapi uvicorn")

    app = FastAPI(
        title="DiffServe",
        description="Online Serving for Diffusion LLMs with CW-SRPT Scheduling",
        version="0.1.0",
    )

    # Store reference to engine
    app.state.engine = engine
    app.state.config = config
    app.state.tokenizer = engine.tokenizer

    # ─── Request/Response models ──────────────────────────────────

    class CompletionRequest(BaseModel):
        prompt: str
        max_tokens: int = Field(default=128, ge=1, le=2048)
        threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
        stream: bool = False

    class CompletionChoice(BaseModel):
        text: str
        index: int = 0
        finish_reason: str = "stop"

    class UsageInfo(BaseModel):
        prompt_tokens: int
        completion_tokens: int
        total_tokens: int

    class CompletionResponse(BaseModel):
        id: str
        object: str = "text_completion"
        created: int
        model: str
        choices: list
        usage: UsageInfo

    # ─── Routes ───────────────────────────────────────────────────

    @app.on_event("startup")
    async def startup():
        await engine.start()
        logger.info(f"DiffServe API server started on {config.host}:{config.port}")

    @app.on_event("shutdown")
    async def shutdown():
        await engine.stop()
        logger.info("DiffServe API server stopped")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "active_requests": engine.active_count,
            "pending_requests": engine.pending_count,
            "policy": config.policy,
            "model": config.model_path.split("/")[-1],
        }

    @app.get("/metrics")
    async def metrics():
        return engine.metrics

    @app.get("/v1/models")
    async def list_models():
        model_name = config.model_path.split("/")[-1]
        return {
            "object": "list",
            "data": [{
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "diffserve",
            }],
        }

    @app.post("/v1/completions")
    async def create_completion(request: CompletionRequest):
        tokenizer = app.state.tokenizer

        # Encode prompt with chat template
        full_prompt = (
            f'<role>SYSTEM</role>detailed thinking off<|role_end|>'
            f'<role>HUMAN</role>{request.prompt}<|role_end|>'
            f'<role>ASSISTANT</role>'
        )
        prompt_ids = tokenizer.encode(
            full_prompt, return_tensors='pt'
        ).squeeze(0).to(engine.device)

        threshold = request.threshold or config.threshold

        # Submit to engine
        rid, future = engine.add_request(
            prompt_ids=prompt_ids,
            gen_length=request.max_tokens,
            threshold=threshold,
        )

        # Wait for completion
        try:
            result = await asyncio.wait_for(future, timeout=120.0)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Request timed out after 120 seconds")

        # Build response
        model_name = config.model_path.split("/")[-1]
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=model_name,
            choices=[CompletionChoice(text=result.output_text)],
            usage=UsageInfo(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
            ),
        )

    return app


def run_server(engine: DiffServeEngine, config: DiffServeConfig):
    """Start the HTTP server (blocking)."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn is required for the HTTP server. "
            "Install with: pip install uvicorn")

    app = create_app(engine, config)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
    )
