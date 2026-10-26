"""CareerPilot AI backend entry point.

This file only wires the web layer together. Request handling lives in
`app/routers`, glue code in `app/services`, and ML logic in `ml/`.
"""

from fastapi import FastAPI

from app.routers import match
from app.schemas import HealthResponse

APP_VERSION = "0.1.0"

app = FastAPI(
    title="CareerPilot AI API",
    version=APP_VERSION,
    description="Job matching and application tracking backend.",
)

app.include_router(match.router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness check: confirms the server process is up and responding."""
    return HealthResponse(status="ok", version=APP_VERSION)
