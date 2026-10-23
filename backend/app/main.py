"""CareerPilot AI backend entry point."""

from fastapi import FastAPI

from app.schemas import HealthResponse

APP_VERSION = "0.1.0"

app = FastAPI(
    title="CareerPilot AI API",
    version=APP_VERSION,
    description="Job matching and application tracking backend.",
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness check: confirms the server process is up and responding."""
    return HealthResponse(status="ok", version=APP_VERSION)
