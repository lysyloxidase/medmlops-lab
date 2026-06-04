# pyright: reportUnusedFunction=false
"""FastAPI app for Phase 4 clinical serving."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

from medmlops.serving.predict import PredictionService, RegistryLoader
from medmlops.serving.schemas import (
    BatchPredictionResponse,
    ClinicalEncounter,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)


def _get_service(request: Request) -> PredictionService:
    service = getattr(request.app.state, "service", None)
    if not isinstance(service, PredictionService):
        msg = "Prediction service is not initialized"
        raise RuntimeError(msg)
    return service


def create_app(
    *,
    service: PredictionService | None = None,
    params_path: str | Path = "params.yaml",
    registry_loader: RegistryLoader | None = None,
) -> FastAPI:
    """Create the FastAPI app and warm-load model artifacts on startup."""

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
        if service is not None:
            app_instance.state.service = service
        else:
            loader = registry_loader
            if loader is None:
                app_instance.state.service = PredictionService.from_params(params_path)
            else:
                app_instance.state.service = PredictionService.from_params(
                    params_path,
                    registry_loader=loader,
                )
        yield

    api = FastAPI(
        title="MedMLOps-Lab Clinical Risk API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @api.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        return _get_service(request).health()

    @api.get("/metrics", response_class=PlainTextResponse)
    def metrics(request: Request) -> PlainTextResponse:
        return PlainTextResponse(
            _get_service(request).prometheus_metrics(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @api.get("/model-info", response_model=ModelInfoResponse)
    def model_info(request: Request) -> ModelInfoResponse:
        return _get_service(request).model_info()

    @api.post("/predict", response_model=PredictionResponse)
    def predict(
        encounter: ClinicalEncounter,
        request: Request,
    ) -> PredictionResponse:
        return _get_service(request).predict_one(encounter)

    @api.post("/batch-predict", response_model=BatchPredictionResponse)
    def batch_predict(
        encounters: list[ClinicalEncounter],
        request: Request,
    ) -> BatchPredictionResponse:
        return _get_service(request).predict_batch(encounters)

    return api


app = create_app()
