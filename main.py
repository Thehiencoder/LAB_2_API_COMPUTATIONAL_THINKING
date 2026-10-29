import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
import uvicorn

from schemas import (
    PredictRequest,
    PredictResponse,
    HealthResponse,
    RootResponse
)
from model import SentimentClassifier, MODEL_NAME

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("api")

# Global classifier instance
classifier = SentimentClassifier(model_name=MODEL_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle model startup and shutdown."""
    logger.info("Initializing API service and pre-loading Hugging Face model...")
    try:
        classifier.load_model()
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
    yield
    logger.info("Shutting down API service...")


app = FastAPI(
    title="Text Sentiment Analysis API",
    description=(
        "FastAPI service deploying Hugging Face's "
        f"'{MODEL_NAME}' for binary sentiment classification (Positive/Negative)."
    ),
    version="1.0.0",
    lifespan=lifespan
)


@app.get(
    "/",
    response_model=RootResponse,
    status_code=status.HTTP_200_OK,
    summary="API Information & Root Welcome",
    tags=["General"]
)
async def root():
    """Returns a brief introduction to the system and available endpoints."""
    return RootResponse(
        title="Text Sentiment Analysis API",
        description=(
            "A Web API built with FastAPI that provides sentiment analysis using "
            f"the fine-tuned DistilBERT model ({MODEL_NAME}) from Hugging Face."
        ),
        version="1.0.0",
        docs_url="/docs",
        endpoints={
            "GET /": "Service introduction and API metadata.",
            "GET /health": "System and model operational health check.",
            "POST /predict": "Submit English text to predict sentiment (POSITIVE/NEGATIVE)."
        }
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="System Health Check",
    tags=["System"]
)
async def health():
    """Checks the operational status of the service and confirms the model is loaded."""
    if not classifier.is_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "model_name": classifier.model_name,
                "model_loaded": False,
                "device": classifier.device,
                "detail": "Model is not loaded or initialization failed."
            }
        )

    return HealthResponse(
        status="healthy",
        model_name=classifier.model_name,
        model_loaded=classifier.is_loaded,
        device=classifier.device
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Sentiment of English Text",
    tags=["Inference"]
)
async def predict(payload: PredictRequest):
    """
    Receives English input text and returns sentiment classification result:
    - **label**: 'POSITIVE' or 'NEGATIVE'
    - **score**: Confidence score between 0.0 and 1.0
    - **inference_time_ms**: Time taken for model inference
    """
    if not classifier.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently not ready to serve predictions."
        )

    try:
        prediction = classifier.predict(payload.text)
        return PredictResponse(
            text=payload.text,
            label=prediction["label"],
            score=prediction["score"],
            inference_time_ms=prediction["inference_time_ms"]
        )
    except Exception as exc:
        logger.error(f"Inference error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {str(exc)}"
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
