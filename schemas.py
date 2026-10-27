from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any


class PredictRequest(BaseModel):
    """Schema for the prediction request."""
    text: str = Field(
        ...,
        description="The English text to analyze sentiment for.",
        examples=["I absolutely love using FastAPI, it makes building APIs effortless and fun!"]
    )

    @field_validator("text")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("The 'text' field cannot be empty or contain only whitespace.")
        if len(trimmed) > 2000:
            raise ValueError("The 'text' field exceeds the maximum limit of 2000 characters.")
        return trimmed


class PredictResponse(BaseModel):
    """Schema for the prediction response."""
    text: str = Field(..., description="The original input text.")
    label: str = Field(..., description="The predicted sentiment label (e.g., POSITIVE or NEGATIVE).")
    score: float = Field(..., description="The confidence score of the prediction (between 0.0 and 1.0).")
    inference_time_ms: float = Field(..., description="Execution time of the model inference in milliseconds.")


class HealthResponse(BaseModel):
    """Schema for the health check response."""
    status: str = Field(..., description="Operational status of the API service.")
    model_name: str = Field(..., description="Name of the Hugging Face model loaded.")
    model_loaded: bool = Field(..., description="Whether the model is currently loaded in memory.")
    device: str = Field(..., description="Device used for inference (CPU or CUDA).")


class RootResponse(BaseModel):
    """Schema for the root welcome/info response."""
    title: str = Field(..., description="API System Title.")
    description: str = Field(..., description="Brief summary of the system and its capabilities.")
    version: str = Field(..., description="Application version.")
    docs_url: str = Field(..., description="URL to interactive Swagger UI documentation.")
    endpoints: Dict[str, str] = Field(..., description="Key available endpoints and their purpose.")
