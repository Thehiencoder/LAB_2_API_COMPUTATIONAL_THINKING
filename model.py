import time
import logging
import torch
from transformers import pipeline
from typing import Dict, Any

logger = logging.getLogger(__name__)

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


class SentimentClassifier:
    """Wrapper class for Hugging Face Sentiment Analysis pipeline."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipeline = None

    def load_model(self) -> None:
        """Loads the Hugging Face pipeline into memory."""
        logger.info(f"Loading model '{self.model_name}' on device: {self.device.upper()}...")
        device_id = 0 if self.device == "cuda" else -1
        self._pipeline = pipeline(
            task="sentiment-analysis",
            model=self.model_name,
            device=device_id
        )
        logger.info(f"Model '{self.model_name}' loaded successfully.")

    @property
    def is_loaded(self) -> bool:
        """Returns True if the model pipeline is loaded."""
        return self._pipeline is not None

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Runs sentiment analysis inference on the input text.
        
        Returns:
            dict containing:
                - label: 'POSITIVE' or 'NEGATIVE'
                - score: float confidence score (0.0 - 1.0)
                - inference_time_ms: inference latency in milliseconds
        """
        if not self.is_loaded:
            raise RuntimeError("Model has not been loaded yet. Please call load_model() first.")

        start_time = time.perf_counter()
        raw_result = self._pipeline(text)[0]
        end_time = time.perf_counter()

        inference_time_ms = round((end_time - start_time) * 1000, 2)
        score = round(float(raw_result["score"]), 4)
        label = raw_result["label"]

        return {
            "label": label,
            "score": score,
            "inference_time_ms": inference_time_ms
        }
