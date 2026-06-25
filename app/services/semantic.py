"""Semantic similarity model management."""
import logging
from sentence_transformers import SentenceTransformer as SenTran
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

logger = logging.getLogger(__name__)

_semantic_model = None
_semantic_model_load_failed = False


def get_semantic_model():
    """Load (or retrieve cached) the sentence-transformers model."""
    global _semantic_model, _semantic_model_load_failed
    if _semantic_model is not None:
        return _semantic_model
    if _semantic_model_load_failed:
        return None
    try:
        _semantic_model = SenTran('distiluse-base-multilingual-cased', local_files_only=True)
        logger.info("Semantic model loaded successfully.")
        return _semantic_model
    except Exception as e:
        _semantic_model_load_failed = True
        logger.error(f"Failed to load semantic model: {e}")
        return None


def compute_batch_semantic_similarity(texts, model=None):
    """Compute pairwise semantic similarity matrix for a list of texts."""
    if model is None:
        model = get_semantic_model()
    if model is None:
        return None
    try:
        embeddings = model.encode(texts, show_progress_bar=False)
        sim_matrix = cosine_similarity(embeddings)
        return sim_matrix
    except Exception as e:
        logger.error(f"Semantic similarity computation failed: {e}")
        return None
