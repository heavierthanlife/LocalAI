"""Semantic similarity model management with language-aware model switching.

Models:
- bge-large-zh-v1.5:        Best Chinese STS (C-MTEB #1), 1024-dim, ~1.3GB
- paraphrase-multilingual-MiniLM-L12-v2:  Multilingual (50+ langs), 384-dim, ~420MB
- distiluse-base-multilingual-cased:      Legacy fallback, 512-dim, ~540MB

Strategy: detect language → pick best model → compute embeddings → cosine similarity
"""
import logging
import os
from sentence_transformers import SentenceTransformer as SenTran
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from app.services.langutils import detect_pair_language

logger = logging.getLogger(__name__)

# Model registry: key → (model_name, is_default_for)
_MODEL_REGISTRY = {
    'zh':   ('BAAI/bge-large-zh-v1.5',   'Chinese (bge-large-zh-v1.5)'),
    'en':   ('paraphrase-multilingual-MiniLM-L12-v2', 'Multilingual (paraphrase-multilingual)'),
    'legacy': ('distiluse-base-multilingual-cased',   'Legacy fallback (distiluse)'),
}

# Runtime cache: {model_key: SentenceTransformer instance}
_model_cache = {}
# Track per-key load failures so we don't retry forever
_load_failures = set()


def _get_model_cache_dir():
    """Custom cache dir so downloads go to a predictable location."""
    base = os.environ.get('HF_HOME') or os.path.join(
        os.path.expanduser('~'), '.cache', 'huggingface'
    )
    return os.path.join(base, 'hub')


def _load_model(model_key: str):
    """Load a model by key, with caching."""
    global _model_cache, _load_failures

    if model_key in _model_cache:
        return _model_cache[model_key]
    if model_key in _load_failures:
        return None

    if model_key not in _MODEL_REGISTRY:
        logger.error(f"Unknown model key: {model_key}")
        _load_failures.add(model_key)
        return None

    model_name, display_name = _MODEL_REGISTRY[model_key]

    try:
        logger.info(f"Loading model [{model_key}]: {model_name} ...")
        model = SenTran(model_name)
        _model_cache[model_key] = model
        logger.info(f"Model [{model_key}] ({display_name}) loaded successfully.")
        return model
    except Exception as e:
        _load_failures.add(model_key)
        logger.error(f"Failed to load model [{model_key}] {model_name}: {e}")
        return None


def get_model_for_language(lang: str):
    """Get the best model for a given language code ('zh' or 'en')."""
    return _load_model(lang)


def get_model_for_texts(text1: str, text2: str):
    """Pick the optimal model based on detected language of the input pair.
    
    Returns (model, lang_code) tuple, or (None, None) if no model available.
    """
    lang = detect_pair_language(text1, text2)
    model = _load_model(lang)
    if model:
        return model, lang

    # Fallback: try the other language
    fallback_lang = 'en' if lang == 'zh' else 'zh'
    logger.warning(f"No model for '{lang}', trying fallback '{fallback_lang}'")
    model = _load_model(fallback_lang)
    if model:
        return model, fallback_lang

    # Last resort: legacy model
    logger.warning("Falling back to legacy distiluse model")
    model = _load_model('legacy')
    if model:
        return model, 'legacy'

    return None, None


# ── Legacy / backward-compat API ──

def get_semantic_model():
    """Load (or retrieve cached) the default legacy sentence-transformers model.
    
    Kept for backward compatibility with existing code.
    Prefer get_model_for_texts() for new code.
    """
    return _load_model('legacy')


def compute_batch_semantic_similarity(texts, model=None, lang_code=None):
    """Compute pairwise semantic similarity matrix for a list of texts.
    
    Args:
        texts: list of strings
        model: optional pre-loaded model. If None and lang_code is set, loads language-specific model.
               If both None, uses legacy.
        lang_code: 'zh', 'en', or None (auto-detect from first text)
    
    Returns cosine_similarity matrix or None on failure.
    """
    if model is None:
        if lang_code:
            model = get_model_for_language(lang_code)
        elif texts:
            model, _ = get_model_for_texts(texts[0], texts[-1] if len(texts) > 1 else texts[0])
        else:
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


def get_loaded_models():
    """Return info about currently loaded models (for diagnostics)."""
    return {
        key: {
            'name': _MODEL_REGISTRY.get(key, ('unknown', 'unknown'))[0],
            'loaded': key in _model_cache,
            'failed': key in _load_failures,
        }
        for key in _MODEL_REGISTRY
    }


def preload_models():
    """Pre-load all configured models at startup (call from app init)."""
    for key in _MODEL_REGISTRY:
        _load_model(key)
