"""Runtime-configurable settings — admin-adjustable via API, persisted to JSON.

Load order: factory presets → runtime_config.json overrides → env var overrides (highest priority).
Factory presets are an immutable baseline saved once; admin can restore to factory at any time.
"""

import os, json, logging
from threading import Lock
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CONFIG_PATH = DATA_DIR / "runtime_config.json"
FACTORY_PATH = DATA_DIR / "runtime_config_factory.json"
_lock = Lock()

# ── Defaults (hardcoded — fallback when no factory preset exists) ──
DEFAULTS = {
    # LLM / Agent
    "llm_timeout_seconds":      120,
    "llm_max_tokens":           1600,
    "llm_max_tokens_min":       100,
    "llm_max_tokens_max":       4800,
    "llm_temperature":          0.7,
    "llm_batch_timeout_seconds": 90,

    # Active LLM provider & model (NOT factory — admin selects, "" = auto-detect)
    "active_llm_provider":      "",
    "active_llm_model":         "",

    # Bocha search cache
    "search_cache_ttl_hours":   72,
    "headroom_enabled":         True,  # compress RAG/file/history before LLM
    "judge_review_enabled":     False, # 2nd-model quality check (costs extra API call)

    # VL model
    "vl_max_image_size":        1024,
    "vl_jpeg_quality":          85,
    "vl_max_tokens":            800,
    "vl_temperature":           0.7,

    # RAG engine
    "rag_chunk_size":           500,
    "rag_chunk_overlap":        100,
    "rag_top_k_default":        8,
    "rag_max_context_chars":    8000,
    "rag_min_chunk_chars":      20,

    # File cache
    "file_cache_max_age_hours": 24,
    "file_cache_max_cached_files": 10,
    "file_cache_max_content_size": 51200,

    # File processing
    "file_template_similarity_threshold": 0.85,
    "file_keywords_top_k":      20,
    "file_semantic_batch_size": 32,
    "file_ocr_zoom":            2.0,
    "file_ocr_max_dim":         2000,
    "file_name_max_len":        40,

    # Session
    "session_title_max_len":    20,

    # Cleanup schedules (days or hours as noted)
    "cleanup_session_days":     15,
    "cleanup_anon_temp_days":   1,
    "cleanup_project_deletion_days": 30,
    "cleanup_share_file_days":  7,
    "cleanup_download_token_hours": 24,
    "cleanup_report_retention_days": 90,
    "cleanup_recycle_bin_days": 3,
    "cleanup_original_file_days": 3,
    "cleanup_message_response_hours": 1,

    # Rate limits
    "ratelimit_admin_max":      5,
    "ratelimit_admin_window_seconds": 1800,
    "ratelimit_credit_max":     10,
    "ratelimit_credit_window_seconds": 300,

    # Anonymous limits
    "anon_max_files":           5,
    "anon_max_file_size_mb":    5,
    "anon_message_max_chars":   10000,
    "storage_warn_threshold_mb": 300,

    # Training data
    "training_min_rating":      3,
    "training_min_length":      100,
    "training_retention_days":  90,   # auto-cleanup training raw/ data older than this
    "export_retention_count":   20,   # keep last N export files, auto-delete older
    "ingest_review_warn_days":  3,    # warn after N days of unreviewed ingest data

    # Auto report
    "report_min_messages":      5,
    "report_retention_days":    90,

    # Web extractor
    "web_extract_retries":      3,
    "web_extract_timeout_seconds": 15,

    # Max upload size (MB, informational — actual change requires server restart)
    "max_upload_size_mb":       50,

    # Task lock timeout
    "task_timeout_seconds":     600,

    # ── Quote anomaly detection thresholds ──
    "quote_anomaly_same_rate_threshold":      0.05,   # relative diff below which bids are "same-rate"
    "quote_anomaly_drop_threshold":           0.30,   # drop below reference/mean that triggers alert
    "quote_anomaly_cv_low_alert":             0.05,   # CV below this = possible collusion
    "quote_anomaly_cv_high_alert":            1.5,    # CV above this = unusual dispersion
    "quote_anomaly_benford_deviation_alert":  0.15,   # MAD from Benford's Law that triggers alert
    "quote_anomaly_clustering_bandwidth":     0.02,   # fraction of value range for cluster detection
    "quote_anomaly_min_cluster_size":         3,      # minimum bids in a cluster to flag
    "quote_anomaly_min_prices_for_benford":   20,     # minimum values needed for Benford analysis

    # ── Relationship extraction ──
    "relation_extraction_ner_provider":  "hanlp",     # "hanlp" or "llm"
    "relation_extraction_llm_fallback":  True,        # use LLM for relationship classification
    "relation_tianyancha_enabled":       False,       # 天眼查 API toggle
    "relation_tianyancha_api_key":       "",          # 天眼查 API key (encrypted at rest)

    # ── Typo detection ──
    "typo_chinese_enabled":       True,
    "typo_english_enabled":       True,
    "typo_numeric_enabled":       True,
    "typo_daxie_enabled":         True,              # 大写金额 validation
    "typo_auto_correct":          False,             # false = suggest only
    "typo_diff_review_enabled":   False,             # opt-in before/after diff mode
    "typo_min_confidence":        0.70,              # minimum confidence to suggest
}

# Keys that are NOT part of factory presets (admin-only runtime choices)
NON_FACTORY_KEYS = {"active_llm_provider", "active_llm_model"}


def _load():
    """Return merged config dict: factory → runtime overrides."""
    cfg = dict(DEFAULTS)
    # Layer 1: factory presets (if saved)
    if FACTORY_PATH.exists():
        try:
            with open(FACTORY_PATH, 'r', encoding='utf-8') as f:
                factory = json.load(f)
            if isinstance(factory, dict):
                for k, v in factory.items():
                    if k in DEFAULTS:
                        cfg[k] = v
        except Exception as e:
            logger.warning(f"Failed to load factory presets: {e}")
    # Layer 2: runtime overrides
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                overrides = json.load(f)
            if isinstance(overrides, dict):
                for k, v in overrides.items():
                    if k in DEFAULTS:
                        cfg[k] = v
        except Exception as e:
            logger.warning(f"Failed to load runtime config: {e}")
    return cfg


# ── Public API ──

def get_all() -> dict:
    """Return full config dict."""
    with _lock:
        return _load()


def get(key: str, default=None):
    """Read a single config value."""
    with _lock:
        cfg = _load()
        return cfg.get(key, default)


def update(updates: dict) -> dict:
    """Merge admin-provided updates into the runtime JSON file. Returns the new full config."""
    with _lock:
        # Build current state
        current = dict(DEFAULTS)
        if FACTORY_PATH.exists():
            try:
                with open(FACTORY_PATH, 'r', encoding='utf-8') as f:
                    current.update(json.load(f))
            except Exception:
                pass
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    current.update(json.load(f))
            except Exception:
                pass

        for k, v in updates.items():
            if k in DEFAULTS:
                current[k] = v

        # Persist only keys that differ from factory (or DEFAULTS if no factory)
        baseline = dict(DEFAULTS)
        if FACTORY_PATH.exists():
            try:
                with open(FACTORY_PATH, 'r', encoding='utf-8') as f:
                    baseline.update(json.load(f))
            except Exception:
                pass
        persisted = {k: v for k, v in current.items() if v != baseline.get(k)}
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(persisted, f, ensure_ascii=False, indent=2)
        return current


def reset_to_defaults() -> dict:
    """Remove runtime overrides (keep factory)."""
    with _lock:
        if CONFIG_PATH.exists():
            CONFIG_PATH.unlink()
        return _load()


# ── Factory presets ──

def save_factory_presets() -> dict:
    """Save current effective config (excluding non-factory keys) as immutable factory baseline.

    This can only be called ONCE — subsequent calls are no-ops unless forced.
    The factory file must NOT be manually edited or deleted by admin.
    """
    with _lock:
        if FACTORY_PATH.exists():
            logger.info("Factory presets already exist — not overwriting.")
            with open(FACTORY_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)

        current = dict(DEFAULTS)
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    current.update(json.load(f))
            except Exception:
                pass

        # Strip non-factory keys
        factory = {k: v for k, v in current.items() if k not in NON_FACTORY_KEYS}

        FACTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FACTORY_PATH, 'w', encoding='utf-8') as f:
            json.dump(factory, f, ensure_ascii=False, indent=2)

        # Also flag as immutable via OS (best-effort, not critical if fails)
        try:
            os.chmod(FACTORY_PATH, 0o444)  # read-only
        except Exception:
            pass

        logger.info(f"Factory presets saved: {len(factory)} keys → {FACTORY_PATH}")
        return factory


def get_factory_presets() -> dict | None:
    """Return saved factory presets, or None if never saved."""
    if not FACTORY_PATH.exists():
        return None
    try:
        with open(FACTORY_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def has_factory_presets() -> bool:
    """Check if factory presets have been saved."""
    return FACTORY_PATH.exists()


def restore_factory_presets() -> dict:
    """Restore all config to factory preset values.

    Removes runtime_config.json. Non-factory keys (active_llm_*) keep their
    current values since they are not covered by factory presets.
    """
    with _lock:
        # Capture current non-factory values before wiping
        non_factory_values = {}
        old_cfg = _load()
        for k in NON_FACTORY_KEYS:
            non_factory_values[k] = old_cfg.get(k, DEFAULTS.get(k, ""))

        if CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

        if not FACTORY_PATH.exists():
            logger.warning("No factory presets to restore — reverting to hardcoded defaults.")
        else:
            logger.info("Restored to factory presets.")

        # Re-persist non-factory keys so they survive the restore
        cfg_after = _load()
        for k, v in non_factory_values.items():
            if v and v != cfg_after.get(k):
                cfg_after[k] = v

        persisted = {}
        baseline = dict(DEFAULTS)
        if FACTORY_PATH.exists():
            try:
                with open(FACTORY_PATH, 'r', encoding='utf-8') as f:
                    baseline.update(json.load(f))
            except Exception:
                pass
        for k, v in cfg_after.items():
            if v != baseline.get(k):
                persisted[k] = v

        if persisted:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(persisted, f, ensure_ascii=False, indent=2)

        return cfg_after
