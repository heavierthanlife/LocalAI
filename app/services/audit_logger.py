"""Detailed audit logger for compare/check operations.

Writes structured, machine-parseable entries to logs/app.log for every
compare or compliance check operation — showing exactly which components
participated, their inputs/outputs, timing, and results.

Log entries use the format:
    [AUDIT] <operation> | <component> | <field>=<value> | ...

Usage:
    from app.services.audit_logger import AuditLogger
    audit = AuditLogger("batch_compare")
    audit.component("text_similarity", input_chars=5230, method="tfidf_cosine")
    audit.result(score=0.85, status="PASS")
    audit.flush()
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("audit")

# Ensure audit logger writes to app.log (inherits from root config)
logger.setLevel(logging.DEBUG)


class AuditLogger:
    """One instance per compare/check operation. Collects component results."""

    def __init__(self, operation: str, operation_id: str = ""):
        self.operation = operation
        self.operation_id = operation_id
        self.start_time = time.time()
        self.components = []
        self._logged_start = False

    def _start(self):
        if not self._logged_start:
            logger.debug(
                f"[AUDIT] BEGIN | op={self.operation} | id={self.operation_id} "
                f"| ts={datetime.now(timezone.utc).isoformat()}"
            )
            self._logged_start = True

    def component(self, name: str, status: str = "OK", **kwargs):
        """Log a component's participation.

        Args:
            name: component identifier (e.g., "text_similarity", "key_info_match")
            status: "OK" | "SKIPPED" | "FAILED" | "DISABLED"
            **kwargs: component-specific metrics (input_chars, method, score, duration_ms, etc.)
        """
        self._start()
        parts = [f"comp={name}", f"status={status}"]
        for k, v in kwargs.items():
            # Truncate long values
            vs = str(v)
            if len(vs) > 200:
                vs = vs[:200] + "..."
            parts.append(f"{k}={vs}")
        logger.debug(f"[AUDIT] COMPONENT | {' | '.join(parts)}")

    def result(self, **kwargs):
        """Log final result summary."""
        self._start()
        parts = [f"op={self.operation}", f"id={self.operation_id}"]
        for k, v in kwargs.items():
            vs = str(v)
            if len(vs) > 500:
                vs = vs[:500] + "..."
            parts.append(f"{k}={vs}")
        elapsed_ms = int((time.time() - self.start_time) * 1000)
        parts.append(f"elapsed_ms={elapsed_ms}")
        logger.debug(f"[AUDIT] RESULT | {' | '.join(parts)}")

    def flush(self):
        """Explicit flush (handlers configured via dictConfig)."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.result(status="ERROR", error=str(exc_val)[:300])
        self.flush()
