"""E2E-test fixtures: full stack (DB + optional real Redis + mock LLM).

Reuses integration fixtures and adds full-stack workflow support.
"""
# Re-export everything from integration conftest
from tests.integration.conftest import *
