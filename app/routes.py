"""Route registration bridges the original app.py into the modular factory.

The original monolithic routes are in _app.py (renamed from app.py to avoid
name conflict with the app/ package). This module provides register_all()
that the app factory calls.
"""
import os
import sys
import logging
import importlib.util

logger = logging.getLogger(__name__)

# Cache the route module to avoid re-importing
_route_module = None


def register_all(flask_app):
    """Register all routes from _app.py onto the Flask app."""
    global _route_module

    # Determine the path to _app.py
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_py_path = os.path.join(base_dir, "app.py")

    # Load the original app module
    spec = importlib.util.spec_from_file_location("_app_module", app_py_path)
    module = importlib.util.module_from_spec(spec)

    # Inject Flask app BEFORE executing the module code
    module.app = flask_app

    # Execute the module (this registers all @app.route decorators)
    sys.modules['_app_module'] = module
    spec.loader.exec_module(module)

    _route_module = module
    logger.info("Routes registered from app.py")

