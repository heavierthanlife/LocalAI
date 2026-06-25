"""Entry point for the AI_Services application.

Usage:
    python run.py
    # or
    waitress-serve --host=0.0.0.0 --port=5000 run:app
"""
import logging
import os
import sys

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, init_services

# Initialize services (DB tables, drivers, models)
init_services()

# Create the Flask application
app = create_app()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(host='0.0.0.0', port=5000, threaded=True)
