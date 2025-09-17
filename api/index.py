# api/index.py
import os, sys

# Make sure Python can import from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import create_app

# Call the factory to get the Flask app
app = create_app()
