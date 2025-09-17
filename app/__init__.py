# app/__init__.py
import os
from flask import Flask
from datetime import datetime

def create_app():
    app = Flask(
        __name__,
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    )

    @app.context_processor
    def inject_now():
        return {"current_year": datetime.now().year}

    from .routes import main
    app.register_blueprint(main)

    return app
