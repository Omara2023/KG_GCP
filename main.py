import os
from flask import Flask
from logging_config import setup_logging
from routes.query import query_bp

def create_app():
    """Factory method to produce Flask App."""
    app_logger = setup_logging()
    app = Flask(__name__)
    app.register_blueprint(query_bp)
    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
