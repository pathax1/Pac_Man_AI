from flask import Flask
from flask_app.routes import bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(bp)
    return app
