from flask import Flask

def create_app():
    app = Flask(__name__)
    from flask_app import routes
    app.register_blueprint(routes.bp)
    return app
