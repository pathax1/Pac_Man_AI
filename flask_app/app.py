from flask import Flask, render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/play')
def play():
    subprocess.Popen(["python", r"C:\Users\Autom\PycharmProjects\Pac_Man_AI\game\game_controller.py"])
    return "<h1>Game Started! Check the Pygame window.</h1>"

if __name__ == "__main__":
    app.run(debug=True)