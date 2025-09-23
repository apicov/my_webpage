from flask import Flask
from flask_cors import CORS
import os

from dotenv import load_dotenv
load_dotenv(override=True)
from agents.chat_agent import ChatAgent

# Import blueprints
from routes.main_routes import main_bp
from routes.stream_routes import stream_bp
from routes.tictactoe_routes import tictactoe_bp

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

name = os.getenv("MY_NAME")
last_name = os.getenv("MY_LAST_NAME")

# Load the summary and resume
with open("./data/summary.txt", "r", encoding="utf-8") as f:
    summary = f.read()
with open("./data/resume.md", "r", encoding="utf-8") as f:
    resume = f.read()

# Use ChatAgent directly instead of orchestrator
chat_agent = ChatAgent(name, last_name, summary, resume)

# Attach chat agent to main blueprint so routes can access it
main_bp.chat_agent = chat_agent

# Register blueprints
app.register_blueprint(main_bp)
app.register_blueprint(stream_bp, url_prefix='/stream')
app.register_blueprint(tictactoe_bp, url_prefix='/tictactoe')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)