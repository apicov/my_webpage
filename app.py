from flask import Flask, render_template, request, jsonify
import json
import time
import os

from dotenv import load_dotenv
load_dotenv(override=True)
print("Loaded .env file")
from  ai_assistant import Assistant


app = Flask(__name__)

name = os.getenv("MY_NAME")
last_name = os.getenv("MY_LAST_NAME")

# Load the summary and resume
with open("./data/summary.txt", "r", encoding="utf-8") as f:
    summary = f.read()
with open("./data/resume.md", "r", encoding="utf-8") as f:
    resume = f.read()

assistant = Assistant(name, last_name, summary, resume)

# Load personal info from JSON file
with open('./data/personal_info.json', 'r', encoding='utf-8') as f:
    PERSONAL_INFO = json.load(f)

def message_to_dict(msg):
    # If it's already a dict, return as is
    if isinstance(msg, dict):
        return msg
    # If it has a .to_dict() method, use it
    if hasattr(msg, 'to_dict'):
        return msg.to_dict()
    # Otherwise, use vars() (works for most objects)
    return vars(msg)


def get_ai_response(messages):
    response = assistant.get_response(messages)
    return response
    
@app.route('/')
def home():
    return render_template('homepage.html', info=PERSONAL_INFO)

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        print(messages)
        
        if not messages:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get AI response
        ai_response = get_ai_response(messages)
        messages_dicts = [message_to_dict(m) for m in ai_response]
        print(messages_dicts)
        #time.sleep(1)

        
        return jsonify({
            'response': messages_dicts,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': 'Something went wrong',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)