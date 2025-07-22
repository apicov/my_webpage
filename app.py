from flask import Flask, render_template, request, jsonify
import json
import time
import os

from dotenv import load_dotenv
load_dotenv(override=True)
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

# Your professional info - customize this
PERSONAL_INFO = {
    "name": "Antonio Pico",
    "title": "Computer Scientist", 
    "bio": "Computer Science PhD with experience in machine learning, deep learning, and embedded systems. Currently expanding my focus into two cutting-edge areas: TinyML and LLM agents.",
    "skills": ["Machine Learning", "Deep Learning", "LLMs & Agents", "IoT & Embedded Systems", "TinyML", "Robotics"],
    "experience": [
        {
            "role": "Senior ML Engineer",
            "company": "Tech Company",
            "period": "2022 - Present",
            "description": "Developed and deployed machine learning models for production systems"
        },
        {
            "role": "Python Developer", 
            "company": "Previous Company",
            "period": "2020 - 2022",
            "description": "Built scalable web applications and data processing pipelines"
        }
    ],
    "education": [
        {
            "degree": "Master's in Computer Science",
            "school": "University Name",
            "year": "2020"
        }
    ]
}

# Knowledge base for your AI assistant
KNOWLEDGE_BASE = {
    "skills": "I'm proficient in Python, machine learning, natural language processing, IoT systems, and embedded programming. I work with frameworks like TensorFlow, PyTorch, scikit-learn, and have experience with both supervised and unsupervised learning.",
    
    "experience": "I have 4+ years of experience in Python development and 3+ years specifically in ML engineering. I've worked on projects ranging from chatbots and recommendation systems to IoT data processing and real-time analytics.",
    
    "projects": "Some of my notable projects include: an AI-powered chatbot for customer service, an IoT sensor network for environmental monitoring, a recommendation engine for e-commerce, and several deep learning models for image and text classification.",
    
    "education": "I have a Master's degree in Computer Science with a focus on AI and machine learning. I'm also constantly learning through online courses, research papers, and hands-on projects.",
    
    "technologies": "I work with Python, TensorFlow, PyTorch, scikit-learn, pandas, NumPy, Flask, FastAPI, Docker, AWS, Git, SQL databases, MongoDB, MQTT for IoT, Arduino, Raspberry Pi, and various cloud platforms.",
    
    "approach": "I believe in building practical, scalable solutions. I focus on understanding business requirements first, then selecting the right tools and techniques. I'm passionate about clean code, proper testing, and continuous learning."
}

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
        
        if not messages:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get AI response (replace this with your actual LLM call)
        ai_response = get_ai_response(messages)
        #time.sleep(1)

        
        return jsonify({
            'response': ai_response,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': 'Something went wrong',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)