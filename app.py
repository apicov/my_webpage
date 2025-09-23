from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import time
import os
import glob
import frontmatter
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)
from agents.chat_agent import ChatAgent
import asyncio


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


async def get_ai_response(messages):
    # Convert messages to LangChain format
    from langchain_core.messages import HumanMessage, AIMessage

    lc_messages = []
    for msg in messages:
        if msg.get("role") == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    # Call chat agent directly
    response = await chat_agent.handle(lc_messages, "web_session", {})

    # Return in expected format
    return [{
        "role": "assistant",
        "content": response.get("content", "I apologize, but I couldn't generate a response.")
    }]
    
@app.route('/chat', methods=['POST'])
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        print(messages)
        
        if not messages:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get AI response
        ai_response = asyncio.run(get_ai_response(messages))
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

@app.route('/user-info')
@app.route('/api/user-info')
def user_info():
    """API endpoint to get user information for React frontend """
    return jsonify(PERSONAL_INFO)

@app.route('/projects')
@app.route('/api/projects')
def get_projects():
    """API endpoint to get all projects from markdown files"""
    try:
        projects = []
        project_dir = Path('./data/projects')
        
        # Create directory if it doesn't exist
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Read all markdown files in the projects directory (including subfolders)
        for md_file in project_dir.glob('**/*.md'):
            with open(md_file, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)
                
                # Extract metadata and content
                project = {
                    'id': post.metadata.get('id', md_file.stem),
                    'title': post.metadata.get('title', 'Untitled Project'),
                    'description': post.metadata.get('description', ''),
                    'technologies': post.metadata.get('technologies', '').split(', ') if isinstance(post.metadata.get('technologies'), str) else post.metadata.get('technologies', []),
                    'status': post.metadata.get('status', 'planned'),
                    'featured': post.metadata.get('featured', False),
                    'githubUrl': post.metadata.get('githubUrl'),
                    'liveUrl': post.metadata.get('liveUrl'),
                    'demoUrl': post.metadata.get('demoUrl'),
                    'startDate': post.metadata.get('startDate'),
                    'endDate': post.metadata.get('endDate'),
                    'thumbnail': post.metadata.get('thumbnail'),
                    'content': post.content  # Full markdown content
                }
                projects.append(project)
        
        # Sort projects: featured first, then by status
        projects.sort(key=lambda x: (not x['featured'], x['status'] != 'completed'))
        
        return jsonify(projects)
    
    except Exception as e:
        print(f"Error loading projects: {e}")
        return jsonify({'error': 'Failed to load projects'}), 500

@app.route('/data/<path:filename>')
@app.route('/api/data/<path:filename>')
def serve_data_files(filename):
    """Serve static files from the data directory"""
    return send_from_directory('./data', filename)

@app.route('/projects/<project_id>')
@app.route('/api/projects/<project_id>')
def get_project(project_id):
    """API endpoint to get a single project by ID"""
    try:
        # Search for the project file recursively
        project_dir = Path('./data/projects')
        project_file = None
        
        # Look for the markdown file in any subfolder
        for md_file in project_dir.glob(f'**/{project_id}.md'):
            project_file = md_file
            break
        
        if not project_file or not project_file.exists():
            return jsonify({'error': 'Project not found'}), 404
        
        with open(project_file, 'r', encoding='utf-8') as f:
            post = frontmatter.load(f)
            
            project = {
                'id': post.metadata.get('id', project_id),
                'title': post.metadata.get('title', 'Untitled Project'),
                'description': post.metadata.get('description', ''),
                'technologies': post.metadata.get('technologies', '').split(', ') if isinstance(post.metadata.get('technologies'), str) else post.metadata.get('technologies', []),
                'status': post.metadata.get('status', 'planned'),
                'featured': post.metadata.get('featured', False),
                'githubUrl': post.metadata.get('githubUrl'),
                'liveUrl': post.metadata.get('liveUrl'),
                'demoUrl': post.metadata.get('demoUrl'),
                'startDate': post.metadata.get('startDate'),
                'endDate': post.metadata.get('endDate'),
                'thumbnail': post.metadata.get('thumbnail'),
                'content': post.content
            }
            
        return jsonify(project)
    
    except Exception as e:
        print(f"Error loading project {project_id}: {e}")
        return jsonify({'error': 'Failed to load project'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)