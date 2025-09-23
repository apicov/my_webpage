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
import threading
import requests
from datetime import datetime, timedelta


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

# Camera Stream Management (imports moved to top)

# Camera state tracking
camera_state = {
    'running': False,
    'last_activity': None,
    'auto_shutdown_timer': None
}

# Configuration
CAMERA_API_BASE = os.getenv('CAMERA_API_URL', 'https://your-camera-api.ngrok.io')  # Camera control API
JANUS_WEBSOCKET_URL = os.getenv('JANUS_WEBSOCKET_URL', 'wss://your-janus.ngrok.io')  # Janus WebRTC server
CAMERA_TIMEOUT_MINUTES = int(os.getenv('CAMERA_TIMEOUT_MINUTES', '5'))  # Auto-shutdown after 5 minutes

def call_camera_api(endpoint, method='GET'):
    """Helper to call Raspberry Pi camera API"""
    try:
        url = f"{CAMERA_API_BASE}{endpoint}"
        if method == 'POST':
            response = requests.post(url, timeout=10)
        else:
            response = requests.get(url, timeout=10)
        return response.json()
    except Exception as e:
        print(f"Camera API error: {e}")
        return {"error": str(e)}

def auto_shutdown_camera():
    """Auto-shutdown camera after timeout"""
    global camera_state
    print("Auto-shutting down camera due to inactivity")
    result = call_camera_api('/camera/stop', 'POST')
    camera_state['running'] = False
    camera_state['last_activity'] = None
    camera_state['auto_shutdown_timer'] = None
    print(f"Camera auto-shutdown result: {result}")

def reset_camera_timer():
    """Reset/restart the camera auto-shutdown timer"""
    global camera_state

    # Cancel existing timer
    if camera_state['auto_shutdown_timer']:
        camera_state['auto_shutdown_timer'].cancel()

    # Start new timer
    camera_state['last_activity'] = datetime.now()
    camera_state['auto_shutdown_timer'] = threading.Timer(
        CAMERA_TIMEOUT_MINUTES * 60,
        auto_shutdown_camera
    )
    camera_state['auto_shutdown_timer'].start()
    print(f"Camera timer reset - will auto-shutdown in {CAMERA_TIMEOUT_MINUTES} minutes")

@app.route('/stream/start', methods=['POST'])
@app.route('/api/stream/start', methods=['POST'])
def start_stream():
    """Start camera and stream - unified endpoint"""
    global camera_state

    try:
        # Check current status first
        status = call_camera_api('/camera/status')

        if status.get('running'):
            # Camera already running - just reset timer
            reset_camera_timer()
            return jsonify({
                'status': 'success',
                'message': 'Stream already active, timer reset',
                'running': True,
                'janus_url': JANUS_WEBSOCKET_URL,
                'timeout_minutes': CAMERA_TIMEOUT_MINUTES
            })

        # Start camera
        print("Starting camera stream...")
        result = call_camera_api('/camera/start', 'POST')

        if result.get('status') in ['started', 'already_running']:
            camera_state['running'] = True
            reset_camera_timer()

            return jsonify({
                'status': 'success',
                'message': 'Stream started successfully',
                'running': True,
                'janus_url': JANUS_WEBSOCKET_URL,
                'timeout_minutes': CAMERA_TIMEOUT_MINUTES,
                'camera_result': result
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to start camera',
                'error': result.get('error', 'Unknown error')
            }), 500

    except Exception as e:
        print(f"Error starting stream: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Stream start failed: {str(e)}'
        }), 500

@app.route('/stream/stop', methods=['POST'])
@app.route('/api/stream/stop', methods=['POST'])
def stop_stream():
    """Stop camera and stream - unified endpoint"""
    global camera_state

    try:
        # Cancel auto-shutdown timer
        if camera_state['auto_shutdown_timer']:
            camera_state['auto_shutdown_timer'].cancel()
            camera_state['auto_shutdown_timer'] = None

        # Stop camera
        result = call_camera_api('/camera/stop', 'POST')

        camera_state['running'] = False
        camera_state['last_activity'] = None

        return jsonify({
            'status': 'success',
            'message': 'Stream stopped successfully',
            'running': False,
            'camera_result': result
        })

    except Exception as e:
        print(f"Error stopping stream: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Stream stop failed: {str(e)}'
        }), 500

@app.route('/stream/status', methods=['GET'])
@app.route('/api/stream/status', methods=['GET'])
def stream_status():
    """Get current stream status"""
    global camera_state

    try:
        # Get actual camera status
        pi_status = call_camera_api('/camera/status')

        # Sync our state with actual camera state
        actual_running = pi_status.get('running', False)
        if actual_running != camera_state['running']:
            camera_state['running'] = actual_running
            if not actual_running and camera_state['auto_shutdown_timer']:
                camera_state['auto_shutdown_timer'].cancel()
                camera_state['auto_shutdown_timer'] = None

        time_remaining = None
        if camera_state['running'] and camera_state['last_activity']:
            elapsed = datetime.now() - camera_state['last_activity']
            remaining = timedelta(minutes=CAMERA_TIMEOUT_MINUTES) - elapsed
            time_remaining = max(0, int(remaining.total_seconds() / 60))

        return jsonify({
            'status': 'success',
            'running': camera_state['running'],
            'last_activity': camera_state['last_activity'].isoformat() if camera_state['last_activity'] else None,
            'timeout_minutes': CAMERA_TIMEOUT_MINUTES,
            'time_remaining_minutes': time_remaining,
            'janus_url': JANUS_WEBSOCKET_URL if camera_state['running'] else None,
            'pi_status': pi_status
        })

    except Exception as e:
        print(f"Error getting stream status: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Status check failed: {str(e)}'
        }), 500

@app.route('/stream/ping', methods=['POST'])
@app.route('/api/stream/ping', methods=['POST'])
def stream_ping():
    """Ping endpoint to reset camera timer (extend session)"""
    global camera_state

    if camera_state['running']:
        reset_camera_timer()
        return jsonify({
            'status': 'success',
            'message': 'Timer reset - session extended',
            'time_remaining_minutes': CAMERA_TIMEOUT_MINUTES
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Stream not running'
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)