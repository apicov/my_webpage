from flask import Blueprint, request, jsonify
import os
import requests
import threading
from datetime import datetime, timedelta

# Create blueprint
stream_bp = Blueprint('stream', __name__)

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
    """Auto-shutdown camera after 5-minute absolute time limit"""
    global camera_state
    print("Auto-shutting down camera - 5 minute limit reached")
    result = call_camera_api('/camera/stop', 'POST')
    camera_state['running'] = False
    camera_state['last_activity'] = None
    camera_state['auto_shutdown_timer'] = None
    camera_state['started_at'] = None
    print(f"Camera auto-shutdown result: {result}")

def start_camera_timer():
    """Start absolute 5-minute timer (no reset allowed)"""
    global camera_state

    # Cancel any existing timer
    if camera_state['auto_shutdown_timer']:
        camera_state['auto_shutdown_timer'].cancel()

    # Start new absolute timer - exactly 5 minutes from now
    camera_state['started_at'] = datetime.now()
    camera_state['last_activity'] = datetime.now()
    camera_state['auto_shutdown_timer'] = threading.Timer(
        CAMERA_TIMEOUT_MINUTES * 60,
        auto_shutdown_camera
    )
    camera_state['auto_shutdown_timer'].start()
    print(f"Camera started - will auto-shutdown in exactly {CAMERA_TIMEOUT_MINUTES} minutes")

def get_time_remaining():
    """Get time remaining in minutes (for display)"""
    global camera_state
    if not camera_state['started_at']:
        return None

    elapsed = datetime.now() - camera_state['started_at']
    remaining = timedelta(minutes=CAMERA_TIMEOUT_MINUTES) - elapsed
    return max(0, int(remaining.total_seconds() / 60))

@stream_bp.route('/start', methods=['POST'])
@stream_bp.route('/api/stream/start', methods=['POST'])
def start_stream():
    """Start camera and stream - unified endpoint"""
    global camera_state

    try:
        # Check current status first
        status = call_camera_api('/camera/status')

        if status.get('running'):
            # Camera already running - no reset, return remaining time
            time_remaining = get_time_remaining()
            return jsonify({
                'status': 'success',
                'message': 'Stream already active',
                'running': True,
                'janus_url': JANUS_WEBSOCKET_URL,
                'timeout_minutes': CAMERA_TIMEOUT_MINUTES,
                'time_remaining_minutes': time_remaining
            })

        # Start camera
        print("Starting camera stream...")
        result = call_camera_api('/camera/start', 'POST')

        if result.get('status') in ['started', 'already_running']:
            camera_state['running'] = True
            start_camera_timer()

            return jsonify({
                'status': 'success',
                'message': 'Stream started successfully',
                'running': True,
                'janus_url': JANUS_WEBSOCKET_URL,
                'timeout_minutes': CAMERA_TIMEOUT_MINUTES,
                'time_remaining_minutes': CAMERA_TIMEOUT_MINUTES,
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

@stream_bp.route('/stop', methods=['POST'])
@stream_bp.route('/api/stream/stop', methods=['POST'])
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

@stream_bp.route('/status', methods=['GET'])
@stream_bp.route('/api/stream/status', methods=['GET'])
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

        time_remaining = get_time_remaining()

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

