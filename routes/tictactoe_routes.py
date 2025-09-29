from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import uuid
import os

# Import the TicTacToe state machine
from agents.tic_tac_toe_state_machine import TicTacToeStateMachine

# Create blueprint
tictactoe_bp = Blueprint('tictactoe', __name__)

# Simple session tracking: user_id -> {'state_machine': TicTacToeStateMachine, 'last_activity': datetime}
user_sessions = {}

# Game timeout (5 minutes of inactivity)
GAME_TIMEOUT_MINUTES = 5

def cleanup_expired_sessions():
    """Remove expired sessions"""
    current_time = datetime.now()
    expired_users = [
        user_id for user_id, session in user_sessions.items()
        if current_time - session['last_activity'] > timedelta(minutes=GAME_TIMEOUT_MINUTES)
    ]
    for user_id in expired_users:
        del user_sessions[user_id]

def get_state_machine(user_id):
    """Get or create TicTacToe state machine for user"""
    cleanup_expired_sessions()

    if user_id not in user_sessions:
        # Create new state machine for this user
        led_matrix_url = os.getenv('MATRIX_URL', 'http://localhost:8080')
        state_machine = TicTacToeStateMachine(
            thread_id=f"ttt-{user_id}",
            led_matrix_url=led_matrix_url
        )
        user_sessions[user_id] = {
            'state_machine': state_machine,
            'last_activity': datetime.now()
        }
    else:
        # Update activity timestamp
        user_sessions[user_id]['last_activity'] = datetime.now()

    return user_sessions[user_id]['state_machine']

@tictactoe_bp.route('/chat', methods=['POST'])
@tictactoe_bp.route('/api/tictactoe/chat', methods=['POST'])
def tictactoe_chat():
    """TicTacToe chat endpoint"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        user_id = data.get('user_id', str(uuid.uuid4()))
        user_state = data.get('state', '')

        print(f"TicTacToe: user_id={user_id}, message={user_message}, state={user_state}")

        # Handle reset command
        if user_state == 'reset':
            if user_id in user_sessions:
                user_sessions[user_id]['state_machine'].reset_state_machine()
                del user_sessions[user_id]
            return jsonify({'message': '🔄 Game reset! Send a message to start new game.', 'state': 'playing'})

        # Get state machine and process message
        state_machine = get_state_machine(user_id)
        response_message = state_machine.step(user_message)

        return jsonify({'message': response_message, 'state': state_machine.state.current_state})

    except Exception as e:
        print(f"TicTacToe error: {e}")
        return jsonify({'message': '🚫 Error occurred. Try again.', 'state': 'error'}), 500

@tictactoe_bp.route('/status', methods=['GET'])
@tictactoe_bp.route('/api/tictactoe/status', methods=['GET'])
def game_status():
    """Get current game status"""
    cleanup_expired_sessions()
    return jsonify({
        'active_sessions': len(user_sessions)
    })

@tictactoe_bp.route('/reset', methods=['POST'])
@tictactoe_bp.route('/api/tictactoe/reset', methods=['POST'])
def reset_game():
    """Reset the current game"""
    try:
        data = request.get_json(silent=True)
        user_id = data.get('user_id') if data else None

        if user_id and user_id in user_sessions:
            user_sessions[user_id]['state_machine'].reset_state_machine()
            del user_sessions[user_id]

        return jsonify({
            'status': 'success',
            'message': '🔄 Game reset successfully. Send a message to start a new game.'
        })

    except Exception as e:
        print(f"TicTacToe reset error: {e}")
        return jsonify({
            'status': 'error',
            'message': '🚫 Reset failed. Please try again.'
        }), 500