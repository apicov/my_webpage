from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import uuid

# Create blueprint
tictactoe_bp = Blueprint('tictactoe', __name__)

# Simple session state - only one game at a time
game_session = {
    'active': False,
    'user_id': None,
    'langraph_state': {},
    'started_at': None,
    'last_activity': None
}

# Game timeout (15 minutes of inactivity)
GAME_TIMEOUT_MINUTES = 15

def reset_game_session():
    """Reset the game session to initial state"""
    global game_session
    game_session.update({
        'active': False,
        'user_id': None,
        'langraph_state': {},
        'started_at': None,
        'last_activity': None
    })

def is_session_expired():
    """Check if current session has expired"""
    if not game_session['last_activity']:
        return False

    elapsed = datetime.now() - game_session['last_activity']
    return elapsed > timedelta(minutes=GAME_TIMEOUT_MINUTES)

def create_response(message='', state='playing'):
    """Create standardized response"""
    return {
        'message': message,
        'state': state
    }

# TODO: Initialize LangGraph agent here
# from agents.tictactoe_agent import TicTacToeAgent
# tictactoe_agent = TicTacToeAgent()

async def call_game_api(user_message, user_id, session_state):
    """Call external TicTacToe game API or LangGraph agent"""
    # TODO: Replace with actual API calls to your game service
    # This is where you'll integrate with your LangGraph agent

    # Placeholder response for now
    return {
        'message': f'Received: {user_message}. Game API integration pending...',
        'state': 'playing'
    }


@tictactoe_bp.route('/chat', methods=['POST'])
@tictactoe_bp.route('/api/tictactoe/chat', methods=['POST'])
async def tictactoe_chat():
    """Simple TicTacToe chat endpoint with state field"""
    global game_session

    try:
        data = request.get_json()
        user_message = data.get('message', '')
        user_id = data.get('user_id', str(uuid.uuid4()))
        user_state = data.get('state', '')

        print(f"TicTacToe: user_id={user_id}, message={user_message}, state={user_state}")

        # Check for expired session
        if game_session['active'] and is_session_expired():
            reset_game_session()

        # Handle reset command
        if user_state == 'reset':
            reset_game_session()
            return jsonify(create_response('🔄 Game reset! Send a message to start new game.', 'playing'))

        # Check if game is busy with different user
        if game_session['active'] and game_session['user_id'] != user_id:
            return jsonify(create_response('🚫 Game busy. Another player is active.', 'busy'))

        # Start new game if not active
        if not game_session['active']:
            game_session.update({
                'active': True,
                'user_id': user_id,
                'started_at': datetime.now(),
                'last_activity': datetime.now()
            })

        # Update activity
        game_session['last_activity'] = datetime.now()

        # Call the game API/agent
        result = await call_game_api(user_message, user_id, game_session['langraph_state'])

        # Update langraph state if provided
        if 'langraph_state' in result:
            game_session['langraph_state'] = result['langraph_state']

        return jsonify(create_response(result['message'], result['state']))

    except Exception as e:
        print(f"TicTacToe error: {e}")
        return jsonify(create_response('🚫 Error occurred. Try again.', 'error')), 500

@tictactoe_bp.route('/status', methods=['GET'])
@tictactoe_bp.route('/api/tictactoe/status', methods=['GET'])
def game_status():
    """Get current game status"""
    global game_session

    # Check for expired session
    if game_session['active'] and is_session_expired():
        reset_game_session()

    return jsonify({
        'active': game_session['active'],
        'user_id': game_session['user_id'] if game_session['active'] else None,
        'started_at': game_session['started_at'].isoformat() if game_session['started_at'] else None
    })

@tictactoe_bp.route('/reset', methods=['POST'])
@tictactoe_bp.route('/api/tictactoe/reset', methods=['POST'])
def reset_game():
    """Reset the current game"""
    global game_session

    try:
        data = request.get_json()
        user_id = data.get('user_id') if data else None

        # Only allow reset if user is the active player or game is inactive
        if game_session['active'] and user_id and game_session['user_id'] != user_id:
            return jsonify({
                'status': 'error',
                'message': '🚫 Cannot reset. Another player is active.'
            }), 400

        reset_game_session()
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