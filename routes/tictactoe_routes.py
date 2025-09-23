from flask import Blueprint, request, jsonify
import json
import time
from datetime import datetime, timedelta
import uuid

# Create blueprint
tictactoe_bp = Blueprint('tictactoe', __name__)

# Global game session state - only one game at a time
game_session = {
    'active': False,
    'user_id': None,
    'board': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # 0=empty, 1=human, 2=AI
    'current_player': 'human',
    'game_state': 'waiting',  # waiting, playing, ai_thinking, finished
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
        'board': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        'current_player': 'human',
        'game_state': 'waiting',
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

def get_valid_moves():
    """Get list of valid move positions (1-9)"""
    moves = []
    for i in range(3):
        for j in range(3):
            if game_session['board'][i][j] == 0:
                position = i * 3 + j + 1  # Convert to 1-9
                moves.append(position)
    return moves

def position_to_coords(position):
    """Convert position 1-9 to [row, col] coordinates"""
    if not (1 <= position <= 9):
        return None
    position -= 1  # Convert to 0-8
    row = position // 3
    col = position % 3
    return [row, col]

def coords_to_position(row, col):
    """Convert [row, col] coordinates to position 1-9"""
    return row * 3 + col + 1

def check_winner():
    """Check if there's a winner or draw"""
    board = game_session['board']

    # Check rows
    for row in board:
        if row[0] == row[1] == row[2] != 0:
            return 'human' if row[0] == 1 else 'ai'

    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != 0:
            return 'human' if board[0][col] == 1 else 'ai'

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != 0:
        return 'human' if board[0][0] == 1 else 'ai'

    if board[0][2] == board[1][1] == board[2][0] != 0:
        return 'human' if board[0][2] == 1 else 'ai'

    # Check for draw
    if not get_valid_moves():
        return 'draw'

    return None

def create_response(message='', state='playing'):
    """Create standardized response"""
    return {
        'message': message,
        'state': state
    }

# TODO: Initialize LangGraph agent here
# from agents.tictactoe_agent import TicTacToeAgent
# tictactoe_agent = TicTacToeAgent()

def generate_ai_response(user_message, game_context):
    """Generate AI response using LangGraph agent"""
    # TODO: Replace with actual LangGraph implementation

    # Check if this is the start of a new game
    if not game_session['active']:
        return "🎮 Welcome to TicTacToe! I'm your AI opponent. You are X (1), I am O (2). Choose positions 1-9:\n\n1|2|3\n4|5|6\n7|8|9\n\nJust tell me the number where you want to place your X. You go first!"

    # Game is active - respond based on context
    if game_session['current_player'] == 'ai':
        return "Let me think about my move... 🤔"
    else:
        return "Your turn! Tell me which position (1-9) you want to place your X."

def make_ai_move():
    """AI makes its move (simple strategy for now)"""
    # TODO: Replace with LangGraph AI decision
    # For now, simple strategy: take center, then corners, then edges

    valid_moves = get_valid_moves()
    if not valid_moves:
        return None

    # Simple AI strategy
    # 1. Try to win
    # 2. Block human from winning
    # 3. Take center
    # 4. Take corners
    # 5. Take edges

    board = game_session['board']

    # Check if AI can win
    for row, col in valid_moves:
        board[row][col] = 2  # Try AI move
        if check_winner() == 'ai':
            return [row, col]  # Winning move found
        board[row][col] = 0  # Undo

    # Check if need to block human
    for row, col in valid_moves:
        board[row][col] = 1  # Try human move
        if check_winner() == 'human':
            board[row][col] = 2  # Block it
            return [row, col]
        board[row][col] = 0  # Undo

    # Take center if available
    if [1, 1] in valid_moves:
        return [1, 1]

    # Take corners
    corners = [[0, 0], [0, 2], [2, 0], [2, 2]]
    for corner in corners:
        if corner in valid_moves:
            return corner

    # Take any edge
    return valid_moves[0]

@tictactoe_bp.route('/chat', methods=['POST'])
@tictactoe_bp.route('/api/tictactoe/chat', methods=['POST'])
def tictactoe_chat():
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
                'game_state': 'playing',
                'started_at': datetime.now(),
                'last_activity': datetime.now()
            })
            ai_response = "🎮 TicTacToe started! You are X, I am O. Choose positions 1-9:\n\n1|2|3\n4|5|6\n7|8|9\n\nYour turn!"
            return jsonify(create_response(ai_response, 'playing'))

        # Update activity
        game_session['last_activity'] = datetime.now()

        # Try to parse position from message (1-9)
        position = None
        try:
            import re
            numbers = re.findall(r'\b[1-9]\b', user_message)
            if numbers:
                position = int(numbers[0])
        except:
            pass

        # Handle move if position detected and it's human's turn
        if position and game_session['current_player'] == 'human':
            coords = position_to_coords(position)
            if coords:
                row, col = coords
                if game_session['board'][row][col] == 0:
                    # Make human move
                    game_session['board'][row][col] = 1

                    # Check for winner after human move
                    winner = check_winner()
                    if winner:
                        game_session['game_state'] = 'finished'
                        if winner == 'human':
                            ai_response = "🎉 You won! 🏆"
                        elif winner == 'draw':
                            ai_response = "🤝 Draw! Good game!"
                        else:
                            ai_response = "🤖 I won! Good game!"
                        return jsonify(create_response(ai_response, 'finished'))

                    # Make AI move
                    game_session['current_player'] = 'ai'
                    ai_position = make_ai_move()
                    if ai_position:
                        ai_coords = position_to_coords(ai_position)
                        if ai_coords:
                            ai_row, ai_col = ai_coords
                            game_session['board'][ai_row][ai_col] = 2

                            # Check for winner after AI move
                            winner = check_winner()
                            if winner:
                                game_session['game_state'] = 'finished'
                                if winner == 'ai':
                                    ai_response = "🤖 I won! Good game!"
                                elif winner == 'draw':
                                    ai_response = "🤝 Draw! Good game!"
                                return jsonify(create_response(ai_response, 'finished'))

                            # Continue game
                            game_session['current_player'] = 'human'
                            ai_response = f"Good move! I played {ai_position}. Your turn!"

                    return jsonify(create_response(ai_response, 'playing'))

                else:
                    ai_response = f"❌ Position {position} taken. Try another!"
            else:
                ai_response = "❌ Choose 1-9"
        else:
            # General chat response
            ai_response = generate_ai_response(user_message, game_session)

        return jsonify(create_response(ai_response, 'playing'))

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
        'game_state': game_session['game_state'],
        'board': game_session['board'],
        'current_player': game_session['current_player'],
        'winner': check_winner(),
        'valid_moves': get_valid_moves(),
        'active_user': game_session['user_id'] if game_session['active'] else None,
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