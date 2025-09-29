# Modular TicTacToe State Machine
import os
import random
from typing import Optional, Literal, List
from abc import ABC, abstractmethod
from enum import Enum
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .tic_tac_toe_tools import TicTacToeTools

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass  # dotenv not available

# =============================================================================
# MODELS AND DATA CLASSES
# =============================================================================

class CameraStatus(str, Enum):
    WORKING = "working"
    NOT_WORKING = "not_working"
    NOT_MENTIONED = "not_mentioned"

class CameraOnAcknowledgement(BaseModel):
    camera_status: CameraStatus = Field(CameraStatus.NOT_MENTIONED, description="Set to WORKING if user confirms camera is visible/working, NOT_WORKING if user reports camera issues, NOT_MENTIONED if user doesn't mention camera at all")
    player_name: Optional[str] = Field(None, description="ONLY extract if user explicitly provides their name. Leave None if not provided.")
    player_symbol: Optional[Literal["X", "O"]] = Field(None, description="ONLY extract if user explicitly chooses X or O. Leave None if not explicitly chosen.")
    reply: str = Field(..., description="A friendly response acknowledging the input and guiding the conversation forward")

class PlayerReadyResponse(BaseModel):
    ready_to_start: Optional[bool] = Field(None, description="ONLY set to True if user explicitly confirms they are ready to start the game. Leave None if not explicitly stated.")
    reply: str = Field(..., description="A friendly response acknowledging the input and guiding the conversation forward")

class MoveResponse(BaseModel):
    move: Optional[int] = Field(None, description="ONLY extract the cell number (1-9) if user explicitly makes a move or select your move if it is your turn. Leave None if not explicitly stated.")
    reply: str = Field(..., description="A friendly response acknowledging the input and guiding the conversation forward")

class UserMoveExtraction(BaseModel):
    user_move: Optional[int] = Field(None, description="Extract the cell number (1-9) if user explicitly specifies a move. Leave None if no move specified.")
    has_move: bool = Field(False, description="True if user is making a move, False if just chatting")

class AIMoveGeneration(BaseModel):
    ai_move: int = Field(..., description="The AI's chosen move position (1-9)")
    response_message: str = Field(..., description="Friendly response about the AI's move and current game state")

class GameState:
    """Centralized game state management"""
    def __init__(self):
        self.current_state = "welcome"
        self.messages = []
        self.player_name = ""
        self.ai_name = "Bot"
        self.player_symbol = None
        self.ai_symbol = None
        self.start_turn = None
        self.add_sys_prompt_flag = True

    def set_player_info(self, name: str, symbol: str):
        self.player_name = name
        self.player_symbol = symbol
        self.ai_symbol = "O" if symbol == "X" else "X"

    def transition_to(self, new_state: str):
        self.current_state = new_state
        self.add_sys_prompt_flag = True

class GameConfig:
    """Configuration constants"""
    def __init__(self, thread_id=None, led_matrix_url=None):
        self.thread_id = thread_id or os.getenv("THREAD_ID", "ttt-1")
        self.led_matrix_url = led_matrix_url or os.getenv("MATRIX_URL")
        self.max_errors = int(os.getenv("MAX_ERRORS", "3"))
        self.llm_model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "1.0"))

        if not self.led_matrix_url:
            raise ValueError("MATRIX_URL environment variable must be set")

# =============================================================================
# SERVICES
# =============================================================================

class LLMService:
    """Centralized LLM interaction service"""
    def __init__(self, config: GameConfig):
        self.groq_llm = ChatGroq(
            groq_api_key=os.getenv('GROQ_API_KEY'),
            model_name=config.llm_model,
            temperature=config.llm_temperature
        )

    def invoke_with_retry(self, messages, response_model, max_retries=2):
        """Retry wrapper for structured output"""
        for attempt in range(max_retries + 1):
            try:
                structured_llm = self.groq_llm.with_structured_output(response_model)
                return structured_llm.invoke(messages)
            except Exception as e:
                print(f"Structured output attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    raise e

    def simple_invoke(self, messages):
        """Simple LLM invocation without structured output"""
        return self.groq_llm.invoke(messages)

class ErrorHandler:
    """Centralized error handling"""
    def __init__(self, max_errors=3):
        self.error_count = 0
        self.max_errors = max_errors

    def handle_error(self, error, context="", fallback_message="I'm having trouble understanding. Let me try again."):
        """Handle errors with retry logic"""
        self.error_count += 1
        print(f"Error in {context}: {error}")

        if self.error_count >= self.max_errors:
            return "I've encountered several issues. Let's start fresh! Please click the 'Connect Camera' button to begin."
        else:
            return fallback_message

    def reset_error_count(self):
        """Reset error count on successful operation"""
        self.error_count = 0

class ToolsManager:
    """Manages tool interactions"""
    def __init__(self, config: GameConfig):
        self.tools = TicTacToeTools(config.led_matrix_url, agent_name="Agent")
        # We no longer need to bind/unbind tools since we're calling API methods directly
        # No ReAct agent needed - using structured approach

    def start_video_stream(self, is_retry=False):
        """Start video stream via local Flask stream endpoint"""
        action = "retry" if is_retry else "start"
        try:
            import requests
            # Call the local Flask stream endpoint (same as frontend does)
            url = "http://localhost:5000/stream/start"
            response = requests.post(url,
                                   headers={'Content-Type': 'application/json'},
                                   timeout=10)
            print(f"Video stream {action} response: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Stream API result: {result}")
                return result.get('status') == 'success'
            return False
        except Exception as e:
            print(f"Failed to {action} video stream: {e}")
            return False

    def setup_game(self, player_name: str, player_symbol: str, ai_name: str, start_turn: str):
        """Setup game on LED matrix"""
        return self.tools.setup_game(
            user_name=player_name,
            user_symbol=player_symbol,
            bot_name=ai_name,
            starting_player=start_turn
        )

    def full_reset_and_welcome(self):
        """Reset game and show welcome screen"""
        try:
            self.tools.full_reset()
            self.tools.welcome_screen()
        except Exception as e:
            print(f"Error resetting LED matrix: {e}")

class PromptManager:
    """Manages system prompts for different states"""

    @staticmethod
    def get_welcome_prompt():
        return """You are a friendly Tic-Tac-Toe game host. This is the very start of the game. The game will be displayed on a led matrix, which will be streamed to the user, through a web camera. You will play the game with the user.

        Your tasks:
        1. Welcome the user enthusiastically and tell them you've automatically turned on the camera stream with the LED matrix
        2. Ask them to wait a few seconds for the video to load, then confirm if they can see the LED matrix camera feed
        3. If user confirms they can see the camera/LED matrix, proceed to ask for their name and whether they'd like to be X or O
        4. If user says they can't see it, offer to try again or troubleshoot

        CRITICAL STRUCTURED OUTPUT RULES:
        - Set camera_status to WORKING if user explicitly confirms camera is working/visible
        - Set camera_status to NOT_WORKING if user explicitly says camera isn't working or asks to try again
        - Set camera_status to NOT_MENTIONED if user doesn't mention camera at all
        - ONLY fill player_name field if user explicitly provides their name
        - ONLY fill player_symbol field if user explicitly chooses X or O
        - If user hasn't provided name/symbol, leave those fields as null/None
        - Always provide a friendly reply regardless

        PERSISTENCE RULES:
        - If user avoids giving their name (jokes, says "secret", etc.), politely insist: "I need your actual name to start the game!"
        - If user doesn't choose X or O clearly, ask again: "Please choose X or O to play as!"
        - If camera confirmation is missing, remind them: "Please wait a few seconds for the video to load. Can you see the LED matrix camera feed?"
        - If user says camera isn't working, offer: "Let me try connecting again. Please wait a moment..."
        - Do NOT accept evasive answers or move forward without ALL required information
        - Keep the conversation friendly but persistent until you get real answers

        Do NOT make assumptions or fill fields based on implications. Wait for explicit user input."""

    @staticmethod
    def get_player_ready_prompt():
        return """You are a friendly Tic-Tac-Toe game host. The user has provided their name and chosen their symbol. You have asked if they are ready to start the game. Now, you need to confirm if the user is ready to start playing."""

    @staticmethod
    def get_playing_prompt(player_name: str, ai_name: str, player_symbol: str, ai_symbol: str):
        return f"""You are a strategic Tic-Tac-Toe player named {ai_name}.

        AVAILABLE TOOLS:
        - ttt_make_move: Make a move by specifying position 1-9
        - ttt_win_animation: Show win animation (winner_name, winner_symbol) - MUST call when game ends with a winner
        - ttt_draw_animation: Show draw/tie animation - MUST call when game ends in a draw

        CRITICAL TURN-BASED BEHAVIOR:
        You will receive clear instructions about whose turn it is. Follow them exactly:

        IF TOLD "IT IS YOUR TURN":
        - Make YOUR move immediately using ttt_make_move with position 1-9
        - Provide a friendly response about your move
        - If you won or game ended, call appropriate animation (win_animation or draw_animation)

        IF TOLD "IT IS {player_name}'S TURN":
        - Ask {player_name} to choose their move (position 1-9)
        - When they respond with a number, call ttt_make_move with their chosen position
        - Provide a friendly response about their move result
        - If they won or game ended, call appropriate animation (win_animation or draw_animation)

        GAME END DETECTION:
        - The system will detect when the game ends automatically
        - When you make a winning move or final move, call the appropriate animation
        - win_animation: when someone wins
        - draw_animation: when game ends in a tie

        INTERACTION RULES:
        - YOU are responsible for ALL tool calls (including human moves)
        - Keep responses friendly and brief
        - Don't show ASCII boards - user sees LED matrix
        - When game ends: congratulate and STOP (don't ask about playing again)

        STRATEGY (when it's your turn):
        - Win if possible (complete your line)
        - Block opponent's winning move
        - Take center if available
        - Take corners over edges

        Current players: {player_name} ({player_symbol}) vs {ai_name} ({ai_symbol})

        REMEMBER:
        - Follow the turn instructions you receive exactly
        - YOU make ALL tool calls to the LED matrix
        - When game ends: show animation and congratulate, don't ask for another game"""

# =============================================================================
# STATE HANDLERS
# =============================================================================

class StateHandler(ABC):
    """Abstract base class for state handlers"""

    @abstractmethod
    def handle(self, state_machine, message: str) -> str:
        pass

class WelcomeStateHandler(StateHandler):
    """Handles welcome state logic"""

    def handle(self, state_machine, message: str) -> str:
        print("Welcome state")

        if state_machine.state.add_sys_prompt_flag:
            state_machine.tools_manager.full_reset_and_welcome()
            state_machine.state.add_sys_prompt_flag = False

            # Turn on video stream automatically
            print("Starting video stream...")
            state_machine.tools_manager.start_video_stream()

        try:
            greeting_message = state_machine.llm_service.invoke_with_retry(
                state_machine.state.messages,
                CameraOnAcknowledgement
            )

            # Reset error count on success
            state_machine.error_handler.reset_error_count()

            # Handle camera status feedback
            if greeting_message.camera_status == CameraStatus.NOT_WORKING:
                print("User reported camera not working - retrying video stream...")
                state_machine.tools_manager.start_video_stream(is_retry=True)

            # Check if user has provided all required info
            if (greeting_message.player_name and \
                greeting_message.player_symbol and \
                greeting_message.camera_status == CameraStatus.WORKING):

                # Set player info
                state_machine.state.set_player_info(
                    greeting_message.player_name,
                    greeting_message.player_symbol
                )

                # Update system prompt for next state
                system_prompt = f"""You are a friendly Tic-Tac-Toe game host. The user has confirmed the camera is working, and has provided their name as {state_machine.state.player_name}, and chosen to be {state_machine.state.player_symbol}. You are {state_machine.state.ai_symbol}. Now ask the user if they are ready to start the game."""

                state_machine.state.messages[0] = SystemMessage(content=system_prompt)
                response = state_machine.llm_service.simple_invoke(state_machine.state.messages)
                state_machine.state.messages.append(AIMessage(content=response.content))

                # Randomly select who starts
                state_machine.state.start_turn = random.choice([
                    state_machine.state.player_name,
                    state_machine.state.ai_name
                ])

                # Setup the game on LED matrix
                state_machine.tools_manager.setup_game(
                    state_machine.state.player_name,
                    state_machine.state.player_symbol,
                    state_machine.state.ai_name,
                    state_machine.state.start_turn
                )

                # Show player vs screen animation
                state_machine.tools_manager.tools.player_vs_screen()

                # Transition to next state
                state_machine.state.transition_to('player_ready')
                return response.content

            state_machine.state.messages.append(AIMessage(content=greeting_message.reply))
            return greeting_message.reply

        except Exception as e:
            return state_machine.error_handler.handle_error(
                e, "welcome state",
                "I'm having trouble processing your response. Could you please confirm if you can see the camera and provide your name and symbol choice (X or O)?"
            )

class PlayerReadyStateHandler(StateHandler):
    """Handles player ready state logic"""

    def handle(self, state_machine, message: str) -> str:
        print("Player ready state")

        if state_machine.state.add_sys_prompt_flag:
            system_prompt = PromptManager.get_player_ready_prompt()
            state_machine.state.messages[0] = SystemMessage(content=system_prompt)
            state_machine.state.add_sys_prompt_flag = False

        try:
            ready_message = state_machine.llm_service.invoke_with_retry(
                state_machine.state.messages,
                PlayerReadyResponse
            )

            # Reset error count on success
            state_machine.error_handler.reset_error_count()

            if ready_message.ready_to_start:
                state_machine.state.transition_to('playing')

                # Start the game
                state_machine.tools_manager.tools.start_game()

                if state_machine.state.start_turn == state_machine.state.player_name:
                    # Player goes first
                    return state_machine._handle_player_first_move()
                else:
                    # AI goes first
                    return state_machine._handle_ai_first_move()
            else:
                return ready_message.reply

        except Exception as e:
            return state_machine.error_handler.handle_error(
                e, "player_ready state",
                "I'm having trouble understanding. Are you ready to start the game? Please say 'yes' or 'ready' to begin."
            )

class EndgameStateHandler(StateHandler):
    """Handles endgame state - minimal handler for game completion"""

    def handle(self, state_machine, message: str) -> str:
        # Reset to welcome state for new game
        state_machine.state.transition_to('welcome')
        # Reset game state
        state_machine.state = GameState()
        # Reset with welcome prompt
        welcome_prompt = PromptManager.get_welcome_prompt()
        state_machine.state.messages = [SystemMessage(content=welcome_prompt)]
        state_machine.state.messages.append(HumanMessage(content=message))

        # Handle the message as a new welcome
        return state_machine.handlers["welcome"].handle(state_machine, message)

class PlayingStateHandler(StateHandler):
    """Handles gameplay state logic with structured approach"""

    def handle(self, state_machine, message: str) -> str:
        if state_machine.state.add_sys_prompt_flag:
            state_machine.state.add_sys_prompt_flag = False

        try:
            # Step 1: Get current game status to understand board state
            game_status = state_machine.tools_manager.tools.get_status()
            current_turn = game_status.get('current_player', '')
            board = game_status.get('board', [])
            available_positions = [i+1 for i, cell in enumerate(board) if cell == ' '] if board else []

            print(f"Current turn: {current_turn}")
            print(f"Available positions: {available_positions}")

            # Step 2: Extract user move (if any) with structured output
            user_move_result = self._extract_user_move(state_machine, message, available_positions)

            # Step 3: Handle user move if provided
            if user_move_result and user_move_result.has_move:
                if user_move_result.user_move in available_positions:
                    # Execute user move programmatically
                    move_result = state_machine.tools_manager.tools.make_move(user_move_result.user_move)
                    print(f"User move {user_move_result.user_move} executed: {move_result}")

                    # Check if game ended after user move
                    game_status = state_machine.tools_manager.tools.get_status()
                    if self._check_game_ended(state_machine, game_status):
                        return self._handle_game_end(state_machine, game_status)
                else:
                    # Invalid move - cell not available
                    return f"Sorry, position {user_move_result.user_move} is not available. Please choose from: {', '.join(map(str, available_positions))}"

            # Step 4: Check if it's AI's turn now
            game_status = state_machine.tools_manager.tools.get_status()
            current_turn = game_status.get('current_player', '')

            if current_turn == state_machine.state.ai_name:
                # Step 5: Generate AI move with structured output
                board = game_status.get('board', [])
                available_positions = [i+1 for i, cell in enumerate(board) if cell == ' '] if board else []

                ai_move_result = self._generate_ai_move(state_machine, available_positions)

                # Step 6: Execute AI move programmatically
                if ai_move_result.ai_move in available_positions:
                    move_result = state_machine.tools_manager.tools.make_move(ai_move_result.ai_move)
                    print(f"AI move {ai_move_result.ai_move} executed: {move_result}")

                    # Step 7: Check if game ended after AI move
                    game_status = state_machine.tools_manager.tools.get_status()
                    if self._check_game_ended(state_machine, game_status):
                        return self._handle_game_end(state_machine, game_status)

                    return ai_move_result.response_message
                else:
                    return "I had trouble making my move. Let me try again."
            else:
                # It's player's turn - ask for their move
                return f"It's your turn! Please choose a position from: {', '.join(map(str, available_positions))}"

        except Exception as e:
            return state_machine.error_handler.handle_error(
                e, "playing state",
                "I'm having trouble processing the game. Please try again."
            )

    def _extract_user_move(self, state_machine, message: str, available_positions: list) -> Optional[UserMoveExtraction]:
        """Extract user move from message using structured output"""
        try:
            extract_prompt = f"""Extract if the user is making a move in this message: "{message}"

Available positions: {available_positions}

Only extract a move if the user explicitly mentions a number 1-9 as their move choice."""

            extraction_messages = [
                SystemMessage(content=extract_prompt),
                HumanMessage(content=message)
            ]

            return state_machine.llm_service.invoke_with_retry(
                extraction_messages,
                UserMoveExtraction
            )
        except Exception as e:
            print(f"Error extracting user move: {e}")
            return None

    def _generate_ai_move(self, state_machine, available_positions: list) -> AIMoveGeneration:
        """Generate AI move using structured output"""
        strategy_prompt = f"""You are a strategic Tic-Tac-Toe AI player named {state_machine.state.ai_name}.

Current available positions: {available_positions}
You are playing as {state_machine.state.ai_symbol} against {state_machine.state.player_name} ({state_machine.state.player_symbol}).

Strategy priority:
1. Win if possible (complete your line)
2. Block opponent's winning move
3. Take center (5) if available
4. Take corners (1,3,7,9) over edges (2,4,6,8)

Choose your move from the available positions and provide a friendly response."""

        ai_messages = [
            SystemMessage(content=strategy_prompt),
            HumanMessage(content="Make your move!")
        ]

        return state_machine.llm_service.invoke_with_retry(
            ai_messages,
            AIMoveGeneration
        )

    def _check_game_ended(self, state_machine, game_status: dict) -> bool:
        """Check if game has ended"""
        status = game_status.get('game_status', '')
        return status in ['ended_draw', 'ended_winner', 'ended']

    def _handle_game_end(self, state_machine, game_status: dict) -> str:
        """Handle game end with appropriate animations and LLM-generated response"""
        status = game_status.get('game_status', '')
        winner = game_status.get('winner', '')

        if status == 'ended_draw':
            # Show draw animation
            state_machine.tools_manager.tools.draw_animation()
            scenario = "draw"
        elif winner:
            # Show win animation
            winner_symbol = state_machine.state.player_symbol if winner == state_machine.state.player_name else state_machine.state.ai_symbol
            state_machine.tools_manager.tools.win_animation(winner, winner_symbol)
            scenario = "player_won" if winner == state_machine.state.player_name else "ai_won"
        else:
            scenario = "ended"

        # Generate natural end game response using LLM
        try:
            end_prompt = f"""Generate a natural, friendly response for the end of a Tic-Tac-Toe game.

Game context:
- Player name: {state_machine.state.player_name}
- AI name: {state_machine.state.ai_name}
- Player symbol: {state_machine.state.player_symbol}
- AI symbol: {state_machine.state.ai_symbol}
- Game outcome: {scenario}

Generate a personalized, conversational response that:
- Acknowledges the game outcome appropriately
- Maintains a friendly, engaging tone
- Feels natural and not robotic
- Congratulates the winner if there is one
- Keeps it brief (1-2 sentences)

Do not ask about playing again - just focus on the current game conclusion."""

            end_messages = [
                SystemMessage(content=end_prompt),
                HumanMessage(content="Generate the end game response")
            ]

            response = state_machine.llm_service.simple_invoke(end_messages)
            final_response = response.content
        except Exception as e:
            print(f"Error generating end game message: {e}")
            # Fallback to simple messages
            if scenario == "draw":
                final_response = "It's a draw! Great game!"
            elif scenario == "player_won":
                final_response = f"Congratulations {state_machine.state.player_name}! You won!"
            elif scenario == "ai_won":
                final_response = f"I won this round! Good game, {state_machine.state.player_name}!"
            else:
                final_response = "Game ended!"

        # Transition to endgame state
        state_machine.state.transition_to('endgame')
        return final_response

# =============================================================================
# MAIN STATE MACHINE
# =============================================================================

class TicTacToeStateMachine:
    """Modular, maintainable Tic-Tac-Toe state machine"""

    def __init__(self, thread_id=None, led_matrix_url=None):
        # Initialize components
        self.config = GameConfig(thread_id, led_matrix_url)
        self.state = GameState()
        self.llm_service = LLMService(self.config)
        self.error_handler = ErrorHandler(self.config.max_errors)
        self.tools_manager = ToolsManager(self.config)

        # Initialize state handlers
        self.handlers = {
            "welcome": WelcomeStateHandler(),
            "player_ready": PlayerReadyStateHandler(),
            "playing": PlayingStateHandler(),
            "endgame": EndgameStateHandler()
        }

        # Initialize conversation with welcome prompt
        welcome_prompt = PromptManager.get_welcome_prompt()
        self.state.messages = [SystemMessage(content=welcome_prompt)]

    def step(self, message: str) -> str:
        """Process a single step in the game state machine"""
        self.state.messages.append(HumanMessage(content=message))

        try:
            handler = self.handlers.get(self.state.current_state)
            if not handler:
                return "Unknown state. Resetting game."

            return handler.handle(self, message)

        except Exception as e:
            return self.error_handler.handle_error(e, f"state {self.state.current_state}")

    def _handle_player_first_move(self) -> str:
        """Handle when player makes the first move"""
        system_prompt = """You are a friendly Tic-Tac-Toe game host. The user is ready to start the game and will make the first move. Prompt the user to make their move by specifying the cell number (1-9) corresponding to the board positions that are already shown in the led matrix. DO NOT show the board again. - DO NOT show ASCII board representations in your responses"""

        self.state.messages[0] = SystemMessage(content=system_prompt)
        response = self.llm_service.simple_invoke(self.state.messages)
        self.state.messages.append(AIMessage(content=response.content))
        return response.content

    def _handle_ai_first_move(self) -> str:
        """Handle when AI makes the first move"""
        system_prompt = """You are a friendly Tic-Tac-Toe game host and very good player. The user is ready to start the game, but you (the AI) will make the first move. Announce your first move by specifying the cell number (1-9) corresponding to the board positions that will be shown in led matrix and then ask the user to make their move. - DO NOT show ASCII board representations in your responses. Extract your move."""

        self.state.messages[0] = SystemMessage(content=system_prompt)

        try:
            ai_move = self.llm_service.invoke_with_retry(self.state.messages, MoveResponse)
            self.state.messages.append(AIMessage(content=ai_move.reply))

            if ai_move.move:
                self.tools_manager.tools.make_move(position=ai_move.move)

            return ai_move.reply

        except Exception as e:
            return self.error_handler.handle_error(e, "AI first move", "I'm having trouble making my move. Let me try again.")

    def reset_state_machine(self):
        """Reset the state machine to initial state"""
        self.state = GameState()
        self.error_handler = ErrorHandler(self.config.max_errors)

        # Reset with welcome prompt
        welcome_prompt = PromptManager.get_welcome_prompt()
        self.state.messages = [SystemMessage(content=welcome_prompt)]

        # Reset LED matrix
        self.tools_manager.full_reset_and_welcome()
