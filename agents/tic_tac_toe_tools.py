"""
TicTacToe Tools for Matrix Portal LED display
Provides HTTP API wrapper for controlling the LED matrix Tic-Tac-Toe game
"""
import os
from typing import Any, Dict, List, Optional
import requests
from langchain_core.tools import StructuredTool


class TicTacToeTools:
    """Thin client for the Matrix Portal Tic-Tac-Toe HTTP API."""

    def __init__(self, base_url: str, session: Optional[requests.Session] = None, agent_name: str = "Agent") -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.agent_name = agent_name
        # Track which tools are currently enabled
        self._enabled_tools = {
            "setup", "status", "move", "welcome", "vs_screen", "start", "reset", "win_anim", "draw_anim"
        }

    def _req(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to the matrix portal API"""
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.request(method, url, timeout=15, **kwargs)
            resp.raise_for_status()
            if resp.headers.get("Content-Type", "").startswith("application/json"):
                return resp.json()
            return {"ok": True, "text": resp.text}
        except requests.RequestException as e:
            return {"error": str(e), "url": url}

    # ==========================================================================
    # API METHODS (Direct HTTP calls)
    # ==========================================================================

    def setup_game(self, user_name: str, bot_name: str, user_symbol: str, starting_player: str) -> Dict[str, Any]:
        """Setup a new game with player information"""
        body = {
            "user_name": user_name,
            "bot_name": bot_name,
            "user_symbol": user_symbol,
            "starting_player": starting_player
        }
        return self._req("POST", "/game/setup", json=body)

    def coinflip(self, user_name: str, ai_name: str, user_symbol: str) -> Dict[str, Any]:
        """Legacy coinflip method for determining who goes first"""
        body = {"user_name": user_name, "ai_name": ai_name, "user_symbol": user_symbol}
        return self._req("POST", "/game/coinflip", json=body)

    def welcome_screen(self) -> Dict[str, Any]:
        """Display welcome screen on LED matrix"""
        return self._req("POST", "/game/welcome")

    def player_vs_screen(self) -> Dict[str, Any]:
        """Display player vs screen animation"""
        return self._req("POST", "/game/playervs")

    def start_game(self) -> Dict[str, Any]:
        """Start the game and show the grid"""
        return self._req("POST", "/game/start")

    def win_animation(self, winner_name: str, winner_symbol: str) -> Dict[str, Any]:
        """Show win animation with winner information"""
        body = {"winner_name": winner_name, "winner_symbol": winner_symbol}
        return self._req("POST", "/game/winanim", json=body)

    def draw_animation(self) -> Dict[str, Any]:
        """Show draw/tie animation"""
        return self._req("POST", "/game/drawanim")

    def get_status(self) -> Dict[str, Any]:
        """Get current game status"""
        return self._req("GET", "/game/status")

    def make_move(self, position: int) -> Dict[str, Any]:
        """Make a move by calling the API directly with just the position."""
        return self._req("POST", "/game/move", json={"position": position})

    def reset(self) -> Dict[str, Any]:
        """Reset the current game"""
        return self._req("POST", "/game/reset")

    def full_reset(self) -> Dict[str, Any]:
        """Full reset - clears everything for new players"""
        return self._req("POST", "/game/fullreset")

    def end(self) -> Dict[str, Any]:
        """End the current game"""
        return self._req("POST", "/game/end")

    # ==========================================================================
    # TOOL MANAGEMENT METHODS
    # ==========================================================================

    def bind_tools(self, tool_names: List[str]) -> None:
        """Enable specific tools by name."""
        valid_tools = {"setup", "status", "move", "welcome", "vs_screen", "start", "reset", "win_anim", "draw_anim"}
        for tool_name in tool_names:
            if tool_name in valid_tools:
                self._enabled_tools.add(tool_name)
            else:
                raise ValueError(f"Unknown tool: {tool_name}. Valid tools: {valid_tools}")

    def unbind_tools(self, tool_names: List[str]) -> None:
        """Disable specific tools by name."""
        for tool_name in tool_names:
            self._enabled_tools.discard(tool_name)

    def get_enabled_tools(self) -> List[str]:
        """Get list of currently enabled tools."""
        return list(self._enabled_tools)

    def get_all_available_tools(self) -> List[str]:
        """Get list of all available tool names."""
        return ["setup", "status", "move", "welcome", "vs_screen", "start", "reset", "win_anim", "draw_anim"]

    def bind_all_tools(self) -> None:
        """Enable all available tools."""
        self._enabled_tools = {"setup", "status", "move", "welcome", "vs_screen", "start", "reset", "win_anim", "draw_anim"}

    def unbind_all_tools(self) -> None:
        """Disable all tools."""
        self._enabled_tools = set()

    # ==========================================================================
    # LANGCHAIN TOOL GENERATION
    # ==========================================================================

    def get_tools(self) -> List[Any]:
        """Return LangChain StructuredTool objects for enabled tools."""

        # Capture instance variables to avoid self reference
        base_url = self.base_url
        agent_name = self.agent_name
        session = self.session

        # Create simple functions without decorators
        def setup_game_func(user_name: str, user_symbol: str, starting_player: str = None, bot_name: str = None):
            """Setup a new tic-tac-toe game with player information"""
            bot = bot_name or agent_name
            starter = starting_player or user_name
            body = {"user_name": user_name, "bot_name": bot, "user_symbol": user_symbol, "starting_player": starter}
            try:
                resp = session.post(f"{base_url}/game/setup", json=body, timeout=15)
                resp.raise_for_status()
                result = resp.json()
                if "user" in result and "bot" in result:
                    agent_num = result["bot"]["player_number"]
                    result["agent_player_number"] = agent_num
                return result
            except Exception as e:
                return {"error": str(e)}

        def get_status_func():
            """Get the current game status including board state and whose turn it is"""
            try:
                resp = session.get(f"{base_url}/game/status", timeout=15)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                return {"error": str(e)}

        def make_move_func(position: int):
            """Make a move at the specified board position (1-9)"""
            try:
                resp = session.post(f"{base_url}/game/move", json={"position": position}, timeout=15)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                return {"error": str(e)}

        def welcome_screen_func():
            """Display the welcome screen on the LED matrix"""
            try:
                resp = session.post(f"{base_url}/game/welcome", timeout=15)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                return {"error": str(e)}

        def player_vs_screen_func():
            """Display the player vs screen animation"""
            try:
                resp = session.post(f"{base_url}/game/playervs", timeout=15)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                return {"error": str(e)}

        def start_game_func():
            """Start the game and display the grid on the LED matrix"""
            try:
                resp = session.post(f"{base_url}/game/start", timeout=15)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                return {"error": str(e)}

        def full_reset_func():
            """Perform a full reset of the game state"""
            try:
                resp = session.post(f"{base_url}/game/fullreset", timeout=15)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                return {"error": str(e)}

        def win_animation_func(winner_name: str, winner_symbol: str):
            """Display win animation with winner information"""
            try:
                resp = session.post(f"{base_url}/game/winanim",
                                  json={"winner_name": winner_name, "winner_symbol": winner_symbol},
                                  timeout=15)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                return {"error": str(e)}

        def draw_animation_func():
            """Display draw/tie animation when game ends in a draw"""
            try:
                resp = session.post(f"{base_url}/game/drawanim", timeout=15)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                return {"error": str(e)}

        # Create StructuredTool objects for enabled tools only
        tools = []

        if "setup" in self._enabled_tools:
            setup_tool = StructuredTool.from_function(
                func=setup_game_func,
                name="ttt_setup_game",
                description="Start a new game with fast setup. Args: user_name, user_symbol (X or O), optional starting_player name, optional bot_name"
            )
            tools.append(setup_tool)

        if "status" in self._enabled_tools:
            status_tool = StructuredTool.from_function(
                func=get_status_func,
                name="ttt_get_status",
                description="Get current game status including board, current_player, winner, game_status"
            )
            tools.append(status_tool)

        if "move" in self._enabled_tools:
            move_tool = StructuredTool.from_function(
                func=make_move_func,
                name="ttt_make_move",
                description="Make a move by sending a board position 1-9"
            )
            tools.append(move_tool)

        if "welcome" in self._enabled_tools:
            welcome_tool = StructuredTool.from_function(
                func=welcome_screen_func,
                name="ttt_welcome_screen",
                description="Show welcome screen with TIC-TAC-TOE WELCOME message"
            )
            tools.append(welcome_tool)

        if "vs_screen" in self._enabled_tools:
            vs_tool = StructuredTool.from_function(
                func=player_vs_screen_func,
                name="ttt_player_vs_screen",
                description="Show player vs screen with current player names"
            )
            tools.append(vs_tool)

        if "start" in self._enabled_tools:
            start_tool = StructuredTool.from_function(
                func=start_game_func,
                name="ttt_start_game",
                description="Start the game and show startup screen + grid (requires setup first)"
            )
            tools.append(start_tool)

        if "reset" in self._enabled_tools:
            reset_tool = StructuredTool.from_function(
                func=full_reset_func,
                name="ttt_full_reset",
                description="Full reset - clears everything for new players"
            )
            tools.append(reset_tool)

        if "win_anim" in self._enabled_tools:
            win_anim_tool = StructuredTool.from_function(
                func=win_animation_func,
                name="ttt_win_animation",
                description="Show win animation with winner name and symbol"
            )
            tools.append(win_anim_tool)

        if "draw_anim" in self._enabled_tools:
            draw_anim_tool = StructuredTool.from_function(
                func=draw_animation_func,
                name="ttt_draw_animation",
                description="Show draw/tie animation when game ends in a draw"
            )
            tools.append(draw_anim_tool)

        return tools


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example usage
    base_url = os.environ.get("MATRIX_URL", "http://192.168.1.100")

    print("✅ TicTacToe Tools Module Ready!")
    print("Usage:")
    print("from tic_tac_toe_tools import TicTacToeTools")
    print(f"tools = TicTacToeTools('{base_url}')")
    print("tools.bind_tools(['status', 'move'])")
    print("langchain_tools = tools.get_tools()")