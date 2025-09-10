# Multi-Agent Architecture with LangGraph

## Table of Contents
- [Overview](#overview)
- [Why LangGraph?](#why-langgraph)
- [Architecture Components](#architecture-components)
- [Code Walkthrough](#code-walkthrough)
- [Adding New Agents](#adding-new-agents)
- [Semantic Routing](#semantic-routing)
- [Comparison with Original](#comparison-with-original)
- [Future Extensions](#future-extensions)

## Overview

This document explains the new multi-agent architecture that replaces your original `AI_career_assistant` with a scalable, intelligent system powered by **LangGraph**.

### What We Built
- **Intelligent Orchestrator**: Routes conversations to specialized agents using semantic understanding
- **Extensible Agent System**: Easy to add new agents (video, sensor, thesis, etc.)
- **Backward Compatibility**: Your Flask app works exactly the same
- **No Keyword Detection**: Uses LLM reasoning instead of hardcoded rules

### Key Benefits
1. **Semantic Routing**: LLM decides which agent to use based on intent, not keywords
2. **Easy Expansion**: Add new agents by just creating a new file
3. **Stateful Workflows**: LangGraph manages conversation state and flow
4. **Tool Integration**: Each agent can have specialized tools
5. **Session Management**: Built-in support for user sessions and locks

---

## Why LangGraph?

### Traditional Multi-Agent Problems
```python
# ❌ Old approach - brittle keyword detection
if "video" in message.lower():
    return video_agent.handle(message)
elif "sensor" in message.lower():
    return sensor_agent.handle(message)
# What if user says "show me the camera feed"? 
# What about "temperature readings"?
```

### LangGraph Solution
```python
# ✅ New approach - semantic understanding
routing_prompt = """
You are an intelligent router. Which agent should handle:
"Can you show me the camera feed from the lab?"

Available agents:
- chat: General conversation and career info
- video: Camera feeds and hardware demonstrations  
- sensor: Environmental data and readings
"""
# LLM responds: "video" (understands intent!)
```

### LangGraph Key Concepts

#### 1. **State Management**
```python
class AgentState(TypedDict):
    messages: List[BaseMessage]      # Conversation history
    session_id: str                  # User session
    current_agent: str               # Which agent is handling this
    agent_response: Dict[str, Any]   # Agent's response
    context: Dict[str, Any]          # Additional context
```

#### 2. **Graph Workflows**
```python
workflow = StateGraph(AgentState)
workflow.add_node("router", self._route_to_agent)      # Step 1: Route
workflow.add_node("execute_agent", self._execute_agent) # Step 2: Execute
workflow.set_entry_point("router")                     # Start here
workflow.add_edge("router", "execute_agent")           # Flow: router → execute
workflow.add_edge("execute_agent", END)                # Flow: execute → end
```

#### 3. **Async Processing**
LangGraph handles async operations naturally, perfect for:
- Multiple API calls
- Video stream management  
- Sensor data polling
- Database operations

---

## Architecture Components

### 1. **Orchestrator** (`agents/orchestrator.py`)
The brain of the system that decides which agent should handle each conversation.

```python
class LangGraphOrchestrator:
    def __init__(self, name, last_name, summary, resume):
        # Initialize routing LLM
        self.router_llm = ChatGroq(...)
        
        # Agent registry - automatically populated
        self.agents = {}
        
        # Create workflow graph
        self.workflow = self._create_workflow()
```

**Key Methods:**
- `_route_to_agent()`: Uses LLM to choose the best agent
- `_execute_agent()`: Runs the chosen agent
- `register_agent()`: Adds new agents dynamically

### 2. **Base Agent** (`agents/base_agent.py`)
Abstract class that all agents inherit from. Ensures consistency.

```python
class BaseAgent(ABC):
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """What can this agent do?"""
        pass
    
    @abstractmethod
    def can_handle(self, message: str, context: Dict = None) -> float:
        """How confident is this agent? (0-1 score)"""
        pass
    
    @abstractmethod
    async def handle(self, messages: List[BaseMessage], session_id: str, context: Dict = None):
        """Process the conversation"""
        pass
```

### 3. **Chat Agent** (`agents/chat_agent.py`)
Handles general conversation and career questions. Replaces your original Assistant.

```python
class ChatAgent(BaseAgent):
    def __init__(self, name, last_name, summary, resume):
        # Uses your exact same prompts and behavior!
        self.system_prompt = self._create_system_prompt()
        
    def get_capabilities(self):
        return ["general_conversation", "career_questions", "professional_information"]
```

### 4. **Flask Integration** (`app.py`)
Direct integration with LangGraph orchestrator.

```python
# Direct import - no wrapper needed
from agents.orchestrator import LangGraphOrchestrator
import asyncio

# Create orchestrator instance
assistant = LangGraphOrchestrator(name, last_name, summary, resume)

def get_ai_response(messages):
    # Handle async LangGraph calls
    return asyncio.run(assistant.get_response(messages))
```

---

## Code Walkthrough

### Flow Example: User asks "What's your experience with IoT?"

#### Step 1: Message Arrives
```python
# In Flask app with LangGraph
messages = [{"role": "user", "content": "What's your experience with IoT?"}]
response = asyncio.run(assistant.get_response(messages))
```

#### Step 2: Orchestrator Routes
```python
# LangGraph workflow starts
initial_state = AgentState(
    messages=[HumanMessage(content="What's your experience with IoT?")],
    session_id="web_session",
    current_agent="",
    agent_response={},
    context={}
)

# Router analyzes the message
routing_prompt = """
Available agents:
- chat: General conversation and career info
- video: Camera feeds and hardware demonstrations  
- sensor: Environmental data and readings

User message: "What's your experience with IoT?"
"""

# LLM responds: "chat" (career question)
state["current_agent"] = "chat"
```

#### Step 3: Agent Executes
```python
# ChatAgent handles the conversation
chat_agent = self.agents["chat"]
response = await chat_agent.handle(
    messages=[HumanMessage(content="What's your experience with IoT?")],
    session_id="web_session"
)

# Uses your original system prompt and resume data
```

#### Step 4: Response Returns
```python
# Back through Flask to frontend
return [{
    "role": "assistant",
    "content": "Based on Antonio Pico's resume, he has extensive IoT experience including..."
}]
```

---

## Adding New Agents

### Example: Video Stream Agent

#### 1. Create the Agent File
```python
# agents/video_agent.py
from .base_agent import BaseAgent
import httpx  # For Raspberry Pi API calls

class VideoStreamAgent(BaseAgent):
    def __init__(self):
        super().__init__("video")
        self.active_sessions = {}  # Track who's using video
        self.pi_api_url = "http://raspberrypi.local:8080"
    
    def get_capabilities(self) -> List[str]:
        return [
            "video_streaming",
            "camera_control", 
            "hardware_demonstration",
            "webrtc_connection"
        ]
    
    def can_handle(self, message: str, context: Dict = None) -> float:
        # Not needed anymore - LLM router decides!
        return 0.9
    
    async def handle(self, messages: List[BaseMessage], session_id: str, context: Dict = None):
        # Check if video stream is available (single user limit)
        if not self._can_start_stream(session_id):
            return {
                "content": "Video stream is currently in use. You're #2 in queue.",
                "ui_commands": [{"type": "show_queue", "position": 2}]
            }
        
        # Start camera on Raspberry Pi
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self.pi_api_url}/camera/start")
            stream_data = response.json()
        
        # Lock session for 5 minutes
        self._lock_session(session_id, timeout=300)
        
        return {
            "content": "Starting camera feed...",
            "media": {
                "type": "webrtc_stream",
                "stream_url": stream_data["stream_url"],
                "ice_servers": stream_data["ice_servers"]
            },
            "ui_commands": [
                {"type": "set_display", "content_type": "video"},
                {"type": "start_webrtc", "config": stream_data}
            ]
        }
    
    def requires_session_lock(self) -> bool:
        return True  # Only one user at a time
    
    def get_session_timeout(self) -> int:
        return 300  # 5 minutes
```

#### 2. Register the Agent
```python
# Option A: In orchestrator.py __init__
from .video_agent import VideoStreamAgent
self.agents['video'] = VideoStreamAgent()

# Option B: In app.py (recommended for dynamic agents)
from agents.video_agent import VideoStreamAgent
assistant.register_agent("video", VideoStreamAgent())
```

#### 3. Update Agent Descriptions
```python
# In orchestrator.py _get_agent_description()
descriptions = {
    "chat": "Handles general conversation and career questions",
    "video": "Controls video streams, camera feeds, and hardware demonstrations via WebRTC",
    # ... other agents
}
```

**That's it!** The LLM router will automatically understand requests like:
- "Show me the camera"
- "I want to see the hardware demo"  
- "Can you start the video feed?"

### Example: Sensor Data Agent

```python
# agents/sensor_agent.py
class SensorAgent(BaseAgent):
    def get_capabilities(self):
        return ["sensor_readings", "environmental_data", "live_monitoring"]
    
    async def handle(self, messages, session_id, context):
        # Read sensor data via MCP
        sensor_data = await self.mcp_client.read_sensors()
        
        return {
            "content": f"Current readings: Temperature {sensor_data['temp']}°C, Humidity {sensor_data['humidity']}%",
            "media": {
                "type": "chart_data",
                "data": sensor_data
            },
            "ui_commands": [{"type": "set_display", "content_type": "chart"}]
        }
```

---

## Semantic Routing

### How It Works

#### Traditional Keyword Approach
```python
# ❌ Brittle and limited
keywords = {
    "video": ["camera", "video", "stream", "show"],
    "sensor": ["temperature", "humidity", "sensor"]
}

# Fails on: "I'd like to see the lab equipment"
# Fails on: "What's the current climate data?"
```

#### LangGraph Semantic Approach
```python
# ✅ Intelligent and flexible
routing_prompt = f"""
You are an expert at understanding user intent. 

Available specialized agents:
- video: Controls camera feeds, hardware demonstrations, live streams
- sensor: Environmental monitoring, temperature/humidity readings, data visualization
- chat: Career information, general conversation, professional background

User request: "{user_message}"

Which agent should handle this? Consider the USER'S INTENT, not just keywords.
Respond with just the agent name.
"""

# Understands:
# "Show me the lab" → video
# "What's the climate like?" → sensor  
# "Tell me about your work" → chat
```

### Routing Decision Tree
```
User Message
     ↓
LLM Analyzes Intent
     ↓
Considers Agent Capabilities
     ↓
Returns Best Agent Name
     ↓
Orchestrator Executes Agent
```

### Routing Examples

| User Input | LLM Reasoning | Chosen Agent |
|------------|---------------|--------------|
| "Show me the camera" | User wants visual feed → video | `video` |
| "What's the temperature?" | User wants sensor data → sensor | `sensor` |
| "Tell me about your experience" | User wants career info → chat | `chat` |
| "I'd like to see your lab setup" | User wants visual demonstration → video | `video` |
| "How's the environment?" | Could be sensor data → sensor | `sensor` |
| "What do you do for work?" | Professional question → chat | `chat` |

---

## Comparison with Original

### Original Architecture
```
Flask App
    ↓
Assistant Class
    ↓
Single Agent with Tools
    ↓
Hardcoded Tool Selection
    ↓
Response
```

### New LangGraph Architecture
```
Flask App (app.py)
    ↓
Direct LangGraph Integration
    ↓
LangGraph Orchestrator
    ↓
Semantic Router (LLM-powered)
    ↓
Specialized Agents
    ↓
Dynamic Tool Selection
    ↓
Enhanced Response
```

### What Stayed the Same
- ✅ Flask endpoints unchanged
- ✅ Frontend interface unchanged  
- ✅ Your exact system prompts
- ✅ Response format
- ✅ Professional behavior

### What Improved
- ✅ Intelligent routing instead of keywords
- ✅ Easy to add new agents
- ✅ Better session management
- ✅ Support for complex workflows
- ✅ Async processing capabilities
- ✅ Enhanced UI control commands

---

## Future Extensions

### 1. RAG Thesis Agent
```python
class ThesisAgent(BaseAgent):
    def __init__(self, thesis_path):
        # Load thesis into vector database
        self.vector_store = self._load_thesis(thesis_path)
        
    async def handle(self, messages, session_id, context):
        query = messages[-1].content
        
        # RAG: Retrieve relevant thesis sections
        relevant_docs = self.vector_store.similarity_search(query, k=3)
        
        # Generate response with context
        context_prompt = f"""
        Based on these thesis sections:
        {relevant_docs}
        
        Answer: {query}
        """
        
        response = await self.llm.ainvoke([SystemMessage(content=context_prompt)])
        
        return {
            "content": response.content,
            "metadata": {
                "sources": [doc.metadata for doc in relevant_docs]
            }
        }
```

### 2. Hardware Control Agent  
```python
class HardwareAgent(BaseAgent):
    async def handle(self, messages, session_id, context):
        # Parse hardware command
        command = self._parse_hardware_command(messages[-1].content)
        
        # Execute on Raspberry Pi
        result = await self.pi_controller.execute(command)
        
        return {
            "content": f"Hardware command executed: {result}",
            "ui_commands": [
                {"type": "show_hardware_status", "status": result}
            ]
        }
```

### 3. Multi-Modal Agent
```python
class MultiModalAgent(BaseAgent):
    async def handle(self, messages, session_id, context):
        # Handle text, images, and video in one agent
        last_message = messages[-1]
        
        if hasattr(last_message, 'image_url'):
            # Process image with vision model
            analysis = await self.vision_model.analyze(last_message.image_url)
            return {"content": f"I can see {analysis}"}
        
        # Handle text normally
        return await super().handle(messages, session_id, context)
```

### 4. Advanced Workflow
```python
# Complex multi-step workflows
workflow = StateGraph(AgentState)

# Add conditional routing
workflow.add_conditional_edges(
    "router",
    lambda state: "video" if state["needs_hardware"] else "chat"
)

# Add parallel processing
workflow.add_node("fetch_data", fetch_sensor_data)
workflow.add_node("process_video", process_video_stream)

# Run multiple agents in parallel when needed
```

---

## Key Learning Points

### 1. **LangGraph State Management**
- State flows through all nodes
- Each node can modify the state
- Type hints ensure consistency

### 2. **Async/Await Patterns**
- All LangGraph operations are async
- Enables parallel processing
- Better for I/O operations (API calls, database)

### 3. **Agent Design Patterns**
- Inherit from BaseAgent for consistency
- Implement required abstract methods
- Use capabilities for documentation

### 4. **Routing Intelligence**
- LLM makes routing decisions
- No hardcoded rules needed
- Easily handles edge cases

### 5. **Extensibility**
- Add agents by creating files
- Register in orchestrator
- System automatically adapts

This architecture gives you a professional, scalable foundation that can grow from a simple chat assistant to a complex multi-agent system controlling hardware, analyzing documents, and managing real-time data streams.

The beauty is that each new capability is just a new agent file away! 🚀