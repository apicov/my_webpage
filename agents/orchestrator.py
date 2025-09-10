from typing import Dict, List, Any, TypedDict
import os
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()

class AgentState(TypedDict):
    """State that flows through the agent workflow"""
    messages: List[BaseMessage]
    session_id: str
    current_agent: str
    agent_response: Dict[str, Any]
    context: Dict[str, Any]

class LangGraphOrchestrator:
    """
    LangGraph-based orchestrator that intelligently routes conversations 
    to appropriate agents based on semantic understanding, not keywords.
    """
    
    def __init__(self, name: str, last_name: str, summary: str, resume: str):
        self.name = name
        self.last_name = last_name
        self.summary = summary
        self.resume = resume
        
        # Initialize the routing LLM
        self.router_llm = ChatGroq(
            groq_api_key=os.getenv('GROQ_API_KEY'),
            model_name="llama-3.3-70b-versatile",
            temperature=0.1  # Low temperature for consistent routing
        )
        
        # Agent registry - will be populated dynamically
        self.agents = {}
        
        # Initialize with chat agent
        from .chat_agent import ChatAgent
        self.agents['chat'] = ChatAgent(name, last_name, summary, resume)
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
    
    def register_agent(self, agent_name: str, agent_instance):
        """Register a new agent - makes adding agents trivial"""
        self.agents[agent_name] = agent_instance
        print(f"✅ Registered agent: {agent_name}")
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        # Define the workflow
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._route_to_agent)
        workflow.add_node("execute_agent", self._execute_agent)
        
        # Define the flow
        workflow.set_entry_point("router")
        workflow.add_edge("router", "execute_agent")
        workflow.add_edge("execute_agent", END)
        
        return workflow.compile()
    
    async def _route_to_agent(self, state: AgentState) -> AgentState:
        """
        Intelligently route to the best agent using LLM reasoning,
        not hardcoded keywords.
        """
        
        if not state["messages"]:
            state["current_agent"] = "chat"
            return state
        
        latest_message = state["messages"][-1]
        
        # Create agent capability descriptions
        agent_descriptions = {}
        for agent_name, agent in self.agents.items():
            capabilities = agent.get_capabilities()
            agent_descriptions[agent_name] = {
                "name": agent_name,
                "capabilities": capabilities,
                "description": self._get_agent_description(agent_name)
            }
        
        # Create routing prompt
        routing_prompt = self._create_routing_prompt(
            latest_message.content, 
            agent_descriptions
        )
        
        try:
            # Get routing decision from LLM
            response = await self.router_llm.ainvoke([
                SystemMessage(content=routing_prompt),
                HumanMessage(content=f"User message: {latest_message.content}")
            ])
            
            # Parse the response to get the agent name
            chosen_agent = self._parse_routing_response(response.content)
            
            # Validate the chosen agent exists
            if chosen_agent not in self.agents:
                chosen_agent = "chat"  # Fallback to chat
            
            state["current_agent"] = chosen_agent
            
        except Exception as e:
            print(f"Routing error: {e}, defaulting to chat agent")
            state["current_agent"] = "chat"
        
        return state
    
    async def _execute_agent(self, state: AgentState) -> AgentState:
        """Execute the chosen agent"""
        
        agent_name = state["current_agent"]
        agent = self.agents.get(agent_name)
        
        if not agent:
            # Fallback to chat agent
            agent = self.agents["chat"]
        
        try:
            # Execute the agent
            response = await agent.handle(
                state["messages"], 
                state["session_id"], 
                state.get("context", {})
            )
            
            state["agent_response"] = response
            
        except Exception as e:
            print(f"Agent execution error: {e}")
            state["agent_response"] = {
                "content": "I apologize, but I encountered an error. Please try again.",
                "error": str(e)
            }
        
        return state
    
    def _create_routing_prompt(self, user_message: str, agent_descriptions: Dict) -> str:
        """Create a prompt for intelligent agent routing"""
        
        agents_info = ""
        for agent_name, info in agent_descriptions.items():
            agents_info += f"\n- {agent_name}: {info['description']}"
        
        return f"""You are an intelligent agent router. Your job is to analyze user messages and determine which specialized agent should handle the conversation.

Available agents:{agents_info}

Rules for routing:
1. Choose the agent that is MOST specialized for the user's request
2. If the request could fit multiple agents, choose the most specific one
3. For general conversation or career questions, use 'chat'
4. Consider the INTENT behind the message, not just keywords
5. If unsure, default to 'chat'

Respond with ONLY the agent name (e.g., 'chat', 'video', 'sensor', etc.). No explanation needed."""
    
    def _get_agent_description(self, agent_name: str) -> str:
        """Get human-readable description of what each agent does"""
        descriptions = {
            "chat": "Handles general conversation, career questions, and professional information about the candidate",
            "video": "Controls video streams, camera feeds, and hardware demonstrations via WebRTC",
            "sensor": "Reads and displays sensor data, environmental monitoring, and live data visualization", 
            "thesis": "Answers questions about research, thesis content, and academic papers using RAG",
            "hardware": "Controls IoT devices, Raspberry Pi hardware, and physical demonstrations"
        }
        return descriptions.get(agent_name, "General purpose agent")
    
    def _parse_routing_response(self, response: str) -> str:
        """Parse the LLM routing response to extract agent name"""
        response = response.strip().lower()
        
        # Look for agent names in the response
        for agent_name in self.agents.keys():
            if agent_name in response:
                return agent_name
        
        # Default to chat if no clear match
        return "chat"
    
    async def get_response(self, messages: List[Dict]) -> List[Dict]:
        """
        Main entry point - replaces the original Assistant.get_response method
        """
        
        # Convert messages to LangChain format
        lc_messages = []
        for msg in messages:
            if msg.get("role") == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
        
        # Create initial state
        initial_state = AgentState(
            messages=lc_messages,
            session_id="web_session",  # You can make this dynamic
            current_agent="",
            agent_response={},
            context={}
        )
        
        try:
            # Run the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Return response in the expected format
            response = final_state["agent_response"]
            
            return [{
                "role": "assistant",
                "content": response.get("content", "I apologize, but I couldn't generate a response.")
            }]
            
        except Exception as e:
            print(f"Orchestrator error: {e}")
            return [{
                "role": "assistant", 
                "content": "I apologize, but I encountered an error. Please try again."
            }]