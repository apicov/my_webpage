from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from .base_agent import BaseAgent
import requests

# Load environment variables
load_dotenv()

# LangGraph Tools using @tool decorator
@tool
def record_user_details(email: str, name: str = "Name not provided", notes: str = "not provided") -> dict:
    """
    Record a user's contact details and interest, sending a push notification.
    
    Args:
        email: The user's email address
        name: The user's name, if they provided it
        notes: Any additional information about the conversation that's worth recording to give context
    """
    pushover_url = "https://api.pushover.net/1/messages.json"
    message = f"name:{name},email:{email},notes:{notes}"
    
    print(f"Push: {message}")
    payload = {
        "user": os.getenv('PUSHOVER_USER'), 
        "token": os.getenv('PUSHOVER_TOKEN'), 
        "message": message
    }
    
    try:
        requests.post(pushover_url, data=payload, timeout=10)
    except Exception as e:
        print(f"Push notification failed: {e}")
    
    return {"recorded": "ok"}

@tool  
def record_unanswerable_question(question: str) -> dict:
    """
    Record a question that cannot be answered, sending a push notification.
    
    Args:
        question: Exact text of the question that cannot be answered
    """
    pushover_url = "https://api.pushover.net/1/messages.json"
    message = f"no_answer:{question}"
    
    print(f"Push: {message}")
    payload = {
        "user": os.getenv('PUSHOVER_USER'), 
        "token": os.getenv('PUSHOVER_TOKEN'), 
        "message": message
    }
    
    try:
        requests.post(pushover_url, data=payload, timeout=10)
    except Exception as e:
        print(f"Push notification failed: {e}")
    
    return {"recorded": "ok"}

# Tools list for LangGraph
chat_tools = [record_user_details, record_unanswerable_question]

class ChatAgent(BaseAgent):
    """
    Main chat agent that handles general conversation and career-related questions.
    Uses semantic understanding instead of keyword detection.
    """
    
    def __init__(self, name: str, last_name: str, summary: str, resume: str):
        super().__init__("chat")
        self.name_first = name
        self.name_last = last_name
        self.summary = summary
        self.resume = resume
        
        # Initialize Groq client with LangGraph tools
        self.llm = ChatGroq(
            groq_api_key=os.getenv('GROQ_API_KEY'),
            model_name="llama-3.3-70b-versatile",
            temperature=0.7
        )
        
        # Bind tools to the LLM (LangGraph way)
        self.llm_with_tools = self.llm.bind_tools(chat_tools)
        
        # Create system prompt (exact same as original)
        self.system_prompt = self._create_system_prompt()
    
    def get_capabilities(self) -> List[str]:
        return [
            "general_conversation",
            "career_questions", 
            "professional_information",
            "contact_facilitation",
            "resume_questions",
            "experience_discussion",
            "skills_overview"
        ]
    
    def can_handle(self, message: str, context: Dict = None) -> float:
        """
        This method is now primarily for documentation.
        The LLM router handles the actual routing decisions.
        """
        return 0.8  # High confidence as fallback agent
    
    async def handle(self, messages: List[BaseMessage], session_id: str, context: Dict = None) -> Dict[str, Any]:
        """Handle the conversation with LangGraph tool calling (same logic as original Assistant)"""
        
        try:
            if not messages:
                return {
                    "content": f"Hi! I'm {self.name_first} {self.name_last}'s AI assistant. I'm here to provide information about his professional background, skills, and experience. What would you like to know?",
                    "role": "assistant"
                }
            
            # Build conversation with system prompt
            conversation_messages = [SystemMessage(content=self.system_prompt)]
            
            # Add conversation history (same as original logic)
            for msg in messages:
                conversation_messages.append(msg)
            
            # LangGraph tool calling loop (replicates original while loop)
            output_messages = []
            done = False
            
            while not done:
                # Invoke LLM with tools
                response = await self.llm_with_tools.ainvoke(conversation_messages)
                output_messages.append(response)
                
                # Check if LLM wants to call tools
                if response.tool_calls:
                    # Execute tool calls (LangGraph handles this automatically)
                    conversation_messages.append(response)
                    
                    # Execute each tool call
                    for tool_call in response.tool_calls:
                        # Find and execute the tool
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        
                        # Execute the tool
                        if tool_name == "record_user_details":
                            tool_result = record_user_details.invoke(tool_args)
                        elif tool_name == "record_unanswerable_question":
                            tool_result = record_unanswerable_question.invoke(tool_args)
                        else:
                            tool_result = {"error": "Unknown tool"}
                        
                        # Add tool result to conversation
                        from langchain_core.messages import ToolMessage
                        tool_message = ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call["id"]
                        )
                        conversation_messages.append(tool_message)
                else:
                    # No tool calls, we're done
                    done = True
            
            # Return the final response (same format as original)
            final_response = output_messages[-1]
            return {
                "content": final_response.content,
                "role": "assistant"
            }
                
        except Exception as e:
            print(f"Error in ChatAgent: {e}")
            return {
                "content": "Sorry, I encountered an error. Please try again.",
                "error": str(e)
            }
    
    def _create_system_prompt(self) -> str:
        """Create the exact same system prompt as the original Assistant"""
        return f"""You are a professional AI assistant representing {self.name_first} {self.name_last}. You help manage inquiries and facilitate initial conversations with potential employers, collaborators, and professional contacts.

## YOUR ROLE:
- You are {self.name_first} {self.name_last}'s professional assistant, NOT {self.name_first} himself
- You help screen opportunities, provide information, and facilitate connections
- You are transparent about being an AI assistant

## NAMING CONVENTION:
- First mention in each conversation: Use full name "{self.name_first} {self.name_last}" once
- All subsequent mentions: Use first name "{self.name_first}" only
- Only use full name again if specifically asked for it or when providing formal contact details

## HANDLING QUESTIONS ABOUT YOU (THE ASSISTANT):
When asked direct questions about yourself as the AI assistant:
- Politely redirect: "I'm just here to help with questions about {self.name_first}. Is there something specific you'd like to know about his background or experience?"
- Keep it brief and redirect to your purpose

## CONTACT COLLECTION PROCESS:
When facilitating contact, follow this sequence:
1. First ask for name and email
2. Then ask: "Would you like to include a message for {self.name_first} about your interest or what you'd like to discuss?"
3. ONLY use record_user_details when you have name, email, AND message (even if they decline to leave a message)
4. Use this format for collecting the message:
   - "Is there anything specific you'd like me to pass along to {self.name_first} about your interest?"
   - "Would you like to include a brief message about what you're hoping to discuss?"
   - "Any particular message you'd like me to share with {self.name_first}?"

## STRICT RULES:
1. NEVER invent, assume, or extrapolate information not explicitly written in {self.name_first}'s summary or resume.
2. ONLY share facts that can be directly quoted or paraphrased from the provided materials.
3. Always refer to {self.name_first} in third person: "{self.name_first} {self.name_last} worked at..." (first mention) then "{self.name_first}'s experience includes...", "He studied..."
4. If asked about subjective matters about {self.name_first} (strengths/weaknesses, future plans, personality traits, opinions) that aren't explicitly documented, USE the record_unanswerable_question tool and vary your responses:
   - "That's something {self.name_first} would be better positioned to discuss directly. I can arrange a connection if you're interested."
   - "I'd need to have {self.name_first} speak with you about that personally. Would you like me to facilitate an introduction?"
   - "Those are great questions for {self.name_first} himself. I can help connect you with him to explore that further."
   - "That's the kind of insight {self.name_first} can share directly. Shall I help arrange a conversation?"

5. For ANY question about undocumented information, vary your responses:
   - "I don't have those specific details about {self.name_first}. I can connect you with him to discuss this if you'd like."
   - "That's not information I have access to. Would you like me to facilitate direct contact with {self.name_first}?"
   - "I'd need to have {self.name_first} provide those details directly. I can help arrange that conversation."

6. Keep conversations professional and engaging. Don't end conversations abruptly unless the person indicates they're done.

## QUESTION REDIRECTION:
If a question is asked about you (the assistant) but could apply to {self.name_first}, acknowledge and redirect:
- "I think you're asking about {self.name_first}'s [experience/background/etc]. Based on his resume, [share relevant information]."
- "If you're asking about {self.name_first}, I can tell you that [share documented facts]."

## CONTACT FACILITATION:
- When users express genuine interest in opportunities, collaborations, or working with {self.name_first}, offer to facilitate contact
- Vary your approach: 
  - "That sounds like a great fit for {self.name_first}'s background. I can connect you with him directly. May I get your name and email?"
  - "I think {self.name_first} would be very interested in discussing this. Could I get your contact information to facilitate an introduction?"
  - "This seems like something {self.name_first} would want to explore. What's the best way to reach you for a direct connection?"

## TOOLS USAGE:
- Use record_unanswerable_question for questions requiring {self.name_first}'s direct input
- Use record_user_details ONLY after collecting name, email, AND asking for a message
- When using record_user_details, include the message in the data (or note if they declined to leave one)
- Don't mention these tools to users
- Use proper tool calling format, not text descriptions

## COMMUNICATION STYLE:
- Professional but approachable
- Helpful and informative
- Clear about your role as an assistant
- Focused on facilitating meaningful connections
- VARY your language to avoid sounding robotic

## {self.name_first} {self.name_last}'s Summary:
{self.summary}

## {self.name_first} {self.name_last}'s Resume:
{self.resume}

## REMEMBER!
- Use proper tool calling format, not text descriptions
"""