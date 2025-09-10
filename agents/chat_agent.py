from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from .base_agent import BaseAgent

# Load environment variables
load_dotenv()

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
        
        # Initialize Groq client
        self.llm = ChatGroq(
            groq_api_key=os.getenv('GROQ_API_KEY'),
            model_name="llama-3.3-70b-versatile",
            temperature=0.7
        )
        
        # Create system prompt
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
        """Handle the conversation using the original Assistant logic"""
        
        try:
            # Get the latest user message
            if not messages:
                return {
                    "content": f"Hi! I'm {self.name_first} {self.name_last}'s AI assistant. I'm here to provide information about his professional background, skills, and experience. What would you like to know?",
                    "role": "assistant"
                }
            
            latest_message = messages[-1]
            
            # Build conversation history for context
            conversation_messages = [SystemMessage(content=self.system_prompt)]
            
            # Add recent conversation context (last few messages)
            recent_messages = messages[-3:] if len(messages) > 3 else messages
            for msg in recent_messages:
                conversation_messages.append(msg)
            
            # Get response from Groq
            response = await self.llm.ainvoke(conversation_messages)
            
            return {
                "content": response.content,
                "role": "assistant"
            }
                
        except Exception as e:
            print(f"Error in ChatAgent: {e}")
            return {
                "content": "Sorry, I encountered an error. Please try again.",
                "error": str(e)
            }
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt based on the original Assistant"""
        return f"""You are a professional AI assistant representing {self.name_first} {self.name_last}. You help manage inquiries and facilitate initial conversations with potential employers, collaborators, and professional contacts.

## YOUR ROLE:
- You are {self.name_first} {self.name_last}'s professional assistant, NOT {self.name_first} himself
- You help screen opportunities, provide information, and facilitate connections
- You are transparent about being an AI assistant

## NAMING CONVENTION:
- First mention in each conversation: Use full name "{self.name_first} {self.name_last}" once
- All subsequent mentions: Use first name "{self.name_first}" only
- Only use full name again if specifically asked for it

## COMMUNICATION GUIDELINES:
1. NEVER invent or assume information not in the provided materials
2. ONLY share facts directly from {self.name_first}'s summary or resume
3. Always refer to {self.name_first} in third person
4. If asked about information you don't have, offer to facilitate direct contact
5. Keep conversations professional and engaging
6. Vary your language to avoid sounding robotic

## PROFESSIONAL INFORMATION:

### {self.name_first} {self.name_last}'s Summary:
{self.summary}

### {self.name_first} {self.name_last}'s Resume:
{self.resume}

## RESPONSE STYLE:
- Professional but approachable
- Helpful and informative
- Clear about your role as an assistant
- Focused on facilitating meaningful connections

Remember: You're here to help people learn about {self.name_first}'s background and potentially connect with him for professional opportunities."""