# LLM Agents Tutorial: Building Autonomous AI Systems

## 🤖 Introduction to LLM Agents

LLM Agents are autonomous AI systems that can plan, reason, and execute complex tasks using language models as their "brain." This tutorial covers everything from basic agent architectures to advanced multi-agent systems.

**What you'll learn:**
- Agent architectures and frameworks
- Planning and reasoning systems
- Tool use and function calling
- Multi-agent coordination
- Building autonomous AI assistants

---

## 🏗️ Chapter 1: Basic Agent Architecture

### Simple ReAct Agent

```python
import openai
import json
import re
from typing import List, Dict, Any

class ReActAgent:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.conversation_history = []
        self.tools = {}
        
    def add_tool(self, name: str, function: callable, description: str):
        """Add a tool that the agent can use"""
        self.tools[name] = {
            'function': function,
            'description': description
        }
    
    def think(self, query: str) -> str:
        """Generate reasoning and action plan"""
        
        # Build system prompt
        system_prompt = f"""You are an AI agent that can use tools to solve problems.

Available tools:
{self._format_tools()}

Use the following format:
Thought: I need to think about this step by step
Action: tool_name
Action Input: input_for_tool
Observation: result_from_tool
... (repeat if needed)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Let's solve: {query}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    def execute(self, query: str) -> str:
        """Execute a query using the ReAct pattern"""
        
        reasoning = self.think(query)
        print(f"Agent reasoning:\n{reasoning}")
        
        # Parse the reasoning to extract actions
        actions = self._parse_actions(reasoning)
        
        for action in actions:
            if action['action'] in self.tools:
                try:
                    result = self.tools[action['action']]['function'](action['input'])
                    print(f"Tool {action['action']} returned: {result}")
                except Exception as e:
                    print(f"Error executing {action['action']}: {e}")
        
        return reasoning
    
    def _format_tools(self) -> str:
        """Format available tools for the prompt"""
        tool_descriptions = []
        for name, tool in self.tools.items():
            tool_descriptions.append(f"- {name}: {tool['description']}")
        return "\n".join(tool_descriptions)
    
    def _parse_actions(self, reasoning: str) -> List[Dict[str, str]]:
        """Parse actions from reasoning text"""
        actions = []
        
        # Simple regex parsing
        action_pattern = r"Action: (\w+)\nAction Input: (.+?)(?=\n|$)"
        matches = re.findall(action_pattern, reasoning, re.MULTILINE)
        
        for match in matches:
            actions.append({
                'action': match[0],
                'input': match[1].strip()
            })
        
        return actions

# Example usage
def search_web(query):
    """Mock web search function"""
    return f"Search results for '{query}': Some relevant information found."

def calculate(expression):
    """Mock calculator function"""
    try:
        return eval(expression)
    except:
        return "Error: Invalid expression"

# Create and configure agent
agent = ReActAgent()
agent.add_tool("search", search_web, "Search the web for information")
agent.add_tool("calculate", calculate, "Perform mathematical calculations")

# Test the agent
result = agent.execute("What is the population of Tokyo and what is 2^10?")
print(f"\nFinal result:\n{result}")
```

### Function Calling Agent

```python
class FunctionCallingAgent:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.functions = []
        
    def add_function(self, name: str, description: str, parameters: Dict, function: callable):
        """Add a function that can be called by the agent"""
        self.functions.append({
            "name": name,
            "description": description,
            "parameters": parameters,
            "function": function
        })
    
    def execute(self, query: str) -> str:
        """Execute query using function calling"""
        
        # Prepare function definitions for OpenAI
        function_definitions = []
        for func in self.functions:
            function_definitions.append({
                "name": func["name"],
                "description": func["description"],
                "parameters": func["parameters"]
            })
        
        # First call to determine if function should be called
        messages = [{"role": "user", "content": query}]
        
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            functions=function_definitions,
            function_call="auto"
        )
        
        response_message = response.choices[0].message
        
        # Check if function was called
        if response_message.get("function_call"):
            function_name = response_message["function_call"]["name"]
            function_args = json.loads(response_message["function_call"]["arguments"])
            
            # Find and execute the function
            for func in self.functions:
                if func["name"] == function_name:
                    try:
                        function_result = func["function"](**function_args)
                        
                        # Second call with function result
                        messages.append(response_message)
                        messages.append({
                            "role": "function",
                            "name": function_name,
                            "content": str(function_result)
                        })
                        
                        final_response = openai.ChatCompletion.create(
                            model=self.model_name,
                            messages=messages
                        )
                        
                        return final_response.choices[0].message.content
                        
                    except Exception as e:
                        return f"Error executing function {function_name}: {e}"
        
        return response_message.content

# Example usage
function_agent = FunctionCallingAgent()

# Add weather function
weather_schema = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
        }
    },
    "required": ["location"]
}

def get_weather(location):
    """Mock weather function"""
    return f"Weather in {location}: Sunny, 75°F"

function_agent.add_function(
    "get_weather",
    "Get the current weather in a given location",
    weather_schema,
    get_weather
)

# Test function calling
result = function_agent.execute("What's the weather like in New York?")
print(result)
```

---

## 🧠 Chapter 2: Planning and Reasoning Agents

### Chain of Thought Agent

```python
class ChainOfThoughtAgent:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.memory = []
        
    def solve_problem(self, problem: str) -> str:
        """Solve a problem using chain of thought reasoning"""
        
        prompt = f"""Let's solve this problem step by step:

Problem: {problem}

Let me think through this step by step:

1) First, I need to understand what's being asked
2) Then, I'll break it down into smaller parts
3) I'll solve each part systematically
4) Finally, I'll combine the results

Let me start:"""

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant that solves problems step by step."},
            {"role": "user", "content": prompt}
        ]
        
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=1000
        )
        
        reasoning = response.choices[0].message.content
        
        # Store in memory
        self.memory.append({
            "problem": problem,
            "reasoning": reasoning,
            "timestamp": datetime.now()
        })
        
        return reasoning
    
    def get_similar_problems(self, problem: str) -> List[Dict]:
        """Find similar problems from memory"""
        # Simple similarity check (could be improved with embeddings)
        similar = []
        for memory_item in self.memory:
            if any(word in memory_item["problem"].lower() 
                   for word in problem.lower().split()):
                similar.append(memory_item)
        return similar

# Example usage
cot_agent = ChainOfThoughtAgent()

problem = """
If a train travels 120 km in 2 hours, and then travels 180 km in 3 hours, 
what is the average speed for the entire journey?
"""

solution = cot_agent.solve_problem(problem)
print(solution)
```

### Tree of Thoughts Agent

```python
class TreeOfThoughtsAgent:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        
    def generate_thoughts(self, problem: str, num_thoughts: int = 3) -> List[str]:
        """Generate multiple thoughts for a problem"""
        
        prompt = f"""Given this problem: {problem}

Generate {num_thoughts} different approaches or thoughts to solve this problem.
Each thought should be a different strategy or perspective.

Thought 1:"""
        
        messages = [
            {"role": "system", "content": "You are a creative problem solver."},
            {"role": "user", "content": prompt}
        ]
        
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=0.8,
            max_tokens=500
        )
        
        # Parse thoughts from response
        thoughts_text = response.choices[0].message.content
        thoughts = []
        
        # Simple parsing (could be improved)
        lines = thoughts_text.split('\n')
        current_thought = ""
        
        for line in lines:
            if line.strip().startswith('Thought'):
                if current_thought:
                    thoughts.append(current_thought.strip())
                current_thought = line
            else:
                current_thought += " " + line
        
        if current_thought:
            thoughts.append(current_thought.strip())
        
        return thoughts[:num_thoughts]
    
    def evaluate_thought(self, problem: str, thought: str) -> float:
        """Evaluate the quality of a thought (0-1)"""
        
        prompt = f"""Problem: {problem}

Thought: {thought}

Rate this thought from 0 to 10 on how likely it is to lead to a solution.
Consider clarity, feasibility, and relevance.

Rating:"""
        
        messages = [
            {"role": "system", "content": "You are an evaluator. Respond only with a number from 0-10."},
            {"role": "user", "content": prompt}
        ]
        
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=10
        )
        
        try:
            rating = float(response.choices[0].message.content.strip())
            return min(max(rating / 10, 0), 1)  # Normalize to 0-1
        except:
            return 0.5
    
    def solve_with_tree_of_thoughts(self, problem: str, max_iterations: int = 3) -> str:
        """Solve problem using tree of thoughts approach"""
        
        best_solution = ""
        best_score = 0
        
        for iteration in range(max_iterations):
            print(f"Iteration {iteration + 1}:")
            
            # Generate thoughts
            thoughts = self.generate_thoughts(problem, num_thoughts=3)
            
            # Evaluate each thought
            for i, thought in enumerate(thoughts):
                score = self.evaluate_thought(problem, thought)
                print(f"Thought {i+1} (Score: {score:.2f}): {thought[:100]}...")
                
                if score > best_score:
                    best_score = score
                    best_solution = thought
            
            # If we have a good solution, stop
            if best_score > 0.8:
                break
        
        return f"Best solution found (score: {best_score:.2f}):\n{best_solution}"

# Example usage
tot_agent = TreeOfThoughtsAgent()

problem = "How can we reduce plastic waste in our community?"

solution = tot_agent.solve_with_tree_of_thoughts(problem)
print(solution)
```

---

## 🔧 Chapter 3: Tool-Using Agents

### Multi-Tool Agent

```python
import requests
import wikipedia
import json
from datetime import datetime

class MultiToolAgent:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.tools = {}
        self.tool_history = []
        
    def add_tool(self, name: str, function: callable, description: str, schema: Dict):
        """Add a tool with schema"""
        self.tools[name] = {
            'function': function,
            'description': description,
            'schema': schema
        }
    
    def execute_with_tools(self, query: str) -> str:
        """Execute query using available tools"""
        
        # Prepare function definitions
        function_definitions = []
        for name, tool in self.tools.items():
            function_definitions.append({
                "name": name,
                "description": tool['description'],
                "parameters": tool['schema']
            })
        
        messages = [{"role": "user", "content": query}]
        
        # First call to determine tool usage
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            functions=function_definitions,
            function_call="auto"
        )
        
        response_message = response.choices[0].message
        
        # Handle function calls
        if response_message.get("function_call"):
            return self._handle_function_call(response_message, messages)
        
        return response_message.content
    
    def _handle_function_call(self, response_message, messages):
        """Handle function call and continue conversation"""
        
        function_name = response_message["function_call"]["name"]
        function_args = json.loads(response_message["function_call"]["arguments"])
        
        # Execute function
        if function_name in self.tools:
            try:
                function_result = self.tools[function_name]['function'](**function_args)
                
                # Log tool usage
                self.tool_history.append({
                    'tool': function_name,
                    'args': function_args,
                    'result': function_result,
                    'timestamp': datetime.now()
                })
                
                # Continue conversation with result
                messages.append(response_message)
                messages.append({
                    "role": "function",
                    "name": function_name,
                    "content": str(function_result)
                })
                
                final_response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=messages
                )
                
                return final_response.choices[0].message.content
                
            except Exception as e:
                return f"Error executing {function_name}: {e}"
        
        return "Tool not found"

# Tool implementations
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information"""
    try:
        results = wikipedia.search(query, results=3)
        if results:
            page = wikipedia.page(results[0])
            return f"Wikipedia result for '{query}': {page.summary[:500]}..."
        else:
            return f"No Wikipedia results found for '{query}'"
    except Exception as e:
        return f"Error searching Wikipedia: {e}"

def get_current_time() -> str:
    """Get current time and date"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate_math(expression: str) -> str:
    """Evaluate mathematical expression"""
    try:
        # Safe evaluation (could be improved with more security)
        allowed_names = {
            k: v for k, v in math.__dict__.items() 
            if not k.startswith('__')
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {e}"

# Create and configure agent
multi_tool_agent = MultiToolAgent()

# Add tools
multi_tool_agent.add_tool(
    "search_wikipedia",
    search_wikipedia,
    "Search Wikipedia for information about a topic",
    {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            }
        },
        "required": ["query"]
    }
)

multi_tool_agent.add_tool(
    "get_current_time",
    get_current_time,
    "Get the current time and date",
    {
        "type": "object",
        "properties": {},
        "required": []
    }
)

multi_tool_agent.add_tool(
    "calculate_math",
    calculate_math,
    "Evaluate a mathematical expression",
    {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate"
            }
        },
        "required": ["expression"]
    }
)

# Test the agent
queries = [
    "What is the current time?",
    "Tell me about artificial intelligence",
    "What is 2^10 + 5*3?"
]

for query in queries:
    print(f"\nQuery: {query}")
    result = multi_tool_agent.execute_with_tools(query)
    print(f"Result: {result}")
```

---

## 🤝 Chapter 4: Multi-Agent Systems

### Collaborative Agents

```python
class CollaborativeAgent:
    def __init__(self, name: str, role: str, model_name="gpt-3.5-turbo"):
        self.name = name
        self.role = role
        self.model_name = model_name
        self.memory = []
        self.collaborators = {}
        
    def add_collaborator(self, agent):
        """Add a collaborator agent"""
        self.collaborators[agent.name] = agent
    
    def think(self, problem: str) -> str:
        """Generate thoughts based on role"""
        
        prompt = f"""You are {self.name}, a {self.role}.

Problem: {problem}

Based on your role and expertise, what are your thoughts on this problem?
What would you contribute to solving it?"""
        
        messages = [
            {"role": "system", "content": f"You are {self.name}, a {self.role}."},
            {"role": "user", "content": prompt}
        ]
        
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7
        )
        
        thoughts = response.choices[0].message.content
        self.memory.append({
            "problem": problem,
            "thoughts": thoughts,
            "timestamp": datetime.now()
        })
        
        return thoughts
    
    def collaborate(self, problem: str, other_thoughts: List[str]) -> str:
        """Collaborate with other agents"""
        
        thoughts_text = "\n".join([f"- {thought}" for thought in other_thoughts])
        
        prompt = f"""You are {self.name}, a {self.role}.

Problem: {problem}

Other agents' thoughts:
{thoughts_text}

How do you build upon these thoughts? What additional insights can you provide?
How can you collaborate with the other agents to solve this problem?"""
        
        messages = [
            {"role": "system", "content": f"You are {self.name}, a {self.role}."},
            {"role": "user", "content": prompt}
        ]
        
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7
        )
        
        collaboration = response.choices[0].message.content
        return collaboration

class MultiAgentSystem:
    def __init__(self):
        self.agents = {}
        self.conversation_history = []
        
    def add_agent(self, agent: CollaborativeAgent):
        """Add an agent to the system"""
        self.agents[agent.name] = agent
        
        # Add collaborators
        for other_agent in self.agents.values():
            if other_agent.name != agent.name:
                agent.add_collaborator(other_agent)
                other_agent.add_collaborator(agent)
    
    def solve_problem(self, problem: str) -> Dict[str, str]:
        """Solve problem using all agents"""
        
        print(f"Solving problem: {problem}\n")
        
        # Phase 1: Individual thoughts
        print("Phase 1: Individual Analysis")
        individual_thoughts = {}
        
        for name, agent in self.agents.items():
            thoughts = agent.think(problem)
            individual_thoughts[name] = thoughts
            print(f"\n{name} ({agent.role}):")
            print(thoughts)
        
        # Phase 2: Collaboration
        print("\n" + "="*50)
        print("Phase 2: Collaboration")
        collaboration_results = {}
        
        for name, agent in self.agents.items():
            other_thoughts = [thoughts for agent_name, thoughts in individual_thoughts.items() 
                            if agent_name != name]
            collaboration = agent.collaborate(problem, other_thoughts)
            collaboration_results[name] = collaboration
            print(f"\n{name} collaboration:")
            print(collaboration)
        
        # Phase 3: Synthesis
        print("\n" + "="*50)
        print("Phase 3: Final Synthesis")
        
        all_contributions = []
        for name, thoughts in individual_thoughts.items():
            all_contributions.append(f"{name}: {thoughts}")
        for name, collaboration in collaboration_results.items():
            all_contributions.append(f"{name} collaboration: {collaboration}")
        
        synthesis_prompt = f"""Problem: {problem}

All agent contributions:
{chr(10).join(all_contributions)}

Synthesize all these contributions into a comprehensive solution:"""
        
        messages = [
            {"role": "system", "content": "You are a synthesis agent that combines multiple perspectives."},
            {"role": "user", "content": synthesis_prompt}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3
        )
        
        final_solution = response.choices[0].message.content
        
        return {
            "individual_thoughts": individual_thoughts,
            "collaboration_results": collaboration_results,
            "final_solution": final_solution
        }

# Example usage
mas = MultiAgentSystem()

# Create specialized agents
researcher = CollaborativeAgent("Alice", "Research Analyst", "gpt-3.5-turbo")
engineer = CollaborativeAgent("Bob", "Systems Engineer", "gpt-3.5-turbo")
designer = CollaborativeAgent("Carol", "UX Designer", "gpt-3.5-turbo")

mas.add_agent(researcher)
mas.add_agent(engineer)
mas.add_agent(designer)

# Solve a complex problem
problem = "Design a sustainable smart home system that reduces energy consumption by 50%"

results = mas.solve_problem(problem)

print(f"\nFinal Solution:\n{results['final_solution']}")
```

---

## 🎯 Chapter 5: Autonomous AI Assistants

### Personal AI Assistant

```python
class PersonalAIAssistant:
    def __init__(self, name: str, model_name="gpt-3.5-turbo"):
        self.name = name
        self.model_name = model_name
        self.memory = []
        self.preferences = {}
        self.skills = {}
        
    def learn_preference(self, category: str, preference: str):
        """Learn user preferences"""
        if category not in self.preferences:
            self.preferences[category] = []
        self.preferences[category].append(preference)
    
    def add_skill(self, skill_name: str, skill_function: callable, description: str):
        """Add a skill to the assistant"""
        self.skills[skill_name] = {
            'function': skill_function,
            'description': description
        }
    
    def remember(self, interaction: Dict):
        """Store interaction in memory"""
        self.memory.append({
            **interaction,
            'timestamp': datetime.now()
        })
    
    def get_context(self, query: str) -> str:
        """Get relevant context from memory"""
        # Simple keyword matching (could be improved with embeddings)
        relevant_memories = []
        
        for memory in self.memory[-10:]:  # Last 10 interactions
            if any(word in memory.get('content', '').lower() 
                   for word in query.lower().split()):
                relevant_memories.append(memory)
        
        if relevant_memories:
            context = "Previous relevant interactions:\n"
            for memory in relevant_memories[-3:]:  # Last 3 relevant
                context += f"- {memory.get('content', '')}\n"
            return context
        return ""
    
    def respond(self, user_input: str) -> str:
        """Generate personalized response"""
        
        # Get context
        context = self.get_context(user_input)
        
        # Build personality prompt
        personality = f"""You are {self.name}, a personal AI assistant.

User preferences:
{self._format_preferences()}

Available skills:
{self._format_skills()}

{context}

Respond in a helpful, personalized way. Use the user's preferences when relevant.
If you need to use a skill, mention it clearly."""

        messages = [
            {"role": "system", "content": personality},
            {"role": "user", "content": user_input}
        ]
        
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7
        )
        
        response_text = response.choices[0].message.content
        
        # Store interaction
        self.remember({
            'user_input': user_input,
            'assistant_response': response_text,
            'content': f"{user_input} -> {response_text}"
        })
        
        return response_text
    
    def _format_preferences(self) -> str:
        """Format user preferences for prompt"""
        if not self.preferences:
            return "No specific preferences learned yet."
        
        formatted = []
        for category, prefs in self.preferences.items():
            formatted.append(f"{category}: {', '.join(prefs)}")
        return "\n".join(formatted)
    
    def _format_skills(self) -> str:
        """Format available skills for prompt"""
        if not self.skills:
            return "No specific skills available."
        
        formatted = []
        for skill_name, skill_info in self.skills.items():
            formatted.append(f"- {skill_name}: {skill_info['description']}")
        return "\n".join(formatted)

# Example usage
assistant = PersonalAIAssistant("Alex")

# Learn preferences
assistant.learn_preference("communication_style", "friendly and casual")
assistant.learn_preference("technical_level", "intermediate")
assistant.learn_preference("interests", "technology, science, programming")

# Add skills
def schedule_reminder(task, time):
    return f"Reminder set for {task} at {time}"

def search_internet(query):
    return f"Search results for '{query}': [mock results]"

assistant.add_skill("schedule_reminder", schedule_reminder, "Schedule reminders and tasks")
assistant.add_skill("search_internet", search_internet, "Search the internet for information")

# Interact with assistant
conversations = [
    "Hello! How are you today?",
    "I'm interested in learning about machine learning",
    "Can you remind me to call the dentist tomorrow at 2 PM?",
    "What's the weather like in San Francisco?"
]

for user_input in conversations:
    print(f"\nUser: {user_input}")
    response = assistant.respond(user_input)
    print(f"Alex: {response}")
```

---

## 🔄 Chapter 6: Agent Memory and Learning

### Long-term Memory System

```python
import pickle
import os
from datetime import datetime, timedelta

class AgentMemory:
    def __init__(self, memory_file: str = "agent_memory.pkl"):
        self.memory_file = memory_file
        self.short_term = []  # Recent interactions
        self.long_term = []   # Important memories
        self.load_memory()
        
    def add_memory(self, content: str, importance: float = 0.5, 
                   memory_type: str = "interaction"):
        """Add a memory with importance score"""
        
        memory = {
            'content': content,
            'importance': importance,
            'type': memory_type,
            'timestamp': datetime.now(),
            'access_count': 0
        }
        
        # Add to short-term memory
        self.short_term.append(memory)
        
        # If important enough, add to long-term memory
        if importance > 0.7:
            self.long_term.append(memory)
        
        # Limit short-term memory size
        if len(self.short_term) > 100:
            self.short_term.pop(0)
        
        self.save_memory()
    
    def retrieve_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """Retrieve relevant memories"""
        
        all_memories = self.short_term + self.long_term
        
        # Simple relevance scoring (could be improved with embeddings)
        scored_memories = []
        query_words = set(query.lower().split())
        
        for memory in all_memories:
            memory_words = set(memory['content'].lower().split())
            overlap = len(query_words.intersection(memory_words))
            relevance = overlap / len(query_words) if query_words else 0
            
            # Boost score for frequently accessed memories
            access_boost = min(memory['access_count'] * 0.1, 0.5)
            
            # Boost score for important memories
            importance_boost = memory['importance'] * 0.3
            
            final_score = relevance + access_boost + importance_boost
            scored_memories.append((final_score, memory))
        
        # Sort by score and return top memories
        scored_memories.sort(reverse=True)
        relevant_memories = [memory for score, memory in scored_memories[:limit]]
        
        # Update access count
        for memory in relevant_memories:
            memory['access_count'] += 1
        
        return relevant_memories
    
    def forget_old_memories(self, days: int = 30):
        """Remove old, unimportant memories"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        self.long_term = [
            memory for memory in self.long_term
            if memory['timestamp'] > cutoff_date or memory['importance'] > 0.8
        ]
        
        self.save_memory()
    
    def save_memory(self):
        """Save memory to file"""
        memory_data = {
            'short_term': self.short_term,
            'long_term': self.long_term
        }
        
        with open(self.memory_file, 'wb') as f:
            pickle.dump(memory_data, f)
    
    def load_memory(self):
        """Load memory from file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'rb') as f:
                    memory_data = pickle.load(f)
                    self.short_term = memory_data.get('short_term', [])
                    self.long_term = memory_data.get('long_term', [])
            except:
                self.short_term = []
                self.long_term = []

class LearningAgent:
    def __init__(self, name: str, model_name="gpt-3.5-turbo"):
        self.name = name
        self.model_name = model_name
        self.memory = AgentMemory(f"{name}_memory.pkl")
        self.learned_patterns = {}
        
    def learn_from_interaction(self, user_input: str, response: str, 
                              feedback: str = None):
        """Learn from an interaction"""
        
        # Store interaction
        self.memory.add_memory(
            f"User: {user_input} | Assistant: {response}",
            importance=0.6,
            memory_type="interaction"
        )
        
        # If feedback provided, learn from it
        if feedback:
            self.memory.add_memory(
                f"Feedback on '{user_input}': {feedback}",
                importance=0.8,
                memory_type="feedback"
            )
            
            # Extract learning patterns
            self._extract_patterns(user_input, response, feedback)
    
    def _extract_patterns(self, user_input: str, response: str, feedback: str):
        """Extract learning patterns from feedback"""
        
        # Simple pattern extraction (could be improved)
        if "good" in feedback.lower() or "great" in feedback.lower():
            pattern = {
                'input_pattern': user_input.lower(),
                'response_pattern': response,
                'feedback': 'positive',
                'timestamp': datetime.now()
            }
            
            pattern_key = f"positive_{hash(user_input) % 1000}"
            self.learned_patterns[pattern_key] = pattern
        
        elif "bad" in feedback.lower() or "wrong" in feedback.lower():
            pattern = {
                'input_pattern': user_input.lower(),
                'response_pattern': response,
                'feedback': 'negative',
                'timestamp': datetime.now()
            }
            
            pattern_key = f"negative_{hash(user_input) % 1000}"
            self.learned_patterns[pattern_key] = pattern
    
    def respond_with_learning(self, user_input: str) -> str:
        """Generate response using learned patterns"""
        
        # Retrieve relevant memories
        relevant_memories = self.memory.retrieve_memories(user_input, limit=3)
        
        # Check for learned patterns
        applicable_patterns = []
        for pattern_key, pattern in self.learned_patterns.items():
            if pattern['input_pattern'] in user_input.lower():
                applicable_patterns.append(pattern)
        
        # Build context from memories and patterns
        context = "Relevant past interactions:\n"
        for memory in relevant_memories:
            context += f"- {memory['content']}\n"
        
        if applicable_patterns:
            context += "\nLearned patterns:\n"
            for pattern in applicable_patterns:
                context += f"- {pattern['feedback']}: {pattern['response_pattern']}\n"
        
        # Generate response
        prompt = f"""You are {self.name}, a learning AI assistant.

{context}

User input: {user_input}

Respond based on your past interactions and learned patterns. 
If you have positive patterns for similar inputs, follow them.
If you have negative patterns, avoid similar responses."""

        messages = [
            {"role": "system", "content": f"You are {self.name}, a learning AI assistant."},
            {"role": "user", "content": prompt}
        ]
        
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7
        )
        
        response_text = response.choices[0].message.content
        
        # Store interaction
        self.memory.add_memory(
            f"User: {user_input} | Assistant: {response_text}",
            importance=0.6,
            memory_type="interaction"
        )
        
        return response_text

# Example usage
learning_agent = LearningAgent("Learner")

# Initial interactions
interactions = [
    ("What's the weather like?", "I don't have access to weather data.", "bad - you should say you can't provide weather info"),
    ("Tell me a joke", "Why don't scientists trust atoms? Because they make up everything!", "good - that was funny"),
    ("What's 2+2?", "2+2 equals 4", "good - correct answer"),
]

for user_input, response, feedback in interactions:
    learning_agent.learn_from_interaction(user_input, response, feedback)

# Test learning
test_input = "What's the weather like?"
response = learning_agent.respond_with_learning(test_input)
print(f"User: {test_input}")
print(f"Learner: {response}")
```

---

## 🎉 Conclusion

You now have a comprehensive understanding of:

✅ **Basic Agent Architecture** - ReAct, function calling, tool use  
✅ **Planning and Reasoning** - Chain of thought, tree of thoughts  
✅ **Multi-Tool Agents** - Wikipedia, calculations, time tools  
✅ **Multi-Agent Systems** - Collaboration, synthesis, coordination  
✅ **Autonomous Assistants** - Personalization, memory, learning  
✅ **Memory Systems** - Long-term memory, pattern learning  

### Key Takeaways:

1. **Start simple** - Begin with basic ReAct agents
2. **Use appropriate tools** - Match tools to agent capabilities
3. **Enable collaboration** - Multi-agent systems are powerful
4. **Implement memory** - Learning and adaptation are key
5. **Focus on user experience** - Personalization matters

### Next Steps:

1. **Build specialized agents** - Domain-specific expertise
2. **Implement advanced reasoning** - More sophisticated planning
3. **Create agent ecosystems** - Multiple agents working together
4. **Add learning capabilities** - Continuous improvement

**Happy agent development!** 🚀

---

*Build intelligent, autonomous AI systems!* 🎯 