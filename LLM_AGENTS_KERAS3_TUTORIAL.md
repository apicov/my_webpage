# LLM Agents Tutorial: Keras 3.0 Edition

## 🤖 Introduction to LLM Agents with Keras 3.0

LLM Agents are autonomous AI systems that can plan, reason, and execute complex tasks using language models as their "brain." This tutorial covers everything from basic agent architectures to advanced multi-agent systems using Keras 3.0.

### What are LLM Agents?

**LLM Agents** are AI systems that combine the language understanding capabilities of Large Language Models with the ability to:
- **Plan and Reason**: Break down complex problems into steps
- **Use Tools**: Interact with external systems, APIs, and databases
- **Make Decisions**: Choose the best course of action based on context
- **Learn and Adapt**: Improve performance through experience
- **Collaborate**: Work with other agents or humans

Think of them as **AI assistants that can actually do things**, not just talk about them.

### How Do LLM Agents Work?

LLM Agents follow a **sense-think-act** cycle:

1. **Sense**: Receive input (user query, environment state, etc.)
2. **Think**: Use the LLM to reason about what to do
3. **Act**: Execute actions (call tools, make decisions, etc.)
4. **Observe**: See the results of actions
5. **Repeat**: Continue until the task is complete

### Key Components of LLM Agents:

1. **Language Model (Brain)**: The reasoning and planning engine
2. **Tools (Hands)**: Functions the agent can call to interact with the world
3. **Memory (Experience)**: Storage for past interactions and knowledge
4. **Planning System (Strategy)**: How the agent breaks down complex tasks
5. **Execution Engine (Actions)**: How the agent carries out plans

### Why Keras 3.0 for LLM Agents?

Keras 3.0 provides several advantages for building LLM agents:

1. **Multi-backend Support**: Choose the best backend for your agent's needs
2. **Unified API**: Consistent interface across different model types
3. **Custom Training**: Fine-tune models for specific agent tasks
4. **Integration**: Works seamlessly with your existing TinyML and IoT projects
5. **Performance**: Optimized for modern hardware and deployment

### Types of LLM Agents:

1. **ReAct Agents**: Reason and act in cycles
2. **Function-Calling Agents**: Use structured function calls
3. **Planning Agents**: Create detailed plans before acting
4. **Multi-Agent Systems**: Multiple agents working together
5. **Autonomous Agents**: Self-directed agents with goals

### Prerequisites:

Before starting this tutorial, you should be familiar with:
- LLM fundamentals (from the previous tutorial)
- Basic Python programming
- Understanding of neural networks
- Your existing Flask and React setup

**What you'll learn:**
- Agent architectures and frameworks with Keras 3.0
- Planning and reasoning systems
- Tool use and function calling
- Multi-agent coordination
- Building autonomous AI assistants

---

## 🏗️ Chapter 1: Basic Agent Architecture with Keras 3.0

### Understanding Agent Architectures

Before we dive into code, let's understand the fundamental concepts behind LLM agents:

#### **The ReAct Pattern**

**ReAct** stands for **Reasoning + Acting**. It's a pattern where the agent:
1. **Reasons** about what to do next
2. **Acts** by calling a tool or function
3. **Observes** the result
4. **Repeats** until the task is complete

This pattern mimics how humans solve problems: we think, we do something, we see what happens, and we adjust our approach.

#### **Why ReAct Works**

The ReAct pattern is powerful because it:
- **Breaks down complex problems** into manageable steps
- **Allows for course correction** based on results
- **Makes reasoning explicit** and debuggable
- **Enables tool use** in a structured way

#### **Components of a ReAct Agent**

1. **Reasoning Engine**: The LLM that generates thoughts and plans
2. **Tool Registry**: Available functions the agent can call
3. **Action Parser**: Extracts actions from the LLM's reasoning
4. **Execution Engine**: Runs the tools and returns results
5. **Memory**: Stores conversation history and results

### Simple ReAct Agent

```python
import keras
import json
import re
from typing import List, Dict, Any
from datetime import datetime

class ReActAgent:
    """
    ReAct Agent - A simple implementation of the Reasoning + Acting pattern
    
    This agent follows the ReAct pattern:
    1. Receives a query
    2. Reasons about what tools to use
    3. Executes actions using available tools
    4. Observes results and continues reasoning
    
    The agent maintains:
    - A registry of available tools
    - Conversation history for context
    - The ability to parse and execute actions
    
    Parameters:
    - model_name: Name of the language model to use for reasoning
    """
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        
        # Store conversation history for context
        # This helps the agent remember previous interactions
        self.conversation_history = []
        
        # Registry of available tools
        # Each tool has a name, function, and description
        self.tools = {}
        
    def add_tool(self, name: str, function: callable, description: str):
        """
        Add a tool that the agent can use
        
        Tools are functions that the agent can call to interact with the world.
        Examples include: searching the web, making calculations, calling APIs,
        reading files, etc.
        
        Parameters:
        - name: The name the agent will use to call this tool
        - function: The actual function to execute
        - description: A description of what the tool does (used in prompts)
        """
        self.tools[name] = {
            'function': function,
            'description': description
        }
    
    def think(self, query: str) -> str:
        """
        Generate reasoning and action plan using the ReAct pattern
        
        This is the core reasoning function of the agent. It:
        1. Constructs a prompt that includes available tools
        2. Asks the LLM to reason step-by-step
        3. Returns the reasoning in a structured format
        
        The ReAct format encourages the LLM to:
        - Think explicitly about each step
        - Choose appropriate tools
        - Provide clear reasoning for decisions
        - Continue until the problem is solved
        
        Parameters:
        - query: The user's question or request
        
        Returns:
        - str: The agent's reasoning and action plan
        """
        
        # Build system prompt that defines the agent's capabilities
        # This prompt tells the LLM what tools are available and how to use them
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

        # Prepare messages for the LLM
        # The system message sets the context, the user message is the query
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # For demonstration, we'll use a mock response
        # In practice, you'd use OpenAI API or a local model
        # This could be replaced with a Keras 3.0 model for local inference
        response = self._mock_llm_call(messages)
        
        return response
    
    def execute(self, query: str) -> str:
        """
        Execute a query using the ReAct pattern
        
        This is the main execution function that:
        1. Gets the agent's reasoning and action plan
        2. Parses the reasoning to extract specific actions
        3. Executes each action using the appropriate tool
        4. Returns the complete reasoning process
        
        The execution follows the ReAct cycle:
        - Reason: Generate thoughts and action plan
        - Act: Execute tools based on the plan
        - Observe: See the results of actions
        
        Parameters:
        - query: The user's question or request
        
        Returns:
        - str: The complete reasoning and execution trace
        """
        
        # Step 1: Generate reasoning and action plan
        # This calls the LLM to think about how to solve the problem
        reasoning = self.think(query)
        print(f"Agent reasoning:\n{reasoning}")
        
        # Step 2: Parse the reasoning to extract specific actions
        # The LLM's output contains structured action commands that we need to extract
        actions = self._parse_actions(reasoning)
        
        # Step 3: Execute each action using the appropriate tool
        # This is where the agent actually "does things" in the world
        for action in actions:
            if action['action'] in self.tools:
                try:
                    # Call the tool function with the provided input
                    result = self.tools[action['action']]['function'](action['input'])
                    print(f"Tool {action['action']} returned: {result}")
                except Exception as e:
                    # Handle errors gracefully
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
    
    def _mock_llm_call(self, messages):
        """Mock LLM call for demonstration"""
        # In practice, replace with actual OpenAI API call
        return "Thought: I need to search for information\nAction: search\nAction Input: the query\nObservation: search results\nThought: I now know the answer\nFinal Answer: Based on the search results..."

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

### 🔍 **Understanding the ReAct Agent**

Now that we've implemented a ReAct agent, let's understand what makes it powerful and how it works:

#### **1. The ReAct Pattern in Action**

The ReAct pattern follows this cycle:

```
Query → Think → Act → Observe → Think → Act → ... → Final Answer
```

**Example Flow:**
1. **Query**: "What is the population of Tokyo and what is 2^10?"
2. **Think**: "I need to search for Tokyo's population and calculate 2^10"
3. **Act**: Call search tool with "Tokyo population"
4. **Observe**: Get search results
5. **Think**: "Now I need to calculate 2^10"
6. **Act**: Call calculate tool with "2**10"
7. **Observe**: Get result "1024"
8. **Think**: "I have both pieces of information"
9. **Final Answer**: Combine both results

#### **2. Key Components Explained**

**Tool Registry (`self.tools`)**:
- Stores available functions the agent can call
- Each tool has a name, function, and description
- The description is used in prompts to tell the LLM what the tool does

**Action Parser (`_parse_actions`)**:
- Extracts structured actions from the LLM's natural language reasoning
- Uses regex to find "Action:" and "Action Input:" patterns
- Converts text reasoning into executable commands

**Execution Engine**:
- Safely calls tool functions with provided inputs
- Handles errors gracefully
- Provides feedback on tool execution

#### **3. Why ReAct is Effective**

**Advantages:**
- **Explicit Reasoning**: You can see exactly how the agent thinks
- **Debuggable**: Easy to understand where things go wrong
- **Flexible**: Can handle complex multi-step problems
- **Extensible**: Easy to add new tools

**Challenges:**
- **Prompt Engineering**: Requires careful prompt design
- **Parsing**: Need robust parsing of LLM outputs
- **Error Handling**: Must handle tool failures gracefully
- **Context Management**: Need to manage conversation history

#### **4. Real-World Applications**

ReAct agents are used for:
- **Customer Support**: Answering complex customer queries
- **Data Analysis**: Breaking down analysis tasks into steps
- **Content Creation**: Researching and writing content
- **Automation**: Automating complex workflows
- **Education**: Tutoring and problem-solving assistance

#### **5. Integration with Keras 3.0**

To use this with Keras 3.0 models:
1. Replace `_mock_llm_call` with a Keras 3.0 model
2. Use the model for text generation
3. Fine-tune the model for better reasoning
4. Optimize for your specific use case

### Function Calling Agent with Keras 3.0
```

### Function Calling Agent with Keras 3.0

#### **Understanding Function Calling**

**Function Calling** is a more structured approach to tool use where:
- Functions are defined with explicit schemas (JSON Schema)
- The LLM chooses which function to call based on the query
- Function parameters are validated before execution
- The system is more reliable and type-safe

**Advantages over ReAct:**
- **Structured**: Functions have defined schemas
- **Reliable**: Better error handling and validation
- **Type-safe**: Parameters are validated
- **Efficient**: Direct function calls without parsing

**When to Use Function Calling:**
- When you have well-defined, structured tools
- When you need reliable parameter validation
- When you want type safety
- When you're building production systems

```python
class FunctionCallingAgent:
    """
    Function Calling Agent - A structured approach to tool use
    
    This agent uses function calling instead of the ReAct pattern. It:
    1. Defines functions with explicit schemas (JSON Schema)
    2. Lets the LLM choose which function to call
    3. Validates parameters before execution
    4. Provides structured, reliable tool use
    
    The key difference from ReAct is that this approach is more structured
    and reliable, but less flexible for complex reasoning chains.
    
    Parameters:
    - model_name: Name of the language model to use
    """
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        
        # Registry of available functions with their schemas
        # Each function has a name, description, parameter schema, and implementation
        self.functions = []
        
    def add_function(self, name: str, description: str, parameters: Dict, function: callable):
        """
        Add a function that can be called by the agent
        
        Functions are defined with JSON Schema for parameters, which provides:
        - Type validation
        - Parameter constraints
        - Clear documentation
        - Structured calling
        
        Parameters:
        - name: Function name (must be unique)
        - description: What the function does
        - parameters: JSON Schema defining the function parameters
        - function: The actual function to execute
        """
        self.functions.append({
            "name": name,
            "description": description,
            "parameters": parameters,
            "function": function
        })
    
    def execute(self, query: str) -> str:
        """
        Execute query using function calling
        
        This method implements the function calling pattern:
        1. Prepare function definitions for the LLM
        2. Ask the LLM if it wants to call a function
        3. If yes, handle the function call
        4. If no, return the direct response
        
        The function calling pattern is more structured than ReAct:
        - Functions have explicit schemas
        - Parameters are validated
        - The LLM makes a single decision about function calling
        - Less complex reasoning, more reliable execution
        
        Parameters:
        - query: The user's question or request
        
        Returns:
        - str: The agent's response or function call result
        """
        
        # Step 1: Prepare function definitions for the LLM
        # Convert our function registry into the format expected by the LLM
        function_definitions = []
        for func in self.functions:
            function_definitions.append({
                "name": func["name"],
                "description": func["description"],
                "parameters": func["parameters"]
            })
        
        # Step 2: First call to determine if function should be called
        # The LLM decides whether to call a function or respond directly
        messages = [{"role": "user", "content": query}]
        
        # Mock response for demonstration
        # In practice, this would be an actual LLM call with function definitions
        response_message = self._mock_function_call(messages, function_definitions)
        
        # Step 3: Check if function was called
        # If the LLM decided to call a function, handle it
        if response_message.get("function_call"):
            return self._handle_function_call(response_message, messages)
        
        # Step 4: Return direct response if no function was called
        return response_message.get("content", "No response generated")
    
    def _handle_function_call(self, response_message, messages):
        """Handle function call and continue conversation"""
        
        function_name = response_message["function_call"]["name"]
        function_args = json.loads(response_message["function_call"]["arguments"])
        
        # Execute function
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
                    
                    final_response = self._mock_llm_call(messages)
                    
                    return final_response
                    
                except Exception as e:
                    return f"Error executing function {function_name}: {e}"
        
        return "Tool not found"
    
    def _mock_function_call(self, messages, function_definitions):
        """Mock function calling for demonstration"""
        # In practice, replace with actual OpenAI API call
        return {
            "function_call": {
                "name": "get_weather",
                "arguments": '{"location": "New York"}'
            }
        }
    
    def _mock_llm_call(self, messages):
        """Mock LLM call for demonstration"""
        return "The weather in New York is sunny and 75°F."

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

### 🔍 **Understanding Function Calling Agents**

Now let's understand the key differences and advantages of function calling over ReAct:

#### **1. Function Calling vs ReAct Comparison**

| Aspect | ReAct | Function Calling |
|--------|-------|------------------|
| **Structure** | Free-form reasoning | Structured schemas |
| **Reliability** | Depends on parsing | Built-in validation |
| **Complexity** | Can handle complex chains | Single function calls |
| **Debugging** | Manual parsing debugging | Schema validation errors |
| **Performance** | Multiple LLM calls | Fewer LLM calls |
| **Use Case** | Complex reasoning | Simple tool use |

#### **2. JSON Schema Benefits**

**JSON Schema** provides several advantages:

1. **Type Safety**: Parameters are validated before execution
2. **Documentation**: Schemas serve as documentation
3. **Constraints**: Can specify required fields, data types, ranges
4. **Error Handling**: Clear error messages for invalid parameters
5. **IDE Support**: Better autocomplete and validation

**Example Schema:**
```json
{
  "type": "object",
  "properties": {
    "location": {
      "type": "string",
      "description": "City name"
    },
    "temperature": {
      "type": "number",
      "minimum": -100,
      "maximum": 150
    }
  },
  "required": ["location"]
}
```

#### **3. Function Calling Flow**

The function calling process follows this pattern:

```
Query → LLM Decision → Function Call → Result → Final Response
```

**Detailed Flow:**
1. **Query**: User asks a question
2. **LLM Decision**: LLM decides if it needs to call a function
3. **Function Call**: If yes, LLM specifies which function and parameters
4. **Execution**: System executes the function with validated parameters
5. **Result**: Function result is returned to LLM
6. **Final Response**: LLM provides final answer incorporating function result

#### **4. When to Use Function Calling**

**Best for:**
- **Simple tool use**: Getting weather, making calculations
- **Production systems**: Where reliability is critical
- **Structured APIs**: When you have well-defined interfaces
- **Type safety**: When parameter validation is important

**Not ideal for:**
- **Complex reasoning**: Multi-step problem solving
- **Dynamic workflows**: Where the plan changes based on results
- **Creative tasks**: Where flexibility is more important than structure

#### **5. Integration with Keras 3.0**

To integrate with Keras 3.0:
1. Replace mock LLM calls with Keras 3.0 models
2. Use the model for function calling decisions
3. Fine-tune for better function selection
4. Optimize for your specific function set

---

## 🧠 Chapter 2: Planning and Reasoning Agents
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
        
        response = self._mock_llm_call(messages)
        
        # Store in memory
        self.memory.append({
            "problem": problem,
            "reasoning": response,
            "timestamp": datetime.now()
        })
        
        return response
    
    def get_similar_problems(self, problem: str) -> List[Dict]:
        """Find similar problems from memory"""
        # Simple similarity check (could be improved with embeddings)
        similar = []
        for memory_item in self.memory:
            if any(word in memory_item["problem"].lower() 
                   for word in problem.lower().split()):
                similar.append(memory_item)
        return similar
    
    def _mock_llm_call(self, messages):
        """Mock LLM call for demonstration"""
        return """Let me solve this step by step:

1) I need to find the total distance traveled: 120 km + 180 km = 300 km
2) I need to find the total time taken: 2 hours + 3 hours = 5 hours
3) Average speed = total distance / total time = 300 km / 5 hours = 60 km/h

The average speed for the entire journey is 60 km/h."""

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
        
        response = self._mock_llm_call(messages)
        
        # Parse thoughts from response
        thoughts_text = response
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
        
        response = self._mock_llm_call(messages)
        
        try:
            rating = float(response.strip())
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
    
    def _mock_llm_call(self, messages):
        """Mock LLM call for demonstration"""
        if "Rate this thought" in messages[-1]["content"]:
            return "8"
        elif "Generate 3 different approaches" in messages[-1]["content"]:
            return """Thought 1: Implement community recycling programs
Thought 2: Create educational campaigns about plastic waste
Thought 3: Develop biodegradable alternatives to plastic"""
        else:
            return "Mock response"

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
        response_message = self._mock_function_call(messages, function_definitions)
        
        # Handle function calls
        if response_message.get("function_call"):
            return self._handle_function_call(response_message, messages)
        
        return response_message.get("content", "No response generated")
    
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
                
                final_response = self._mock_llm_call(messages)
                
                return final_response
                
            except Exception as e:
                return f"Error executing {function_name}: {e}"
        
        return "Tool not found"
    
    def _mock_function_call(self, messages, function_definitions):
        """Mock function calling for demonstration"""
        # In practice, replace with actual OpenAI API call
        return {
            "function_call": {
                "name": "search_wikipedia",
                "arguments": '{"query": "artificial intelligence"}'
            }
        }
    
    def _mock_llm_call(self, messages):
        """Mock LLM call for demonstration"""
        return "Based on the Wikipedia search, artificial intelligence is a field of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence."

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
            k: v for k, v in __import__('math').__dict__.items() 
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
        
        response = self._mock_llm_call(messages)
        
        thoughts = response
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
        
        response = self._mock_llm_call(messages)
        
        collaboration = response
        return collaboration
    
    def _mock_llm_call(self, messages):
        """Mock LLM call for demonstration"""
        role = self.role.lower()
        if "research" in role:
            return "From a research perspective, we should gather data on current energy consumption patterns and analyze trends."
        elif "engineer" in role:
            return "As an engineer, I can design energy-efficient systems and optimize existing infrastructure."
        elif "designer" in role:
            return "From a design perspective, we should focus on user experience and intuitive interfaces for energy management."
        else:
            return "I can contribute my expertise to solve this problem."

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
        
        final_response = self._mock_llm_call(messages)
        
        return {
            "individual_thoughts": individual_thoughts,
            "collaboration_results": collaboration_results,
            "final_solution": final_response
        }
    
    def _mock_llm_call(self, messages):
        """Mock LLM call for demonstration"""
        return """Based on all contributions, here's a comprehensive solution:

1. Research Phase: Gather data on current energy consumption patterns
2. Engineering Phase: Design energy-efficient systems and optimize infrastructure
3. Design Phase: Create intuitive user interfaces for energy management
4. Integration: Combine all approaches into a unified smart home system

This multi-disciplinary approach ensures both technical feasibility and user adoption."""

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
        
        response = self._mock_llm_call(messages)
        
        # Store interaction
        self.remember({
            'user_input': user_input,
            'assistant_response': response,
            'content': f"{user_input} -> {response}"
        })
        
        return response
    
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
    
    def _mock_llm_call(self, messages):
        """Mock LLM call for demonstration"""
        return "Hello! I'm your personal AI assistant. I can help you with various tasks and remember your preferences. How can I assist you today?"

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
        
        response = self._mock_llm_call(messages)
        
        # Store interaction
        self.memory.add_memory(
            f"User: {user_input} | Assistant: {response}",
            importance=0.6,
            memory_type="interaction"
        )
        
        return response
    
    def _mock_llm_call(self, messages):
        """Mock LLM call for demonstration"""
        return "Based on my learning, I'll provide a helpful response that builds on our previous interactions."

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

You now have a comprehensive understanding of LLM Agents with Keras 3.0:

✅ **Basic Agent Architecture** - ReAct, function calling, tool use  
✅ **Planning and Reasoning** - Chain of thought, tree of thoughts  
✅ **Multi-Tool Agents** - Wikipedia, calculations, time tools  
✅ **Multi-Agent Systems** - Collaboration, synthesis, coordination  
✅ **Autonomous Assistants** - Personalization, memory, learning  
✅ **Memory Systems** - Long-term memory, pattern learning  

### Key Advantages of Keras 3.0 for LLM Agents:

1. **Multi-backend support** - TensorFlow, PyTorch, JAX
2. **Unified API** - Consistent interface across backends
3. **Better performance** - Optimized for modern hardware
4. **Easier deployment** - Simplified model serving
5. **Integration** - Works seamlessly with your TinyML projects

### Next Steps:

1. **Build specialized agents** - Domain-specific expertise
2. **Implement advanced reasoning** - More sophisticated planning
3. **Create agent ecosystems** - Multiple agents working together
4. **Add learning capabilities** - Continuous improvement

**Happy agent development with Keras 3.0!** 🚀

---

*Build intelligent, autonomous AI systems with Keras 3.0!* 🎯 