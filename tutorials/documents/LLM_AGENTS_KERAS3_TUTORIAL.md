# LLM Agents Tutorial: Mastering Autonomous AI Systems from First Principles

## 📚 Welcome to the Complete LLM Agents Learning Journey!

This comprehensive tutorial transforms you from someone who uses language models into an **LLM agents expert** who can build sophisticated autonomous AI systems. You'll master the theoretical foundations, implement agent architectures from scratch, understand tool usage patterns, and create production-ready multi-agent systems integrated with your actual chat platform.

**What Makes This Tutorial Unique:**
- **Complete Self-Contained Learning**: From agent theory to production autonomous systems
- **Build From Scratch**: Implement every agent pattern and tool interface manually
- **Theory + Practice**: Deep understanding of agent reasoning combined with hands-on coding
- **Real Integration**: Apply advanced agent techniques to your actual working platform
- **Cutting-Edge Architectures**: Latest advances in ReAct, planning, and multi-agent systems

### **The AI Agents Revolution: Understanding Autonomous Intelligence**

LLM Agents represent the next evolutionary step beyond language models. While LLMs can understand and generate text, agents can **reason, plan, and act** in the world. They transform static AI assistants into dynamic, goal-oriented systems that can operate autonomously to achieve complex objectives.

**Historical Context:**
- **2022**: ReAct pattern introduces reasoning and acting in language models
- **2023**: AutoGPT demonstrates autonomous task execution capabilities
- **2023**: LangChain enables tool-using agents at scale
- **2024**: Multi-agent systems show emergent collaborative intelligence
- **Today**: Building foundations for artificial general intelligence through agents

**Why Agent Mastery Matters:**
- **Beyond Conversation**: Move from reactive chat to proactive assistance
- **Autonomous Capabilities**: Build systems that operate independently
- **Complex Problem Solving**: Handle multi-step reasoning and execution
- **Real-World Impact**: Create AI that interacts with external systems
- **Future-Proof Skills**: Foundation for AGI and advanced AI systems

---

## 🎯 Complete Learning Objectives

### **Chapter 1: Foundations of AI Agents**
**Learning Goals:**
- Understand the mathematical foundations of agent behavior
- Master the difference between reactive and deliberative agents
- Learn agent architectures and reasoning patterns
- Grasp the theoretical framework of autonomous systems

**What You'll Be Able to Do:**
- Design agent architectures for specific problem domains
- Implement basic agent reasoning loops from scratch
- Understand agent evaluation metrics and performance measures
- Debug agent behavior and improve decision-making processes

### **Chapter 2: Tool Usage and Interface Patterns**
**Learning Goals:**
- Master different tool interface formats (JSON, functions, APIs)
- Understand tool discovery and dynamic binding mechanisms
- Learn error handling and recovery in tool usage
- Implement secure and robust tool execution systems

**What You'll Be Able to Do:**
- Design flexible tool interfaces for various systems
- Implement secure tool execution with proper error handling
- Create dynamic tool discovery and binding systems
- Build tool usage monitoring and optimization systems

### **Chapter 3: Advanced Agent Architectures**
**Learning Goals:**
- Master ReAct (Reasoning and Acting) pattern implementation
- Understand planning agents and goal decomposition
- Learn memory systems and long-term context management
- Implement learning and adaptation mechanisms

**What You'll Be Able to Do:**
- Build sophisticated reasoning agents that plan and execute
- Implement memory systems for complex task management
- Create adaptive agents that improve over time
- Design agent architectures for specific domain requirements

### **Chapter 4: Multi-Agent Systems**
**Learning Goals:**
- Understand agent communication protocols and coordination
- Master distributed problem solving and task allocation
- Learn emergent behavior in multi-agent systems
- Implement conflict resolution and consensus mechanisms

**What You'll Be Able to Do:**
- Design multi-agent architectures for complex systems
- Implement agent communication and coordination protocols
- Create distributed problem-solving systems
- Build fault-tolerant multi-agent networks

### **Chapter 5: Production Agent Systems**
**Learning Goals:**
- Master agent deployment and monitoring in production
- Understand scalability and performance optimization
- Learn safety, security, and alignment considerations
- Implement agent system management and orchestration

**What You'll Be Able to Do:**
- Deploy autonomous agent systems at scale
- Monitor and optimize agent performance in production
- Implement safety measures and alignment techniques
- Build complete agent platforms with management interfaces

---

## 🧠 Chapter 0: Mathematical Foundations of Agent Systems

Before building agents, we need to understand the mathematical and theoretical foundations that make autonomous behavior possible. This groundwork will make everything else crystal clear.

### Understanding Agent Behavior Mathematically

**An Agent as a Mathematical Function:**

At its core, an agent is a function that maps observations to actions over time:

```
Agent: (Observation × Memory) → (Action × Updated_Memory)
```

More formally, an agent can be represented as:
```
πθ(at | ot, mt) = P(action_at_time_t | observation_at_time_t, memory_at_time_t)
```

Where θ represents the agent's parameters (learned through training or hardcoded rules).

**The Agent-Environment Loop:**

```python
# fundamental_agent_theory.py - Mathematical foundations of agents
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

class Environment:
    """
    Abstract environment that agents interact with.
    
    This represents the mathematical concept of an environment
    in agent theory - anything the agent can observe and act upon.
    """
    
    def __init__(self):
        self.state = None
        self.history = []
    
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """
        Execute action and return (observation, reward, done, info).
        
        This is the fundamental environment interface:
        - observation: What the agent can perceive
        - reward: Feedback signal for the action
        - done: Whether the episode is complete
        - info: Additional environment information
        """
        raise NotImplementedError
    
    def reset(self) -> Any:
        """Reset environment and return initial observation."""
        raise NotImplementedError

class Agent(ABC):
    """
    Abstract agent following the mathematical agent definition.
    
    An agent perceives environment state and takes actions
    to maximize some objective function over time.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.memory = {}
        self.action_history = []
        self.observation_history = []
    
    @abstractmethod
    def perceive(self, observation: Any) -> None:
        """Process observation and update internal state."""
        pass
    
    @abstractmethod
    def decide(self) -> Any:
        """Decide on next action based on current state."""
        pass
    
    @abstractmethod
    def act(self, action: Any) -> Any:
        """Execute action in environment."""
        pass
    
    def agent_loop(self, environment: Environment, max_steps: int = 1000):
        """
        Execute the fundamental agent-environment interaction loop.
        
        This implements the core mathematical relationship:
        observation → reasoning → action → environment response → repeat
        """
        
        observation = environment.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Perceive current state
            self.perceive(observation)
            
            # Decide on action
            action = self.decide()
            
            # Act in environment
            observation, reward, done, info = environment.step(action)
            total_reward += reward
            
            # Update history
            self.action_history.append(action)
            self.observation_history.append(observation)
            
            if done:
                break
        
        return total_reward, step + 1

def demonstrate_agent_mathematics():
    """
    Demonstrate the mathematical foundations of agent behavior.
    """
    
    print("🧮 Mathematical Foundations of AI Agents")
    print("=" * 50)
    
    # Define different types of agent decision functions
    
    def reactive_agent(observation):
        """
        Reactive agent: action = f(current_observation)
        No memory, purely stimulus-response
        """
        # Simple rule-based response
        if "help" in str(observation).lower():
            return "provide_help"
        elif "analyze" in str(observation).lower():
            return "perform_analysis"
        else:
            return "default_response"
    
    def deliberative_agent(observation, memory):
        """
        Deliberative agent: action = f(observation, memory, goals)
        Uses internal state and planning
        """
        # Update memory with observation
        memory['observations'] = memory.get('observations', [])
        memory['observations'].append(observation)
        
        # Deliberate based on history and goals
        if len(memory['observations']) > 3:
            # If we've seen multiple observations, plan multi-step response
            return "complex_planned_action"
        else:
            return "simple_response"
    
    def learning_agent(observation, memory, experience):
        """
        Learning agent: action = f(observation, memory, learned_policy)
        Adapts behavior based on past outcomes
        """
        # Update memory
        memory['total_interactions'] = memory.get('total_interactions', 0) + 1
        
        # Learn from experience (simplified)
        success_rate = experience.get('success_rate', 0.5)
        
        if success_rate > 0.7:
            return "confident_action"
        elif success_rate < 0.3:
            return "cautious_action"
        else:
            return "exploratory_action"
    
    # Demonstrate different agent types
    sample_observation = "User needs help analyzing data"
    sample_memory = {}
    sample_experience = {'success_rate': 0.8}
    
    print("Agent Decision Comparison:")
    print(f"Observation: '{sample_observation}'")
    print()
    
    reactive_action = reactive_agent(sample_observation)
    print(f"Reactive Agent:     {reactive_action}")
    print("  - Decision based only on current observation")
    print("  - No memory or learning")
    
    deliberative_action = deliberative_agent(sample_observation, sample_memory.copy())
    print(f"Deliberative Agent: {deliberative_action}")
    print("  - Decision based on observation + memory")
    print("  - Can plan multi-step actions")
    
    learning_action = learning_agent(sample_observation, sample_memory.copy(), sample_experience)
    print(f"Learning Agent:     {learning_action}")
    print("  - Decision based on observation + memory + experience")
    print("  - Adapts behavior based on past outcomes")
    
    print(f"\n🔍 Key Insight: More sophisticated agents consider more information")
    print(f"  Reactive: O(1) - constant time decisions")
    print(f"  Deliberative: O(n) - proportional to memory size") 
    print(f"  Learning: O(n + m) - memory + experience")

demonstrate_agent_mathematics()
```

### The Information Theory of Agent Decision Making

**Decision Making as Information Processing:**

Agent decisions can be understood through information theory:

```python
def analyze_agent_information_processing():
    """
    Analyze agent decision-making from an information theory perspective.
    
    This helps us understand why certain agent architectures work better
    than others for different types of problems.
    """
    
    import math
    
    print("\n📊 Information Theory of Agent Decisions")
    print("=" * 50)
    
    # Simulate different agent information processing capabilities
    
    def calculate_decision_entropy(possible_actions, action_probabilities):
        """Calculate entropy of agent's decision distribution."""
        entropy = 0
        for prob in action_probabilities:
            if prob > 0:
                entropy -= prob * math.log2(prob)
        return entropy
    
    # Example scenarios
    scenarios = {
        "Simple Task": {
            "actions": ["yes", "no"],
            "reactive_probs": [0.9, 0.1],        # High confidence, low entropy
            "deliberative_probs": [0.7, 0.3],    # Moderate confidence
            "learning_probs": [0.95, 0.05]       # Very high confidence after learning
        },
        "Complex Task": {
            "actions": ["plan_a", "plan_b", "plan_c", "wait", "ask_for_help"],
            "reactive_probs": [0.2, 0.2, 0.2, 0.2, 0.2],    # High entropy, confused
            "deliberative_probs": [0.5, 0.3, 0.1, 0.05, 0.05], # Lower entropy, more confident
            "learning_probs": [0.8, 0.1, 0.05, 0.03, 0.02]     # Lowest entropy, learned optimal
        }
    }
    
    for scenario_name, scenario in scenarios.items():
        print(f"\n📋 Scenario: {scenario_name}")
        actions = scenario["actions"]
        
        for agent_type in ["reactive", "deliberative", "learning"]:
            probs = scenario[f"{agent_type}_probs"]
            entropy = calculate_decision_entropy(actions, probs)
            
            # Calculate information content (surprise)
            max_entropy = math.log2(len(actions))  # Maximum possible entropy
            confidence = 1 - (entropy / max_entropy)  # Confidence as inverse of normalized entropy
            
            print(f"  {agent_type.capitalize():<12}: entropy={entropy:.2f}, confidence={confidence:.2f}")
    
    print(f"\n💡 Key Insights:")
    print(f"  - Lower entropy = more confident decisions")
    print(f"  - Learning agents develop lower entropy over time")
    print(f"  - Complex tasks require deliberative processing")
    print(f"  - Information processing capacity limits agent performance")

analyze_agent_information_processing()
```

### The Cognitive Architecture of Reasoning

**Understanding Different Reasoning Patterns:**

LLM agents use different reasoning patterns depending on the task:

```python
class ReasoningPattern:
    """
    Base class for different reasoning patterns used by LLM agents.
    
    Each pattern represents a different way of processing information
    and making decisions.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.steps = []
    
    def reason(self, problem: str) -> List[str]:
        """Apply reasoning pattern to a problem."""
        raise NotImplementedError

class ChainOfThoughtReasoning(ReasoningPattern):
    """
    Chain-of-Thought: Step-by-step reasoning through problems.
    
    This pattern breaks complex problems into sequential steps,
    making the reasoning process explicit and traceable.
    """
    
    def __init__(self):
        super().__init__("Chain-of-Thought")
    
    def reason(self, problem: str) -> List[str]:
        """
        Apply chain-of-thought reasoning.
        
        Steps:
        1. Understand the problem
        2. Break into sub-problems  
        3. Solve each sub-problem
        4. Combine solutions
        """
        
        reasoning_steps = [
            f"Problem: {problem}",
            "Step 1: Let me understand what's being asked...",
            "Step 2: I need to break this into smaller parts...", 
            "Step 3: For each part, I'll...",
            "Step 4: Combining these solutions..."
        ]
        
        return reasoning_steps

class ReActReasoning(ReasoningPattern):
    """
    ReAct: Reasoning + Acting pattern.
    
    This pattern interleaves reasoning (thought) with actions (tool use),
    allowing agents to gather information and act on it iteratively.
    """
    
    def __init__(self):
        super().__init__("ReAct")
        self.available_tools = ["search", "calculate", "analyze", "execute"]
    
    def reason(self, problem: str) -> List[str]:
        """
        Apply ReAct reasoning pattern.
        
        Alternates between:
        - Thought: reasoning about the problem
        - Action: using tools to gather information or execute
        - Observation: processing results of actions
        """
        
        reasoning_steps = [
            f"Problem: {problem}",
            "Thought: I need to understand this problem better.",
            "Action: search(problem context)",
            "Observation: Found relevant information about...",
            "Thought: Based on this information, I should...", 
            "Action: calculate(specific computation)",
            "Observation: The calculation shows...",
            "Thought: Now I can provide a complete solution."
        ]
        
        return reasoning_steps

class PlanExecuteReasoning(ReasoningPattern):
    """
    Plan-Execute: High-level planning followed by detailed execution.
    
    This pattern separates strategic planning from tactical execution,
    allowing for better handling of complex, multi-step tasks.
    """
    
    def __init__(self):
        super().__init__("Plan-Execute")
    
    def reason(self, problem: str) -> List[str]:
        """
        Apply plan-execute reasoning.
        
        Steps:
        1. Create high-level plan
        2. Detail each step
        3. Execute steps sequentially
        4. Monitor and adapt
        """
        
        reasoning_steps = [
            f"Problem: {problem}",
            "Planning Phase:",
            "  - Goal: Define what success looks like",
            "  - Strategy: High-level approach",
            "  - Steps: Detailed action sequence",
            "Execution Phase:",
            "  - Step 1: Execute first action",
            "  - Step 2: Execute second action",
            "  - Monitor: Check progress against plan",
            "  - Adapt: Modify plan if needed"
        ]
        
        return reasoning_steps

def demonstrate_reasoning_patterns():
    """
    Demonstrate different reasoning patterns and their applications.
    """
    
    print("\n🧠 Cognitive Reasoning Patterns")
    print("=" * 50)
    
    # Create reasoning pattern instances
    patterns = [
        ChainOfThoughtReasoning(),
        ReActReasoning(), 
        PlanExecuteReasoning()
    ]
    
    # Test problem
    test_problem = "How can I optimize my portfolio to showcase advanced AI skills?"
    
    print(f"Test Problem: {test_problem}")
    print()
    
    for pattern in patterns:
        print(f"🎯 {pattern.name} Reasoning:")
        steps = pattern.reason(test_problem)
        
        for i, step in enumerate(steps):
            if step.startswith("  "):
                print(f"    {step}")
            else:
                print(f"  {step}")
        print()
    
    print("🔍 When to Use Each Pattern:")
    print("  Chain-of-Thought: Mathematical problems, logical reasoning")
    print("  ReAct: Information gathering, tool-using tasks")
    print("  Plan-Execute: Complex projects, multi-step objectives")
    print()
    print("💡 Advanced agents can switch between patterns based on task type")

demonstrate_reasoning_patterns()
```

---

## 🔍 Chapter 1: Tool Usage and Interface Patterns

Tool usage is fundamental to agent capabilities. Agents need to interact with external systems, APIs, and functions to accomplish real-world tasks. Understanding different tool interface patterns is crucial for building practical agents.

### Understanding Tool Interface Formats

**The Tool Interface Spectrum:**

```python
# tool_interfaces.py - Complete guide to agent tool usage
import json
import inspect
from typing import Any, Dict, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum

class ToolFormat(Enum):
    """Different formats for tool definitions and usage."""
    JSON_SCHEMA = "json_schema"           # OpenAI-style function calling
    PYTHON_FUNCTION = "python_function"   # Direct function calls
    REST_API = "rest_api"                 # HTTP API endpoints
    COMMAND_LINE = "command_line"         # Shell commands
    DATABASE = "database"                 # Database queries

@dataclass
class ToolResult:
    """Standardized result from tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Optional[Dict] = None

class Tool:
    """
    Base tool class supporting multiple interface formats.
    
    This provides a unified interface for tools regardless of their
    underlying implementation (JSON, functions, APIs, etc.).
    """
    
    def __init__(self, name: str, description: str, format_type: ToolFormat):
        self.name = name
        self.description = description
        self.format_type = format_type
        self.usage_count = 0
        self.success_count = 0
    
    def execute(self, *args, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        self.usage_count += 1
        try:
            result = self._execute_impl(*args, **kwargs)
            self.success_count += 1
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
    
    def _execute_impl(self, *args, **kwargs) -> Any:
        """Subclass-specific implementation."""
        raise NotImplementedError
    
    def get_schema(self) -> Dict:
        """Get tool schema for agent understanding."""
        raise NotImplementedError
    
    def get_success_rate(self) -> float:
        """Calculate tool success rate."""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count

class JSONSchemaTool(Tool):
    """
    JSON Schema tool following OpenAI function calling format.
    
    This format is used by many LLM APIs and provides structured
    parameter definitions with validation.
    """
    
    def __init__(self, name: str, description: str, parameters: Dict, function: Callable):
        super().__init__(name, description, ToolFormat.JSON_SCHEMA)
        self.parameters = parameters
        self.function = function
    
    def get_schema(self) -> Dict:
        """Return OpenAI-compatible function schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
    
    def _execute_impl(self, **kwargs) -> Any:
        """Execute using JSON parameters."""
        return self.function(**kwargs)

class PythonFunctionTool(Tool):
    """
    Direct Python function tool.
    
    This format allows agents to call Python functions directly,
    with automatic parameter inspection and validation.
    """
    
    def __init__(self, function: Callable):
        name = function.__name__
        description = function.__doc__ or f"Python function: {name}"
        super().__init__(name, description, ToolFormat.PYTHON_FUNCTION)
        self.function = function
        self.signature = inspect.signature(function)
    
    def get_schema(self) -> Dict:
        """Auto-generate schema from function signature."""
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param_name, param in self.signature.parameters.items():
            param_info = {"type": "string"}  # Default type
            
            # Try to infer type from annotation
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_info["type"] = "integer"
                elif param.annotation == float:
                    param_info["type"] = "number"
                elif param.annotation == bool:
                    param_info["type"] = "boolean"
            
            parameters["properties"][param_name] = param_info
            
            # Check if parameter is required
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": parameters
        }
    
    def _execute_impl(self, *args, **kwargs) -> Any:
        """Execute Python function directly."""
        return self.function(*args, **kwargs)

class RestAPITool(Tool):
    """
    REST API tool for external service integration.
    
    This format allows agents to interact with web services,
    APIs, and external systems through HTTP requests.
    """
    
    def __init__(self, name: str, base_url: str, endpoints: Dict):
        description = f"REST API tool for {base_url}"
        super().__init__(name, description, ToolFormat.REST_API)
        self.base_url = base_url
        self.endpoints = endpoints
        self.session = requests.Session()
    
    def get_schema(self) -> Dict:
        """Generate schema from API endpoints."""
        return {
            "name": self.name,
            "description": self.description,
            "endpoints": self.endpoints,
            "base_url": self.base_url
        }
    
    def _execute_impl(self, endpoint: str, method: str = "GET", **kwargs) -> Any:
        """Execute API call."""
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()

class ToolRegistry:
    """
    Centralized tool registry for agent access.
    
    This provides discovery, categorization, and access control
    for all available tools in the system.
    """
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.categories: Dict[str, List[str]] = {}
        self.security_policies: Dict[str, Dict] = {}
    
    def register_tool(self, tool: Tool, category: str = "general", 
                     security_policy: Optional[Dict] = None):
        """Register a tool with optional categorization and security."""
        self.tools[tool.name] = tool
        
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(tool.name)
        
        if security_policy:
            self.security_policies[tool.name] = security_policy
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self.tools.get(name)
    
    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """List available tools, optionally filtered by category."""
        if category:
            return self.categories.get(category, [])
        return list(self.tools.keys())
    
    def get_tools_schema(self) -> List[Dict]:
        """Get schema for all tools."""
        return [tool.get_schema() for tool in self.tools.values()]

# Example tool implementations
def calculate_math(expression: str) -> float:
    """
    Safely evaluate mathematical expressions.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
    
    Returns:
        Result of the calculation
    """
    # Safe evaluation using ast
    import ast
    import operator
    
    # Supported operations
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }
    
    def eval_expr(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
        elif isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](eval_expr(node.operand))
        else:
            raise TypeError(f"Unsupported operation: {type(node)}")
    
    tree = ast.parse(expression, mode='eval')
    return eval_expr(tree.body)

def search_web(query: str, num_results: int = 5) -> List[Dict]:
    """
    Search the web for information.
    
    Args:
        query: Search query
        num_results: Number of results to return
    
    Returns:
        List of search results with title, url, and snippet
    """
    # This is a mock implementation
    # In production, you'd integrate with a real search API
    return [
        {
            "title": f"Result {i+1} for: {query}",
            "url": f"https://example.com/result{i+1}",
            "snippet": f"This is a sample search result {i+1} for the query '{query}'"
        }
        for i in range(num_results)
    ]

def get_weather(location: str) -> Dict:
    """
    Get current weather for a location.
    
    Args:
        location: City name or coordinates
    
    Returns:
        Weather information including temperature, conditions, etc.
    """
    # Mock weather data
    import random
    
    conditions = ["sunny", "cloudy", "rainy", "snowy", "partly cloudy"]
    
    return {
        "location": location,
        "temperature": random.randint(-10, 35),
        "condition": random.choice(conditions),
        "humidity": random.randint(30, 90),
        "wind_speed": random.randint(0, 20)
    }

# Create and populate tool registry
tool_registry = ToolRegistry()

# Register Python function tools
math_tool = PythonFunctionTool(calculate_math)
weather_tool = PythonFunctionTool(get_weather)
search_tool = PythonFunctionTool(search_web)

tool_registry.register_tool(math_tool, "computation")
tool_registry.register_tool(weather_tool, "information")
tool_registry.register_tool(search_tool, "information")

print("✅ Tool Usage Framework Complete!")
print(f"Registered tools: {tool_registry.list_tools()}")
print(f"Categories: {list(tool_registry.categories.keys())}")
```

**Key Takeaways:**

1. **Tool Abstraction**: Different tool formats (JSON Schema, Python functions, REST APIs) unified under a common interface
2. **Security Patterns**: Built-in error handling, validation, and access control
3. **Registry System**: Centralized tool discovery and management
4. **Production Ready**: Proper error handling, monitoring, and scalability considerations

**Exercise: Extend the Framework**
Try adding a new tool type for command-line programs, or implement a tool that reads/writes files. Consider security implications and error handling patterns.

---

## 🧠 Chapter 3: Advanced Agent Architectures and Reasoning Patterns

Now that we understand tools, let's explore sophisticated agent architectures that enable complex reasoning and autonomous behavior.

### The ReAct Pattern: Reasoning + Acting

**ReAct** (Reasoning and Acting) is a breakthrough pattern that combines thinking and acting in language models, enabling step-by-step problem solving.

**The ReAct Loop:**
1. **Thought**: Reason about the current situation
2. **Action**: Take an action based on reasoning
3. **Observation**: Observe the result of the action
4. **Repeat**: Continue until goal is achieved

```python
# react_agent.py - Implementation of the ReAct reasoning pattern
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class ReActStepType(Enum):
    THOUGHT = "Thought"
    ACTION = "Action"
    OBSERVATION = "Observation"

@dataclass
class ReActStep:
    """Single step in ReAct reasoning trace."""
    step_type: ReActStepType
    content: str
    step_number: int

class ReActAgent:
    """
    ReAct Agent implementing Reasoning + Acting pattern.
    
    This agent alternates between reasoning about the situation
    and taking actions to gather information or make progress.
    """
    
    def __init__(self, llm_client, tool_registry: ToolRegistry, max_steps: int = 10):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.trace: List[ReActStep] = []
    
    def solve(self, question: str) -> str:
        """
        Solve a question using ReAct reasoning.
        
        Args:
            question: The question or task to solve
            
        Returns:
            Final answer or result
        """
        self.trace = []
        
        # Initial system prompt for ReAct
        system_prompt = """You are a helpful assistant that uses tools to answer questions.

You have access to the following tools:
{tools}

Use the following format for your responses:

Thought: [Your reasoning about what to do next]
Action: [The action to take - must be one of the available tools]
Action Input: [The input to the action]
Observation: [The result of the action]

You can repeat Thought/Action/Observation multiple times.
When you have enough information to answer the question, provide:

Final Answer: [Your final answer to the question]

Question: {question}"""

        tools_description = self._format_tools_for_prompt()
        prompt = system_prompt.format(tools=tools_description, question=question)
        
        step_count = 0
        
        while step_count < self.max_steps:
            # Get response from LLM
            response = self._get_llm_response(prompt)
            
            # Parse the response
            parsed_steps = self._parse_react_response(response)
            
            # Process each step
            for step in parsed_steps:
                self.trace.append(step)
                
                if step.step_type == ReActStepType.ACTION:
                    # Execute the action
                    observation = self._execute_action(step.content)
                    obs_step = ReActStep(
                        step_type=ReActStepType.OBSERVATION,
                        content=observation,
                        step_number=len(self.trace)
                    )
                    self.trace.append(obs_step)
                    
                    # Add observation to prompt for next iteration
                    prompt += f"\nObservation: {observation}"
            
            # Check if we have a final answer
            if "Final Answer:" in response:
                final_answer = response.split("Final Answer:")[-1].strip()
                return final_answer
            
            step_count += 1
        
        return "Maximum steps reached without finding an answer."
    
    def _format_tools_for_prompt(self) -> str:
        """Format available tools for the prompt."""
        tools_list = []
        for tool_name in self.tool_registry.list_tools():
            tool = self.tool_registry.get_tool(tool_name)
            tools_list.append(f"- {tool.name}: {tool.description}")
        return "\n".join(tools_list)
    
    def _get_llm_response(self, prompt: str) -> str:
        """Get response from language model."""
        # This would integrate with your actual LLM client
        # For now, we'll simulate a response
        return "Thought: I need to search for information about this topic.\nAction: search_web\nAction Input: " + prompt.split("Question:")[-1].strip()
    
    def _parse_react_response(self, response: str) -> List[ReActStep]:
        """Parse ReAct format response into steps."""
        steps = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith("Thought:"):
                steps.append(ReActStep(
                    step_type=ReActStepType.THOUGHT,
                    content=line[8:].strip(),
                    step_number=len(self.trace) + len(steps)
                ))
            elif line.startswith("Action:"):
                steps.append(ReActStep(
                    step_type=ReActStepType.ACTION,
                    content=line[7:].strip(),
                    step_number=len(self.trace) + len(steps)
                ))
        
        return steps
    
    def _execute_action(self, action_line: str) -> str:
        """Execute an action and return observation."""
        # Parse action and input
        parts = action_line.split('\n')
        action_name = parts[0].strip()
        
        # Find action input
        action_input = ""
        for part in parts[1:]:
            if part.strip().startswith("Action Input:"):
                action_input = part.strip()[13:].strip()
                break
        
        # Execute tool
        tool = self.tool_registry.get_tool(action_name)
        if not tool:
            return f"Error: Tool '{action_name}' not found"
        
        try:
            # For simplicity, assume action_input is the main parameter
            result = tool.execute(action_input)
            if result.success:
                return str(result.data)
            else:
                return f"Error: {result.error}"
        except Exception as e:
            return f"Error executing {action_name}: {str(e)}"

### Planning Agents: Hierarchical Task Decomposition

```python
# planning_agent.py - Hierarchical planning and execution
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    """Represents a single task in a plan."""
    id: str
    description: str
    dependencies: List[str]
    estimated_complexity: int  # 1-10 scale
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    subtasks: List['Task'] = None
    
    def __post_init__(self):
        if self.subtasks is None:
            self.subtasks = []

class PlanningAgent:
    """
    Hierarchical planning agent that decomposes complex tasks.
    
    This agent creates plans by breaking down complex objectives
    into manageable subtasks with dependencies.
    """
    
    def __init__(self, llm_client, tool_registry: ToolRegistry):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.current_plan: List[Task] = []
        self.completed_tasks: List[Task] = []
    
    def create_plan(self, objective: str) -> List[Task]:
        """
        Create a hierarchical plan for achieving an objective.
        
        Args:
            objective: High-level goal to achieve
            
        Returns:
            List of tasks representing the plan
        """
        # Use LLM to decompose the objective
        planning_prompt = f"""
        Break down this objective into specific, actionable tasks:
        
        Objective: {objective}
        
        Available tools: {', '.join(self.tool_registry.list_tools())}
        
        For each task, provide:
        1. Clear description
        2. Dependencies (which tasks must complete first)
        3. Complexity estimate (1-10)
        4. Whether it needs subtasks
        
        Format as JSON list of tasks.
        """
        
        # This would use your actual LLM client
        plan_response = self._get_planning_response(planning_prompt)
        
        # Parse response into Task objects
        tasks = self._parse_plan_response(plan_response)
        
        # Create hierarchical structure
        self.current_plan = self._create_task_hierarchy(tasks)
        
        return self.current_plan
    
    def execute_plan(self) -> Dict[str, Any]:
        """
        Execute the current plan using topological sorting.
        
        Returns:
            Execution results and statistics
        """
        execution_order = self._topological_sort()
        results = {
            "completed_tasks": [],
            "failed_tasks": [],
            "total_time": 0,
            "success_rate": 0
        }
        
        for task in execution_order:
            if self._can_execute_task(task):
                try:
                    task.status = TaskStatus.IN_PROGRESS
                    result = self._execute_task(task)
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    results["completed_tasks"].append(task.id)
                    self.completed_tasks.append(task)
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.result = f"Error: {str(e)}"
                    results["failed_tasks"].append(task.id)
        
        # Calculate success rate
        total_tasks = len(results["completed_tasks"]) + len(results["failed_tasks"])
        if total_tasks > 0:
            results["success_rate"] = len(results["completed_tasks"]) / total_tasks
        
        return results
    
    def _get_planning_response(self, prompt: str) -> str:
        """Get planning response from LLM."""
        # Simulate LLM response for planning
        return """[
            {
                "id": "task_1",
                "description": "Gather initial information",
                "dependencies": [],
                "complexity": 3
            },
            {
                "id": "task_2", 
                "description": "Analyze gathered data",
                "dependencies": ["task_1"],
                "complexity": 5
            },
            {
                "id": "task_3",
                "description": "Generate final report",
                "dependencies": ["task_2"],
                "complexity": 4
            }
        ]"""
    
    def _parse_plan_response(self, response: str) -> List[Task]:
        """Parse LLM response into Task objects."""
        import json
        
        try:
            task_data = json.loads(response)
            tasks = []
            
            for data in task_data:
                task = Task(
                    id=data["id"],
                    description=data["description"],
                    dependencies=data.get("dependencies", []),
                    estimated_complexity=data.get("complexity", 5)
                )
                tasks.append(task)
            
            return tasks
        except Exception as e:
            # Fallback to simple task creation
            return [Task(
                id="fallback_task",
                description="Execute objective directly",
                dependencies=[],
                estimated_complexity=5
            )]
    
    def _create_task_hierarchy(self, tasks: List[Task]) -> List[Task]:
        """Create hierarchical task structure."""
        # For complex tasks, break them down further
        hierarchical_tasks = []
        
        for task in tasks:
            if task.estimated_complexity > 7:
                # Break down complex tasks into subtasks
                subtasks = self._decompose_complex_task(task)
                task.subtasks = subtasks
            
            hierarchical_tasks.append(task)
        
        return hierarchical_tasks
    
    def _decompose_complex_task(self, task: Task) -> List[Task]:
        """Decompose a complex task into subtasks."""
        # Simple decomposition strategy
        subtasks = [
            Task(
                id=f"{task.id}_prep",
                description=f"Prepare for: {task.description}",
                dependencies=[],
                estimated_complexity=2
            ),
            Task(
                id=f"{task.id}_execute",
                description=f"Execute: {task.description}",
                dependencies=[f"{task.id}_prep"],
                estimated_complexity=task.estimated_complexity - 3
            ),
            Task(
                id=f"{task.id}_verify",
                description=f"Verify: {task.description}",
                dependencies=[f"{task.id}_execute"],
                estimated_complexity=2
            )
        ]
        
        return subtasks
    
    def _topological_sort(self) -> List[Task]:
        """Sort tasks by dependencies using topological sort."""
        # Flatten all tasks including subtasks
        all_tasks = []
        for task in self.current_plan:
            all_tasks.append(task)
            all_tasks.extend(task.subtasks)
        
        # Simple topological sort implementation
        sorted_tasks = []
        remaining_tasks = all_tasks.copy()
        
        while remaining_tasks:
            # Find tasks with no unmet dependencies
            ready_tasks = []
            for task in remaining_tasks:
                deps_met = all(
                    any(t.id == dep_id and t.status == TaskStatus.COMPLETED 
                        for t in sorted_tasks)
                    for dep_id in task.dependencies
                ) if task.dependencies else True
                
                if deps_met:
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # Circular dependency or other issue
                ready_tasks = [remaining_tasks[0]]  # Force progress
            
            # Add ready tasks to sorted list
            for task in ready_tasks:
                sorted_tasks.append(task)
                remaining_tasks.remove(task)
        
        return sorted_tasks
    
    def _can_execute_task(self, task: Task) -> bool:
        """Check if a task can be executed."""
        if task.status != TaskStatus.PENDING:
            return False
        
        # Check if dependencies are met
        for dep_id in task.dependencies:
            dep_completed = any(
                t.id == dep_id and t.status == TaskStatus.COMPLETED
                for t in self.completed_tasks
            )
            if not dep_completed:
                return False
        
        return True
    
    def _execute_task(self, task: Task) -> Any:
        """Execute a single task."""
        # Determine which tool to use based on task description
        if "search" in task.description.lower():
            tool = self.tool_registry.get_tool("search_web")
            result = tool.execute(task.description)
            return result.data
        elif "calculate" in task.description.lower():
            tool = self.tool_registry.get_tool("calculate_math")
            # Extract mathematical expression from description
            result = tool.execute("2 + 2")  # Simplified
            return result.data
        else:
            # Default execution
            return f"Completed: {task.description}"

print("✅ Advanced Agent Architectures Complete!")
print("Key patterns: ReAct reasoning, Hierarchical planning, Task decomposition")
```

### Memory and State Management

```python
# memory_agent.py - Agent memory and state management
from typing import Dict, List, Any, Optional
import json
import hashlib
from datetime import datetime, timedelta

class MemoryType(Enum):
    WORKING = "working"        # Short-term, current task
    EPISODIC = "episodic"      # Specific experiences/events
    SEMANTIC = "semantic"      # General knowledge
    PROCEDURAL = "procedural"  # How to do things

@dataclass
class Memory:
    """Single memory entry."""
    id: str
    content: Any
    memory_type: MemoryType
    timestamp: datetime
    importance: float  # 0.0 to 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    related_memories: List[str] = None
    
    def __post_init__(self):
        if self.related_memories is None:
            self.related_memories = []

class MemoryManager:
    """
    Sophisticated memory management for agents.
    
    Implements different types of memory with forgetting curves,
    importance weighting, and associative recall.
    """
    
    def __init__(self, max_working_memory: int = 7, max_total_memory: int = 1000):
        self.memories: Dict[str, Memory] = {}
        self.max_working_memory = max_working_memory
        self.max_total_memory = max_total_memory
        self.working_memory: List[str] = []  # IDs of active memories
    
    def store_memory(self, content: Any, memory_type: MemoryType, 
                    importance: float = 0.5) -> str:
        """Store a new memory."""
        memory_id = self._generate_memory_id(content)
        
        memory = Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            timestamp=datetime.now(),
            importance=importance
        )
        
        self.memories[memory_id] = memory
        
        # Add to working memory if appropriate
        if memory_type == MemoryType.WORKING:
            self._add_to_working_memory(memory_id)
        
        # Manage memory limits
        self._manage_memory_limits()
        
        return memory_id
    
    def recall_memory(self, query: str, memory_type: Optional[MemoryType] = None,
                     limit: int = 5) -> List[Memory]:
        """Recall memories based on query."""
        relevant_memories = []
        
        for memory in self.memories.values():
            if memory_type and memory.memory_type != memory_type:
                continue
            
            # Simple relevance scoring
            relevance = self._calculate_relevance(query, memory)
            if relevance > 0.1:  # Threshold for relevance
                memory.access_count += 1
                memory.last_accessed = datetime.now()
                relevant_memories.append((memory, relevance))
        
        # Sort by relevance and importance
        relevant_memories.sort(key=lambda x: x[1] * x[0].importance, reverse=True)
        
        return [mem for mem, _ in relevant_memories[:limit]]
    
    def get_working_memory(self) -> List[Memory]:
        """Get current working memory contents."""
        return [self.memories[mem_id] for mem_id in self.working_memory 
                if mem_id in self.memories]
    
    def _generate_memory_id(self, content: Any) -> str:
        """Generate unique ID for memory content."""
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.md5(content_str.encode()).hexdigest()[:16]
    
    def _add_to_working_memory(self, memory_id: str):
        """Add memory to working memory, managing capacity."""
        if memory_id in self.working_memory:
            # Move to front (most recent)
            self.working_memory.remove(memory_id)
        
        self.working_memory.insert(0, memory_id)
        
        # Limit working memory size
        if len(self.working_memory) > self.max_working_memory:
            self.working_memory = self.working_memory[:self.max_working_memory]
    
    def _calculate_relevance(self, query: str, memory: Memory) -> float:
        """Calculate relevance of memory to query."""
        # Simple text-based relevance
        query_words = set(query.lower().split())
        memory_text = str(memory.content).lower()
        memory_words = set(memory_text.split())
        
        # Jaccard similarity
        intersection = len(query_words & memory_words)
        union = len(query_words | memory_words)
        
        if union == 0:
            return 0.0
        
        base_relevance = intersection / union
        
        # Adjust for recency and importance
        recency_factor = self._calculate_recency_factor(memory)
        
        return base_relevance * memory.importance * recency_factor
    
    def _calculate_recency_factor(self, memory: Memory) -> float:
        """Calculate recency factor using forgetting curve."""
        time_since = datetime.now() - memory.timestamp
        days = time_since.total_seconds() / (24 * 3600)
        
        # Exponential decay with different rates for different memory types
        decay_rates = {
            MemoryType.WORKING: 0.5,    # Fast decay
            MemoryType.EPISODIC: 0.1,   # Medium decay
            MemoryType.SEMANTIC: 0.01,  # Slow decay
            MemoryType.PROCEDURAL: 0.005  # Very slow decay
        }
        
        decay_rate = decay_rates.get(memory.memory_type, 0.1)
        return max(0.1, 1.0 - (days * decay_rate))
    
    def _manage_memory_limits(self):
        """Manage memory limits by forgetting less important memories."""
        if len(self.memories) <= self.max_total_memory:
            return
        
        # Calculate forgetting scores
        forgetting_scores = []
        for memory_id, memory in self.memories.items():
            if memory.memory_type == MemoryType.WORKING:
                continue  # Don't forget working memory
            
            # Score based on importance, recency, and access frequency
            recency_factor = self._calculate_recency_factor(memory)
            access_factor = min(1.0, memory.access_count / 10.0)
            
            forgetting_score = (1.0 - memory.importance) * (1.0 - recency_factor) * (1.0 - access_factor)
            forgetting_scores.append((memory_id, forgetting_score))
        
        # Sort by forgetting score (highest = most forgettable)
        forgetting_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Remove memories until under limit
        memories_to_remove = len(self.memories) - self.max_total_memory
        for i in range(memories_to_remove):
            if i < len(forgetting_scores):
                memory_id = forgetting_scores[i][0]
                del self.memories[memory_id]
                if memory_id in self.working_memory:
                    self.working_memory.remove(memory_id)

class MemoryAugmentedAgent:
    """
    Agent with sophisticated memory capabilities.
    
    Combines memory management with reasoning and tool usage
    for persistent, context-aware behavior.
    """
    
    def __init__(self, llm_client, tool_registry: ToolRegistry):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.memory_manager = MemoryManager()
        self.conversation_history: List[Dict] = []
    
    def process_input(self, user_input: str) -> str:
        """Process user input with memory-augmented reasoning."""
        # Store user input in memory
        self.memory_manager.store_memory(
            content={"type": "user_input", "text": user_input},
            memory_type=MemoryType.EPISODIC,
            importance=0.7
        )
        
        # Recall relevant memories
        relevant_memories = self.memory_manager.recall_memory(user_input)
        
        # Get working memory context
        working_memory = self.memory_manager.get_working_memory()
        
        # Build context for LLM
        context = self._build_context(user_input, relevant_memories, working_memory)
        
        # Generate response
        response = self._generate_response(context)
        
        # Store response in memory
        self.memory_manager.store_memory(
            content={"type": "agent_response", "text": response},
            memory_type=MemoryType.EPISODIC,
            importance=0.6
        )
        
        # Update conversation history
        self.conversation_history.append({
            "user": user_input,
            "agent": response,
            "timestamp": datetime.now()
        })
        
        return response
    
    def _build_context(self, user_input: str, relevant_memories: List[Memory],
                      working_memory: List[Memory]) -> str:
        """Build context string for LLM."""
        context_parts = []
        
        # Add relevant memories
        if relevant_memories:
            context_parts.append("Relevant past memories:")
            for memory in relevant_memories:
                context_parts.append(f"- {memory.content}")
        
        # Add working memory
        if working_memory:
            context_parts.append("\nCurrent working memory:")
            for memory in working_memory:
                context_parts.append(f"- {memory.content}")
        
        # Add current input
        context_parts.append(f"\nCurrent input: {user_input}")
        
        return "\n".join(context_parts)
    
    def _generate_response(self, context: str) -> str:
        """Generate response using LLM with context."""
        # This would use your actual LLM client
        # For now, return a simulated response
        return f"Based on the context and my memories, I understand you're asking about: {context.split('Current input:')[-1].strip()}"

print("✅ Memory and State Management Complete!")
print("Features: Working memory, Episodic/Semantic memory, Forgetting curves, Associative recall")
```

---

## 🤝 Chapter 4: Multi-Agent Systems and Coordination

Multi-agent systems represent the cutting edge of AI coordination, where multiple specialized agents work together to solve complex problems.

### Agent Communication Protocols

```python
# multi_agent_system.py - Advanced multi-agent coordination
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
from datetime import datetime

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    COORDINATION = "coordination"
    STATUS_UPDATE = "status_update"

class AgentRole(Enum):
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    EXECUTOR = "executor"
    MONITOR = "monitor"

@dataclass
class Message:
    """Inter-agent communication message."""
    id: str
    sender_id: str
    receiver_id: Optional[str]  # None for broadcasts
    message_type: MessageType
    content: Any
    timestamp: datetime
    priority: int = 5  # 1-10, higher = more urgent
    requires_response: bool = False

class AgentProtocol:
    """Communication protocol for agent interactions."""
    
    def __init__(self):
        self.message_queue: Dict[str, List[Message]] = {}
        self.agents: Dict[str, 'BaseAgent'] = {}
        self.message_history: List[Message] = []
    
    def register_agent(self, agent: 'BaseAgent'):
        """Register an agent with the protocol."""
        self.agents[agent.agent_id] = agent
        self.message_queue[agent.agent_id] = []
    
    def send_message(self, message: Message):
        """Send a message between agents."""
        self.message_history.append(message)
        
        if message.receiver_id:
            # Direct message
            if message.receiver_id in self.message_queue:
                self.message_queue[message.receiver_id].append(message)
        else:
            # Broadcast message
            for agent_id in self.message_queue:
                if agent_id != message.sender_id:
                    self.message_queue[agent_id].append(message)
    
    def get_messages(self, agent_id: str) -> List[Message]:
        """Get pending messages for an agent."""
        messages = self.message_queue.get(agent_id, [])
        self.message_queue[agent_id] = []  # Clear after retrieval
        return sorted(messages, key=lambda m: m.priority, reverse=True)

class BaseAgent:
    """Base class for multi-agent system participants."""
    
    def __init__(self, agent_id: str, role: AgentRole, llm_client=None):
        self.agent_id = agent_id
        self.role = role
        self.llm_client = llm_client
        self.protocol: Optional[AgentProtocol] = None
        self.capabilities: List[str] = []
        self.current_tasks: List[Dict] = []
        self.status = "idle"
    
    def join_protocol(self, protocol: AgentProtocol):
        """Join a communication protocol."""
        self.protocol = protocol
        protocol.register_agent(self)
    
    def send_message(self, receiver_id: Optional[str], message_type: MessageType,
                    content: Any, priority: int = 5):
        """Send a message to another agent."""
        if not self.protocol:
            raise ValueError("Agent not connected to protocol")
        
        message = Message(
            id=f"{self.agent_id}_{datetime.now().timestamp()}",
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            timestamp=datetime.now(),
            priority=priority
        )
        
        self.protocol.send_message(message)
    
    def process_messages(self) -> List[Any]:
        """Process incoming messages."""
        if not self.protocol:
            return []
        
        messages = self.protocol.get_messages(self.agent_id)
        responses = []
        
        for message in messages:
            response = self.handle_message(message)
            if response:
                responses.append(response)
        
        return responses
    
    def handle_message(self, message: Message) -> Optional[Any]:
        """Handle a received message."""
        # Override in subclasses
        return None
    
    def advertise_capabilities(self):
        """Broadcast capabilities to other agents."""
        self.send_message(
            receiver_id=None,  # Broadcast
            message_type=MessageType.BROADCAST,
            content={
                "type": "capabilities",
                "agent_id": self.agent_id,
                "role": self.role.value,
                "capabilities": self.capabilities
            }
        )

class EnhancedAssistantAgent(BaseAgent):
    """
    Enhanced agent that wraps YOUR existing Assistant class with autonomous capabilities.
    This preserves all your existing functionality while adding agent features.
    """
    
    def __init__(self, agent_id: str, role: AgentRole, your_assistant, capabilities: List[str] = None):
        # Initialize base agent
        super().__init__(agent_id, role, None)  # No LLM client - we use YOUR assistant's
        
        # Wrap YOUR existing Assistant
        self.your_assistant = your_assistant
        self.capabilities = capabilities or ["professional_representation", "career_guidance", "tool_usage"]
        
        # Agent-specific enhancements
        self.conversation_context = []
        self.autonomous_mode = False
        self.goals = []
        self.memory_manager = None  # Will be set later if needed
    
    def handle_message(self, message: Message) -> Optional[Any]:
        """Handle message using YOUR Assistant enhanced with agent capabilities."""
        try:
            # Convert agent message to your assistant's format
            conversation_messages = [{"role": "user", "content": str(message.content)}]
            
            # Add conversation context for continuity
            if self.conversation_context:
                conversation_messages = self.conversation_context + conversation_messages
            
            # Use YOUR existing Assistant's get_response method
            assistant_responses = self.your_assistant.get_response(conversation_messages)
            
            # Extract response content
            response_content = ""
            if assistant_responses:
                first_response = assistant_responses[0]
                if hasattr(first_response, 'content'):
                    response_content = first_response.content
                elif isinstance(first_response, dict):
                    response_content = first_response.get('content', str(first_response))
                else:
                    response_content = str(first_response)
            
            # Update conversation context (keep last 10 exchanges)
            self.conversation_context.extend([
                {"role": "user", "content": str(message.content)},
                {"role": "assistant", "content": response_content}
            ])
            self.conversation_context = self.conversation_context[-20:]  # Keep last 10 exchanges
            
            # Store in memory if available
            if self.memory_manager:
                self.memory_manager.store_memory(Memory(
                    memory_id=f"conv_{datetime.now().timestamp()}",
                    memory_type=MemoryType.EPISODIC,
                    content=f"User: {message.content}\nAssistant: {response_content}",
                    importance=0.8,
                    timestamp=datetime.now()
                ))
            
            # Send response back if it's not a broadcast
            if message.receiver_id and message.receiver_id != "all":
                self.send_message(
                    receiver_id=message.sender_id,
                    message_type=MessageType.RESPONSE,
                    content=response_content
                )
            
            return response_content
            
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {str(e)}. Please try again."
            if message.receiver_id and message.receiver_id != "all":
                self.send_message(
                    receiver_id=message.sender_id,
                    message_type=MessageType.RESPONSE,
                    content=error_msg
                )
            return error_msg
    
    def process_input(self, user_input: str) -> str:
        """Direct input processing for chat interface compatibility."""
        try:
            # Use YOUR assistant directly for simple input
            conversation_messages = [{"role": "user", "content": user_input}]
            
            # Add context if available
            if self.conversation_context:
                conversation_messages = self.conversation_context + conversation_messages
            
            responses = self.your_assistant.get_response(conversation_messages)
            
            if responses:
                first_response = responses[0]
                if hasattr(first_response, 'content'):
                    return first_response.content
                elif isinstance(first_response, dict):
                    return first_response.get('content', str(first_response))
                else:
                    return str(first_response)
            
            return "I'm sorry, I couldn't process that request."
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try again."
    
    def enable_autonomous_mode(self, goals: List[str]):
        """Enable autonomous operation with specific goals."""
        self.autonomous_mode = True
        self.goals = goals
        self.status = "autonomous"
        
    def get_status_report(self) -> Dict:
        """Get detailed status of the enhanced assistant."""
        memory_count = 0
        if self.memory_manager and hasattr(self.memory_manager, 'memories'):
            memory_count = len(self.memory_manager.memories)
        
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "autonomous_mode": self.autonomous_mode,
            "goals": self.goals,
            "conversation_turns": len(self.conversation_context) // 2,
            "memory_count": memory_count,
            "capabilities": self.capabilities,
            "assistant_model": getattr(self.your_assistant, 'model', 'llama-3.3-70b-versatile')
        }

class CoordinatorAgent(BaseAgent):
    """
    Coordinator agent that manages task distribution and execution.
    
    Responsibilities:
    - Decompose complex tasks
    - Assign tasks to appropriate specialists
    - Monitor progress and coordinate execution
    - Handle conflicts and resource allocation
    """
    
    def __init__(self, agent_id: str, llm_client=None):
        super().__init__(agent_id, AgentRole.COORDINATOR, llm_client)
        self.capabilities = ["task_decomposition", "resource_allocation", "conflict_resolution"]
        self.agent_directory: Dict[str, Dict] = {}
        self.active_projects: Dict[str, Dict] = {}
    
    def coordinate_task(self, task_description: str) -> str:
        """Coordinate execution of a complex task."""
        # Decompose task
        subtasks = self._decompose_task(task_description)
        
        # Create project
        project_id = f"project_{datetime.now().timestamp()}"
        self.active_projects[project_id] = {
            "description": task_description,
            "subtasks": subtasks,
            "assigned_agents": {},
            "status": "planning",
            "start_time": datetime.now()
        }
        
        # Assign subtasks to agents
        for subtask in subtasks:
            assigned_agent = self._find_best_agent_for_task(subtask)
            if assigned_agent:
                self._assign_task_to_agent(assigned_agent, subtask, project_id)
        
        return project_id
    
    def _decompose_task(self, task_description: str) -> List[Dict]:
        """Decompose complex task into subtasks."""
        # Use LLM for intelligent task decomposition
        subtasks = [
            {
                "id": f"subtask_1",
                "description": f"Research phase for: {task_description}",
                "required_capabilities": ["information_gathering"],
                "priority": 8
            },
            {
                "id": f"subtask_2", 
                "description": f"Analysis phase for: {task_description}",
                "required_capabilities": ["data_analysis"],
                "priority": 6,
                "dependencies": ["subtask_1"]
            },
            {
                "id": f"subtask_3",
                "description": f"Synthesis phase for: {task_description}",
                "required_capabilities": ["synthesis", "reporting"],
                "priority": 7,
                "dependencies": ["subtask_2"]
            }
        ]
        return subtasks
    
    def _find_best_agent_for_task(self, subtask: Dict) -> Optional[str]:
        """Find the best agent for a specific subtask."""
        required_caps = subtask.get("required_capabilities", [])
        
        best_agent = None
        best_score = 0
        
        for agent_id, agent_info in self.agent_directory.items():
            agent_caps = agent_info.get("capabilities", [])
            
            # Calculate capability match score
            matching_caps = len(set(required_caps) & set(agent_caps))
            total_caps = len(required_caps)
            
            if total_caps == 0:
                score = 1.0
            else:
                score = matching_caps / total_caps
            
            # Consider agent availability
            if agent_info.get("status") == "busy":
                score *= 0.5
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
        
        return best_agent
    
    def _assign_task_to_agent(self, agent_id: str, subtask: Dict, project_id: str):
        """Assign a subtask to a specific agent."""
        self.send_message(
            receiver_id=agent_id,
            message_type=MessageType.REQUEST,
            content={
                "type": "task_assignment",
                "project_id": project_id,
                "subtask": subtask,
                "deadline": "1 hour"  # Could be calculated based on priority
            },
            priority=subtask.get("priority", 5)
        )
    
    def handle_message(self, message: Message) -> Optional[Any]:
        """Handle incoming messages."""
        content = message.content
        
        if content.get("type") == "capabilities":
            # Update agent directory
            self.agent_directory[message.sender_id] = {
                "role": content.get("role"),
                "capabilities": content.get("capabilities"),
                "last_seen": datetime.now(),
                "status": "available"
            }
        
        elif content.get("type") == "task_completion":
            # Handle task completion
            project_id = content.get("project_id")
            subtask_id = content.get("subtask_id")
            result = content.get("result")
            
            if project_id in self.active_projects:
                # Update project status
                project = self.active_projects[project_id]
                project["assigned_agents"][message.sender_id] = {
                    "subtask_id": subtask_id,
                    "result": result,
                    "completion_time": datetime.now()
                }
                
                # Check if project is complete
                self._check_project_completion(project_id)
        
        elif content.get("type") == "status_update":
            # Update agent status
            if message.sender_id in self.agent_directory:
                self.agent_directory[message.sender_id]["status"] = content.get("status")
        
        return None
    
    def _check_project_completion(self, project_id: str):
        """Check if a project is complete."""
        project = self.active_projects[project_id]
        subtasks = project["subtasks"]
        completed_subtasks = len(project["assigned_agents"])
        
        if completed_subtasks >= len(subtasks):
            project["status"] = "completed"
            project["completion_time"] = datetime.now()
            
            # Notify all participants
            self.send_message(
                receiver_id=None,  # Broadcast
                message_type=MessageType.BROADCAST,
                content={
                    "type": "project_completion",
                    "project_id": project_id,
                    "results": project["assigned_agents"]
                }
            )

class SpecialistAgent(BaseAgent):
    """
    Specialist agent with specific domain expertise.
    
    Examples: ResearchAgent, AnalysisAgent, SynthesisAgent
    """
    
    def __init__(self, agent_id: str, specialization: str, llm_client=None):
        super().__init__(agent_id, AgentRole.SPECIALIST, llm_client)
        self.specialization = specialization
        self.capabilities = self._define_capabilities()
        self.tool_registry = ToolRegistry()  # Agent-specific tools
    
    def _define_capabilities(self) -> List[str]:
        """Define capabilities based on specialization."""
        capability_map = {
            "research": ["information_gathering", "web_search", "data_collection"],
            "analysis": ["data_analysis", "statistical_analysis", "pattern_recognition"],
            "synthesis": ["synthesis", "reporting", "summarization"],
            "execution": ["task_execution", "automation", "workflow_management"]
        }
        return capability_map.get(self.specialization, ["general"])
    
    def handle_message(self, message: Message) -> Optional[Any]:
        """Handle task assignments and coordination messages."""
        content = message.content
        
        if content.get("type") == "task_assignment":
            # Execute assigned task
            subtask = content.get("subtask")
            project_id = content.get("project_id")
            
            self.status = "busy"
            
            # Execute the subtask
            result = self._execute_subtask(subtask)
            
            self.status = "available"
            
            # Report completion
            self.send_message(
                receiver_id=message.sender_id,
                message_type=MessageType.RESPONSE,
                content={
                    "type": "task_completion",
                    "project_id": project_id,
                    "subtask_id": subtask.get("id"),
                    "result": result,
                    "agent_id": self.agent_id
                }
            )
        
        return None
    
    def _execute_subtask(self, subtask: Dict) -> Any:
        """Execute a specific subtask."""
        description = subtask.get("description", "")
        
        # Use specialization-specific logic
        if self.specialization == "research":
            return self._research_task(description)
        elif self.specialization == "analysis":
            return self._analysis_task(description)
        elif self.specialization == "synthesis":
            return self._synthesis_task(description)
        else:
            return f"Completed: {description}"
    
    def _research_task(self, description: str) -> Dict:
        """Execute research task."""
        # Use research tools
        search_tool = self.tool_registry.get_tool("search_web")
        if search_tool:
            result = search_tool.execute(description)
            return {
                "type": "research_results",
                "query": description,
                "findings": result.data if result.success else "No results found",
                "confidence": 0.8
            }
        return {"type": "research_results", "query": description, "findings": "Mock research results"}
    
    def _analysis_task(self, description: str) -> Dict:
        """Execute analysis task."""
        return {
            "type": "analysis_results",
            "input": description,
            "analysis": "Detailed analysis of the provided data",
            "insights": ["Key insight 1", "Key insight 2"],
            "confidence": 0.85
        }
    
    def _synthesis_task(self, description: str) -> Dict:
        """Execute synthesis task."""
        return {
            "type": "synthesis_results",
            "input": description,
            "synthesis": "Comprehensive synthesis of all inputs",
            "recommendations": ["Recommendation 1", "Recommendation 2"],
            "confidence": 0.9
        }

class MultiAgentSystem:
    """
    Complete multi-agent system orchestrator.
    
    Manages agent lifecycle, communication, and system-wide coordination.
    """
    
    def __init__(self):
        self.protocol = AgentProtocol()
        self.agents: Dict[str, BaseAgent] = {}
        self.system_status = "initializing"
    
    def add_agent(self, agent: BaseAgent):
        """Add an agent to the system."""
        self.agents[agent.agent_id] = agent
        agent.join_protocol(self.protocol)
        
        # Have agent advertise its capabilities
        agent.advertise_capabilities()
    
    def create_research_team(self) -> List[str]:
        """Create a specialized research team."""
        team_agents = []
        
        # Create coordinator
        coordinator = CoordinatorAgent("coordinator_1")
        self.add_agent(coordinator)
        team_agents.append(coordinator.agent_id)
        
        # Create specialists
        research_agent = SpecialistAgent("researcher_1", "research")
        analysis_agent = SpecialistAgent("analyst_1", "analysis")
        synthesis_agent = SpecialistAgent("synthesizer_1", "synthesis")
        
        self.add_agent(research_agent)
        self.add_agent(analysis_agent)
        self.add_agent(synthesis_agent)
        
        team_agents.extend([
            research_agent.agent_id,
            analysis_agent.agent_id,
            synthesis_agent.agent_id
        ])
        
        return team_agents
    
    def execute_collaborative_task(self, task_description: str) -> str:
        """Execute a task using the multi-agent system."""
        # Get coordinator
        coordinator = None
        for agent in self.agents.values():
            if agent.role == AgentRole.COORDINATOR:
                coordinator = agent
                break
        
        if not coordinator:
            return "No coordinator available"
        
        # Start task coordination
        project_id = coordinator.coordinate_task(task_description)
        
        # Process messages until completion
        max_iterations = 50
        iteration = 0
        
        while iteration < max_iterations:
            # Process messages for all agents
            for agent in self.agents.values():
                agent.process_messages()
            
            # Check if project is complete
            if project_id in coordinator.active_projects:
                project = coordinator.active_projects[project_id]
                if project.get("status") == "completed":
                    return f"Project {project_id} completed successfully"
            
            iteration += 1
            # In a real system, you'd add proper async handling here
        
        return f"Project {project_id} timed out"

# Example usage and testing
async def demo_multi_agent_system():
    """Demonstrate multi-agent system capabilities."""
    
    # Create system
    mas = MultiAgentSystem()
    
    # Create research team
    team = mas.create_research_team()
    print(f"✅ Created research team: {team}")
    
    # Execute collaborative task
    task = "Analyze the impact of AI on software development practices"
    result = mas.execute_collaborative_task(task)
    print(f"✅ Task result: {result}")
    
    return mas

print("✅ Multi-Agent Systems Complete!")
print("Features: Agent communication, Task coordination, Role specialization, Collaborative execution")
```

---

## 🔗 Chapter 5: Integration with Your Platform

Now let's integrate all these advanced agent capabilities with your existing Flask/React platform to create a production-ready AI-powered system.

### Flask Backend Enhancement for Agent Integration

```python
# enhanced_app.py - Production-ready agent-powered Flask backend
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any

# Import our agent framework
from multi_agent_system import MultiAgentSystem, CoordinatorAgent, SpecialistAgent

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global agent system
agent_system = MultiAgentSystem()

class PlatformAgentManager:
    """
    Manages agents specifically for the web platform.
    
    Provides web-friendly interfaces and real-time communication
    between the frontend and agent system.
    """
    
    def __init__(self, mas: MultiAgentSystem):
        self.mas = mas
        self.active_sessions: Dict[str, Dict] = {}
        self.conversation_agents: Dict[str, str] = {}  # session_id -> agent_id
        
        # Initialize default agents
        self._initialize_platform_agents()
    
    def _initialize_platform_agents(self):
        """Initialize agents for platform use."""
        # Create conversation coordinator
        conversation_coordinator = CoordinatorAgent("conversation_coordinator")
        self.mas.add_agent(conversation_coordinator)
        
        # Create specialized agents for different capabilities
        research_agent = SpecialistAgent("web_researcher", "research")
        analysis_agent = SpecialistAgent("data_analyst", "analysis") 
        synthesis_agent = SpecialistAgent("content_synthesizer", "synthesis")
        
        self.mas.add_agent(research_agent)
        self.mas.add_agent(analysis_agent)
        self.mas.add_agent(synthesis_agent)
    
    def create_conversation_session(self, session_id: str, name: str, last_name: str, summary: str, resume: str) -> Dict:
        """Create a new conversation session that enhances YOUR existing Assistant."""
        # Import YOUR existing Assistant class
        from AI_career_assistant.ai_assistant.assistant import Assistant
        
        # Create an enhanced agent that wraps YOUR Assistant
        personal_agent = EnhancedAssistantAgent(
            agent_id=f"assistant_{session_id}",
            role=AgentRole.SPECIALIST,
            your_assistant=Assistant(name, last_name, summary, resume),  # YOUR existing Assistant
            capabilities=["professional_representation", "career_guidance", "tool_usage", "autonomous_reasoning"]
        )
        
        # Add memory capabilities to your assistant
        personal_agent.memory_manager = MemoryManager()
        
        self.mas.add_agent(personal_agent)
        self.conversation_agents[session_id] = personal_agent.agent_id
        
        self.active_sessions[session_id] = {
            "start_time": datetime.now(),
            "agent_id": personal_agent.agent_id,
            "conversation_history": [],
            "active_projects": [],
            "assistant_name": f"{name} {last_name}"
        }
        
        return {
            "session_id": session_id,
            "agent_id": personal_agent.agent_id,
            "capabilities": personal_agent.capabilities,
            "status": "ready",
            "assistant_name": f"{name} {last_name}"
        }
    
    def process_conversation_message(self, session_id: str, message: str) -> Dict:
        """Process a conversation message through the agent system."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        agent_id = session["agent_id"]
        
        # Get the personal agent
        personal_agent = self.mas.agents.get(agent_id)
        if not personal_agent:
            return {"error": "Agent not found"}
        
        # Process message with memory
        if hasattr(personal_agent, 'memory_manager'):
            response = personal_agent.process_input(message)
        else:
            # Fallback to simple processing
            response = f"I understand you said: {message}"
        
        # Store in session history
        session["conversation_history"].append({
            "user": message,
            "agent": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "response": response,
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id
        }
    
    def start_collaborative_project(self, session_id: str, project_description: str) -> Dict:
        """Start a collaborative project using multiple agents."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        # Get coordinator
        coordinator = None
        for agent in self.mas.agents.values():
            if agent.role.value == "coordinator":
                coordinator = agent
                break
        
        if not coordinator:
            return {"error": "No coordinator available"}
        
        # Start project
        project_id = coordinator.coordinate_task(project_description)
        
        # Track in session
        session = self.active_sessions[session_id]
        session["active_projects"].append({
            "project_id": project_id,
            "description": project_description,
            "start_time": datetime.now().isoformat(),
            "status": "active"
        })
        
        return {
            "project_id": project_id,
            "description": project_description,
            "status": "started",
            "session_id": session_id
        }
    
    def get_session_status(self, session_id: str) -> Dict:
        """Get comprehensive status of a session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        
        return {
            "session_id": session_id,
            "agent_id": session["agent_id"],
            "start_time": session["start_time"].isoformat(),
            "conversation_length": len(session["conversation_history"]),
            "active_projects": len(session["active_projects"]),
            "projects": session["active_projects"]
        }

# Initialize agent manager
agent_manager = PlatformAgentManager(agent_system)

@app.route('/api/agent/session', methods=['POST'])
def create_agent_session():
    """Create a new agent conversation session."""
    session_id = request.json.get('session_id', f"session_{datetime.now().timestamp()}")
    
    result = agent_manager.create_conversation_session(session_id)
    return jsonify(result)

@app.route('/api/agent/chat', methods=['POST'])
def agent_chat():
    """Send a message to the agent system."""
    data = request.json
    session_id = data.get('session_id')
    message = data.get('message')
    
    if not session_id or not message:
        return jsonify({"error": "Missing session_id or message"}), 400
    
    result = agent_manager.process_conversation_message(session_id, message)
    
    # Emit real-time update via WebSocket
    socketio.emit('agent_response', result, room=session_id)
    
    return jsonify(result)

@app.route('/api/agent/project', methods=['POST'])
def start_agent_project():
    """Start a collaborative agent project."""
    data = request.json
    session_id = data.get('session_id')
    project_description = data.get('description')
    
    if not session_id or not project_description:
        return jsonify({"error": "Missing session_id or description"}), 400
    
    result = agent_manager.start_collaborative_project(session_id, project_description)
    
    # Emit project update
    socketio.emit('project_started', result, room=session_id)
    
    return jsonify(result)

@app.route('/api/agent/status/<session_id>', methods=['GET'])
def get_agent_status(session_id):
    """Get agent session status."""
    result = agent_manager.get_session_status(session_id)
    return jsonify(result)

@app.route('/api/agent/capabilities', methods=['GET'])
def get_agent_capabilities():
    """Get available agent capabilities."""
    capabilities = {
        "available_agents": len(agent_system.agents),
        "agent_types": list(set(agent.role.value for agent in agent_system.agents.values())),
        "capabilities": list(set(
            cap for agent in agent_system.agents.values() 
            for cap in getattr(agent, 'capabilities', [])
        ))
    }
    return jsonify(capabilities)

# WebSocket events for real-time agent interaction
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f"Client connected: {request.sid}")

@socketio.on('join_session')
def handle_join_session(data):
    """Join an agent session for real-time updates."""
    session_id = data.get('session_id')
    if session_id:
        join_room(session_id)
        emit('joined_session', {"session_id": session_id})

@socketio.on('agent_message')
def handle_agent_message(data):
    """Handle real-time agent messages."""
    session_id = data.get('session_id')
    message = data.get('message')
    
    if session_id and message:
        # Process through agent system
        result = agent_manager.process_conversation_message(session_id, message)
        
        # Emit to session room
        emit('agent_response', result, room=session_id)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
```

### React Frontend Enhancement for Agent Integration

```typescript
// AgentInterface.tsx - Advanced React component for agent interaction
import React, { useState, useEffect, useCallback, useRef } from 'react';
import io, { Socket } from 'socket.io-client';

interface Message {
  id: string;
  type: 'user' | 'agent' | 'system';
  content: string;
  timestamp: string;
  agent_id?: string;
}

interface Project {
  project_id: string;
  description: string;
  start_time: string;
  status: string;
}

interface AgentSession {
  session_id: string;
  agent_id: string;
  capabilities: string[];
  status: string;
}

const AgentInterface: React.FC = () => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [session, setSession] = useState<AgentSession | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [agentCapabilities, setAgentCapabilities] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Initialize socket connection
  useEffect(() => {
    const newSocket = io('http://localhost:5000');
    setSocket(newSocket);

    // Set up event listeners
    newSocket.on('agent_response', (data) => {
      addMessage({
        id: Date.now().toString(),
        type: 'agent',
        content: data.response,
        timestamp: data.timestamp,
        agent_id: data.agent_id
      });
      setIsLoading(false);
    });

    newSocket.on('project_started', (data) => {
      setProjects(prev => [...prev, data]);
      addMessage({
        id: Date.now().toString(),
        type: 'system',
        content: `Started project: ${data.description}`,
        timestamp: new Date().toISOString()
      });
    });

    return () => newSocket.close();
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Initialize agent session
  useEffect(() => {
    initializeSession();
    loadAgentCapabilities();
  }, []);

  const initializeSession = async () => {
    try {
      const sessionId = `session_${Date.now()}`;
      const response = await fetch('/api/agent/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
      });

      const sessionData = await response.json();
      setSession(sessionData);

      // Join socket room
      if (socket) {
        socket.emit('join_session', { session_id: sessionId });
      }

      addMessage({
        id: Date.now().toString(),
        type: 'system',
        content: `Agent session initialized. Your personal agent ID: ${sessionData.agent_id}`,
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error('Failed to initialize session:', error);
    }
  };

  const loadAgentCapabilities = async () => {
    try {
      const response = await fetch('/api/agent/capabilities');
      const data = await response.json();
      setAgentCapabilities(data.capabilities);
    } catch (error) {
      console.error('Failed to load capabilities:', error);
    }
  };

  const addMessage = useCallback((message: Message) => {
    setMessages(prev => [...prev, message]);
  }, []);

  const sendMessage = async () => {
    if (!currentMessage.trim() || !session) return;

    // Add user message
    addMessage({
      id: Date.now().toString(),
      type: 'user',
      content: currentMessage,
      timestamp: new Date().toISOString()
    });

    setIsLoading(true);
    setCurrentMessage('');

    try {
      await fetch('/api/agent/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: session.session_id,
          message: currentMessage
        })
      });
    } catch (error) {
      console.error('Failed to send message:', error);
      setIsLoading(false);
    }
  };

  const startProject = async (description: string) => {
    if (!session) return;

    try {
      await fetch('/api/agent/project', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: session.session_id,
          description
        })
      });
    } catch (error) {
      console.error('Failed to start project:', error);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="agent-interface">
      <style jsx>{`
        .agent-interface {
          display: flex;
          flex-direction: column;
          height: 100vh;
          max-width: 1200px;
          margin: 0 auto;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        .header {
          background: rgba(255, 255, 255, 0.1);
          backdrop-filter: blur(10px);
          padding: 1rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .session-info {
          color: white;
          font-size: 0.9rem;
        }
        
        .capabilities {
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
          margin-top: 0.5rem;
        }
        
        .capability-tag {
          background: rgba(255, 255, 255, 0.2);
          color: white;
          padding: 0.2rem 0.5rem;
          border-radius: 1rem;
          font-size: 0.8rem;
        }
        
        .main-content {
          display: flex;
          flex: 1;
          overflow: hidden;
        }
        
        .chat-section {
          flex: 2;
          display: flex;
          flex-direction: column;
        }
        
        .messages {
          flex: 1;
          overflow-y: auto;
          padding: 1rem;
          background: rgba(255, 255, 255, 0.05);
        }
        
        .message {
          margin-bottom: 1rem;
          padding: 0.75rem;
          border-radius: 1rem;
          max-width: 80%;
        }
        
        .message.user {
          background: rgba(255, 255, 255, 0.2);
          color: white;
          margin-left: auto;
          text-align: right;
        }
        
        .message.agent {
          background: rgba(255, 255, 255, 0.9);
          color: #333;
        }
        
        .message.system {
          background: rgba(255, 193, 7, 0.2);
          color: white;
          text-align: center;
          margin: 0 auto;
          font-style: italic;
        }
        
        .message-meta {
          font-size: 0.7rem;
          opacity: 0.7;
          margin-top: 0.25rem;
        }
        
        .input-section {
          padding: 1rem;
          background: rgba(255, 255, 255, 0.1);
          backdrop-filter: blur(10px);
        }
        
        .input-container {
          display: flex;
          gap: 0.5rem;
        }
        
        .message-input {
          flex: 1;
          padding: 0.75rem;
          border: none;
          border-radius: 1rem;
          background: rgba(255, 255, 255, 0.9);
          font-size: 1rem;
        }
        
        .send-button {
          padding: 0.75rem 1.5rem;
          border: none;
          border-radius: 1rem;
          background: #007bff;
          color: white;
          cursor: pointer;
          font-weight: 600;
          transition: all 0.2s;
        }
        
        .send-button:hover {
          background: #0056b3;
          transform: translateY(-1px);
        }
        
        .send-button:disabled {
          background: #6c757d;
          cursor: not-allowed;
          transform: none;
        }
        
        .projects-section {
          flex: 1;
          background: rgba(255, 255, 255, 0.05);
          padding: 1rem;
          border-left: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .projects-header {
          color: white;
          margin-bottom: 1rem;
          font-size: 1.1rem;
          font-weight: 600;
        }
        
        .project-card {
          background: rgba(255, 255, 255, 0.1);
          padding: 1rem;
          border-radius: 0.5rem;
          margin-bottom: 1rem;
          color: white;
        }
        
        .project-title {
          font-weight: 600;
          margin-bottom: 0.5rem;
        }
        
        .project-meta {
          font-size: 0.8rem;
          opacity: 0.8;
        }
        
        .quick-actions {
          margin-top: 1rem;
        }
        
        .action-button {
          display: block;
          width: 100%;
          padding: 0.5rem;
          margin-bottom: 0.5rem;
          border: none;
          border-radius: 0.5rem;
          background: rgba(255, 255, 255, 0.2);
          color: white;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .action-button:hover {
          background: rgba(255, 255, 255, 0.3);
        }
        
        .loading {
          text-align: center;
          color: white;
          font-style: italic;
          padding: 1rem;
        }
      `}</style>

      {/* Header */}
      <div className="header">
        <div className="session-info">
          {session ? (
            <>
              <div>🤖 Agent Session Active - ID: {session.agent_id}</div>
              <div className="capabilities">
                {agentCapabilities.map(cap => (
                  <span key={cap} className="capability-tag">{cap}</span>
                ))}
              </div>
            </>
          ) : (
            <div>🔄 Initializing Agent System...</div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        {/* Chat Section */}
        <div className="chat-section">
          <div className="messages">
            {messages.map(message => (
              <div key={message.id} className={`message ${message.type}`}>
                <div>{message.content}</div>
                <div className="message-meta">
                  {new Date(message.timestamp).toLocaleTimeString()}
                  {message.agent_id && ` - Agent: ${message.agent_id}`}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="loading">🤔 Agent is thinking...</div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-section">
            <div className="input-container">
              <textarea
                className="message-input"
                value={currentMessage}
                onChange={(e) => setCurrentMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask your AI agents anything..."
                rows={2}
                disabled={!session}
              />
              <button
                className="send-button"
                onClick={sendMessage}
                disabled={!currentMessage.trim() || isLoading || !session}
              >
                Send
              </button>
            </div>
          </div>
        </div>

        {/* Projects Section */}
        <div className="projects-section">
          <div className="projects-header">🚀 Active Projects</div>
          
          {projects.map(project => (
            <div key={project.project_id} className="project-card">
              <div className="project-title">{project.description}</div>
              <div className="project-meta">
                Status: {project.status} | Started: {new Date(project.start_time).toLocaleString()}
              </div>
            </div>
          ))}

          <div className="quick-actions">
            <button
              className="action-button"
              onClick={() => startProject("Research AI trends in 2024")}
            >
              🔍 Start Research Project
            </button>
            <button
              className="action-button"
              onClick={() => startProject("Analyze user engagement data")}
            >
              📊 Start Analysis Project
            </button>
            <button
              className="action-button"
              onClick={() => startProject("Create comprehensive report")}
            >
              📝 Start Synthesis Project
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgentInterface;
```

### Production Deployment Configuration

```dockerfile
# Dockerfile - Production-ready containerization
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Start application
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "enhanced_app:app"]
```

```yaml
# docker-compose.yml - Complete production stack
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - SECRET_KEY=${SECRET_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - postgres
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=agents
      - POSTGRES_USER=agent_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - web
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
```

## 🎯 Comprehensive Learning Assessment and Next Steps

### **🏆 What You've Mastered**

After completing this comprehensive tutorial, you've gained expertise in:

#### **1. Mathematical Foundations**
- ✅ **Agent theory**: Formal mathematical definitions and behavior models
- ✅ **Information theory**: Decision entropy and confidence metrics
- ✅ **Cognitive patterns**: Chain-of-Thought, ReAct, Plan-Execute architectures

#### **2. Tool Usage Mastery**
- ✅ **Multi-format tools**: JSON Schema, Python functions, REST APIs, command-line
- ✅ **Security frameworks**: Validation, rate limiting, error handling
- ✅ **Dynamic discovery**: Runtime tool binding and capability advertisement

#### **3. Advanced Architectures**
- ✅ **ReAct agents**: Reasoning + Acting pattern implementation
- ✅ **Planning systems**: Hierarchical task decomposition and execution
- ✅ **Memory management**: Working, episodic, semantic, and procedural memory

#### **4. Multi-Agent Coordination**
- ✅ **Communication protocols**: Message passing and coordination patterns
- ✅ **Role specialization**: Coordinators, specialists, executors, monitors
- ✅ **Collaborative execution**: Team formation and task distribution

#### **5. Production Integration**
- ✅ **Platform enhancement**: Flask backend with agent integration
- ✅ **Real-time interfaces**: WebSocket communication and React frontend
- ✅ **Deployment ready**: Docker, nginx, database integration

### **🚀 Advanced Challenges to Master**

#### **Challenge 1: Custom Agent Architecture**
Create a specialized agent for your domain:
```python
class YourDomainAgent(SpecialistAgent):
    def __init__(self):
        super().__init__("domain_expert", "your_specialization")
        # Add domain-specific tools and reasoning patterns
```

#### **Challenge 2: Advanced Memory Patterns**
Implement episodic memory with emotional weighting:
```python
class EmotionalMemory(Memory):
    def __init__(self, content, emotional_valence: float):
        super().__init__(content, MemoryType.EPISODIC)
        self.emotional_valence = emotional_valence  # -1.0 to 1.0
```

#### **Challenge 3: Multi-Modal Agent Integration**
Extend agents to handle images, audio, and video:
```python
class MultiModalAgent(BaseAgent):
    def process_image(self, image_data: bytes) -> Dict:
        # Implement vision capabilities
    
    def process_audio(self, audio_data: bytes) -> Dict:
        # Implement speech/audio processing
```

### **🎓 Graduation Project: Build Your AI Company**

**Create a complete AI-powered platform that demonstrates mastery:**

1. **Design Multi-Agent Architecture** (25 points)
   - Coordinator, 3+ specialists, memory management
   - Custom communication protocols
   - Real-time collaboration monitoring

2. **Implement Advanced Reasoning** (25 points)
   - ReAct pattern with custom tools
   - Hierarchical planning with dependencies
   - Memory-augmented decision making

3. **Build Production Interface** (25 points)
   - Enhanced Flask backend with agent integration
   - Real-time React frontend with WebSocket communication
   - Authentication and session management

4. **Deploy and Scale** (25 points)
   - Docker containerization
   - Load balancing and scaling
   - Monitoring and error handling

### **🌟 Certification Criteria**

**To earn "LLM Agents Expert" certification:**

- [ ] **Theory Mastery**: Explain agent architectures and mathematical foundations
- [ ] **Implementation Skills**: Build working multi-agent systems from scratch
- [ ] **Tool Integration**: Create custom tools with security and error handling
- [ ] **Platform Development**: Deploy production-ready agent-powered applications
- [ ] **Innovation**: Demonstrate novel agent capabilities or architectures

### **📚 Recommended Next Learning Paths**

#### **1. Advanced AI Research**
- **Reinforcement Learning**: RLHF and agent training
- **Multimodal AI**: Vision, speech, and language integration
- **Reasoning Systems**: Formal logic and symbolic reasoning

#### **2. Production Specialization**
- **MLOps for Agents**: Model deployment and monitoring
- **Agent Security**: Adversarial robustness and safety
- **Scalable Architecture**: Distributed agent systems

#### **3. Domain Applications**
- **Business Automation**: Enterprise agent deployment
- **Creative AI**: Content generation and artistic agents
- **Scientific Research**: Research assistant and discovery agents

---

## 🎯 Final Thoughts: The Future of Autonomous Intelligence

You've now mastered the complete spectrum of LLM agents - from mathematical foundations to production deployment. This knowledge positions you at the forefront of the AI revolution, capable of building systems that think, reason, and act autonomously.

**Key Insights from Your Journey:**

1. **Agents are More Than Language Models**: They're reasoning systems that can plan, remember, and coordinate
2. **Tool Usage is Fundamental**: The ability to interact with external systems makes agents truly powerful
3. **Memory Enables Continuity**: Sophisticated memory management creates persistent, context-aware intelligence
4. **Coordination Enables Emergence**: Multi-agent systems exhibit capabilities beyond individual agents
5. **Production Integration is Critical**: Real-world impact requires seamless platform integration

**The Path Forward:**

As you continue building agent systems, remember that this is still the early days of autonomous AI. The patterns and architectures you've learned will evolve, but the fundamental principles - reasoning, memory, coordination, and tool usage - will remain central to artificial intelligence.

**Your Mission:**
Build agent systems that augment human intelligence, solve real problems, and push the boundaries of what's possible with autonomous AI. The future of intelligence is collaborative, and you're now equipped to lead that collaboration between humans and artificial agents.

**🚀 Go build the future of AI!**

---

*This tutorial represents the complete educational journey from agent theory to production deployment. Continue exploring, building, and innovating in the exciting field of LLM agents!*
    handling authentication, retries, and error recovery.
    """
    
    def __init__(self, name: str, description: str, base_url: str, 
                 method: str = "GET", headers: Optional[Dict] = None):
        super().__init__(name, description, ToolFormat.REST_API)
        self.base_url = base_url
        self.method = method
        self.headers = headers or {}
    
    def get_schema(self) -> Dict:
        """Return API schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string", "description": "API endpoint path"},
                    "params": {"type": "object", "description": "Query parameters"},
                    "data": {"type": "object", "description": "Request body data"}
                },
                "required": ["endpoint"]
            }
        }
    
    def _execute_impl(self, endpoint: str, params: Optional[Dict] = None, 
                      data: Optional[Dict] = None) -> Any:
        """Execute API request."""
        # Simplified implementation - in practice, use requests library
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Mock API response for demonstration
        return {
            "url": url,
            "method": self.method,
            "params": params,
            "data": data,
            "response": "Mock API response"
        }

def create_example_tools():
    """
    Create example tools showcasing different formats.
    
    These tools demonstrate how agents can interact with
    various types of external systems and services.
    """
    
    print("🛠️ Tool Interface Formats")
    print("=" * 40)
    
    # 1. JSON Schema Tool (OpenAI-style)
    def analyze_portfolio(projects: List[str], skills: List[str], goal: str) -> Dict:
        """Analyze portfolio and suggest improvements."""
        return {
            "analysis": f"Analyzed {len(projects)} projects and {len(skills)} skills",
            "recommendation": f"To achieve '{goal}', focus on advanced ML projects",
            "next_steps": ["Add transformer implementation", "Showcase production deployment"]
        }
    
    portfolio_tool = JSONSchemaTool(
        name="analyze_portfolio",
        description="Analyze portfolio and suggest improvements based on career goals",
        parameters={
            "type": "object",
            "properties": {
                "projects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of current projects"
                },
                "skills": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of current skills"
                },
                "goal": {
                    "type": "string",
                    "description": "Career goal or objective"
                }
            },
            "required": ["projects", "skills", "goal"]
        },
        function=analyze_portfolio
    )
    
    # 2. Python Function Tool (direct function)
    def calculate_learning_progress(completed_tutorials: int, total_tutorials: int, 
                                  hours_spent: float) -> Dict:
        """Calculate learning progress metrics."""
        completion_rate = completed_tutorials / total_tutorials if total_tutorials > 0 else 0
        average_time = hours_spent / completed_tutorials if completed_tutorials > 0 else 0
        
        return {
            "completion_percentage": completion_rate * 100,
            "tutorials_remaining": total_tutorials - completed_tutorials,
            "average_hours_per_tutorial": average_time,
            "estimated_time_remaining": average_time * (total_tutorials - completed_tutorials)
        }
    
    learning_tool = PythonFunctionTool(calculate_learning_progress)
    
    # 3. REST API Tool (external service)
    system_monitor = RestAPITool(
        name="monitor_system",
        description="Monitor system performance and resource usage",
        base_url="http://localhost:5000/api",
        method="GET",
        headers={"Authorization": "Bearer token"}
    )
    
    tools = [portfolio_tool, learning_tool, system_monitor]
    
    # Demonstrate tool schemas
    print("📋 Tool Schemas:")
    for tool in tools:
        print(f"\n{tool.name} ({tool.format_type.value}):")
        schema = tool.get_schema()
        print(f"  Description: {schema['description']}")
        if 'parameters' in schema:
            print(f"  Parameters: {len(schema['parameters'].get('properties', {}))} defined")
    
    # Demonstrate tool execution
    print(f"\n⚡ Tool Execution Examples:")
    
    # Execute JSON Schema tool
    result1 = portfolio_tool.execute(
        projects=["Chat Platform", "TinyML System", "Transformer Implementation"],
        skills=["Python", "Keras", "Flask", "React"],
        goal="Senior AI Engineer position"
    )
    print(f"\n{portfolio_tool.name}: {result1.success}")
    if result1.success:
        print(f"  Recommendation: {result1.data['recommendation']}")
    
    # Execute Python function tool
    result2 = learning_tool.execute(
        completed_tutorials=3,
        total_tutorials=8,
        hours_spent=24.5
    )
    print(f"\n{learning_tool.name}: {result2.success}")
    if result2.success:
        print(f"  Progress: {result2.data['completion_percentage']:.1f}%")
        print(f"  Time remaining: {result2.data['estimated_time_remaining']:.1f} hours")
    
    # Execute REST API tool
    result3 = system_monitor.execute(
        endpoint="/status",
        params={"detailed": True}
    )
    print(f"\n{system_monitor.name}: {result3.success}")
    if result3.success:
        print(f"  Response: {result3.data['response']}")
    
    return tools

# Create and demonstrate tools
example_tools = create_example_tools()
```

### Dynamic Tool Discovery and Binding

**Advanced Tool Management:**

```python
class ToolRegistry:
    """
    Advanced tool registry with dynamic discovery and binding.
    
    This allows agents to discover available tools at runtime,
    bind to them dynamically, and manage tool lifecycle.
    """
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.categories: Dict[str, List[str]] = {}
        self.usage_stats: Dict[str, Dict] = {}
    
    def register_tool(self, tool: Tool, category: str = "general"):
        """Register a tool with the registry."""
        self.tools[tool.name] = tool
        
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(tool.name)
        
        self.usage_stats[tool.name] = {
            "registrations": 1,
            "last_used": None,
            "average_execution_time": 0.0
        }
        
        print(f"✅ Registered tool: {tool.name} in category: {category}")
    
    def discover_tools(self, query: str) -> List[Tool]:
        """
        Discover relevant tools based on query.
        
        This uses semantic matching to find tools that might
        be relevant to the agent's current task.
        """
        relevant_tools = []
        query_lower = query.lower()
        
        for tool_name, tool in self.tools.items():
            # Simple keyword matching (in practice, use embeddings)
            if (query_lower in tool.name.lower() or 
                query_lower in tool.description.lower()):
                relevant_tools.append(tool)
        
        # Sort by success rate
        relevant_tools.sort(key=lambda t: t.get_success_rate(), reverse=True)
        
        return relevant_tools
    
    def get_tools_by_category(self, category: str) -> List[Tool]:
        """Get all tools in a specific category."""
        tool_names = self.categories.get(category, [])
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    def get_tool_schema_collection(self) -> List[Dict]:
        """Get schemas for all registered tools."""
        return [tool.get_schema() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name with error handling."""
        if tool_name not in self.tools:
            return ToolResult(
                success=False, 
                data=None, 
                error=f"Tool '{tool_name}' not found"
            )
        
        tool = self.tools[tool_name]
        
        # Update usage statistics
        import time
        start_time = time.time()
        
        result = tool.execute(**kwargs)
        
        execution_time = time.time() - start_time
        self.usage_stats[tool_name]["last_used"] = time.time()
        
        # Update average execution time
        current_avg = self.usage_stats[tool_name]["average_execution_time"]
        usage_count = tool.usage_count
        new_avg = ((current_avg * (usage_count - 1)) + execution_time) / usage_count
        self.usage_stats[tool_name]["average_execution_time"] = new_avg
        
        return result

def demonstrate_dynamic_tool_management():
    """
    Demonstrate dynamic tool discovery and management.
    """
    
    print("\n🔍 Dynamic Tool Management")
    print("=" * 40)
    
    # Create tool registry
    registry = ToolRegistry()
    
    # Register tools from different categories
    
    # Portfolio management tools
    def update_portfolio(project_name: str, description: str, technologies: List[str]) -> Dict:
        """Add or update a project in the portfolio."""
        return {
            "project": project_name,
            "status": "updated",
            "tech_count": len(technologies),
            "last_modified": "2024-01-01"
        }
    
    registry.register_tool(
        PythonFunctionTool(update_portfolio),
        category="portfolio"
    )
    
    # System management tools  
    def check_system_health() -> Dict:
        """Check overall system health and performance."""
        return {
            "cpu_usage": 45.2,
            "memory_usage": 67.8,
            "disk_usage": 23.1,
            "status": "healthy"
        }
    
    registry.register_tool(
        PythonFunctionTool(check_system_health),
        category="system"
    )
    
    # Learning tools
    def track_tutorial_progress(tutorial_name: str, completion_percentage: float) -> Dict:
        """Track progress on a specific tutorial."""
        return {
            "tutorial": tutorial_name,
            "progress": completion_percentage,
            "estimated_completion": "2024-02-15",
            "next_milestone": "Chapter 5"
        }
    
    registry.register_tool(
        PythonFunctionTool(track_tutorial_progress), 
        category="learning"
    )
    
    print(f"📚 Tool Registry Status:")
    print(f"  Total tools: {len(registry.tools)}")
    print(f"  Categories: {list(registry.categories.keys())}")
    print(f"  Tools per category: {[(cat, len(tools)) for cat, tools in registry.categories.items()]}")
    
    # Demonstrate tool discovery
    print(f"\n🔍 Tool Discovery Examples:")
    
    queries = [
        "portfolio management",
        "system monitoring", 
        "tutorial progress",
        "performance optimization"
    ]
    
    for query in queries:
        relevant_tools = registry.discover_tools(query)
        print(f"\nQuery: '{query}'")
        print(f"  Found {len(relevant_tools)} relevant tools:")
        for tool in relevant_tools:
            print(f"    - {tool.name}: {tool.description[:50]}...")
    
    # Demonstrate tool execution through registry
    print(f"\n⚡ Tool Execution Through Registry:")
    
    result = registry.execute_tool(
        "track_tutorial_progress",
        tutorial_name="LLM Agents",
        completion_percentage=75.0
    )
    
    if result.success:
        print(f"✅ Tutorial tracking: {result.data['progress']}% complete")
    else:
        print(f"❌ Error: {result.error}")
    
    # Show usage statistics
    print(f"\n📊 Tool Usage Statistics:")
    for tool_name, stats in registry.usage_stats.items():
        tool = registry.tools[tool_name]
        print(f"  {tool_name}:")
        print(f"    Success rate: {tool.get_success_rate():.2f}")
        print(f"    Avg execution time: {stats['average_execution_time']:.3f}s")

demonstrate_dynamic_tool_management()
```

### Security and Error Handling in Tool Usage

**Production-Ready Tool Execution:**

```python
class SecureToolExecutor:
    """
    Secure tool executor with comprehensive error handling.
    
    This handles security, validation, retries, and monitoring
    for production agent systems.
    """
    
    def __init__(self, max_retries: int = 3, timeout: float = 30.0):
        self.max_retries = max_retries
        self.timeout = timeout
        self.execution_log = []
        self.security_policies = {}
    
    def add_security_policy(self, tool_name: str, policy: Dict):
        """Add security policy for a specific tool."""
        self.security_policies[tool_name] = policy
    
    def validate_parameters(self, tool: Tool, parameters: Dict) -> Tuple[bool, Optional[str]]:
        """Validate tool parameters against schema."""
        schema = tool.get_schema()
        
        if 'parameters' not in schema:
            return True, None
        
        required_params = schema['parameters'].get('required', [])
        provided_params = set(parameters.keys())
        required_set = set(required_params)
        
        missing_params = required_set - provided_params
        if missing_params:
            return False, f"Missing required parameters: {list(missing_params)}"
        
        return True, None
    
    def check_security_policy(self, tool_name: str, parameters: Dict) -> Tuple[bool, Optional[str]]:
        """Check if tool execution is allowed under security policy."""
        if tool_name not in self.security_policies:
            return True, None  # No policy = allowed
        
        policy = self.security_policies[tool_name]
        
        # Check rate limiting
        if 'max_calls_per_minute' in policy:
            recent_calls = len([
                log for log in self.execution_log 
                if (log['tool'] == tool_name and 
                    time.time() - log['timestamp'] < 60)
            ])
            
            if recent_calls >= policy['max_calls_per_minute']:
                return False, "Rate limit exceeded"
        
        # Check parameter restrictions
        if 'forbidden_params' in policy:
            forbidden = set(policy['forbidden_params']) & set(parameters.keys())
            if forbidden:
                return False, f"Forbidden parameters: {list(forbidden)}"
        
        return True, None
    
    def execute_with_retry(self, tool: Tool, parameters: Dict) -> ToolResult:
        """
        Execute tool with retry logic and comprehensive error handling.
        """
        import time
        
        tool_name = tool.name
        start_time = time.time()
        
        # Validate parameters
        valid, error = self.validate_parameters(tool, parameters)
        if not valid:
            return ToolResult(success=False, data=None, error=f"Validation error: {error}")
        
        # Check security policy
        allowed, error = self.check_security_policy(tool_name, parameters)
        if not allowed:
            return ToolResult(success=False, data=None, error=f"Security policy violation: {error}")
        
        # Execute with retries
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                # Add timeout handling in practice
                result = tool.execute(**parameters)
                
                # Log execution
                self.execution_log.append({
                    'tool': tool_name,
                    'timestamp': time.time(),
                    'attempt': attempt + 1,
                    'success': result.success,
                    'execution_time': time.time() - start_time
                })
                
                if result.success:
                    return result
                else:
                    last_error = result.error
                    
            except Exception as e:
                last_error = str(e)
                
            # Wait before retry (exponential backoff)
            if attempt < self.max_retries:
                wait_time = 2 ** attempt  # 1s, 2s, 4s...
                time.sleep(wait_time)
        
        # All retries failed
        return ToolResult(
            success=False,
            data=None,
            error=f"Failed after {self.max_retries + 1} attempts. Last error: {last_error}"
        )
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics for monitoring."""
        if not self.execution_log:
            return {"total_executions": 0}
        
        total_executions = len(self.execution_log)
        successful_executions = sum(1 for log in self.execution_log if log['success'])
        
        execution_times = [log['execution_time'] for log in self.execution_log]
        avg_execution_time = sum(execution_times) / len(execution_times)
        
        tool_usage = {}
        for log in self.execution_log:
            tool_name = log['tool']
            if tool_name not in tool_usage:
                tool_usage[tool_name] = {'count': 0, 'success_count': 0}
            tool_usage[tool_name]['count'] += 1
            if log['success']:
                tool_usage[tool_name]['success_count'] += 1
        
        return {
            'total_executions': total_executions,
            'success_rate': successful_executions / total_executions,
            'average_execution_time': avg_execution_time,
            'tool_usage': tool_usage
        }

def demonstrate_secure_tool_execution():
    """
    Demonstrate secure tool execution with error handling.
    """
    
    print("\n🔒 Secure Tool Execution")
    print("=" * 40)
    
    import time
    
    # Create secure executor
    executor = SecureToolExecutor(max_retries=2, timeout=10.0)
    
    # Create a tool that might fail
    def unreliable_tool(data: str, fail_rate: float = 0.3) -> Dict:
        """Tool that randomly fails to demonstrate retry logic."""
        import random
        if random.random() < fail_rate:
            raise Exception("Random failure for demonstration")
        return {"processed_data": data.upper(), "status": "success"}
    
    tool = PythonFunctionTool(unreliable_tool)
    
    # Add security policy
    executor.add_security_policy("unreliable_tool", {
        'max_calls_per_minute': 5,
        'forbidden_params': ['admin_mode']
    })
    
    print("🛡️ Security Policy Applied:")
    print("  - Max 5 calls per minute")
    print("  - 'admin_mode' parameter forbidden")
    
    # Test normal execution
    print("\n✅ Normal Execution:")
    result1 = executor.execute_with_retry(tool, {"data": "test input", "fail_rate": 0.1})
    print(f"  Success: {result1.success}")
    if result1.success:
        print(f"  Result: {result1.data}")
    
    # Test parameter validation
    print("\n❌ Parameter Validation Test:")
    result2 = executor.execute_with_retry(tool, {})  # Missing required parameter
    print(f"  Success: {result2.success}")
    print(f"  Error: {result2.error}")
    
    # Test security policy violation
    print("\n🚫 Security Policy Test:")
    result3 = executor.execute_with_retry(tool, {
        "data": "test", 
        "admin_mode": True  # Forbidden parameter
    })
    print(f"  Success: {result3.success}")
    print(f"  Error: {result3.error}")
    
    # Test retry logic
    print("\n🔄 Retry Logic Test:")
    result4 = executor.execute_with_retry(tool, {"data": "test", "fail_rate": 0.8})  # High failure rate
    print(f"  Success: {result4.success}")
    if not result4.success:
        print(f"  Error: {result4.error}")
    
    # Show execution statistics
    print("\n📊 Execution Statistics:")
    stats = executor.get_execution_stats()
    print(f"  Total executions: {stats['total_executions']}")
    print(f"  Success rate: {stats['success_rate']:.2f}")
    print(f"  Average execution time: {stats['average_execution_time']:.3f}s")

demonstrate_secure_tool_execution()
```

This comprehensive tool usage foundation provides:

1. **Multiple Interface Formats**: JSON Schema, Python functions, REST APIs
2. **Dynamic Tool Discovery**: Runtime tool finding and binding
3. **Security and Validation**: Production-ready safety measures
4. **Error Handling**: Comprehensive retry and recovery logic
5. **Monitoring**: Usage statistics and performance tracking

In the next chapters, we'll build on this foundation to create sophisticated agent architectures that can reason, plan, and coordinate using these tools. 