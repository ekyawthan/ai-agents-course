# Agentic Frameworks

## Introduction to Agent Frameworks

Frameworks provide pre-built components, patterns, and tools for building agents faster and more reliably. They handle common challenges so you can focus on your specific use case.

### Why Use Frameworks?

**Benefits**:
- Faster development
- Battle-tested patterns
- Community support
- Built-in best practices
- Easier maintenance
- Rich ecosystem

**Trade-offs**:
- Learning curve
- Framework lock-in
- Less control
- Overhead
- Version dependencies

### Popular Frameworks

1. **LangChain**: Comprehensive, modular
2. **LangGraph**: State machines for agents
3. **AutoGPT**: Autonomous agents
4. **CrewAI**: Multi-agent collaboration
5. **AutoGen**: Conversational agents

## LangChain and LangGraph

### LangChain Basics

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory

class LangChainAgent:
    """Agent built with LangChain"""
    
    def __init__(self):
        self.llm = OpenAI(temperature=0.7)
        self.memory = ConversationBufferMemory()
        self.tools = self._create_tools()
        self.agent = self._create_agent()
    
    def _create_tools(self) -> List[Tool]:
        """Create agent tools"""
        
        def search_tool(query: str) -> str:
            """Search for information"""
            return f"Search results for: {query}"
        
        def calculator_tool(expression: str) -> str:
            """Calculate mathematical expression"""
            try:
                return str(eval(expression))
            except:
                return "Error in calculation"
        
        tools = [
            Tool(
                name="Search",
                func=search_tool,
                description="Search for information. Input should be a search query."
            ),
            Tool(
                name="Calculator",
                func=calculator_tool,
                description="Calculate mathematical expressions. Input should be a math expression."
            )
        ]
        
        return tools
    
    def _create_agent(self):
        """Create ReAct agent"""
        
        prompt = PromptTemplate.from_template("""
Answer the following question using available tools.

Tools:
{tools}

Question: {input}

{agent_scratchpad}
""")
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=5
        )
        
        return agent_executor
    
    def run(self, query: str) -> str:
        """Run agent"""
        result = self.agent.invoke({"input": query})
        return result["output"]

# Usage
agent = LangChainAgent()
response = agent.run("What is 25 * 17?")
print(response)
```

### LangChain Chains

```python
from langchain.chains import SequentialChain, TransformChain
from langchain.chains.llm import LLMChain

class ChainedAgent:
    """Agent using LangChain chains"""
    
    def __init__(self):
        self.llm = OpenAI(temperature=0.5)
    
    def create_research_chain(self):
        """Create multi-step research chain"""
        
        # Step 1: Generate search queries
        query_prompt = PromptTemplate(
            input_variables=["topic"],
            template="Generate 3 search queries to research: {topic}\n\nQueries:"
        )
        query_chain = LLMChain(llm=self.llm, prompt=query_prompt, output_key="queries")
        
        # Step 2: Search (simplified)
        def search_transform(inputs: dict) -> dict:
            queries = inputs["queries"].split('\n')
            results = [f"Results for: {q}" for q in queries if q.strip()]
            return {"search_results": "\n".join(results)}
        
        search_chain = TransformChain(
            input_variables=["queries"],
            output_variables=["search_results"],
            transform=search_transform
        )
        
        # Step 3: Synthesize
        synthesis_prompt = PromptTemplate(
            input_variables=["topic", "search_results"],
            template="""Synthesize information about {topic} from these results:

{search_results}

Summary:"""
        )
        synthesis_chain = LLMChain(llm=self.llm, prompt=synthesis_prompt, output_key="summary")
        
        # Combine into sequential chain
        overall_chain = SequentialChain(
            chains=[query_chain, search_chain, synthesis_chain],
            input_variables=["topic"],
            output_variables=["summary"],
            verbose=True
        )
        
        return overall_chain
    
    def research(self, topic: str) -> str:
        """Conduct research using chain"""
        chain = self.create_research_chain()
        result = chain({"topic": topic})
        return result["summary"]

# Usage
chained_agent = ChainedAgent()
summary = chained_agent.research("AI agent architectures")
print(summary)
```

### LangGraph State Machines

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    """State for agent"""
    messages: Annotated[list, operator.add]
    current_step: str
    data: dict

class LangGraphAgent:
    """Agent using LangGraph state machine"""
    
    def __init__(self):
        self.llm = OpenAI()
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build state machine graph"""
        
        workflow = StateGraph(AgentState)
        
        # Define nodes (states)
        workflow.add_node("start", self.start_node)
        workflow.add_node("research", self.research_node)
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("respond", self.respond_node)
        
        # Define edges (transitions)
        workflow.set_entry_point("start")
        workflow.add_edge("start", "research")
        workflow.add_edge("research", "analyze")
        workflow.add_edge("analyze", "respond")
        workflow.add_edge("respond", END)
        
        return workflow.compile()
    
    def start_node(self, state: AgentState) -> AgentState:
        """Initial state"""
        print("📍 Starting...")
        state["current_step"] = "start"
        return state
    
    def research_node(self, state: AgentState) -> AgentState:
        """Research state"""
        print("🔍 Researching...")
        
        # Simulate research
        query = state["messages"][-1] if state["messages"] else ""
        state["data"]["research_results"] = f"Research results for: {query}"
        state["current_step"] = "research"
        
        return state
    
    def analyze_node(self, state: AgentState) -> AgentState:
        """Analysis state"""
        print("📊 Analyzing...")
        
        results = state["data"].get("research_results", "")
        state["data"]["analysis"] = f"Analysis of: {results}"
        state["current_step"] = "analyze"
        
        return state
    
    def respond_node(self, state: AgentState) -> AgentState:
        """Response state"""
        print("💬 Responding...")
        
        analysis = state["data"].get("analysis", "")
        response = f"Based on analysis: {analysis}"
        state["messages"].append(response)
        state["current_step"] = "respond"
        
        return state
    
    def run(self, query: str) -> str:
        """Run agent through state machine"""
        
        initial_state = {
            "messages": [query],
            "current_step": "init",
            "data": {}
        }
        
        final_state = self.graph.invoke(initial_state)
        
        return final_state["messages"][-1]

# Usage
langgraph_agent = LangGraphAgent()
response = langgraph_agent.run("Explain quantum computing")
print(response)
```

## AutoGPT and BabyAGI

### AutoGPT Pattern

```python
class AutoGPTAgent:
    """Autonomous agent inspired by AutoGPT"""
    
    def __init__(self, objective: str):
        self.objective = objective
        self.client = openai.OpenAI()
        self.task_list = []
        self.completed_tasks = []
        self.memory = []
    
    def run(self, max_iterations: int = 10):
        """Run autonomous agent"""
        
        print(f"🎯 Objective: {self.objective}\n")
        
        # Generate initial tasks
        self.task_list = self.generate_tasks(self.objective)
        
        for iteration in range(max_iterations):
            if not self.task_list:
                print("✅ All tasks completed!")
                break
            
            # Get next task
            current_task = self.task_list.pop(0)
            print(f"\n📋 Task {iteration + 1}: {current_task}")
            
            # Execute task
            result = self.execute_task(current_task)
            print(f"✓ Result: {result[:200]}...")
            
            # Store in memory
            self.memory.append({
                "task": current_task,
                "result": result
            })
            self.completed_tasks.append(current_task)
            
            # Generate new tasks based on result
            new_tasks = self.generate_new_tasks(current_task, result)
            self.task_list.extend(new_tasks)
            
            # Prioritize tasks
            self.task_list = self.prioritize_tasks(self.task_list)
        
        return self.summarize_results()
    
    def generate_tasks(self, objective: str) -> List[str]:
        """Generate initial task list"""
        
        prompt = f"""Given this objective: {objective}

Break it down into 3-5 specific, actionable tasks.
List them in order of execution.

Tasks:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        tasks_text = response.choices[0].message.content
        tasks = [t.strip('0123456789.- ').strip() for t in tasks_text.split('\n') if t.strip()]
        
        return tasks
    
    def execute_task(self, task: str) -> str:
        """Execute a single task"""
        
        # Build context from memory
        context = self.build_context()
        
        prompt = f"""Objective: {self.objective}

Previous tasks completed:
{context}

Current task: {task}

Execute this task and provide the result:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def generate_new_tasks(self, completed_task: str, result: str) -> List[str]:
        """Generate new tasks based on result"""
        
        prompt = f"""Objective: {self.objective}

Completed task: {completed_task}
Result: {result}

Based on this result, what new tasks (if any) should be added?
Only suggest tasks that help achieve the objective.

New tasks (or "none"):"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        tasks_text = response.choices[0].message.content
        
        if "none" in tasks_text.lower():
            return []
        
        tasks = [t.strip('0123456789.- ').strip() for t in tasks_text.split('\n') if t.strip()]
        return tasks
    
    def prioritize_tasks(self, tasks: List[str]) -> List[str]:
        """Prioritize task list"""
        
        if not tasks:
            return []
        
        prompt = f"""Objective: {self.objective}

Tasks to prioritize:
{chr(10).join([f"{i+1}. {t}" for i, t in enumerate(tasks)])}

Reorder these tasks by priority (most important first).
Return just the task list in order.

Prioritized tasks:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        prioritized_text = response.choices[0].message.content
        prioritized = [t.strip('0123456789.- ').strip() for t in prioritized_text.split('\n') if t.strip()]
        
        return prioritized
    
    def build_context(self) -> str:
        """Build context from memory"""
        if not self.memory:
            return "None"
        
        context = []
        for item in self.memory[-5:]:  # Last 5 tasks
            context.append(f"- {item['task']}: {item['result'][:100]}...")
        
        return "\n".join(context)
    
    def summarize_results(self) -> str:
        """Summarize all results"""
        
        prompt = f"""Objective: {self.objective}

Completed tasks and results:
{chr(10).join([f"{i+1}. {m['task']}: {m['result']}" for i, m in enumerate(self.memory)])}

Provide a comprehensive summary of what was accomplished:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        return response.choices[0].message.content

# Usage
autogpt = AutoGPTAgent("Research and summarize the top 3 AI agent frameworks")
summary = autogpt.run(max_iterations=5)
print(f"\n📝 Final Summary:\n{summary}")
```

## CrewAI and AutoGen

### Multi-Agent Collaboration

```python
class Agent:
    """Individual agent in crew"""
    
    def __init__(self, role: str, goal: str, backstory: str):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.client = openai.OpenAI()
    
    def execute_task(self, task: str, context: str = "") -> str:
        """Execute task as this agent"""
        
        prompt = f"""You are a {self.role}.

Your goal: {self.goal}

Background: {self.backstory}

{f"Context: {context}" if context else ""}

Task: {task}

Response:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content

class Crew:
    """Crew of collaborating agents"""
    
    def __init__(self):
        self.agents = []
        self.tasks = []
    
    def add_agent(self, agent: Agent):
        """Add agent to crew"""
        self.agents.append(agent)
        print(f"👤 Added agent: {agent.role}")
    
    def add_task(self, description: str, agent_role: str, dependencies: List[str] = None):
        """Add task to crew"""
        self.tasks.append({
            "description": description,
            "agent_role": agent_role,
            "dependencies": dependencies or [],
            "status": "pending",
            "result": None
        })
    
    def run(self) -> Dict:
        """Execute all tasks with crew"""
        
        print("\n🚀 Starting crew execution\n")
        
        completed = set()
        
        while len(completed) < len(self.tasks):
            # Find ready tasks
            ready_tasks = [
                task for task in self.tasks
                if task["status"] == "pending" and
                all(dep in completed for dep in task["dependencies"])
            ]
            
            if not ready_tasks:
                break
            
            # Execute ready tasks
            for task in ready_tasks:
                # Find agent
                agent = next((a for a in self.agents if a.role == task["agent_role"]), None)
                
                if not agent:
                    print(f"⚠️  No agent found for role: {task['agent_role']}")
                    task["status"] = "failed"
                    continue
                
                # Build context from dependencies
                context = self.build_context(task["dependencies"])
                
                # Execute
                print(f"▶️  {agent.role}: {task['description']}")
                result = agent.execute_task(task["description"], context)
                
                task["result"] = result
                task["status"] = "completed"
                completed.add(task["description"])
                
                print(f"✓ Completed\n")
        
        return self.generate_report()
    
    def build_context(self, dependencies: List[str]) -> str:
        """Build context from completed dependencies"""
        context_parts = []
        
        for dep in dependencies:
            dep_task = next((t for t in self.tasks if t["description"] == dep), None)
            if dep_task and dep_task["result"]:
                context_parts.append(f"{dep}: {dep_task['result'][:200]}...")
        
        return "\n\n".join(context_parts)
    
    def generate_report(self) -> Dict:
        """Generate execution report"""
        completed = sum(1 for t in self.tasks if t["status"] == "completed")
        
        return {
            "total_tasks": len(self.tasks),
            "completed": completed,
            "failed": len(self.tasks) - completed,
            "tasks": self.tasks
        }

# Usage
crew = Crew()

# Add agents
researcher = Agent(
    role="Researcher",
    goal="Find and analyze information",
    backstory="Expert researcher with deep analytical skills"
)

writer = Agent(
    role="Writer",
    goal="Create clear, engaging content",
    backstory="Professional writer skilled at explaining complex topics"
)

reviewer = Agent(
    role="Reviewer",
    goal="Ensure quality and accuracy",
    backstory="Detail-oriented reviewer with high standards"
)

crew.add_agent(researcher)
crew.add_agent(writer)
crew.add_agent(reviewer)

# Add tasks
crew.add_task(
    "Research the top 3 AI agent frameworks",
    "Researcher"
)

crew.add_task(
    "Write a comparison article based on the research",
    "Writer",
    dependencies=["Research the top 3 AI agent frameworks"]
)

crew.add_task(
    "Review the article for accuracy and clarity",
    "Reviewer",
    dependencies=["Write a comparison article based on the research"]
)

# Execute
report = crew.run()
print(f"\n📊 Report: {report['completed']}/{report['total_tasks']} tasks completed")
```

## Custom Framework Design

### Building Your Own Framework

```python
class CustomAgentFramework:
    """Custom agent framework"""
    
    def __init__(self):
        self.agents = {}
        self.tools = {}
        self.memory = {}
        self.middleware = []
    
    def register_agent(self, name: str, agent_class):
        """Register agent type"""
        self.agents[name] = agent_class
        print(f"✅ Registered agent: {name}")
    
    def register_tool(self, name: str, tool_func):
        """Register tool"""
        self.tools[name] = tool_func
        print(f"🔧 Registered tool: {name}")
    
    def add_middleware(self, middleware_func):
        """Add middleware for request processing"""
        self.middleware.append(middleware_func)
    
    def create_agent(self, agent_type: str, **kwargs):
        """Create agent instance"""
        if agent_type not in self.agents:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_class = self.agents[agent_type]
        agent = agent_class(framework=self, **kwargs)
        
        return agent
    
    def execute_tool(self, tool_name: str, **params):
        """Execute tool"""
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        return self.tools[tool_name](**params)
    
    def process_request(self, agent, request: str) -> str:
        """Process request through middleware"""
        
        # Apply middleware
        for middleware in self.middleware:
            request = middleware(request)
        
        # Execute agent
        response = agent.process(request)
        
        return response

# Usage
framework = CustomAgentFramework()

# Register components
framework.register_tool("search", lambda query: f"Results for: {query}")
framework.register_tool("calculate", lambda expr: str(eval(expr)))

# Add middleware
def logging_middleware(request):
    print(f"📝 Request: {request}")
    return request

framework.add_middleware(logging_middleware)

# Create and use agent
# agent = framework.create_agent("research_agent")
# response = framework.process_request(agent, "Find information about AI")
```

## Best Practices

1. **Choose right framework**: Match to your needs
2. **Start simple**: Don't over-engineer
3. **Understand abstractions**: Know what framework does
4. **Customize carefully**: Extend, don't fight framework
5. **Keep updated**: Follow framework updates
6. **Test thoroughly**: Framework bugs affect you
7. **Monitor performance**: Track overhead
8. **Document usage**: Help team understand
9. **Plan migration**: Have exit strategy
10. **Contribute back**: Share improvements

## Next Steps

Chapter 7 (Advanced Topics) is complete! You now have deep knowledge of agent learning, multimodal capabilities, and frameworks. This prepares you for enterprise-scale deployments in Module 8.
