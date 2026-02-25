# Tool Integration

## Function Calling APIs

Modern LLMs support native function calling, making tool integration more reliable than text parsing.

### OpenAI Function Calling

```python
import openai

client = openai.OpenAI()

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g., 'San Francisco'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Call LLM with tools
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto"  # Let model decide when to use tools
)

# Check if model wants to call a function
message = response.choices[0].message
if message.tool_calls:
    tool_call = message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    print(f"Calling: {function_name}({arguments})")
```

### Anthropic Tool Use

```python
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }
    }
]

response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "Weather in Paris?"}]
)

# Check for tool use
for block in response.content:
    if block.type == "tool_use":
        print(f"Tool: {block.name}")
        print(f"Input: {block.input}")
```

### Benefits of Native Function Calling

1. **Structured output**: JSON instead of text parsing
2. **Type safety**: Parameters validated by LLM
3. **Reliability**: Less prone to format errors
4. **Parallel calls**: Multiple tools at once

## Tool Schemas and Descriptions

Good tool definitions are critical for agent performance.

### Anatomy of a Tool Schema

```python
{
    "name": "search_database",  # Clear, descriptive name
    "description": "Search the product database for items matching criteria. Returns up to 10 results.",  # When and why to use
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (e.g., 'red shoes size 10')"
            },
            "category": {
                "type": "string",
                "enum": ["electronics", "clothing", "books"],
                "description": "Product category to search within"
            },
            "max_price": {
                "type": "number",
                "description": "Maximum price in USD"
            }
        },
        "required": ["query"]  # Only query is mandatory
    }
}
```

### Writing Effective Descriptions

**Bad**: "Search function"
**Good**: "Search the product database for items. Use when user asks about products, availability, or prices."

**Bad**: "Gets data"
**Good**: "Retrieve user profile data including name, email, and preferences. Use for personalization or account queries."

### Description Best Practices

1. **Be specific**: Explain exactly what the tool does
2. **Include examples**: Show typical parameter values
3. **State limitations**: Mention constraints or edge cases
4. **Clarify use cases**: When should this tool be used?
5. **Avoid ambiguity**: Use precise language

```python
# Good example
{
    "name": "calculate_shipping",
    "description": """Calculate shipping cost for an order.
    
    Use when: User asks about shipping costs or delivery fees
    Returns: Cost in USD and estimated delivery days
    Limitations: Only works for US addresses
    
    Example: calculate_shipping(weight=2.5, zip_code="94102")
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "weight": {
                "type": "number",
                "description": "Package weight in pounds (e.g., 2.5)"
            },
            "zip_code": {
                "type": "string",
                "description": "5-digit US ZIP code (e.g., '94102')"
            }
        },
        "required": ["weight", "zip_code"]
    }
}
```

## Parameter Validation

Always validate parameters before execution.

### Basic Validation

```python
def validate_parameters(tool_name, params):
    """Validate tool parameters"""
    validators = {
        "search": validate_search,
        "calculate": validate_calculate,
        "send_email": validate_email
    }
    
    if tool_name not in validators:
        return False, f"Unknown tool: {tool_name}"
    
    return validators[tool_name](params)

def validate_search(params):
    """Validate search parameters"""
    if "query" not in params:
        return False, "Missing required parameter: query"
    
    if not isinstance(params["query"], str):
        return False, "Query must be a string"
    
    if len(params["query"]) < 2:
        return False, "Query too short (minimum 2 characters)"
    
    if len(params["query"]) > 200:
        return False, "Query too long (maximum 200 characters)"
    
    return True, "Valid"
```

### Type Validation

```python
def validate_type(value, expected_type):
    """Validate parameter type"""
    type_map = {
        "string": str,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict
    }
    
    expected = type_map.get(expected_type)
    if not isinstance(value, expected):
        return False, f"Expected {expected_type}, got {type(value).__name__}"
    
    return True, "Valid"
```

### Schema-Based Validation

```python
import jsonschema

def validate_with_schema(params, schema):
    """Validate parameters against JSON schema"""
    try:
        jsonschema.validate(instance=params, schema=schema)
        return True, "Valid"
    except jsonschema.ValidationError as e:
        return False, str(e)

# Example usage
schema = {
    "type": "object",
    "properties": {
        "email": {
            "type": "string",
            "format": "email"
        },
        "age": {
            "type": "integer",
            "minimum": 0,
            "maximum": 150
        }
    },
    "required": ["email"]
}

valid, message = validate_with_schema(
    {"email": "user@example.com", "age": 25},
    schema
)
```

### Sanitization

Clean inputs before use:

```python
def sanitize_string(s, max_length=1000):
    """Sanitize string input"""
    # Remove null bytes
    s = s.replace('\x00', '')
    
    # Trim whitespace
    s = s.strip()
    
    # Limit length
    s = s[:max_length]
    
    return s

def sanitize_sql_input(s):
    """Prevent SQL injection"""
    # Use parameterized queries instead
    # This is just for demonstration
    dangerous = ["'", '"', ';', '--', '/*', '*/']
    for char in dangerous:
        s = s.replace(char, '')
    return s
```

## Response Parsing

Handle tool outputs consistently.

### Structured Responses

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ToolResponse:
    """Standardized tool response"""
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    metadata: Optional[dict] = None

def execute_tool(tool_name, params):
    """Execute tool and return structured response"""
    try:
        result = TOOLS[tool_name](params)
        return ToolResponse(
            success=True,
            data=result,
            metadata={"tool": tool_name, "timestamp": time.time()}
        )
    except Exception as e:
        return ToolResponse(
            success=False,
            error=str(e),
            metadata={"tool": tool_name}
        )
```

### Formatting for LLM

```python
def format_tool_response(response: ToolResponse) -> str:
    """Format tool response for LLM consumption"""
    if response.success:
        return f"Success: {json.dumps(response.data, indent=2)}"
    else:
        return f"Error: {response.error}"

# Usage in agent loop
result = execute_tool("search", {"query": "AI agents"})
observation = format_tool_response(result)
messages.append({"role": "user", "content": f"Observation: {observation}"})
```

### Handling Different Response Types

```python
def parse_tool_output(output, expected_type="string"):
    """Parse and validate tool output"""
    if expected_type == "json":
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response"}
    
    elif expected_type == "number":
        try:
            return float(output)
        except ValueError:
            return None
    
    elif expected_type == "boolean":
        return output.lower() in ["true", "yes", "1"]
    
    else:  # string
        return str(output)
```

## Building a Tool Registry

Organize tools for easy management.

### Simple Registry

```python
class ToolRegistry:
    """Manage available tools"""
    
    def __init__(self):
        self.tools = {}
    
    def register(self, name, function, schema):
        """Register a new tool"""
        self.tools[name] = {
            "function": function,
            "schema": schema
        }
    
    def get_tool(self, name):
        """Get tool by name"""
        return self.tools.get(name)
    
    def list_tools(self):
        """List all available tools"""
        return list(self.tools.keys())
    
    def get_schemas(self):
        """Get all tool schemas for LLM"""
        return [tool["schema"] for tool in self.tools.values()]
    
    def execute(self, name, params):
        """Execute a tool"""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        
        return tool["function"](params)

# Usage
registry = ToolRegistry()

# Register tools
registry.register(
    name="search",
    function=search_function,
    schema={
        "name": "search",
        "description": "Search the web",
        "parameters": {...}
    }
)

# Use in agent
schemas = registry.get_schemas()
result = registry.execute("search", {"query": "AI"})
```

### Advanced Registry with Decorators

```python
class ToolRegistry:
    def __init__(self):
        self.tools = {}
    
    def tool(self, name, description, parameters):
        """Decorator to register tools"""
        def decorator(func):
            self.tools[name] = {
                "function": func,
                "schema": {
                    "name": name,
                    "description": description,
                    "parameters": parameters
                }
            }
            return func
        return decorator

# Create registry
registry = ToolRegistry()

# Register tools with decorator
@registry.tool(
    name="calculate",
    description="Evaluate mathematical expressions",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string"}
        },
        "required": ["expression"]
    }
)
def calculate(expression):
    """Calculate mathematical expression"""
    return eval(expression)

@registry.tool(
    name="get_time",
    description="Get current time",
    parameters={"type": "object", "properties": {}}
)
def get_time():
    """Get current time"""
    from datetime import datetime
    return datetime.now().isoformat()
```

## Complete Tool Integration Example

```python
import openai
import json
from typing import Dict, Any, List

class Agent:
    """Agent with integrated tool system"""
    
    def __init__(self, model="gpt-4"):
        self.client = openai.OpenAI()
        self.model = model
        self.registry = ToolRegistry()
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register built-in tools"""
        @self.registry.tool(
            name="search",
            description="Search for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        )
        def search(query):
            # Implement search
            return f"Search results for: {query}"
        
        @self.registry.tool(
            name="calculate",
            description="Evaluate math expressions",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        )
        def calculate(expression):
            try:
                return str(eval(expression))
            except Exception as e:
                return f"Error: {str(e)}"
    
    def run(self, user_input: str, max_steps: int = 10) -> str:
        """Run agent with tool integration"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant with access to tools."},
            {"role": "user", "content": user_input}
        ]
        
        for step in range(max_steps):
            # Call LLM with tools
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.registry.get_schemas(),
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            messages.append(message)
            
            # Check if done
            if not message.tool_calls:
                return message.content
            
            # Execute tool calls
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                # Execute tool
                result = self.registry.execute(function_name, arguments)
                
                # Add result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
        
        return "Max steps reached"

# Usage
agent = Agent()
response = agent.run("What is 25 * 17?")
print(response)
```

## Best Practices

1. **Clear naming**: Use descriptive, unambiguous tool names
2. **Comprehensive descriptions**: Help the LLM understand when to use each tool
3. **Validate everything**: Check parameters before execution
4. **Handle errors gracefully**: Return useful error messages
5. **Keep tools focused**: One tool, one purpose
6. **Document examples**: Show typical usage in descriptions
7. **Version your tools**: Track changes to tool interfaces
8. **Test thoroughly**: Verify tools work with various inputs

## Common Patterns

### Conditional Tool Access

```python
def get_available_tools(user_role):
    """Return tools based on user permissions"""
    base_tools = ["search", "calculate"]
    
    if user_role == "admin":
        base_tools.extend(["delete_data", "modify_settings"])
    
    return [registry.get_tool(name) for name in base_tools]
```

### Tool Chaining

```python
# Tools can call other tools
@registry.tool(name="research", ...)
def research(topic):
    # Search for information
    results = registry.execute("search", {"query": topic})
    
    # Summarize results
    summary = registry.execute("summarize", {"text": results})
    
    return summary
```

### Async Tool Execution

```python
import asyncio

async def execute_tool_async(tool_name, params):
    """Execute tool asynchronously"""
    tool = registry.get_tool(tool_name)
    return await tool["function"](params)

# Execute multiple tools in parallel
results = await asyncio.gather(
    execute_tool_async("search", {"query": "AI"}),
    execute_tool_async("search", {"query": "ML"}),
    execute_tool_async("search", {"query": "agents"})
)
```

## Next Steps

Now that you understand tool integration, let's build a complete hands-on project in the next section where you'll create a research assistant agent with multiple tools!
