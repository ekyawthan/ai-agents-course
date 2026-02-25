# Code Execution

## Why Agents Need Code Execution

Code execution allows agents to:
- Perform precise calculations
- Process data programmatically
- Generate and test code
- Automate complex operations
- Verify results deterministically

**Without code execution**: "The sum of 1 to 100 is approximately 5050"
**With code execution**: "The sum of 1 to 100 is exactly 5050" (calculated)

## Sandboxed Environments

Never execute untrusted code directly. Always use sandboxing.

### Why Sandboxing?

**Risks of unsandboxed execution**:
- File system access (delete files)
- Network access (data exfiltration)
- System commands (malicious operations)
- Resource exhaustion (infinite loops)

### Docker Sandbox

```python
import docker
import tempfile

class DockerSandbox:
    """Execute code in Docker container"""
    
    def __init__(self, image="python:3.11-slim"):
        self.client = docker.from_env()
        self.image = image
    
    def execute(self, code: str, timeout: int = 30) -> dict:
        """Execute Python code in container"""
        try:
            # Create temporary file with code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                code_file = f.name
            
            # Run container
            container = self.client.containers.run(
                self.image,
                f"python {code_file}",
                detach=True,
                mem_limit="128m",
                network_disabled=True,
                remove=True
            )
            
            # Wait for completion
            result = container.wait(timeout=timeout)
            logs = container.logs().decode('utf-8')
            
            return {
                "success": result['StatusCode'] == 0,
                "output": logs,
                "exit_code": result['StatusCode']
            }
            
        except docker.errors.ContainerError as e:
            return {
                "success": False,
                "output": str(e),
                "exit_code": -1
            }
        except Exception as e:
            return {
                "success": False,
                "output": f"Error: {str(e)}",
                "exit_code": -1
            }
```

### RestrictedPython

```python
from RestrictedPython import compile_restricted, safe_globals
import io
import sys

class RestrictedExecutor:
    """Execute Python with restrictions"""
    
    def __init__(self):
        self.safe_builtins = {
            'print': print,
            'range': range,
            'len': len,
            'sum': sum,
            'max': max,
            'min': min,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'list': list,
            'dict': dict,
            'set': set,
            'str': str,
            'int': int,
            'float': float,
        }
    
    def execute(self, code: str, timeout: int = 5) -> dict:
        """Execute restricted Python code"""
        try:
            # Compile with restrictions
            byte_code = compile_restricted(
                code,
                filename='<inline>',
                mode='exec'
            )
            
            if byte_code.errors:
                return {
                    "success": False,
                    "output": "\n".join(byte_code.errors)
                }
            
            # Capture output
            output_buffer = io.StringIO()
            sys.stdout = output_buffer
            
            # Execute with safe globals
            exec(byte_code, {
                "__builtins__": self.safe_builtins,
                "_print_": print,
                "_getattr_": getattr,
            })
            
            # Restore stdout
            sys.stdout = sys.__stdout__
            
            return {
                "success": True,
                "output": output_buffer.getvalue()
            }
            
        except Exception as e:
            sys.stdout = sys.__stdout__
            return {
                "success": False,
                "output": f"Error: {str(e)}"
            }
```

### E2B Code Interpreter

```python
from e2b import Sandbox

class E2BSandbox:
    """Execute code using E2B"""
    
    def __init__(self):
        self.sandbox = Sandbox()
    
    def execute_python(self, code: str) -> dict:
        """Execute Python code"""
        try:
            execution = self.sandbox.run_code(code)
            
            return {
                "success": not execution.error,
                "output": execution.stdout,
                "error": execution.stderr,
                "logs": execution.logs
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e)
            }
    
    def execute_bash(self, command: str) -> dict:
        """Execute bash command"""
        try:
            result = self.sandbox.process.start_and_wait(command)
            
            return {
                "success": result.exit_code == 0,
                "output": result.stdout,
                "error": result.stderr
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
```

## Code Generation and Validation

### Generate Code

```python
def generate_code(task: str, language: str = "python") -> str:
    """Generate code for a task"""
    prompt = f"""Write {language} code to accomplish this task:

Task: {task}

Requirements:
- Include error handling
- Add comments
- Return result clearly
- Keep it simple and readable

Code:"""
    
    response = llm.generate(prompt, temperature=0.2)
    return extract_code(response)

def extract_code(response: str) -> str:
    """Extract code from markdown"""
    import re
    
    # Look for code blocks
    pattern = r"```(?:python)?\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    return response.strip()
```

### Validate Code

```python
import ast

def validate_python_code(code: str) -> dict:
    """Validate Python code syntax"""
    try:
        ast.parse(code)
        return {
            "valid": True,
            "errors": []
        }
    except SyntaxError as e:
        return {
            "valid": False,
            "errors": [f"Line {e.lineno}: {e.msg}"]
        }

def check_dangerous_operations(code: str) -> dict:
    """Check for dangerous operations"""
    dangerous_patterns = [
        (r'import\s+os', "OS module import"),
        (r'import\s+sys', "System module import"),
        (r'import\s+subprocess', "Subprocess import"),
        (r'open\s*\(', "File operations"),
        (r'eval\s*\(', "Eval usage"),
        (r'exec\s*\(', "Exec usage"),
        (r'__import__', "Dynamic imports"),
    ]
    
    issues = []
    for pattern, description in dangerous_patterns:
        if re.search(pattern, code):
            issues.append(description)
    
    return {
        "safe": len(issues) == 0,
        "issues": issues
    }
```

### Test Generated Code

```python
def test_code(code: str, test_cases: List[dict]) -> dict:
    """Test code with test cases"""
    sandbox = RestrictedExecutor()
    results = []
    
    for test in test_cases:
        # Prepare test code
        test_code = f"""
{code}

# Test case
result = {test['call']}
print(result)
"""
        
        # Execute
        output = sandbox.execute(test_code)
        
        # Check result
        expected = str(test['expected'])
        actual = output['output'].strip()
        
        results.append({
            "test": test['call'],
            "expected": expected,
            "actual": actual,
            "passed": actual == expected
        })
    
    return {
        "total": len(results),
        "passed": sum(1 for r in results if r['passed']),
        "results": results
    }

# Example usage
code = """
def add(a, b):
    return a + b
"""

test_cases = [
    {"call": "add(2, 3)", "expected": 5},
    {"call": "add(-1, 1)", "expected": 0},
    {"call": "add(0, 0)", "expected": 0}
]

results = test_code(code, test_cases)
```

## Debugging and Error Recovery

### Parse Errors

```python
def parse_error(error_message: str) -> dict:
    """Parse error message for useful info"""
    import re
    
    # Extract line number
    line_match = re.search(r'line (\d+)', error_message)
    line_num = int(line_match.group(1)) if line_match else None
    
    # Extract error type
    type_match = re.search(r'(\w+Error):', error_message)
    error_type = type_match.group(1) if type_match else "Unknown"
    
    return {
        "type": error_type,
        "line": line_num,
        "message": error_message
    }
```

### Auto-Fix Errors

```python
def fix_code_error(code: str, error: str) -> str:
    """Attempt to fix code based on error"""
    prompt = f"""This code has an error:

Code:
```python
{code}
```

Error:
{error}

Provide the corrected code:"""
    
    response = llm.generate(prompt, temperature=0.1)
    return extract_code(response)

def iterative_fix(code: str, max_attempts: int = 3) -> dict:
    """Iteratively fix code until it works"""
    sandbox = RestrictedExecutor()
    
    for attempt in range(max_attempts):
        # Try to execute
        result = sandbox.execute(code)
        
        if result['success']:
            return {
                "success": True,
                "code": code,
                "attempts": attempt + 1
            }
        
        # Try to fix
        code = fix_code_error(code, result['output'])
    
    return {
        "success": False,
        "code": code,
        "attempts": max_attempts,
        "error": "Max attempts reached"
    }
```

## Security Considerations

### Input Validation

```python
def validate_code_input(code: str) -> dict:
    """Validate code before execution"""
    
    # Check length
    if len(code) > 10000:
        return {
            "valid": False,
            "reason": "Code too long (max 10000 chars)"
        }
    
    # Check for null bytes
    if '\x00' in code:
        return {
            "valid": False,
            "reason": "Invalid characters in code"
        }
    
    # Check syntax
    syntax_check = validate_python_code(code)
    if not syntax_check['valid']:
        return {
            "valid": False,
            "reason": f"Syntax error: {syntax_check['errors']}"
        }
    
    # Check for dangerous operations
    safety_check = check_dangerous_operations(code)
    if not safety_check['safe']:
        return {
            "valid": False,
            "reason": f"Unsafe operations: {safety_check['issues']}"
        }
    
    return {"valid": True}
```

### Resource Limits

```python
class ResourceLimitedExecutor:
    """Execute code with resource limits"""
    
    def __init__(self):
        self.max_execution_time = 30  # seconds
        self.max_memory = 128 * 1024 * 1024  # 128 MB
        self.max_output_size = 10000  # characters
    
    def execute(self, code: str) -> dict:
        """Execute with limits"""
        import signal
        import resource
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Execution timeout")
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.max_execution_time)
        
        # Set memory limit
        resource.setrlimit(
            resource.RLIMIT_AS,
            (self.max_memory, self.max_memory)
        )
        
        try:
            # Execute code
            result = self._execute_code(code)
            
            # Limit output size
            if len(result['output']) > self.max_output_size:
                result['output'] = result['output'][:self.max_output_size] + "...(truncated)"
            
            return result
            
        except TimeoutError:
            return {
                "success": False,
                "output": "Execution timeout"
            }
        except MemoryError:
            return {
                "success": False,
                "output": "Memory limit exceeded"
            }
        finally:
            signal.alarm(0)  # Cancel alarm
```

## Complete Code Execution Agent

```python
class CodeExecutionAgent:
    """Agent that can generate and execute code"""
    
    def __init__(self):
        self.sandbox = RestrictedExecutor()
        self.client = openai.OpenAI()
    
    def solve_with_code(self, problem: str) -> str:
        """Solve problem by generating and executing code"""
        
        # Generate code
        print("💻 Generating code...")
        code = self.generate_solution(problem)
        print(f"Generated:\n{code}\n")
        
        # Validate
        validation = validate_code_input(code)
        if not validation['valid']:
            return f"Invalid code: {validation['reason']}"
        
        # Execute
        print("▶️  Executing code...")
        result = self.sandbox.execute(code)
        
        if result['success']:
            print(f"✓ Output: {result['output']}\n")
            return self.format_result(problem, code, result['output'])
        else:
            # Try to fix and retry
            print("⚠️  Error occurred, attempting fix...")
            fixed = iterative_fix(code)
            
            if fixed['success']:
                result = self.sandbox.execute(fixed['code'])
                return self.format_result(problem, fixed['code'], result['output'])
            else:
                return f"Failed to execute: {result['output']}"
    
    def generate_solution(self, problem: str) -> str:
        """Generate code to solve problem"""
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": f"""Write Python code to solve this problem:

{problem}

Requirements:
- Use only standard library
- Print the final result
- Handle edge cases
- Keep it simple

Provide only the code, no explanations."""
            }],
            temperature=0.2
        )
        
        return extract_code(response.choices[0].message.content)
    
    def format_result(self, problem: str, code: str, output: str) -> str:
        """Format final result"""
        return f"""Problem: {problem}

Solution:
```python
{code}
```

Result: {output}"""

# Usage
agent = CodeExecutionAgent()
result = agent.solve_with_code("Calculate the sum of all prime numbers less than 100")
print(result)
```

## Advanced Use Cases

### Data Analysis

```python
def analyze_data_with_code(data: List[dict], question: str) -> str:
    """Analyze data using generated code"""
    
    # Generate analysis code
    code = f"""
import json

data = {json.dumps(data)}

# Analysis code will be generated here
"""
    
    analysis_code = generate_code(
        f"Analyze this data to answer: {question}\nData structure: {data[0] if data else {}}"
    )
    
    full_code = code + "\n" + analysis_code
    
    # Execute
    sandbox = RestrictedExecutor()
    result = sandbox.execute(full_code)
    
    return result['output']
```

### Mathematical Computation

```python
def compute_math(expression: str) -> str:
    """Safely compute mathematical expression"""
    
    code = f"""
import math

result = {expression}
print(result)
"""
    
    sandbox = RestrictedExecutor()
    result = sandbox.execute(code)
    
    if result['success']:
        return result['output'].strip()
    else:
        return f"Error: {result['output']}"
```

### Code Transformation

```python
def transform_code(code: str, transformation: str) -> str:
    """Transform code (refactor, optimize, etc.)"""
    
    prompt = f"""Transform this code:

Original:
```python
{code}
```

Transformation: {transformation}

Transformed code:"""
    
    response = llm.generate(prompt)
    return extract_code(response)

# Example
original = "for i in range(len(items)): print(items[i])"
transformed = transform_code(original, "Make it more Pythonic")
# Result: "for item in items: print(item)"
```

## Best Practices

1. **Always sandbox**: Never execute untrusted code directly
2. **Set timeouts**: Prevent infinite loops
3. **Limit resources**: Memory, CPU, network
4. **Validate inputs**: Check code before execution
5. **Handle errors gracefully**: Don't crash on bad code
6. **Test generated code**: Verify it works
7. **Log executions**: Track what code runs
8. **Isolate environments**: One execution shouldn't affect others
9. **Clean up**: Remove temporary files and containers
10. **Monitor usage**: Track resource consumption

## Common Pitfalls

### Pitfall 1: Trusting Generated Code
**Problem**: LLM generates code with bugs
**Solution**: Always test and validate

### Pitfall 2: No Timeout
**Problem**: Infinite loops hang the system
**Solution**: Set execution timeouts

### Pitfall 3: Unrestricted Access
**Problem**: Code can access file system
**Solution**: Use proper sandboxing

### Pitfall 4: Poor Error Messages
**Problem**: User doesn't understand what went wrong
**Solution**: Parse and explain errors clearly

## Next Steps

You now understand code execution for agents! Next, we'll explore data access and retrieval, including databases, APIs, and RAG systems.
