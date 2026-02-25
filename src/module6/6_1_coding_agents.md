# Coding Agents

## Module 6: Learning Objectives

By the end of this module, you will:
- ✓ Build coding agents that analyze and generate code
- ✓ Create research agents with multi-source verification
- ✓ Implement task automation with workflow orchestration
- ✓ Design specialized agents for specific domains
- ✓ Integrate advanced capabilities into focused agents

---

## Introduction to Coding Agents

Coding agents are specialized AI systems that understand, generate, modify, and debug code. They're among the most powerful and practical agent applications.

### What Makes Coding Agents Special?

**Unique Capabilities**:
- Understand code semantics and structure
- Generate syntactically correct code
- Refactor and optimize existing code
- Debug and fix errors
- Write tests and documentation
- Work across multiple programming languages

**Key Challenges**:
- Code must be syntactically correct
- Logic must be sound
- Must handle edge cases
- Need to understand context and dependencies
- Security vulnerabilities must be avoided

### Types of Coding Agents

1. **Code Generation Agents**: Write new code from specifications
2. **Code Review Agents**: Analyze and suggest improvements
3. **Debugging Agents**: Find and fix bugs
4. **Refactoring Agents**: Improve code structure
5. **Testing Agents**: Generate and run tests
6. **Documentation Agents**: Write comments and docs

## Code Understanding and Generation

### Understanding Code Structure

```python
import ast
from typing import Dict, List, Any

class CodeAnalyzer:
    """Analyze code structure and semantics"""
    
    def __init__(self):
        self.client = openai.OpenAI()
    
    def parse_python_code(self, code: str) -> Dict[str, Any]:
        """Parse Python code into AST"""
        try:
            tree = ast.parse(code)
            
            analysis = {
                "functions": [],
                "classes": [],
                "imports": [],
                "variables": [],
                "complexity": 0
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis["functions"].append({
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "line": node.lineno,
                        "docstring": ast.get_docstring(node)
                    })
                
                elif isinstance(node, ast.ClassDef):
                    methods = [
                        n.name for n in node.body 
                        if isinstance(n, ast.FunctionDef)
                    ]
                    analysis["classes"].append({
                        "name": node.name,
                        "methods": methods,
                        "line": node.lineno,
                        "docstring": ast.get_docstring(node)
                    })
                
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis["imports"].append(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        analysis["imports"].append(f"{module}.{alias.name}")
            
            return analysis
            
        except SyntaxError as e:
            return {
                "error": "Syntax error",
                "message": str(e),
                "line": e.lineno
            }
    
    def analyze_complexity(self, code: str) -> Dict[str, Any]:
        """Analyze code complexity"""
        try:
            tree = ast.parse(code)
            
            complexity = {
                "cyclomatic": 1,  # Base complexity
                "lines_of_code": len(code.split('\n')),
                "num_functions": 0,
                "num_classes": 0,
                "max_nesting": 0
            }
            
            for node in ast.walk(tree):
                # Count decision points for cyclomatic complexity
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity["cyclomatic"] += 1
                
                elif isinstance(node, ast.FunctionDef):
                    complexity["num_functions"] += 1
                
                elif isinstance(node, ast.ClassDef):
                    complexity["num_classes"] += 1
            
            return complexity
            
        except Exception as e:
            return {"error": str(e)}
    
    def extract_dependencies(self, code: str) -> List[str]:
        """Extract external dependencies"""
        try:
            tree = ast.parse(code)
            dependencies = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Get top-level package
                        pkg = alias.name.split('.')[0]
                        dependencies.add(pkg)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        pkg = node.module.split('.')[0]
                        dependencies.add(pkg)
            
            # Filter out standard library
            stdlib = {'os', 'sys', 'json', 're', 'time', 'datetime', 'math'}
            external = dependencies - stdlib
            
            return sorted(external)
            
        except Exception as e:
            return []
    
    def understand_code_intent(self, code: str) -> str:
        """Use LLM to understand what code does"""
        prompt = f"""Analyze this code and explain what it does:

```python
{code}
```

Provide:
1. High-level purpose
2. Key functionality
3. Input/output
4. Any notable patterns or techniques

Explanation:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content

# Usage
analyzer = CodeAnalyzer()

code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def add(self, a, b):
        return a + b
"""

analysis = analyzer.parse_python_code(code)
print(f"Functions: {[f['name'] for f in analysis['functions']]}")
print(f"Classes: {[c['name'] for c in analysis['classes']]}")

complexity = analyzer.analyze_complexity(code)
print(f"Cyclomatic complexity: {complexity['cyclomatic']}")

intent = analyzer.understand_code_intent(code)
print(f"Intent: {intent}")
```

### Generating Code from Specifications

```python
class CodeGenerator:
    """Generate code from natural language specifications"""
    
    def __init__(self):
        self.client = openai.OpenAI()
    
    def generate_function(self, 
                         description: str,
                         language: str = "python",
                         include_tests: bool = False) -> Dict[str, str]:
        """Generate function from description"""
        
        prompt = f"""Generate a {language} function based on this description:

{description}

Requirements:
- Include type hints (if applicable)
- Add docstring with description, parameters, and return value
- Handle edge cases
- Include error handling
- Follow best practices
- Keep it simple and readable

{"Also generate unit tests for this function." if include_tests else ""}

Provide the code:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        code = response.choices[0].message.content
        
        # Extract code and tests
        parts = self.extract_code_blocks(code)
        
        return {
            "code": parts.get("main", code),
            "tests": parts.get("tests", "") if include_tests else None
        }
    
    def generate_class(self,
                      description: str,
                      methods: List[str] = None) -> str:
        """Generate class from description"""
        
        methods_str = ""
        if methods:
            methods_str = f"\nMethods to implement:\n" + "\n".join(f"- {m}" for m in methods)
        
        prompt = f"""Generate a Python class based on this description:

{description}{methods_str}

Requirements:
- Include __init__ method
- Add docstrings for class and methods
- Use type hints
- Follow PEP 8 style guide
- Include example usage in docstring

Provide the code:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        return self.extract_code_blocks(response.choices[0].message.content)["main"]
    
    def generate_from_signature(self, signature: str) -> str:
        """Generate function implementation from signature"""
        
        prompt = f"""Implement this function:

```python
{signature}
    pass
```

Provide a complete, working implementation with:
- Proper logic
- Error handling
- Edge case handling
- Comments for complex parts

Implementation:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        return self.extract_code_blocks(response.choices[0].message.content)["main"]
    
    def extract_code_blocks(self, text: str) -> Dict[str, str]:
        """Extract code blocks from markdown"""
        import re
        
        # Find all code blocks
        pattern = r'```(?:python)?\n(.*?)```'
        blocks = re.findall(pattern, text, re.DOTALL)
        
        if not blocks:
            return {"main": text}
        
        result = {"main": blocks[0]}
        
        if len(blocks) > 1:
            result["tests"] = blocks[1]
        
        return result

# Usage
generator = CodeGenerator()

# Generate function
result = generator.generate_function(
    "Create a function that calculates the factorial of a number",
    include_tests=True
)

print("Generated code:")
print(result["code"])

if result["tests"]:
    print("\nGenerated tests:")
    print(result["tests"])

# Generate class
class_code = generator.generate_class(
    "A simple cache that stores key-value pairs with expiration",
    methods=["set", "get", "delete", "clear"]
)

print("\nGenerated class:")
print(class_code)
```


## Refactoring and Optimization

### Automated Refactoring

```python
class RefactoringAgent:
    """Refactor and improve code quality"""
    
    def __init__(self):
        self.client = openai.OpenAI()
    
    def refactor_for_readability(self, code: str) -> Dict[str, str]:
        """Improve code readability"""
        prompt = f"""Refactor this code for better readability:

```python
{code}
```

Apply these improvements:
- Better variable names
- Extract complex expressions
- Add comments
- Simplify logic
- Follow PEP 8

Provide:
1. Refactored code
2. List of changes made

Response:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return self.parse_response(response.choices[0].message.content)
    
    def optimize_performance(self, code: str) -> Dict[str, str]:
        """Optimize code for performance"""
        prompt = f"""Optimize this code for better performance:

```python
{code}
```

Consider:
- Algorithm complexity
- Data structure choices
- Unnecessary operations
- Caching opportunities
- Memory usage

Provide optimized code with explanation:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return self.parse_response(response.choices[0].message.content)
    
    def apply_design_pattern(self, code: str, pattern: str) -> Dict[str, str]:
        """Apply design pattern to code"""
        prompt = f"""Refactor this code to use the {pattern} design pattern:

```python
{code}
```

Explain:
- Why this pattern is appropriate
- How it improves the code
- What changed

Refactored code:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return self.parse_response(response.choices[0].message.content)
    
    def extract_method(self, code: str, lines: tuple) -> Dict[str, str]:
        """Extract method refactoring"""
        prompt = f"""Extract lines {lines[0]}-{lines[1]} into a separate method:

```python
{code}
```

Provide:
- New method with good name
- Updated original code
- Method signature

Result:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        return self.parse_response(response.choices[0].message.content)

# Usage
refactorer = RefactoringAgent()

# Improve readability
messy_code = """
def f(x,y,z):
    if x>0:
        if y>0:
            if z>0:
                return x+y+z
    return 0
"""

result = refactorer.refactor_for_readability(messy_code)
print("Refactored:", result["code"])

# Optimize performance
slow_code = """
def find_duplicates(items):
    duplicates = []
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates
"""

result = refactorer.optimize_performance(slow_code)
print("Optimized:", result["code"])
```

## Test Generation

### Comprehensive Test Generation

```python
class TestGenerator:
    """Generate comprehensive unit tests"""
    
    def __init__(self):
        self.client = openai.OpenAI()
    
    def generate_unit_tests(self, code: str, framework: str = "pytest") -> str:
        """Generate unit tests with full coverage"""
        prompt = f"""Generate comprehensive {framework} tests for this code:

```python
{code}
```

Include tests for:
1. Normal/happy path cases
2. Edge cases (empty, None, boundaries)
3. Error cases (invalid input, exceptions)
4. Integration scenarios
5. Fixtures and setup if needed

Use descriptive test names and add comments.

Tests:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        return self.extract_code(response.choices[0].message.content)
    
    def generate_property_tests(self, code: str) -> str:
        """Generate property-based tests using Hypothesis"""
        prompt = f"""Generate property-based tests using Hypothesis for:

```python
{code}
```

Create tests that verify properties like:
- Invariants
- Idempotence
- Commutativity
- Round-trip properties

Tests:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        return self.extract_code(response.choices[0].message.content)
    
    def generate_integration_tests(self, code: str, dependencies: List[str]) -> str:
        """Generate integration tests"""
        deps_str = ", ".join(dependencies)
        
        prompt = f"""Generate integration tests for this code that interacts with: {deps_str}

```python
{code}
```

Include:
- Mocking external dependencies
- Testing interactions
- Setup and teardown
- Error scenarios

Tests:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        return self.extract_code(response.choices[0].message.content)
    
    def extract_code(self, text: str) -> str:
        """Extract code from markdown"""
        import re
        pattern = r'```(?:python)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0] if matches else text

# Usage
test_gen = TestGenerator()

code_to_test = """
def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
"""

tests = test_gen.generate_unit_tests(code_to_test)
print("Generated tests:")
print(tests)
```

## Debugging and Error Fixing

### Automated Debugging Agent

```python
class DebuggingAgent:
    """Find and fix bugs in code"""
    
    def __init__(self):
        self.client = openai.OpenAI()
        self.sandbox = CodeExecutor()  # From previous modules
    
    def debug_code(self, code: str, error_message: str = None) -> Dict:
        """Debug code and suggest fixes"""
        
        # Try to execute and capture error if not provided
        if not error_message:
            result = self.sandbox.execute(code)
            if not result["success"]:
                error_message = result["output"]
        
        prompt = f"""Debug this code:

```python
{code}
```

Error: {error_message}

Provide:
1. Root cause analysis
2. Fixed code
3. Explanation of the fix
4. How to prevent similar bugs

Response:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        return self.parse_debug_response(response.choices[0].message.content)
    
    def find_logical_errors(self, code: str, expected_behavior: str) -> Dict:
        """Find logical errors (code runs but wrong output)"""
        prompt = f"""This code runs without errors but produces wrong results:

```python
{code}
```

Expected behavior: {expected_behavior}

Analyze:
1. What's the logical error?
2. Why does it produce wrong results?
3. How to fix it?
4. Test cases to verify the fix

Analysis:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return self.parse_debug_response(response.choices[0].message.content)
    
    def suggest_improvements(self, code: str, issue: str) -> List[str]:
        """Suggest multiple ways to fix an issue"""
        prompt = f"""Suggest 3 different ways to fix this issue:

Code:
```python
{code}
```

Issue: {issue}

For each solution, provide:
- The fix
- Pros and cons
- When to use it

Solutions:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        return self.parse_solutions(response.choices[0].message.content)
    
    def iterative_fix(self, code: str, max_attempts: int = 3) -> Dict:
        """Iteratively fix code until it works"""
        for attempt in range(max_attempts):
            # Try to execute
            result = self.sandbox.execute(code)
            
            if result["success"]:
                return {
                    "success": True,
                    "code": code,
                    "attempts": attempt + 1
                }
            
            # Try to fix
            fix_result = self.debug_code(code, result["output"])
            code = fix_result["fixed_code"]
        
        return {
            "success": False,
            "code": code,
            "attempts": max_attempts,
            "last_error": result["output"]
        }

# Usage
debugger = DebuggingAgent()

buggy_code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

# This will crash on empty list
result = calculate_average([])
"""

fix = debugger.debug_code(buggy_code)
print("Root cause:", fix["root_cause"])
print("Fixed code:", fix["fixed_code"])
```

## Repository-Level Operations

### Codebase Understanding

```python
from pathlib import Path
import json

class CodebaseAgent:
    """Understand and navigate entire codebases"""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.index = {}
        self.dependency_graph = {}
        self.client = openai.OpenAI()
    
    def index_codebase(self):
        """Index all Python files in codebase"""
        print("Indexing codebase...")
        
        for py_file in self.root_path.rglob("*.py"):
            if "venv" in str(py_file) or ".git" in str(py_file):
                continue
            
            try:
                with open(py_file) as f:
                    code = f.read()
                
                analyzer = CodeAnalyzer()
                analysis = analyzer.parse_python_code(code)
                
                self.index[str(py_file.relative_to(self.root_path))] = {
                    "analysis": analysis,
                    "size": len(code),
                    "lines": len(code.split('\n'))
                }
            except Exception as e:
                print(f"Error indexing {py_file}: {e}")
        
        print(f"Indexed {len(self.index)} files")
    
    def find_function_definition(self, function_name: str) -> List[Dict]:
        """Find where a function is defined"""
        results = []
        
        for file_path, data in self.index.items():
            for func in data["analysis"].get("functions", []):
                if func["name"] == function_name:
                    results.append({
                        "file": file_path,
                        "line": func["line"],
                        "signature": f"{func['name']}({', '.join(func['args'])})"
                    })
        
        return results
    
    def find_class_definition(self, class_name: str) -> List[Dict]:
        """Find where a class is defined"""
        results = []
        
        for file_path, data in self.index.items():
            for cls in data["analysis"].get("classes", []):
                if cls["name"] == class_name:
                    results.append({
                        "file": file_path,
                        "line": cls["line"],
                        "methods": cls["methods"]
                    })
        
        return results
    
    def find_usages(self, symbol: str) -> List[Dict]:
        """Find where a symbol is used"""
        usages = []
        
        for py_file in self.root_path.rglob("*.py"):
            if "venv" in str(py_file):
                continue
            
            try:
                with open(py_file) as f:
                    for i, line in enumerate(f, 1):
                        if symbol in line:
                            usages.append({
                                "file": str(py_file.relative_to(self.root_path)),
                                "line": i,
                                "content": line.strip()
                            })
            except:
                pass
        
        return usages
    
    def analyze_dependencies(self):
        """Build dependency graph"""
        for file_path, data in self.index.items():
            imports = data["analysis"].get("imports", [])
            self.dependency_graph[file_path] = imports
    
    def get_codebase_summary(self) -> Dict:
        """Get high-level codebase summary"""
        total_files = len(self.index)
        total_functions = sum(
            len(data["analysis"].get("functions", []))
            for data in self.index.values()
        )
        total_classes = sum(
            len(data["analysis"].get("classes", []))
            for data in self.index.values()
        )
        total_lines = sum(
            data["lines"]
            for data in self.index.values()
        )
        
        return {
            "total_files": total_files,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "total_lines": total_lines,
            "avg_lines_per_file": total_lines / total_files if total_files > 0 else 0
        }
    
    def explain_codebase(self) -> str:
        """Generate high-level explanation of codebase"""
        summary = self.get_codebase_summary()
        
        # Get file structure
        files = list(self.index.keys())
        
        prompt = f"""Explain this codebase structure:

Files: {len(files)}
Functions: {summary['total_functions']}
Classes: {summary['total_classes']}
Lines of code: {summary['total_lines']}

File structure:
{chr(10).join(files[:20])}

Provide:
1. What this codebase likely does
2. Main components/modules
3. Architecture pattern
4. Key areas of functionality

Explanation:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content

# Usage
codebase = CodebaseAgent("./my_project")
codebase.index_codebase()

# Find function
results = codebase.find_function_definition("process_data")
print(f"Found in: {results}")

# Get summary
summary = codebase.get_codebase_summary()
print(f"Codebase: {summary['total_files']} files, {summary['total_lines']} lines")

# Explain codebase
explanation = codebase.explain_codebase()
print(explanation)
```

## Complete Coding Agent System

```python
class CompleteCodingAgent:
    """Full-featured coding agent"""
    
    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.generator = CodeGenerator()
        self.refactorer = RefactoringAgent()
        self.test_gen = TestGenerator()
        self.debugger = DebuggingAgent()
        self.client = openai.OpenAI()
    
    def process_request(self, request: str, code: str = None, context: Dict = None) -> Dict:
        """Process any coding request"""
        
        # Classify intent
        intent = self.classify_intent(request)
        
        if intent == "generate":
            return self.handle_generation(request)
        
        elif intent == "analyze":
            return self.handle_analysis(code)
        
        elif intent == "refactor":
            return self.handle_refactoring(code, request)
        
        elif intent == "test":
            return self.handle_test_generation(code)
        
        elif intent == "debug":
            return self.handle_debugging(code, context)
        
        elif intent == "explain":
            return self.handle_explanation(code)
        
        else:
            return {"error": "Could not understand request"}
    
    def handle_generation(self, request: str) -> Dict:
        """Handle code generation requests"""
        code = self.generator.generate_function(request)
        
        # Validate generated code
        validation = self.analyzer.parse_python_code(code)
        
        if "error" in validation:
            # Try to fix
            fixed = self.debugger.debug_code(code, validation["error"])
            code = fixed["fixed_code"]
        
        # Generate tests
        tests = self.test_gen.generate_unit_tests(code)
        
        return {
            "type": "generation",
            "code": code,
            "tests": tests,
            "validated": True
        }
    
    def handle_analysis(self, code: str) -> Dict:
        """Handle code analysis requests"""
        # Parse structure
        structure = self.analyzer.parse_python_code(code)
        
        # Analyze complexity
        complexity = self.analyzer.analyze_complexity(code)
        
        # Get explanation
        explanation = self.analyzer.understand_code_intent(code)
        
        return {
            "type": "analysis",
            "structure": structure,
            "complexity": complexity,
            "explanation": explanation
        }
    
    def handle_refactoring(self, code: str, request: str) -> Dict:
        """Handle refactoring requests"""
        if "performance" in request.lower():
            result = self.refactorer.optimize_performance(code)
        elif "readable" in request.lower():
            result = self.refactorer.refactor_for_readability(code)
        else:
            result = self.refactorer.refactor_code(code)
        
        return {
            "type": "refactoring",
            "original": code,
            "refactored": result["code"],
            "changes": result.get("changes", [])
        }
    
    def handle_test_generation(self, code: str) -> Dict:
        """Handle test generation requests"""
        unit_tests = self.test_gen.generate_unit_tests(code)
        
        return {
            "type": "tests",
            "code": code,
            "tests": unit_tests
        }
    
    def handle_debugging(self, code: str, context: Dict) -> Dict:
        """Handle debugging requests"""
        error_msg = context.get("error") if context else None
        
        result = self.debugger.debug_code(code, error_msg)
        
        return {
            "type": "debugging",
            "original": code,
            "fixed": result["fixed_code"],
            "explanation": result.get("explanation", "")
        }
    
    def handle_explanation(self, code: str) -> Dict:
        """Handle code explanation requests"""
        explanation = self.analyzer.understand_code_intent(code)
        structure = self.analyzer.parse_python_code(code)
        
        return {
            "type": "explanation",
            "explanation": explanation,
            "structure": structure
        }
    
    def classify_intent(self, request: str) -> str:
        """Classify user intent"""
        request_lower = request.lower()
        
        keywords = {
            "generate": ["generate", "create", "write", "implement"],
            "analyze": ["analyze", "understand", "explain what"],
            "refactor": ["refactor", "improve", "optimize", "clean"],
            "test": ["test", "unittest", "pytest"],
            "debug": ["debug", "fix", "error", "bug"],
            "explain": ["explain", "what does", "how does"]
        }
        
        for intent, words in keywords.items():
            if any(word in request_lower for word in words):
                return intent
        
        return "unknown"

# Usage
agent = CompleteCodingAgent()

# Generate code
result = agent.process_request("Create a function to validate email addresses")
print("Generated code:")
print(result["code"])
print("\nTests:")
print(result["tests"])

# Analyze code
code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

result = agent.process_request("Analyze this code", code=code)
print("\nComplexity:", result["complexity"])
print("Explanation:", result["explanation"])

# Refactor code
result = agent.process_request("Optimize this code for performance", code=code)
print("\nRefactored:")
print(result["refactored"])
```

## Best Practices for Coding Agents

### 1. Code Quality Checks

Always validate generated code:
- Syntax checking (AST parsing)
- Style checking (PEP 8, linting)
- Security scanning (bandit, safety)
- Type checking (mypy)

### 2. Testing Strategy

- Generate tests alongside code
- Run tests automatically
- Achieve high coverage
- Include edge cases

### 3. Context Awareness

- Understand existing codebase
- Match coding style
- Respect conventions
- Consider dependencies

### 4. Iterative Improvement

- Start with simple solution
- Refine based on feedback
- Test incrementally
- Document changes

### 5. Security Considerations

- Validate all inputs
- Avoid SQL injection
- Check for XSS vulnerabilities
- Use secure libraries
- Never expose secrets

### 6. Performance Optimization

- Profile before optimizing
- Choose right algorithms
- Consider memory usage
- Cache when appropriate
- Benchmark improvements

### 7. Documentation

- Generate docstrings
- Add inline comments
- Create README files
- Document APIs
- Explain complex logic

### 8. Version Control

- Commit frequently
- Write clear messages
- Use branches
- Review changes
- Tag releases

### 9. Collaboration

- Follow team standards
- Request code reviews
- Share knowledge
- Document decisions
- Communicate changes

### 10. Continuous Learning

- Learn from mistakes
- Study good code
- Stay updated
- Experiment safely
- Share learnings

## Advanced Topics

### Multi-Language Support

```python
class MultiLanguageAgent:
    """Support multiple programming languages"""
    
    def __init__(self):
        self.client = openai.OpenAI()
        self.supported_languages = ["python", "javascript", "java", "go", "rust"]
    
    def generate_code(self, description: str, language: str) -> str:
        """Generate code in specified language"""
        if language not in self.supported_languages:
            raise ValueError(f"Unsupported language: {language}")
        
        prompt = f"""Generate {language} code for:

{description}

Follow {language} best practices and conventions.

Code:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        return response.choices[0].message.content
    
    def translate_code(self, code: str, from_lang: str, to_lang: str) -> str:
        """Translate code between languages"""
        prompt = f"""Translate this {from_lang} code to {to_lang}:

```{from_lang}
{code}
```

Maintain:
- Same functionality
- Idiomatic {to_lang} style
- Best practices

{to_lang} code:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        return response.choices[0].message.content
```

### Code Review Agent

```python
class CodeReviewAgent:
    """Automated code review"""
    
    def __init__(self):
        self.client = openai.OpenAI()
    
    def review_code(self, code: str) -> Dict:
        """Comprehensive code review"""
        prompt = f"""Review this code:

```python
{code}
```

Provide feedback on:
1. Code quality (readability, maintainability)
2. Potential bugs or issues
3. Performance concerns
4. Security vulnerabilities
5. Best practice violations
6. Suggestions for improvement

Rate each category 1-5 and provide specific feedback.

Review:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return self.parse_review(response.choices[0].message.content)
    
    def suggest_improvements(self, code: str) -> List[Dict]:
        """Suggest specific improvements"""
        review = self.review_code(code)
        
        improvements = []
        for issue in review.get("issues", []):
            improvements.append({
                "issue": issue,
                "suggestion": self.generate_fix(code, issue),
                "priority": self.assess_priority(issue)
            })
        
        return improvements
```

## Next Steps

You now have comprehensive knowledge of coding agents! Next, we'll explore research agents that gather and synthesize information from multiple sources.
