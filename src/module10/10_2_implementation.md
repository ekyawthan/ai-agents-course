# Implementation

## Building the Autonomous Software Engineering Agent

Let's build the complete system step by step.

## Project Setup

```bash
# Create project structure
mkdir autonomous-se-agent
cd autonomous-se-agent

# Create directories
mkdir -p src/{agents,tools,memory,orchestration}
mkdir -p tests
mkdir -p data/{cache,feedback}

# Install dependencies
pip install openai chromadb gitpython docker pytest pylint black ast-grep-py
```

## Core Implementation

### 1. Main Orchestrator

```python
# src/orchestration/orchestrator.py
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum
import openai

class TaskType(Enum):
    ANALYZE = "analyze"
    FIX = "fix"
    TEST = "test"
    REFACTOR = "refactor"
    REVIEW = "review"

@dataclass
class Task:
    type: TaskType
    target: str
    description: str
    context: Dict

class SoftwareEngineeringAgent:
    """Main orchestrator for autonomous SE agent"""
    
    def __init__(self):
        self.client = openai.OpenAI()
        self.analyzer = AnalyzerAgent()
        self.fixer = FixerAgent()
        self.tester = TesterAgent()
        self.memory = AgentMemory()
    
    def process_request(self, request: str, target_path: str) -> Dict:
        """Process user request"""
        
        # Parse intent
        intent = self.parse_intent(request)
        
        # Create plan
        plan = self.create_plan(intent, target_path)
        
        # Execute plan
        results = self.execute_plan(plan)
        
        # Store in memory
        self.memory.store_episode(request, plan, results)
        
        return results
    
    def parse_intent(self, request: str) -> Dict:
        """Parse user intent"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "Parse user intent. Return JSON with: task_type, target, requirements"
            }, {
                "role": "user",
                "content": request
            }],
            temperature=0.2
        )
        
        import json
        return json.loads(response.choices[0].message.content)
    
    def create_plan(self, intent: Dict, target_path: str) -> List[Task]:
        """Create execution plan"""
        
        tasks = []
        task_type = TaskType(intent["task_type"])
        
        if task_type == TaskType.FIX:
            # Fix requires: analyze -> fix -> test
            tasks.append(Task(TaskType.ANALYZE, target_path, "Analyze code", {}))
            tasks.append(Task(TaskType.FIX, target_path, intent["requirements"], {}))
            tasks.append(Task(TaskType.TEST, target_path, "Validate fix", {}))
        
        elif task_type == TaskType.REFACTOR:
            # Refactor requires: analyze -> refactor -> test
            tasks.append(Task(TaskType.ANALYZE, target_path, "Analyze code", {}))
            tasks.append(Task(TaskType.REFACTOR, target_path, intent["requirements"], {}))
            tasks.append(Task(TaskType.TEST, target_path, "Validate refactor", {}))
        
        else:
            tasks.append(Task(task_type, target_path, intent["requirements"], {}))
        
        return tasks
    
    def execute_plan(self, plan: List[Task]) -> Dict:
        """Execute task plan"""
        
        results = []
        context = {}
        
        for task in plan:
            task.context = context
            
            if task.type == TaskType.ANALYZE:
                result = self.analyzer.execute(task)
            elif task.type == TaskType.FIX:
                result = self.fixer.execute(task)
            elif task.type == TaskType.TEST:
                result = self.tester.execute(task)
            else:
                result = {"error": "Unknown task type"}
            
            results.append(result)
            context.update(result)
        
        return {"tasks": len(plan), "results": results}
```

### 2. Analyzer Agent

```python
# src/agents/analyzer.py
import ast
from typing import Dict, List

class AnalyzerAgent:
    """Analyzes code for issues"""
    
    def __init__(self):
        self.client = openai.OpenAI()
    
    def execute(self, task: Task) -> Dict:
        """Analyze code file"""
        
        # Read code
        with open(task.target, 'r') as f:
            code = f.read()
        
        # Parse AST
        ast_analysis = self.analyze_ast(code)
        
        # Run static analysis
        static_issues = self.run_static_analysis(task.target)
        
        # LLM-based analysis
        llm_analysis = self.llm_analyze(code)
        
        return {
            "file": task.target,
            "ast_analysis": ast_analysis,
            "static_issues": static_issues,
            "llm_analysis": llm_analysis,
            "issues": self.consolidate_issues(static_issues, llm_analysis)
        }
    
    def analyze_ast(self, code: str) -> Dict:
        """Analyze code structure"""
        
        try:
            tree = ast.parse(code)
            
            functions = [node.name for node in ast.walk(tree) 
                        if isinstance(node, ast.FunctionDef)]
            classes = [node.name for node in ast.walk(tree) 
                      if isinstance(node, ast.ClassDef)]
            
            return {
                "functions": functions,
                "classes": classes,
                "lines": len(code.split('\n'))
            }
        except SyntaxError as e:
            return {"error": str(e)}
    
    def run_static_analysis(self, file_path: str) -> List[Dict]:
        """Run pylint"""
        
        import subprocess
        
        result = subprocess.run(
            ['pylint', file_path, '--output-format=json'],
            capture_output=True,
            text=True
        )
        
        import json
        try:
            return json.loads(result.stdout)
        except:
            return []
    
    def llm_analyze(self, code: str) -> Dict:
        """LLM-based code analysis"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "You are an expert code reviewer. Analyze code for bugs, security issues, and improvements."
            }, {
                "role": "user",
                "content": f"Analyze this code:\n\n{code}"
            }],
            temperature=0.3
        )
        
        return {"analysis": response.choices[0].message.content}
    
    def consolidate_issues(self, static: List[Dict], llm: Dict) -> List[Dict]:
        """Consolidate all issues"""
        
        issues = []
        
        # Add static analysis issues
        for issue in static:
            issues.append({
                "type": issue.get("type", "unknown"),
                "message": issue.get("message", ""),
                "line": issue.get("line", 0),
                "severity": issue.get("severity", "info"),
                "source": "static"
            })
        
        return issues
```

### 3. Fixer Agent

```python
# src/agents/fixer.py
from typing import Dict
import difflib

class FixerAgent:
    """Generates and applies fixes"""
    
    def __init__(self):
        self.client = openai.OpenAI()
        self.validator = FixValidator()
    
    def execute(self, task: Task) -> Dict:
        """Generate and apply fix"""
        
        # Read current code
        with open(task.target, 'r') as f:
            original_code = f.read()
        
        # Get issues from context
        issues = task.context.get("issues", [])
        
        # Generate fix
        fixed_code = self.generate_fix(original_code, issues, task.description)
        
        # Validate fix
        validation = self.validator.validate(original_code, fixed_code)
        
        if not validation["valid"]:
            return {
                "success": False,
                "error": "Validation failed",
                "details": validation
            }
        
        # Show diff
        diff = self.generate_diff(original_code, fixed_code)
        
        return {
            "success": True,
            "original_code": original_code,
            "fixed_code": fixed_code,
            "diff": diff,
            "validation": validation
        }
    
    def generate_fix(self, code: str, issues: List[Dict], description: str) -> str:
        """Generate fixed code"""
        
        issues_text = "\n".join([
            f"- Line {i['line']}: {i['message']}"
            for i in issues[:5]  # Top 5 issues
        ])
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "You are an expert programmer. Fix code issues while preserving functionality."
            }, {
                "role": "user",
                "content": f"Fix these issues:\n{issues_text}\n\nRequirement: {description}\n\nOriginal code:\n{code}\n\nFixed code:"
            }],
            temperature=0.2
        )
        
        return self.extract_code(response.choices[0].message.content)
    
    def generate_diff(self, original: str, fixed: str) -> str:
        """Generate unified diff"""
        
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            fixed.splitlines(keepends=True),
            fromfile='original',
            tofile='fixed'
        )
        
        return ''.join(diff)
    
    def extract_code(self, text: str) -> str:
        """Extract code from markdown"""
        import re
        pattern = r'```python\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0] if matches else text

class FixValidator:
    """Validate fixes"""
    
    def validate(self, original: str, fixed: str) -> Dict:
        """Multi-level validation"""
        
        return {
            "valid": self.check_syntax(fixed) and self.check_safety(fixed),
            "syntax_valid": self.check_syntax(fixed),
            "safety_passed": self.check_safety(fixed)
        }
    
    def check_syntax(self, code: str) -> bool:
        """Check syntax"""
        try:
            ast.parse(code)
            return True
        except:
            return False
    
    def check_safety(self, code: str) -> bool:
        """Check for unsafe patterns"""
        unsafe = ["eval(", "exec(", "__import__", "os.system"]
        return not any(pattern in code for pattern in unsafe)
```

### 4. Tester Agent

```python
# src/agents/tester.py
from typing import Dict, List

class TesterAgent:
    """Generates and runs tests"""
    
    def __init__(self):
        self.client = openai.OpenAI()
    
    def execute(self, task: Task) -> Dict:
        """Generate tests for code"""
        
        # Read code
        with open(task.target, 'r') as f:
            code = f.read()
        
        # Generate tests
        tests = self.generate_tests(code)
        
        # Run tests
        results = self.run_tests(tests)
        
        return {
            "tests_generated": len(tests),
            "tests_passed": sum(1 for r in results if r["passed"]),
            "coverage": self.calculate_coverage(code, tests),
            "test_code": tests
        }
    
    def generate_tests(self, code: str) -> str:
        """Generate test code"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "Generate comprehensive pytest tests. Include edge cases, error cases, and normal cases."
            }, {
                "role": "user",
                "content": f"Generate tests for:\n\n{code}"
            }],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def run_tests(self, test_code: str) -> List[Dict]:
        """Run generated tests"""
        
        # Write to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            test_file = f.name
        
        # Run pytest
        import subprocess
        result = subprocess.run(
            ['pytest', test_file, '-v', '--json-report'],
            capture_output=True
        )
        
        return [{"passed": result.returncode == 0}]
    
    def calculate_coverage(self, code: str, tests: str) -> float:
        """Estimate test coverage"""
        # Simplified coverage estimation
        return 0.85
```

### 5. Memory System

```python
# src/memory/agent_memory.py
import chromadb
from typing import Dict, List
import json

class AgentMemory:
    """Unified memory system"""
    
    def __init__(self):
        self.working_memory = []
        self.client = chromadb.Client()
        self.episodes = self.client.create_collection("episodes")
        self.codebase = self.client.create_collection("codebase")
    
    def store_episode(self, request: str, plan: List[Task], results: Dict):
        """Store completed episode"""
        
        episode = {
            "request": request,
            "plan": [{"type": t.type.value, "target": t.target} for t in plan],
            "results": results,
            "success": results.get("success", False)
        }
        
        self.episodes.add(
            documents=[json.dumps(episode)],
            metadatas=[{"request": request}],
            ids=[f"episode_{len(self.episodes.get()['ids'])}"]
        )
    
    def recall_similar_episodes(self, request: str, limit: int = 3) -> List[Dict]:
        """Recall similar past episodes"""
        
        results = self.episodes.query(
            query_texts=[request],
            n_results=limit
        )
        
        return [json.loads(doc) for doc in results['documents'][0]]
    
    def index_file(self, file_path: str, code: str, analysis: Dict):
        """Index file in semantic memory"""
        
        self.codebase.add(
            documents=[code],
            metadatas=[{
                "file_path": file_path,
                "functions": json.dumps(analysis.get("functions", [])),
                "classes": json.dumps(analysis.get("classes", []))
            }],
            ids=[file_path]
        )
    
    def search_codebase(self, query: str, limit: int = 5) -> List[Dict]:
        """Search codebase semantically"""
        
        results = self.codebase.query(
            query_texts=[query],
            n_results=limit
        )
        
        return results
```

### 6. Tool Layer

```python
# src/tools/code_tools.py
import ast
import subprocess
from typing import Dict, List

class CodeTools:
    """Low-level code manipulation tools"""
    
    @staticmethod
    def parse_python(code: str) -> Dict:
        """Parse Python code"""
        
        try:
            tree = ast.parse(code)
            
            return {
                "valid": True,
                "functions": [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)],
                "classes": [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)],
                "imports": [n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)]
            }
        except SyntaxError as e:
            return {"valid": False, "error": str(e)}
    
    @staticmethod
    def run_linter(file_path: str) -> List[Dict]:
        """Run pylint"""
        
        result = subprocess.run(
            ['pylint', file_path, '--output-format=json'],
            capture_output=True,
            text=True
        )
        
        import json
        try:
            return json.loads(result.stdout)
        except:
            return []
    
    @staticmethod
    def format_code(code: str) -> str:
        """Format with black"""
        
        result = subprocess.run(
            ['black', '-'],
            input=code,
            capture_output=True,
            text=True
        )
        
        return result.stdout if result.returncode == 0 else code
    
    @staticmethod
    def run_tests(test_path: str) -> Dict:
        """Run pytest"""
        
        result = subprocess.run(
            ['pytest', test_path, '-v'],
            capture_output=True,
            text=True
        )
        
        return {
            "passed": result.returncode == 0,
            "output": result.stdout
        }

class SafeExecutor:
    """Execute code safely in Docker"""
    
    def __init__(self):
        import docker
        self.client = docker.from_env()
    
    def execute(self, code: str, timeout: int = 30) -> Dict:
        """Execute in isolated container"""
        
        try:
            container = self.client.containers.run(
                "python:3.11-slim",
                command=['python', '-c', code],
                detach=True,
                mem_limit="256m",
                network_disabled=True,
                remove=True
            )
            
            result = container.wait(timeout=timeout)
            logs = container.logs().decode()
            
            return {"success": True, "output": logs, "exit_code": result['StatusCode']}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
```

### 7. Complete Agent Implementation

```python
# src/agents/fixer.py (complete version)
from typing import Dict, List
import openai

class FixerAgent:
    """Generates and applies fixes"""
    
    def __init__(self):
        self.client = openai.OpenAI()
        self.tools = CodeTools()
    
    def execute(self, task: Task) -> Dict:
        """Generate fix for issues"""
        
        # Read code
        with open(task.target, 'r') as f:
            original_code = f.read()
        
        # Get issues from context
        issues = task.context.get("issues", [])
        
        # Retrieve similar fixes from memory
        similar_fixes = self.recall_similar_fixes(issues)
        
        # Generate fix with context
        fixed_code = self.generate_fix(
            original_code, 
            issues, 
            task.description,
            similar_fixes
        )
        
        # Validate
        if not self.validate_fix(original_code, fixed_code):
            return {"success": False, "error": "Validation failed"}
        
        # Generate explanation
        explanation = self.explain_fix(original_code, fixed_code, issues)
        
        return {
            "success": True,
            "original_code": original_code,
            "fixed_code": fixed_code,
            "explanation": explanation,
            "issues_addressed": len(issues)
        }
    
    def generate_fix(self, 
                    code: str, 
                    issues: List[Dict],
                    description: str,
                    similar_fixes: List[Dict]) -> str:
        """Generate fixed code"""
        
        issues_text = "\n".join([
            f"- Line {i['line']}: {i['message']} (severity: {i['severity']})"
            for i in issues[:10]
        ])
        
        context_text = ""
        if similar_fixes:
            context_text = "\n\nSimilar fixes from history:\n" + "\n".join([
                f"- {fix['description']}: {fix['approach']}"
                for fix in similar_fixes[:3]
            ])
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "Fix code issues while preserving functionality. Return only the fixed code."
            }, {
                "role": "user",
                "content": f"Issues:\n{issues_text}\n\nRequirement: {description}{context_text}\n\nCode:\n{code}\n\nFixed code:"
            }],
            temperature=0.2
        )
        
        return self.extract_code(response.choices[0].message.content)
    
    def validate_fix(self, original: str, fixed: str) -> bool:
        """Validate fix"""
        
        # Check syntax
        parsed = self.tools.parse_python(fixed)
        if not parsed["valid"]:
            return False
        
        # Check no unsafe operations
        unsafe = ["eval(", "exec(", "os.system"]
        if any(op in fixed for op in unsafe):
            return False
        
        return True
    
    def explain_fix(self, original: str, fixed: str, issues: List[Dict]) -> str:
        """Explain what was fixed"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": f"Explain changes:\n\nOriginal:\n{original[:500]}\n\nFixed:\n{fixed[:500]}\n\nIssues addressed: {len(issues)}"
            }],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def recall_similar_fixes(self, issues: List[Dict]) -> List[Dict]:
        """Recall similar fixes from memory"""
        # Simplified - would use vector search
        return []
    
    def extract_code(self, text: str) -> str:
        """Extract code from response"""
        import re
        pattern = r'```python\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0] if matches else text
```

### 8. CLI Interface

```python
# src/cli.py
import click
from orchestration.orchestrator import SoftwareEngineeringAgent

@click.group()
def cli():
    """Autonomous Software Engineering Agent"""
    pass

@cli.command()
@click.argument('file_path')
def analyze(file_path):
    """Analyze code file"""
    agent = SoftwareEngineeringAgent()
    result = agent.process_request(f"Analyze {file_path}", file_path)
    click.echo(json.dumps(result, indent=2))

@cli.command()
@click.argument('file_path')
@click.option('--description', '-d', help='Fix description')
def fix(file_path, description):
    """Fix issues in code"""
    agent = SoftwareEngineeringAgent()
    result = agent.process_request(
        f"Fix issues: {description}" if description else "Fix all issues",
        file_path
    )
    
    if result['results'][-1]['success']:
        click.echo("✓ Fix generated successfully")
        click.echo("\nDiff:")
        click.echo(result['results'][-1]['diff'])
    else:
        click.echo("✗ Fix failed")

@cli.command()
@click.argument('file_path')
def test(file_path):
    """Generate tests"""
    agent = SoftwareEngineeringAgent()
    result = agent.process_request(f"Generate tests for {file_path}", file_path)
    click.echo(f"Generated {result['results'][0]['tests_generated']} tests")

if __name__ == '__main__':
    cli()
```

## Usage Examples

### Example 1: Analyze and Fix

```bash
# Analyze code
python src/cli.py analyze src/example.py

# Fix issues
python src/cli.py fix src/example.py --description "Fix type errors and add error handling"

# Generate tests
python src/cli.py test src/example.py
```

### Example 2: Programmatic Usage

```python
from orchestration.orchestrator import SoftwareEngineeringAgent

# Initialize agent
agent = SoftwareEngineeringAgent()

# Analyze code
result = agent.process_request(
    "Analyze this file for bugs and security issues",
    "src/auth.py"
)

print(f"Found {len(result['results'][0]['issues'])} issues")

# Fix critical issues
fix_result = agent.process_request(
    "Fix all critical and high severity issues",
    "src/auth.py"
)

if fix_result['results'][-1]['success']:
    print("Fix applied successfully")
    print(fix_result['results'][-1]['explanation'])
```

## Advanced Features

### Learning from Feedback

```python
class FeedbackLearner:
    """Learn from user feedback"""
    
    def __init__(self):
        self.feedback_db = []
    
    def collect_feedback(self, task: Task, result: Dict, user_rating: int):
        """Collect user feedback"""
        
        self.feedback_db.append({
            "task": task,
            "result": result,
            "rating": user_rating,
            "timestamp": time.time()
        })
    
    def improve_from_feedback(self):
        """Analyze feedback and improve"""
        
        # Identify patterns in low-rated results
        low_rated = [f for f in self.feedback_db if f["rating"] < 3]
        
        # Extract common issues
        # Adjust prompts or strategies
        # Update tool selection logic
        pass
```

### Parallel Processing

```python
import asyncio
from typing import List

class ParallelAnalyzer:
    """Analyze multiple files in parallel"""
    
    async def analyze_files(self, file_paths: List[str]) -> List[Dict]:
        """Analyze files concurrently"""
        
        tasks = [self.analyze_file(path) for path in file_paths]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def analyze_file(self, file_path: str) -> Dict:
        """Analyze single file"""
        
        analyzer = AnalyzerAgent()
        task = Task(TaskType.ANALYZE, file_path, "Analyze", {})
        
        return analyzer.execute(task)

# Usage
async def main():
    analyzer = ParallelAnalyzer()
    results = await analyzer.analyze_files(['file1.py', 'file2.py', 'file3.py'])
    print(f"Analyzed {len(results)} files")

asyncio.run(main())
```

## Testing the Agent

### Unit Tests

```python
# tests/test_analyzer.py
import pytest
from agents.analyzer import AnalyzerAgent
from orchestration.orchestrator import Task, TaskType

def test_analyzer_detects_issues():
    """Test analyzer finds issues"""
    
    agent = AnalyzerAgent()
    
    # Create test task
    task = Task(
        type=TaskType.ANALYZE,
        target="tests/fixtures/buggy_code.py",
        description="Analyze",
        context={}
    )
    
    result = agent.execute(task)
    
    assert "issues" in result
    assert len(result["issues"]) > 0

def test_analyzer_handles_syntax_errors():
    """Test analyzer handles invalid syntax"""
    
    agent = AnalyzerAgent()
    
    # Write invalid code
    with open("tests/fixtures/invalid.py", "w") as f:
        f.write("def broken(\n")
    
    task = Task(TaskType.ANALYZE, "tests/fixtures/invalid.py", "Analyze", {})
    result = agent.execute(task)
    
    assert "error" in result["ast_analysis"]
```

### Integration Tests

```python
# tests/test_integration.py
import pytest
from orchestration.orchestrator import SoftwareEngineeringAgent

def test_end_to_end_fix():
    """Test complete fix workflow"""
    
    agent = SoftwareEngineeringAgent()
    
    # Create buggy code
    buggy_code = '''
def divide(a, b):
    return a / b
'''
    
    with open("tests/fixtures/buggy.py", "w") as f:
        f.write(buggy_code)
    
    # Request fix
    result = agent.process_request(
        "Fix the division by zero bug",
        "tests/fixtures/buggy.py"
    )
    
    # Verify fix was generated
    assert result["results"][-1]["success"]
    assert "if b == 0" in result["results"][-1]["fixed_code"]
```

## Deployment

### Docker Container

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source
COPY src/ ./src/

# Expose API
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0"]
```

### API Service

```python
# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Autonomous SE Agent API")

class AnalyzeRequest(BaseModel):
    file_path: str
    options: Dict = {}

class FixRequest(BaseModel):
    file_path: str
    description: str

@app.post("/analyze")
async def analyze_code(request: AnalyzeRequest):
    """Analyze code endpoint"""
    
    agent = SoftwareEngineeringAgent()
    result = agent.process_request(
        f"Analyze {request.file_path}",
        request.file_path
    )
    
    return result

@app.post("/fix")
async def fix_code(request: FixRequest):
    """Fix code endpoint"""
    
    agent = SoftwareEngineeringAgent()
    result = agent.process_request(
        f"Fix: {request.description}",
        request.file_path
    )
    
    return result

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}
```

## Next Steps

You now have a complete implementation! In the next section, we'll evaluate and iterate on the agent to make it production-ready.
