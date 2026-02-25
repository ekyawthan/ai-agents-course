# Evaluation & Iteration

## Evaluating Your Agent

Now that you've built the Autonomous Software Engineering Agent, let's evaluate its performance and iterate to improve it.

## Evaluation Framework

### Test Suite Design

```python
# tests/evaluation/test_suite.py
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class TestCase:
    name: str
    input_code: str
    expected_issues: List[str]
    expected_fix_pattern: str
    difficulty: str  # easy, medium, hard

class EvaluationSuite:
    """Comprehensive evaluation suite"""
    
    def __init__(self):
        self.test_cases = self.create_test_cases()
        self.results = []
    
    def create_test_cases(self) -> List[TestCase]:
        """Create diverse test cases"""
        
        return [
            TestCase(
                name="Division by zero",
                input_code="def divide(a, b): return a / b",
                expected_issues=["ZeroDivisionError"],
                expected_fix_pattern="if b == 0",
                difficulty="easy"
            ),
            TestCase(
                name="SQL injection",
                input_code='query = f"SELECT * FROM users WHERE id = {user_id}"',
                expected_issues=["SQL injection"],
                expected_fix_pattern="parameterized",
                difficulty="medium"
            ),
            TestCase(
                name="Race condition",
                input_code="""
counter = 0
def increment():
    global counter
    temp = counter
    counter = temp + 1
""",
                expected_issues=["race condition"],
                expected_fix_pattern="lock",
                difficulty="hard"
            )
        ]
    
    def run_evaluation(self, agent) -> Dict:
        """Run full evaluation"""
        
        results = {
            "total": len(self.test_cases),
            "passed": 0,
            "by_difficulty": {"easy": 0, "medium": 0, "hard": 0}
        }
        
        for test_case in self.test_cases:
            result = self.evaluate_test_case(agent, test_case)
            self.results.append(result)
            
            if result["passed"]:
                results["passed"] += 1
                results["by_difficulty"][test_case.difficulty] += 1
        
        results["accuracy"] = results["passed"] / results["total"]
        
        return results
    
    def evaluate_test_case(self, agent, test_case: TestCase) -> Dict:
        """Evaluate single test case"""
        
        # Write test code to file
        test_file = f"tests/fixtures/{test_case.name.replace(' ', '_')}.py"
        with open(test_file, 'w') as f:
            f.write(test_case.input_code)
        
        # Run agent
        result = agent.process_request(
            f"Analyze and fix issues in {test_file}",
            test_file
        )
        
        # Check if issues detected
        issues_found = result["results"][0].get("issues", [])
        detected_expected = any(
            expected in str(issues_found).lower()
            for expected in test_case.expected_issues
        )
        
        # Check if fix applied correctly
        fixed_code = result["results"][1].get("fixed_code", "")
        fix_correct = test_case.expected_fix_pattern.lower() in fixed_code.lower()
        
        return {
            "test_case": test_case.name,
            "passed": detected_expected and fix_correct,
            "issues_detected": detected_expected,
            "fix_correct": fix_correct,
            "difficulty": test_case.difficulty
        }
```

### Performance Benchmarks

```python
# tests/evaluation/benchmarks.py
import time
from typing import Dict

class PerformanceBenchmark:
    """Benchmark agent performance"""
    
    def __init__(self):
        self.metrics = {}
    
    def benchmark_analysis_speed(self, agent, file_sizes: List[int]) -> Dict:
        """Benchmark analysis speed"""
        
        results = {}
        
        for size in file_sizes:
            # Generate code of specific size
            code = self.generate_code(size)
            test_file = f"tests/fixtures/size_{size}.py"
            
            with open(test_file, 'w') as f:
                f.write(code)
            
            # Time analysis
            start = time.time()
            agent.process_request(f"Analyze {test_file}", test_file)
            duration = time.time() - start
            
            results[size] = {
                "duration": duration,
                "lines_per_second": size / duration
            }
        
        return results
    
    def benchmark_fix_quality(self, agent, test_cases: List[TestCase]) -> Dict:
        """Benchmark fix quality"""
        
        metrics = {
            "fixes_attempted": 0,
            "fixes_successful": 0,
            "fixes_optimal": 0,
            "avg_fix_time": []
        }
        
        for test_case in test_cases:
            start = time.time()
            
            # Generate fix
            result = agent.process_request(
                f"Fix issues in {test_case.name}",
                test_case.name
            )
            
            duration = time.time() - start
            metrics["avg_fix_time"].append(duration)
            metrics["fixes_attempted"] += 1
            
            if result["results"][-1]["success"]:
                metrics["fixes_successful"] += 1
                
                # Check if optimal
                if self.is_optimal_fix(result["results"][-1]["fixed_code"]):
                    metrics["fixes_optimal"] += 1
        
        return metrics
    
    def generate_code(self, lines: int) -> str:
        """Generate code of specific size"""
        return "\n".join([f"# Line {i}" for i in range(lines)])
    
    def is_optimal_fix(self, code: str) -> bool:
        """Check if fix is optimal"""
        # Simplified check
        return "try" in code or "if" in code
```

## Real-World Testing

### Beta Testing Strategy

```python
class BetaTester:
    """Coordinate beta testing"""
    
    def __init__(self):
        self.testers = []
        self.feedback = []
    
    def run_beta_test(self, agent, duration_days: int = 7) -> Dict:
        """Run beta test program"""
        
        print(f"Starting {duration_days}-day beta test...")
        
        # Collect usage data
        usage_data = self.collect_usage_data(agent, duration_days)
        
        # Collect feedback
        feedback = self.collect_feedback()
        
        # Analyze results
        analysis = self.analyze_beta_results(usage_data, feedback)
        
        return analysis
    
    def collect_usage_data(self, agent, days: int) -> Dict:
        """Collect usage metrics"""
        
        return {
            "total_requests": 0,
            "successful_requests": 0,
            "avg_response_time": 0,
            "most_common_tasks": [],
            "error_rate": 0
        }
    
    def collect_feedback(self) -> List[Dict]:
        """Collect user feedback"""
        
        return [
            {
                "user": "tester1",
                "rating": 4,
                "comments": "Works well for simple bugs",
                "issues": ["Slow on large files"]
            }
        ]
    
    def analyze_beta_results(self, usage: Dict, feedback: List[Dict]) -> Dict:
        """Analyze beta test results"""
        
        avg_rating = sum(f["rating"] for f in feedback) / len(feedback)
        
        return {
            "usage_stats": usage,
            "avg_rating": avg_rating,
            "key_issues": self.extract_key_issues(feedback),
            "recommendations": self.generate_recommendations(usage, feedback)
        }
    
    def extract_key_issues(self, feedback: List[Dict]) -> List[str]:
        """Extract common issues"""
        
        all_issues = []
        for f in feedback:
            all_issues.extend(f.get("issues", []))
        
        # Count frequency
        from collections import Counter
        return [issue for issue, count in Counter(all_issues).most_common(5)]
    
    def generate_recommendations(self, usage: Dict, feedback: List[Dict]) -> List[str]:
        """Generate improvement recommendations"""
        
        recommendations = []
        
        if usage["error_rate"] > 0.1:
            recommendations.append("Improve error handling")
        
        if usage["avg_response_time"] > 10:
            recommendations.append("Optimize performance")
        
        return recommendations
```

## Iteration Process

### Continuous Improvement Loop

```python
class ImprovementLoop:
    """Continuous improvement system"""
    
    def __init__(self, agent):
        self.agent = agent
        self.version = 1
        self.performance_history = []
    
    def iterate(self, evaluation_results: Dict) -> Dict:
        """Improve based on evaluation"""
        
        # Identify weaknesses
        weaknesses = self.identify_weaknesses(evaluation_results)
        
        # Generate improvements
        improvements = self.generate_improvements(weaknesses)
        
        # Apply improvements
        self.apply_improvements(improvements)
        
        # Re-evaluate
        new_results = self.evaluate()
        
        # Track progress
        self.performance_history.append({
            "version": self.version,
            "results": new_results
        })
        
        self.version += 1
        
        return {
            "improvements_made": len(improvements),
            "performance_change": self.calculate_improvement(evaluation_results, new_results)
        }
    
    def identify_weaknesses(self, results: Dict) -> List[str]:
        """Identify areas needing improvement"""
        
        weaknesses = []
        
        if results["accuracy"] < 0.8:
            weaknesses.append("low_accuracy")
        
        if results.get("avg_response_time", 0) > 10:
            weaknesses.append("slow_performance")
        
        if results.get("error_rate", 0) > 0.05:
            weaknesses.append("high_error_rate")
        
        return weaknesses
    
    def generate_improvements(self, weaknesses: List[str]) -> List[Dict]:
        """Generate improvement strategies"""
        
        improvements = []
        
        for weakness in weaknesses:
            if weakness == "low_accuracy":
                improvements.append({
                    "area": "prompts",
                    "action": "Refine analysis prompts with more examples"
                })
            
            elif weakness == "slow_performance":
                improvements.append({
                    "area": "caching",
                    "action": "Add caching for repeated analyses"
                })
            
            elif weakness == "high_error_rate":
                improvements.append({
                    "area": "error_handling",
                    "action": "Add more robust error handling"
                })
        
        return improvements
    
    def apply_improvements(self, improvements: List[Dict]):
        """Apply improvements to agent"""
        
        for improvement in improvements:
            print(f"Applying: {improvement['action']}")
            # Apply improvement
            # In practice, would modify agent configuration or code
    
    def evaluate(self) -> Dict:
        """Run evaluation"""
        suite = EvaluationSuite()
        return suite.run_evaluation(self.agent)
    
    def calculate_improvement(self, old: Dict, new: Dict) -> float:
        """Calculate improvement percentage"""
        
        old_acc = old.get("accuracy", 0)
        new_acc = new.get("accuracy", 0)
        
        return ((new_acc - old_acc) / old_acc * 100) if old_acc > 0 else 0
```

## Production Deployment

### Deployment Checklist

- [ ] All tests passing
- [ ] Performance benchmarks met
- [ ] Security audit completed
- [ ] Documentation updated
- [ ] Monitoring configured
- [ ] Rollback plan ready
- [ ] User training completed
- [ ] Feedback system active

### Monitoring Setup

```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
requests_total = Counter('agent_requests_total', 'Total requests', ['task_type'])
request_duration = Histogram('agent_request_duration_seconds', 'Request duration')
active_tasks = Gauge('agent_active_tasks', 'Active tasks')
errors_total = Counter('agent_errors_total', 'Total errors', ['error_type'])

class MonitoredAgent:
    """Agent with monitoring"""
    
    def __init__(self, agent):
        self.agent = agent
    
    def process_request(self, request: str, target: str) -> Dict:
        """Process with monitoring"""
        
        active_tasks.inc()
        start = time.time()
        
        try:
            result = self.agent.process_request(request, target)
            
            # Record metrics
            requests_total.labels(task_type=result.get("task_type", "unknown")).inc()
            request_duration.observe(time.time() - start)
            
            return result
            
        except Exception as e:
            errors_total.labels(error_type=type(e).__name__).inc()
            raise
        
        finally:
            active_tasks.dec()
```

### Logging Strategy

```python
# src/monitoring/logging_config.py
import logging
import json

class StructuredLogger:
    """Structured logging for agent"""
    
    def __init__(self):
        self.logger = logging.getLogger("se_agent")
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler("agent.log")
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
    
    def log_request(self, request: str, target: str):
        """Log incoming request"""
        self.logger.info(json.dumps({
            "event": "request",
            "request": request,
            "target": target,
            "timestamp": time.time()
        }))
    
    def log_result(self, result: Dict):
        """Log result"""
        self.logger.info(json.dumps({
            "event": "result",
            "success": result.get("success"),
            "timestamp": time.time()
        }))
    
    def log_error(self, error: Exception):
        """Log error"""
        self.logger.error(json.dumps({
            "event": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time()
        }))
```

## User Feedback Collection

### Feedback System

```python
# src/feedback/collector.py
from typing import Dict, Optional
import sqlite3

class FeedbackCollector:
    """Collect and analyze user feedback"""
    
    def __init__(self, db_path: str = "data/feedback.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize feedback database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY,
                task_id TEXT,
                rating INTEGER,
                comments TEXT,
                accepted BOOLEAN,
                timestamp REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def collect(self, task_id: str, rating: int, comments: str, accepted: bool):
        """Store feedback"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback (task_id, rating, comments, accepted, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (task_id, rating, comments, accepted, time.time()))
        
        conn.commit()
        conn.close()
    
    def analyze_feedback(self) -> Dict:
        """Analyze collected feedback"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get statistics
        cursor.execute('SELECT AVG(rating), COUNT(*) FROM feedback')
        avg_rating, total = cursor.fetchone()
        
        cursor.execute('SELECT COUNT(*) FROM feedback WHERE accepted = 1')
        accepted = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "avg_rating": avg_rating,
            "total_feedback": total,
            "acceptance_rate": accepted / total if total > 0 else 0
        }
    
    def get_improvement_suggestions(self) -> List[str]:
        """Extract improvement suggestions from feedback"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get low-rated feedback
        cursor.execute('SELECT comments FROM feedback WHERE rating < 3')
        low_rated = cursor.fetchall()
        
        conn.close()
        
        # Extract common themes
        suggestions = []
        for (comment,) in low_rated:
            if comment:
                suggestions.append(comment)
        
        return suggestions
```

## A/B Testing

### Comparing Agent Versions

```python
class ABTester:
    """A/B test different agent versions"""
    
    def __init__(self, agent_a, agent_b):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.results_a = []
        self.results_b = []
    
    def run_ab_test(self, test_cases: List[TestCase]) -> Dict:
        """Run A/B test"""
        
        import random
        
        for test_case in test_cases:
            # Randomly assign to A or B
            if random.random() < 0.5:
                result = self.test_agent(self.agent_a, test_case)
                self.results_a.append(result)
            else:
                result = self.test_agent(self.agent_b, test_case)
                self.results_b.append(result)
        
        # Compare results
        return self.compare_results()
    
    def test_agent(self, agent, test_case: TestCase) -> Dict:
        """Test single agent"""
        
        start = time.time()
        result = agent.process_request(test_case.name, test_case.name)
        duration = time.time() - start
        
        return {
            "success": result.get("success", False),
            "duration": duration
        }
    
    def compare_results(self) -> Dict:
        """Compare A vs B"""
        
        a_success = sum(1 for r in self.results_a if r["success"]) / len(self.results_a)
        b_success = sum(1 for r in self.results_b if r["success"]) / len(self.results_b)
        
        a_speed = sum(r["duration"] for r in self.results_a) / len(self.results_a)
        b_speed = sum(r["duration"] for r in self.results_b) / len(self.results_b)
        
        return {
            "agent_a": {"success_rate": a_success, "avg_duration": a_speed},
            "agent_b": {"success_rate": b_success, "avg_duration": b_speed},
            "winner": "A" if a_success > b_success else "B"
        }
```

## Iteration Examples

### Iteration 1: Improve Accuracy

**Problem**: Agent missing 30% of bugs

**Analysis**:
```python
# Analyze false negatives
false_negatives = [
    "Off-by-one errors",
    "Null pointer issues",
    "Type mismatches"
]
```

**Solution**:
```python
# Enhanced analysis prompt
enhanced_prompt = """Analyze code for:
1. Logic errors (off-by-one, boundary conditions)
2. Null/None handling
3. Type safety
4. Resource leaks
5. Concurrency issues

Be thorough and check edge cases."""

# Update analyzer
analyzer.system_prompt = enhanced_prompt
```

**Result**: Accuracy improved from 70% → 85%

### Iteration 2: Optimize Performance

**Problem**: Analysis takes 15s per file (target: <5s)

**Analysis**:
```python
# Profile performance
import cProfile

profiler = cProfile.Profile()
profiler.enable()
agent.process_request("Analyze file.py", "file.py")
profiler.disable()
profiler.print_stats(sort='cumtime')
```

**Solution**:
```python
# Add caching
class CachedAnalyzer:
    def __init__(self):
        self.cache = {}
    
    def analyze(self, file_path: str) -> Dict:
        # Check cache
        file_hash = self.hash_file(file_path)
        
        if file_hash in self.cache:
            return self.cache[file_hash]
        
        # Analyze
        result = self.do_analysis(file_path)
        
        # Cache result
        self.cache[file_hash] = result
        
        return result
```

**Result**: Analysis time reduced to 3s per file

### Iteration 3: Reduce False Positives

**Problem**: 40% of reported issues are false positives

**Analysis**:
```python
# Analyze false positives
fp_analysis = {
    "style_issues_as_bugs": 15,
    "context_misunderstanding": 12,
    "overly_strict_checks": 8
}
```

**Solution**:
```python
# Add confidence scoring
class ConfidenceScorer:
    def score_issue(self, issue: Dict) -> float:
        """Score issue confidence"""
        
        score = 0.5  # Base
        
        # Increase for multiple sources
        if issue["source"] == "static" and issue.get("llm_confirmed"):
            score += 0.3
        
        # Increase for severity
        if issue["severity"] == "critical":
            score += 0.2
        
        return min(score, 1.0)

# Filter low-confidence issues
filtered_issues = [i for i in issues if scorer.score_issue(i) > 0.6]
```

**Result**: False positive rate reduced from 40% → 15%

## Production Metrics

### Key Metrics to Track

```python
class ProductionMetrics:
    """Track production metrics"""
    
    def __init__(self):
        self.metrics = {
            "requests_per_day": 0,
            "success_rate": 0,
            "avg_response_time": 0,
            "user_satisfaction": 0,
            "bugs_fixed": 0,
            "tests_generated": 0,
            "code_quality_improvement": 0
        }
    
    def daily_report(self) -> Dict:
        """Generate daily metrics report"""
        
        return {
            "date": time.strftime("%Y-%m-%d"),
            "metrics": self.metrics,
            "alerts": self.check_alerts()
        }
    
    def check_alerts(self) -> List[str]:
        """Check for metric alerts"""
        
        alerts = []
        
        if self.metrics["success_rate"] < 0.9:
            alerts.append("Success rate below threshold")
        
        if self.metrics["avg_response_time"] > 10:
            alerts.append("Response time above threshold")
        
        return alerts
```

## Final Evaluation

### Comprehensive Assessment

```python
def final_evaluation(agent) -> Dict:
    """Comprehensive final evaluation"""
    
    # Run test suite
    suite = EvaluationSuite()
    test_results = suite.run_evaluation(agent)
    
    # Run benchmarks
    benchmark = PerformanceBenchmark()
    perf_results = benchmark.benchmark_analysis_speed(agent, [100, 500, 1000])
    
    # Analyze feedback
    feedback = FeedbackCollector()
    feedback_analysis = feedback.analyze_feedback()
    
    # Generate report
    report = {
        "test_results": test_results,
        "performance": perf_results,
        "user_feedback": feedback_analysis,
        "overall_score": calculate_overall_score(test_results, perf_results, feedback_analysis)
    }
    
    return report

def calculate_overall_score(tests: Dict, perf: Dict, feedback: Dict) -> float:
    """Calculate overall score"""
    
    # Weighted average
    test_score = tests["accuracy"] * 0.4
    perf_score = (1.0 if perf[100]["duration"] < 5 else 0.5) * 0.3
    feedback_score = feedback["acceptance_rate"] * 0.3
    
    return test_score + perf_score + feedback_score
```

## Congratulations!

You've completed the capstone project! You've built a sophisticated Autonomous Software Engineering Agent that:

✅ Analyzes code for bugs and quality issues
✅ Generates fixes with explanations
✅ Writes comprehensive tests
✅ Operates safely with validation
✅ Learns from feedback
✅ Scales to production workloads

---

## Practice Exercises

### Exercise 1: Add Code Review Agent (Medium)
**Task**: Add a ReviewerAgent that analyzes pull requests.

<details>
<summary>Click to see solution</summary>

```python
class ReviewerAgent:
    def review_pr(self, diff: str) -> Dict:
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": f"Review this code change:\n{diff}\n\nProvide: issues, suggestions, approval"
            }]
        )
        return {"review": response.choices[0].message.content}
```
</details>

### Exercise 2: Implement Learning System (Hard)
**Task**: Make the agent learn from user corrections.

<details>
<summary>Click to see solution</summary>

```python
class LearningAgent:
    def __init__(self):
        self.corrections = []
    
    def learn_from_correction(self, original: str, corrected: str):
        self.corrections.append({"original": original, "corrected": corrected})
        
        # Use corrections as few-shot examples
        if len(self.corrections) > 5:
            self.update_prompts()
```
</details>

---

> **✅ Chapter 10 Summary**
>
> You've completed the capstone project:
> - **Designed** a multi-agent software engineering system
> - **Implemented** specialized agents (analyzer, fixer, tester)
> - **Integrated** all concepts from previous chapters
> - **Evaluated** with comprehensive test suites
> - **Deployed** with monitoring and feedback loops
>
> This capstone demonstrates how to combine planning, memory, tools, safety, and learning into a production-ready autonomous system.

### What You've Learned

Throughout this course, you've mastered:
- **Foundations**: Agent architecture and LLM fundamentals
- **Building**: ReAct patterns and tool integration
- **Advanced Patterns**: Planning, memory, multi-agent systems
- **Tools**: Code execution, data access, web interaction
- **Production**: Reliability, testing, monitoring
- **Specialization**: Coding, research, automation agents
- **Advanced Topics**: Learning, multimodal, frameworks
- **Enterprise**: Architecture, security, cost optimization
- **Research**: Frontier capabilities, emerging paradigms
- **Capstone**: Complete production-ready agent

### Next Steps

1. **Deploy your agent**: Put it into production
2. **Contribute**: Share your implementation
3. **Research**: Explore open problems
4. **Build more**: Create specialized agents
5. **Teach**: Share your knowledge

Thank you for completing the Agentic Guide to AI Agents course!
