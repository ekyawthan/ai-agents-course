# Evaluation & Testing

## Agent Benchmarks

Measure agent performance systematically.

### Creating Test Suites

```python
from dataclasses import dataclass
from typing import List, Callable
import time

@dataclass
class TestCase:
    """Single test case"""
    name: str
    input: str
    expected_output: str = None
    expected_behavior: str = None
    timeout: int = 30

@dataclass
class TestResult:
    """Test result"""
    test_name: str
    passed: bool
    actual_output: str
    expected_output: str
    execution_time: float
    error: str = None

class AgentTestSuite:
    """Test suite for agents"""
    
    def __init__(self, agent):
        self.agent = agent
        self.test_cases = []
        self.results = []
    
    def add_test(self, test_case: TestCase):
        """Add test case"""
        self.test_cases.append(test_case)
    
    def run_tests(self) -> dict:
        """Run all tests"""
        self.results = []
        
        for test in self.test_cases:
            print(f"Running: {test.name}...")
            result = self.run_single_test(test)
            self.results.append(result)
        
        return self.generate_report()
    
    def run_single_test(self, test: TestCase) -> TestResult:
        """Run single test"""
        start_time = time.time()
        
        try:
            # Execute agent
            actual_output = self.agent.process(test.input)
            execution_time = time.time() - start_time
            
            # Check result
            if test.expected_output:
                passed = self.check_output_match(actual_output, test.expected_output)
            elif test.expected_behavior:
                passed = self.check_behavior(actual_output, test.expected_behavior)
            else:
                passed = True  # Just check it doesn't crash
            
            return TestResult(
                test_name=test.name,
                passed=passed,
                actual_output=actual_output,
                expected_output=test.expected_output or test.expected_behavior,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name=test.name,
                passed=False,
                actual_output="",
                expected_output=test.expected_output or test.expected_behavior,
                execution_time=execution_time,
                error=str(e)
            )
    
    def check_output_match(self, actual: str, expected: str) -> bool:
        """Check if output matches expected"""
        # Exact match
        if actual.strip() == expected.strip():
            return True
        
        # Contains expected
        if expected.lower() in actual.lower():
            return True
        
        return False
    
    def check_behavior(self, output: str, behavior: str) -> bool:
        """Check if output exhibits expected behavior"""
        # Use LLM to judge
        prompt = f"""Does this output exhibit the expected behavior?

Output: {output}

Expected behavior: {behavior}

Answer with just 'yes' or 'no':"""
        
        response = llm.generate(prompt).strip().lower()
        return response == 'yes'
    
    def generate_report(self) -> dict:
        """Generate test report"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        avg_time = sum(r.execution_time for r in self.results) / total if total > 0 else 0
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0,
            "avg_execution_time": avg_time,
            "results": self.results
        }

# Usage
suite = AgentTestSuite(agent)

suite.add_test(TestCase(
    name="Basic math",
    input="What is 2 + 2?",
    expected_output="4"
))

suite.add_test(TestCase(
    name="Tool usage",
    input="Search for information about Python",
    expected_behavior="Uses search tool and provides relevant information"
))

report = suite.run_tests()
print(f"Pass rate: {report['pass_rate']:.1%}")
```

### Standard Benchmarks

```python
class StandardBenchmarks:
    """Common agent benchmarks"""
    
    @staticmethod
    def get_math_benchmark() -> List[TestCase]:
        """Math reasoning tests"""
        return [
            TestCase("Addition", "What is 123 + 456?", "579"),
            TestCase("Multiplication", "What is 25 * 17?", "425"),
            TestCase("Word problem", "If I have 3 apples and buy 2 more, how many do I have?", "5"),
            TestCase("Percentage", "What is 15% of 200?", "30"),
        ]
    
    @staticmethod
    def get_reasoning_benchmark() -> List[TestCase]:
        """Logical reasoning tests"""
        return [
            TestCase(
                "Deduction",
                "All cats are animals. Fluffy is a cat. Is Fluffy an animal?",
                expected_behavior="Correctly deduces that Fluffy is an animal"
            ),
            TestCase(
                "Planning",
                "I need to make dinner. What steps should I take?",
                expected_behavior="Provides logical sequence of steps"
            ),
        ]
    
    @staticmethod
    def get_tool_usage_benchmark() -> List[TestCase]:
        """Tool usage tests"""
        return [
            TestCase(
                "Search",
                "Find information about the Eiffel Tower",
                expected_behavior="Uses search tool and provides facts"
            ),
            TestCase(
                "Calculation",
                "Calculate the compound interest on $1000 at 5% for 3 years",
                expected_behavior="Uses calculator tool"
            ),
        ]
```

## Success Metrics

Define what success means for your agent.

### Quantitative Metrics

```python
class AgentMetrics:
    """Track agent performance metrics"""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_execution_time": 0,
            "tool_calls": 0,
            "tokens_used": 0,
            "cost": 0.0
        }
    
    def record_request(self, 
                      success: bool,
                      execution_time: float,
                      tool_calls: int = 0,
                      tokens: int = 0,
                      cost: float = 0.0):
        """Record request metrics"""
        self.metrics["total_requests"] += 1
        
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        self.metrics["total_execution_time"] += execution_time
        self.metrics["tool_calls"] += tool_calls
        self.metrics["tokens_used"] += tokens
        self.metrics["cost"] += cost
    
    def get_summary(self) -> dict:
        """Get metrics summary"""
        total = self.metrics["total_requests"]
        
        if total == 0:
            return self.metrics
        
        return {
            **self.metrics,
            "success_rate": self.metrics["successful_requests"] / total,
            "avg_execution_time": self.metrics["total_execution_time"] / total,
            "avg_tool_calls": self.metrics["tool_calls"] / total,
            "avg_tokens": self.metrics["tokens_used"] / total,
            "avg_cost": self.metrics["cost"] / total
        }
    
    def print_summary(self):
        """Print formatted summary"""
        summary = self.get_summary()
        
        print("Agent Performance Metrics")
        print("=" * 40)
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Avg Execution Time: {summary['avg_execution_time']:.2f}s")
        print(f"Avg Tool Calls: {summary['avg_tool_calls']:.1f}")
        print(f"Avg Tokens: {summary['avg_tokens']:.0f}")
        print(f"Avg Cost: ${summary['avg_cost']:.4f}")
        print(f"Total Cost: ${summary['cost']:.2f}")
```

### Qualitative Metrics

```python
class QualityEvaluator:
    """Evaluate response quality"""
    
    def __init__(self):
        self.client = openai.OpenAI()
    
    def evaluate_response(self, 
                         question: str,
                         response: str,
                         criteria: List[str] = None) -> dict:
        """Evaluate response quality"""
        
        if criteria is None:
            criteria = [
                "Accuracy: Is the information correct?",
                "Completeness: Does it fully answer the question?",
                "Clarity: Is it easy to understand?",
                "Relevance: Does it stay on topic?"
            ]
        
        scores = {}
        
        for criterion in criteria:
            score = self.score_criterion(question, response, criterion)
            criterion_name = criterion.split(':')[0]
            scores[criterion_name] = score
        
        return {
            "scores": scores,
            "average": sum(scores.values()) / len(scores),
            "passed": all(score >= 3 for score in scores.values())
        }
    
    def score_criterion(self, question: str, response: str, criterion: str) -> int:
        """Score response on single criterion (1-5)"""
        prompt = f"""Rate this response on the following criterion (1-5):

Question: {question}

Response: {response}

Criterion: {criterion}

Provide only a number from 1 (poor) to 5 (excellent):"""
        
        result = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        try:
            score = int(result.choices[0].message.content.strip())
            return max(1, min(5, score))  # Clamp to 1-5
        except:
            return 3  # Default to middle score
```

## Unit and Integration Testing

### Unit Tests for Components

```python
import unittest

class TestAgentComponents(unittest.TestCase):
    """Unit tests for agent components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.agent = MyAgent()
    
    def test_input_validation(self):
        """Test input validation"""
        validator = InputValidator()
        
        # Valid input
        result = validator.validate_text_input("Hello world")
        self.assertTrue(result['valid'])
        
        # Invalid input (too long)
        long_text = "x" * 20000
        result = validator.validate_text_input(long_text)
        self.assertFalse(result['valid'])
    
    def test_tool_execution(self):
        """Test tool execution"""
        result = self.agent.execute_tool("calculate", {"expression": "2 + 2"})
        self.assertEqual(result, "4")
    
    def test_memory_storage(self):
        """Test memory system"""
        self.agent.memory.add("user_name", "Alice")
        retrieved = self.agent.memory.get("user_name")
        self.assertEqual(retrieved, "Alice")
    
    def test_error_handling(self):
        """Test error handling"""
        # Should not crash on invalid tool
        result = self.agent.execute_tool("nonexistent_tool", {})
        self.assertIn("error", result.lower())
    
    def tearDown(self):
        """Clean up"""
        pass

# Run tests
if __name__ == '__main__':
    unittest.main()
```

### Integration Tests

```python
class TestAgentIntegration(unittest.TestCase):
    """Integration tests for full agent"""
    
    def test_end_to_end_query(self):
        """Test complete query flow"""
        agent = MyAgent()
        
        response = agent.process("What is 2 + 2?")
        
        self.assertIsNotNone(response)
        self.assertIn("4", response)
    
    def test_multi_step_task(self):
        """Test multi-step task execution"""
        agent = MyAgent()
        
        response = agent.process("Search for Python tutorials and summarize the top result")
        
        # Should use search tool
        self.assertTrue(agent.tool_used("search"))
        
        # Should provide summary
        self.assertGreater(len(response), 50)
    
    def test_error_recovery(self):
        """Test error recovery"""
        agent = MyAgent()
        
        # Simulate tool failure
        agent.tools["search"] = lambda x: raise_error()
        
        response = agent.process("Search for something")
        
        # Should handle gracefully
        self.assertIsNotNone(response)
        self.assertNotIn("Traceback", response)
    
    def test_rate_limiting(self):
        """Test rate limiting"""
        agent = MyAgent()
        
        # Make many requests
        for i in range(150):
            response = agent.process(f"Request {i}")
        
        # Should be rate limited
        self.assertTrue(agent.was_rate_limited())
```

### Property-Based Testing

```python
from hypothesis import given, strategies as st

class TestAgentProperties(unittest.TestCase):
    """Property-based tests"""
    
    @given(st.text(min_size=1, max_size=1000))
    def test_agent_handles_any_text(self, text):
        """Agent should handle any text input without crashing"""
        agent = MyAgent()
        
        try:
            response = agent.process(text)
            # Should return something
            self.assertIsNotNone(response)
        except Exception as e:
            # Should not crash
            self.fail(f"Agent crashed on input: {text[:50]}... Error: {e}")
    
    @given(st.integers(min_value=-1000, max_value=1000))
    def test_calculator_tool(self, number):
        """Calculator should handle any integer"""
        agent = MyAgent()
        
        result = agent.execute_tool("calculate", {"expression": f"{number} + 1"})
        expected = str(number + 1)
        
        self.assertEqual(result, expected)
```

## Human Evaluation Frameworks

### Collecting Human Feedback

```python
class HumanEvaluator:
    """Collect human evaluations"""
    
    def __init__(self):
        self.evaluations = []
    
    def request_evaluation(self, 
                          question: str,
                          response: str,
                          evaluator_id: str) -> dict:
        """Request human evaluation"""
        
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"\nResponse: {response}")
        print(f"\n{'='*60}")
        
        # Collect ratings
        ratings = {}
        
        criteria = [
            ("accuracy", "Is the response accurate? (1-5)"),
            ("helpfulness", "Is the response helpful? (1-5)"),
            ("clarity", "Is the response clear? (1-5)"),
        ]
        
        for key, prompt in criteria:
            while True:
                try:
                    score = int(input(f"{prompt}: "))
                    if 1 <= score <= 5:
                        ratings[key] = score
                        break
                except ValueError:
                    pass
        
        # Collect feedback
        feedback = input("\nAdditional feedback (optional): ")
        
        evaluation = {
            "question": question,
            "response": response,
            "evaluator_id": evaluator_id,
            "ratings": ratings,
            "feedback": feedback,
            "timestamp": time.time()
        }
        
        self.evaluations.append(evaluation)
        return evaluation
    
    def get_summary(self) -> dict:
        """Get evaluation summary"""
        if not self.evaluations:
            return {}
        
        # Average ratings
        avg_ratings = {}
        for criterion in ["accuracy", "helpfulness", "clarity"]:
            scores = [e["ratings"][criterion] for e in self.evaluations]
            avg_ratings[criterion] = sum(scores) / len(scores)
        
        return {
            "total_evaluations": len(self.evaluations),
            "average_ratings": avg_ratings,
            "overall_score": sum(avg_ratings.values()) / len(avg_ratings)
        }
```

### A/B Testing

```python
class ABTest:
    """A/B test different agent versions"""
    
    def __init__(self, agent_a, agent_b):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.results = {"a": [], "b": []}
    
    def run_test(self, test_cases: List[str], evaluator) -> dict:
        """Run A/B test"""
        
        for i, test_case in enumerate(test_cases):
            # Alternate between agents
            if i % 2 == 0:
                agent = self.agent_a
                variant = "a"
            else:
                agent = self.agent_b
                variant = "b"
            
            # Get response
            response = agent.process(test_case)
            
            # Evaluate
            evaluation = evaluator.evaluate_response(test_case, response)
            
            self.results[variant].append(evaluation)
        
        return self.compare_results()
    
    def compare_results(self) -> dict:
        """Compare A vs B"""
        avg_a = sum(r["average"] for r in self.results["a"]) / len(self.results["a"])
        avg_b = sum(r["average"] for r in self.results["b"]) / len(self.results["b"])
        
        return {
            "agent_a_score": avg_a,
            "agent_b_score": avg_b,
            "winner": "a" if avg_a > avg_b else "b",
            "difference": abs(avg_a - avg_b)
        }
```

## Automated Testing Pipeline

```python
class TestPipeline:
    """Automated testing pipeline"""
    
    def __init__(self, agent):
        self.agent = agent
        self.test_suite = AgentTestSuite(agent)
        self.metrics = AgentMetrics()
        self.evaluator = QualityEvaluator()
    
    def run_full_pipeline(self) -> dict:
        """Run complete test pipeline"""
        results = {}
        
        # 1. Unit tests
        print("Running unit tests...")
        results["unit_tests"] = self.run_unit_tests()
        
        # 2. Integration tests
        print("Running integration tests...")
        results["integration_tests"] = self.run_integration_tests()
        
        # 3. Benchmark tests
        print("Running benchmarks...")
        results["benchmarks"] = self.run_benchmarks()
        
        # 4. Quality evaluation
        print("Running quality evaluation...")
        results["quality"] = self.run_quality_evaluation()
        
        # 5. Performance metrics
        print("Collecting performance metrics...")
        results["performance"] = self.metrics.get_summary()
        
        # 6. Generate report
        report = self.generate_report(results)
        
        return report
    
    def run_unit_tests(self) -> dict:
        """Run unit tests"""
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestAgentComponents)
        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)
        
        return {
            "total": result.testsRun,
            "passed": result.testsRun - len(result.failures) - len(result.errors),
            "failed": len(result.failures) + len(result.errors)
        }
    
    def run_integration_tests(self) -> dict:
        """Run integration tests"""
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestAgentIntegration)
        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)
        
        return {
            "total": result.testsRun,
            "passed": result.testsRun - len(result.failures) - len(result.errors),
            "failed": len(result.failures) + len(result.errors)
        }
    
    def run_benchmarks(self) -> dict:
        """Run benchmark tests"""
        # Add standard benchmarks
        for test in StandardBenchmarks.get_math_benchmark():
            self.test_suite.add_test(test)
        
        for test in StandardBenchmarks.get_reasoning_benchmark():
            self.test_suite.add_test(test)
        
        return self.test_suite.run_tests()
    
    def run_quality_evaluation(self) -> dict:
        """Run quality evaluation"""
        test_cases = [
            ("What is Python?", "Python is a high-level programming language..."),
            ("How do I sort a list?", "You can use the sorted() function..."),
        ]
        
        evaluations = []
        for question, response in test_cases:
            eval_result = self.evaluator.evaluate_response(question, response)
            evaluations.append(eval_result)
        
        avg_score = sum(e["average"] for e in evaluations) / len(evaluations)
        
        return {
            "evaluations": evaluations,
            "average_score": avg_score
        }
    
    def generate_report(self, results: dict) -> dict:
        """Generate comprehensive report"""
        return {
            "timestamp": time.time(),
            "summary": {
                "unit_tests_passed": results["unit_tests"]["passed"],
                "integration_tests_passed": results["integration_tests"]["passed"],
                "benchmark_pass_rate": results["benchmarks"]["pass_rate"],
                "quality_score": results["quality"]["average_score"],
                "success_rate": results["performance"]["success_rate"]
            },
            "details": results
        }

# Usage
pipeline = TestPipeline(agent)
report = pipeline.run_full_pipeline()

print("\nTest Report Summary")
print("=" * 40)
for key, value in report["summary"].items():
    print(f"{key}: {value}")
```

## Best Practices

1. **Test early and often**: Continuous testing during development
2. **Automate testing**: Run tests automatically on changes
3. **Use multiple metrics**: Quantitative and qualitative
4. **Test edge cases**: Unusual inputs, errors, limits
5. **Benchmark regularly**: Track performance over time
6. **Get human feedback**: Automated tests aren't enough
7. **Test in production**: Monitor real usage
8. **Version your tests**: Track test changes
9. **Document failures**: Learn from what breaks
10. **Iterate based on results**: Use tests to improve

## Next Steps

You now understand evaluation and testing! Next, we'll explore monitoring and observability for production agents.
