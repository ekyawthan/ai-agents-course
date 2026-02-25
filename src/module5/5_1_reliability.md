# Reliability & Safety

## Input Validation and Sanitization

Never trust user input. Always validate and sanitize.

### Input Validation

```python
from typing import Optional
import re

class InputValidator:
    """Validate user inputs"""
    
    def __init__(self):
        self.max_input_length = 10000
        self.max_file_size = 10 * 1024 * 1024  # 10MB
    
    def validate_text_input(self, text: str) -> dict:
        """Validate text input"""
        errors = []
        
        # Check type
        if not isinstance(text, str):
            return {"valid": False, "errors": ["Input must be string"]}
        
        # Check length
        if len(text) > self.max_input_length:
            errors.append(f"Input too long (max {self.max_input_length} chars)")
        
        # Check for null bytes
        if '\x00' in text:
            errors.append("Invalid characters detected")
        
        # Check for control characters
        if any(ord(c) < 32 and c not in '\n\r\t' for c in text):
            errors.append("Control characters not allowed")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def validate_url(self, url: str) -> dict:
        """Validate URL"""
        if not isinstance(url, str):
            return {"valid": False, "errors": ["URL must be string"]}
        
        # Basic URL pattern
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        if not url_pattern.match(url):
            return {"valid": False, "errors": ["Invalid URL format"]}
        
        # Check for dangerous protocols
        if url.startswith(('file://', 'javascript:', 'data:')):
            return {"valid": False, "errors": ["Unsafe URL protocol"]}
        
        return {"valid": True, "errors": []}
    
    def validate_file_path(self, path: str, allowed_extensions: list = None) -> dict:
        """Validate file path"""
        errors = []
        
        # Check for path traversal
        if '..' in path or path.startswith('/'):
            errors.append("Path traversal detected")
        
        # Check extension
        if allowed_extensions:
            ext = path.split('.')[-1].lower()
            if ext not in allowed_extensions:
                errors.append(f"File type not allowed. Allowed: {allowed_extensions}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def sanitize_text(self, text: str) -> str:
        """Sanitize text input"""
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Remove control characters except newlines and tabs
        text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\r\t')
        
        # Trim whitespace
        text = text.strip()
        
        # Limit length
        if len(text) > self.max_input_length:
            text = text[:self.max_input_length]
        
        return text
```

### SQL Injection Prevention

```python
import sqlite3

class SafeDatabase:
    """Database access with SQL injection prevention"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def query(self, sql: str, params: tuple = ()) -> list:
        """Execute query with parameterized statements"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Always use parameterized queries
            cursor.execute(sql, params)
            results = cursor.fetchall()
            conn.close()
            return results
        except Exception as e:
            conn.close()
            raise Exception(f"Query error: {str(e)}")
    
    def safe_search(self, table: str, column: str, value: str) -> list:
        """Safe search with validation"""
        # Validate table and column names (whitelist)
        allowed_tables = ['users', 'products', 'orders']
        allowed_columns = ['name', 'email', 'description', 'title']
        
        if table not in allowed_tables:
            raise ValueError(f"Invalid table: {table}")
        
        if column not in allowed_columns:
            raise ValueError(f"Invalid column: {column}")
        
        # Use parameterized query
        sql = f"SELECT * FROM {table} WHERE {column} LIKE ?"
        return self.query(sql, (f"%{value}%",))
```

## Output Guardrails

Ensure agent outputs are safe and appropriate.

### Content Filtering

```python
class OutputGuardrails:
    """Filter and validate agent outputs"""
    
    def __init__(self):
        self.client = openai.OpenAI()
        self.blocked_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{16}\b',  # Credit card
            r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',  # Email (if needed)
        ]
    
    def check_output(self, text: str) -> dict:
        """Check if output is safe"""
        issues = []
        
        # Check for PII
        for pattern in self.blocked_patterns:
            if re.search(pattern, text):
                issues.append(f"Potential PII detected: {pattern}")
        
        # Check for harmful content
        if self.contains_harmful_content(text):
            issues.append("Potentially harmful content detected")
        
        # Check length
        if len(text) > 50000:
            issues.append("Output too long")
        
        return {
            "safe": len(issues) == 0,
            "issues": issues
        }
    
    def contains_harmful_content(self, text: str) -> bool:
        """Check for harmful content using moderation API"""
        try:
            response = self.client.moderations.create(input=text)
            result = response.results[0]
            
            # Check if any category is flagged
            return any([
                result.categories.hate,
                result.categories.violence,
                result.categories.self_harm,
                result.categories.sexual,
            ])
        except:
            return False
    
    def redact_pii(self, text: str) -> str:
        """Redact PII from text"""
        # Redact SSN
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED-SSN]', text)
        
        # Redact credit cards
        text = re.sub(r'\b\d{16}\b', '[REDACTED-CC]', text)
        
        # Redact emails (if needed)
        text = re.sub(
            r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',
            '[REDACTED-EMAIL]',
            text
        )
        
        return text
    
    def filter_output(self, text: str) -> dict:
        """Filter and clean output"""
        check = self.check_output(text)
        
        if not check['safe']:
            # Redact PII
            text = self.redact_pii(text)
            
            # Re-check
            check = self.check_output(text)
        
        return {
            "text": text,
            "safe": check['safe'],
            "issues": check['issues']
        }
```

### Response Validation

```python
class ResponseValidator:
    """Validate agent responses"""
    
    def validate_response(self, response: str, expected_format: str = None) -> dict:
        """Validate response format and content"""
        errors = []
        
        # Check not empty
        if not response or not response.strip():
            errors.append("Empty response")
        
        # Check format if specified
        if expected_format == 'json':
            try:
                json.loads(response)
            except json.JSONDecodeError:
                errors.append("Invalid JSON format")
        
        elif expected_format == 'markdown':
            # Basic markdown validation
            if not any(marker in response for marker in ['#', '*', '-', '`']):
                errors.append("Not valid markdown")
        
        # Check for refusal patterns
        refusal_patterns = [
            "I cannot", "I'm unable to", "I can't",
            "I don't have access", "I'm not able to"
        ]
        
        if any(pattern.lower() in response.lower() for pattern in refusal_patterns):
            errors.append("Agent refused to complete task")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
```

## Rate Limiting and Cost Control

Prevent runaway costs and abuse.

### Rate Limiter

```python
import time
from collections import defaultdict
from threading import Lock

class RateLimiter:
    """Rate limit API calls"""
    
    def __init__(self):
        self.requests = defaultdict(list)
        self.lock = Lock()
    
    def check_rate_limit(self, 
                         user_id: str,
                         max_requests: int = 100,
                         window_seconds: int = 3600) -> dict:
        """Check if user is within rate limit"""
        with self.lock:
            current_time = time.time()
            
            # Remove old requests outside window
            self.requests[user_id] = [
                req_time for req_time in self.requests[user_id]
                if current_time - req_time < window_seconds
            ]
            
            # Check limit
            if len(self.requests[user_id]) >= max_requests:
                return {
                    "allowed": False,
                    "remaining": 0,
                    "reset_in": window_seconds - (current_time - self.requests[user_id][0])
                }
            
            # Add current request
            self.requests[user_id].append(current_time)
            
            return {
                "allowed": True,
                "remaining": max_requests - len(self.requests[user_id]),
                "reset_in": window_seconds
            }
```

### Cost Tracker

```python
class CostTracker:
    """Track and limit API costs"""
    
    def __init__(self, max_cost_per_user: float = 10.0):
        self.costs = defaultdict(float)
        self.max_cost_per_user = max_cost_per_user
        self.lock = Lock()
    
    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for API call"""
        # Pricing per 1K tokens (example rates)
        pricing = {
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
            'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
        }
        
        if model not in pricing:
            model = 'gpt-4'  # Default to most expensive
        
        cost = (
            (input_tokens / 1000) * pricing[model]['input'] +
            (output_tokens / 1000) * pricing[model]['output']
        )
        
        return cost
    
    def check_budget(self, user_id: str, estimated_cost: float) -> dict:
        """Check if user has budget for request"""
        with self.lock:
            current_cost = self.costs[user_id]
            
            if current_cost + estimated_cost > self.max_cost_per_user:
                return {
                    "allowed": False,
                    "current_cost": current_cost,
                    "max_cost": self.max_cost_per_user,
                    "remaining": self.max_cost_per_user - current_cost
                }
            
            return {
                "allowed": True,
                "current_cost": current_cost,
                "remaining": self.max_cost_per_user - current_cost - estimated_cost
            }
    
    def record_cost(self, user_id: str, cost: float):
        """Record actual cost"""
        with self.lock:
            self.costs[user_id] += cost
    
    def reset_user_cost(self, user_id: str):
        """Reset user's cost (e.g., monthly)"""
        with self.lock:
            self.costs[user_id] = 0.0
```

## Failure Modes and Fallbacks

Handle failures gracefully.

### Retry Logic

```python
import time
from functools import wraps

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retry with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    
                    delay = base_delay * (2 ** attempt)
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
        
        return wrapper
    return decorator

# Usage
@retry_with_backoff(max_retries=3, base_delay=1.0)
def call_api(prompt: str) -> str:
    """API call with retry"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

### Circuit Breaker

```python
class CircuitBreaker:
    """Circuit breaker pattern for API calls"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker"""
        if self.state == 'open':
            # Check if timeout has passed
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'half-open'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset if in half-open
            if self.state == 'half-open':
                self.state = 'closed'
                self.failures = 0
            
            return result
            
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            
            if self.failures >= self.failure_threshold:
                self.state = 'open'
            
            raise e
```

### Fallback Strategies

```python
class FallbackAgent:
    """Agent with fallback strategies"""
    
    def __init__(self):
        self.primary_model = "gpt-4"
        self.fallback_model = "gpt-3.5-turbo"
        self.client = openai.OpenAI()
    
    def generate_with_fallback(self, prompt: str) -> dict:
        """Try primary model, fallback to cheaper model if fails"""
        try:
            response = self.client.chat.completions.create(
                model=self.primary_model,
                messages=[{"role": "user", "content": prompt}],
                timeout=30
            )
            
            return {
                "success": True,
                "response": response.choices[0].message.content,
                "model": self.primary_model
            }
            
        except Exception as e:
            print(f"Primary model failed: {e}. Trying fallback...")
            
            try:
                response = self.client.chat.completions.create(
                    model=self.fallback_model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=30
                )
                
                return {
                    "success": True,
                    "response": response.choices[0].message.content,
                    "model": self.fallback_model,
                    "fallback": True
                }
                
            except Exception as e2:
                return {
                    "success": False,
                    "error": str(e2)
                }
    
    def execute_with_fallback(self, task: str, strategies: list) -> dict:
        """Try multiple strategies in order"""
        for i, strategy in enumerate(strategies):
            try:
                result = strategy(task)
                return {
                    "success": True,
                    "result": result,
                    "strategy": i
                }
            except Exception as e:
                if i == len(strategies) - 1:
                    return {
                        "success": False,
                        "error": f"All strategies failed. Last error: {e}"
                    }
                continue
```

## Complete Safe Agent

```python
class SafeAgent:
    """Production-ready agent with safety features"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.validator = InputValidator()
        self.guardrails = OutputGuardrails()
        self.rate_limiter = RateLimiter()
        self.cost_tracker = CostTracker()
        self.circuit_breaker = CircuitBreaker()
        self.client = openai.OpenAI()
    
    def process(self, user_input: str) -> dict:
        """Process user input safely"""
        
        # 1. Validate input
        validation = self.validator.validate_text_input(user_input)
        if not validation['valid']:
            return {
                "success": False,
                "error": "Invalid input",
                "details": validation['errors']
            }
        
        # 2. Check rate limit
        rate_check = self.rate_limiter.check_rate_limit(self.user_id)
        if not rate_check['allowed']:
            return {
                "success": False,
                "error": "Rate limit exceeded",
                "reset_in": rate_check['reset_in']
            }
        
        # 3. Sanitize input
        clean_input = self.validator.sanitize_text(user_input)
        
        # 4. Estimate cost
        estimated_tokens = len(clean_input.split()) * 1.3  # Rough estimate
        estimated_cost = self.cost_tracker.estimate_cost(
            'gpt-4',
            int(estimated_tokens),
            500  # Estimated output
        )
        
        # 5. Check budget
        budget_check = self.cost_tracker.check_budget(self.user_id, estimated_cost)
        if not budget_check['allowed']:
            return {
                "success": False,
                "error": "Budget exceeded",
                "remaining": budget_check['remaining']
            }
        
        # 6. Generate response with circuit breaker
        try:
            response = self.circuit_breaker.call(
                self._generate_response,
                clean_input
            )
        except Exception as e:
            return {
                "success": False,
                "error": f"Generation failed: {str(e)}"
            }
        
        # 7. Validate output
        filtered = self.guardrails.filter_output(response)
        
        if not filtered['safe']:
            return {
                "success": False,
                "error": "Output failed safety check",
                "issues": filtered['issues']
            }
        
        # 8. Record actual cost
        self.cost_tracker.record_cost(self.user_id, estimated_cost)
        
        return {
            "success": True,
            "response": filtered['text'],
            "cost": estimated_cost,
            "remaining_budget": budget_check['remaining'] - estimated_cost
        }
    
    @retry_with_backoff(max_retries=3)
    def _generate_response(self, prompt: str) -> str:
        """Generate response with retry"""
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Never share personal information or harmful content."
                },
                {"role": "user", "content": prompt}
            ],
            timeout=30
        )
        
        return response.choices[0].message.content

# Usage
agent = SafeAgent(user_id="user123")
result = agent.process("What is the capital of France?")

if result['success']:
    print(result['response'])
else:
    print(f"Error: {result['error']}")
```

## Best Practices

1. **Validate everything**: Never trust input
2. **Sanitize data**: Clean before processing
3. **Rate limit**: Prevent abuse
4. **Track costs**: Monitor spending
5. **Filter outputs**: Check for harmful content
6. **Implement retries**: Handle transient failures
7. **Use circuit breakers**: Prevent cascading failures
8. **Have fallbacks**: Multiple strategies
9. **Log everything**: Track for debugging
10. **Test failure modes**: Ensure graceful degradation

## Next Steps

You now understand reliability and safety! Next, we'll explore evaluation and testing to ensure your agents work correctly.
