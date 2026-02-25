# Cost Optimization

## Introduction to Cost Management

Managing costs is critical for sustainable agent systems. This section covers strategies to optimize spending while maintaining performance.

### Cost Drivers

**API Costs**:
- LLM API calls (tokens)
- Embedding generation
- Image generation
- Audio processing

**Infrastructure**:
- Compute resources
- Storage
- Network bandwidth
- Database operations

**Third-Party Services**:
- Search APIs
- Data providers
- Monitoring tools

## Token Usage Optimization

### Token Counting and Budgeting

```python
import tiktoken
from typing import Dict, List

class TokenOptimizer:
    """Optimize token usage"""
    
    def __init__(self, model: str = "gpt-4"):
        self.encoding = tiktoken.encoding_for_model(model)
        self.model = model
        self.token_costs = {
            "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
        }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def estimate_cost(self, input_text: str, output_tokens: int) -> float:
        """Estimate API call cost"""
        input_tokens = self.count_tokens(input_text)
        
        costs = self.token_costs.get(self.model, self.token_costs["gpt-4"])
        
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        
        return input_cost + output_cost
    
    def optimize_prompt(self, prompt: str, max_tokens: int) -> str:
        """Optimize prompt to fit token budget"""
        tokens = self.count_tokens(prompt)
        
        if tokens <= max_tokens:
            return prompt
        
        # Truncate to fit budget
        words = prompt.split()
        while tokens > max_tokens and words:
            words.pop()
            prompt = " ".join(words)
            tokens = self.count_tokens(prompt)
        
        return prompt
    
    def compress_context(self, messages: List[Dict], max_tokens: int) -> List[Dict]:
        """Compress conversation context"""
        total_tokens = sum(self.count_tokens(m["content"]) for m in messages)
        
        if total_tokens <= max_tokens:
            return messages
        
        # Keep system message and recent messages
        compressed = [messages[0]]  # System message
        
        # Add recent messages until budget
        for msg in reversed(messages[1:]):
            msg_tokens = self.count_tokens(msg["content"])
            if total_tokens - msg_tokens >= 0:
                compressed.insert(1, msg)
                total_tokens -= msg_tokens
            else:
                break
        
        return compressed

# Usage
optimizer = TokenOptimizer("gpt-4")

prompt = "This is a long prompt..." * 100
tokens = optimizer.count_tokens(prompt)
cost = optimizer.estimate_cost(prompt, 500)

print(f"Tokens: {tokens}, Estimated cost: ${cost:.4f}")

# Optimize
optimized = optimizer.optimize_prompt(prompt, max_tokens=1000)
```

### Caching Strategies

```python
from functools import lru_cache
import hashlib
import json
from typing import Optional

class ResponseCache:
    """Cache LLM responses"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get_cache_key(self, prompt: str, model: str, temperature: float) -> str:
        """Generate cache key"""
        key_data = f"{prompt}:{model}:{temperature}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, prompt: str, model: str, temperature: float) -> Optional[str]:
        """Get cached response"""
        key = self.get_cache_key(prompt, model, temperature)
        
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, prompt: str, model: str, temperature: float, response: str):
        """Cache response"""
        key = self.get_cache_key(prompt, model, temperature)
        
        # Evict oldest if full
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = response
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache)
        }

# Cached Agent
class CachedAgent:
    """Agent with response caching"""
    
    def __init__(self):
        self.client = openai.OpenAI()
        self.cache = ResponseCache()
    
    def generate(self, prompt: str, model: str = "gpt-4", temperature: float = 0.7) -> str:
        """Generate with caching"""
        
        # Check cache
        cached = self.cache.get(prompt, model, temperature)
        if cached:
            print("✓ Cache hit")
            return cached
        
        # Generate
        print("✗ Cache miss - calling API")
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        
        result = response.choices[0].message.content
        
        # Cache result
        self.cache.set(prompt, model, temperature, result)
        
        return result

# Usage
agent = CachedAgent()

# First call - cache miss
response1 = agent.generate("What is AI?")

# Second call - cache hit
response2 = agent.generate("What is AI?")

# Stats
stats = agent.cache.get_stats()
print(f"Cache hit rate: {stats['hit_rate']:.1%}")
```

## Model Selection

### Cost-Performance Trade-offs

```python
class ModelSelector:
    """Select optimal model based on requirements"""
    
    def __init__(self):
        self.models = {
            "gpt-4": {
                "cost_per_1k": 0.03,
                "quality": 10,
                "speed": 5
            },
            "gpt-4-turbo": {
                "cost_per_1k": 0.01,
                "quality": 9,
                "speed": 8
            },
            "gpt-3.5-turbo": {
                "cost_per_1k": 0.0005,
                "quality": 7,
                "speed": 10
            }
        }
    
    def select_model(self, 
                    priority: str = "balanced",
                    complexity: str = "medium") -> str:
        """Select best model"""
        
        if priority == "cost":
            return "gpt-3.5-turbo"
        elif priority == "quality":
            return "gpt-4"
        elif priority == "speed":
            return "gpt-3.5-turbo"
        else:  # balanced
            if complexity == "high":
                return "gpt-4-turbo"
            else:
                return "gpt-3.5-turbo"
    
    def estimate_monthly_cost(self, 
                             requests_per_day: int,
                             avg_tokens: int,
                             model: str) -> float:
        """Estimate monthly cost"""
        
        cost_per_1k = self.models[model]["cost_per_1k"]
        daily_cost = (requests_per_day * avg_tokens / 1000) * cost_per_1k
        monthly_cost = daily_cost * 30
        
        return monthly_cost

# Usage
selector = ModelSelector()

# Select for simple task
model = selector.select_model(priority="cost", complexity="low")
print(f"Selected: {model}")

# Estimate costs
monthly = selector.estimate_monthly_cost(
    requests_per_day=10000,
    avg_tokens=500,
    model="gpt-3.5-turbo"
)
print(f"Estimated monthly cost: ${monthly:.2f}")
```

## Batch Processing

### Batch API Usage

```python
class BatchProcessor:
    """Process requests in batches"""
    
    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
        self.client = openai.OpenAI()
    
    def process_batch(self, requests: List[str]) -> List[str]:
        """Process multiple requests efficiently"""
        
        results = []
        
        # Process in batches
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            
            # Process batch
            batch_results = self.process_single_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def process_single_batch(self, batch: List[str]) -> List[str]:
        """Process single batch"""
        
        # Combine into single prompt for efficiency
        combined_prompt = "Process these requests:\n\n"
        for i, req in enumerate(batch, 1):
            combined_prompt += f"{i}. {req}\n"
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": combined_prompt}]
        )
        
        # Parse results
        result_text = response.choices[0].message.content
        results = result_text.split('\n')
        
        return results[:len(batch)]

# Usage
processor = BatchProcessor(batch_size=5)
requests = [f"Summarize topic {i}" for i in range(20)]
results = processor.process_batch(requests)
```

## Resource Optimization

### Compute Optimization

```python
class ResourceOptimizer:
    """Optimize compute resources"""
    
    def __init__(self):
        self.metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "response_times": []
        }
    
    def monitor_resources(self):
        """Monitor resource usage"""
        import psutil
        
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().percent
        
        self.metrics["cpu_usage"].append(cpu)
        self.metrics["memory_usage"].append(memory)
        
        return {"cpu": cpu, "memory": memory}
    
    def should_scale(self) -> Dict:
        """Determine if scaling is needed"""
        
        if not self.metrics["cpu_usage"]:
            return {"scale": False}
        
        avg_cpu = sum(self.metrics["cpu_usage"][-10:]) / min(10, len(self.metrics["cpu_usage"]))
        avg_memory = sum(self.metrics["memory_usage"][-10:]) / min(10, len(self.metrics["memory_usage"]))
        
        scale_up = avg_cpu > 80 or avg_memory > 80
        scale_down = avg_cpu < 20 and avg_memory < 20
        
        return {
            "scale": scale_up or scale_down,
            "direction": "up" if scale_up else "down",
            "cpu": avg_cpu,
            "memory": avg_memory
        }

# Usage
optimizer = ResourceOptimizer()
resources = optimizer.monitor_resources()
scaling = optimizer.should_scale()

if scaling["scale"]:
    print(f"Scale {scaling['direction']}: CPU={scaling['cpu']:.1f}%, Memory={scaling['memory']:.1f}%")
```

## Cost Monitoring

### Real-Time Cost Tracking

```python
class CostMonitor:
    """Monitor and track costs"""
    
    def __init__(self, budget: float = 1000.0):
        self.budget = budget
        self.costs = []
        self.alerts = []
    
    def record_cost(self, amount: float, service: str, metadata: Dict = None):
        """Record cost"""
        
        cost_entry = {
            "amount": amount,
            "service": service,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        self.costs.append(cost_entry)
        
        # Check budget
        total = self.get_total_cost()
        if total > self.budget * 0.8:
            self.add_alert("warning", f"80% of budget used: ${total:.2f}")
        
        if total > self.budget:
            self.add_alert("critical", f"Budget exceeded: ${total:.2f}")
    
    def get_total_cost(self) -> float:
        """Get total cost"""
        return sum(c["amount"] for c in self.costs)
    
    def get_cost_by_service(self) -> Dict:
        """Get costs grouped by service"""
        by_service = {}
        
        for cost in self.costs:
            service = cost["service"]
            by_service[service] = by_service.get(service, 0) + cost["amount"]
        
        return by_service
    
    def add_alert(self, level: str, message: str):
        """Add cost alert"""
        alert = {
            "level": level,
            "message": message,
            "timestamp": time.time()
        }
        
        self.alerts.append(alert)
        print(f"🚨 {level.upper()}: {message}")
    
    def get_report(self) -> Dict:
        """Generate cost report"""
        total = self.get_total_cost()
        by_service = self.get_cost_by_service()
        
        return {
            "total_cost": total,
            "budget": self.budget,
            "remaining": self.budget - total,
            "utilization": (total / self.budget) * 100,
            "by_service": by_service,
            "alerts": self.alerts
        }

# Usage
monitor = CostMonitor(budget=100.0)

# Record costs
monitor.record_cost(15.50, "openai", {"model": "gpt-4"})
monitor.record_cost(2.30, "pinecone", {"operation": "query"})

# Get report
report = monitor.get_report()
print(f"Total: ${report['total_cost']:.2f}")
print(f"Budget utilization: {report['utilization']:.1f}%")
```

## Best Practices

1. **Monitor costs**: Track spending in real-time
2. **Set budgets**: Implement spending limits
3. **Cache responses**: Avoid redundant API calls
4. **Optimize prompts**: Minimize token usage
5. **Choose right model**: Balance cost and quality
6. **Batch requests**: Process multiple items together
7. **Use cheaper models**: For simple tasks
8. **Implement rate limiting**: Prevent runaway costs
9. **Regular audits**: Review and optimize
10. **Alert on anomalies**: Detect unusual spending

## Next Steps

**Module 8 (Enterprise & Scale) is complete!** You now understand architecture patterns, security & compliance, and cost optimization for production agent systems.

We've completed 8 out of 10 modules! Only Chapters 9 and 10 remain. Would you like to continue?
