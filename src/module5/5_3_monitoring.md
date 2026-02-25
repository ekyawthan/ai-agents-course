# Monitoring & Observability

## Logging and Tracing

Track what your agent is doing at every step.

### Structured Logging

```python
import logging
import json
from datetime import datetime
from typing import Any, Dict

class AgentLogger:
    """Structured logging for agents"""
    
    def __init__(self, agent_id: str, log_file: str = "agent.log"):
        self.agent_id = agent_id
        self.logger = logging.getLogger(agent_id)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        
        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        self.logger.addHandler(console)
    
    def log_event(self, 
                  event_type: str,
                  data: Dict[str, Any],
                  level: str = "info"):
        """Log structured event"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": self.agent_id,
            "event_type": event_type,
            "data": data
        }
        
        log_message = json.dumps(log_entry)
        
        if level == "info":
            self.logger.info(log_message)
        elif level == "warning":
            self.logger.warning(log_message)
        elif level == "error":
            self.logger.error(log_message)
        elif level == "debug":
            self.logger.debug(log_message)
    
    def log_request(self, user_id: str, input_text: str):
        """Log incoming request"""
        self.log_event("request", {
            "user_id": user_id,
            "input": input_text[:200],  # Truncate long inputs
            "input_length": len(input_text)
        })
    
    def log_response(self, user_id: str, output_text: str, execution_time: float):
        """Log response"""
        self.log_event("response", {
            "user_id": user_id,
            "output": output_text[:200],
            "output_length": len(output_text),
            "execution_time": execution_time
        })
    
    def log_tool_call(self, tool_name: str, parameters: dict, result: Any):
        """Log tool execution"""
        self.log_event("tool_call", {
            "tool": tool_name,
            "parameters": parameters,
            "result": str(result)[:200],
            "success": result is not None
        })
    
    def log_error(self, error_type: str, error_message: str, context: dict = None):
        """Log error"""
        self.log_event("error", {
            "error_type": error_type,
            "message": error_message,
            "context": context or {}
        }, level="error")

# Usage
logger = AgentLogger("agent-001")
logger.log_request("user123", "What is the weather?")
logger.log_tool_call("weather_api", {"location": "NYC"}, {"temp": 72})
logger.log_response("user123", "It's 72°F in NYC", 1.5)
```

### Distributed Tracing

```python
import uuid
from contextlib import contextmanager
from typing import Optional

class Tracer:
    """Distributed tracing for agent operations"""
    
    def __init__(self):
        self.traces = {}
        self.current_trace = None
    
    @contextmanager
    def trace(self, operation_name: str, parent_id: Optional[str] = None):
        """Create trace span"""
        span_id = str(uuid.uuid4())
        trace_id = parent_id or str(uuid.uuid4())
        
        span = {
            "span_id": span_id,
            "trace_id": trace_id,
            "operation": operation_name,
            "start_time": time.time(),
            "parent_id": parent_id,
            "children": [],
            "metadata": {}
        }
        
        # Store current trace
        previous_trace = self.current_trace
        self.current_trace = span_id
        self.traces[span_id] = span
        
        try:
            yield span
        finally:
            # End span
            span["end_time"] = time.time()
            span["duration"] = span["end_time"] - span["start_time"]
            
            # Restore previous trace
            self.current_trace = previous_trace
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to current span"""
        if self.current_trace:
            self.traces[self.current_trace]["metadata"][key] = value
    
    def get_trace(self, trace_id: str) -> dict:
        """Get full trace"""
        spans = [s for s in self.traces.values() if s["trace_id"] == trace_id]
        
        # Build tree
        root = [s for s in spans if s["parent_id"] is None][0]
        self._build_tree(root, spans)
        
        return root
    
    def _build_tree(self, node: dict, all_spans: list):
        """Build trace tree"""
        children = [s for s in all_spans if s["parent_id"] == node["span_id"]]
        node["children"] = children
        
        for child in children:
            self._build_tree(child, all_spans)

# Usage
tracer = Tracer()

with tracer.trace("agent_request") as trace:
    tracer.add_metadata("user_id", "user123")
    
    with tracer.trace("tool_call", parent_id=trace["span_id"]):
        tracer.add_metadata("tool", "search")
        # Execute tool
        pass
    
    with tracer.trace("generate_response", parent_id=trace["span_id"]):
        # Generate response
        pass

# View trace
full_trace = tracer.get_trace(trace["trace_id"])
```

## Performance Metrics

Track agent performance in real-time.

### Metrics Collector

```python
from collections import defaultdict
from threading import Lock
import time

class MetricsCollector:
    """Collect and aggregate metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.lock = Lock()
    
    def record_metric(self, name: str, value: float, tags: dict = None):
        """Record a metric value"""
        with self.lock:
            self.metrics[name].append({
                "value": value,
                "timestamp": time.time(),
                "tags": tags or {}
            })
    
    def increment_counter(self, name: str, amount: int = 1):
        """Increment counter"""
        with self.lock:
            self.counters[name] += amount
    
    def get_stats(self, name: str, window_seconds: int = 3600) -> dict:
        """Get statistics for metric"""
        with self.lock:
            current_time = time.time()
            
            # Filter to time window
            values = [
                m["value"] for m in self.metrics[name]
                if current_time - m["timestamp"] < window_seconds
            ]
            
            if not values:
                return {}
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "p50": self._percentile(values, 50),
                "p95": self._percentile(values, 95),
                "p99": self._percentile(values, 99)
            }
    
    def _percentile(self, values: list, percentile: int) -> float:
        """Calculate percentile"""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_counter(self, name: str) -> int:
        """Get counter value"""
        with self.lock:
            return self.counters[name]
    
    def reset(self):
        """Reset all metrics"""
        with self.lock:
            self.metrics.clear()
            self.counters.clear()

# Usage
metrics = MetricsCollector()

# Record metrics
metrics.record_metric("response_time", 1.5, {"user": "user123"})
metrics.record_metric("response_time", 2.1, {"user": "user456"})
metrics.increment_counter("total_requests")
metrics.increment_counter("successful_requests")

# Get stats
stats = metrics.get_stats("response_time")
print(f"Avg response time: {stats['avg']:.2f}s")
print(f"P95 response time: {stats['p95']:.2f}s")
```

### Real-Time Dashboard

```python
class MetricsDashboard:
    """Real-time metrics dashboard"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
    
    def display(self):
        """Display current metrics"""
        print("\n" + "="*60)
        print("AGENT METRICS DASHBOARD")
        print("="*60)
        
        # Request metrics
        total = self.metrics.get_counter("total_requests")
        successful = self.metrics.get_counter("successful_requests")
        failed = self.metrics.get_counter("failed_requests")
        
        print(f"\nRequests:")
        print(f"  Total: {total}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        if total > 0:
            print(f"  Success Rate: {successful/total:.1%}")
        
        # Response time
        response_stats = self.metrics.get_stats("response_time")
        if response_stats:
            print(f"\nResponse Time:")
            print(f"  Average: {response_stats['avg']:.2f}s")
            print(f"  P50: {response_stats['p50']:.2f}s")
            print(f"  P95: {response_stats['p95']:.2f}s")
            print(f"  P99: {response_stats['p99']:.2f}s")
        
        # Tool usage
        tool_calls = self.metrics.get_counter("tool_calls")
        print(f"\nTool Calls: {tool_calls}")
        
        # Cost
        total_cost = self.metrics.get_counter("total_cost_cents") / 100
        print(f"\nTotal Cost: ${total_cost:.2f}")
        
        print("="*60 + "\n")
```

## Cost Tracking

Monitor spending in real-time.

### Cost Monitor

```python
class CostMonitor:
    """Monitor and alert on costs"""
    
    def __init__(self, budget_limit: float = 100.0):
        self.budget_limit = budget_limit
        self.costs = defaultdict(float)
        self.lock = Lock()
        self.alerts = []
    
    def record_cost(self, 
                   user_id: str,
                   cost: float,
                   model: str,
                   tokens: int):
        """Record cost"""
        with self.lock:
            self.costs[user_id] += cost
            
            # Check for alerts
            if self.costs[user_id] > self.budget_limit * 0.8:
                self.add_alert(
                    "warning",
                    f"User {user_id} at 80% of budget: ${self.costs[user_id]:.2f}"
                )
            
            if self.costs[user_id] > self.budget_limit:
                self.add_alert(
                    "critical",
                    f"User {user_id} exceeded budget: ${self.costs[user_id]:.2f}"
                )
    
    def add_alert(self, level: str, message: str):
        """Add alert"""
        alert = {
            "level": level,
            "message": message,
            "timestamp": time.time()
        }
        self.alerts.append(alert)
        
        # Log alert
        if level == "critical":
            logger.log_event("cost_alert", alert, level="error")
        else:
            logger.log_event("cost_alert", alert, level="warning")
    
    def get_user_cost(self, user_id: str) -> dict:
        """Get user's cost"""
        with self.lock:
            cost = self.costs[user_id]
            return {
                "cost": cost,
                "budget": self.budget_limit,
                "remaining": self.budget_limit - cost,
                "percentage": (cost / self.budget_limit) * 100
            }
    
    def get_total_cost(self) -> float:
        """Get total cost across all users"""
        with self.lock:
            return sum(self.costs.values())
    
    def get_alerts(self, level: str = None) -> list:
        """Get alerts"""
        if level:
            return [a for a in self.alerts if a["level"] == level]
        return self.alerts
```

## User Feedback Loops

Collect and act on user feedback.

### Feedback Collector

```python
class FeedbackCollector:
    """Collect user feedback"""
    
    def __init__(self):
        self.feedback = []
        self.ratings = defaultdict(list)
    
    def collect_rating(self, 
                      user_id: str,
                      interaction_id: str,
                      rating: int,
                      comment: str = ""):
        """Collect user rating (1-5)"""
        feedback = {
            "user_id": user_id,
            "interaction_id": interaction_id,
            "rating": rating,
            "comment": comment,
            "timestamp": time.time()
        }
        
        self.feedback.append(feedback)
        self.ratings[user_id].append(rating)
        
        # Log feedback
        logger.log_event("user_feedback", feedback)
        
        # Alert on low ratings
        if rating <= 2:
            logger.log_event("low_rating", feedback, level="warning")
    
    def get_average_rating(self, user_id: str = None) -> float:
        """Get average rating"""
        if user_id:
            ratings = self.ratings[user_id]
        else:
            ratings = [f["rating"] for f in self.feedback]
        
        if not ratings:
            return 0.0
        
        return sum(ratings) / len(ratings)
    
    def get_recent_feedback(self, limit: int = 10) -> list:
        """Get recent feedback"""
        return sorted(
            self.feedback,
            key=lambda x: x["timestamp"],
            reverse=True
        )[:limit]
    
    def get_low_ratings(self, threshold: int = 2) -> list:
        """Get low-rated interactions"""
        return [
            f for f in self.feedback
            if f["rating"] <= threshold
        ]
```

### Feedback Analysis

```python
class FeedbackAnalyzer:
    """Analyze feedback patterns"""
    
    def __init__(self, feedback_collector: FeedbackCollector):
        self.collector = feedback_collector
        self.client = openai.OpenAI()
    
    def analyze_trends(self) -> dict:
        """Analyze feedback trends"""
        recent = self.collector.get_recent_feedback(limit=100)
        
        if not recent:
            return {}
        
        # Calculate trends
        ratings = [f["rating"] for f in recent]
        
        return {
            "average_rating": sum(ratings) / len(ratings),
            "total_feedback": len(recent),
            "rating_distribution": {
                "5_star": sum(1 for r in ratings if r == 5),
                "4_star": sum(1 for r in ratings if r == 4),
                "3_star": sum(1 for r in ratings if r == 3),
                "2_star": sum(1 for r in ratings if r == 2),
                "1_star": sum(1 for r in ratings if r == 1),
            }
        }
    
    def identify_issues(self) -> list:
        """Identify common issues from feedback"""
        low_ratings = self.collector.get_low_ratings()
        
        if not low_ratings:
            return []
        
        # Extract comments
        comments = [f["comment"] for f in low_ratings if f["comment"]]
        
        if not comments:
            return []
        
        # Use LLM to identify themes
        prompt = f"""Analyze these negative feedback comments and identify common themes:

{chr(10).join(comments[:20])}

List the top 3 issues:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content.split('\n')
```

## Complete Monitoring System

```python
class AgentMonitor:
    """Complete monitoring system"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger = AgentLogger(agent_id)
        self.tracer = Tracer()
        self.metrics = MetricsCollector()
        self.cost_monitor = CostMonitor()
        self.feedback = FeedbackCollector()
    
    def monitor_request(self, user_id: str, input_text: str):
        """Monitor incoming request"""
        self.logger.log_request(user_id, input_text)
        self.metrics.increment_counter("total_requests")
        
        return {
            "trace_id": str(uuid.uuid4()),
            "start_time": time.time()
        }
    
    def monitor_response(self, 
                        user_id: str,
                        output_text: str,
                        context: dict):
        """Monitor response"""
        execution_time = time.time() - context["start_time"]
        
        self.logger.log_response(user_id, output_text, execution_time)
        self.metrics.record_metric("response_time", execution_time)
        self.metrics.increment_counter("successful_requests")
    
    def monitor_tool_call(self, tool_name: str, parameters: dict, result: Any):
        """Monitor tool execution"""
        self.logger.log_tool_call(tool_name, parameters, result)
        self.metrics.increment_counter("tool_calls")
        self.metrics.increment_counter(f"tool_calls_{tool_name}")
    
    def monitor_cost(self, 
                    user_id: str,
                    model: str,
                    tokens: int,
                    cost: float):
        """Monitor cost"""
        self.cost_monitor.record_cost(user_id, cost, model, tokens)
        self.metrics.increment_counter("total_cost_cents", int(cost * 100))
    
    def monitor_error(self, error_type: str, error_message: str, context: dict):
        """Monitor error"""
        self.logger.log_error(error_type, error_message, context)
        self.metrics.increment_counter("failed_requests")
        self.metrics.increment_counter(f"error_{error_type}")
    
    def get_health_status(self) -> dict:
        """Get system health status"""
        total = self.metrics.get_counter("total_requests")
        successful = self.metrics.get_counter("successful_requests")
        failed = self.metrics.get_counter("failed_requests")
        
        success_rate = successful / total if total > 0 else 0
        
        response_stats = self.metrics.get_stats("response_time")
        avg_response_time = response_stats.get("avg", 0) if response_stats else 0
        
        # Determine health
        if success_rate < 0.9 or avg_response_time > 10:
            health = "unhealthy"
        elif success_rate < 0.95 or avg_response_time > 5:
            health = "degraded"
        else:
            health = "healthy"
        
        return {
            "status": health,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "total_requests": total,
            "failed_requests": failed,
            "total_cost": self.cost_monitor.get_total_cost()
        }
    
    def generate_report(self) -> dict:
        """Generate monitoring report"""
        return {
            "agent_id": self.agent_id,
            "timestamp": time.time(),
            "health": self.get_health_status(),
            "metrics": {
                "response_time": self.metrics.get_stats("response_time"),
                "requests": {
                    "total": self.metrics.get_counter("total_requests"),
                    "successful": self.metrics.get_counter("successful_requests"),
                    "failed": self.metrics.get_counter("failed_requests")
                },
                "tool_calls": self.metrics.get_counter("tool_calls")
            },
            "cost": {
                "total": self.cost_monitor.get_total_cost(),
                "alerts": self.cost_monitor.get_alerts()
            },
            "feedback": {
                "average_rating": self.feedback.get_average_rating(),
                "recent": self.feedback.get_recent_feedback(limit=5)
            }
        }

# Usage
monitor = AgentMonitor("agent-001")

# Monitor request
context = monitor.monitor_request("user123", "What is Python?")

# Monitor tool call
monitor.monitor_tool_call("search", {"query": "Python"}, "Results...")

# Monitor cost
monitor.monitor_cost("user123", "gpt-4", 500, 0.015)

# Monitor response
monitor.monitor_response("user123", "Python is...", context)

# Get health status
health = monitor.get_health_status()
print(f"System health: {health['status']}")

# Generate report
report = monitor.generate_report()
```

## Alerting

Set up alerts for critical issues.

### Alert Manager

```python
class AlertManager:
    """Manage alerts and notifications"""
    
    def __init__(self):
        self.alert_rules = []
        self.active_alerts = []
    
    def add_rule(self, 
                 name: str,
                 condition: Callable,
                 severity: str,
                 message: str):
        """Add alert rule"""
        self.alert_rules.append({
            "name": name,
            "condition": condition,
            "severity": severity,
            "message": message
        })
    
    def check_alerts(self, metrics: dict):
        """Check all alert rules"""
        new_alerts = []
        
        for rule in self.alert_rules:
            if rule["condition"](metrics):
                alert = {
                    "name": rule["name"],
                    "severity": rule["severity"],
                    "message": rule["message"],
                    "timestamp": time.time(),
                    "metrics": metrics
                }
                new_alerts.append(alert)
                self.trigger_alert(alert)
        
        self.active_alerts.extend(new_alerts)
        return new_alerts
    
    def trigger_alert(self, alert: dict):
        """Trigger alert notification"""
        print(f"\n🚨 ALERT [{alert['severity']}]: {alert['name']}")
        print(f"   {alert['message']}")
        
        # In production, send to:
        # - Email
        # - Slack
        # - PagerDuty
        # - etc.
    
    def get_active_alerts(self, severity: str = None) -> list:
        """Get active alerts"""
        if severity:
            return [a for a in self.active_alerts if a["severity"] == severity]
        return self.active_alerts

# Setup alerts
alerts = AlertManager()

# High error rate
alerts.add_rule(
    name="High Error Rate",
    condition=lambda m: m.get("success_rate", 1) < 0.9,
    severity="critical",
    message="Success rate below 90%"
)

# Slow response time
alerts.add_rule(
    name="Slow Response Time",
    condition=lambda m: m.get("avg_response_time", 0) > 5,
    severity="warning",
    message="Average response time above 5 seconds"
)

# High cost
alerts.add_rule(
    name="High Cost",
    condition=lambda m: m.get("total_cost", 0) > 50,
    severity="warning",
    message="Total cost exceeded $50"
)
```

## Best Practices

1. **Log everything**: Requests, responses, errors, tool calls
2. **Use structured logging**: JSON format for easy parsing
3. **Track key metrics**: Response time, success rate, cost
4. **Set up alerts**: Be notified of issues immediately
5. **Monitor costs**: Track spending in real-time
6. **Collect feedback**: Learn from users
7. **Create dashboards**: Visualize metrics
8. **Trace requests**: Follow execution flow
9. **Analyze trends**: Look for patterns over time
10. **Act on insights**: Use data to improve

## Next Steps

Chapter 5 (Production-Ready Agents) is complete! You now understand reliability, testing, and monitoring. You're ready to build production-grade agents that are safe, tested, and observable.

Would you like to continue with Chapter 6 (Specialized Agent Types)?
