# Architecture Patterns

## Module 8: Learning Objectives

By the end of this module, you will:
- ✓ Design microservices and event-driven architectures
- ✓ Implement enterprise security and compliance
- ✓ Optimize costs through caching and model selection
- ✓ Scale agents to handle production workloads
- ✓ Deploy on Kubernetes and serverless platforms

---

## Introduction to Enterprise Architecture

Enterprise-scale agent systems require robust, scalable, and maintainable architectures. This section covers proven patterns for production deployments.

### Key Requirements

**Scalability**:
- Handle increasing load
- Horizontal scaling
- Resource efficiency
- Performance optimization

**Reliability**:
- High availability (99.9%+)
- Fault tolerance
- Graceful degradation
- Disaster recovery

**Maintainability**:
- Clear separation of concerns
- Easy updates and rollbacks
- Monitoring and debugging
- Documentation

**Security**:
- Authentication and authorization
- Data encryption
- Audit logging
- Compliance

## Microservices for Agents

### Agent Microservices Architecture

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn

# Agent Service
class AgentService:
    """Core agent microservice"""
    
    def __init__(self):
        self.app = FastAPI(title="Agent Service")
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/agent/process")
        async def process_request(request: AgentRequest):
            """Process agent request"""
            try:
                result = await self.process(request)
                return {"success": True, "result": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/agent/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "service": "agent"}
    
    async def process(self, request: AgentRequest) -> Dict:
        """Process agent request"""
        # Agent logic here
        return {"response": "Processed"}
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run service"""
        uvicorn.run(self.app, host=host, port=port)

class AgentRequest(BaseModel):
    """Agent request model"""
    user_id: str
    input: str
    context: Optional[Dict[str, Any]] = None

# Tool Service
class ToolService:
    """Tool execution microservice"""
    
    def __init__(self):
        self.app = FastAPI(title="Tool Service")
        self.tools = {}
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/tools/execute")
        async def execute_tool(request: ToolRequest):
            """Execute tool"""
            try:
                result = await self.execute(request)
                return {"success": True, "result": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/tools/list")
        async def list_tools():
            """List available tools"""
            return {"tools": list(self.tools.keys())}
    
    async def execute(self, request: ToolRequest) -> Any:
        """Execute tool"""
        if request.tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {request.tool_name}")
        
        tool = self.tools[request.tool_name]
        return tool(**request.parameters)
    
    def register_tool(self, name: str, func):
        """Register tool"""
        self.tools[name] = func

class ToolRequest(BaseModel):
    """Tool request model"""
    tool_name: str
    parameters: Dict[str, Any]

# Memory Service
class MemoryService:
    """Memory management microservice"""
    
    def __init__(self):
        self.app = FastAPI(title="Memory Service")
        self.storage = {}
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/memory/store")
        async def store_memory(request: MemoryRequest):
            """Store memory"""
            self.storage[request.key] = request.value
            return {"success": True}
        
        @self.app.get("/memory/retrieve/{key}")
        async def retrieve_memory(key: str):
            """Retrieve memory"""
            value = self.storage.get(key)
            if value is None:
                raise HTTPException(status_code=404, detail="Memory not found")
            return {"key": key, "value": value}
        
        @self.app.delete("/memory/delete/{key}")
        async def delete_memory(key: str):
            """Delete memory"""
            if key in self.storage:
                del self.storage[key]
            return {"success": True}

class MemoryRequest(BaseModel):
    """Memory request model"""
    key: str
    value: Any

# API Gateway
class APIGateway:
    """API Gateway for routing requests"""
    
    def __init__(self):
        self.app = FastAPI(title="API Gateway")
        self.services = {
            "agent": "http://localhost:8000",
            "tools": "http://localhost:8001",
            "memory": "http://localhost:8002"
        }
        self.setup_routes()
    
    def setup_routes(self):
        """Setup gateway routes"""
        
        @self.app.post("/api/chat")
        async def chat(request: ChatRequest):
            """Chat endpoint"""
            import httpx
            
            # Route to agent service
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.services['agent']}/agent/process",
                    json=request.dict()
                )
                return response.json()
        
        @self.app.get("/api/health")
        async def health():
            """Check health of all services"""
            import httpx
            
            health_status = {}
            async with httpx.AsyncClient() as client:
                for service, url in self.services.items():
                    try:
                        response = await client.get(f"{url}/health", timeout=5)
                        health_status[service] = "healthy"
                    except:
                        health_status[service] = "unhealthy"
            
            return {"services": health_status}

class ChatRequest(BaseModel):
    """Chat request model"""
    user_id: str
    message: str

# Usage
if __name__ == "__main__":
    # Start services on different ports
    agent_service = AgentService()
    # agent_service.run(port=8000)
    
    tool_service = ToolService()
    # tool_service.run(port=8001)
    
    memory_service = MemoryService()
    # memory_service.run(port=8002)
    
    gateway = APIGateway()
    # gateway.app.run(port=8080)
```

### Service Communication

```python
import httpx
from typing import Optional
import asyncio

class ServiceClient:
    """Client for inter-service communication"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def call_service(self, 
                          endpoint: str,
                          method: str = "POST",
                          data: Optional[Dict] = None) -> Dict:
        """Call another service"""
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "POST":
                response = await self.client.post(url, json=data)
            elif method == "GET":
                response = await self.client.get(url)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            return {"error": str(e)}
    
    async def close(self):
        """Close client"""
        await self.client.aclose()

# Circuit Breaker for service calls
class CircuitBreaker:
    """Circuit breaker for service resilience"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func, *args, **kwargs):
        """Call function with circuit breaker"""
        
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0
            
            return result
            
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            
            if self.failures >= self.failure_threshold:
                self.state = "open"
            
            raise e

# Service Registry
class ServiceRegistry:
    """Service discovery and registration"""
    
    def __init__(self):
        self.services = {}
    
    def register(self, service_name: str, url: str, metadata: Dict = None):
        """Register service"""
        self.services[service_name] = {
            "url": url,
            "metadata": metadata or {},
            "registered_at": time.time()
        }
        print(f"✅ Registered service: {service_name} at {url}")
    
    def discover(self, service_name: str) -> Optional[str]:
        """Discover service URL"""
        service = self.services.get(service_name)
        return service["url"] if service else None
    
    def list_services(self) -> Dict:
        """List all services"""
        return self.services

# Usage
registry = ServiceRegistry()
registry.register("agent-service", "http://localhost:8000")
registry.register("tool-service", "http://localhost:8001")

# Get service URL
agent_url = registry.discover("agent-service")
```

## Event-Driven Architectures

### Message Queue Integration

```python
import json
from typing import Callable, Dict
import asyncio
from queue import Queue
import threading

class MessageBroker:
    """Simple message broker"""
    
    def __init__(self):
        self.queues = {}
        self.subscribers = {}
    
    def create_queue(self, queue_name: str):
        """Create message queue"""
        if queue_name not in self.queues:
            self.queues[queue_name] = Queue()
            self.subscribers[queue_name] = []
    
    def publish(self, queue_name: str, message: Dict):
        """Publish message to queue"""
        if queue_name not in self.queues:
            self.create_queue(queue_name)
        
        self.queues[queue_name].put(message)
        print(f"📤 Published to {queue_name}: {message}")
    
    def subscribe(self, queue_name: str, handler: Callable):
        """Subscribe to queue"""
        if queue_name not in self.queues:
            self.create_queue(queue_name)
        
        self.subscribers[queue_name].append(handler)
        print(f"📥 Subscribed to {queue_name}")
    
    def start_consumer(self, queue_name: str):
        """Start consuming messages"""
        
        def consume():
            while True:
                try:
                    message = self.queues[queue_name].get(timeout=1)
                    
                    # Call all subscribers
                    for handler in self.subscribers[queue_name]:
                        try:
                            handler(message)
                        except Exception as e:
                            print(f"❌ Handler error: {e}")
                    
                except:
                    continue
        
        thread = threading.Thread(target=consume, daemon=True)
        thread.start()

# Event-Driven Agent
class EventDrivenAgent:
    """Agent using event-driven architecture"""
    
    def __init__(self, broker: MessageBroker):
        self.broker = broker
        self.setup_subscriptions()
    
    def setup_subscriptions(self):
        """Setup event subscriptions"""
        self.broker.subscribe("user_request", self.handle_user_request)
        self.broker.subscribe("tool_result", self.handle_tool_result)
    
    def handle_user_request(self, message: Dict):
        """Handle user request event"""
        print(f"🤖 Processing request: {message}")
        
        # Process and publish result
        result = {"response": f"Processed: {message.get('input')}"}
        self.broker.publish("agent_response", result)
    
    def handle_tool_result(self, message: Dict):
        """Handle tool result event"""
        print(f"🔧 Tool result: {message}")

# Usage
broker = MessageBroker()
agent = EventDrivenAgent(broker)

# Start consumers
broker.start_consumer("user_request")
broker.start_consumer("tool_result")

# Publish event
broker.publish("user_request", {"user_id": "123", "input": "Hello"})
```

### Kafka Integration

```python
from kafka import KafkaProducer, KafkaConsumer
import json

class KafkaAgentSystem:
    """Agent system using Kafka"""
    
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.bootstrap_servers = bootstrap_servers
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    
    def publish_event(self, topic: str, event: Dict):
        """Publish event to Kafka"""
        self.producer.send(topic, event)
        self.producer.flush()
        print(f"📤 Published to {topic}")
    
    def create_consumer(self, topic: str, group_id: str):
        """Create Kafka consumer"""
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        return consumer
    
    def consume_events(self, topic: str, group_id: str, handler: Callable):
        """Consume events from Kafka"""
        consumer = self.create_consumer(topic, group_id)
        
        for message in consumer:
            try:
                handler(message.value)
            except Exception as e:
                print(f"❌ Error processing message: {e}")

# Usage
# kafka_system = KafkaAgentSystem()
# kafka_system.publish_event("agent-requests", {"user_id": "123", "input": "Hello"})
```

## Serverless Deployments

### AWS Lambda Agent

```python
import json
import boto3
from typing import Dict, Any

class LambdaAgent:
    """Agent deployed as AWS Lambda"""
    
    def __init__(self):
        self.client = openai.OpenAI()
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table('agent-memory')
    
    def handler(self, event: Dict, context: Any) -> Dict:
        """Lambda handler function"""
        
        try:
            # Parse request
            body = json.loads(event.get('body', '{}'))
            user_id = body.get('user_id')
            input_text = body.get('input')
            
            # Get user memory
            memory = self.get_memory(user_id)
            
            # Process request
            response = self.process(input_text, memory)
            
            # Update memory
            self.update_memory(user_id, response)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'response': response
                })
            }
            
        except Exception as e:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': str(e)
                })
            }
    
    def process(self, input_text: str, memory: Dict) -> str:
        """Process request"""
        # Build context from memory
        context = memory.get('context', '')
        
        messages = [
            {"role": "system", "content": f"Context: {context}"},
            {"role": "user", "content": input_text}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        return response.choices[0].message.content
    
    def get_memory(self, user_id: str) -> Dict:
        """Get user memory from DynamoDB"""
        try:
            response = self.table.get_item(Key={'user_id': user_id})
            return response.get('Item', {})
        except:
            return {}
    
    def update_memory(self, user_id: str, response: str):
        """Update user memory"""
        try:
            self.table.put_item(
                Item={
                    'user_id': user_id,
                    'context': response,
                    'updated_at': int(time.time())
                }
            )
        except Exception as e:
            print(f"Error updating memory: {e}")

# Lambda function
def lambda_handler(event, context):
    """AWS Lambda entry point"""
    agent = LambdaAgent()
    return agent.handler(event, context)
```

### Serverless Framework Configuration

```yaml
# serverless.yml
service: agent-service

provider:
  name: aws
  runtime: python3.11
  region: us-east-1
  environment:
    OPENAI_API_KEY: ${env:OPENAI_API_KEY}
  iamRoleStatements:
    - Effect: Allow
      Action:
        - dynamodb:GetItem
        - dynamodb:PutItem
      Resource: "arn:aws:dynamodb:*:*:table/agent-memory"

functions:
  agent:
    handler: handler.lambda_handler
    events:
      - http:
          path: agent/process
          method: post
          cors: true
    timeout: 30
    memorySize: 512

resources:
  Resources:
    AgentMemoryTable:
      Type: AWS::DynamoDB::Table
      Properties:
        TableName: agent-memory
        AttributeDefinitions:
          - AttributeName: user_id
            AttributeType: S
        KeySchema:
          - AttributeName: user_id
            KeyType: HASH
        BillingMode: PAY_PER_REQUEST
```

## Scaling Strategies

### Horizontal Scaling

```python
from multiprocessing import Pool, cpu_count
import concurrent.futures

class ScalableAgentPool:
    """Pool of agent workers for horizontal scaling"""
    
    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or cpu_count()
        self.pool = Pool(processes=self.num_workers)
        print(f"🔧 Created pool with {self.num_workers} workers")
    
    def process_batch(self, requests: List[Dict]) -> List[Dict]:
        """Process batch of requests in parallel"""
        results = self.pool.map(self.process_single, requests)
        return results
    
    def process_single(self, request: Dict) -> Dict:
        """Process single request"""
        # Agent processing logic
        return {"response": f"Processed: {request.get('input')}"}
    
    def close(self):
        """Close pool"""
        self.pool.close()
        self.pool.join()

# Async scaling
class AsyncAgentPool:
    """Async agent pool"""
    
    def __init__(self, max_workers: int = 10):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_batch(self, requests: List[Dict]) -> List[Dict]:
        """Process batch asynchronously"""
        loop = asyncio.get_event_loop()
        
        tasks = [
            loop.run_in_executor(self.executor, self.process_single, req)
            for req in requests
        ]
        
        results = await asyncio.gather(*tasks)
        return results
    
    def process_single(self, request: Dict) -> Dict:
        """Process single request"""
        return {"response": f"Processed: {request.get('input')}"}

# Usage
pool = ScalableAgentPool(num_workers=4)

requests = [
    {"input": f"Request {i}"} for i in range(100)
]

results = pool.process_batch(requests)
print(f"Processed {len(results)} requests")

pool.close()
```

### Load Balancing

```python
from typing import List
import random

class LoadBalancer:
    """Load balancer for agent instances"""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.instances = []
        self.current_index = 0
        self.instance_loads = {}
    
    def register_instance(self, instance_url: str):
        """Register agent instance"""
        self.instances.append(instance_url)
        self.instance_loads[instance_url] = 0
        print(f"✅ Registered instance: {instance_url}")
    
    def get_instance(self) -> str:
        """Get instance based on strategy"""
        
        if self.strategy == "round_robin":
            return self.round_robin()
        elif self.strategy == "least_connections":
            return self.least_connections()
        elif self.strategy == "random":
            return self.random_selection()
        else:
            return self.round_robin()
    
    def round_robin(self) -> str:
        """Round-robin selection"""
        if not self.instances:
            raise Exception("No instances available")
        
        instance = self.instances[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.instances)
        return instance
    
    def least_connections(self) -> str:
        """Select instance with least connections"""
        if not self.instances:
            raise Exception("No instances available")
        
        return min(self.instance_loads, key=self.instance_loads.get)
    
    def random_selection(self) -> str:
        """Random selection"""
        if not self.instances:
            raise Exception("No instances available")
        
        return random.choice(self.instances)
    
    def record_request(self, instance_url: str):
        """Record request to instance"""
        self.instance_loads[instance_url] += 1
    
    def record_completion(self, instance_url: str):
        """Record request completion"""
        self.instance_loads[instance_url] -= 1

# Usage
lb = LoadBalancer(strategy="least_connections")
lb.register_instance("http://agent1:8000")
lb.register_instance("http://agent2:8000")
lb.register_instance("http://agent3:8000")

# Route request
instance = lb.get_instance()
print(f"Routing to: {instance}")
```

## Container Orchestration

### Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  agent-service:
    build: ./agent-service
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - postgres
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1'
          memory: 1G
  
  tool-service:
    build: ./tool-service
    ports:
      - "8001:8001"
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
  
  memory-service:
    build: ./memory-service
    ports:
      - "8002:8002"
    environment:
      - POSTGRES_URL=postgresql://user:pass@postgres:5432/agentdb
    depends_on:
      - postgres
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
  
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=agentdb
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - agent-service

volumes:
  redis-data:
  postgres-data:
```

### Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-service
  template:
    metadata:
      labels:
        app: agent-service
    spec:
      containers:
      - name: agent
        image: agent-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: agent-service
spec:
  selector:
    app: agent-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Best Practices

1. **Decouple services**: Loose coupling, high cohesion
2. **Stateless design**: Store state externally
3. **Idempotent operations**: Safe to retry
4. **Circuit breakers**: Prevent cascading failures
5. **Health checks**: Monitor service health
6. **Graceful shutdown**: Clean resource cleanup
7. **Configuration management**: Externalize config
8. **Service discovery**: Dynamic service location
9. **API versioning**: Backward compatibility
10. **Documentation**: Clear API contracts

## Next Steps

You now understand enterprise architecture patterns! Next, we'll explore security and compliance for production agent systems.
