# Task Automation Agents

## Introduction to Task Automation

Task automation agents handle repetitive workflows, orchestrate complex processes, and integrate with existing tools to save time and reduce errors.

### What Makes Automation Agents Special?

**Core Capabilities**:
- Workflow orchestration
- Event-driven triggers
- Integration with multiple tools
- Scheduled operations
- Error handling and recovery
- State management across tasks

**Key Benefits**:
- Eliminate repetitive work
- Reduce human error
- 24/7 operation
- Consistent execution
- Scalable processing
- Audit trails

### Types of Automation Agents

1. **Workflow Agents**: Multi-step process automation
2. **Scheduling Agents**: Time-based task execution
3. **Integration Agents**: Connect different systems
4. **Monitoring Agents**: Watch and respond to events
5. **Data Processing Agents**: ETL and transformation

## Workflow Orchestration

### Building Workflow Engine

```python
from dataclasses import dataclass
from typing import List, Dict, Callable, Any
from enum import Enum
import time

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class Task:
    """Single task in workflow"""
    id: str
    name: str
    action: Callable
    params: Dict[str, Any]
    dependencies: List[str] = None
    retry_count: int = 3
    timeout: int = 300
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = None

class WorkflowEngine:
    """Orchestrate complex workflows"""
    
    def __init__(self):
        self.tasks = {}
        self.execution_log = []
    
    def add_task(self, task: Task):
        """Add task to workflow"""
        self.tasks[task.id] = task
    
    def execute_workflow(self) -> Dict:
        """Execute all tasks respecting dependencies"""
        print("🚀 Starting workflow execution\n")
        
        completed = set()
        failed = set()
        
        while len(completed) + len(failed) < len(self.tasks):
            # Find tasks ready to execute
            ready_tasks = self.get_ready_tasks(completed, failed)
            
            if not ready_tasks:
                # Check if we're stuck
                pending = [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]
                if pending:
                    print("⚠️  Workflow stuck - circular dependencies or all tasks failed")
                    break
                else:
                    break
            
            # Execute ready tasks
            for task in ready_tasks:
                result = self.execute_task(task)
                
                if result['success']:
                    completed.add(task.id)
                else:
                    failed.add(task.id)
        
        return self.generate_report(completed, failed)
    
    def get_ready_tasks(self, completed: set, failed: set) -> List[Task]:
        """Get tasks ready to execute"""
        ready = []
        
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            
            # Check dependencies
            if task.dependencies:
                deps_met = all(dep in completed for dep in task.dependencies)
                deps_failed = any(dep in failed for dep in task.dependencies)
                
                if deps_failed:
                    task.status = TaskStatus.SKIPPED
                    task.error = "Dependency failed"
                    continue
                
                if not deps_met:
                    continue
            
            ready.append(task)
        
        return ready
    
    def execute_task(self, task: Task) -> Dict:
        """Execute single task with retry logic"""
        print(f"▶️  Executing: {task.name}")
        task.status = TaskStatus.RUNNING
        
        for attempt in range(task.retry_count):
            try:
                # Execute task action
                start_time = time.time()
                result = task.action(**task.params)
                execution_time = time.time() - start_time
                
                # Success
                task.status = TaskStatus.COMPLETED
                task.result = result
                
                log_entry = {
                    "task_id": task.id,
                    "task_name": task.name,
                    "status": "success",
                    "execution_time": execution_time,
                    "attempt": attempt + 1
                }
                self.execution_log.append(log_entry)
                
                print(f"✅ Completed: {task.name} ({execution_time:.2f}s)\n")
                
                return {"success": True, "result": result}
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Attempt {attempt + 1} failed: {error_msg}")
                
                if attempt < task.retry_count - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Final failure
                    task.status = TaskStatus.FAILED
                    task.error = error_msg
                    
                    log_entry = {
                        "task_id": task.id,
                        "task_name": task.name,
                        "status": "failed",
                        "error": error_msg,
                        "attempts": task.retry_count
                    }
                    self.execution_log.append(log_entry)
                    
                    print(f"💥 Failed: {task.name}\n")
                    
                    return {"success": False, "error": error_msg}
    
    def generate_report(self, completed: set, failed: set) -> Dict:
        """Generate execution report"""
        total = len(self.tasks)
        skipped = sum(1 for t in self.tasks.values() if t.status == TaskStatus.SKIPPED)
        
        report = {
            "total_tasks": total,
            "completed": len(completed),
            "failed": len(failed),
            "skipped": skipped,
            "success_rate": len(completed) / total if total > 0 else 0,
            "execution_log": self.execution_log
        }
        
        print("=" * 50)
        print("WORKFLOW EXECUTION REPORT")
        print("=" * 50)
        print(f"Total Tasks: {total}")
        print(f"Completed: {len(completed)}")
        print(f"Failed: {len(failed)}")
        print(f"Skipped: {skipped}")
        print(f"Success Rate: {report['success_rate']:.1%}")
        print("=" * 50)
        
        return report

# Usage
workflow = WorkflowEngine()

# Define tasks
def fetch_data(source):
    print(f"  Fetching from {source}...")
    time.sleep(1)
    return {"data": f"Data from {source}"}

def process_data(data):
    print(f"  Processing data...")
    time.sleep(1)
    return {"processed": True}

def save_results(data):
    print(f"  Saving results...")
    time.sleep(1)
    return {"saved": True}

# Add tasks
workflow.add_task(Task(
    id="fetch",
    name="Fetch Data",
    action=fetch_data,
    params={"source": "API"}
))

workflow.add_task(Task(
    id="process",
    name="Process Data",
    action=process_data,
    params={"data": {}},
    dependencies=["fetch"]
))

workflow.add_task(Task(
    id="save",
    name="Save Results",
    action=save_results,
    params={"data": {}},
    dependencies=["process"]
))

# Execute
report = workflow.execute_workflow()
```

### Parallel Workflow Execution

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ParallelWorkflowEngine(WorkflowEngine):
    """Execute independent tasks in parallel"""
    
    def __init__(self, max_workers: int = 4):
        super().__init__()
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def execute_workflow_async(self) -> Dict:
        """Execute workflow with parallel execution"""
        print("🚀 Starting parallel workflow execution\n")
        
        completed = set()
        failed = set()
        
        while len(completed) + len(failed) < len(self.tasks):
            # Get ready tasks
            ready_tasks = self.get_ready_tasks(completed, failed)
            
            if not ready_tasks:
                break
            
            # Execute tasks in parallel
            tasks_futures = [
                self.execute_task_async(task)
                for task in ready_tasks
            ]
            
            results = await asyncio.gather(*tasks_futures)
            
            # Update completed/failed
            for task, result in zip(ready_tasks, results):
                if result['success']:
                    completed.add(task.id)
                else:
                    failed.add(task.id)
        
        return self.generate_report(completed, failed)
    
    async def execute_task_async(self, task: Task) -> Dict:
        """Execute task asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.execute_task,
            task
        )

# Usage
async def main():
    workflow = ParallelWorkflowEngine(max_workers=3)
    
    # Add independent tasks that can run in parallel
    for i in range(5):
        workflow.add_task(Task(
            id=f"task_{i}",
            name=f"Task {i}",
            action=lambda x: time.sleep(1) or f"Result {x}",
            params={"x": i}
        ))
    
    report = await workflow.execute_workflow_async()

# Run
# asyncio.run(main())
```

## Scheduled Operations

### Task Scheduler

```python
from datetime import datetime, timedelta
import schedule
import threading

class TaskScheduler:
    """Schedule tasks to run at specific times"""
    
    def __init__(self):
        self.scheduled_tasks = []
        self.running = False
        self.thread = None
    
    def schedule_task(self, 
                     task: Callable,
                     schedule_type: str,
                     time_spec: str = None,
                     **kwargs):
        """Schedule a task"""
        
        if schedule_type == "daily":
            job = schedule.every().day.at(time_spec).do(task, **kwargs)
        
        elif schedule_type == "hourly":
            job = schedule.every().hour.do(task, **kwargs)
        
        elif schedule_type == "interval":
            minutes = int(time_spec)
            job = schedule.every(minutes).minutes.do(task, **kwargs)
        
        elif schedule_type == "weekly":
            day, time = time_spec.split()
            job = getattr(schedule.every(), day.lower()).at(time).do(task, **kwargs)
        
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        self.scheduled_tasks.append({
            "job": job,
            "task": task.__name__,
            "schedule": schedule_type,
            "time_spec": time_spec
        })
        
        print(f"📅 Scheduled: {task.__name__} - {schedule_type} {time_spec or ''}")
    
    def start(self):
        """Start scheduler"""
        self.running = True
        self.thread = threading.Thread(target=self._run_scheduler)
        self.thread.daemon = True
        self.thread.start()
        print("🕐 Scheduler started")
    
    def stop(self):
        """Stop scheduler"""
        self.running = False
        if self.thread:
            self.thread.join()
        print("🛑 Scheduler stopped")
    
    def _run_scheduler(self):
        """Run scheduler loop"""
        while self.running:
            schedule.run_pending()
            time.sleep(1)
    
    def list_scheduled_tasks(self) -> List[Dict]:
        """List all scheduled tasks"""
        return self.scheduled_tasks

# Usage
scheduler = TaskScheduler()

def backup_database():
    print(f"💾 Running database backup at {datetime.now()}")
    # Backup logic here

def send_report():
    print(f"📊 Sending daily report at {datetime.now()}")
    # Report logic here

def cleanup_temp_files():
    print(f"🧹 Cleaning temp files at {datetime.now()}")
    # Cleanup logic here

# Schedule tasks
scheduler.schedule_task(backup_database, "daily", "02:00")
scheduler.schedule_task(send_report, "daily", "09:00")
scheduler.schedule_task(cleanup_temp_files, "interval", "60")  # Every hour

# Start scheduler
scheduler.start()

# Keep running
# try:
#     while True:
#         time.sleep(1)
# except KeyboardInterrupt:
#     scheduler.stop()
```

### Cron-Style Scheduling

```python
from crontab import CronTab

class CronScheduler:
    """Cron-style task scheduling"""
    
    def __init__(self):
        self.cron = CronTab(user=True)
    
    def add_cron_job(self, 
                     command: str,
                     schedule: str,
                     comment: str = None):
        """Add cron job
        
        Schedule format: "minute hour day month weekday"
        Examples:
        - "0 2 * * *" - Daily at 2 AM
        - "*/15 * * * *" - Every 15 minutes
        - "0 9 * * 1-5" - Weekdays at 9 AM
        """
        job = self.cron.new(command=command, comment=comment)
        job.setall(schedule)
        self.cron.write()
        
        print(f"✅ Added cron job: {comment or command}")
        print(f"   Schedule: {schedule}")
    
    def list_jobs(self) -> List[Dict]:
        """List all cron jobs"""
        jobs = []
        for job in self.cron:
            jobs.append({
                "command": job.command,
                "schedule": str(job.slices),
                "comment": job.comment,
                "enabled": job.is_enabled()
            })
        return jobs
    
    def remove_job(self, comment: str):
        """Remove job by comment"""
        self.cron.remove_all(comment=comment)
        self.cron.write()
        print(f"🗑️  Removed job: {comment}")

# Usage
# cron = CronScheduler()
# cron.add_cron_job(
#     "python /path/to/backup.py",
#     "0 2 * * *",
#     "Daily backup"
# )
```

## Event-Driven Triggers

### Event Listener System

```python
from typing import Callable, Dict, List
from queue import Queue
import threading

class EventType(Enum):
    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    API_CALL = "api_call"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    ERROR_OCCURRED = "error_occurred"

@dataclass
class Event:
    """Event data"""
    type: EventType
    data: Dict[str, Any]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class EventDrivenAgent:
    """Agent that responds to events"""
    
    def __init__(self):
        self.handlers = {}
        self.event_queue = Queue()
        self.running = False
        self.thread = None
    
    def register_handler(self, event_type: EventType, handler: Callable):
        """Register event handler"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        
        self.handlers[event_type].append(handler)
        print(f"📝 Registered handler for {event_type.value}")
    
    def emit_event(self, event: Event):
        """Emit an event"""
        self.event_queue.put(event)
    
    def start(self):
        """Start event processing"""
        self.running = True
        self.thread = threading.Thread(target=self._process_events)
        self.thread.daemon = True
        self.thread.start()
        print("🎯 Event processor started")
    
    def stop(self):
        """Stop event processing"""
        self.running = False
        if self.thread:
            self.thread.join()
        print("🛑 Event processor stopped")
    
    def _process_events(self):
        """Process events from queue"""
        while self.running:
            try:
                event = self.event_queue.get(timeout=1)
                self._handle_event(event)
            except:
                continue
    
    def _handle_event(self, event: Event):
        """Handle single event"""
        print(f"⚡ Event: {event.type.value}")
        
        handlers = self.handlers.get(event.type, [])
        
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"❌ Handler error: {e}")

# Usage
agent = EventDrivenAgent()

# Register handlers
def on_file_created(event: Event):
    print(f"  📄 File created: {event.data['filename']}")
    # Process new file

def on_threshold_exceeded(event: Event):
    print(f"  ⚠️  Threshold exceeded: {event.data['metric']} = {event.data['value']}")
    # Send alert

def on_error(event: Event):
    print(f"  💥 Error occurred: {event.data['error']}")
    # Log and notify

agent.register_handler(EventType.FILE_CREATED, on_file_created)
agent.register_handler(EventType.THRESHOLD_EXCEEDED, on_threshold_exceeded)
agent.register_handler(EventType.ERROR_OCCURRED, on_error)

# Start processing
agent.start()

# Emit events
agent.emit_event(Event(
    type=EventType.FILE_CREATED,
    data={"filename": "data.csv"}
))

agent.emit_event(Event(
    type=EventType.THRESHOLD_EXCEEDED,
    data={"metric": "cpu_usage", "value": 95}
))
```

### File System Watcher

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FileWatcher(FileSystemEventHandler):
    """Watch file system for changes"""
    
    def __init__(self, agent: EventDrivenAgent):
        self.agent = agent
    
    def on_created(self, event):
        """File created"""
        if not event.is_directory:
            self.agent.emit_event(Event(
                type=EventType.FILE_CREATED,
                data={"path": event.src_path}
            ))
    
    def on_modified(self, event):
        """File modified"""
        if not event.is_directory:
            self.agent.emit_event(Event(
                type=EventType.FILE_MODIFIED,
                data={"path": event.src_path}
            ))
    
    def on_deleted(self, event):
        """File deleted"""
        if not event.is_directory:
            self.agent.emit_event(Event(
                type=EventType.FILE_DELETED,
                data={"path": event.src_path}
            ))

def start_file_watcher(path: str, agent: EventDrivenAgent):
    """Start watching directory"""
    event_handler = FileWatcher(agent)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    print(f"👁️  Watching: {path}")
    return observer

# Usage
# observer = start_file_watcher("/path/to/watch", agent)
```

## Integration with Existing Tools

### Tool Integration Framework

```python
class ToolIntegration:
    """Integrate with external tools"""
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, name: str, connector: Callable):
        """Register tool connector"""
        self.tools[name] = connector
        print(f"🔌 Registered tool: {name}")
    
    def execute_tool(self, name: str, action: str, **params) -> Dict:
        """Execute tool action"""
        if name not in self.tools:
            return {"success": False, "error": f"Tool not found: {name}"}
        
        try:
            result = self.tools[name](action, **params)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

# Example integrations

def slack_connector(action: str, **params):
    """Slack integration"""
    if action == "send_message":
        channel = params.get("channel")
        message = params.get("message")
        # Send to Slack API
        print(f"📱 Slack: Sending to {channel}: {message}")
        return {"sent": True}
    
    elif action == "get_messages":
        channel = params.get("channel")
        # Get from Slack API
        return {"messages": []}

def email_connector(action: str, **params):
    """Email integration"""
    if action == "send":
        to = params.get("to")
        subject = params.get("subject")
        body = params.get("body")
        # Send email
        print(f"📧 Email: Sending to {to}")
        return {"sent": True}

def database_connector(action: str, **params):
    """Database integration"""
    if action == "query":
        sql = params.get("sql")
        # Execute query
        print(f"🗄️  Database: Executing query")
        return {"rows": []}
    
    elif action == "insert":
        table = params.get("table")
        data = params.get("data")
        # Insert data
        return {"inserted": True}

# Setup
integrations = ToolIntegration()
integrations.register_tool("slack", slack_connector)
integrations.register_tool("email", email_connector)
integrations.register_tool("database", database_connector)

# Use
integrations.execute_tool(
    "slack",
    "send_message",
    channel="#general",
    message="Task completed!"
)
```

## Complete Automation Agent

```python
class AutomationAgent:
    """Complete task automation agent"""
    
    def __init__(self):
        self.workflow_engine = WorkflowEngine()
        self.scheduler = TaskScheduler()
        self.event_agent = EventDrivenAgent()
        self.integrations = ToolIntegration()
        self.client = openai.OpenAI()
    
    def create_automation(self, description: str) -> Dict:
        """Create automation from natural language"""
        
        # Parse description to understand automation
        automation_spec = self.parse_automation_description(description)
        
        # Create workflow
        workflow_id = self.create_workflow(automation_spec)
        
        # Setup triggers
        if automation_spec.get("trigger_type") == "schedule":
            self.setup_scheduled_trigger(workflow_id, automation_spec)
        elif automation_spec.get("trigger_type") == "event":
            self.setup_event_trigger(workflow_id, automation_spec)
        
        return {
            "workflow_id": workflow_id,
            "automation_spec": automation_spec,
            "status": "active"
        }
    
    def parse_automation_description(self, description: str) -> Dict:
        """Parse natural language automation description"""
        prompt = f"""Parse this automation request into a structured specification:

"{description}"

Provide JSON with:
- trigger_type: "schedule" or "event"
- trigger_spec: schedule time or event type
- steps: list of actions to perform
- integrations: tools needed

Specification:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        import json
        return json.loads(response.choices[0].message.content)
    
    def create_workflow(self, spec: Dict) -> str:
        """Create workflow from specification"""
        workflow_id = f"workflow_{int(time.time())}"
        
        for i, step in enumerate(spec.get("steps", [])):
            task = Task(
                id=f"{workflow_id}_step_{i}",
                name=step.get("name"),
                action=self.create_action_from_spec(step),
                params=step.get("params", {}),
                dependencies=step.get("dependencies", [])
            )
            self.workflow_engine.add_task(task)
        
        return workflow_id
    
    def create_action_from_spec(self, step_spec: Dict) -> Callable:
        """Create executable action from step specification"""
        action_type = step_spec.get("action_type")
        
        if action_type == "api_call":
            def action(**params):
                return self.integrations.execute_tool(
                    step_spec["tool"],
                    step_spec["action"],
                    **params
                )
            return action
        
        elif action_type == "data_processing":
            def action(**params):
                # Process data
                return {"processed": True}
            return action
        
        else:
            def action(**params):
                print(f"Executing: {step_spec.get('name')}")
                return {"done": True}
            return action
    
    def setup_scheduled_trigger(self, workflow_id: str, spec: Dict):
        """Setup scheduled trigger for workflow"""
        def run_workflow():
            print(f"🔄 Running scheduled workflow: {workflow_id}")
            self.workflow_engine.execute_workflow()
        
        self.scheduler.schedule_task(
            run_workflow,
            spec["trigger_spec"]["type"],
            spec["trigger_spec"]["time"]
        )
    
    def setup_event_trigger(self, workflow_id: str, spec: Dict):
        """Setup event trigger for workflow"""
        event_type = EventType[spec["trigger_spec"]["event"]]
        
        def on_event(event: Event):
            print(f"🎯 Event triggered workflow: {workflow_id}")
            self.workflow_engine.execute_workflow()
        
        self.event_agent.register_handler(event_type, on_event)

# Usage
agent = AutomationAgent()

# Create automation from description
automation = agent.create_automation("""
Every day at 9 AM:
1. Fetch data from the API
2. Process and analyze the data
3. Generate a report
4. Send the report via email to team@company.com
""")

print(f"Created automation: {automation['workflow_id']}")
```

## Best Practices

1. **Idempotency**: Tasks should be safely re-runnable
2. **Error handling**: Always handle failures gracefully
3. **Logging**: Track all automation executions
4. **Monitoring**: Alert on failures
5. **Testing**: Test workflows before production
6. **Documentation**: Document automation logic
7. **Versioning**: Track automation changes
8. **Rollback**: Ability to revert changes
9. **Rate limiting**: Don't overwhelm systems
10. **Security**: Secure credentials and access

---

## Practice Exercises

### Exercise 1: Email Automation Agent (Medium)
**Task**: Build an agent that processes emails and takes actions.

<details>
<summary>Click to see solution</summary>

```python
class EmailAgent:
    def process_email(self, email: Dict) -> Dict:
        # Classify email
        category = self.classify(email["subject"])
        
        # Route based on category
        if category == "urgent":
            return self.escalate(email)
        elif category == "question":
            return self.auto_respond(email)
        else:
            return self.archive(email)
```
</details>

### Exercise 2: Workflow Orchestrator (Hard)
**Task**: Create an orchestrator that manages complex multi-step workflows.

<details>
<summary>Click to see solution</summary>

```python
class WorkflowOrchestrator:
    def execute_workflow(self, workflow: Dict) -> Dict:
        results = {}
        for step in workflow["steps"]:
            if self.check_conditions(step, results):
                result = self.execute_step(step)
                results[step["id"]] = result
        return results
```
</details>

---

> **✅ Chapter 6 Summary**
>
> You've mastered specialized agent types:
> - **Coding Agents**: Analyze, generate, refactor, and test code
> - **Research Agents**: Multi-source search, verification, and synthesis
> - **Automation Agents**: Workflow orchestration, scheduling, and event-driven tasks
>
> These specialized agents demonstrate how to focus agent capabilities on specific domains for maximum effectiveness.

## Next Steps

Chapter 6 (Specialized Agent Types) is complete! You now have deep knowledge of coding agents, research agents, and task automation agents. These specialized agents form the foundation for building powerful, domain-specific AI systems.
