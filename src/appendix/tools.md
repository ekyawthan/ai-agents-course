# Tools & Libraries

## Core Libraries

### LLM APIs

**OpenAI**
```bash
pip install openai
```
```python
from openai import OpenAI
client = OpenAI(api_key="your-key")
```
- Models: GPT-4, GPT-3.5-turbo
- Function calling support
- Streaming responses
- [Documentation](https://platform.openai.com/docs)

**Anthropic Claude**
```bash
pip install anthropic
```
```python
import anthropic
client = anthropic.Anthropic(api_key="your-key")
```
- Models: Claude 3 (Opus, Sonnet, Haiku)
- Long context windows (200K tokens)
- [Documentation](https://docs.anthropic.com)

**AWS Bedrock**
```bash
pip install boto3
```
```python
import boto3
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
```
- Multiple model providers
- Enterprise features
- [Documentation](https://docs.aws.amazon.com/bedrock)

### Agent Frameworks

**LangChain**
```bash
pip install langchain langchain-openai
```
- Chains, agents, tools
- Memory management
- [Documentation](https://python.langchain.com)

**LangGraph**
```bash
pip install langgraph
```
- Graph-based workflows
- State management
- [Documentation](https://langchain-ai.github.io/langgraph)

**AutoGPT**
```bash
git clone https://github.com/Significant-Gravitas/AutoGPT
```
- Autonomous task execution
- Plugin system

**CrewAI**
```bash
pip install crewai
```
- Multi-agent orchestration
- Role-based agents

### Vector Databases

**ChromaDB**
```bash
pip install chromadb
```
```python
import chromadb
client = chromadb.Client()
collection = client.create_collection("docs")
```
- Embedded database
- Simple API

**Pinecone**
```bash
pip install pinecone-client
```
- Managed service
- High performance
- Scalable

**Weaviate**
```bash
pip install weaviate-client
```
- Open source
- Hybrid search
- GraphQL API

### Code Analysis

**AST Tools**
```bash
pip install ast-grep-py
```
- Python: Built-in `ast` module
- Multi-language: tree-sitter

**Linters**
```bash
pip install pylint ruff mypy
```
- pylint: Comprehensive checking
- ruff: Fast linting
- mypy: Type checking

**Formatters**
```bash
pip install black isort
```
- black: Code formatting
- isort: Import sorting

### Testing

**pytest**
```bash
pip install pytest pytest-asyncio pytest-cov
```
- Unit testing
- Async support
- Coverage reports

**unittest**
- Built-in Python testing
- Standard library

### Monitoring

**Prometheus**
```bash
pip install prometheus-client
```
- Metrics collection
- Time series data

**OpenTelemetry**
```bash
pip install opentelemetry-api opentelemetry-sdk
```
- Distributed tracing
- Metrics and logs

### Utilities

**Docker SDK**
```bash
pip install docker
```
- Container management
- Safe code execution

**GitPython**
```bash
pip install gitpython
```
- Git operations
- Repository management

**Requests**
```bash
pip install requests httpx
```
- HTTP requests
- API integration

## Development Tools

### IDEs & Editors

- **VS Code**: Python, Jupyter extensions
- **PyCharm**: Professional Python IDE
- **Cursor**: AI-powered editor
- **Jupyter**: Interactive notebooks

### Debugging

- **pdb**: Python debugger
- **ipdb**: Enhanced debugger
- **pytest-pdb**: Test debugging

### Documentation

- **Sphinx**: Python documentation
- **MkDocs**: Markdown documentation
- **mdBook**: Rust-based book tool

## Deployment Tools

### Containerization

- **Docker**: Container platform
- **Docker Compose**: Multi-container apps

### Orchestration

- **Kubernetes**: Container orchestration
- **AWS ECS**: Managed containers
- **AWS Lambda**: Serverless functions

### CI/CD

- **GitHub Actions**: Automated workflows
- **GitLab CI**: Integrated CI/CD
- **AWS CodePipeline**: AWS-native CI/CD

## Quick Start Template

```python
# requirements.txt
openai==1.12.0
langchain==0.1.0
chromadb==0.4.22
fastapi==0.109.0
uvicorn==0.27.0
pytest==8.0.0
```

```python
# agent.py
from openai import OpenAI

class SimpleAgent:
    def __init__(self):
        self.client = OpenAI()
    
    def run(self, task: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": task}]
        )
        return response.choices[0].message.content

agent = SimpleAgent()
result = agent.run("Hello!")
print(result)
```
