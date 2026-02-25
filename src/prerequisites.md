# Prerequisites

## Required Knowledge

### Programming Fundamentals
- **Python proficiency**: Functions, classes, decorators, async/await
- **Data structures**: Lists, dicts, sets, queues
- **Error handling**: Try/except, custom exceptions
- **File I/O**: Reading/writing files

### Basic Concepts
- **APIs**: REST APIs, HTTP methods, JSON
- **Command line**: Basic bash/terminal commands
- **Git**: Version control basics
- **Environment variables**: Configuration management

### Recommended (Not Required)
- Machine learning basics
- Natural language processing concepts
- Docker/containerization
- Cloud platforms (AWS, Azure, GCP)

## Technical Requirements

### Software
- **Python 3.9+**: [Download](https://www.python.org/downloads/)
- **pip**: Package manager (comes with Python)
- **Git**: [Download](https://git-scm.com/downloads)
- **Code editor**: VS Code, PyCharm, or similar
- **Terminal**: Command line access

### Accounts
- **OpenAI API key**: [Get key](https://platform.openai.com/api-keys)
  - Or Anthropic, AWS Bedrock, etc.
- **GitHub account**: For version control
- **Optional**: Cloud provider account (AWS, GCP, Azure)

### Hardware
- **Minimum**: 8GB RAM, modern CPU
- **Recommended**: 16GB RAM, GPU for local models
- **Internet**: Stable connection for API calls

## Setup Instructions

### 1. Install Python

```bash
# Check Python version
python --version  # Should be 3.9+

# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 2. Install Core Libraries

```bash
pip install openai langchain chromadb fastapi uvicorn pytest
```

### 3. Configure API Keys

```bash
# Create .env file
echo "OPENAI_API_KEY=your-key-here" > .env

# Or export directly
export OPENAI_API_KEY="your-key-here"
```

### 4. Verify Setup

```python
# test_setup.py
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
print("✓ Setup successful!")
print(response.choices[0].message.content)
```

## Time Commitment

- **Total course**: 40-60 hours
- **Per module**: 4-6 hours
- **Capstone project**: 10-15 hours

**Recommended pace**: 2-3 modules per week

## Learning Path

### Beginner Track (Start Here)
1. Module 1: Foundations
2. Module 2: Building Your First Agent
3. Module 4: Agent Tools & Capabilities
4. Module 5: Production-Ready Agents

### Intermediate Track
5. Module 3: Advanced Agent Patterns
6. Module 6: Specialized Agent Types
7. Module 7: Advanced Topics

### Advanced Track
8. Module 8: Enterprise & Scale
9. Module 9: Cutting-Edge Research
10. Module 10: Capstone Project

## Getting Help

- **GitHub Issues**: Report errors or ask questions
- **Discussions**: Share projects and get feedback
- **Community**: Join Discord/Slack communities (see Resources)

## Ready to Start?

If you meet the prerequisites, you're ready to begin! Start with the [Introduction](./introduction.md) and then dive into [Module 1](./module1/1_1_what_are_agents.md).
