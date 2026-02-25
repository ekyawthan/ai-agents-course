# LLM Fundamentals for Agents

## How Language Models Work

Large Language Models (LLMs) are the "brain" of modern AI agents. Understanding how they work helps you build better agents.

### The Basics

LLMs are trained to predict the next token (word or word piece) given previous tokens:

```
Input:  "The capital of France is"
Output: "Paris" (most likely next token)
```

This simple mechanism enables:
- Text generation
- Question answering
- Reasoning
- Code generation
- Tool use

### From Prediction to Reasoning

Modern LLMs don't just predict—they reason:

**Chain-of-Thought**: Breaking down problems step by step
```
Question: "If I have 3 apples and buy 2 more, then give away 1, how many do I have?"

LLM reasoning:
1. Start with 3 apples
2. Buy 2 more: 3 + 2 = 5
3. Give away 1: 5 - 1 = 4
Answer: 4 apples
```

**Tool Use**: Recognizing when to call external functions
```
User: "What's the weather in Tokyo?"
LLM: I should use the weather_api tool with location="Tokyo"
```

### Key Capabilities for Agents

- **Instruction following**: Understanding and executing commands
- **Context understanding**: Maintaining awareness of conversation history
- **Function calling**: Invoking tools with correct parameters
- **Error recovery**: Adapting when things go wrong
- **Self-reflection**: Evaluating own outputs

## Prompting Strategies for Agents

How you prompt an LLM dramatically affects agent performance.

### System Prompts

Define the agent's role, capabilities, and constraints:

```
You are a research assistant agent. Your goal is to help users 
find and synthesize information from multiple sources.

Available tools:
- web_search(query): Search the internet
- read_url(url): Extract content from a webpage
- summarize(text): Create concise summaries

Always:
1. Break complex requests into steps
2. Verify information from multiple sources
3. Cite your sources
4. Ask for clarification if needed
```

### Few-Shot Examples

Show the agent how to behave through examples:

```
Example 1:
User: "Find recent news about AI"
Agent: I'll search for recent AI news.
Action: web_search("AI news 2026")
Result: [search results]
Agent: Here are the top 3 recent AI developments...

Example 2:
User: "What's on that page?"
Agent: I need a URL to read a page. Could you provide the link?
```

### ReAct Pattern

The most common prompting pattern for agents:

```
Thought: What do I need to do?
Action: [tool_name](parameters)
Observation: [result from tool]
Thought: What does this mean?
Action: [next tool or final answer]
```

### Structured Outputs

Guide the LLM to produce consistent formats:

```
Respond in this format:
{
  "reasoning": "Your thought process",
  "action": "tool_name",
  "parameters": {"param": "value"},
  "confidence": 0.95
}
```

## Context Windows and Token Limits

Every LLM has a maximum context window—the amount of text it can process at once.

### Common Context Sizes

- **GPT-4**: 8K, 32K, 128K tokens
- **Claude**: 200K tokens
- **Gemini**: 1M+ tokens

### What Fits in Context?

Approximate token counts:
- 1 token ≈ 4 characters
- 1 token ≈ 0.75 words
- 100 tokens ≈ 75 words
- 1,000 tokens ≈ 750 words

### Context Management Strategies

**1. Summarization**
Compress old conversation history:
```
[Full conversation history]
    ↓
[Summary of key points] + [Recent messages]
```

**2. Sliding Window**
Keep only the most recent N messages:
```
Message 1, 2, 3, 4, 5, 6, 7, 8
                    └─────────┘ (keep last 4)
```

**3. Selective Retention**
Keep important messages, discard routine ones:
```
System prompt + Key decisions + Recent context
```

**4. External Memory**
Store information outside context, retrieve as needed:
```
Context: [Current task]
Memory DB: [All past information]
           ↓ (retrieve relevant)
Context: [Current task] + [Relevant memories]
```

### Token Budget Management

For agents, allocate tokens wisely:
```
System prompt:     500 tokens
Tools definition:  1,000 tokens
Conversation:      5,000 tokens
Working memory:    1,500 tokens
Reserve:           1,000 tokens (for response)
─────────────────────────────
Total:             9,000 tokens (fits in 8K with buffer)
```

## Temperature, Top-p, and Sampling Parameters

These parameters control how the LLM generates text.

### Temperature

Controls randomness (0.0 to 2.0):

**Low temperature (0.0 - 0.3)**: Deterministic, focused
```
Temperature: 0.1
"The capital of France is Paris" (always)
```
**Use for**: Tool calling, structured tasks, factual responses

**Medium temperature (0.5 - 0.8)**: Balanced
```
Temperature: 0.7
"The capital of France is Paris, a beautiful city known for..."
```
**Use for**: General agent behavior, conversational responses

**High temperature (1.0 - 2.0)**: Creative, random
```
Temperature: 1.5
"The capital of France? Ah, the magnificent Paris, where..."
```
**Use for**: Creative tasks, brainstorming, diverse outputs

### Top-p (Nucleus Sampling)

Controls diversity by probability mass (0.0 to 1.0):

**Low top-p (0.1 - 0.5)**: Conservative choices
- Considers only the most likely tokens
- More focused and consistent

**High top-p (0.9 - 1.0)**: Diverse choices
- Considers a wider range of tokens
- More varied and creative

**Typical for agents**: 0.9-0.95

### Top-k

Limits to top K most likely tokens:
- **top-k=1**: Always pick most likely (deterministic)
- **top-k=10**: Choose from 10 most likely
- **top-k=50**: More diversity

### Practical Guidelines for Agents

**For tool calling and structured tasks:**
```python
temperature = 0.1
top_p = 0.9
```

**For conversational responses:**
```python
temperature = 0.7
top_p = 0.95
```

**For creative tasks:**
```python
temperature = 1.0
top_p = 0.95
```

## Other Important Parameters

### Max Tokens
Maximum length of generated response:
- Set based on expected output length
- Leave room for tool calls and reasoning
- Typical: 500-2000 for agent responses

### Stop Sequences
Tokens that halt generation:
```python
stop_sequences = ["</tool>", "DONE", "\n\nUser:"]
```
Useful for controlling agent output format.

### Frequency/Presence Penalty
Reduce repetition:
- **Frequency penalty**: Penalize tokens based on how often they appear
- **Presence penalty**: Penalize tokens that have appeared at all
- Typical: 0.0-0.5 for agents

## Prompt Engineering Best Practices

### 1. Be Specific
❌ "Help me with this"
✅ "Search for recent papers on transformer architectures and summarize the key innovations"

### 2. Provide Context
```
You are helping a software engineer debug a Python application.
The user has intermediate Python knowledge.
Focus on practical solutions.
```

### 3. Use Delimiters
```
User input: """
{user_message}
"""

Available tools: ###
{tool_definitions}
###
```

### 4. Specify Output Format
```
Respond with:
1. Your reasoning
2. The action to take
3. Expected outcome
```

### 5. Handle Edge Cases
```
If the user's request is unclear, ask for clarification.
If a tool fails, try an alternative approach.
If you cannot complete the task, explain why.
```

## Testing and Iteration

### Evaluate Prompts Systematically

1. **Create test cases**: Common scenarios your agent should handle
2. **Run experiments**: Try different prompts and parameters
3. **Measure performance**: Success rate, quality, efficiency
4. **Iterate**: Refine based on results

### Common Issues and Fixes

**Issue**: Agent doesn't use tools
**Fix**: Add explicit examples of tool usage

**Issue**: Agent is too verbose
**Fix**: Lower temperature, add "be concise" instruction

**Issue**: Agent hallucinates
**Fix**: Emphasize "only use provided tools", add verification steps

**Issue**: Agent gets stuck in loops
**Fix**: Add step counter, max iterations limit

## Choosing the Right Model

Different models for different needs:

### For Agents

**GPT-4 / GPT-4 Turbo**
- Excellent reasoning
- Reliable tool calling
- Good for complex tasks

**Claude 3 (Opus/Sonnet)**
- Long context (200K)
- Strong reasoning
- Good safety features

**GPT-3.5 Turbo**
- Fast and cheap
- Good for simple agents
- Lower reasoning capability

### Trade-offs

- **Cost vs. Capability**: Stronger models cost more
- **Speed vs. Quality**: Faster models may be less accurate
- **Context vs. Price**: Longer context costs more

## Next Steps

With these LLM fundamentals, you're ready to build your first agent! In Chapter 2, we'll implement a simple ReAct agent that puts these concepts into practice.
