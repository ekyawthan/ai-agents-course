# Frequently Asked Questions

## Getting Started

### Which LLM should I use?

**For learning**: Start with OpenAI's GPT-3.5-turbo
- Affordable ($0.50-2 per million tokens)
- Fast responses
- Good function calling support

**For production**: Consider:
- **GPT-4**: Best reasoning, higher cost
- **Claude 3**: Long context (200K tokens), excellent for complex tasks
- **AWS Bedrock**: Enterprise features, multiple models
- **Open source** (Llama, Mistral): Self-hosted, no API costs

### How much does it cost to run agents?

**Development** (100 requests/day):
- GPT-3.5: ~$5-10/month
- GPT-4: ~$30-50/month

**Production** (10K requests/day):
- GPT-3.5: ~$500-1000/month
- GPT-4: ~$3000-5000/month

**Cost optimization**:
- Use caching (50-70% reduction)
- Smaller models for simple tasks
- Batch requests when possible

### Do I need a GPU?

**No** for most agent development:
- API-based LLMs run in the cloud
- Your code just makes HTTP requests

**Yes** if you want to:
- Run local models (Llama, Mistral)
- Fine-tune models
- Process large batches offline

### Can I use this commercially?

**Yes**, but check:
- LLM provider terms (OpenAI, Anthropic allow commercial use)
- Open source licenses for frameworks
- Data privacy regulations (GDPR, etc.)
- Your specific use case compliance needs

## Technical Questions

### How do I handle rate limits?

```python
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
def call_llm(prompt):
    return client.chat.completions.create(...)
```

### How do I reduce latency?

1. **Streaming**: Stream responses as they generate
2. **Caching**: Cache repeated queries
3. **Smaller models**: Use GPT-3.5 for simple tasks
4. **Parallel calls**: Run independent calls concurrently
5. **Prompt optimization**: Shorter prompts = faster responses

### How do I prevent hallucinations?

1. **Require tool use**: Force agents to use tools, not memory
2. **Validation**: Verify outputs before using them
3. **Lower temperature**: Use 0.2-0.3 for factual tasks
4. **Structured outputs**: Use JSON mode or function calling
5. **Retrieval**: Use RAG to ground responses in facts

### How do I debug agent failures?

1. **Log everything**: All thoughts, actions, observations
2. **Trace execution**: Use tools like LangSmith
3. **Test incrementally**: Start simple, add complexity
4. **Validate tools**: Test tools independently
5. **Check prompts**: Ensure clear instructions

## Architecture Questions

### Single agent vs multi-agent?

**Single agent** when:
- Task is focused and well-defined
- Simplicity is important
- Low latency is critical

**Multi-agent** when:
- Task requires diverse expertise
- Parallel processing helps
- Checks and balances needed
- Scaling beyond single agent

### How do I handle long-running tasks?

1. **Async processing**: Use background jobs
2. **Checkpointing**: Save state periodically
3. **Progress updates**: Stream status to user
4. **Timeouts**: Set reasonable limits
5. **Resumability**: Allow restart from checkpoint

### How do I scale to production?

1. **Horizontal scaling**: Multiple agent instances
2. **Load balancing**: Distribute requests
3. **Caching**: Redis for responses
4. **Queue systems**: RabbitMQ, SQS for async tasks
5. **Monitoring**: Track performance and errors

## Safety & Security

### How do I make agents safe?

1. **Sandboxing**: Isolate code execution (Docker)
2. **Validation**: Check all inputs and outputs
3. **Rate limiting**: Prevent abuse
4. **Human approval**: For critical actions
5. **Audit logging**: Track all actions
6. **Guardrails**: Block harmful requests

### What about prompt injection?

**Defense strategies**:
1. **Input sanitization**: Remove suspicious patterns
2. **Separate contexts**: User input vs system instructions
3. **Output validation**: Check for unexpected behavior
4. **Monitoring**: Detect anomalies
5. **Least privilege**: Limit tool access

### How do I handle sensitive data?

1. **Encryption**: Encrypt data at rest and in transit
2. **Access control**: Role-based permissions
3. **Data minimization**: Only collect what's needed
4. **Anonymization**: Remove PII when possible
5. **Compliance**: Follow GDPR, HIPAA, etc.

## Development Questions

### Which framework should I use?

**LangChain**: Best for rapid prototyping
- Lots of integrations
- Active community
- Good documentation

**LangGraph**: Best for complex workflows
- Graph-based state management
- Better control flow
- Production-ready

**Custom**: Best for specific needs
- Full control
- No framework overhead
- Optimized for your use case

### How do I test agents?

1. **Unit tests**: Test individual components
2. **Integration tests**: Test agent workflows
3. **Evaluation sets**: Benchmark on standard tasks
4. **A/B testing**: Compare agent versions
5. **User testing**: Real-world feedback

### How long does it take to build an agent?

**Simple agent** (ReAct with 3-5 tools): 1-2 days
**Production agent** (with testing, monitoring): 1-2 weeks
**Complex multi-agent system**: 1-3 months
**Enterprise deployment**: 3-6 months

## Common Issues

### "My agent gets stuck in loops"

**Solutions**:
- Set max_steps limit
- Add loop detection
- Improve prompts to avoid repetition
- Use planning instead of pure ReAct

### "Tool calls fail frequently"

**Solutions**:
- Validate tool schemas
- Add retry logic with exponential backoff
- Improve tool descriptions
- Test tools independently
- Add error handling

### "Agent is too slow"

**Solutions**:
- Use faster models (GPT-3.5 vs GPT-4)
- Enable streaming
- Cache repeated queries
- Optimize prompts (shorter = faster)
- Run tools in parallel

### "Costs are too high"

**Solutions**:
- Cache aggressively
- Use smaller models when possible
- Optimize prompt length
- Batch requests
- Set usage limits

## Learning Path

### I'm a beginner programmer. Can I take this course?

You need:
- Python basics (functions, classes)
- API concepts
- Command line comfort

If you're missing these, spend 2-4 weeks on Python fundamentals first, then return to this course.

### Should I take this course or learn LangChain first?

**Take this course** if you want to:
- Understand agent fundamentals
- Build from scratch
- Know what's happening under the hood

**Learn LangChain first** if you want to:
- Build quickly with existing tools
- Focus on applications, not internals

Ideally: Take this course, then use frameworks with deeper understanding.

### How do I stay current with agent research?

1. **Follow researchers**: Twitter/X, blogs
2. **Read papers**: ArXiv, conferences
3. **Join communities**: Discord, Reddit
4. **Experiment**: Try new techniques
5. **Contribute**: Open source projects

## Still Have Questions?

- **GitHub Discussions**: [Ask the community](https://github.com/ekyawthan/ai-agents-course/discussions)
- **Issues**: [Report problems](https://github.com/ekyawthan/ai-agents-course/issues)
- **Contributing**: [Improve the course](./contributing.md)
