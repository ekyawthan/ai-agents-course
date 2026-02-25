# Glossary

## A

**Agent** - An autonomous system that perceives its environment and takes actions to achieve goals.

**Agentic Framework** - A software framework designed for building AI agents (e.g., LangChain, AutoGPT).

**API (Application Programming Interface)** - Interface for software components to communicate.

**AST (Abstract Syntax Tree)** - Tree representation of code structure.

## B

**Backoff** - Strategy for retrying failed operations with increasing delays.

**Benchmark** - Standardized test for measuring performance.

**Beam Search** - Search algorithm that explores multiple paths simultaneously.

## C

**Chain-of-Thought (CoT)** - Prompting technique that encourages step-by-step reasoning.

**Checkpoint** - Saved state of a model or agent for recovery.

**Context Window** - Maximum amount of text an LLM can process at once.

**Constitutional AI** - Approach to align AI behavior with principles.

## D

**Deterministic** - Producing the same output given the same input.

**Distributed Tracing** - Tracking requests across multiple services.

**Docker** - Platform for containerizing applications.

## E

**Embedding** - Vector representation of text or data.

**Episodic Memory** - Memory of specific past events or experiences.

**Evaluation Metric** - Quantitative measure of performance.

## F

**Few-Shot Learning** - Learning from a small number of examples.

**Fine-Tuning** - Training a pre-trained model on specific data.

**Function Calling** - LLM capability to invoke external functions.

## G

**Generalization** - Ability to perform well on unseen data.

**Guardrails** - Safety mechanisms to prevent harmful behavior.

**GPU (Graphics Processing Unit)** - Hardware for parallel computation.

## H

**Hallucination** - When LLMs generate false or nonsensical information.

**Human-in-the-Loop (HITL)** - System requiring human approval for decisions.

**Hyperparameter** - Configuration parameter for model training.

## I

**Inference** - Using a trained model to make predictions.

**Interpretability** - Ability to understand model decisions.

## K

**Kubernetes (K8s)** - Container orchestration platform.

## L

**Latency** - Time delay between request and response.

**LLM (Large Language Model)** - Neural network trained on vast text data.

**Long-Horizon Planning** - Planning over extended time periods.

## M

**Memory System** - Component for storing and retrieving information.

**Meta-Learning** - Learning how to learn.

**Microservices** - Architecture pattern with independent services.

**Multimodal** - Processing multiple types of data (text, images, audio).

## N

**Neural Network** - Computing system inspired by biological brains.

**NLP (Natural Language Processing)** - Processing and understanding human language.

## O

**Observability** - Ability to understand system internal state from outputs.

**Orchestration** - Coordinating multiple components or agents.

## P

**Perception-Reasoning-Action Loop** - Core agent cycle: observe, think, act.

**Prompt Engineering** - Crafting effective prompts for LLMs.

**Production** - Live environment serving real users.

## R

**RAG (Retrieval-Augmented Generation)** - Combining retrieval with generation.

**ReAct** - Pattern combining reasoning and acting.

**Reinforcement Learning (RL)** - Learning through rewards and penalties.

**RLHF (Reinforcement Learning from Human Feedback)** - Training with human preferences.

## S

**Sandbox** - Isolated environment for safe code execution.

**Semantic Memory** - Memory of facts and knowledge.

**Semantic Search** - Search based on meaning, not keywords.

**Self-Improvement** - Agent's ability to improve its own capabilities.

**Streaming** - Sending responses incrementally as generated.

## T

**Temperature** - Parameter controlling randomness in LLM outputs (0=deterministic, 1=creative).

**Token** - Unit of text processed by LLMs (roughly 0.75 words).

**Tool** - External function or API an agent can use.

**Tree of Thoughts** - Exploring multiple reasoning paths.

## V

**Vector Database** - Database optimized for similarity search on embeddings.

**Validation** - Checking if outputs meet requirements.

## W

**Working Memory** - Short-term memory for current task.

## Z

**Zero-Shot** - Performing tasks without specific training examples.

---

## Common Acronyms

- **AI** - Artificial Intelligence
- **API** - Application Programming Interface
- **AST** - Abstract Syntax Tree
- **CI/CD** - Continuous Integration/Continuous Deployment
- **CoT** - Chain-of-Thought
- **GPU** - Graphics Processing Unit
- **HITL** - Human-in-the-Loop
- **LLM** - Large Language Model
- **ML** - Machine Learning
- **NLP** - Natural Language Processing
- **RAG** - Retrieval-Augmented Generation
- **RL** - Reinforcement Learning
- **RLHF** - Reinforcement Learning from Human Feedback
- **SLA** - Service Level Agreement
- **ToT** - Tree of Thoughts
- **UI/UX** - User Interface/User Experience

## Model Parameters

**Temperature** - Controls randomness (0.0-2.0)
- 0.0-0.3: Focused, deterministic
- 0.4-0.7: Balanced
- 0.8-1.0: Creative
- 1.0+: Very random

**Top-p (Nucleus Sampling)** - Alternative to temperature (0.0-1.0)
- 0.1: Very focused
- 0.5: Balanced
- 0.9: Diverse

**Max Tokens** - Maximum length of response

**Frequency Penalty** - Reduces repetition (-2.0 to 2.0)

**Presence Penalty** - Encourages new topics (-2.0 to 2.0)

## HTTP Status Codes

- **200** - Success
- **400** - Bad Request
- **401** - Unauthorized
- **429** - Rate Limited
- **500** - Server Error
- **503** - Service Unavailable
