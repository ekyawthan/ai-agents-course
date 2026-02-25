# Agent Learning & Adaptation

## Introduction to Agent Learning

Learning and adaptation enable agents to improve over time, personalize to users, and handle new situations without explicit reprogramming.

### Why Learning Matters

**Benefits**:
- Improved performance over time
- Personalization to individual users
- Adaptation to changing environments
- Reduced need for manual updates
- Discovery of better strategies

**Challenges**:
- Avoiding catastrophic forgetting
- Balancing exploration vs exploitation
- Ensuring safe learning
- Managing computational costs
- Maintaining consistency

### Types of Learning

1. **Few-Shot Learning**: Learn from minimal examples
2. **Reinforcement Learning**: Learn from feedback
3. **Continuous Learning**: Ongoing improvement
4. **Transfer Learning**: Apply knowledge to new domains
5. **Meta-Learning**: Learn how to learn

## Few-Shot Learning

### In-Context Learning

```python
from typing import List, Dict
import openai

class FewShotLearner:
    """Learn from few examples in context"""
    
    def __init__(self):
        self.client = openai.OpenAI()
        self.examples = []
    
    def add_example(self, input_text: str, output_text: str, explanation: str = None):
        """Add training example"""
        example = {
            "input": input_text,
            "output": output_text,
            "explanation": explanation
        }
        self.examples.append(example)
        print(f"✅ Added example: {input_text[:50]}...")
    
    def learn_from_examples(self, examples: List[Dict]):
        """Batch add examples"""
        for ex in examples:
            self.add_example(ex["input"], ex["output"], ex.get("explanation"))
    
    def predict(self, input_text: str, temperature: float = 0.3) -> str:
        """Make prediction using learned examples"""
        
        # Build prompt with examples
        prompt = self.build_few_shot_prompt(input_text)
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        
        return response.choices[0].message.content
    
    def build_few_shot_prompt(self, input_text: str) -> str:
        """Build prompt with examples"""
        prompt = "Learn from these examples:\n\n"
        
        for i, example in enumerate(self.examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Input: {example['input']}\n"
            prompt += f"Output: {example['output']}\n"
            if example.get('explanation'):
                prompt += f"Why: {example['explanation']}\n"
            prompt += "\n"
        
        prompt += f"Now apply what you learned:\n"
        prompt += f"Input: {input_text}\n"
        prompt += f"Output:"
        
        return prompt
    
    def evaluate(self, test_cases: List[Dict]) -> Dict:
        """Evaluate performance on test cases"""
        correct = 0
        total = len(test_cases)
        
        for test in test_cases:
            prediction = self.predict(test["input"])
            expected = test["output"]
            
            # Simple exact match (can be more sophisticated)
            if prediction.strip().lower() == expected.strip().lower():
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }

# Usage
learner = FewShotLearner()

# Teach sentiment analysis
learner.add_example(
    "This product is amazing!",
    "positive",
    "Enthusiastic language indicates positive sentiment"
)
learner.add_example(
    "Terrible experience, very disappointed",
    "negative",
    "Words like 'terrible' and 'disappointed' indicate negative sentiment"
)
learner.add_example(
    "It's okay, nothing special",
    "neutral",
    "Lukewarm language indicates neutral sentiment"
)

# Test
result = learner.predict("I love this so much!")
print(f"Prediction: {result}")

# Evaluate
test_cases = [
    {"input": "Best purchase ever!", "output": "positive"},
    {"input": "Waste of money", "output": "negative"},
    {"input": "It works fine", "output": "neutral"}
]
evaluation = learner.evaluate(test_cases)
print(f"Accuracy: {evaluation['accuracy']:.1%}")
```

### Dynamic Example Selection

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class AdaptiveFewShotLearner(FewShotLearner):
    """Select most relevant examples dynamically"""
    
    def __init__(self, max_examples: int = 5):
        super().__init__()
        self.max_examples = max_examples
        self.example_embeddings = []
    
    def add_example(self, input_text: str, output_text: str, explanation: str = None):
        """Add example with embedding"""
        super().add_example(input_text, output_text, explanation)
        
        # Get embedding
        embedding = self.get_embedding(input_text)
        self.example_embeddings.append(embedding)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get text embedding"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def select_relevant_examples(self, input_text: str) -> List[Dict]:
        """Select most relevant examples for input"""
        if not self.examples:
            return []
        
        # Get input embedding
        input_embedding = self.get_embedding(input_text)
        
        # Calculate similarities
        similarities = []
        for i, example_embedding in enumerate(self.example_embeddings):
            similarity = cosine_similarity(
                input_embedding.reshape(1, -1),
                example_embedding.reshape(1, -1)
            )[0][0]
            similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Select top examples
        selected_indices = [idx for idx, _ in similarities[:self.max_examples]]
        selected_examples = [self.examples[i] for i in selected_indices]
        
        return selected_examples
    
    def predict(self, input_text: str, temperature: float = 0.3) -> str:
        """Predict using most relevant examples"""
        # Select relevant examples
        relevant_examples = self.select_relevant_examples(input_text)
        
        # Temporarily use only relevant examples
        original_examples = self.examples
        self.examples = relevant_examples
        
        # Make prediction
        result = super().predict(input_text, temperature)
        
        # Restore all examples
        self.examples = original_examples
        
        return result

# Usage
adaptive_learner = AdaptiveFewShotLearner(max_examples=3)

# Add many examples
examples = [
    ("Great product!", "positive"),
    ("Horrible quality", "negative"),
    ("Works as expected", "neutral"),
    ("Absolutely love it!", "positive"),
    ("Complete waste", "negative"),
    ("It's fine", "neutral"),
]

for inp, out in examples:
    adaptive_learner.add_example(inp, out)

# Predict - will use most relevant examples
result = adaptive_learner.predict("This is fantastic!")
print(f"Prediction: {result}")
```

## Reinforcement Learning from Feedback

### Human Feedback Collection

```python
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class Feedback:
    """User feedback on agent response"""
    response_id: str
    rating: int  # 1-5
    comment: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class FeedbackCollector:
    """Collect and manage user feedback"""
    
    def __init__(self):
        self.feedback_history = []
        self.response_cache = {}
    
    def record_response(self, response_id: str, prompt: str, response: str):
        """Record agent response"""
        self.response_cache[response_id] = {
            "prompt": prompt,
            "response": response,
            "timestamp": time.time()
        }
    
    def collect_feedback(self, response_id: str, rating: int, comment: str = None) -> Feedback:
        """Collect feedback on response"""
        feedback = Feedback(
            response_id=response_id,
            rating=rating,
            comment=comment
        )
        
        self.feedback_history.append(feedback)
        print(f"📝 Feedback recorded: {rating}/5")
        
        return feedback
    
    def get_average_rating(self) -> float:
        """Get average rating"""
        if not self.feedback_history:
            return 0.0
        
        total = sum(f.rating for f in self.feedback_history)
        return total / len(self.feedback_history)
    
    def get_positive_examples(self, threshold: int = 4) -> List[Dict]:
        """Get highly-rated examples"""
        positive = []
        
        for feedback in self.feedback_history:
            if feedback.rating >= threshold:
                response_data = self.response_cache.get(feedback.response_id)
                if response_data:
                    positive.append({
                        "prompt": response_data["prompt"],
                        "response": response_data["response"],
                        "rating": feedback.rating
                    })
        
        return positive
    
    def get_negative_examples(self, threshold: int = 2) -> List[Dict]:
        """Get poorly-rated examples"""
        negative = []
        
        for feedback in self.feedback_history:
            if feedback.rating <= threshold:
                response_data = self.response_cache.get(feedback.response_id)
                if response_data:
                    negative.append({
                        "prompt": response_data["prompt"],
                        "response": response_data["response"],
                        "rating": feedback.rating,
                        "comment": feedback.comment
                    })
        
        return negative

# Usage
collector = FeedbackCollector()

# Record response
response_id = "resp_001"
collector.record_response(
    response_id,
    "What is Python?",
    "Python is a programming language..."
)

# Collect feedback
collector.collect_feedback(response_id, 5, "Very helpful!")

# Get positive examples for learning
positive_examples = collector.get_positive_examples()
print(f"Positive examples: {len(positive_examples)}")
```

### Learning from Feedback

```python
class RLFHAgent:
    """Agent that learns from human feedback"""
    
    def __init__(self):
        self.client = openai.OpenAI()
        self.feedback_collector = FeedbackCollector()
        self.learner = AdaptiveFewShotLearner()
    
    def respond(self, prompt: str, response_id: str = None) -> str:
        """Generate response"""
        if response_id is None:
            response_id = f"resp_{int(time.time())}"
        
        # Use learned examples
        positive_examples = self.feedback_collector.get_positive_examples()
        
        # Build prompt with positive examples
        enhanced_prompt = self.build_prompt_with_examples(prompt, positive_examples)
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": enhanced_prompt}],
            temperature=0.7
        )
        
        response_text = response.choices[0].message.content
        
        # Record for feedback
        self.feedback_collector.record_response(response_id, prompt, response_text)
        
        return response_text
    
    def build_prompt_with_examples(self, prompt: str, examples: List[Dict]) -> str:
        """Build prompt incorporating learned examples"""
        if not examples:
            return prompt
        
        enhanced = "Here are examples of good responses:\n\n"
        
        for ex in examples[:5]:  # Use top 5
            enhanced += f"Q: {ex['prompt']}\n"
            enhanced += f"A: {ex['response']}\n\n"
        
        enhanced += f"Now respond to:\nQ: {prompt}\nA:"
        
        return enhanced
    
    def learn_from_feedback(self, response_id: str, rating: int, comment: str = None):
        """Learn from user feedback"""
        feedback = self.feedback_collector.collect_feedback(response_id, rating, comment)
        
        # If positive, add to examples
        if rating >= 4:
            response_data = self.feedback_collector.response_cache.get(response_id)
            if response_data:
                self.learner.add_example(
                    response_data["prompt"],
                    response_data["response"],
                    f"User rated {rating}/5"
                )
                print("✅ Learned from positive feedback")
        
        # If negative, analyze and improve
        elif rating <= 2:
            self.analyze_negative_feedback(response_id, comment)
    
    def analyze_negative_feedback(self, response_id: str, comment: str):
        """Analyze negative feedback to improve"""
        response_data = self.feedback_collector.response_cache.get(response_id)
        if not response_data:
            return
        
        prompt = f"""Analyze this negative feedback:

Original prompt: {response_data['prompt']}
Response: {response_data['response']}
User feedback: {comment}

What went wrong and how to improve?"""
        
        analysis = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        print(f"📊 Analysis: {analysis.choices[0].message.content[:200]}...")
    
    def get_performance_metrics(self) -> Dict:
        """Get learning performance metrics"""
        avg_rating = self.feedback_collector.get_average_rating()
        total_feedback = len(self.feedback_collector.feedback_history)
        positive_count = len(self.feedback_collector.get_positive_examples())
        
        return {
            "average_rating": avg_rating,
            "total_feedback": total_feedback,
            "positive_examples": positive_count,
            "learned_examples": len(self.learner.examples)
        }

# Usage
agent = RLFHAgent()

# Interact and learn
response_id = "resp_001"
response = agent.respond("Explain machine learning", response_id)
print(f"Response: {response}")

# User provides feedback
agent.learn_from_feedback(response_id, 5, "Clear and concise!")

# Check improvement
metrics = agent.get_performance_metrics()
print(f"Metrics: {metrics}")
```

## Continuous Learning

### Online Learning System

```python
class ContinuousLearner:
    """Agent that continuously learns from interactions"""
    
    def __init__(self, memory_size: int = 1000):
        self.client = openai.OpenAI()
        self.memory_size = memory_size
        self.interaction_history = []
        self.performance_history = []
    
    def interact(self, prompt: str) -> Dict:
        """Interact and learn"""
        # Generate response
        response = self.generate_response(prompt)
        
        # Record interaction
        interaction = {
            "prompt": prompt,
            "response": response,
            "timestamp": time.time()
        }
        self.interaction_history.append(interaction)
        
        # Trim history if too large
        if len(self.interaction_history) > self.memory_size:
            self.interaction_history = self.interaction_history[-self.memory_size:]
        
        return {
            "response": response,
            "interaction_id": len(self.interaction_history) - 1
        }
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using learned knowledge"""
        # Get relevant past interactions
        relevant = self.get_relevant_interactions(prompt)
        
        # Build context
        context = self.build_context(relevant)
        
        # Generate
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def get_relevant_interactions(self, prompt: str, top_k: int = 5) -> List[Dict]:
        """Get relevant past interactions"""
        if not self.interaction_history:
            return []
        
        # Simple keyword matching (can use embeddings for better results)
        prompt_words = set(prompt.lower().split())
        
        scored = []
        for interaction in self.interaction_history:
            interaction_words = set(interaction["prompt"].lower().split())
            overlap = len(prompt_words & interaction_words)
            scored.append((interaction, overlap))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [interaction for interaction, _ in scored[:top_k]]
    
    def build_context(self, relevant_interactions: List[Dict]) -> str:
        """Build context from relevant interactions"""
        if not relevant_interactions:
            return "You are a helpful assistant."
        
        context = "You are a helpful assistant. Here are relevant past interactions:\n\n"
        
        for interaction in relevant_interactions:
            context += f"Q: {interaction['prompt']}\n"
            context += f"A: {interaction['response']}\n\n"
        
        context += "Use this knowledge to inform your response."
        
        return context
    
    def update_from_feedback(self, interaction_id: int, feedback: Dict):
        """Update based on feedback"""
        if interaction_id >= len(self.interaction_history):
            return
        
        interaction = self.interaction_history[interaction_id]
        interaction["feedback"] = feedback
        
        # Track performance
        self.performance_history.append({
            "timestamp": time.time(),
            "rating": feedback.get("rating", 0)
        })
    
    def get_learning_curve(self) -> List[float]:
        """Get performance over time"""
        if not self.performance_history:
            return []
        
        # Calculate moving average
        window = 10
        curve = []
        
        for i in range(len(self.performance_history)):
            start = max(0, i - window + 1)
            window_ratings = [
                p["rating"] for p in self.performance_history[start:i+1]
            ]
            avg = sum(window_ratings) / len(window_ratings)
            curve.append(avg)
        
        return curve

# Usage
learner = ContinuousLearner()

# Continuous interaction
for i in range(10):
    result = learner.interact(f"Question {i}: What is AI?")
    print(f"Response {i}: {result['response'][:50]}...")
    
    # Simulate feedback
    learner.update_from_feedback(result["interaction_id"], {"rating": 4})

# Check learning curve
curve = learner.get_learning_curve()
print(f"Learning curve: {curve}")
```

## Fine-Tuning for Specific Tasks

### Preparing Training Data

```python
class FineTuningDataPrep:
    """Prepare data for fine-tuning"""
    
    def __init__(self):
        self.training_data = []
    
    def add_training_example(self, 
                            system_message: str,
                            user_message: str,
                            assistant_message: str):
        """Add training example"""
        example = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message}
            ]
        }
        self.training_data.append(example)
    
    def load_from_feedback(self, feedback_collector: FeedbackCollector, min_rating: int = 4):
        """Load training data from positive feedback"""
        positive_examples = feedback_collector.get_positive_examples(threshold=min_rating)
        
        for example in positive_examples:
            self.add_training_example(
                "You are a helpful assistant.",
                example["prompt"],
                example["response"]
            )
        
        print(f"Loaded {len(positive_examples)} training examples")
    
    def export_jsonl(self, filename: str):
        """Export to JSONL format for fine-tuning"""
        import json
        
        with open(filename, 'w') as f:
            for example in self.training_data:
                f.write(json.dumps(example) + '\n')
        
        print(f"Exported {len(self.training_data)} examples to {filename}")
    
    def validate_data(self) -> Dict:
        """Validate training data quality"""
        if not self.training_data:
            return {"valid": False, "error": "No training data"}
        
        issues = []
        
        for i, example in enumerate(self.training_data):
            # Check structure
            if "messages" not in example:
                issues.append(f"Example {i}: Missing 'messages' field")
                continue
            
            messages = example["messages"]
            
            # Check message count
            if len(messages) < 2:
                issues.append(f"Example {i}: Too few messages")
            
            # Check roles
            roles = [m["role"] for m in messages]
            if "user" not in roles or "assistant" not in roles:
                issues.append(f"Example {i}: Missing required roles")
        
        return {
            "valid": len(issues) == 0,
            "total_examples": len(self.training_data),
            "issues": issues
        }

# Usage
prep = FineTuningDataPrep()

# Add examples
prep.add_training_example(
    "You are a Python expert.",
    "How do I sort a list?",
    "Use the sorted() function or list.sort() method..."
)

# Validate
validation = prep.validate_data()
print(f"Valid: {validation['valid']}")

# Export
prep.export_jsonl("training_data.jsonl")
```

## Transfer Learning

### Domain Adaptation

```python
class DomainAdapter:
    """Adapt agent to new domain"""
    
    def __init__(self, base_agent):
        self.base_agent = base_agent
        self.domain_examples = []
        self.client = openai.OpenAI()
    
    def add_domain_knowledge(self, domain: str, examples: List[Dict]):
        """Add domain-specific examples"""
        self.domain_examples.extend(examples)
        print(f"Added {len(examples)} examples for domain: {domain}")
    
    def adapt_response(self, prompt: str, domain: str) -> str:
        """Generate domain-adapted response"""
        # Get domain examples
        domain_context = self.build_domain_context(domain)
        
        # Generate with domain context
        messages = [
            {"role": "system", "content": domain_context},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.5
        )
        
        return response.choices[0].message.content
    
    def build_domain_context(self, domain: str) -> str:
        """Build context for specific domain"""
        context = f"You are an expert in {domain}.\n\n"
        context += "Domain-specific examples:\n\n"
        
        # Filter examples for this domain
        relevant = [ex for ex in self.domain_examples if ex.get("domain") == domain]
        
        for ex in relevant[:5]:
            context += f"Q: {ex['input']}\n"
            context += f"A: {ex['output']}\n\n"
        
        return context

# Usage
adapter = DomainAdapter(base_agent=None)

# Add medical domain knowledge
medical_examples = [
    {
        "domain": "medical",
        "input": "What is hypertension?",
        "output": "Hypertension is high blood pressure..."
    }
]

adapter.add_domain_knowledge("medical", medical_examples)

# Adapt to medical domain
response = adapter.adapt_response(
    "Explain diabetes",
    domain="medical"
)
print(response)
```

## Meta-Learning

### Learning to Learn

```python
class MetaLearner:
    """Learn how to learn new tasks quickly"""
    
    def __init__(self):
        self.client = openai.OpenAI()
        self.task_history = []
        self.learning_strategies = []
    
    def learn_new_task(self, task_description: str, examples: List[Dict]) -> Dict:
        """Learn a new task"""
        print(f"📚 Learning new task: {task_description}")
        
        # Analyze task
        task_analysis = self.analyze_task(task_description, examples)
        
        # Select learning strategy
        strategy = self.select_strategy(task_analysis)
        
        # Apply strategy
        learned_model = self.apply_strategy(strategy, examples)
        
        # Record
        self.task_history.append({
            "description": task_description,
            "analysis": task_analysis,
            "strategy": strategy,
            "examples_count": len(examples)
        })
        
        return {
            "task": task_description,
            "strategy": strategy,
            "model": learned_model
        }
    
    def analyze_task(self, description: str, examples: List[Dict]) -> Dict:
        """Analyze task characteristics"""
        prompt = f"""Analyze this learning task:

Task: {description}

Examples: {len(examples)}
Sample: {examples[0] if examples else 'None'}

Determine:
1. Task type (classification, generation, etc.)
2. Complexity (simple, medium, complex)
3. Required examples (few, many)
4. Best learning approach

Analysis:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        # Parse analysis (simplified)
        return {
            "type": "classification",
            "complexity": "medium",
            "analysis": response.choices[0].message.content
        }
    
    def select_strategy(self, task_analysis: Dict) -> str:
        """Select learning strategy based on task"""
        complexity = task_analysis.get("complexity", "medium")
        
        if complexity == "simple":
            return "few-shot"
        elif complexity == "medium":
            return "adaptive-few-shot"
        else:
            return "fine-tuning"
    
    def apply_strategy(self, strategy: str, examples: List[Dict]) -> Any:
        """Apply selected learning strategy"""
        if strategy == "few-shot":
            learner = FewShotLearner()
            for ex in examples:
                learner.add_example(ex["input"], ex["output"])
            return learner
        
        elif strategy == "adaptive-few-shot":
            learner = AdaptiveFewShotLearner()
            for ex in examples:
                learner.add_example(ex["input"], ex["output"])
            return learner
        
        else:
            # Would implement fine-tuning
            return None
    
    def get_learning_insights(self) -> Dict:
        """Get insights from learning history"""
        if not self.task_history:
            return {}
        
        strategies_used = {}
        for task in self.task_history:
            strategy = task["strategy"]
            strategies_used[strategy] = strategies_used.get(strategy, 0) + 1
        
        return {
            "total_tasks_learned": len(self.task_history),
            "strategies_used": strategies_used,
            "avg_examples_per_task": sum(t["examples_count"] for t in self.task_history) / len(self.task_history)
        }

# Usage
meta_learner = MetaLearner()

# Learn multiple tasks
tasks = [
    {
        "description": "Sentiment analysis",
        "examples": [
            {"input": "Great!", "output": "positive"},
            {"input": "Terrible", "output": "negative"}
        ]
    },
    {
        "description": "Language detection",
        "examples": [
            {"input": "Hello", "output": "English"},
            {"input": "Bonjour", "output": "French"}
        ]
    }
]

for task in tasks:
    result = meta_learner.learn_new_task(task["description"], task["examples"])
    print(f"Learned using: {result['strategy']}")

# Get insights
insights = meta_learner.get_learning_insights()
print(f"Insights: {insights}")
```

## Best Practices

1. **Start simple**: Begin with few-shot learning
2. **Collect feedback**: Continuously gather user input
3. **Monitor performance**: Track learning metrics
4. **Avoid overfitting**: Don't memorize, generalize
5. **Safe learning**: Validate before deploying
6. **Incremental updates**: Small, frequent improvements
7. **A/B testing**: Compare learned vs baseline
8. **Human oversight**: Review learned behaviors
9. **Version control**: Track model versions
10. **Rollback capability**: Revert if performance degrades

## Next Steps

You now understand agent learning and adaptation in depth! Next, we'll explore multimodal agents that work with images, audio, and other modalities.
