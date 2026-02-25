# Memory Systems

## Why Agents Need Memory

Without memory, agents are like people with amnesia—they can't learn from experience, maintain context, or build on previous interactions.

**Without Memory**:
```
User: "My name is Alice"
Agent: "Nice to meet you!"
[Later]
User: "What's my name?"
Agent: "I don't know your name."
```

**With Memory**:
```
User: "My name is Alice"
Agent: "Nice to meet you, Alice!" [stores: user_name = "Alice"]
[Later]
User: "What's my name?"
Agent: "Your name is Alice." [retrieves: user_name]
```

## Types of Memory

### Short-Term Memory (Working Memory)

Temporary storage for the current task.

**Characteristics**:
- Limited capacity (context window)
- Cleared after task completion
- Fast access
- Stored in conversation history

**What to store**:
- Current conversation
- Intermediate results
- Active plan
- Tool outputs

### Long-Term Memory (Persistent Memory)

Permanent storage across sessions.

**Characteristics**:
- Unlimited capacity (database)
- Persists across sessions
- Slower access (requires retrieval)
- Stored in external systems

**What to store**:
- User preferences
- Past conversations
- Learned facts
- Successful strategies

## Conversation History Management

Managing the conversation context efficiently.

### Basic History Tracking

```python
class ConversationMemory:
    """Simple conversation history"""
    
    def __init__(self, max_messages=20):
        self.messages = []
        self.max_messages = max_messages
    
    def add_message(self, role: str, content: str):
        """Add message to history"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        # Trim if too long
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_messages(self) -> List[dict]:
        """Get conversation history"""
        return self.messages
    
    def clear(self):
        """Clear history"""
        self.messages = []
```

### Sliding Window

Keep only recent messages:

```python
class SlidingWindowMemory:
    """Keep last N messages"""
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.messages = []
    
    def add(self, message: dict):
        """Add message and maintain window"""
        self.messages.append(message)
        
        # Keep only last N messages
        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size:]
    
    def get_context(self) -> List[dict]:
        """Get current window"""
        return self.messages
```

### Token-Based Truncation

Manage by token count instead of message count:

```python
import tiktoken

class TokenAwareMemory:
    """Manage memory by token budget"""
    
    def __init__(self, max_tokens=4000, model="gpt-4"):
        self.max_tokens = max_tokens
        self.messages = []
        self.encoding = tiktoken.encoding_for_model(model)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def get_total_tokens(self) -> int:
        """Count total tokens in history"""
        total = 0
        for msg in self.messages:
            total += self.count_tokens(msg["content"])
        return total
    
    def add(self, message: dict):
        """Add message and trim if needed"""
        self.messages.append(message)
        
        # Trim oldest messages if over budget
        while self.get_total_tokens() > self.max_tokens and len(self.messages) > 1:
            self.messages.pop(0)  # Remove oldest
    
    def get_context(self) -> List[dict]:
        """Get messages within token budget"""
        return self.messages
```

### Summarization Strategy

Compress old messages:

```python
class SummarizingMemory:
    """Summarize old conversations"""
    
    def __init__(self, summary_threshold=20):
        self.messages = []
        self.summary = None
        self.summary_threshold = summary_threshold
    
    def add(self, message: dict):
        """Add message and summarize if needed"""
        self.messages.append(message)
        
        if len(self.messages) > self.summary_threshold:
            self.summarize_old_messages()
    
    def summarize_old_messages(self):
        """Summarize and compress old messages"""
        # Take first half of messages
        to_summarize = self.messages[:len(self.messages)//2]
        
        # Create summary
        summary_text = self.create_summary(to_summarize)
        
        # Update summary
        if self.summary:
            self.summary += f"\n\n{summary_text}"
        else:
            self.summary = summary_text
        
        # Keep only recent messages
        self.messages = self.messages[len(self.messages)//2:]
    
    def create_summary(self, messages: List[dict]) -> str:
        """Generate summary of messages"""
        conversation = "\n".join([
            f"{m['role']}: {m['content']}" for m in messages
        ])
        
        prompt = f"""Summarize this conversation concisely:

{conversation}

Summary:"""
        
        return llm.generate(prompt)
    
    def get_context(self) -> List[dict]:
        """Get context with summary"""
        context = []
        
        if self.summary:
            context.append({
                "role": "system",
                "content": f"Previous conversation summary:\n{self.summary}"
            })
        
        context.extend(self.messages)
        return context
```

## Vector Databases for Semantic Memory

Store and retrieve information by meaning, not just keywords.

### Why Vector Databases?

Traditional search: "Find messages containing 'Python'"
Semantic search: "Find messages about programming languages"

### Basic Vector Memory

```python
import numpy as np
from typing import List, Tuple

class VectorMemory:
    """Simple vector-based memory"""
    
    def __init__(self):
        self.memories = []
        self.embeddings = []
    
    def add(self, text: str, metadata: dict = None):
        """Store memory with embedding"""
        # Get embedding
        embedding = self.get_embedding(text)
        
        self.memories.append({
            "text": text,
            "metadata": metadata or {},
            "timestamp": time.time()
        })
        self.embeddings.append(embedding)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        # Using OpenAI embeddings
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Search for relevant memories"""
        if not self.memories:
            return []
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Calculate similarities
        similarities = []
        for i, emb in enumerate(self.embeddings):
            similarity = self.cosine_similarity(query_embedding, emb)
            similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        results = []
        for i, score in similarities[:top_k]:
            result = self.memories[i].copy()
            result["similarity"] = score
            results.append(result)
        
        return results
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### Using Chroma

```python
import chromadb
from chromadb.config import Settings

class ChromaMemory:
    """Memory using ChromaDB"""
    
    def __init__(self, collection_name="agent_memory"):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Agent memory storage"}
        )
    
    def add(self, text: str, metadata: dict = None):
        """Add memory"""
        doc_id = f"mem_{int(time.time() * 1000)}"
        
        self.collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )
    
    def search(self, query: str, n_results: int = 5) -> List[dict]:
        """Search memories"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        memories = []
        for i in range(len(results['documents'][0])):
            memories.append({
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
        
        return memories
    
    def delete_all(self):
        """Clear all memories"""
        self.client.delete_collection(self.collection.name)
```

### Using Pinecone

```python
import pinecone

class PineconeMemory:
    """Memory using Pinecone"""
    
    def __init__(self, index_name="agent-memory"):
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENV")
        )
        
        # Create index if doesn't exist
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding size
                metric="cosine"
            )
        
        self.index = pinecone.Index(index_name)
    
    def add(self, text: str, metadata: dict = None):
        """Add memory"""
        # Get embedding
        embedding = self.get_embedding(text)
        
        # Generate ID
        doc_id = f"mem_{int(time.time() * 1000)}"
        
        # Upsert to Pinecone
        self.index.upsert([(
            doc_id,
            embedding,
            {
                "text": text,
                **(metadata or {})
            }
        )])
    
    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Search memories"""
        query_embedding = self.get_embedding(query)
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        memories = []
        for match in results['matches']:
            memories.append({
                "text": match['metadata']['text'],
                "score": match['score'],
                "metadata": match['metadata']
            })
        
        return memories
```

## Entity Tracking and State Management

Track entities (people, places, things) mentioned in conversations.

### Entity Extraction

```python
class EntityTracker:
    """Track entities across conversation"""
    
    def __init__(self):
        self.entities = {}
    
    def extract_entities(self, text: str) -> dict:
        """Extract entities from text"""
        prompt = f"""Extract entities from this text:

Text: {text}

Return as JSON:
{{
  "people": ["name1", "name2"],
  "places": ["place1"],
  "organizations": ["org1"],
  "dates": ["date1"],
  "other": ["thing1"]
}}"""
        
        response = llm.generate(prompt)
        return json.loads(response)
    
    def update(self, text: str):
        """Update entity tracking"""
        entities = self.extract_entities(text)
        
        for entity_type, items in entities.items():
            if entity_type not in self.entities:
                self.entities[entity_type] = {}
            
            for item in items:
                if item not in self.entities[entity_type]:
                    self.entities[entity_type][item] = {
                        "first_seen": time.time(),
                        "mentions": 0,
                        "context": []
                    }
                
                self.entities[entity_type][item]["mentions"] += 1
                self.entities[entity_type][item]["context"].append(text)
    
    def get_entity_info(self, entity: str) -> dict:
        """Get information about an entity"""
        for entity_type, items in self.entities.items():
            if entity in items:
                return {
                    "type": entity_type,
                    **items[entity]
                }
        return None
```

### State Management

```python
class StateManager:
    """Manage agent state"""
    
    def __init__(self):
        self.state = {
            "user_info": {},
            "current_task": None,
            "preferences": {},
            "context": {}
        }
    
    def update(self, key: str, value: any):
        """Update state"""
        keys = key.split('.')
        current = self.state
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def get(self, key: str, default=None):
        """Get state value"""
        keys = key.split('.')
        current = self.state
        
        for k in keys:
            if k not in current:
                return default
            current = current[k]
        
        return current
    
    def save(self, filepath: str):
        """Save state to file"""
        with open(filepath, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def load(self, filepath: str):
        """Load state from file"""
        with open(filepath, 'r') as f:
            self.state = json.load(f)
```

## Memory Retrieval Strategies

How to find relevant memories efficiently.

### Recency-Based Retrieval

```python
def get_recent_memories(memories: List[dict], n: int = 5) -> List[dict]:
    """Get most recent memories"""
    sorted_memories = sorted(
        memories,
        key=lambda x: x.get('timestamp', 0),
        reverse=True
    )
    return sorted_memories[:n]
```

### Relevance-Based Retrieval

```python
def get_relevant_memories(
    query: str,
    memories: List[dict],
    n: int = 5
) -> List[dict]:
    """Get most relevant memories using embeddings"""
    query_embedding = get_embedding(query)
    
    scored_memories = []
    for memory in memories:
        memory_embedding = memory.get('embedding')
        if memory_embedding:
            score = cosine_similarity(query_embedding, memory_embedding)
            scored_memories.append((memory, score))
    
    scored_memories.sort(key=lambda x: x[1], reverse=True)
    return [m for m, s in scored_memories[:n]]
```

### Hybrid Retrieval

Combine multiple factors:

```python
def hybrid_retrieval(
    query: str,
    memories: List[dict],
    n: int = 5,
    recency_weight: float = 0.3,
    relevance_weight: float = 0.7
) -> List[dict]:
    """Combine recency and relevance"""
    
    query_embedding = get_embedding(query)
    current_time = time.time()
    
    scored_memories = []
    for memory in memories:
        # Relevance score
        relevance = cosine_similarity(
            query_embedding,
            memory['embedding']
        )
        
        # Recency score (decay over time)
        age = current_time - memory['timestamp']
        recency = np.exp(-age / (24 * 3600))  # Decay over days
        
        # Combined score
        score = (
            relevance_weight * relevance +
            recency_weight * recency
        )
        
        scored_memories.append((memory, score))
    
    scored_memories.sort(key=lambda x: x[1], reverse=True)
    return [m for m, s in scored_memories[:n]]
```

### Importance-Based Retrieval

```python
def get_important_memories(
    memories: List[dict],
    n: int = 5
) -> List[dict]:
    """Get memories marked as important"""
    
    # Score by importance
    scored = []
    for memory in memories:
        importance = memory.get('importance', 0)
        scored.append((memory, importance))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    return [m for m, s in scored[:n]]

def calculate_importance(memory: dict) -> float:
    """Calculate memory importance"""
    prompt = f"""Rate the importance of remembering this information (0-10):

{memory['text']}

Consider:
- Is it about user preferences?
- Is it a key fact?
- Will it be useful later?

Importance (0-10):"""
    
    response = llm.generate(prompt)
    return float(response.strip())
```

## Complete Memory System

```python
class ComprehensiveMemory:
    """Full-featured memory system"""
    
    def __init__(self):
        # Short-term memory
        self.conversation = TokenAwareMemory(max_tokens=4000)
        
        # Long-term memory
        self.long_term = ChromaMemory()
        
        # Entity tracking
        self.entities = EntityTracker()
        
        # State management
        self.state = StateManager()
    
    def add_message(self, role: str, content: str):
        """Add message to conversation"""
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        
        # Add to short-term
        self.conversation.add(message)
        
        # Extract and track entities
        if role == "user":
            self.entities.update(content)
        
        # Store important messages in long-term
        if self.is_important(content):
            self.long_term.add(
                content,
                metadata={
                    "role": role,
                    "timestamp": time.time()
                }
            )
    
    def is_important(self, text: str) -> bool:
        """Determine if message should be stored long-term"""
        keywords = [
            "my name is", "i prefer", "remember",
            "always", "never", "i like", "i don't like"
        ]
        return any(kw in text.lower() for kw in keywords)
    
    def get_context(self, query: str = None) -> List[dict]:
        """Get relevant context for current query"""
        context = []
        
        # Add relevant long-term memories
        if query:
            relevant = self.long_term.search(query, n_results=3)
            if relevant:
                context.append({
                    "role": "system",
                    "content": "Relevant information from past:\n" +
                               "\n".join([m['text'] for m in relevant])
                })
        
        # Add recent conversation
        context.extend(self.conversation.get_context())
        
        return context
    
    def save(self, filepath: str):
        """Save memory state"""
        data = {
            "entities": self.entities.entities,
            "state": self.state.state,
            "timestamp": time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load memory state"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.entities.entities = data.get('entities', {})
        self.state.state = data.get('state', {})
```

## Using Memory in Agents

```python
class MemoryAgent:
    """Agent with comprehensive memory"""
    
    def __init__(self):
        self.memory = ComprehensiveMemory()
        self.client = openai.OpenAI()
    
    def chat(self, user_input: str) -> str:
        """Chat with memory"""
        
        # Add user message to memory
        self.memory.add_message("user", user_input)
        
        # Get context with relevant memories
        context = self.memory.get_context(query=user_input)
        
        # Generate response
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=context
        )
        
        assistant_message = response.choices[0].message.content
        
        # Add assistant response to memory
        self.memory.add_message("assistant", assistant_message)
        
        return assistant_message
    
    def save_session(self):
        """Save memory for later"""
        self.memory.save("session_memory.json")
    
    def load_session(self):
        """Load previous session"""
        self.memory.load("session_memory.json")
```

## Best Practices

1. **Separate short and long-term**: Different storage for different needs
2. **Be selective**: Don't store everything
3. **Use semantic search**: Find by meaning, not keywords
4. **Track importance**: Prioritize valuable information
5. **Manage token budgets**: Don't overflow context
6. **Summarize old conversations**: Compress history
7. **Update entities**: Track what's mentioned
8. **Persist critical data**: Save to disk/database
9. **Retrieve strategically**: Balance recency, relevance, importance
10. **Test retrieval**: Ensure you find what you need

## Next Steps

With memory systems in place, agents can maintain context and learn from experience. Next, we'll explore multi-agent systems where multiple agents collaborate!
