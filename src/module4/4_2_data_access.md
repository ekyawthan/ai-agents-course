# Data Access & Retrieval

## RAG (Retrieval Augmented Generation)

RAG combines retrieval with generation to provide accurate, grounded responses.

### Why RAG?

**Without RAG**:
- LLM relies on training data (may be outdated)
- Can hallucinate facts
- No access to private/recent information

**With RAG**:
- Retrieves relevant documents first
- Grounds responses in actual data
- Works with private knowledge bases
- Always up-to-date

### Basic RAG Pipeline

```python
class SimpleRAG:
    """Basic RAG implementation"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.client = openai.OpenAI()
    
    def add_document(self, text: str, metadata: dict = None):
        """Add document to knowledge base"""
        # Create embedding
        embedding = self.get_embedding(text)
        
        self.documents.append({
            "text": text,
            "metadata": metadata or {},
            "id": len(self.documents)
        })
        self.embeddings.append(embedding)
    
    def get_embedding(self, text: str) -> list:
        """Get embedding for text"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Retrieve relevant documents"""
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        # Sort and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, score in similarities[:top_k]:
            doc = self.documents[i].copy()
            doc['score'] = score
            results.append(doc)
        
        return results
    
    def query(self, question: str) -> str:
        """Answer question using RAG"""
        # Retrieve relevant documents
        docs = self.retrieve(question, top_k=3)
        
        # Build context
        context = "\n\n".join([
            f"Document {i+1}:\n{doc['text']}"
            for i, doc in enumerate(docs)
        ])
        
        # Generate answer
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "Answer questions based on the provided context. If the answer isn't in the context, say so."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ]
        )
        
        return response.choices[0].message.content
    
    def cosine_similarity(self, a: list, b: list) -> float:
        """Calculate cosine similarity"""
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Usage
rag = SimpleRAG()

# Add documents
rag.add_document("Python is a high-level programming language.")
rag.add_document("JavaScript is used for web development.")
rag.add_document("Python is popular for data science and AI.")

# Query
answer = rag.query("What is Python used for?")
print(answer)
```

### Advanced RAG with LangChain

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

class AdvancedRAG:
    """RAG using LangChain"""
    
    def __init__(self, persist_directory="./chroma_db"):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vectorstore = None
        self.persist_directory = persist_directory
    
    def load_documents(self, documents: list):
        """Load and process documents"""
        # Split documents into chunks
        chunks = self.text_splitter.create_documents(documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
    
    def query(self, question: str) -> dict:
        """Query with source attribution"""
        if not self.vectorstore:
            return {"answer": "No documents loaded", "sources": []}
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0),
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        # Query
        result = qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "sources": [doc.page_content for doc in result["source_documents"]]
        }
```

### Chunking Strategies

```python
class DocumentChunker:
    """Different chunking strategies"""
    
    def chunk_by_tokens(self, text: str, chunk_size: int = 512, overlap: int = 50) -> list:
        """Chunk by token count"""
        import tiktoken
        
        encoding = tiktoken.encoding_for_model("gpt-4")
        tokens = encoding.encode(text)
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end - overlap
        
        return chunks
    
    def chunk_by_sentences(self, text: str, sentences_per_chunk: int = 5) -> list:
        """Chunk by sentences"""
        import re
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = ". ".join(sentences[i:i+sentences_per_chunk]) + "."
            chunks.append(chunk)
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> list:
        """Chunk by paragraphs"""
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
    
    def semantic_chunking(self, text: str, similarity_threshold: float = 0.7) -> list:
        """Chunk based on semantic similarity"""
        sentences = self.split_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # Check similarity with current chunk
            chunk_text = " ".join(current_chunk)
            similarity = self.calculate_similarity(chunk_text, sentences[i])
            
            if similarity >= similarity_threshold:
                current_chunk.append(sentences[i])
            else:
                # Start new chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
        
        # Add last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
```

## Database Queries

### SQL Databases

```python
import sqlite3
from typing import List, Dict

class SQLAgent:
    """Agent that can query SQL databases"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.client = openai.OpenAI()
    
    def get_schema(self) -> str:
        """Get database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        schema = []
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            schema.append(f"Table: {table_name}")
            for col in columns:
                schema.append(f"  - {col[1]} ({col[2]})")
        
        conn.close()
        return "\n".join(schema)
    
    def natural_language_query(self, question: str) -> Dict:
        """Convert natural language to SQL and execute"""
        # Generate SQL
        sql = self.generate_sql(question)
        
        # Execute SQL
        results = self.execute_sql(sql)
        
        # Format response
        answer = self.format_results(question, results)
        
        return {
            "question": question,
            "sql": sql,
            "results": results,
            "answer": answer
        }
    
    def generate_sql(self, question: str) -> str:
        """Generate SQL from natural language"""
        schema = self.get_schema()
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a SQL expert. Convert natural language questions to SQL queries.

Database schema:
{schema}

Rules:
- Return only the SQL query, no explanations
- Use proper SQL syntax
- Be careful with column names
- Use appropriate JOINs when needed"""
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.1
        )
        
        sql = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        sql = sql.replace("```sql", "").replace("```", "").strip()
        
        return sql
    
    def execute_sql(self, sql: str) -> List[Dict]:
        """Execute SQL query safely"""
        # Validate query (read-only)
        if not self.is_safe_query(sql):
            raise ValueError("Only SELECT queries are allowed")
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
            
            # Convert to list of dicts
            results = [dict(row) for row in rows]
            
            conn.close()
            return results
            
        except Exception as e:
            conn.close()
            raise Exception(f"SQL execution error: {str(e)}")
    
    def is_safe_query(self, sql: str) -> bool:
        """Check if query is safe (read-only)"""
        sql_upper = sql.upper().strip()
        
        # Only allow SELECT
        if not sql_upper.startswith("SELECT"):
            return False
        
        # Disallow dangerous keywords
        dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE"]
        for keyword in dangerous:
            if keyword in sql_upper:
                return False
        
        return True
    
    def format_results(self, question: str, results: List[Dict]) -> str:
        """Format results as natural language"""
        if not results:
            return "No results found."
        
        # Convert results to text
        results_text = "\n".join([str(row) for row in results[:10]])
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": f"""Answer this question based on the query results:

Question: {question}

Results:
{results_text}

Provide a clear, natural language answer:"""
                }
            ]
        )
        
        return response.choices[0].message.content

# Usage
agent = SQLAgent("company.db")
result = agent.natural_language_query("How many employees are in the sales department?")
print(result['answer'])
```

### NoSQL Databases

```python
from pymongo import MongoClient

class MongoDBAgent:
    """Agent for MongoDB queries"""
    
    def __init__(self, connection_string: str, database: str):
        self.client = MongoClient(connection_string)
        self.db = self.client[database]
        self.llm = openai.OpenAI()
    
    def query(self, question: str, collection: str) -> dict:
        """Query MongoDB using natural language"""
        # Generate MongoDB query
        query_dict = self.generate_query(question, collection)
        
        # Execute query
        results = list(self.db[collection].find(query_dict).limit(10))
        
        # Format response
        answer = self.format_results(question, results)
        
        return {
            "question": question,
            "query": query_dict,
            "results": results,
            "answer": answer
        }
    
    def generate_query(self, question: str, collection: str) -> dict:
        """Generate MongoDB query from natural language"""
        # Get sample document
        sample = self.db[collection].find_one()
        
        response = self.llm.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": f"""Convert natural language to MongoDB query.

Collection: {collection}
Sample document: {sample}

Return only valid JSON for MongoDB find() query."""
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.1
        )
        
        import json
        query_str = response.choices[0].message.content.strip()
        return json.loads(query_str)
```

## API Integrations

### REST API Client

```python
import requests
from typing import Optional

class APIAgent:
    """Agent that can call REST APIs"""
    
    def __init__(self):
        self.client = openai.OpenAI()
        self.session = requests.Session()
    
    def call_api(self, 
                 url: str,
                 method: str = "GET",
                 headers: Optional[dict] = None,
                 params: Optional[dict] = None,
                 data: Optional[dict] = None) -> dict:
        """Make API call"""
        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=data,
                timeout=30
            )
            
            response.raise_for_status()
            
            return {
                "success": True,
                "status_code": response.status_code,
                "data": response.json() if response.content else None
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def natural_language_api_call(self, request: str, api_spec: dict) -> dict:
        """Convert natural language to API call"""
        # Generate API call parameters
        params = self.generate_api_params(request, api_spec)
        
        # Make API call
        result = self.call_api(**params)
        
        # Format response
        if result['success']:
            answer = self.format_api_response(request, result['data'])
            return {
                "request": request,
                "api_call": params,
                "response": result['data'],
                "answer": answer
            }
        else:
            return {
                "request": request,
                "error": result['error']
            }
    
    def generate_api_params(self, request: str, api_spec: dict) -> dict:
        """Generate API parameters from natural language"""
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": f"""Convert natural language to API call parameters.

API Specification:
{json.dumps(api_spec, indent=2)}

Return JSON with: url, method, headers, params, data"""
                },
                {
                    "role": "user",
                    "content": request
                }
            ],
            temperature=0.1
        )
        
        import json
        return json.loads(response.choices[0].message.content)
```

### GraphQL Client

```python
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

class GraphQLAgent:
    """Agent for GraphQL APIs"""
    
    def __init__(self, endpoint: str):
        transport = RequestsHTTPTransport(url=endpoint)
        self.client = Client(transport=transport, fetch_schema_from_transport=True)
        self.llm = openai.OpenAI()
    
    def query(self, natural_language_query: str) -> dict:
        """Execute GraphQL query from natural language"""
        # Generate GraphQL query
        graphql_query = self.generate_graphql(natural_language_query)
        
        # Execute query
        query = gql(graphql_query)
        result = self.client.execute(query)
        
        # Format response
        answer = self.format_results(natural_language_query, result)
        
        return {
            "question": natural_language_query,
            "graphql": graphql_query,
            "result": result,
            "answer": answer
        }
    
    def generate_graphql(self, question: str) -> str:
        """Generate GraphQL query"""
        schema = self.client.schema
        
        response = self.llm.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": f"""Generate GraphQL query from natural language.

Schema: {schema}

Return only the GraphQL query."""
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )
        
        return response.choices[0].message.content.strip()
```

## File System Operations

### Safe File Access

```python
import os
from pathlib import Path

class FileSystemAgent:
    """Agent with safe file system access"""
    
    def __init__(self, allowed_directory: str):
        self.allowed_directory = Path(allowed_directory).resolve()
    
    def is_safe_path(self, path: str) -> bool:
        """Check if path is within allowed directory"""
        try:
            requested_path = (self.allowed_directory / path).resolve()
            return requested_path.is_relative_to(self.allowed_directory)
        except:
            return False
    
    def read_file(self, path: str) -> dict:
        """Read file safely"""
        if not self.is_safe_path(path):
            return {"success": False, "error": "Access denied"}
        
        try:
            full_path = self.allowed_directory / path
            with open(full_path, 'r') as f:
                content = f.read()
            
            return {
                "success": True,
                "content": content,
                "size": len(content)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_files(self, path: str = ".") -> dict:
        """List files in directory"""
        if not self.is_safe_path(path):
            return {"success": False, "error": "Access denied"}
        
        try:
            full_path = self.allowed_directory / path
            files = []
            
            for item in full_path.iterdir():
                files.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None
                })
            
            return {
                "success": True,
                "files": files
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_files(self, pattern: str, path: str = ".") -> dict:
        """Search for files matching pattern"""
        if not self.is_safe_path(path):
            return {"success": False, "error": "Access denied"}
        
        try:
            full_path = self.allowed_directory / path
            matches = list(full_path.rglob(pattern))
            
            results = [
                {
                    "path": str(m.relative_to(self.allowed_directory)),
                    "name": m.name,
                    "size": m.stat().st_size if m.is_file() else None
                }
                for m in matches
            ]
            
            return {
                "success": True,
                "matches": results
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
```

## Complete Data Access Agent

```python
class DataAccessAgent:
    """Unified agent for data access"""
    
    def __init__(self):
        self.rag = SimpleRAG()
        self.sql_agent = None
        self.api_agent = APIAgent()
        self.fs_agent = None
        self.client = openai.OpenAI()
    
    def configure_sql(self, db_path: str):
        """Configure SQL access"""
        self.sql_agent = SQLAgent(db_path)
    
    def configure_filesystem(self, allowed_dir: str):
        """Configure file system access"""
        self.fs_agent = FileSystemAgent(allowed_dir)
    
    def query(self, question: str) -> str:
        """Answer question using appropriate data source"""
        # Determine which data source to use
        source = self.determine_source(question)
        
        if source == "rag":
            return self.rag.query(question)
        elif source == "sql" and self.sql_agent:
            result = self.sql_agent.natural_language_query(question)
            return result['answer']
        elif source == "api":
            # Would need API spec
            return "API access requires configuration"
        elif source == "filesystem" and self.fs_agent:
            # Would need to determine file operation
            return "File system access requires specific operation"
        else:
            return "Unable to determine appropriate data source"
    
    def determine_source(self, question: str) -> str:
        """Determine which data source to use"""
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": f"""Which data source should be used for this question?

Question: {question}

Options: rag, sql, api, filesystem

Answer with just the option:"""
                }
            ],
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip().lower()
```

## Best Practices

1. **Validate queries**: Check SQL/API calls before execution
2. **Limit results**: Don't return huge datasets
3. **Cache responses**: Avoid redundant queries
4. **Handle errors**: Graceful failure handling
5. **Secure credentials**: Never expose API keys
6. **Rate limiting**: Respect API limits
7. **Chunk large documents**: Better retrieval
8. **Use appropriate embeddings**: Match your use case
9. **Monitor costs**: Track API usage
10. **Test thoroughly**: Verify data access works

## Next Steps

You now understand data access and retrieval! Next, we'll explore web interaction including browser automation and scraping.
