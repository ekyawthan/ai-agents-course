# Research Agents

## Introduction to Research Agents

Research agents are specialized AI systems that gather, analyze, and synthesize information from multiple sources to answer complex questions or investigate topics in depth.

### What Makes Research Agents Unique?

**Core Capabilities**:
- Multi-source information gathering
- Source credibility assessment
- Information synthesis and summarization
- Citation management
- Fact verification
- Deep topic exploration

**Key Challenges**:
- Information overload
- Source reliability
- Conflicting information
- Bias detection
- Citation accuracy
- Staying current

### Types of Research Agents

1. **Academic Research Agents**: Literature reviews, paper analysis
2. **Market Research Agents**: Competitive analysis, trends
3. **Investigative Agents**: Deep dives, fact-checking
4. **News Aggregation Agents**: Current events, monitoring
5. **Technical Research Agents**: Documentation, specifications

## Information Gathering Strategies

### Multi-Source Search

```python
from typing import List, Dict
import requests
from bs4 import BeautifulSoup

class MultiSourceSearcher:
    """Search across multiple sources"""
    
    def __init__(self):
        self.client = openai.OpenAI()
        self.sources = {
            "web": self.search_web,
            "academic": self.search_academic,
            "news": self.search_news,
            "social": self.search_social
        }
    
    def search_all_sources(self, query: str, sources: List[str] = None) -> Dict:
        """Search across all specified sources"""
        if sources is None:
            sources = list(self.sources.keys())
        
        results = {}
        
        for source in sources:
            if source in self.sources:
                print(f"Searching {source}...")
                results[source] = self.sources[source](query)
        
        return results
    
    def search_web(self, query: str) -> List[Dict]:
        """Search general web"""
        # Using a search API (example with Google Custom Search)
        api_key = os.getenv("GOOGLE_API_KEY")
        search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": search_engine_id,
            "q": query,
            "num": 10
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("items", []):
                results.append({
                    "title": item["title"],
                    "url": item["link"],
                    "snippet": item["snippet"],
                    "source": "web"
                })
            
            return results
        except Exception as e:
            print(f"Web search error: {e}")
            return []
    
    def search_academic(self, query: str) -> List[Dict]:
        """Search academic sources (arXiv, PubMed, etc.)"""
        # Example with arXiv API
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": 10
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse XML response
            from xml.etree import ElementTree as ET
            root = ET.fromstring(response.content)
            
            results = []
            for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                title = entry.find("{http://www.w3.org/2005/Atom}title").text
                summary = entry.find("{http://www.w3.org/2005/Atom}summary").text
                link = entry.find("{http://www.w3.org/2005/Atom}id").text
                
                results.append({
                    "title": title.strip(),
                    "url": link,
                    "snippet": summary.strip()[:200],
                    "source": "academic"
                })
            
            return results
        except Exception as e:
            print(f"Academic search error: {e}")
            return []
    
    def search_news(self, query: str) -> List[Dict]:
        """Search news sources"""
        # Example with News API
        api_key = os.getenv("NEWS_API_KEY")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": api_key,
            "pageSize": 10,
            "sortBy": "relevancy"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for article in data.get("articles", []):
                results.append({
                    "title": article["title"],
                    "url": article["url"],
                    "snippet": article["description"],
                    "source": "news",
                    "published": article.get("publishedAt")
                })
            
            return results
        except Exception as e:
            print(f"News search error: {e}")
            return []
    
    def search_social(self, query: str) -> List[Dict]:
        """Search social media (Twitter, Reddit, etc.)"""
        # Example implementation for Reddit
        url = f"https://www.reddit.com/search.json"
        params = {
            "q": query,
            "limit": 10,
            "sort": "relevance"
        }
        headers = {"User-Agent": "ResearchAgent/1.0"}
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for post in data["data"]["children"]:
                post_data = post["data"]
                results.append({
                    "title": post_data["title"],
                    "url": f"https://reddit.com{post_data['permalink']}",
                    "snippet": post_data.get("selftext", "")[:200],
                    "source": "social",
                    "score": post_data.get("score", 0)
                })
            
            return results
        except Exception as e:
            print(f"Social search error: {e}")
            return []

# Usage
searcher = MultiSourceSearcher()
results = searcher.search_all_sources("artificial intelligence agents")

for source, items in results.items():
    print(f"\n{source.upper()} Results: {len(items)}")
    for item in items[:3]:
        print(f"  - {item['title']}")
```

### Deep Content Extraction

```python
class ContentExtractor:
    """Extract and process content from sources"""
    
    def __init__(self):
        self.client = openai.OpenAI()
    
    def extract_from_url(self, url: str) -> Dict:
        """Extract main content from URL"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator='\n', strip=True)
            
            # Extract metadata
            title = soup.find('title')
            title_text = title.string if title else ""
            
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc['content'] if meta_desc else ""
            
            return {
                "url": url,
                "title": title_text,
                "description": description,
                "content": text[:10000],  # Limit content
                "word_count": len(text.split())
            }
            
        except Exception as e:
            return {
                "url": url,
                "error": str(e)
            }
    
    def extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content"""
        prompt = f"""Extract the key points from this content:

{content[:4000]}

Provide 5-7 bullet points of the most important information:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        points = response.choices[0].message.content.strip().split('\n')
        return [p.strip('- ').strip() for p in points if p.strip()]
    
    def extract_quotes(self, content: str, topic: str) -> List[Dict]:
        """Extract relevant quotes"""
        prompt = f"""Find relevant quotes about "{topic}" from this content:

{content[:4000]}

Provide 3-5 direct quotes with context:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        # Parse quotes
        quotes_text = response.choices[0].message.content
        # Simple parsing - in production, use more robust method
        quotes = []
        for line in quotes_text.split('\n'):
            if line.strip().startswith('"'):
                quotes.append({"quote": line.strip(), "context": ""})
        
        return quotes

# Usage
extractor = ContentExtractor()

# Extract content
content = extractor.extract_from_url("https://example.com/article")
print(f"Title: {content['title']}")
print(f"Words: {content['word_count']}")

# Extract key points
key_points = extractor.extract_key_points(content['content'])
for point in key_points:
    print(f"  • {point}")
```

## Source Verification

### Credibility Assessment

```python
class SourceVerifier:
    """Verify source credibility and reliability"""
    
    def __init__(self):
        self.client = openai.OpenAI()
        self.trusted_domains = {
            "academic": [".edu", ".gov", "arxiv.org", "pubmed.gov"],
            "news": ["reuters.com", "apnews.com", "bbc.com"],
            "tech": ["github.com", "stackoverflow.com"]
        }
    
    def assess_credibility(self, url: str, content: str = None) -> Dict:
        """Assess source credibility"""
        from urllib.parse import urlparse
        
        domain = urlparse(url).netloc
        
        # Check against trusted domains
        trust_level = "unknown"
        for category, domains in self.trusted_domains.items():
            if any(trusted in domain for trusted in domains):
                trust_level = "high"
                break
        
        # Analyze content if provided
        content_score = None
        if content:
            content_score = self.analyze_content_quality(content)
        
        return {
            "url": url,
            "domain": domain,
            "trust_level": trust_level,
            "content_quality": content_score,
            "is_trusted": trust_level == "high"
        }
    
    def analyze_content_quality(self, content: str) -> Dict:
        """Analyze content quality indicators"""
        prompt = f"""Analyze the quality and credibility of this content:

{content[:2000]}

Rate (1-5) on:
1. Factual accuracy (based on claims made)
2. Objectivity (bias level)
3. Citation quality (references provided)
4. Writing quality (clarity, professionalism)
5. Depth of analysis

Provide scores and brief explanation:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return self.parse_quality_scores(response.choices[0].message.content)
    
    def cross_reference(self, claim: str, sources: List[Dict]) -> Dict:
        """Cross-reference a claim across sources"""
        confirmations = 0
        contradictions = 0
        
        for source in sources:
            result = self.check_claim_in_source(claim, source.get("content", ""))
            
            if result == "confirms":
                confirmations += 1
            elif result == "contradicts":
                contradictions += 1
        
        return {
            "claim": claim,
            "confirmations": confirmations,
            "contradictions": contradictions,
            "confidence": confirmations / len(sources) if sources else 0
        }
    
    def check_claim_in_source(self, claim: str, content: str) -> str:
        """Check if source confirms, contradicts, or is neutral on claim"""
        prompt = f"""Does this content confirm, contradict, or neither regarding this claim?

Claim: {claim}

Content: {content[:1000]}

Answer with just: confirms, contradicts, or neutral"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip().lower()

# Usage
verifier = SourceVerifier()

# Assess credibility
credibility = verifier.assess_credibility(
    "https://arxiv.org/abs/2023.12345",
    "This paper presents..."
)
print(f"Trust level: {credibility['trust_level']}")

# Cross-reference claim
claim = "AI agents can autonomously complete complex tasks"
sources = [
    {"content": "Research shows AI agents are capable of..."},
    {"content": "Studies indicate autonomous agents can..."}
]
verification = verifier.cross_reference(claim, sources)
print(f"Confidence: {verification['confidence']:.0%}")
```

## Synthesis and Summarization

### Information Synthesis

```python
class InformationSynthesizer:
    """Synthesize information from multiple sources"""
    
    def __init__(self):
        self.client = openai.OpenAI()
    
    def synthesize_sources(self, 
                          query: str,
                          sources: List[Dict],
                          style: str = "comprehensive") -> str:
        """Synthesize information from multiple sources"""
        
        # Prepare source summaries
        source_texts = []
        for i, source in enumerate(sources[:10], 1):  # Limit to 10 sources
            source_texts.append(f"""
Source {i}: {source.get('title', 'Unknown')}
URL: {source.get('url', 'N/A')}
Content: {source.get('snippet', source.get('content', ''))[:500]}
""")
        
        sources_combined = "\n---\n".join(source_texts)
        
        style_instructions = {
            "comprehensive": "Provide a detailed, thorough analysis",
            "concise": "Provide a brief, focused summary",
            "academic": "Use formal, academic tone with citations",
            "casual": "Use conversational, accessible language"
        }
        
        prompt = f"""Synthesize information about: {query}

Sources:
{sources_combined}

{style_instructions.get(style, style_instructions['comprehensive'])}.

Requirements:
- Integrate information from multiple sources
- Identify common themes and patterns
- Note any contradictions
- Cite sources [1], [2], etc.
- Provide balanced perspective

Synthesis:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=2000
        )
        
        synthesis = response.choices[0].message.content
        
        # Add source list
        source_list = "\n\nSources:\n"
        for i, source in enumerate(sources[:10], 1):
            source_list += f"[{i}] {source.get('title', 'Unknown')} - {source.get('url', 'N/A')}\n"
        
        return synthesis + source_list
    
    def identify_themes(self, sources: List[Dict]) -> List[Dict]:
        """Identify common themes across sources"""
        # Combine content
        combined_content = "\n\n".join([
            s.get('snippet', s.get('content', ''))[:500]
            for s in sources[:20]
        ])
        
        prompt = f"""Identify the main themes in these sources:

{combined_content}

List 5-7 key themes with:
- Theme name
- Brief description
- How many sources mention it

Themes:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return self.parse_themes(response.choices[0].message.content)
    
    def find_contradictions(self, sources: List[Dict]) -> List[Dict]:
        """Find contradictions between sources"""
        contradictions = []
        
        # Compare sources pairwise (simplified)
        for i in range(min(5, len(sources))):
            for j in range(i+1, min(5, len(sources))):
                source_a = sources[i]
                source_b = sources[j]
                
                prompt = f"""Do these sources contradict each other?

Source A: {source_a.get('snippet', '')[:300]}

Source B: {source_b.get('snippet', '')[:300]}

If yes, explain the contradiction. If no, say "no contradiction".

Analysis:"""
                
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                
                result = response.choices[0].message.content
                
                if "no contradiction" not in result.lower():
                    contradictions.append({
                        "source_a": source_a.get('title'),
                        "source_b": source_b.get('title'),
                        "contradiction": result
                    })
        
        return contradictions

# Usage
synthesizer = InformationSynthesizer()

sources = [
    {"title": "AI Agents Overview", "url": "...", "snippet": "AI agents are..."},
    {"title": "Agent Architectures", "url": "...", "snippet": "Modern agents use..."},
    # ... more sources
]

# Synthesize
synthesis = synthesizer.synthesize_sources(
    "What are AI agents?",
    sources,
    style="comprehensive"
)
print(synthesis)

# Identify themes
themes = synthesizer.identify_themes(sources)
for theme in themes:
    print(f"Theme: {theme}")
```

## Citation Management

### Automatic Citation Generation

```python
class CitationManager:
    """Manage citations and references"""
    
    def __init__(self):
        self.citations = []
        self.citation_style = "APA"  # APA, MLA, Chicago
    
    def add_citation(self, source: Dict) -> int:
        """Add source and return citation number"""
        self.citations.append(source)
        return len(self.citations)
    
    def format_citation(self, source: Dict, style: str = None) -> str:
        """Format citation in specified style"""
        style = style or self.citation_style
        
        if style == "APA":
            return self.format_apa(source)
        elif style == "MLA":
            return self.format_mla(source)
        elif style == "Chicago":
            return self.format_chicago(source)
        else:
            return self.format_simple(source)
    
    def format_apa(self, source: Dict) -> str:
        """Format in APA style"""
        author = source.get('author', 'Unknown')
        year = source.get('year', 'n.d.')
        title = source.get('title', 'Untitled')
        url = source.get('url', '')
        
        return f"{author}. ({year}). {title}. Retrieved from {url}"
    
    def format_mla(self, source: Dict) -> str:
        """Format in MLA style"""
        author = source.get('author', 'Unknown')
        title = source.get('title', 'Untitled')
        website = source.get('website', 'Web')
        url = source.get('url', '')
        
        return f'{author}. "{title}." {website}. {url}.'
    
    def format_simple(self, source: Dict) -> str:
        """Simple format"""
        title = source.get('title', 'Untitled')
        url = source.get('url', '')
        return f"{title} - {url}"
    
    def generate_bibliography(self) -> str:
        """Generate full bibliography"""
        bibliography = "References:\n\n"
        
        for i, source in enumerate(self.citations, 1):
            citation = self.format_citation(source)
            bibliography += f"{i}. {citation}\n"
        
        return bibliography
    
    def inline_cite(self, text: str, citation_num: int) -> str:
        """Add inline citation to text"""
        return f"{text} [{citation_num}]"

# Usage
citations = CitationManager()

# Add sources
source1 = {
    "author": "Smith, J.",
    "year": "2023",
    "title": "Understanding AI Agents",
    "url": "https://example.com/article"
}

cite_num = citations.add_citation(source1)

# Use in text
text = citations.inline_cite("AI agents are autonomous systems", cite_num)
print(text)  # "AI agents are autonomous systems [1]"

# Generate bibliography
print(citations.generate_bibliography())
```

## Complete Research Agent

```python
class ResearchAgent:
    """Complete research agent system"""
    
    def __init__(self):
        self.searcher = MultiSourceSearcher()
        self.extractor = ContentExtractor()
        self.verifier = SourceVerifier()
        self.synthesizer = InformationSynthesizer()
        self.citations = CitationManager()
        self.client = openai.OpenAI()
    
    def research(self, 
                query: str,
                depth: str = "medium",
                sources: List[str] = None) -> Dict:
        """Conduct comprehensive research"""
        
        print(f"🔍 Researching: {query}\n")
        
        # 1. Search multiple sources
        print("📚 Gathering sources...")
        search_results = self.searcher.search_all_sources(query, sources)
        
        all_sources = []
        for source_type, results in search_results.items():
            all_sources.extend(results)
        
        print(f"Found {len(all_sources)} sources\n")
        
        # 2. Extract and verify content
        print("📖 Extracting content...")
        verified_sources = []
        
        for source in all_sources[:20]:  # Limit processing
            # Extract content
            if 'content' not in source:
                content_data = self.extractor.extract_from_url(source['url'])
                source['content'] = content_data.get('content', source.get('snippet', ''))
            
            # Verify credibility
            credibility = self.verifier.assess_credibility(
                source['url'],
                source.get('content', '')
            )
            
            if credibility['is_trusted'] or credibility['trust_level'] != 'low':
                source['credibility'] = credibility
                verified_sources.append(source)
                
                # Add citation
                cite_num = self.citations.add_citation(source)
                source['citation_num'] = cite_num
        
        print(f"Verified {len(verified_sources)} sources\n")
        
        # 3. Synthesize information
        print("✍️  Synthesizing findings...")
        synthesis = self.synthesizer.synthesize_sources(
            query,
            verified_sources,
            style="comprehensive" if depth == "deep" else "concise"
        )
        
        # 4. Identify themes
        themes = self.synthesizer.identify_themes(verified_sources)
        
        # 5. Find contradictions
        contradictions = self.synthesizer.find_contradictions(verified_sources)
        
        # 6. Generate bibliography
        bibliography = self.citations.generate_bibliography()
        
        return {
            "query": query,
            "synthesis": synthesis,
            "themes": themes,
            "contradictions": contradictions,
            "sources": verified_sources,
            "bibliography": bibliography,
            "source_count": len(verified_sources)
        }
    
    def deep_dive(self, topic: str, subtopics: List[str] = None) -> Dict:
        """Deep research on topic with subtopics"""
        
        if not subtopics:
            # Generate subtopics
            subtopics = self.generate_subtopics(topic)
        
        results = {
            "topic": topic,
            "subtopics": {}
        }
        
        for subtopic in subtopics:
            print(f"\n📌 Researching subtopic: {subtopic}")
            result = self.research(f"{topic}: {subtopic}", depth="medium")
            results["subtopics"][subtopic] = result
        
        # Create overall synthesis
        print("\n🔗 Creating overall synthesis...")
        overall = self.synthesize_deep_dive(topic, results["subtopics"])
        results["overall_synthesis"] = overall
        
        return results
    
    def generate_subtopics(self, topic: str) -> List[str]:
        """Generate relevant subtopics"""
        prompt = f"""Generate 5 key subtopics for researching: {topic}

Subtopics should:
- Cover different aspects
- Be specific and focused
- Be researchable

List:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        subtopics = response.choices[0].message.content.strip().split('\n')
        return [s.strip('- 0123456789.').strip() for s in subtopics if s.strip()]
    
    def synthesize_deep_dive(self, topic: str, subtopic_results: Dict) -> str:
        """Synthesize results from deep dive"""
        # Combine all syntheses
        combined = f"# Comprehensive Research: {topic}\n\n"
        
        for subtopic, result in subtopic_results.items():
            combined += f"## {subtopic}\n\n"
            combined += result['synthesis'] + "\n\n"
        
        return combined
    
    def fact_check(self, claim: str) -> Dict:
        """Fact-check a specific claim"""
        print(f"🔎 Fact-checking: {claim}\n")
        
        # Search for information about the claim
        results = self.research(claim, depth="medium")
        
        # Cross-reference
        verification = self.verifier.cross_reference(
            claim,
            results['sources']
        )
        
        # Determine verdict
        if verification['confidence'] > 0.7:
            verdict = "Likely True"
        elif verification['confidence'] < 0.3:
            verdict = "Likely False"
        else:
            verdict = "Unclear/Mixed Evidence"
        
        return {
            "claim": claim,
            "verdict": verdict,
            "confidence": verification['confidence'],
            "confirmations": verification['confirmations'],
            "contradictions": verification['contradictions'],
            "sources": results['sources'][:5],
            "explanation": results['synthesis']
        }

# Usage
agent = ResearchAgent()

# Basic research
result = agent.research("What are the latest developments in AI agents?")
print(result['synthesis'])
print(f"\nSources: {result['source_count']}")

# Deep dive
deep_result = agent.deep_dive(
    "AI Agent Architectures",
    subtopics=["ReAct Pattern", "Memory Systems", "Tool Use"]
)

# Fact check
fact_result = agent.fact_check("AI agents can autonomously write production code")
print(f"Verdict: {fact_result['verdict']}")
print(f"Confidence: {fact_result['confidence']:.0%}")
```

## Best Practices

1. **Multi-source verification**: Never rely on single source
2. **Assess credibility**: Check source reliability
3. **Cite properly**: Always attribute information
4. **Check recency**: Ensure information is current
5. **Cross-reference**: Verify claims across sources
6. **Note contradictions**: Highlight conflicting information
7. **Maintain objectivity**: Present balanced view
8. **Track sources**: Keep detailed records
9. **Update regularly**: Refresh research periodically
10. **Human review**: Critical research needs expert review

## Next Steps

You now have comprehensive knowledge of research agents! Next, we'll explore task automation agents that handle repetitive workflows.
