# Hands-On Project: Shopping Research Assistant

## Project Overview

Build a **Shopping Research Assistant** that helps users make informed purchasing decisions by:
- Searching for products across multiple sources
- Comparing prices and features
- Reading product reviews
- Summarizing pros and cons
- Providing recommendations with reasoning

This project combines everything you've learned: ReAct pattern, tool integration, multi-step reasoning, and error handling.

## What You'll Build

An agent that can handle queries like:
- "Find the best laptop under $1000 for programming"
- "Compare noise-canceling headphones"
- "What are the top-rated coffee makers?"
- "Should I buy the iPhone 15 or Samsung S24?"

## Project Setup

### Dependencies

```bash
pip install openai requests beautifulsoup4 python-dotenv
```

### Project Structure

```
shopping_agent/
├── agent.py           # Main agent implementation
├── tools.py           # Tool definitions
├── config.py          # Configuration
├── .env              # API keys
└── test_agent.py     # Test cases
```

### Configuration

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4"
MAX_STEPS = 15
TEMPERATURE = 0.7
```

## Implement the Tools

### Tool 1: Product Search

```python
# tools.py
import requests
from typing import Dict, List

def search_products(query: str, max_results: int = 5) -> str:
    """
    Search for products matching the query.
    Returns product names, prices, and URLs.
    """
    try:
        # Using a mock API for demonstration
        # In production, use real APIs like Amazon Product API, eBay, etc.
        
        # Simulate search results
        results = [
            {
                "name": f"Product {i+1} for {query}",
                "price": f"${100 + i*50}",
                "rating": f"{4.0 + i*0.2:.1f}/5.0",
                "url": f"https://example.com/product-{i+1}"
            }
            for i in range(max_results)
        ]
        
        # Format results
        output = f"Found {len(results)} products:\n\n"
        for i, product in enumerate(results, 1):
            output += f"{i}. {product['name']}\n"
            output += f"   Price: {product['price']}\n"
            output += f"   Rating: {product['rating']}\n"
            output += f"   URL: {product['url']}\n\n"
        
        return output
    
    except Exception as e:
        return f"Error searching products: {str(e)}"


def search_products_real(query: str, max_results: int = 5) -> str:
    """
    Real implementation using web search.
    Searches Google Shopping or similar.
    """
    try:
        # Example with Google Custom Search API
        api_key = os.getenv("GOOGLE_API_KEY")
        search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": search_engine_id,
            "q": query + " buy price",
            "num": max_results
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        items = data.get("items", [])
        
        output = f"Found {len(items)} products:\n\n"
        for i, item in enumerate(items, 1):
            output += f"{i}. {item['title']}\n"
            output += f"   {item['snippet']}\n"
            output += f"   URL: {item['link']}\n\n"
        
        return output
    
    except Exception as e:
        return f"Error: {str(e)}"
```

### Tool 2: Get Product Details

```python
from bs4 import BeautifulSoup

def get_product_details(url: str) -> str:
    """
    Extract detailed information from a product page.
    Returns specs, description, and reviews summary.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text (simplified)
        # In production, use specific selectors for each site
        text = soup.get_text(separator='\n', strip=True)
        
        # Limit length
        max_length = 2000
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return f"Product details from {url}:\n\n{text}"
    
    except Exception as e:
        return f"Error fetching product details: {str(e)}"
```

### Tool 3: Compare Products

```python
def compare_products(product_list: str) -> str:
    """
    Compare multiple products based on provided information.
    Input: Comma-separated product names or descriptions.
    Returns: Comparison table.
    """
    try:
        products = [p.strip() for p in product_list.split(',')]
        
        if len(products) < 2:
            return "Error: Need at least 2 products to compare"
        
        output = "Product Comparison:\n\n"
        output += "To compare these products effectively, I need their details.\n"
        output += "Please use get_product_details for each product first.\n\n"
        output += f"Products to compare: {', '.join(products)}"
        
        return output
    
    except Exception as e:
        return f"Error comparing products: {str(e)}"
```

### Tool 4: Get Reviews Summary

```python
def get_reviews_summary(product_name: str) -> str:
    """
    Get a summary of customer reviews for a product.
    Returns common pros, cons, and overall sentiment.
    """
    try:
        # Mock implementation
        # In production, scrape from Amazon, Reddit, review sites
        
        reviews = {
            "overall_rating": "4.3/5.0",
            "total_reviews": 1247,
            "pros": [
                "Excellent build quality",
                "Great performance",
                "Good value for money"
            ],
            "cons": [
                "Battery life could be better",
                "Slightly heavy",
                "Limited color options"
            ],
            "common_themes": [
                "Users love the performance",
                "Some complaints about weight",
                "Generally recommended"
            ]
        }
        
        output = f"Reviews Summary for {product_name}:\n\n"
        output += f"Overall Rating: {reviews['overall_rating']} ({reviews['total_reviews']} reviews)\n\n"
        output += "Pros:\n"
        for pro in reviews['pros']:
            output += f"  ✓ {pro}\n"
        output += "\nCons:\n"
        for con in reviews['cons']:
            output += f"  ✗ {con}\n"
        output += "\nCommon Themes:\n"
        for theme in reviews['common_themes']:
            output += f"  • {theme}\n"
        
        return output
    
    except Exception as e:
        return f"Error getting reviews: {str(e)}"
```

### Tool 5: Price History

```python
def get_price_history(product_name: str) -> str:
    """
    Get price history and trends for a product.
    Helps determine if current price is good.
    """
    try:
        # Mock implementation
        # In production, use CamelCamelCamel API, Keepa, etc.
        
        history = {
            "current_price": "$899",
            "lowest_price": "$799 (3 months ago)",
            "highest_price": "$999 (6 months ago)",
            "average_price": "$879",
            "trend": "stable",
            "recommendation": "Current price is close to average. Good time to buy."
        }
        
        output = f"Price History for {product_name}:\n\n"
        output += f"Current Price: {history['current_price']}\n"
        output += f"Lowest Price: {history['lowest_price']}\n"
        output += f"Highest Price: {history['highest_price']}\n"
        output += f"Average Price: {history['average_price']}\n"
        output += f"Trend: {history['trend']}\n\n"
        output += f"💡 {history['recommendation']}"
        
        return output
    
    except Exception as e:
        return f"Error getting price history: {str(e)}"
```

## Build the Agent

### Tool Registry

```python
# agent.py
from tools import (
    search_products,
    get_product_details,
    compare_products,
    get_reviews_summary,
    get_price_history
)

class ShoppingAgent:
    """Shopping Research Assistant Agent"""
    
    def __init__(self):
        self.tools = self._create_tool_schemas()
        self.client = openai.OpenAI()
    
    def _create_tool_schemas(self):
        """Define tool schemas for OpenAI function calling"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_products",
                    "description": "Search for products matching a query. Use when user asks to find or search for products.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Product search query (e.g., 'laptop under $1000')"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_product_details",
                    "description": "Get detailed information about a specific product from its URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "Product page URL"
                            }
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_reviews_summary",
                    "description": "Get summary of customer reviews including pros, cons, and ratings.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_name": {
                                "type": "string",
                                "description": "Product name"
                            }
                        },
                        "required": ["product_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_price_history",
                    "description": "Get price history and determine if current price is good.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_name": {
                                "type": "string",
                                "description": "Product name"
                            }
                        },
                        "required": ["product_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_products",
                    "description": "Compare multiple products. Use after gathering details about each product.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_list": {
                                "type": "string",
                                "description": "Comma-separated list of product names"
                            }
                        },
                        "required": ["product_list"]
                    }
                }
            }
        ]
    
    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return result"""
        tool_map = {
            "search_products": search_products,
            "get_product_details": get_product_details,
            "compare_products": compare_products,
            "get_reviews_summary": get_reviews_summary,
            "get_price_history": get_price_history
        }
        
        if tool_name not in tool_map:
            return f"Error: Unknown tool {tool_name}"
        
        try:
            result = tool_map[tool_name](**arguments)
            return result
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
    def run(self, user_query: str, max_steps: int = 15) -> str:
        """Run the shopping assistant agent"""
        messages = [
            {
                "role": "system",
                "content": """You are a helpful shopping research assistant. 
                
Your goal is to help users make informed purchasing decisions by:
1. Searching for relevant products
2. Gathering detailed information and reviews
3. Comparing options
4. Providing clear recommendations with reasoning

Always:
- Search for products before making recommendations
- Check reviews and ratings
- Consider price history when available
- Compare multiple options when relevant
- Cite specific information from your research
- Be honest about limitations

Format your final recommendation clearly with pros, cons, and reasoning."""
            },
            {"role": "user", "content": user_query}
        ]
        
        print(f"🛍️  User: {user_query}\n")
        
        for step in range(max_steps):
            # Get LLM response
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.7
            )
            
            message = response.choices[0].message
            
            # If no tool calls, we're done
            if not message.tool_calls:
                print(f"🤖 Assistant: {message.content}\n")
                return message.content
            
            # Add assistant message
            messages.append(message)
            
            # Execute tool calls
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                print(f"🔧 Using tool: {function_name}({arguments})")
                
                # Execute tool
                result = self._execute_tool(function_name, arguments)
                print(f"📊 Result: {result[:200]}...\n")
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        
        return "⚠️  Max steps reached without completing the task"
```

## Complete Implementation

```python
# agent.py (complete file)
import openai
import json
from config import OPENAI_API_KEY, MODEL
from tools import (
    search_products,
    get_product_details,
    compare_products,
    get_reviews_summary,
    get_price_history
)

openai.api_key = OPENAI_API_KEY

# [ShoppingAgent class from above]

def main():
    """Test the shopping agent"""
    agent = ShoppingAgent()
    
    # Example queries
    queries = [
        "Find the best noise-canceling headphones under $300",
        "Compare iPhone 15 Pro and Samsung Galaxy S24",
        "What's a good coffee maker for home use?"
    ]
    
    for query in queries:
        print("=" * 60)
        result = agent.run(query)
        print("=" * 60)
        print()

if __name__ == "__main__":
    main()
```

## Test Cases

```python
# test_agent.py
from agent import ShoppingAgent

def test_product_search():
    """Test basic product search"""
    agent = ShoppingAgent()
    result = agent.run("Find wireless keyboards under $50")
    assert "Product" in result or "keyboard" in result.lower()
    print("✓ Product search test passed")

def test_comparison():
    """Test product comparison"""
    agent = ShoppingAgent()
    result = agent.run("Compare MacBook Air vs Dell XPS 13")
    assert len(result) > 100  # Should have substantial response
    print("✓ Comparison test passed")

def test_reviews():
    """Test review gathering"""
    agent = ShoppingAgent()
    result = agent.run("What do people say about AirPods Pro?")
    assert "review" in result.lower() or "rating" in result.lower()
    print("✓ Reviews test passed")

if __name__ == "__main__":
    test_product_search()
    test_comparison()
    test_reviews()
    print("\n✅ All tests passed!")
```

## Debug Common Issues

### Issue 1: Agent Doesn't Use Tools

**Problem**: Agent responds without searching

**Solution**: Strengthen system prompt
```python
"You MUST use the search_products tool before making any recommendations.
Never rely on prior knowledge about products or prices."
```

### Issue 2: Infinite Search Loop

**Problem**: Agent keeps searching without concluding

**Solution**: Add step tracking and guidance
```python
# Track tool usage
tool_usage = {}
if tool_name in tool_usage:
    tool_usage[tool_name] += 1
    if tool_usage[tool_name] > 3:
        return "You've used this tool multiple times. Please synthesize your findings."
```

### Issue 3: Hallucinated Product Info

**Problem**: Agent invents product details

**Solution**: Emphasize tool-only information
```python
"CRITICAL: Only use information from tool results. 
If a tool doesn't return information, say so explicitly.
Never make up product names, prices, or specifications."
```

### Issue 4: Poor Recommendations

**Problem**: Recommendations lack depth

**Solution**: Add structured output requirement
```python
"Format your final recommendation as:

**Recommendation**: [Product name]

**Why**: [2-3 key reasons]

**Pros**:
- [Pro 1]
- [Pro 2]

**Cons**:
- [Con 1]
- [Con 2]

**Price**: [Current price and value assessment]"
```

## Enhancements

### 1. Add Budget Tracking
```python
def check_budget(price: str, budget: float) -> bool:
    """Check if price is within budget"""
    # Extract numeric price
    price_num = float(price.replace('$', '').replace(',', ''))
    return price_num <= budget
```

### 2. Save Research Sessions
```python
def save_research(query: str, results: str):
    """Save research for later reference"""
    with open(f"research_{timestamp}.txt", "w") as f:
        f.write(f"Query: {query}\n\n{results}")
```

### 3. Multi-Store Price Comparison
```python
def compare_prices_across_stores(product: str) -> dict:
    """Check prices at Amazon, Walmart, Best Buy, etc."""
    stores = ["Amazon", "Walmart", "Best Buy"]
    prices = {}
    for store in stores:
        prices[store] = search_store_price(store, product)
    return prices
```

### 4. Deal Alerts
```python
def check_for_deals(product: str) -> str:
    """Check if product is on sale or has coupons"""
    # Check deal sites, coupon codes, etc.
    pass
```

### 5. Personalization
```python
def get_user_preferences() -> dict:
    """Load user preferences (brands, price range, features)"""
    return {
        "preferred_brands": ["Sony", "Apple"],
        "max_price": 500,
        "must_have_features": ["wireless", "noise-canceling"]
    }
```

## Next Steps

Congratulations! You've built a complete shopping research assistant. You now understand:
- ✅ ReAct pattern implementation
- ✅ Tool integration and validation
- ✅ Multi-step reasoning
- ✅ Error handling and debugging
- ✅ Real-world agent applications

In Chapter 3, we'll explore advanced agent patterns including planning, memory systems, and multi-agent collaboration!
