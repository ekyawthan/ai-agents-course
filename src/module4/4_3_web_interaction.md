# Web Interaction

## Browser Automation

Agents can interact with websites like humans do—clicking, typing, scrolling, and extracting information.

### Why Browser Automation?

- Access dynamic content (JavaScript-rendered)
- Interact with web applications
- Fill forms and submit data
- Navigate multi-page workflows
- Handle authentication

### Playwright Basics

```python
from playwright.sync_api import sync_playwright
from typing import Optional

class BrowserAgent:
    """Agent with browser automation capabilities"""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.page = None
    
    def start(self):
        """Start browser"""
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.page = self.browser.new_page()
    
    def stop(self):
        """Stop browser"""
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
    
    def navigate(self, url: str) -> dict:
        """Navigate to URL"""
        try:
            self.page.goto(url, wait_until="networkidle")
            return {
                "success": True,
                "url": self.page.url,
                "title": self.page.title()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def click(self, selector: str) -> dict:
        """Click element"""
        try:
            self.page.click(selector)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def type_text(self, selector: str, text: str) -> dict:
        """Type text into element"""
        try:
            self.page.fill(selector, text)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_text(self, selector: str) -> Optional[str]:
        """Get text from element"""
        try:
            return self.page.text_content(selector)
        except:
            return None
    
    def screenshot(self, path: str = "screenshot.png") -> dict:
        """Take screenshot"""
        try:
            self.page.screenshot(path=path)
            return {"success": True, "path": path}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_page_content(self) -> str:
        """Get full page HTML"""
        return self.page.content()

# Usage
agent = BrowserAgent()
agent.start()

# Navigate
agent.navigate("https://example.com")

# Interact
agent.type_text("#search", "AI agents")
agent.click("button[type='submit']")

# Extract
results = agent.get_text(".results")

agent.stop()
```

### Selenium Alternative

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class SeleniumAgent:
    """Browser automation with Selenium"""
    
    def __init__(self):
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 10)
    
    def navigate(self, url: str):
        """Navigate to URL"""
        self.driver.get(url)
    
    def click(self, selector: str, by: By = By.CSS_SELECTOR):
        """Click element"""
        element = self.wait.until(
            EC.element_to_be_clickable((by, selector))
        )
        element.click()
    
    def type_text(self, selector: str, text: str, by: By = By.CSS_SELECTOR):
        """Type text"""
        element = self.wait.until(
            EC.presence_of_element_located((by, selector))
        )
        element.clear()
        element.send_keys(text)
    
    def get_text(self, selector: str, by: By = By.CSS_SELECTOR) -> str:
        """Get element text"""
        element = self.wait.until(
            EC.presence_of_element_located((by, selector))
        )
        return element.text
    
    def close(self):
        """Close browser"""
        self.driver.quit()
```

## Web Scraping

Extract structured data from websites.

### BeautifulSoup Scraping

```python
import requests
from bs4 import BeautifulSoup
from typing import List, Dict

class WebScraper:
    """Web scraping agent"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse page"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def extract_links(self, url: str) -> List[str]:
        """Extract all links from page"""
        soup = self.fetch_page(url)
        if not soup:
            return []
        
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            # Convert relative to absolute
            if href.startswith('/'):
                from urllib.parse import urljoin
                href = urljoin(url, href)
            links.append(href)
        
        return links
    
    def extract_text(self, url: str, selector: Optional[str] = None) -> str:
        """Extract text from page"""
        soup = self.fetch_page(url)
        if not soup:
            return ""
        
        if selector:
            element = soup.select_one(selector)
            return element.get_text(strip=True) if element else ""
        else:
            return soup.get_text(separator='\n', strip=True)
    
    def extract_structured_data(self, url: str, schema: dict) -> List[Dict]:
        """Extract structured data based on schema"""
        soup = self.fetch_page(url)
        if not soup:
            return []
        
        results = []
        
        # Find all items matching container selector
        items = soup.select(schema['container'])
        
        for item in items:
            data = {}
            for field, selector in schema['fields'].items():
                element = item.select_one(selector)
                if element:
                    data[field] = element.get_text(strip=True)
            
            if data:
                results.append(data)
        
        return results

# Usage
scraper = WebScraper()

# Extract structured data
schema = {
    'container': '.product',
    'fields': {
        'name': '.product-name',
        'price': '.product-price',
        'rating': '.product-rating'
    }
}

products = scraper.extract_structured_data('https://example.com/products', schema)
```

### Handling Dynamic Content

```python
class DynamicScraper:
    """Scrape JavaScript-rendered content"""
    
    def __init__(self):
        self.browser = BrowserAgent()
        self.browser.start()
    
    def scrape_dynamic(self, url: str, wait_selector: str = None) -> str:
        """Scrape page with JavaScript"""
        self.browser.navigate(url)
        
        # Wait for content to load
        if wait_selector:
            self.browser.page.wait_for_selector(wait_selector)
        else:
            self.browser.page.wait_for_load_state("networkidle")
        
        # Get rendered HTML
        return self.browser.get_page_content()
    
    def scrape_infinite_scroll(self, url: str, max_scrolls: int = 10) -> str:
        """Scrape infinite scroll pages"""
        self.browser.navigate(url)
        
        for _ in range(max_scrolls):
            # Scroll to bottom
            self.browser.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            
            # Wait for new content
            self.browser.page.wait_for_timeout(1000)
        
        return self.browser.get_page_content()
    
    def close(self):
        """Close browser"""
        self.browser.stop()
```

## Form Filling and Navigation

### Automated Form Submission

```python
class FormAgent:
    """Agent that can fill and submit forms"""
    
    def __init__(self):
        self.browser = BrowserAgent()
        self.browser.start()
    
    def fill_form(self, url: str, form_data: dict) -> dict:
        """Fill and submit form"""
        try:
            # Navigate to page
            self.browser.navigate(url)
            
            # Fill fields
            for selector, value in form_data.items():
                if isinstance(value, str):
                    self.browser.type_text(selector, value)
                elif value.get('type') == 'click':
                    self.browser.click(selector)
                elif value.get('type') == 'select':
                    self.browser.page.select_option(selector, value['value'])
            
            # Submit form
            submit_button = form_data.get('submit_button', 'button[type="submit"]')
            self.browser.click(submit_button)
            
            # Wait for response
            self.browser.page.wait_for_load_state("networkidle")
            
            return {
                "success": True,
                "url": self.browser.page.url,
                "title": self.browser.page.title()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def close(self):
        """Close browser"""
        self.browser.stop()

# Usage
agent = FormAgent()

form_data = {
    '#name': 'John Doe',
    '#email': 'john@example.com',
    '#message': 'Hello from agent!',
    'submit_button': '#submit-btn'
}

result = agent.fill_form('https://example.com/contact', form_data)
agent.close()
```

### Multi-Step Navigation

```python
class NavigationAgent:
    """Agent for multi-step web workflows"""
    
    def __init__(self):
        self.browser = BrowserAgent()
        self.browser.start()
        self.history = []
    
    def execute_workflow(self, steps: List[dict]) -> dict:
        """Execute multi-step workflow"""
        results = []
        
        for i, step in enumerate(steps):
            print(f"Step {i+1}: {step['action']}")
            
            try:
                if step['action'] == 'navigate':
                    result = self.browser.navigate(step['url'])
                
                elif step['action'] == 'click':
                    result = self.browser.click(step['selector'])
                
                elif step['action'] == 'type':
                    result = self.browser.type_text(step['selector'], step['text'])
                
                elif step['action'] == 'wait':
                    self.browser.page.wait_for_timeout(step['duration'])
                    result = {"success": True}
                
                elif step['action'] == 'extract':
                    text = self.browser.get_text(step['selector'])
                    result = {"success": True, "data": text}
                
                elif step['action'] == 'screenshot':
                    result = self.browser.screenshot(step.get('path', f'step_{i}.png'))
                
                else:
                    result = {"success": False, "error": "Unknown action"}
                
                results.append({
                    "step": i + 1,
                    "action": step['action'],
                    "result": result
                })
                
                self.history.append({
                    "url": self.browser.page.url,
                    "title": self.browser.page.title()
                })
                
                if not result.get('success', False):
                    break
                    
            except Exception as e:
                results.append({
                    "step": i + 1,
                    "action": step['action'],
                    "result": {"success": False, "error": str(e)}
                })
                break
        
        return {
            "completed": len(results),
            "total": len(steps),
            "results": results,
            "history": self.history
        }
    
    def close(self):
        """Close browser"""
        self.browser.stop()

# Usage
agent = NavigationAgent()

workflow = [
    {"action": "navigate", "url": "https://example.com"},
    {"action": "click", "selector": "#login-btn"},
    {"action": "type", "selector": "#username", "text": "user@example.com"},
    {"action": "type", "selector": "#password", "text": "password123"},
    {"action": "click", "selector": "#submit"},
    {"action": "wait", "duration": 2000},
    {"action": "extract", "selector": ".welcome-message"},
    {"action": "screenshot", "path": "logged-in.png"}
]

result = agent.execute_workflow(workflow)
agent.close()
```

## Screenshot and Visual Understanding

### Taking Screenshots

```python
class ScreenshotAgent:
    """Agent for visual capture and analysis"""
    
    def __init__(self):
        self.browser = BrowserAgent()
        self.browser.start()
        self.client = openai.OpenAI()
    
    def capture_and_analyze(self, url: str, question: str) -> dict:
        """Capture screenshot and analyze with vision model"""
        # Navigate and capture
        self.browser.navigate(url)
        screenshot_path = "temp_screenshot.png"
        self.browser.screenshot(screenshot_path)
        
        # Analyze with vision model
        import base64
        with open(screenshot_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        return {
            "url": url,
            "question": question,
            "analysis": response.choices[0].message.content,
            "screenshot": screenshot_path
        }
    
    def compare_pages(self, url1: str, url2: str) -> dict:
        """Compare two pages visually"""
        # Capture both
        self.browser.navigate(url1)
        self.browser.screenshot("page1.png")
        
        self.browser.navigate(url2)
        self.browser.screenshot("page2.png")
        
        # Compare with vision model
        question = "What are the main differences between these two pages?"
        
        # Would need to send both images to vision model
        # Implementation depends on specific vision API
        
        return {
            "url1": url1,
            "url2": url2,
            "screenshot1": "page1.png",
            "screenshot2": "page2.png"
        }
    
    def close(self):
        """Close browser"""
        self.browser.stop()
```

### Element Detection

```python
class ElementDetector:
    """Detect and locate elements on page"""
    
    def __init__(self):
        self.browser = BrowserAgent()
        self.browser.start()
    
    def find_element_by_description(self, url: str, description: str) -> Optional[str]:
        """Find element selector by natural language description"""
        self.browser.navigate(url)
        
        # Get page structure
        elements = self.browser.page.evaluate("""
            () => {
                const elements = [];
                document.querySelectorAll('button, a, input, select, textarea').forEach(el => {
                    elements.push({
                        tag: el.tagName,
                        text: el.textContent.trim(),
                        id: el.id,
                        class: el.className,
                        type: el.type
                    });
                });
                return elements;
            }
        """)
        
        # Use LLM to match description to element
        prompt = f"""Find the element matching this description: {description}

Available elements:
{json.dumps(elements, indent=2)}

Return the best CSS selector to target this element:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content.strip()
    
    def close(self):
        """Close browser"""
        self.browser.stop()
```

## Complete Web Interaction Agent

```python
class WebAgent:
    """Complete web interaction agent"""
    
    def __init__(self):
        self.browser = BrowserAgent()
        self.browser.start()
        self.scraper = WebScraper()
        self.client = openai.OpenAI()
    
    def execute_task(self, task: str, url: str) -> str:
        """Execute web task from natural language"""
        # Generate action plan
        plan = self.generate_plan(task, url)
        
        # Execute plan
        results = []
        for step in plan:
            result = self.execute_step(step)
            results.append(result)
        
        # Summarize results
        return self.summarize_results(task, results)
    
    def generate_plan(self, task: str, url: str) -> List[dict]:
        """Generate action plan for task"""
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """Generate a step-by-step plan for web automation.

Available actions:
- navigate: Go to URL
- click: Click element (provide selector)
- type: Type text (provide selector and text)
- extract: Extract text (provide selector)
- wait: Wait for duration (milliseconds)
- screenshot: Take screenshot

Return JSON array of steps."""
                },
                {
                    "role": "user",
                    "content": f"Task: {task}\nStarting URL: {url}"
                }
            ],
            temperature=0.2
        )
        
        import json
        return json.loads(response.choices[0].message.content)
    
    def execute_step(self, step: dict) -> dict:
        """Execute single step"""
        action = step['action']
        
        try:
            if action == 'navigate':
                return self.browser.navigate(step['url'])
            elif action == 'click':
                return self.browser.click(step['selector'])
            elif action == 'type':
                return self.browser.type_text(step['selector'], step['text'])
            elif action == 'extract':
                text = self.browser.get_text(step['selector'])
                return {"success": True, "data": text}
            elif action == 'wait':
                self.browser.page.wait_for_timeout(step['duration'])
                return {"success": True}
            elif action == 'screenshot':
                return self.browser.screenshot(step.get('path', 'screenshot.png'))
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def summarize_results(self, task: str, results: List[dict]) -> str:
        """Summarize execution results"""
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": f"""Summarize the results of this web automation task:

Task: {task}

Results:
{json.dumps(results, indent=2)}

Provide a clear summary of what was accomplished:"""
                }
            ]
        )
        
        return response.choices[0].message.content
    
    def close(self):
        """Close browser"""
        self.browser.stop()

# Usage
agent = WebAgent()
result = agent.execute_task(
    "Search for 'AI agents' on the website and extract the top 3 results",
    "https://example.com"
)
print(result)
agent.close()
```

## Best Practices

1. **Respect robots.txt**: Check if scraping is allowed
2. **Rate limiting**: Don't overwhelm servers
3. **Use headless mode**: Faster and less resource-intensive
4. **Handle timeouts**: Set reasonable wait times
5. **Error recovery**: Retry failed operations
6. **Clean up resources**: Close browsers properly
7. **User agent**: Identify your bot appropriately
8. **Cache responses**: Avoid redundant requests
9. **Validate selectors**: Check elements exist before interacting
10. **Monitor performance**: Track execution time

## Common Pitfalls

### Pitfall 1: Stale Selectors
**Problem**: Element selectors change
**Solution**: Use more robust selectors (data attributes, ARIA labels)

### Pitfall 2: Race Conditions
**Problem**: Clicking before element is ready
**Solution**: Use explicit waits

### Pitfall 3: Memory Leaks
**Problem**: Not closing browsers
**Solution**: Always close in finally block or use context managers

### Pitfall 4: Detection
**Problem**: Website blocks automated access
**Solution**: Use stealth plugins, rotate user agents, add delays

## Next Steps

Chapter 4 (Agent Tools & Capabilities) is complete! You now understand code execution, data access, and web interaction. In Chapter 5, we'll explore production-ready agents including reliability, testing, and monitoring.
