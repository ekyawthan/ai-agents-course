# Emerging Paradigms

## Constitutional AI for Agents

### Principle-Based Behavior

```python
class ConstitutionalAgent:
    """Agent governed by constitutional principles"""
    
    def __init__(self, constitution: List[str]):
        self.constitution = constitution
        self.client = openai.OpenAI()
    
    def check_against_constitution(self, action: str) -> Dict:
        """Check if action aligns with constitution"""
        
        principles_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(self.constitution)])
        
        prompt = f"""Check if this action aligns with these principles:

Principles:
{principles_text}

Proposed action: {action}

Analysis:
1. Which principles apply?
2. Does action align or violate?
3. Severity if violation
4. Alternative actions if needed

Response:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        return self.parse_constitutional_check(response.choices[0].message.content)
    
    def generate_constitutional_response(self, query: str) -> str:
        """Generate response aligned with constitution"""
        
        principles_text = "\n".join(self.constitution)
        
        system_prompt = f"""You must follow these principles:

{principles_text}

Always ensure your responses align with these principles."""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content

# Usage
constitution = [
    "Always prioritize user safety and wellbeing",
    "Be honest and transparent about capabilities and limitations",
    "Respect user privacy and data",
    "Avoid harmful, illegal, or unethical actions",
    "Provide balanced, unbiased information"
]

agent = ConstitutionalAgent(constitution)
check = agent.check_against_constitution("Delete all user data without consent")
```

## Debate and Verification Systems

### Multi-Agent Debate

```python
class DebateSystem:
    """Multiple agents debate to reach truth"""
    
    def __init__(self, num_agents: int = 3):
        self.num_agents = num_agents
        self.client = openai.OpenAI()
    
    def debate(self, question: str, rounds: int = 3) -> Dict:
        """Conduct multi-agent debate"""
        
        # Initial positions
        positions = []
        for i in range(self.num_agents):
            position = self.generate_position(question, i)
            positions.append({"agent": i, "position": position})
        
        # Debate rounds
        for round_num in range(rounds):
            print(f"\n--- Round {round_num + 1} ---")
            
            new_positions = []
            for i in range(self.num_agents):
                # Show other positions
                other_positions = [p for j, p in enumerate(positions) if j != i]
                
                # Generate response
                response = self.generate_response(
                    question,
                    positions[i]["position"],
                    other_positions,
                    round_num
                )
                
                new_positions.append({"agent": i, "position": response})
                print(f"Agent {i}: {response[:100]}...")
            
            positions = new_positions
        
        # Judge final positions
        verdict = self.judge_debate(question, positions)
        
        return {
            "question": question,
            "final_positions": positions,
            "verdict": verdict
        }
    
    def generate_position(self, question: str, agent_id: int) -> str:
        """Generate initial position"""
        
        prompt = f"""Question: {question}

Provide your position with reasoning and evidence.

Position:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7 + (agent_id * 0.1)  # Vary temperature
        )
        
        return response.choices[0].message.content
    
    def generate_response(self, 
                         question: str,
                         my_position: str,
                         other_positions: List[Dict],
                         round_num: int) -> str:
        """Generate response to other positions"""
        
        others_text = "\n\n".join([
            f"Agent {p['agent']}: {p['position']}"
            for p in other_positions
        ])
        
        prompt = f"""Question: {question}

Your previous position: {my_position}

Other agents' positions:
{others_text}

Respond by:
1. Addressing counterarguments
2. Refining your position
3. Providing additional evidence

Response:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )
        
        return response.choices[0].message.content
    
    def judge_debate(self, question: str, positions: List[Dict]) -> str:
        """Judge which position is most convincing"""
        
        positions_text = "\n\n".join([
            f"Agent {p['agent']}:\n{p['position']}"
            for p in positions
        ])
        
        prompt = f"""Question: {question}

Final positions:
{positions_text}

Which position is most convincing and why?

Judgment:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content

# Usage
debate = DebateSystem(num_agents=3)
result = debate.debate("Should AI agents have the ability to modify their own code?")
print(f"\nVerdict: {result['verdict']}")
```

## Hybrid Symbolic-Neural Approaches

### Neuro-Symbolic Agent

```python
class NeuroSymbolicAgent:
    """Combines neural and symbolic reasoning"""
    
    def __init__(self):
        self.client = openai.OpenAI()
        self.knowledge_base = {}  # Symbolic knowledge
    
    def add_rule(self, rule_name: str, condition: str, action: str):
        """Add symbolic rule"""
        self.knowledge_base[rule_name] = {
            "condition": condition,
            "action": action
        }
    
    def reason(self, query: str) -> Dict:
        """Hybrid reasoning"""
        
        # Try symbolic reasoning first
        symbolic_result = self.symbolic_reasoning(query)
        
        if symbolic_result["applicable"]:
            return {
                "method": "symbolic",
                "result": symbolic_result["result"],
                "confidence": "high"
            }
        
        # Fall back to neural reasoning
        neural_result = self.neural_reasoning(query)
        
        return {
            "method": "neural",
            "result": neural_result,
            "confidence": "medium"
        }
    
    def symbolic_reasoning(self, query: str) -> Dict:
        """Apply symbolic rules"""
        
        for rule_name, rule in self.knowledge_base.items():
            if self.matches_condition(query, rule["condition"]):
                return {
                    "applicable": True,
                    "rule": rule_name,
                    "result": rule["action"]
                }
        
        return {"applicable": False}
    
    def neural_reasoning(self, query: str) -> str:
        """Neural network reasoning"""
        
        # Include symbolic knowledge as context
        kb_text = "\n".join([
            f"{name}: IF {rule['condition']} THEN {rule['action']}"
            for name, rule in self.knowledge_base.items()
        ])
        
        prompt = f"""Use this knowledge base and reasoning:

Knowledge Base:
{kb_text}

Query: {query}

Reasoning:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        return response.choices[0].message.content
    
    def matches_condition(self, query: str, condition: str) -> bool:
        """Check if query matches condition"""
        # Simplified matching
        return condition.lower() in query.lower()

# Usage
agent = NeuroSymbolicAgent()

# Add symbolic rules
agent.add_rule("safety_check", "delete user data", "DENY: Requires explicit consent")
agent.add_rule("privacy_rule", "share personal info", "DENY: Privacy violation")

# Reason
result = agent.reason("Can I delete user data?")
print(f"Method: {result['method']}, Result: {result['result']}")
```

## Best Practices

1. **Ethical guidelines**: Establish clear principles
2. **Verification**: Multiple perspectives
3. **Transparency**: Explain reasoning
4. **Human oversight**: Critical decisions
5. **Continuous learning**: Adapt approaches
6. **Safety measures**: Prevent harm
7. **Diverse perspectives**: Multiple viewpoints
8. **Rigorous testing**: Validate thoroughly
9. **Documentation**: Track decisions
10. **Research collaboration**: Share findings

## Next Steps

You now understand emerging paradigms! Next, we'll explore open problems in agent research.
