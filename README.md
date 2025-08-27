# 🤖 AI Agents for Enhanced Chatbots

**Three intelligent AI agents that work together to create smarter, safer, and more efficient chatbots.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-orange.svg)](https://openai.com)

## 🎯 Overview

This project provides three specialized AI agents that can be integrated into any chatbot system to enhance its capabilities:

1. **🛡️ Security Agent** - Detects malicious content and security threats
2. **🧠 Context Agent** - Analyzes query relevance and conversation flow
3. **🎯 Model Selection Agent** - Intelligently selects optimal LLM models

## ✨ Key Features

- **🔒 Enhanced Security**: Advanced threat detection using GPT-4o
- **🧠 Smart Context**: Intelligent query relevance analysis
- **💰 Cost Optimization**: Automatic model selection for cost-performance balance
- **🚀 Easy Integration**: Simple API for any chatbot platform
- **⚙️ Configurable**: Customizable thresholds and preferences
- **📊 Analytics**: Detailed analysis and performance metrics

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-agents-chatbot.git
cd ai-agents-chatbot

# Create virtual environment
python -m venv venv 
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Add your OpenAI API key
echo "OPENAI_API_KEY=your_api_key_here" >> .env
```

### 3. Run Demo

```bash
# Start the demo Streamlit chatbot
streamlit run examples/streamlit_demo.py
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Security      │    │   Context       │    │   Model         │
│   Agent         │    │   Agent         │    │   Selection     │
│                 │    │                 │    │   Agent         │
│ • Threat        │    │ • Relevance     │    │ • Query        │
│   Detection     │    │   Analysis      │    │   Analysis     │
│ • Content       │    │ • Domain        │    │ • Model        │
│   Safety        │    │   Detection     │    │   Ranking      │
│ • Malicious     │    │ • Flow          │    │ • Cost         │
│   Prompt        │    │   Analysis      │    │   Optimization │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Chatbot       │
                    │   Orchestrator  │
                    │                 │
                    │ • Coordinates   │
                    │   agents        │
                    │ • Manages flow  │
                    │ • Generates     │
                    │   responses     │
                    └─────────────────┘
```

## 🔧 Integration Guide

### Basic Integration

```python
from ai_agents import SecurityAgent, ContextAgent, ModelSelectionAgent

# Initialize agents
security_agent = SecurityAgent(
    model="gpt-4o",
    threat_threshold=0.7
)

context_agent = ContextAgent(
    chatbot_name="My Bot",
    chatbot_description="A helpful assistant",
    keywords=["help", "assist", "support"]
)

model_agent = ModelSelectionAgent(
    cost_sensitivity="medium",
    performance_preference="balanced"
)

# Process user query
def handle_query(user_query, conversation_history):
    # Security check
    security_result = security_agent.analyze_security(user_query)
    if security_result['blocked']:
        return "I cannot process that request for security reasons."
    
    # Context analysis
    context_result = context_agent.analyze_context(user_query, conversation_history)
    
    # Model selection
    model_result = model_agent.select_model(user_query, conversation_history)
    
    # Generate response using selected model
    response = generate_response(user_query, model_result['selected_model'])
    
    return response
```

### Advanced Configuration

```python
# Custom security thresholds
security_agent = SecurityAgent(
    model="gpt-4o",
    threat_threshold=0.5,  # More strict
    enable_detailed_analysis=True
)

# Domain-specific context
context_agent = ContextAgent(
    chatbot_name="Health Assistant",
    chatbot_description="Medical information and health advice",
    keywords=["health", "medical", "symptoms", "treatment"],
    chatbot_prompt="You are a medical AI assistant..."
)

# Cost-optimized model selection
model_agent = ModelSelectionAgent(
    cost_sensitivity="high",  # Prefer cheaper models
    performance_preference="speed"  # Prioritize speed over quality
)
```

## 📚 API Reference

### Security Agent

```python
class SecurityAgent:
    def analyze_security(self, user_query: str, 
                        conversation_context: str = None,
                        user_profile: Dict = None) -> Dict:
        """
        Analyze user query for security threats
        
        Returns:
            {
                'is_malicious': bool,
                'threat_level': str,  # 'safe', 'low', 'medium', 'high', 'critical'
                'threat_score': float,  # 0.0-1.0
                'confidence_score': float,
                'blocked': bool,
                'warnings': List[str],
                'llm_analysis': Dict
            }
        """
```

### Context Agent

```python
class ContextAgent:
    def analyze_context(self, user_query: str, 
                       conversation_history: List[Dict] = None,
                       user_profile: Dict = None) -> Dict:
        """
        Analyze query relevance and context
        
        Returns:
            {
                'is_contextual': bool,
                'relevance_score': float,  # 0.0-1.0
                'relevance_level': str,  # 'irrelevant', 'low', 'medium', 'high'
                'reasoning': str,
                'context_shift': bool,
                'domain_alignment': float,
                'chatbot_context': Dict
            }
        """
```

### Model Selection Agent

```python
class ModelSelectionAgent:
    def select_model(self, user_query: str, 
                    conversation_context: str = None,
                    user_preferences: Dict = None) -> Dict:
        """
        Select optimal LLM model for query
        
        Returns:
            {
                'selected_model': str,
                'model_info': Dict,
                'selection_reasoning': str,
                'confidence_score': float,
                'estimated_cost': float,
                'estimated_tokens': int,
                'query_analysis': Dict
            }
        """
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific agent tests
python tests/test_security_agent.py
python tests/test_context_agent.py
python tests/test_model_selection_agent.py

# Run integration tests
python tests/test_integration.py
```

## 📊 Performance

- **Security Analysis**: ~2-4 seconds (GPT-4o)
- **Context Analysis**: ~1-2 seconds (GPT-3.5-turbo)
- **Model Selection**: ~0.5-1 second (GPT-3.5-turbo)
- **Total Overhead**: ~3-7 seconds per query
- **Cost**: ~$0.01-0.05 per query (depending on models used)

## 🔒 Security Features

- **Threat Detection**: 20+ threat categories
- **Prompt Injection**: Advanced jailbreak detection
- **Content Safety**: Comprehensive safety analysis
- **Fallback Protection**: Conservative blocking on errors
- **Configurable Thresholds**: Adjustable sensitivity levels

## 🌟 Use Cases

### Healthcare Chatbots
- **Security**: Detect medical misinformation and harmful advice
- **Context**: Ensure queries are health-related
- **Model Selection**: Use high-quality models for medical queries

### E-commerce Assistants
- **Security**: Prevent fraud and malicious requests
- **Context**: Identify shopping-related queries
- **Model Selection**: Balance cost and response quality

### Educational Bots
- **Security**: Filter inappropriate content
- **Context**: Maintain educational focus
- **Model Selection**: Optimize for learning outcomes

### Customer Service
- **Security**: Protect against abuse and spam
- **Context**: Route queries to appropriate departments
- **Model Selection**: Ensure consistent service quality

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 ai_agents/ tests/
black ai_agents/ tests/

# Run tests with coverage
pytest --cov=ai_agents tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing the GPT models
- Streamlit for the demo interface
- The open source community for inspiration and feedback

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ai-agents-chatbot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-agents-chatbot/discussions)
- **Documentation**: [Full Documentation](docs/)

## 🚀 Roadmap

- [ ] **Multi-provider Support**: Google Gemini, Anthropic Claude
- [ ] **Custom Model Training**: Fine-tune agents for specific domains
- [ ] **Real-time Learning**: Improve agents based on user feedback
- [ ] **Advanced Analytics**: Detailed performance insights
- [ ] **API Service**: Hosted service for easy integration
- [ ] **Mobile SDK**: Native mobile app support

---

**Ready to make your chatbot smarter?** 🚀

Start with our [Quick Start Guide](docs/QUICKSTART.md) or explore the [examples](examples/) directory!

Made with ❤️ by the AI Agents Team
