# 🚀 AI Agents Package - Comprehensive Guide

**One guide to rule them all!** This comprehensive guide covers everything you need to know about our AI agents package.

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [Installation & Setup](#-installation--setup)
3. [Agent Types & Factory Functions](#-agent-types--factory-functions)
4. [RAG Capabilities](#-rag-capabilities)
5. [Multi-User & Multi-LLM Support](#-multi-user--multi-llm-support)
6. [Vector Database & ChromaDB](#-vector-database--chromadb)
7. [Advanced Features](#-advanced-features)
8. [Usage Examples](#-usage-examples)
9. [Troubleshooting](#-troubleshooting)

---

## 🚀 Quick Start

### **Basic Setup (30 seconds)**
```bash
# Clone and setup
git clone <your-repo>
cd model_classification
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install everything
pip install -r requirements.txt
pip install -e .

# Run showcase
python showcase_agent_capabilities.py
```

### **First Agent (No API key needed)**
```python
from ai_agents import create_basic_agent

agent = create_basic_agent()
response = agent.invoke("user_123", "Hello! What can you do?")
print(response)
```

---

## 🔧 Installation & Setup

### **Dependencies Overview**

#### **Core Dependencies** (Required)
```bash
pip install langchain langchain-community langchain-openai python-dotenv
```

#### **RAG Support** (Document Retrieval)
```bash
pip install sentence-transformers chromadb faiss-cpu
```

#### **Tool Integration** (Web Search, Wikipedia)
```bash
pip install duckduckgo-search ddgs wikipedia
```

#### **Full Installation** (All Features)
```bash
pip install -r requirements.txt
```

### **Environment Setup**
```bash
# Copy environment template
cp env.example .env

# Add your API keys
echo "OPENAI_API_KEY=your_api_key_here" >> .env
echo "ANTHROPIC_API_KEY=your_anthropic_key_here" >> .env
echo "GOOGLE_API_KEY=your_google_key_here" >> .env
```

---

## 🤖 Agent Types & Factory Functions

### **1. Basic Agent** (`create_basic_agent`)
```python
from ai_agents import create_basic_agent

# No API key needed - perfect for development
agent = create_basic_agent()

# Features:
# ✅ Conversation memory
# ✅ Basic RAG (if enabled)
# ✅ User session management
# ✅ Performance metrics
```

### **2. OpenAI Agent** (`create_openai_agent`)
```python
from ai_agents import create_openai_agent

# Basic OpenAI agent (chat only)
agent = create_openai_agent()

# With RAG for document context
agent = create_openai_agent(enable_rag=True)

# With tools for real-time information
agent = create_openai_agent(enable_tools=True)

# Full-featured agent
agent = create_openai_agent(
    enable_rag=True, 
    enable_tools=True, 
    enable_security=True
)
```

### **3. Enterprise Agent** (`create_enterprise_agent`)
```python
from ai_agents import create_enterprise_agent

# Standard enterprise agent (all features enabled)
agent = create_enterprise_agent()

# Enterprise agent without RAG (chat only)
agent = create_enterprise_agent(enable_rag=False)

# Enterprise agent with custom model
agent = create_enterprise_agent(model_name="gpt-4-turbo")
```

### **Feature Comparison Table**

| Feature | Basic | OpenAI | Enterprise |
|---------|-------|---------|------------|
| **LLM Provider** | Fake LLM | OpenAI | OpenAI |
| **Memory** | ✅ | ✅ | ✅ |
| **RAG** | Optional | Optional | ✅ Default |
| **Tools** | ❌ | Optional | ✅ Default |
| **Security** | ❌ | Optional | ✅ Default |
| **Metrics** | ✅ | ✅ | ✅ |
| **Persistence** | File | File | PostgreSQL |
| **Cost** | Free | API costs | API costs |

---

## 🔍 RAG Capabilities

### **What is RAG?**
**Retrieval Augmented Generation** allows the AI to retrieve relevant documents and use them as context when answering questions.

### **How RAG Works**
1. **User asks a question**
2. **System searches vector database** for relevant documents
3. **Relevant context is retrieved**
4. **Question + context is sent to LLM**
5. **LLM provides context-aware response**

### **RAG Benefits**
- ✅ **More accurate responses**
- ✅ **Up-to-date information**
- ✅ **Company-specific knowledge**
- ✅ **Reduced hallucinations**
- ✅ **Better user experience**

### **RAG Usage Examples**
```python
from ai_agents import create_openai_agent
from langchain_core.documents import Document

# Create RAG-enabled agent
agent = create_openai_agent(
    enable_rag=True,
    vector_store_path="./company_docs"  # Persistent storage
)

# Add documents
documents = [
    Document(page_content="Our company policy states..."),
    Document(page_content="The company's mission is...")
]
agent.add_documents_to_vector_store(documents)

# Query with RAG context
response = agent.invoke("user_123", "What are our company policies?")
print(response)
```

---

## 👥 Multi-User & Multi-LLM Support

### **Multi-User Conversation History**
```python
# Each user gets their own conversation history
agent.invoke("alice", "Hello, I'm Alice")
agent.invoke("bob", "Hello, I'm Bob")

# Get individual user history
alice_history = agent.get_conversation_history("alice")
bob_history = agent.get_conversation_history("bob")

# Clear specific user session
agent.clear_user_session("alice")  # Only clears Alice's data
```

### **Multi-LLM Provider Support**

#### **OpenAI Models**
```python
agent = create_openai_agent(model="gpt-4-turbo")
agent = create_openai_agent(model="gpt-3.5-turbo")
```

#### **Anthropic Models**
```python
agent = create_enterprise_agent(
    llm_provider="anthropic",
    model_name="claude-3-sonnet-20240229"
)
```

#### **Google Models**
```python
agent = create_enterprise_agent(
    llm_provider="google",
    model_name="gemini-pro"
```

#### **Local Models (Ollama)**
```python
agent = create_enterprise_agent(
    llm_provider="ollama",
    model_name="llama2"
)
```

### **Multi-User Analytics**
```python
# Get metrics for all users
all_metrics = agent.get_all_metrics()

for user_id, metrics in all_metrics.items():
    print(f"{user_id}: {metrics.total_messages} messages")
    print(f"  Total cost: ${metrics.total_cost:.4f}")
    print(f"  Average response time: {metrics.avg_response_time:.2f}s")
```

---

## 🗄️ Vector Database & ChromaDB

### **Storage Types**

#### **FAISS (In-Memory - Default)**
```python
# When no path is specified:
agent = create_openai_agent(enable_rag=True)
agent.add_documents_to_vector_store(documents)

# Documents are stored in FAISS (RAM)
# ❌ Lost when script restarts
# ❌ Lost when program stops
# ✅ Very fast access
```

#### **ChromaDB (Persistent - Recommended)**
```python
# When path is specified:
agent = create_openai_agent(
    enable_rag=True,
    vector_store_path="./my_documents"  # Custom directory
)
agent.add_documents_to_vector_store(documents)

# Documents are stored in ChromaDB (file system)
# ✅ Persist across script restarts
# ✅ Persist across program stops
# ✅ Can be shared between agents
# ✅ Perfect for production
```

### **What "Program Restart" Means**
- ❌ **NOT system restart** (your computer stays on)
- ❌ **NOT browser restart**
- ✅ **YES, when you stop and restart your Python script**
- ✅ **YES, when you restart your Streamlit app**
- ✅ **YES, when you restart your web server**

### **Accessing Documents**

#### **Through Our Agent (Recommended)**
```python
# Access documents through agent
results = agent._vector_store.similarity_search("your query", k=5)
for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```

#### **Direct ChromaDB Access (Advanced)**
```python
import chromadb

# Connect to the same database
client = chromadb.PersistentClient(path="./company_docs")
collection = client.get_collection("langchain")

# Query documents
results = collection.query(
    query_texts=["your query"],
    n_results=5
)
```

### **Sharing Between Agents**
```python
# Agent 1: Adds documents
agent1 = create_openai_agent(
    enable_rag=True,
    vector_store_path="./shared_docs"
)
agent1.add_documents_to_vector_store(documents)

# Agent 2: Accesses the same documents
agent2 = create_openai_agent(
    enable_rag=True,
    vector_store_path="./shared_docs"  # Same path!
)
# Documents are automatically available!
```

---

## ⚡ Advanced Features

### **Security Integration**
```python
agent = create_openai_agent(
    enable_security=True,
    security_threshold=0.7
)

# Agent automatically filters:
# ✅ Malicious prompts
# ✅ Inappropriate content
# ✅ Security threats
```

### **Performance Monitoring**
```python
# Get real-time metrics
metrics = agent.get_user_metrics("user_123")
print(f"Messages: {metrics.total_messages}")
print(f"Cost: ${metrics.total_cost:.4f}")
print(f"Response time: {metrics.avg_response_time:.2f}s")

# System status
status = agent.get_system_status()
print(f"Active sessions: {status['active_sessions']}")
print(f"Total users: {status['total_users']}")
```

### **Async & Streaming Support**
```python
# Async invocation
response = await agent.ainvoke("user_123", "Hello!")

# Streaming responses
async for chunk in agent.astream("user_123", "Tell me a story"):
    print(chunk, end="", flush=True)
```

---

## 📚 Usage Examples

### **Basic Chatbot**
```python
from ai_agents import create_basic_agent

agent = create_basic_agent()

# Simple conversation
response = agent.invoke("user_123", "Hello!")
print(response)

# Multi-turn conversation
response1 = agent.invoke("user_123", "What is Python?")
response2 = agent.invoke("user_123", "Tell me more about what we discussed")
```

### **Company Knowledge Base**
```python
from ai_agents import create_openai_agent
from langchain_core.documents import Document

# Create RAG-enabled agent
agent = create_openai_agent(
    enable_rag=True,
    vector_store_path="./company_knowledge"
)

# Add company documents
company_docs = [
    Document(page_content="Our company policy..."),
    Document(page_content="Product specifications..."),
    Document(page_content="Customer service guidelines...")
]
agent.add_documents_to_vector_store(company_docs)

# Query company knowledge
response = agent.invoke("employee_123", "What are our customer service guidelines?")
print(response)
```

### **Multi-User Support System**
```python
from ai_agents import create_enterprise_agent

# Create enterprise agent
agent = create_enterprise_agent()

# Multiple users
users = ["alice", "bob", "charlie"]
for user in users:
    response = agent.invoke(user, f"Hello, I'm {user.capitalize()}")
    print(f"{user}: {response}")

# Get analytics
all_metrics = agent.get_all_metrics()
for user_id, metrics in all_metrics.items():
    print(f"{user_id}: {metrics.total_messages} messages")
```

---

## 🔍 Troubleshooting

### **Common Issues**

#### **1. Import Errors**
```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Reinstall the package
pip install -e .
```

#### **2. Missing Dependencies**
```bash
# Install missing packages
pip install package_name

# Or reinstall all requirements
pip install -r requirements.txt
```

#### **3. API Key Issues**
```bash
# Check your .env file
cat .env

# Verify environment variables
echo $OPENAI_API_KEY
```

#### **4. Vector Store Issues**
```bash
# Install vector store dependencies
pip install faiss-cpu chromadb sentence-transformers

# For GPU support (if available)
pip install faiss-gpu
```

### **Performance Issues**

#### **Slow RAG Operations**
- Use `faiss-gpu` instead of `faiss-cpu` if GPU available
- Consider using smaller embedding models
- Implement caching for frequently accessed documents

#### **Memory Issues**
- Use `MemoryType.BUFFER` instead of `MemoryType.SUMMARY_BUFFER`
- Implement conversation cleanup for long sessions
- Monitor memory usage in production

---

## 🎯 Next Steps

### **1. Try the Examples**
- `examples/agent_smoke_test.py` - Basic functionality test
- `examples/streamlit_demo.py` - Interactive web demo
- `examples/advanced_agent_examples.py` - Advanced usage examples

### **2. Explore the Code**
- `ai_agents/advanced_conversation_agent.py` - Main agent implementation
- `ai_agents/__init__.py` - Package exports and factory functions

### **3. Customize for Your Needs**
- Modify `AgentConfig` for custom settings
- Add custom tools and integrations
- Implement custom history backends

---

## 🆘 Getting Help

### **Check the Logs**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### **Common Questions**

**Q: Which agent should I use?**
A: Start with `create_basic_agent()` for testing, then use `create_openai_agent()` for production with specific features enabled.

**Q: How do I enable RAG?**
A: Use `create_openai_agent(enable_rag=True)` and install `faiss-cpu` and `chromadb`.

**Q: How do I add custom tools?**
A: Extend the `AdvancedConversationAgent` class and override the `_create_tools` method.

**Q: Can I use multiple LLM providers?**
A: Yes! Create different agents with different providers and switch between them.

---

## 🎉 Success!

Once you've completed the installation and testing, you'll have:
- ✅ Working AI agents with conversation memory
- ✅ RAG capabilities for document retrieval
- ✅ Tool integration for enhanced functionality
- ✅ Multi-user support with analytics
- ✅ Production-ready architecture

**Happy coding with AI agents! 🚀**

---

## 📚 Additional Resources

- **Main README**: `README.md` - Project overview and features
- **Examples README**: `examples/README.md` - Examples documentation
- **Advanced RAG Plan**: `ADVANCED_RAG_IMPLEMENTATION_PLAN.md` - Future RAG features
- **Showcase Script**: `showcase_agent_capabilities.py` - Live demonstrations
