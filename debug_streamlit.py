#!/usr/bin/env python3
"""
Debug script to test the exact streamlit demo logic
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_agents.security_agent import SecurityAgent
from ai_agents.context_agent import ContextAgent
from ai_agents.model_selection_agent import ModelSelectionAgent
import openai

def test_streamlit_logic():
    """Test the exact logic from streamlit demo"""
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Initialize agents (exact same as streamlit)
    print("🔧 Initializing agents...")
    security_agent = SecurityAgent(
        model="gpt-4o",
        threat_threshold=0.7,
        enable_detailed_analysis=True
    )
    
    context_agent = ContextAgent(
        chatbot_name="HealthCheck AI",
        chatbot_description="A medical assistant chatbot that provides health information, symptom analysis, and wellness guidance",
        keywords=[
            "health", "medical", "symptoms", "diagnosis", "treatment", "medicine", "doctor", "hospital",
            "pain", "fever", "headache", "cough", "cold", "flu", "vaccine", "medication", "prescription",
            "wellness", "fitness", "nutrition", "diet", "exercise", "mental health", "anxiety", "depression",
            "chronic", "acute", "emergency", "urgent", "checkup", "screening", "prevention", "lifestyle"
        ],
        chatbot_prompt="You are HealthCheck AI, a medical assistant designed to provide health information, symptom analysis, and wellness guidance. Always remind users that you are not a substitute for professional medical advice and encourage them to consult healthcare providers for serious concerns.",
        model="gpt-4o"
    )
    
    model_selection_agent = ModelSelectionAgent(
        cost_sensitivity="medium",
        performance_preference="balanced"
    )
    
    # Test query
    test_query = "helo how are you ?"
    print(f"\n🧪 Testing with query: '{test_query}'")
    
    # Step 1: Security Analysis
    print("\n🛡️ Security Analysis...")
    security_result = security_agent.analyze_security(test_query)
    print(f"✅ Security: {security_result['threat_level']} (Score: {security_result['threat_score']:.2f})")
    print(f"   Blocked: {security_result['blocked']}")
    
    # Step 2: Context Analysis
    print("\n🧠 Context Analysis...")
    context_result = context_agent.analyze_context(test_query)
    print(f"✅ Context: {context_result['relevance_level']} (Score: {context_result['relevance_score']:.2f})")
    
    # Step 3: Model Selection
    print("\n🎯 Model Selection...")
    model_result = model_selection_agent.select_model(
        test_query, 
        context_relevance=context_result['relevance_score']
    )
    print(f"✅ Model: {model_result['model_info']['name']} (Confidence: {model_result['confidence_score']:.2f})")
    print(f"   Selected Model ID: {model_result['selected_model']}")
    
    # Step 4: Generate response (exact same as streamlit)
    print("\n🤖 Response Generation...")
    try:
        # Prepare system message (exact same as streamlit)
        system_message = f"""You are HealthCheck AI, a medical assistant designed to provide health information, symptom analysis, and wellness guidance.

Context Analysis: {context_result['reasoning']}
Security Status: {'Safe' if not security_result['blocked'] else 'Blocked'}
Selected Model: {model_result['model_info']['name']} - {model_result['model_info']['description']}

IMPORTANT: You are not a substitute for professional medical advice. For serious health concerns, always recommend consulting a healthcare provider. Provide evidence-based information while maintaining a caring and professional tone."""
        
        print(f"System Message Length: {len(system_message)}")
        print(f"User Query: {test_query}")
        print(f"Model: {model_result['selected_model']}")
        print(f"Max Tokens: {max(model_result['estimated_tokens'], 100)}")
        
        # Generate response (exact same as streamlit)
        response = client.chat.completions.create(
            model=model_result['selected_model'],  # Use the selected model
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": test_query}
            ],
            max_completion_tokens=max(model_result['estimated_tokens'], 100)
        )
        
        response_text = response.choices[0].message.content
        print(f"✅ Raw Response: {response_text}")
        print(f"✅ Response Length: {len(response_text)}")
        
        # Add agent information (exact same as streamlit)
        agent_info = f"""

---
🤖 **AI Agents Analysis:**
🛡️ **Security:** {security_result['threat_level'].title()} (Score: {security_result['threat_score']:.2f})
🧠 **Context:** {context_result['relevance_level'].title()} (Score: {context_result['relevance_score']:.2f})
🎯 **Model:** {model_result['model_info']['name']} (Confidence: {model_result['confidence_score']:.2f})
⏱️ **Processing Time:** {model_result['processing_time']:.2f}s
💰 **Estimated Cost:** ${model_result['estimated_cost']:.4f}
"""
        
        final_response = response_text + agent_info
        print(f"✅ Final Response Length: {len(final_response)}")
        print(f"✅ Final Response Preview: {final_response[:300]}...")
        
        return final_response
        
    except Exception as e:
        print(f"❌ Response Generation Error: {e}")
        import traceback
        traceback.print_exc()
        return f"I apologize, but I encountered an error while generating a response: {str(e)}"

if __name__ == "__main__":
    result = test_streamlit_logic()
    print(f"\n🎯 Final Result: {result}")
