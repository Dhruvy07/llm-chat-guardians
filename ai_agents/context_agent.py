"""
Context Agent
An AI agent that analyzes query relevance and context for chatbots
"""

import openai
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
import json
import time

load_dotenv()

class ContextAgent:
    """
    AI Context Agent for analyzing query relevance and context
    
    Features:
    - Dynamic domain detection
    - Context relevance scoring
    - Conversation flow analysis
    - Multi-domain support
    - Configurable relevance thresholds
    """
    
    def __init__(self, 
                 chatbot_name: str = "AI Assistant",
                 chatbot_description: str = "A helpful AI assistant",
                 keywords: List[str] = None,
                 chatbot_prompt: str = None,
                 model: str = "gpt-3.5-turbo",
                 relevance_thresholds: Dict = None):
        """
        Initialize the Context Agent
        
        Args:
            chatbot_name: Name of the chatbot/app
            chatbot_description: Description of what the chatbot does
            keywords: List of keywords that enhance detection
            chatbot_prompt: Optional system prompt used by the chatbot
            model: OpenAI model to use for analysis
            relevance_thresholds: Custom relevance thresholds
        """
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.chatbot_name = chatbot_name
        self.chatbot_description = chatbot_description
        self.keywords = keywords or []
        self.chatbot_prompt = chatbot_prompt
        self.model = model
        
        # Default relevance thresholds
        self.relevance_thresholds = relevance_thresholds or {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4,
            'irrelevant': 0.2
        }
        
        # Agent metadata
        self.agent_info = {
            'name': 'Context Agent',
            'version': '1.0.0',
            'description': 'AI agent for analyzing query relevance and context',
            'capabilities': ['context_analysis', 'relevance_scoring', 'domain_detection', 'conversation_flow'],
            'model': self.model,
            'chatbot_name': self.chatbot_name,
            'chatbot_description': self.chatbot_description,
            'keywords_count': len(self.keywords),
            'has_prompt': bool(self.chatbot_prompt)
        }
        
        # Conversation history
        self.conversation_history = []
    
    def analyze_context(self, 
                       user_query: str, 
                       conversation_history: List[Dict] = None,
                       user_profile: Dict = None) -> Dict:
        """
        Analyze the contextual relevance of a user query
        
        Args:
            user_query: The current user query
            conversation_history: List of previous conversation turns
            user_profile: Optional user profile for context
            
        Returns:
            Dictionary containing context analysis results
        """
        start_time = time.time()
        
        # Update conversation history
        if conversation_history:
            self.conversation_history = conversation_history
        
        # Prepare conversation context
        conversation_context = self._prepare_conversation_context()
        
        # LLM-based context analysis
        llm_analysis = self._llm_context_analysis(
            user_query, 
            conversation_context,
            user_profile
        )
        
        # Determine relevance level
        relevance_level = self._get_relevance_level(llm_analysis['relevance_score'])
        
        # Compile results
        processing_time = time.time() - start_time
        
        return {
            'is_contextual': llm_analysis['relevance_score'] > self.relevance_thresholds['low'],
            'relevance_score': llm_analysis['relevance_score'],
            'relevance_level': relevance_level,
            'reasoning': llm_analysis['reasoning'],
            'suggested_response': llm_analysis['suggested_response'],
            'context_shift': llm_analysis['context_shift'],
            'domain_alignment': llm_analysis['domain_alignment'],
            'conversation_flow': llm_analysis.get('conversation_flow', 'smooth'),
            'processing_time': processing_time,
            'agent_info': self.agent_info,
            'chatbot_context': {
                'name': self.chatbot_name,
                'description': self.chatbot_description,
                'keywords_used': self._get_matching_keywords(user_query),
                'prompt_available': bool(self.chatbot_prompt),
                'total_keywords': len(self.keywords)
            },
            'metadata': {
                'model_used': self.model,
                'relevance_thresholds': self.relevance_thresholds.copy(),
                'conversation_length': len(self.conversation_history)
            }
        }
    
    def _prepare_conversation_context(self) -> str:
        """Prepare conversation history for context analysis"""
        if not self.conversation_history:
            return "No previous conversation context available."
        
        # Take last 5 turns for context (to avoid token limits)
        recent_history = self.conversation_history[-5:]
        
        context_parts = []
        for turn in recent_history:
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            context_parts.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(context_parts)
    
    def _get_matching_keywords(self, query: str) -> List[str]:
        """Get keywords that match the current query"""
        query_lower = query.lower()
        matching_keywords = []
        
        for keyword in self.keywords:
            if keyword.lower() in query_lower:
                matching_keywords.append(keyword)
        
        return matching_keywords
    
    def _llm_context_analysis(self, 
                            current_query: str, 
                            conversation_context: str,
                            user_profile: Dict = None) -> Dict:
        """Use LLM to analyze context relevance"""
        try:
            # Build context information
            context_info = f"""
Chatbot Information:
- Name: {self.chatbot_name}
- Description: {self.chatbot_description}
- Keywords: {', '.join(self.keywords) if self.keywords else 'None provided'}
"""
            
            if self.chatbot_prompt:
                context_info += f"- System Prompt: {self.chatbot_prompt}\n"
            
            if user_profile:
                context_info += f"- User Profile: {json.dumps(user_profile)}\n"
            
            # Create the analysis prompt
            prompt = f"""
            You are an AI Context Agent specialized in analyzing conversational context and relevance.
            
            Analyze the contextual relevance of the current query in relation to the chatbot's purpose and conversation history.
            
            {context_info}
            
            Conversation History:
            {conversation_context}
            
            Current Query: "{current_query}"
            
            Consider:
            1. How well the query aligns with the chatbot's purpose and description
            2. Whether the query uses any of the provided keywords
            3. If the query fits within the chatbot's system prompt scope
            4. Whether this represents a topic shift from the conversation history
            5. The overall relevance to what this chatbot is designed to help with
            6. The natural flow of conversation
            
            Provide a JSON response with:
            - relevance_score: float between 0.0 (completely irrelevant) and 1.0 (highly relevant)
            - reasoning: string explaining why the query is or isn't contextually relevant
            - context_shift: boolean indicating if this query represents a significant topic shift
            - domain_alignment: float between 0.0 and 1.0 indicating alignment with chatbot's purpose
            - suggested_response: string suggesting how to handle non-contextual queries
            - conversation_flow: string indicating flow quality ('smooth', 'moderate_shift', 'major_shift')
            
            Response format:
            {{
                "relevance_score": 0.8,
                "reasoning": "This query is relevant to the chatbot's purpose because...",
                "context_shift": false,
                "domain_alignment": 0.9,
                "suggested_response": "I can help you with that. Let me provide relevant information.",
                "conversation_flow": "smooth"
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert AI Context Agent. Analyze conversational context and relevance accurately and thoroughly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=400
            )
            
            # Parse the response
            try:
                analysis = json.loads(response.choices[0].message.content.strip())
                return analysis
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return self._fallback_context_analysis(current_query)
                
        except Exception as e:
            print(f"LLM context analysis error: {e}")
            return self._fallback_context_analysis(current_query)
    
    def _fallback_context_analysis(self, current_query: str) -> Dict:
        """Fallback context analysis when LLM fails"""
        # Basic keyword-based relevance
        matching_keywords = self._get_matching_keywords(current_query)
        keyword_score = min(len(matching_keywords) / max(len(self.keywords), 1) * 2, 1.0) if self.keywords else 0.5
        
        return {
            "relevance_score": keyword_score,
            "reasoning": f"Fallback analysis based on keyword matching. Found {len(matching_keywords)} matching keywords.",
            "context_shift": False,
            "domain_alignment": keyword_score,
            "suggested_response": "I'm here to help. How can I assist you?",
            "conversation_flow": "smooth"
        }
    
    def _get_relevance_level(self, score: float) -> str:
        """Get relevance level based on score"""
        if score >= self.relevance_thresholds['high']:
            return 'high'
        elif score >= self.relevance_thresholds['medium']:
            return 'medium'
        elif score >= self.relevance_thresholds['low']:
            return 'low'
        else:
            return 'irrelevant'
    
    def update_config(self, 
                     chatbot_name: str = None,
                     chatbot_description: str = None,
                     keywords: List[str] = None,
                     chatbot_prompt: str = None,
                     model: str = None,
                     relevance_thresholds: Dict = None):
        """Update the agent configuration"""
        if chatbot_name:
            self.chatbot_name = chatbot_name
            self.agent_info['chatbot_name'] = chatbot_name
        if chatbot_description:
            self.chatbot_description = chatbot_description
            self.agent_info['chatbot_description'] = chatbot_description
        if keywords is not None:
            self.keywords = keywords
            self.agent_info['keywords_count'] = len(keywords)
        if chatbot_prompt is not None:
            self.chatbot_prompt = chatbot_prompt
            self.agent_info['has_prompt'] = bool(chatbot_prompt)
        if model:
            self.model = model
            self.agent_info['model'] = model
        if relevance_thresholds:
            self.relevance_thresholds = relevance_thresholds
    
    def get_config(self) -> Dict:
        """Get current configuration"""
        return {
            'chatbot_name': self.chatbot_name,
            'chatbot_description': self.chatbot_description,
            'keywords': self.keywords.copy(),
            'chatbot_prompt': self.chatbot_prompt,
            'model': self.model,
            'relevance_thresholds': self.relevance_thresholds.copy()
        }
    
    def get_agent_info(self) -> Dict:
        """Get agent information and capabilities"""
        return self.agent_info.copy()
    
    def is_contextual(self, current_query: str, conversation_history: List[Dict] = None) -> bool:
        """Quick check if query is contextual"""
        result = self.analyze_context(current_query, conversation_history)
        return result['is_contextual']
    
    def get_context_score(self, current_query: str, conversation_history: List[Dict] = None) -> float:
        """Get context relevance score"""
        result = self.analyze_context(current_query, conversation_history)
        return result['relevance_score']
    
    def add_keywords(self, new_keywords: List[str]):
        """Add new keywords to the agent"""
        for keyword in new_keywords:
            if keyword not in self.keywords:
                self.keywords.append(keyword)
        self.agent_info['keywords_count'] = len(self.keywords)
    
    def remove_keywords(self, keywords_to_remove: List[str]):
        """Remove keywords from the agent"""
        for keyword in keywords_to_remove:
            if keyword in self.keywords:
                self.keywords.remove(keyword)
        self.agent_info['keywords_count'] = len(self.keywords)
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
        self.agent_info['conversation_length'] = 0
