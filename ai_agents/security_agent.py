"""
Security Agent
An AI agent that detects malicious content and security threats in user queries
"""

import openai
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
import json
import time

load_dotenv()

class SecurityAgent:
    """
    AI Security Agent for detecting malicious content and security threats
    
    Features:
    - Malicious prompt detection
    - Threat classification and scoring
    - Jailbreak attempt detection
    - Content safety analysis
    - Configurable threat thresholds
    """
    
    def __init__(self, 
                 model: str = "gpt-4o",
                 threat_threshold: float = 0.7,
                 enable_detailed_analysis: bool = True):
        """
        Initialize the Security Agent
        
        Args:
            model: OpenAI model to use for analysis
            threat_threshold: Threshold for blocking content (0.0-1.0)
            enable_detailed_analysis: Enable detailed threat analysis
        """
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = model
        self.threat_threshold = threat_threshold
        self.enable_detailed_analysis = enable_detailed_analysis
        
        # Threat categories
        self.threat_categories = [
            'PROMPT_INJECTION', 'JAILBREAK', 'VIOLENCE', 'WEAPONS', 'CRIME',
            'SELF_HARM', 'CHILD_EXPLOITATION', 'MALWARE', 'PHISHING', 'SEXUAL',
            'HATE_SPEECH', 'PROFANITY', 'HARASSMENT', 'PRIVACY', 'SPAM',
            'MISINFORMATION', 'MANIPULATION', 'INAPPROPRIATE', 'SPAM_LIKE', 'OFF_TOPIC'
        ]
        
        # Agent metadata
        self.agent_info = {
            'name': 'Security Agent',
            'version': '1.0.0',
            'description': 'AI agent for detecting malicious content and security threats',
            'capabilities': ['threat_detection', 'content_safety', 'malicious_prompt_detection'],
            'model': self.model,
            'threat_threshold': self.threat_threshold
        }
    
    def analyze_security(self, 
                        user_query: str, 
                        conversation_context: str = None,
                        user_profile: Dict = None) -> Dict:
        """
        Analyze user query for security threats
        
        Args:
            user_query: The user's input query
            conversation_context: Optional conversation history
            user_profile: Optional user profile for context
            
        Returns:
            Dictionary containing security analysis results
        """
        start_time = time.time()
        
        try:
            # LLM-based security analysis
            llm_analysis = self._llm_security_analysis(
                user_query, 
                conversation_context, 
                user_profile
            )
            
            # Determine threat level and blocking decision
            threat_level = self._determine_threat_level(llm_analysis['threat_score'])
            should_block = llm_analysis['threat_score'] >= self.threat_threshold
            
            # Compile results
            processing_time = time.time() - start_time
            
            return {
                'is_malicious': should_block,
                'threat_level': threat_level,
                'threat_score': llm_analysis['threat_score'],
                'confidence_score': llm_analysis['confidence'],
                'blocked': should_block,
                'warnings': llm_analysis.get('warnings', []),
                'llm_analysis': llm_analysis,
                'processing_time': processing_time,
                'agent_info': self.agent_info,
                'metadata': {
                    'model_used': self.model,
                    'threat_threshold': self.threat_threshold,
                    'detailed_analysis': self.enable_detailed_analysis
                }
            }
            
        except Exception as e:
            # Fallback to basic analysis
            return self._fallback_security_analysis(user_query, str(e))
    
    def _llm_security_analysis(self, 
                              user_query: str, 
                              conversation_context: str = None,
                              user_profile: Dict = None) -> Dict:
        """Use LLM to analyze security threats"""
        try:
            # Build context information
            context_info = ""
            if conversation_context:
                context_info += f"\nConversation Context: {conversation_context}"
            if user_profile:
                context_info += f"\nUser Profile: {json.dumps(user_profile)}"
            
            # Create comprehensive security prompt
            prompt = f"""
            You are an AI Security Agent specialized in detecting malicious content and security threats.
            
            Analyze the following user query for security threats and malicious intent.
            
            User Query: "{user_query}"{context_info}
            
            Threat Categories to Detect:
            {', '.join(self.threat_categories)}
            
            Provide a JSON response with:
            - threat_score: float between 0.0 (safe) and 1.0 (highly dangerous)
            - threat_type: string indicating the primary threat category
            - confidence: float between 0.0 and 1.0 indicating analysis confidence
            - reasoning: string explaining the threat analysis
            - should_block: boolean indicating if this content should be blocked
            - detected_threats: list of all detected threat categories
            - warnings: list of specific security warnings
            
            Response format:
            {{
                "threat_score": 0.8,
                "threat_type": "WEAPONS",
                "confidence": 0.95,
                "reasoning": "This query contains dangerous content...",
                "should_block": true,
                "detected_threats": ["weapons", "violence"],
                "warnings": ["Query involves illegal activities", "Potential harm to others"]
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert AI Security Agent. Analyze content for security threats accurately and thoroughly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse the response
            try:
                analysis = json.loads(response.choices[0].message.content.strip())
                return analysis
            except json.JSONDecodeError:
                # Fallback parsing
                return self._parse_fallback(response.choices[0].message.content, user_query)
                
        except Exception as e:
            print(f"LLM security analysis error: {e}")
            return self._parse_fallback("", user_query)
    
    def _parse_fallback(self, llm_response: str, user_query: str) -> Dict:
        """Fallback parsing if LLM response is invalid"""
        # Basic keyword-based threat detection
        query_lower = user_query.lower()
        
        # Threat indicators
        threat_indicators = {
            'weapons': ['gun', 'weapon', 'bomb', 'explosive', 'knife', 'ak-47', 'rifle', 'pistol'],
            'violence': ['kill', 'murder', 'attack', 'harm', 'hurt', 'fight', 'violence'],
            'illegal': ['illegal', 'crime', 'steal', 'rob', 'hack', 'cheat', 'fraud'],
            'inappropriate': ['porn', 'sexual', 'explicit', 'adult content'],
            'hate': ['hate', 'racist', 'discrimination', 'bigotry']
        }
        
        detected_threats = []
        max_threat_score = 0.0
        
        for threat_type, keywords in threat_indicators.items():
            for keyword in keywords:
                if keyword in query_lower:
                    detected_threats.append(threat_type)
                    max_threat_score = max(max_threat_score, 0.8)
                    break
        
        return {
            "threat_score": max_threat_score,
            "threat_type": detected_threats[0] if detected_threats else "UNKNOWN",
            "confidence": 0.6,
            "reasoning": f"Fallback analysis detected threats: {detected_threats}",
            "should_block": max_threat_score >= self.threat_threshold,
            "detected_threats": detected_threats,
            "warnings": [f"Content may contain {threat} content" for threat in detected_threats]
        }
    
    def _determine_threat_level(self, threat_score: float) -> str:
        """Determine threat level based on score"""
        if threat_score >= 0.8:
            return 'critical'
        elif threat_score >= 0.6:
            return 'high'
        elif threat_score >= 0.4:
            return 'medium'
        elif threat_score >= 0.2:
            return 'low'
        else:
            return 'safe'
    
    def _fallback_security_analysis(self, user_query: str, error: str) -> Dict:
        """Fallback security analysis when LLM fails"""
        return {
            'is_malicious': True,  # Conservative approach
            'threat_level': 'high',
            'threat_score': 0.8,
            'confidence_score': 0.3,
            'blocked': True,
            'warnings': [f"Security analysis failed: {error}. Blocking for safety."],
            'llm_analysis': {
                'threat_score': 0.8,
                'threat_type': 'ANALYSIS_ERROR',
                'confidence': 0.3,
                'reasoning': f"Security analysis failed: {error}",
                'should_block': True,
                'detected_threats': ['analysis_error'],
                'warnings': [f"Security analysis failed: {error}"]
            },
            'processing_time': 0.0,
            'agent_info': self.agent_info,
            'metadata': {
                'model_used': 'fallback',
                'threat_threshold': self.threat_threshold,
                'detailed_analysis': False
            }
        }
    
    def update_config(self, 
                     model: str = None,
                     threat_threshold: float = None,
                     enable_detailed_analysis: bool = None):
        """Update agent configuration"""
        if model:
            self.model = model
            self.agent_info['model'] = model
        if threat_threshold is not None:
            self.threat_threshold = threat_threshold
            self.agent_info['threat_threshold'] = threat_threshold
        if enable_detailed_analysis is not None:
            self.enable_detailed_analysis = enable_detailed_analysis
    
    def get_agent_info(self) -> Dict:
        """Get agent information and capabilities"""
        return self.agent_info.copy()
    
    def get_threat_categories(self) -> List[str]:
        """Get available threat categories"""
        return self.threat_categories.copy()
    
    def add_threat_category(self, category: str):
        """Add a new threat category"""
        if category not in self.threat_categories:
            self.threat_categories.append(category)
    
    def is_safe(self, user_query: str) -> bool:
        """Quick safety check"""
        result = self.analyze_security(user_query)
        return not result['blocked']
