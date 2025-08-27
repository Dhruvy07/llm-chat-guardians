"""
Model Selection Agent
An AI agent that intelligently selects the optimal LLM model for queries
"""

import openai
import os
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import json
import time

load_dotenv()

class ModelSelectionAgent:
    """
    AI Model Selection Agent for choosing optimal LLM models
    
    Features:
    - Intelligent model selection based on query characteristics
    - Cost optimization and performance balancing
    - Dynamic model configuration
    - Query complexity analysis
    - Multi-provider support
    """
    
    def __init__(self, 
                 available_models: Dict = None,
                 default_model: str = "gpt-3.5-turbo",
                 cost_sensitivity: str = "medium",
                 performance_preference: str = "balanced"):
        """
        Initialize the Model Selection Agent
        
        Args:
            available_models: Dictionary of available models with their capabilities
            default_model: Default model to use as fallback
            cost_sensitivity: Cost sensitivity level ('low', 'medium', 'high')
            performance_preference: Performance preference ('speed', 'balanced', 'quality')
        """
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.default_model = default_model
        self.cost_sensitivity = cost_sensitivity
        self.performance_preference = performance_preference
        
        # Default available models if none provided
        self.available_models = available_models or self._get_default_models()
        
        # Agent metadata
        self.agent_info = {
            'name': 'Model Selection Agent',
            'version': '1.0.0',
            'description': 'AI agent for intelligent LLM model selection',
            'capabilities': ['model_selection', 'query_analysis', 'cost_optimization', 'performance_balancing'],
            'default_model': self.default_model,
            'cost_sensitivity': self.cost_sensitivity,
            'performance_preference': self.performance_preference,
            'available_models_count': len(self.available_models)
        }
        
        # Selection history for learning
        self.selection_history = []
    
    def _get_default_models(self) -> Dict:
        """Get default available models configuration"""
        return {
            'gpt-3.5-turbo': {
                'name': 'GPT-3.5 Turbo',
                'provider': 'OpenAI',
                'description': 'Fast and cost-effective model for general tasks',
                'capabilities': ['general_conversation', 'basic_analysis', 'fast_response'],
                'max_tokens': 4096,
                'cost_per_1k_tokens': 0.0015,
                'speed_rating': 0.9,
                'quality_rating': 0.7,
                'best_for': ['simple_queries', 'general_conversation', 'cost_sensitive_tasks']
            },
            'gpt-4o': {
                'name': 'GPT-4o',
                'provider': 'OpenAI',
                'description': 'High-performance model for complex tasks',
                'capabilities': ['complex_analysis', 'detailed_explanations', 'high_accuracy'],
                'max_tokens': 128000,
                'cost_per_1k_tokens': 0.005,
                'speed_rating': 0.8,
                'quality_rating': 0.95,
                'best_for': ['complex_queries', 'detailed_analysis', 'high_quality_responses']
            },
            'gpt-4o-mini': {
                'name': 'GPT-4o Mini',
                'provider': 'OpenAI',
                'description': 'Balanced model with good performance and reasonable cost',
                'capabilities': ['moderate_analysis', 'good_accuracy', 'balanced_performance'],
                'max_tokens': 16384,
                'cost_per_1k_tokens': 0.00015,
                'speed_rating': 0.85,
                'quality_rating': 0.8,
                'best_for': ['moderate_complexity', 'balanced_tasks', 'cost_conscious_quality']
            }
        }
    
    def select_model(self, 
                    user_query: str, 
                    conversation_context: str = None,
                    user_preferences: Dict = None,
                    query_metadata: Dict = None) -> Dict:
        """
        Select the optimal model for a user query
        
        Args:
            user_query: The user's input query
            conversation_context: Optional conversation context
            user_preferences: Optional user preferences
            query_metadata: Optional query metadata
            
        Returns:
            Dictionary containing model selection results
        """
        start_time = time.time()
        
        try:
            # Step 1: Analyze query requirements
            query_analysis = self._analyze_query_requirements(
                user_query, 
                conversation_context, 
                user_preferences
            )
            
            # Step 2: Get candidate models
            candidate_models = self._get_candidate_models(query_analysis)
            
            # Step 3: Score and rank models
            ranked_models = self._rank_models(candidate_models, query_analysis)
            
            # Step 4: Select best model
            selected_model = ranked_models[0] if ranked_models else self.available_models[self.default_model]
            # Ensure selected_model has an 'id' field
            if 'id' not in selected_model:
                selected_model['id'] = self.default_model
            
            # Step 5: Compile results
            processing_time = time.time() - start_time
            
            # Update selection history
            self._update_selection_history(selected_model, query_analysis, processing_time)
            
            return {
                'selected_model': selected_model['id'],
                'model_info': selected_model,
                'selection_reasoning': self._generate_selection_reasoning(selected_model, query_analysis),
                'confidence_score': self._calculate_selection_confidence(selected_model, query_analysis),
                'estimated_cost': self._estimate_cost(selected_model, query_analysis),
                'estimated_tokens': query_analysis['estimated_tokens'],
                'query_analysis': query_analysis,
                'candidate_models': candidate_models,
                'processing_time': processing_time,
                'agent_info': self.agent_info,
                'metadata': {
                    'cost_sensitivity': self.cost_sensitivity,
                    'performance_preference': self.performance_preference,
                    'models_considered': len(candidate_models),
                    'selection_criteria': list(query_analysis.keys())
                }
            }
            
        except Exception as e:
            print(f"Model selection error: {e}")
            return self._fallback_model_selection(user_query, str(e))
    
    def _analyze_query_requirements(self, 
                                  user_query: str, 
                                  conversation_context: str = None,
                                  user_preferences: Dict = None) -> Dict:
        """Analyze query to determine requirements"""
        try:
            # LLM-based query analysis
            llm_analysis = self._llm_query_analysis(
                user_query, 
                conversation_context, 
                user_preferences
            )
            
            return llm_analysis
            
        except Exception as e:
            print(f"LLM query analysis error: {e}")
            return self._fallback_query_analysis(user_query)
    
    def _llm_query_analysis(self, 
                           user_query: str, 
                           conversation_context: str = None,
                           user_preferences: Dict = None) -> Dict:
        """Use LLM to analyze query requirements"""
        try:
            # Build context information
            context_info = ""
            if conversation_context:
                context_info += f"\nConversation Context: {conversation_context}"
            if user_preferences:
                context_info += f"\nUser Preferences: {json.dumps(user_preferences)}"
            
            # Create analysis prompt
            prompt = f"""
            You are an AI Model Selection Agent specialized in analyzing queries to determine optimal LLM model requirements.
            
            Analyze the following user query to determine:
            1. Query complexity level
            2. Required capabilities
            3. Estimated token usage
            4. Performance requirements
            5. Cost considerations
            
            User Query: "{user_query}"{context_info}
            
            Provide a JSON response with:
            - complexity_score: float between 0.0 (simple) and 1.0 (very complex)
            - domain: string indicating the query domain (e.g., "health", "technology", "general")
            - required_capabilities: list of required capabilities
            - estimated_tokens: integer estimate of token usage
            - performance_priority: string indicating priority ("speed", "quality", "balanced")
            - cost_sensitivity: string indicating cost sensitivity ("low", "medium", "high")
            
            Response format:
            {{
                "complexity_score": 0.7,
                "domain": "health",
                "required_capabilities": ["detailed_analysis", "medical_knowledge"],
                "estimated_tokens": 500,
                "performance_priority": "quality",
                "cost_sensitivity": "medium"
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert AI Model Selection Agent. Analyze queries accurately to determine optimal model requirements."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            # Parse the response
            try:
                analysis = json.loads(response.choices[0].message.content.strip())
                return analysis
            except json.JSONDecodeError:
                return self._fallback_query_analysis(user_query)
                
        except Exception as e:
            print(f"LLM query analysis error: {e}")
            return self._fallback_query_analysis(user_query)
    
    def _fallback_query_analysis(self, user_query: str) -> Dict:
        """Fallback query analysis when LLM fails"""
        # Basic keyword-based analysis
        query_lower = user_query.lower()
        
        # Simple complexity estimation
        complexity_indicators = [
            'explain', 'analyze', 'compare', 'discuss', 'evaluate', 'research',
            'detailed', 'comprehensive', 'thorough', 'in-depth', 'complex'
        ]
        
        complexity_score = 0.3  # Default to simple
        for indicator in complexity_indicators:
            if indicator in query_lower:
                complexity_score += 0.2
        
        complexity_score = min(complexity_score, 1.0)
        
        return {
            "complexity_score": complexity_score,
            "domain": "general",
            "required_capabilities": ["general_conversation"],
            "estimated_tokens": 300,
            "performance_priority": "balanced",
            "cost_sensitivity": "medium"
        }
    
    def _get_candidate_models(self, query_analysis: Dict) -> List[Dict]:
        """Get candidate models based on query requirements"""
        candidates = []
        
        for model_id, model_info in self.available_models.items():
            # Check if model has required capabilities
            if self._model_matches_requirements(model_info, query_analysis):
                candidate = model_info.copy()
                candidate['id'] = model_id
                candidates.append(candidate)
        
        return candidates
    
    def _model_matches_requirements(self, model_info: Dict, query_analysis: Dict) -> bool:
        """Check if model matches query requirements"""
        required_capabilities = query_analysis.get('required_capabilities', [])
        
        # Check capability match
        for capability in required_capabilities:
            if capability not in model_info.get('capabilities', []):
                return False
        
        return True
    
    def _rank_models(self, candidate_models: List[Dict], query_analysis: Dict) -> List[Dict]:
        """Rank candidate models by suitability"""
        if not candidate_models:
            return []
        
        # Score each candidate
        scored_models = []
        for model in candidate_models:
            score = self._calculate_model_score(model, query_analysis)
            scored_models.append((model, score))
        
        # Sort by score (highest first)
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        # Return models in ranked order
        return [model for model, score in scored_models]
    
    def _calculate_model_score(self, model: Dict, query_analysis: Dict) -> float:
        """Calculate suitability score for a model"""
        complexity_score = query_analysis.get('complexity_score', 0.5)
        performance_priority = query_analysis.get('performance_priority', 'balanced')
        cost_sensitivity = query_analysis.get('cost_sensitivity', 'medium')
        
        # Base score from model ratings
        base_score = (model.get('quality_rating', 0.5) + model.get('speed_rating', 0.5)) / 2
        
        # Complexity adjustment
        if complexity_score > 0.7:
            # Complex queries prefer quality
            quality_weight = 0.8
            speed_weight = 0.2
        elif complexity_score < 0.3:
            # Simple queries prefer speed
            quality_weight = 0.2
            speed_weight = 0.8
        else:
            # Balanced queries
            quality_weight = 0.5
            speed_weight = 0.5
        
        # Performance preference adjustment
        if performance_priority == 'quality':
            quality_weight *= 1.5
        elif performance_priority == 'speed':
            speed_weight *= 1.5
        
        # Cost sensitivity adjustment
        cost_multiplier = 1.0
        if cost_sensitivity == 'high':
            cost_multiplier = 0.7  # Prefer cheaper models
        elif cost_sensitivity == 'low':
            cost_multiplier = 1.3  # Prefer better models
        
        # Calculate final score
        final_score = (
            base_score * 0.4 +
            model.get('quality_rating', 0.5) * quality_weight * 0.3 +
            model.get('speed_rating', 0.5) * speed_weight * 0.3
        ) * cost_multiplier
        
        return min(max(final_score, 0.0), 1.0)
    
    def _generate_selection_reasoning(self, selected_model: Dict, query_analysis: Dict) -> str:
        """Generate human-readable reasoning for model selection"""
        complexity = query_analysis.get('complexity_score', 0.5)
        domain = query_analysis.get('domain', 'general')
        capabilities = query_analysis.get('required_capabilities', [])
        
        reasoning_parts = []
        
        if complexity > 0.7:
            reasoning_parts.append("Complex query requiring detailed analysis")
        elif complexity < 0.3:
            reasoning_parts.append("Simple query suitable for fast response")
        else:
            reasoning_parts.append("Moderate complexity query")
        
        if domain != 'general':
            reasoning_parts.append(f"Domain-specific requirements for {domain}")
        
        if capabilities:
            reasoning_parts.append(f"Required capabilities: {', '.join(capabilities)}")
        
        reasoning_parts.append(f"Selected {selected_model['name']} for optimal performance")
        
        return ". ".join(reasoning_parts) + "."
    
    def _calculate_selection_confidence(self, selected_model: Dict, query_analysis: Dict) -> float:
        """Calculate confidence in the model selection"""
        # Base confidence from model capabilities match
        required_capabilities = query_analysis.get('required_capabilities', [])
        available_capabilities = selected_model.get('capabilities', [])
        
        capability_match = sum(1 for cap in required_capabilities if cap in available_capabilities)
        capability_score = capability_match / max(len(required_capabilities), 1)
        
        # Complexity alignment
        complexity_score = query_analysis.get('complexity_score', 0.5)
        model_quality = selected_model.get('quality_rating', 0.5)
        
        complexity_alignment = 1.0 - abs(complexity_score - model_quality)
        
        # Overall confidence
        confidence = (capability_score * 0.6 + complexity_alignment * 0.4)
        
        return min(max(confidence, 0.0), 1.0)
    
    def _estimate_cost(self, selected_model: Dict, query_analysis: Dict) -> float:
        """Estimate cost for the selected model"""
        estimated_tokens = query_analysis.get('estimated_tokens', 300)
        cost_per_1k = selected_model.get('cost_per_1k_tokens', 0.001)
        
        return (estimated_tokens / 1000) * cost_per_1k
    
    def _update_selection_history(self, selected_model: Dict, query_analysis: Dict, processing_time: float):
        """Update selection history for learning"""
        history_entry = {
            'timestamp': time.time(),
            'selected_model': selected_model['id'],
            'query_analysis': query_analysis,
            'processing_time': processing_time
        }
        
        self.selection_history.append(history_entry)
        
        # Keep only last 100 entries
        if len(self.selection_history) > 100:
            self.selection_history = self.selection_history[-100:]
    
    def _fallback_model_selection(self, user_query: str, error: str) -> Dict:
        """Fallback model selection when analysis fails"""
        default_model_info = self.available_models[self.default_model].copy()
        default_model_info['id'] = self.default_model
        
        return {
            'selected_model': self.default_model,
            'model_info': default_model_info,
            'selection_reasoning': f"Fallback to default model due to error: {error}",
            'confidence_score': 0.3,
            'estimated_cost': 0.001,
            'estimated_tokens': 300,
            'query_analysis': {'complexity_score': 0.5, 'domain': 'general'},
            'candidate_models': [default_model_info],
            'processing_time': 0.0,
            'agent_info': self.agent_info,
            'metadata': {
                'cost_sensitivity': self.cost_sensitivity,
                'performance_preference': self.performance_preference,
                'models_considered': 1,
                'selection_criteria': ['fallback'],
                'error': error
            }
        }
    
    def update_config(self, 
                     available_models: Dict = None,
                     default_model: str = None,
                     cost_sensitivity: str = None,
                     performance_preference: str = None):
        """Update agent configuration"""
        if available_models:
            self.available_models = available_models
            self.agent_info['available_models_count'] = len(available_models)
        if default_model:
            self.default_model = default_model
            self.agent_info['default_model'] = default_model
        if cost_sensitivity:
            self.cost_sensitivity = cost_sensitivity
            self.agent_info['cost_sensitivity'] = cost_sensitivity
        if performance_preference:
            self.performance_preference = performance_preference
            self.agent_info['performance_preference'] = performance_preference
    
    def get_agent_info(self) -> Dict:
        """Get agent information and capabilities"""
        return self.agent_info.copy()
    
    def get_available_models(self) -> Dict:
        """Get available models configuration"""
        return self.available_models.copy()
    
    def add_model(self, model_id: str, model_info: Dict):
        """Add a new model to available models"""
        self.available_models[model_id] = model_info
        self.agent_info['available_models_count'] = len(self.available_models)
    
    def remove_model(self, model_id: str):
        """Remove a model from available models"""
        if model_id in self.available_models and model_id != self.default_model:
            del self.available_models[model_id]
            self.agent_info['available_models_count'] = len(self.available_models)
    
    def get_selection_history(self) -> List[Dict]:
        """Get model selection history"""
        return self.selection_history.copy()
    
    def clear_selection_history(self):
        """Clear selection history"""
        self.selection_history = []
