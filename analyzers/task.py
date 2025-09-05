
import re
from typing import Dict, Any, List, Optional
from .base import BaseAnalyzer

class TaskAnalyzer(BaseAnalyzer):
    """Analyzes task/computational complexity"""
    
    def analyze(self, prompt: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        features = {}
        
        # Task type classification
        features['task_type'] = self._classify_task_type(prompt)
        features['computational_indicators'] = self._detect_computational_tasks(prompt)
        features['creative_indicators'] = self._detect_creative_tasks(prompt)
        features['analytical_indicators'] = self._detect_analytical_tasks(prompt)
        
        # Multi-modal requirements
        features['multimodal_requirements'] = self._detect_multimodal_needs(prompt)
        
        # Output format complexity
        features['output_format'] = self._analyze_output_requirements(prompt)
        
        # Calculate composite score
        score = self._calculate_task_score(features)
        
        return {
            'score': min(10.0, max(0.0, score)),
            'features': features,
            'reasoning': self._generate_reasoning(features, score)
        }
    
    def _classify_task_type(self, text: str) -> str:
        text_lower = text.lower()
        if any(word in text_lower for word in ['calculate', 'compute', 'solve', 'algorithm', 'optimize']):
            return 'computational'
        elif any(word in text_lower for word in ['write', 'create', 'generate', 'compose', 'design']):
            return 'creative'
        elif any(word in text_lower for word in ['analyze', 'compare', 'evaluate', 'assess', 'critique']):
            return 'analytical'
        elif any(word in text_lower for word in ['what is', 'define', 'explain', 'describe', 'tell me']):
            return 'informational'
        elif any(word in text_lower for word in ['summarize', 'combine', 'integrate', 'synthesize']):
            return 'synthesis'
        else:
            return 'general'
    
    def _detect_computational_tasks(self, text: str) -> List[str]:
        return re.findall(r'\b(calculate|compute|solve|optimize|equation)\b', text.lower())
    
    def _detect_creative_tasks(self, text: str) -> List[str]:
        return re.findall(r'\b(write|generate|create|design|draw|paint|compose)\b', text.lower())
    
    def _detect_analytical_tasks(self, text: str) -> List[str]:
        return re.findall(r'\b(analyze|compare|evaluate|assess|critique|diagnose)\b', text.lower())
    
    def _detect_multimodal_needs(self, text: str) -> List[str]:
        return re.findall(r'\b(chart|graph|diagram|visualize|image|plot|table)\b', text.lower())
    
    def _analyze_output_requirements(self, text: str) -> List[str]:
        return re.findall(r'\b(json|csv|markdown|table|pdf|code block|formatted)\b', text.lower())
    
    def _calculate_task_score(self, features: Dict[str, Any]) -> float:
        score = 0.0
        if features['task_type'] in ['computational', 'creative', 'analytical']:
            score += 3
        score += min(2, len(features['computational_indicators']) * 0.5)
        score += min(2, len(features['creative_indicators']) * 0.5)
        score += min(2, len(features['analytical_indicators']) * 0.5)
        score += min(1, len(features['multimodal_requirements']) * 0.25)
        score += min(1, len(features['output_format']) * 0.25)
        return score
    
    def _generate_reasoning(self, features: Dict[str, Any], score: float) -> str:
        reasons = [f"Task type: {features['task_type']}"]
        if features['computational_indicators']:
            reasons.append("Includes computational tasks")
        if features['creative_indicators']:
            reasons.append("Includes creative tasks")
        if features['analytical_indicators']:
            reasons.append("Includes analytical tasks")
        if features['multimodal_requirements']:
            reasons.append("Requires visual or structured output")
        if features['output_format']:
            reasons.append("Has specific output format requirements")
        return "; ".join(reasons)
