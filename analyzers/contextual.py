
import re
from typing import Dict, Any, List, Optional
from .base import BaseAnalyzer

class ContextualAnalyzer(BaseAnalyzer):
    """Analyzes contextual dependency complexity"""
    
    def analyze(self, prompt: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        features = {}
        
        # Pronoun analysis
        features['pronouns'] = self._count_pronouns(prompt)
        features['pronoun_density'] = features['pronouns'] / max(1, len(prompt.split()))
        
        # Underspecified elements
        features['vague_verbs'] = self._detect_vague_verbs(prompt)
        features['implicit_references'] = self._detect_implicit_references(prompt)
        
        # Hedging and politeness markers
        features['hedging_markers'] = self._detect_hedging(prompt)
        
        # Multi-topic detection
        features['topic_shifts'] = self._detect_topic_shifts(prompt)
        
        # Context dependency
        if context:
            features['context_references'] = self._analyze_context_dependency(prompt, context)
        else:
            features['context_references'] = 0
        
        # Calculate composite score
        score = self._calculate_contextual_score(features)
        
        return {
            'score': min(10.0, max(0.0, score)),
            'features': features,
            'reasoning': self._generate_reasoning(features, score)
        }
    
    def _count_pronouns(self, text: str) -> int:
        pronouns = r'\b(he|she|it|they|this|that|these|those|which|what|who)\b'
        return len(re.findall(pronouns, text.lower()))
    
    def _detect_vague_verbs(self, text: str) -> List[str]:
        vague_verbs = {
            'do', 'make', 'get', 'have', 'take', 'give', 'put',
            'go', 'come', 'see', 'look', 'find', 'use', 'work'
        }
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word in vague_verbs]
    
    def _detect_implicit_references(self, text: str) -> List[str]:
        patterns = [
            r'\bthe\s+above\b',
            r'\bthe\s+following\b',
            r'\bthe\s+previous\b',
            r'\bas\s+mentioned\b',
            r'\bthe\s+thing\b',
            r'\bthe\s+stuff\b',
            r'\bthe\s+issue\b'
        ]
        found = []
        for pattern in patterns:
            found.extend(re.findall(pattern, text.lower()))
        return found
    
    def _detect_hedging(self, text: str) -> List[str]:
        patterns = [
            r'could\s+you\s+possibly',
            r'if\s+you\s+don\'?t\s+mind',
            r'would\s+it\s+be\s+possible',
            r'i\s+was\s+wondering',
            r'perhaps\s+you\s+could',
            r'maybe\s+you\s+can',
            r'sort\s+of',
            r'kind\s+of',
            r'i\s+think\s+maybe'
        ]
        found = []
        for pattern in patterns:
            found.extend(re.findall(pattern, text.lower()))
        return found
    
    def _detect_topic_shifts(self, text: str) -> int:
        shift_markers = [
            'also', 'additionally', 'furthermore', 'moreover', 'by the way',
            'incidentally', 'separately', 'on another note', 'while we\'re at it'
        ]
        return sum(len(re.findall(r'\b' + re.escape(marker) + r'\b', text.lower())) for marker in shift_markers)
    
    def _analyze_context_dependency(self, prompt: str, context: List[str]) -> int:
        context_text = ' '.join(context[-3:]).lower()
        prompt_lower = prompt.lower()
        dependency_score = 0
        if any(word in context_text for word in prompt_lower.split()):
            dependency_score += 1
        continuation_patterns = [
            r'\bcontinue\b', r'\bkeep\s+going\b', r'\bmore\s+on\b',
            r'\belaborate\b', r'\bexpand\b', r'\bfollow\s+up\b'
        ]
        for pattern in continuation_patterns:
            if re.search(pattern, prompt_lower):
                dependency_score += 1
        return dependency_score
    
    def _calculate_contextual_score(self, features: Dict[str, Any]) -> float:
        score = 0.0
        if features['pronoun_density'] > 0.1:
            score += 3
        elif features['pronoun_density'] > 0.05:
            score += 2
        elif features['pronoun_density'] > 0.02:
            score += 1
        score += min(2, len(features['vague_verbs']) * 0.2)
        score += min(2, len(features['implicit_references']))
        score += min(2, len(features['hedging_markers']) * 0.5)
        if features['context_references'] > 0:
            score += 1
        return score
    
    def _generate_reasoning(self, features: Dict[str, Any], score: float) -> str:
        reasons = []
        if features['pronoun_density'] > 0.05:
            reasons.append(f"High pronoun density ({features['pronoun_density']:.3f})")
        if len(features['vague_verbs']) > 5:
            reasons.append("Contains many vague verbs")
        if len(features['implicit_references']) > 0:
            reasons.append("Contains implicit references")
        if features['context_references'] > 0:
            reasons.append("Depends on conversation context")
        return "; ".join(reasons) if reasons else "Low contextual dependency"
