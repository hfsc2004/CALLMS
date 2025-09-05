# prompt_complexity/cli.py
import argparse
import json
import sys
from typing import Optional
from .core import PromptAnalyzer
from .config import ComplexityConfig

def main():
    """Command line interface for prompt complexity analysis"""
    parser = argparse.ArgumentParser(description="Analyze LLM prompt complexity")
    
    parser.add_argument("prompt", nargs="?", help="Prompt to analyze (or use --file)")
    parser.add_argument("--file", "-f", help="Read prompt from file")
    parser.add_argument("--output", "-o", choices=["json", "text"], default="text", 
                       help="Output format")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--weights", help="Custom dimension weights (comma-separated)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Get prompt text
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.prompt:
        prompt = args.prompt
    else:
        # Read from stdin
        prompt = sys.stdin.read().strip()
    
    if not prompt:
        print("Error: No prompt provided", file=sys.stderr)
        sys.exit(1)
    
    # Load configuration
    config = ComplexityConfig()
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
                # Update config with loaded values
                for key, value in config_dict.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Parse custom weights
    if args.weights:
        try:
            weights = [float(w) for w in args.weights.split(',')]
            if len(weights) == 5:
                config.dimension_weights = {
                    'structural': weights[0],
                    'semantic': weights[1],
                    'contextual': weights[2],
                    'task': weights[3],
                    'domain': weights[4]
                }
            else:
                print("Error: Must provide exactly 5 weights", file=sys.stderr)
                sys.exit(1)
        except ValueError:
            print("Error: Invalid weight values", file=sys.stderr)
            sys.exit(1)
    
    # Analyze prompt
    analyzer = PromptAnalyzer(config)
    result = analyzer.analyze(prompt)
    
    # Output results
    if args.output == "json":
        print(result.to_json())
    else:
        print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print(f"Overall Complexity Score: {result.overall_score:.2f}/10.0")
        print(f"Confidence: {result.confidence:.2f}")
        print("\nDimension Scores:")
        for dim, score in result.dimension_scores.items():
            print(f"  {dim.capitalize()}: {score:.2f}")
        
        if args.verbose:
            print("\nDetailed Analysis:")
            for dim, reasoning in result.reasoning.items():
                print(f"  {dim.capitalize()}: {reasoning}")

# prompt_complexity/utils.py
from typing import List, Dict, Tuple, Optional
import json
from .core import PromptAnalyzer, ComplexityResult

class BatchAnalyzer:
    """Utility for analyzing multiple prompts efficiently"""
    
    def __init__(self, analyzer: Optional[PromptAnalyzer] = None):
        self.analyzer = analyzer or PromptAnalyzer()
    
    def analyze_batch(self, prompts: List[str]) -> List[ComplexityResult]:
        """Analyze a batch of prompts"""
        results = []
        for prompt in prompts:
            result = self.analyzer.analyze(prompt)
            results.append(result)
        return results
    
    def analyze_from_file(self, filepath: str) -> List[ComplexityResult]:
        """Analyze prompts from a file (one per line)"""
        with open(filepath, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        return self.analyze_batch(prompts)
    
    def export_results(self, results: List[ComplexityResult], filepath: str, format: str = 'json'):
        """Export results to file"""
        if format == 'json':
            data = [result.to_dict() for result in results]
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        elif format == 'csv':
            import csv
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Header
                writer.writerow(['overall_score', 'structural', 'semantic', 'contextual', 'task', 'domain', 'confidence'])
                # Data
                for result in results:
                    row = [result.overall_score]
                    for dim in ['structural', 'semantic', 'contextual', 'task', 'domain']:
                        row.append(result.dimension_scores.get(dim, 0.0))
                    row.append(result.confidence)
                    writer.writerow(row)

class ComplexityRouter:
    """Route prompts to different models based on complexity"""
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.analyzer = PromptAnalyzer()
        self.thresholds = thresholds or {
            'simple': 3.0,
            'medium': 6.0,
            'complex': 8.0
        }
    
    def route_prompt(self, prompt: str) -> Tuple[str, ComplexityResult]:
        """
        Route prompt to appropriate model tier
        
        Returns:
            Tuple of (model_tier, complexity_result)
        """
        result = self.analyzer.analyze(prompt)
        score = result.overall_score
        
        if score <= self.thresholds['simple']:
            tier = 'simple'  # Use lightweight/local model
        elif score <= self.thresholds['medium']:
            tier = 'medium'  # Use standard cloud model
        elif score <= self.thresholds['complex']:
            tier = 'complex'  # Use advanced model
        else:
            tier = 'expert'   # Use most capable model
        
        return tier, result

# prompt_complexity/extensions.py
from typing import Optional, Dict, Any
from .analyzers.base import BaseAnalyzer

class LLMSelfRater(BaseAnalyzer):
    """Use an LLM to self-rate prompt complexity"""
    
    def __init__(self, config, llm_client=None):
        super().__init__(config)
        self.llm_client = llm_client
    
    def analyze(self, prompt: str, context: Optional[list] = None) -> Dict[str, Any]:
        """Use LLM to analyze its own prompt complexity"""
        if not self.llm_client:
            return {'score': 5.0, 'features': {'error': 'No LLM client provided'}}
        
        rating_prompt = f"""
        Rate the complexity of this prompt on a scale of 0-10, considering:
        - Structural complexity (grammar, vocabulary, length)
        - Semantic complexity (reasoning required, abstraction)
        - Contextual complexity (ambiguity, references)
        - Task complexity (computational requirements)
        - Domain complexity (specialized knowledge needed)
        
        Prompt: "{prompt}"
        
        Respond with just a number from 0-10 and brief reasoning.
        """
        
        try:
            response = self.llm_client.generate(rating_prompt)
            # Parse response to extract score
            score = self._extract_score(response)
            return {
                'score': score,
                'features': {'llm_response': response},
                'reasoning': f"LLM self-assessment: {response[:100]}..."
            }
        except Exception as e:
            return {'score': 5.0, 'features': {'error': str(e)}}
    
    def _extract_score(self, response: str) -> float:
        """Extract numerical score from LLM response"""
        import re
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)
        if numbers:
            score = float(numbers[0])
            return min(10.0, max(0.0, score))
        return 5.0

# prompt_complexity/benchmarks.py
"""Benchmarking utilities for complexity analysis"""

class ComplexityBenchmark:
    """Benchmark dataset and evaluation utilities"""
    
    BENCHMARK_PROMPTS = [
        # Simple prompts (expected score 0-3)
        ("What is 2+2?", 1.0),
        ("Define the word 'cat'", 1.5),
        ("What is the capital of France?", 1.2),
        
        # Medium prompts (expected score 3-6)
        ("Explain how photosynthesis works in plants", 4.0),
        ("Compare and contrast democracy and authoritarianism", 5.0),
        ("Write a short story about a time traveler", 4.5),
        
        # Complex prompts (expected score 6-9)
        ("Analyze the philosophical implications of quantum mechanics on free will", 7.5),
        ("Design an algorithm to optimize multi-objective machine learning models", 8.0),
        ("Synthesize current research on consciousness and propose a unified theory", 8.5),
        
        # Expert prompts (expected score 9-10)
        ("Develop a comprehensive framework integrating quantum field theory with consciousness studies while addressing the hard problem of phenomenology", 9.5),
        ("Create a novel mathematical proof demonstrating the relationship between Gödel's incompleteness theorems and computational complexity in recursive function theory", 9.8)
    ]
    
    def __init__(self, analyzer: Optional['PromptAnalyzer'] = None):
        from .core import PromptAnalyzer
        self.analyzer = analyzer or PromptAnalyzer()
    
    def run_benchmark(self) -> Dict[str, float]:# prompt_complexity/analyzers/contextual.py
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
        """Count ambiguous pronouns"""
        pronouns = r'\b(he|she|it|they|this|that|these|those|which|what|who)\b'
        return len(re.findall(pronouns, text.lower()))
    
    def _detect_vague_verbs(self, text: str) -> List[str]:
        """Detect vague or underspecified verbs"""
        vague_verbs = {
            'do', 'make', 'get', 'have', 'take', 'give', 'put',
            'go', 'come', 'see', 'look', 'find', 'use', 'work'
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word in vague_verbs]
    
    def _detect_implicit_references(self, text: str) -> List[str]:
        """Detect implicit or unclear references"""
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
            matches = re.findall(pattern, text.lower())
            found.extend(matches)
        
        return found
    
    def _detect_hedging(self, text: str) -> List[str]:
        """Detect hedging and politeness markers"""
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
            matches = re.findall(pattern, text.lower())
            found.extend(matches)
        
        return found
    
    def _detect_topic_shifts(self, text: str) -> int:
        """Detect abrupt topic changes"""
        shift_markers = [
            'also', 'additionally', 'furthermore', 'moreover', 'by the way',
            'incidentally', 'separately', 'on another note', 'while we\'re at it'
        ]
        
        count = 0
        for marker in shift_markers:
            count += len(re.findall(r'\b' + re.escape(marker) + r'\b', text.lower()))
        
        return count
    
    def _analyze_context_dependency(self, prompt: str, context: List[str]) -> int:
        """Analyze how much the prompt depends on conversation context"""
        context_text = ' '.join(context[-3:]).lower()  # Last 3 messages
        prompt_lower = prompt.lower()
        
        dependency_score = 0
        
        # Check for references to previous content
        if any(word in context_text for word in prompt_lower.split()):
            dependency_score += 1
        
        # Check for continuation phrases
        continuation_patterns = [
            r'\bcontinue\b', r'\bkeep\s+going\b', r'\bmore\s+on\b',
            r'\belaborate\b', r'\bexpand\b', r'\bfollow\s+up\b'
        ]
        
        for pattern in continuation_patterns:
            if re.search(pattern, prompt_lower):
                dependency_score += 1
        
        return dependency_score
    
    def _calculate_contextual_score(self, features: Dict[str, Any]) -> float:
        """Calculate contextual complexity score"""
        score = 0.0
        
        # Pronoun density (0-3 points)
        if features['pronoun_density'] > 0.1:
            score += 3
        elif features['pronoun_density'] > 0.05:
            score += 2
        elif features['pronoun_density'] > 0.02:
            score += 1
        
        # Vague verbs (0-2 points)
        score += min(2, len(features['vague_verbs']) * 0.2)
        
        # Implicit references (0-2 points)
        score += min(2, len(features['implicit_references']))
        
        # Hedging markers (0-2 points)
        score += min(2, len(features['hedging_markers']) * 0.5)
        
        # Context dependency (0-1 point)
        if features['context_references'] > 0:
            score += 1
        
        return score
    
    def _generate_reasoning(self, features: Dict[str, Any], score: float) -> str:
        """Generate human-readable reasoning"""
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

# prompt_complexity/analyzers/task.py
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
        """Classify the primary task type"""
        text_lower = text.lower()
        
        # Computational tasks
        if any(word in text_lower for word in ['calculate', 'compute', 'solve', 'algorithm', 'optimize']):
            return 'computational'
        
        # Creative tasks
        elif any(word in text_lower for word in ['write', 'create', 'generate', 'compose', 'design']):
            return 'creative'
        
        # Analytical tasks
        elif any(word in text_lower for word in ['analyze', 'compare', 'evaluate', 'assess', 'critique']):
            return 'analytical'
        
        # Information retrieval
        elif any(word in text_lower for word in ['what is', 'define', 'explain', 'describe', 'tell me']):
            return 'informational'
        
        # Synthesis tasks
        elif any(word in text_lower for word in ['summarize', 'combine', 'integrate', 'synthesize']):
            return 'synthesis'
        
        else:
            return 'general'
