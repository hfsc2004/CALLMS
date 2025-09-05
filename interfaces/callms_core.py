
from typing import Dict, Optional, List
from callms.analyzers.contextual import ContextualAnalyzer
from callms.analyzers.task import TaskAnalyzer

class CALLMS:
    def __init__(self):
        self.contextual_analyzer = ContextualAnalyzer()
        self.task_analyzer = TaskAnalyzer()

    def score_prompt(self, prompt: str, context: Optional[List[str]] = None) -> Dict[str, any]:
        contextual_result = self.contextual_analyzer.analyze(prompt, context)
        task_result = self.task_analyzer.analyze(prompt, context)

        # Composite score calculation
        weights = {
            'contextual': 0.5,
            'task': 0.5
        }
        final_score = (
            weights['contextual'] * contextual_result['score'] +
            weights['task'] * task_result['score']
        )

        return {
            'total_score': round(final_score, 2),
            'contextual_score': contextual_result['score'],
            'contextual_reasoning': contextual_result['reasoning'],
            'task_score': task_result['score'],
            'task_reasoning': task_result['reasoning'],
            'features': {
                'contextual': contextual_result['features'],
                'task': task_result['features']
            }
        }
