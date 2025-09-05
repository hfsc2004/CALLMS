
# CALLMS v1.1 – Complexity Scoring for LLM Prompts

CALLMS (Complexity Analytics for LLM Scoring) is a Python package that evaluates the complexity of natural language prompts across multiple dimensions (contextual, task, etc.) to assist in routing, throttling, or evaluating LLM input.

What are other practical applications of complexity scores?

Complexity scores—especially when applied to outputs from large language models (LLMs)—are surprisingly versatile. They go beyond academic curiosity and can power real-world decisions, automation, and personalization. Here’s a breakdown of practical applications across different domains:
 Education & Learning Platforms

    Adaptive tutoring: Tailor explanations based on the learner’s level. If a response is too complex, simplify it; if too basic, enrich it.

    Curriculum design: Automatically classify content into beginner, intermediate, or advanced tiers.

    Student assessment: Evaluate the sophistication of student-generated text (e.g., essays or answers) to gauge understanding.

 Business & Customer Support

    Response routing: Direct complex queries to human agents, while simpler ones are handled by bots.

    Tone calibration: Adjust the complexity of automated replies based on the customer’s profile or sentiment.

    Knowledge base optimization: Flag overly complex articles for rewriting or segmentation.

 AI Model Development

    Prompt engineering: Score outputs to refine prompts that yield appropriately complex responses.

    Model comparison: Benchmark models not just on accuracy, but on how nuanced or sophisticated their outputs are.

    Hallucination detection: Extremely high complexity might correlate with fabricated or incoherent responses—useful for filtering.

 Content Creation & Publishing

    Audience targeting: Match content complexity to the intended audience (e.g., technical vs. layperson).

    SEO optimization: Adjust complexity to improve readability scores and engagement metrics.

    Editorial review: Highlight sections that may need simplification or elaboration.

 Healthcare & Legal Tech

    Patient communication: Ensure medical explanations are not overly complex for patients.

    Legal document review: Flag clauses that are unnecessarily convoluted or ambiguous.

 Game Design & Interactive Media

    NPC dialogue tuning: Adjust the complexity of character responses based on player progress or skill level.

    Story branching: Use complexity scores to guide narrative depth and pacing.

If you’re building something with LLMs, complexity scoring can be a powerful lever for control, personalization, and quality assurance. Want help integrating it into a specific workflow or product idea?


Bottom Line: CALLMS is unique in its simplicity and specificity. If your goal is to trigger actions based on how complex or nuanced a model’s response is—not just whether it’s right—CALLMS is a rare and valuable tool.

---

## Installation

```bash
git clone https://github.com/hfsc2004/CALLMS.git
cd CALLMS
python3 -m venv venv
source venv/bin/activate
```

> Dependencies are minimal. If needed:
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

---

## Usage (CLI)

Run directly using the CLI interface:

```bash
python callms_cli.py --text "Could you explain the philosophy behind Pascal’s Wager?"
```

With prior messages (context):

```bash
python callms_cli.py --text "Can you elaborate on that?" \
  --context "Earlier we discussed religious game theory" \
            "You mentioned Pascal's Wager before"
```

---

## Output Example

```
=== CALLMS Complexity Analytics ===
Total Score: 7.6
Contextual Score: 6.5
→ High pronoun density; Contains vague verbs; Depends on conversation context
Task Score: 8.8
→ Task type: analytical; Includes analytical tasks; Requires structured output

Features:
  [contextual]
    pronouns: 3
    vague_verbs: ['do', 'make']
    ...
```

---

## Components

| Module               | Purpose                              |
|----------------------|--------------------------------------|
| `contextual.py`      | Detects ambiguity, pronouns, hedging |
| `task.py`            | Detects creative/computation tasks   |
| `callms_core.py`     | Combines analyzers into a score      |
| `callms_cli.py`      | CLI runner                           |

---

## TODO

- Add Structural + Semantic analyzers
- Add LLM tier fallback logic
- Publish as `pip` package
- Add GitHub Actions test suite

---

## License

CALLMS is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). This means you are free to:

    Share – Copy and redistribute the material in any medium or format.
    Adapt – Remix, transform, and build upon the material.

However, you may not use CALLMS for commercial purposes without explicit permission. See the full license details here.
https://creativecommons.org/licenses/by-nc/4.0/

---

## Contributors

- Aaron French
- Chad Marshall
- Elira
