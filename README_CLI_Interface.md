
# CALLMS v1.1 – Complexity Scoring for LLM Prompts

CALLMS (Complexity Analytics for LLM Scoring) is a Python package that evaluates the complexity of natural language prompts across multiple dimensions (contextual, task, etc.) to assist in routing, throttling, or evaluating LLM input.

---

## Installation

```bash
git clone https://github.com/hfsc2004/CALLMS.git
cd CALLMS
python3 -m venv venv
source venv/bin/activate
```

>  Dependencies are minimal. If needed:
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
=== CALLMS Complexity Analysis ===
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
