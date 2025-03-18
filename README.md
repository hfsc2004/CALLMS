# CALLMS (Complexity Analytics for LLM Scoring)

## Overview

CALLMS (Complexity Analytics for LLM Scoring) is an open-source framework designed to evaluate and classify the complexity of user queries in real-time before they are processed by a Large Language Model (LLM). By analyzing sentence structure, token density, and semantic depth, CALLMS provides a dynamic complexity score that enables AI systems to optimize response depth, ensuring more structured and contextually aware answers when needed.

This project is part of a broader initiative to enhance AI reasoning by introducing **complexity-aware processing**, allowing AI models to determine when deeper computational thought is required versus when a lightweight response is sufficient.

---

## Key Features

- **Real-Time Complexity Scoring** – Evaluates the complexity of user input before sending it to an LLM.
- **Token & Syntax Analysis** – Examines token density, sentence structure, and clause depth.
- **Semantic Depth Measurement** – Uses NLP-based embeddings to compare queries against known complex question datasets.
- **Adaptive Query Handling** – Enables AI models to decide whether to trigger additional recursive thought processing (such as with Deep Thought Systems).
- **Lightweight & Efficient** – Designed to run locally with minimal resource overhead.
- **Privacy-Focused** – Ensures that query analysis happens entirely on-device, with no external API calls.

---

## Installation

### Prerequisites

Before installing CALLMS, ensure that your system meets the following requirements:

- **Python 3.8+**
- **pip package manager**
- **Optional:** SpaCy NLP models for advanced syntax analysis

### Install via GitHub

```bash
# Clone the repository
git clone https://github.com/hfsc2004/CALLMS.git
cd CALLMS

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Basic Complexity Scoring

CALLMS provides a simple way to analyze query complexity:

```python
from callms import ComplexityAnalyzer

analyzer = ComplexityAnalyzer()
query = "What are the implications of quantum entanglement on causality?"
score = analyzer.get_complexity_score(query)

print(f"Complexity Score: {score}")
```

### Complexity Score Levels

- **0 - 30:** Simple (Short, straightforward queries)
- **31 - 60:** Moderate (Multi-part or layered questions)
- **61 - 100:** Complex (Highly nuanced, deep reasoning required)

---

## Configuration & Customization

CALLMS allows customization of its complexity evaluation parameters. Users can configure:

- **Token weight scaling**
- **Syntax parsing depth**
- **Semantic similarity thresholds**

Modify `config.json` to tweak scoring weights:

```json
{
  "token_weight": 1.2,
  "syntax_depth_weight": 1.5,
  "semantic_similarity_threshold": 0.85
}
```

---

## Contributing

We welcome community contributions! To get started:

1. **Fork the repository**
2. **Create a new feature branch**
3. **Submit a pull request (PR)**

Guidelines:

- Follow **PEP 8** for Python code style.
- Ensure proper documentation of all functions.
- Write tests where applicable.

---

## License

CALLMS is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**. This means you are free to:

- **Share** – Copy and redistribute the material in any medium or format.
- **Adapt** – Remix, transform, and build upon the material.

**However, you may not use CALLMS for commercial purposes without explicit permission.** See the full license details [here](https://creativecommons.org/licenses/by-nc/4.0/).

---

## 📞 Contact & Support

For questions, discussions, or feature requests, please open an issue on GitHub or contact the maintainers directly.

📌 **Repository:** [GitHub - CALLMS](https://github.com/hfsc2004/CALLMS) 📧 **Email:** [support@pseudoscifi.com](mailto:support@pseudoscifi.com)
