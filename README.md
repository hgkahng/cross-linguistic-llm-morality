# Cross-Linguistic LLM Morality

Investigating cross-linguistic moral preferences in Large Language Models (LLMs) using distributive justice scenarios.

## Overview

This project examines how LLMs make moral decisions in distributive justice contexts, comparing responses across:
- **Languages**: English and Korean
- **Conditions**: Baseline (no persona) vs. persona-injected
- **Domains**: 10 professional domains (economics, law, philosophy, etc.)

The experimental design uses 6 distributive justice scenarios adapted from behavioral economics literature (Charness & Rabin, 2002).

## Experimental Design

### Conditions

| Condition | Description | Sample Size |
|-----------|-------------|-------------|
| **Baseline** | LLM responds without persona context | 100 repetitions × 6 scenarios × 2 languages |
| **Persona-injected** | LLM responds as domain expert | 10,000 personas × 6 scenarios × 2 languages |

### Distributive Justice Scenarios

Each scenario presents a binary choice between two monetary allocations for Person A and Person B. The LLM plays the role of Person B.

| # | Name | Left (A, B) | Right (A, B) |
|---|------|-------------|--------------|
| 1 | Berk29 | ($400, $400) | ($750, $400) |
| 2 | Berk26 | ($0, $800) | ($400, $400) |
| 3 | Berk23 | ($800, $200) | ($0, $0) |
| 4 | Berk15 | ($200, $700) | ($600, $600) |
| 5 | Barc8 | ($300, $600) | ($700, $500) |
| 6 | Barc2 | ($400, $400) | ($750, $375) |

### Persona Domains

10,000 personas sampled from [PersonaHub](https://huggingface.co/datasets/proj-persona/PersonaHub) across 10 domains:

- Economics
- Law
- Philosophy
- History
- Sociology
- Environmental Science
- Mathematics
- Finance
- Engineering
- Computer Science

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/cross-linguistic-llm-morality.git
cd cross-linguistic-llm-morality

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Requirements

- Python 3.9+
- Google API key (for Gemini 2.0 Flash)

## Usage

### 1. Sample Personas

Sample 10,000 personas from PersonaHub (1,000 per domain):

```bash
python sample_elite_personas.py --quota 1000 --output data/personas/en/
```

### 2. Translate Personas to Korean

```bash
# Initial translation
python src/translate_personas.py \
    --input data/personas/en/ \
    --output data/personas/kr_draft/

# Verification pipeline
python src/verify_translations.py \
    --input data/personas/kr_draft/ \
    --original data/personas/en/ \
    --output data/personas/kr/
```

### 3. Generate Responses

**Baseline condition:**

```bash
python -m src.generate_responses \
    --condition baseline \
    --language en \
    --repetitions 100 \
    --output-dir data/responses/baseline/en/
```

**Persona-injected condition:**

```bash
python -m src.generate_responses \
    --condition persona \
    --language en \
    --personas-dir data/personas/en/ \
    --output-dir data/responses/persona/en/
```

**Test with a subset:**

```bash
# Sample 10 personas per domain for testing
python -m src.generate_responses \
    --condition persona \
    --language en \
    --personas-dir data/personas/en/ \
    --output-dir data/responses/persona/en_test/ \
    --personas-per-domain 10
```

## Project Structure

```
cross-linguistic-llm-morality/
├── src/
│   ├── scenarios.py          # Scenario definitions
│   ├── prompt_templates.py   # EN/KR prompt templates
│   ├── schema.py             # Pydantic response schemas
│   ├── generate_responses.py # Main generation script
│   ├── translate_personas.py # Translation pipeline
│   ├── verify_translations.py
│   └── backtranslate_verify.py
├── data/
│   ├── personas/
│   │   ├── en/               # English personas (10 domains)
│   │   └── kr/               # Korean personas (translated)
│   └── responses/
│       ├── baseline/         # Baseline responses
│       │   ├── en/
│       │   └── kr/
│       └── persona/          # Persona-injected responses
│           ├── en/
│           └── kr/
├── sample_elite_personas.py
├── requirements.txt
└── README.md
```

## Data Format

### Persona File (JSONL)

```json
{"domain": "economics", "persona": "An economist specializing in behavioral economics..."}
```

### Baseline Response

```json
{
  "scenario_id": 1,
  "metric": "Berk29",
  "options": {
    "left": {"A": 400, "B": 400},
    "right": {"A": 750, "B": 400}
  },
  "repetition": 1,
  "thought": "reasoning...",
  "answer": "Right"
}
```

### Persona-Injected Response

```json
{
  "scenario_1": {
    "persona_id": 1,
    "persona_desc": "An economist...",
    "metric": "Berk29",
    "options": {...},
    "thought": "reasoning...",
    "answer": "Left"
  },
  "scenario_2": {...}
}
```

## Model Configuration

- **Model**: Gemini 2.0 Flash (`gemini-2.0-flash`)
- **Temperature**: 1.0 (enables response variation)
- **Output**: Structured JSON via Pydantic validation

## References

### Data Sources

- **PersonaHub**: Ge et al. (2024). "Scaling Synthetic Data Creation with 1,000,000,000 Personas." [arXiv:2406.20094](https://arxiv.org/abs/2406.20094)

### Methodological Foundations

- Charness, G., & Rabin, M. (2002). Understanding social preferences with simple tests. *Quarterly Journal of Economics*, 117(3), 817-869.
- Horton, J. J. (2023). Large language models as simulated economic agents: What can we learn from homo silicus? *NBER Working Paper*.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this code or data in your research, please cite:

```bibtex
@misc{jang2025crosslinguistic,
  author = {Seongyu Jang, Chaewon Jeong, Jimin Kim, Hyungu Kahng},
  title = {Cross-Linguistic LLM Morality},
  year = {2025},
  url = {https://github.com/hgkahng/cross-linguistic-llm-morality}
}
```
