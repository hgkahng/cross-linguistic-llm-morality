"""
Prompt templates for distributive justice scenarios.

Supports baseline (no persona) and persona-injected conditions in English and Korean.
"""

from typing import Literal


Language = Literal["en", "kr"]
Condition = Literal["baseline", "persona"]


# English Templates
BASELINE_TEMPLATE_EN = """You are Person B in a distributive justice scenario.

Person A and B (you) must allocate money between themselves. You, as Person B, choose between:
- Left: A receives ${A_left}, B receives ${B_left}
- Right: A receives ${A_right}, B receives ${B_right}

Please explain your reasoning and then state your choice as either "Left" or "Right".

Return your response in the following JSON format:
{{
  "thought": "your reasoning here",
  "answer": "Left or Right"
}}"""


PERSONA_TEMPLATE_EN = """{persona_description}

You are Person B in a distributive justice scenario.

Person A and B (you) must allocate money between themselves. You, as Person B, choose between:
- Left: A receives ${A_left}, B receives ${B_left}
- Right: A receives ${A_right}, B receives ${B_right}

Please explain your reasoning and then state your choice as either "Left" or "Right".

Return your response in the following JSON format:
{{
  "thought": "your reasoning here",
  "answer": "Left or Right"
}}"""


# Korean Templates
BASELINE_TEMPLATE_KR = """당신은 분배 정의 시나리오에서 사람 B입니다.

사람 A와 B(당신)는 둘 사이에 돈을 배분해야 합니다. 당신은 사람 B로서 다음 중 하나를 선택합니다:
- 왼쪽: A는 ${A_left}를 받고, B는 ${B_left}를 받습니다
- 오른쪽: A는 ${A_right}를 받고, B는 ${B_right}를 받습니다

당신의 추론을 설명한 다음 "왼쪽" 또는 "오른쪽"으로 선택을 명시하십시오.

다음 JSON 형식으로 응답을 반환하십시오:
{{
  "thought": "여기에 당신의 추론",
  "answer": "왼쪽 또는 오른쪽"
}}"""


PERSONA_TEMPLATE_KR = """{persona_description}

당신은 분배 정의 시나리오에서 사람 B입니다.

사람 A와 B(당신)는 둘 사이에 돈을 배분해야 합니다. 당신은 사람 B로서 다음 중 하나를 선택합니다:
- 왼쪽: A는 ${A_left}를 받고, B는 ${B_left}를 받습니다
- 오른쪽: A는 ${A_right}를 받고, B는 ${B_right}를 받습니다

당신의 추론을 설명한 다음 "왼쪽" 또는 "오른쪽"으로 선택을 명시하십시오.

다음 JSON 형식으로 응답을 반환하십시오:
{{
  "thought": "여기에 당신의 추론",
  "answer": "왼쪽 또는 오른쪽"
}}"""


# Translation Pipeline Templates (Stages 1-4)

# Stage 1 & 3: English to Korean Translation
TRANSLATION_PROMPT_TEMPLATE = """Translate the following English persona description to Korean with particular attention to:
1. Semantic accuracy - preserve exact meaning and nuance
2. Domain terminology - use correct professional terminology
3. Natural Korean formal register - ensure native-like expression
4. Professional framing - maintain disciplinary perspective

English persona:
{persona_description}

Provide only the Korean translation, ensuring it reads naturally while preserving all technical details."""


# Stage 2-3: Translation Verification
VERIFICATION_PROMPT_TEMPLATE = """Evaluate this Korean translation of an English persona description.

Original English:
{english_persona}

Korean Translation:
{korean_persona}

Evaluate on these criteria:
1. Semantic accuracy: Does the Korean version preserve the meaning of the English?
2. Domain terminology: Are professional terms correctly translated and appropriate?
3. Korean naturalness: Does it read naturally in formal Korean?
4. Professional framing: Is the disciplinary perspective preserved?

Respond in JSON format with:
- "accept": true/false - whether to accept this translation
- "justification": Brief explanation of your decision (1-2 sentences)
- "criteria_scores": Object with scores 1-5 for each criterion:
  - "semantic_accuracy": 1-5
  - "domain_terminology": 1-5
  - "naturalness": 1-5
  - "professional_framing": 1-5
"""


# Stage 4: Korean to English Back-Translation
BACKTRANSLATION_PROMPT_TEMPLATE = """Translate the following Korean persona description back to English, preserving the meaning and professional context:

Korean:
{korean_persona}

Provide only the English translation."""


# Stage 4: Equivalence Check
EQUIVALENCE_CHECK_PROMPT_TEMPLATE = """Compare these two English persona descriptions for semantic equivalence:

Original English:
{original_english}

Back-translated English (from Korean):
{backtranslated_english}

Do these descriptions convey the same professional identity, expertise, and perspective?
Evaluate whether the Korean translation preserved the essential meaning.

Consider:
- Do they describe the same professional role and expertise?
- Are the key details and specializations preserved?
- Is the disciplinary framing consistent?

Respond in JSON format with:
- "equivalent": true/false - whether descriptions are semantically equivalent
- "divergence_explanation": Explain any meaningful differences (or "None" if equivalent)
- "severity": "none", "minor", or "major"
  - "none": No meaningful divergence
  - "minor": Small differences that don't change core meaning
  - "major": Significant differences that alter professional identity or expertise
"""


def get_template(language: Language, condition: Condition) -> str:
    """
    Get prompt template for given language and condition.

    Args:
        language: Language code ("en" or "kr")
        condition: Experimental condition ("baseline" or "persona")

    Returns:
        Prompt template string

    Raises:
        ValueError: If language or condition is invalid
    """
    templates = {
        ("en", "baseline"): BASELINE_TEMPLATE_EN,
        ("en", "persona"): PERSONA_TEMPLATE_EN,
        ("kr", "baseline"): BASELINE_TEMPLATE_KR,
        ("kr", "persona"): PERSONA_TEMPLATE_KR,
    }

    key = (language, condition)
    if key not in templates:
        raise ValueError(f"Invalid language '{language}' or condition '{condition}'")

    return templates[key]


def format_prompt(
    language: Language,
    condition: Condition,
    A_left: int,
    B_left: int,
    A_right: int,
    B_right: int,
    persona_description: str = None,
) -> str:
    """
    Format prompt with scenario values.

    Args:
        language: Language code ("en" or "kr")
        condition: Experimental condition ("baseline" or "persona")
        A_left: Amount A receives in left option
        B_left: Amount B receives in left option
        A_right: Amount A receives in right option
        B_right: Amount B receives in right option
        persona_description: Persona description (required if condition="persona")

    Returns:
        Formatted prompt string

    Raises:
        ValueError: If persona_description is required but not provided
    """
    if condition == "persona" and not persona_description:
        raise ValueError("persona_description is required for persona condition")

    template = get_template(language, condition)

    # Format template
    prompt = template.format(
        A_left=A_left,
        B_left=B_left,
        A_right=A_right,
        B_right=B_right,
        persona_description=persona_description or "",
    )

    return prompt


if __name__ == "__main__":
    # Test prompt generation
    print("=" * 70)
    print("BASELINE PROMPT (English)")
    print("=" * 70)
    print(format_prompt("en", "baseline", 400, 400, 750, 400))

    print("\n" + "=" * 70)
    print("PERSONA PROMPT (English)")
    print("=" * 70)
    test_persona = "You are a professor of economics specializing in behavioral economics and decision theory."
    print(format_prompt("en", "persona", 400, 400, 750, 400, persona_description=test_persona))

    print("\n" + "=" * 70)
    print("BASELINE PROMPT (Korean)")
    print("=" * 70)
    print(format_prompt("kr", "baseline", 400, 400, 750, 400))

    print("\n" + "=" * 70)
    print("PERSONA PROMPT (Korean)")
    print("=" * 70)
    test_persona_kr = "당신은 행동경제학과 의사결정 이론을 전문으로 하는 경제학 교수입니다."
    print(format_prompt("kr", "persona", 400, 400, 750, 400, persona_description=test_persona_kr))
