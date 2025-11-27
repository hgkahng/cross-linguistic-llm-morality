"""
Pydantic schemas for structured output from LLM responses.

These schemas are used with LangChain's with_structured_output() method
to ensure responses match the expected format.
"""

from pydantic import BaseModel, Field
from typing import Literal


class DistributiveJusticeResponse(BaseModel):
    """
    Response schema for distributive justice scenarios.

    The LLM must provide reasoning and choose between Left or Right options.
    """
    thought: str = Field(
        description="Explanation of reasoning behind the choice"
    )
    answer: Literal["Left", "Right"] = Field(
        description="Choice between Left or Right option"
    )


class DistributiveJusticeResponseKR(BaseModel):
    """
    Response schema for distributive justice scenarios (Korean).

    The LLM must provide reasoning and choose between 왼쪽 or 오른쪽 options.
    """
    thought: str = Field(
        description="선택에 대한 추론 설명"
    )
    answer: Literal["왼쪽", "오른쪽"] = Field(
        description="왼쪽 또는 오른쪽 선택"
    )


def get_response_schema(language: str = "en"):
    """
    Get the appropriate response schema for the given language.

    Args:
        language: Language code ("en" or "kr")

    Returns:
        Pydantic model class for the response schema

    Raises:
        ValueError: If language is not supported
    """
    schemas = {
        "en": DistributiveJusticeResponse,
        "kr": DistributiveJusticeResponseKR,
    }

    if language not in schemas:
        raise ValueError(f"Unsupported language: {language}. Must be 'en' or 'kr'")

    return schemas[language]


if __name__ == "__main__":
    # Test schemas
    print("Testing Pydantic schemas...")
    print("=" * 70)

    # English schema
    print("\nEnglish Response Schema:")
    en_response = DistributiveJusticeResponse(
        thought="I prefer equality",
        answer="Left"
    )
    print(f"  Schema: {DistributiveJusticeResponse.model_json_schema()}")
    print(f"  Example: {en_response.model_dump()}")

    # Korean schema
    print("\nKorean Response Schema:")
    kr_response = DistributiveJusticeResponseKR(
        thought="평등을 선호합니다",
        answer="왼쪽"
    )
    print(f"  Schema: {DistributiveJusticeResponseKR.model_json_schema()}")
    print(f"  Example: {kr_response.model_dump()}")

    # Test validation
    print("\n" + "=" * 70)
    print("Testing validation...")
    try:
        invalid = DistributiveJusticeResponse(
            thought="Test",
            answer="Maybe"  # This should fail
        )
    except Exception as e:
        print(f"  ✓ Validation correctly rejected invalid answer: {type(e).__name__}")
