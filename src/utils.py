"""
Utility functions for response generation.

Includes JSON validation, rate limiting, retry logic, and file I/O helpers.
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from functools import wraps


logger = logging.getLogger(__name__)


# Expected response format
EXPECTED_KEYS = {"thought", "answer"}
VALID_ANSWERS = {"Left", "Right", "왼쪽", "오른쪽"}


def validate_response(response: Dict[str, Any], language: str = "en") -> tuple[bool, Optional[str]]:
    """
    Validate LLM response format and content.

    Args:
        response: Parsed JSON response from LLM
        language: Language code ("en" or "kr")

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if response is valid
        - error_message: Description of validation error, or None if valid
    """
    # Check if response is a dictionary
    if not isinstance(response, dict):
        return False, "Response is not a dictionary"

    # Check for required keys
    if not EXPECTED_KEYS.issubset(response.keys()):
        missing_keys = EXPECTED_KEYS - response.keys()
        return False, f"Missing required keys: {missing_keys}"

    # Validate 'thought' field
    thought = response.get("thought")
    if not isinstance(thought, str) or not thought.strip():
        return False, "'thought' must be a non-empty string"

    # Validate 'answer' field
    answer = response.get("answer")
    if not isinstance(answer, str):
        return False, "'answer' must be a string"

    # Normalize answer for validation
    answer_normalized = answer.strip()

    # Check if answer is valid
    if answer_normalized not in VALID_ANSWERS:
        return False, f"Invalid answer '{answer}'. Must be 'Left', 'Right', '왼쪽', or '오른쪽'"

    return True, None


def parse_json_response(raw_response: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from LLM response, handling potential formatting issues.

    Args:
        raw_response: Raw string response from LLM

    Returns:
        Parsed JSON dictionary, or None if parsing fails
    """
    # Try direct JSON parsing
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code blocks
    if "```json" in raw_response:
        try:
            start = raw_response.find("```json") + 7
            end = raw_response.find("```", start)
            json_str = raw_response[start:end].strip()
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

    # Try extracting JSON between curly braces
    try:
        start = raw_response.find("{")
        end = raw_response.rfind("}") + 1
        if start != -1 and end > start:
            json_str = raw_response[start:end]
            return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator to retry a function on failure with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch

    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed.")

            raise last_exception

        return wrapper
    return decorator


class RateLimiter:
    """
    Simple rate limiter to control API request frequency.

    Uses a sliding window approach to enforce rate limits.
    """

    def __init__(self, calls_per_minute: int = 60):
        """
        Initialize rate limiter.

        Args:
            calls_per_minute: Maximum number of calls allowed per minute
        """
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call_time = 0.0

    def wait_if_needed(self):
        """Wait if necessary to maintain rate limit."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time

        if time_since_last_call < self.min_interval:
            sleep_time = self.min_interval - time_since_last_call
            time.sleep(sleep_time)

        self.last_call_time = time.time()


def save_response_to_file(
    response_data: Dict[str, Any],
    output_path: Path,
    mode: str = "w",
):
    """
    Save response data to JSON file.

    Args:
        response_data: Response data to save
        output_path: Path to output file
        mode: File open mode ("w" for write, "a" for append)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, mode, encoding="utf-8") as f:
        json.dump(response_data, f, ensure_ascii=False, indent=2)
        if mode == "a":
            f.write("\n")


def load_personas_from_file(file_path: Path) -> list[Dict[str, Any]]:
    """
    Load personas from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of persona dictionaries
    """
    personas = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                persona = json.loads(line.strip())
                personas.append(persona)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
                continue

    return personas


def normalize_answer(answer: str, target_language: str = "en") -> str:
    """
    Normalize answer to standard format.

    Args:
        answer: Raw answer string
        target_language: Target language for normalization ("en" or "kr")

    Returns:
        Normalized answer ("Left"/"Right" for English, "왼쪽"/"오른쪽" for Korean)
    """
    answer_lower = answer.strip().lower()

    # English normalization
    if target_language == "en":
        if answer_lower in ["left", "왼쪽"]:
            return "Left"
        elif answer_lower in ["right", "오른쪽"]:
            return "Right"

    # Korean normalization
    elif target_language == "kr":
        if answer_lower in ["left", "왼쪽"]:
            return "왼쪽"
        elif answer_lower in ["right", "오른쪽"]:
            return "오른쪽"

    # Return original if no match
    return answer.strip()


if __name__ == "__main__":
    # Test validation
    print("Testing validation functions...")
    print("=" * 70)

    # Valid response
    valid_response = {"thought": "I prefer equality.", "answer": "Left"}
    is_valid, error = validate_response(valid_response)
    print(f"Valid response: {is_valid} (error: {error})")

    # Invalid response - missing key
    invalid_response = {"thought": "Missing answer key"}
    is_valid, error = validate_response(invalid_response)
    print(f"Invalid response (missing key): {is_valid} (error: {error})")

    # Invalid response - bad answer
    invalid_response2 = {"thought": "Test", "answer": "Maybe"}
    is_valid, error = validate_response(invalid_response2)
    print(f"Invalid response (bad answer): {is_valid} (error: {error})")

    print("\n" + "=" * 70)
    print("Testing JSON parsing...")

    # Test with code block
    markdown_response = '''Here is my response:
```json
{
  "thought": "This is a test",
  "answer": "Left"
}
```
'''
    parsed = parse_json_response(markdown_response)
    print(f"Parsed from markdown: {parsed}")

    # Test rate limiter
    print("\n" + "=" * 70)
    print("Testing rate limiter (2 calls/minute)...")
    limiter = RateLimiter(calls_per_minute=2)

    for i in range(3):
        start = time.time()
        limiter.wait_if_needed()
        elapsed = time.time() - start
        print(f"Call {i+1}: waited {elapsed:.2f}s")
