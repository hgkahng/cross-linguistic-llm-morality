"""
    Generate LLM responses for distributive justice scenarios.

    This script generates responses using Gemini 2.0 Flash via LangChain for:
    - Baseline condition (no persona): 100 repetitions per scenario
    - Persona-injected condition: 10 repetitions per persona-scenario pair

    Supports both English and Korean languages.
    Uses LangChain's with_structured_output() for automatic JSON parsing and validation.
"""

import logging
import argparse
import random
from pathlib import Path
from typing import Optional
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

try:
    # Try relative imports (when imported as a module)
    from .scenarios import SCENARIOS, format_scenario_prompt
    from .prompt_templates import format_prompt
    from .schema import get_response_schema
    from .utils import (
        RateLimiter,
        save_response_to_file,
        load_personas_from_file,
    )
except ImportError:
    # Fall back to absolute imports (when run as a script)
    from scenarios import SCENARIOS, format_scenario_prompt
    from prompt_templates import format_prompt
    from schema import get_response_schema
    from utils import (
        RateLimiter,
        save_response_to_file,
        load_personas_from_file,
    )


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def init_model(model_name: str = "gemini-2.0-flash", temperature: float = 1.0, language: str = "en"):
    """
    Initialize LangChain chat model with Gemini and structured output.

    Args:
        model_name: Gemini model name
        temperature: Sampling temperature
        language: Language code for response schema

    Returns:
        Initialized chat model with structured output
    """
    try:
        from langchain.chat_models import init_chat_model

        # Initialize base model
        base_model = init_chat_model(
            model=model_name,
            model_provider="google_genai",
            temperature=temperature,
        )

        # Get response schema for language
        response_schema = get_response_schema(language)

        # Bind structured output schema
        model = base_model.with_structured_output(response_schema)

        logger.info(f"Initialized model: {model_name} (temperature={temperature}, language={language})")
        logger.info(f"Using structured output with schema: {response_schema.__name__}")
        return model

    except ImportError:
        logger.error("langchain not installed. Install with: pip install langchain langchain-google-genai")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise


def generate_single_response(model, prompt: str) -> dict:
    """
    Generate a single response from the model using structured output.

    The model is already configured with structured output schema,
    so the response is automatically parsed and validated.

    Args:
        model: LangChain chat model with structured output
        prompt: Formatted prompt string

    Returns:
        Response dictionary with 'thought' and 'answer' keys
    """
    # Generate response - automatically parsed and validated by structured output
    response = model.invoke(prompt)

    # Convert Pydantic model to dictionary
    if hasattr(response, 'model_dump'):
        return response.model_dump()
    elif hasattr(response, 'dict'):
        return response.dict()
    else:
        # Fallback for dict-like responses
        return dict(response)


def generate_baseline_responses(
    model,
    language: str,
    repetitions: int,
    output_dir: Path,
    rate_limiter: Optional[RateLimiter] = None,
    batch_size: int = 10,
    max_concurrency: int = 5,
):
    """
    Generate baseline responses (no persona) for all scenarios.

    Args:
        model: LangChain chat model
        language: Language code ("en" or "kr")
        repetitions: Number of repetitions per scenario
        output_dir: Output directory for responses
        rate_limiter: Optional rate limiter
        batch_size: Number of requests to batch together
        max_concurrency: Maximum number of concurrent requests
    """
    logger.info(f"Generating baseline responses ({language})")
    logger.info(f"Scenarios: {len(SCENARIOS)}, Repetitions: {repetitions}")
    logger.info(f"Batch size: {batch_size}, Max concurrency: {max_concurrency}")

    output_dir.mkdir(parents=True, exist_ok=True)
    total_queries = len(SCENARIOS) * repetitions

    with tqdm(total=total_queries, desc=f"Baseline ({language})") as pbar:
        for scenario in SCENARIOS:
            scenario_id = scenario["scenario_id"]
            responses = []

            # Format scenario values
            scenario_values = format_scenario_prompt(scenario)

            # Generate prompt
            prompt = format_prompt(
                language=language,
                condition="baseline",
                **scenario_values,
            )

            # Generate repetitions in batches
            for batch_start in range(0, repetitions, batch_size):
                batch_end = min(batch_start + batch_size, repetitions)
                current_batch_size = batch_end - batch_start

                try:
                    if rate_limiter:
                        rate_limiter.wait_if_needed()

                    # Create batch of identical prompts
                    prompts = [prompt] * current_batch_size

                    # Batch process
                    batch_responses = model.batch(
                        prompts,
                        config={"max_concurrency": max_concurrency}
                    )

                    # Process batch results
                    for rep_idx, response in enumerate(batch_responses):
                        rep = batch_start + rep_idx + 1

                        # Convert Pydantic model to dict if needed
                        if hasattr(response, 'model_dump'):
                            response_dict = response.model_dump()
                        elif hasattr(response, 'dict'):
                            response_dict = response.dict()
                        else:
                            response_dict = dict(response)

                        # Add metadata
                        response_data = {
                            "scenario_id": scenario_id,
                            "scenario_name": scenario["scenario_name"],
                            "options": {
                                "left": scenario["left"],
                                "right": scenario["right"],
                            },
                            "repetition": rep,
                            "thought": response_dict["thought"],
                            "answer": response_dict["answer"],
                        }

                        responses.append(response_data)

                    pbar.update(current_batch_size)

                except Exception as e:
                    logger.error(
                        f"Failed batch for scenario {scenario_id}, "
                        f"reps {batch_start+1}-{batch_end}: {e}"
                    )
                    pbar.update(current_batch_size)
                    continue

            # Save responses for this scenario
            output_file = output_dir / f"scenario_{scenario_id}_baseline.json"
            save_response_to_file(responses, output_file)
            logger.info(f"Saved {len(responses)} responses to {output_file}")


def generate_persona_responses(
    model,
    language: str,
    personas_dir: Path,
    repetitions: int,
    output_dir: Path,
    rate_limiter: Optional[RateLimiter] = None,
    max_personas: Optional[int] = None,
    personas_per_domain: Optional[int] = None,
    batch_size: int = 10,
    max_concurrency: int = 5,
):
    """
    Generate persona-injected responses for all scenarios.

    Args:
        model: LangChain chat model
        language: Language code ("en" or "kr")
        personas_dir: Directory containing persona JSONL files
        repetitions: Number of repetitions per persona-scenario pair
        output_dir: Output directory for responses
        rate_limiter: Optional rate limiter
        max_personas: Optional limit on number of personas to process (total)
        personas_per_domain: Optional random sample size per domain file
        batch_size: Number of requests to batch together
        max_concurrency: Maximum number of concurrent requests
    """
    logger.info(f"Generating persona-injected responses ({language})")
    logger.info(f"Batch size: {batch_size}, Max concurrency: {max_concurrency}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all personas
    all_personas = []
    persona_files = sorted(personas_dir.glob("*.jsonl"))
    logger.info(f"Found {len(persona_files)} persona files")

    for persona_file in persona_files:
        personas = load_personas_from_file(persona_file)

        # Random sample per domain if specified
        if personas_per_domain and len(personas) > personas_per_domain:
            personas = random.sample(personas, personas_per_domain)
            logger.info(f"Sampled {len(personas)} personas from {persona_file.name}")

        all_personas.extend(personas)

    logger.info(f"Loaded {len(all_personas)} total personas")

    # Limit personas if specified (for backward compatibility)
    if max_personas:
        all_personas = all_personas[:max_personas]
        logger.info(f"Limited to {len(all_personas)} personas")

    total_queries = len(all_personas) * len(SCENARIOS) * repetitions
    logger.info(f"Total queries to generate: {total_queries:,}")

    with tqdm(total=total_queries, desc=f"Persona ({language})") as pbar:
        for persona_idx, persona_data in enumerate(all_personas):
            persona_desc = persona_data.get("persona", "")
            if language == 'kr' and persona_desc == "":
                persona_desc = persona_data.get("persona_kr", "")
            domain = persona_data.get("domain", "unknown")

            # Generate responses for all scenarios
            persona_responses = {}

            for scenario in SCENARIOS:
                scenario_id = scenario["scenario_id"]
                scenario_key = f"scenario_{scenario_id}"
                scenario_reps = []

                # Format scenario values
                scenario_values = format_scenario_prompt(scenario)

                # Generate prompt with persona
                prompt = format_prompt(
                    language=language,
                    condition="persona",
                    persona_description=persona_desc,
                    **scenario_values,
                )

                # Generate repetitions in batches
                for batch_start in range(0, repetitions, batch_size):
                    batch_end = min(batch_start + batch_size, repetitions)
                    current_batch_size = batch_end - batch_start

                    try:
                        if rate_limiter:
                            rate_limiter.wait_if_needed()

                        # Create batch of identical prompts
                        prompts = [prompt] * current_batch_size

                        # Batch process
                        batch_responses = model.batch(
                            prompts,
                            config={"max_concurrency": max_concurrency}
                        )

                        # Process batch results
                        for rep_idx, response in enumerate(batch_responses):
                            rep = batch_start + rep_idx + 1

                            # Convert Pydantic model to dict if needed
                            if hasattr(response, 'model_dump'):
                                response_dict = response.model_dump()
                            elif hasattr(response, 'dict'):
                                response_dict = response.dict()
                            else:
                                response_dict = dict(response)

                            # Add response data
                            response_data = {
                                "repetition": rep,
                                "thought": response_dict["thought"],
                                "answer": response_dict["answer"],
                            }

                            scenario_reps.append(response_data)

                        pbar.update(current_batch_size)

                    except Exception as e:
                        logger.error(
                            f"Failed batch for persona {persona_idx+1}, "
                            f"scenario {scenario_id}, reps {batch_start+1}-{batch_end}: {e}"
                        )
                        pbar.update(current_batch_size)
                        continue

                # Store scenario data
                persona_responses[scenario_key] = {
                    "persona_id": persona_idx + 1,
                    "persona_desc": persona_desc,
                    "domain": domain,
                    "scenario_name": scenario["scenario_name"],
                    "options": {
                        "left": scenario["left"],
                        "right": scenario["right"],
                    },
                    "responses": scenario_reps,
                }

            # Save responses for this persona (organized by domain)
            domain_dir = output_dir / domain
            domain_dir.mkdir(parents=True, exist_ok=True)
            output_file = domain_dir / f"persona_{persona_idx+1:05d}.json"
            save_response_to_file(persona_responses, output_file)

            if (persona_idx + 1) % 100 == 0:
                logger.info(f"Completed {persona_idx+1}/{len(all_personas)} personas")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate LLM responses for distributive justice scenarios")
    parser.add_argument(
        "--condition",
        type=str,
        choices=["baseline", "persona"],
        required=True,
        help="Experimental condition"
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["en", "kr"],
        required=True,
        help="Language for prompts"
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="Number of repetitions (default: 100 for baseline, 10 for persona)"
    )
    parser.add_argument(
        "--personas-dir",
        type=Path,
        default=Path("data/personas/en"),
        help="Directory containing persona files (for persona condition)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/responses/{condition}/{language})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="Gemini model name"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=60,
        help="API calls per minute (0 to disable)"
    )
    parser.add_argument(
        "--max-personas",
        type=int,
        default=None,
        help="Maximum number of personas to process (for testing)"
    )
    parser.add_argument(
        "--personas-per-domain",
        type=int,
        default=None,
        help="Random sample N personas from each domain file (default: use all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of requests to batch together (default: 10)"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent requests in a batch (default: 5)"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Set default repetitions
    if args.repetitions is None:
        args.repetitions = 100 if args.condition == "baseline" else 10

    # Set default output directory
    if args.output_dir is None:
        args.output_dir = Path(f"data/responses/{args.condition}/{args.language}")

    logger.info("=" * 70)
    logger.info("LLM Response Generation")
    logger.info("=" * 70)
    logger.info(f"Condition: {args.condition}")
    logger.info(f"Language: {args.language}")
    logger.info(f"Repetitions: {args.repetitions}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max concurrency: {args.max_concurrency}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("")

    # Initialize model with language-specific schema
    model = init_model(args.model, args.temperature, args.language)

    # Initialize rate limiter
    rate_limiter = RateLimiter(args.rate_limit) if args.rate_limit > 0 else None
    if rate_limiter:
        logger.info(f"Rate limit: {args.rate_limit} calls/minute")

    # Generate responses
    start_time = datetime.now()

    if args.condition == "baseline":
        generate_baseline_responses(
            model=model,
            language=args.language,
            repetitions=args.repetitions,
            output_dir=args.output_dir,
            rate_limiter=rate_limiter,
            batch_size=args.batch_size,
            max_concurrency=args.max_concurrency,
        )
    else:  # persona
        generate_persona_responses(
            model=model,
            language=args.language,
            personas_dir=args.personas_dir,
            repetitions=args.repetitions,
            output_dir=args.output_dir,
            rate_limiter=rate_limiter,
            max_personas=args.max_personas,
            personas_per_domain=args.personas_per_domain,
            batch_size=args.batch_size,
            max_concurrency=args.max_concurrency,
        )

    elapsed = datetime.now() - start_time
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Completed in {elapsed}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
