"""
Stage 1: Initial Translation of personas from English to Korean.

This script translates persona descriptions using Gemini 2.0 Flash,
following the four-stage LLM-as-a-Judge methodology (Stage 1).

Translation principles:
1. Formal register appropriate for academic contexts
2. Semantic meaning and pragmatic force preservation
3. Domain-specific terminology and professional framing
4. Natural Korean expression patterns
"""

import json
import logging
import argparse
import time
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from dotenv import load_dotenv

try:
    from .utils import RateLimiter
    from .prompt_templates import TRANSLATION_PROMPT_TEMPLATE
except ImportError:
    from utils import RateLimiter
    from prompt_templates import TRANSLATION_PROMPT_TEMPLATE


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def init_translation_model(model_name: str = "gemini-2.0-flash", temperature: float = 0.3):
    """
    Initialize LangChain chat model for translation.

    Uses lower temperature (0.3) for more consistent translations
    while maintaining some flexibility for natural Korean expression.

    Args:
        model_name: Gemini model name
        temperature: Sampling temperature (lower = more consistent)

    Returns:
        Initialized chat model
    """
    try:
        from langchain.chat_models import init_chat_model

        model = init_chat_model(
            model=model_name,
            model_provider="google_genai",
            temperature=temperature,
        )

        logger.info(f"Initialized translation model: {model_name} (temperature={temperature})")
        return model

    except ImportError:
        logger.error("langchain not installed. Install with: pip install langchain langchain-google-genai")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise


def translate_persona(persona_en: str, model, rate_limiter: Optional[RateLimiter] = None) -> Optional[str]:
    """
    Translate a single persona description to Korean.

    Args:
        persona_en: English persona description
        model: LangChain chat model
        rate_limiter: Optional rate limiter for API calls

    Returns:
        Korean translation, or None if translation failed
    """
    try:
        # Apply rate limiting if configured
        if rate_limiter:
            rate_limiter.wait_if_needed()

        # Format translation prompt
        prompt = TRANSLATION_PROMPT_TEMPLATE.format(persona_description=persona_en)

        # Invoke model
        response = model.invoke(prompt)

        # Extract content from response
        if hasattr(response, 'content'):
            translation = response.content.strip()
        else:
            translation = str(response).strip()

        return translation

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return None


def translate_persona_with_retry(
    persona_en: str,
    model,
    rate_limiter: Optional[RateLimiter] = None,
    max_retries: int = 3
) -> Optional[str]:
    """
    Translate persona with retry logic.

    Args:
        persona_en: English persona description
        model: LangChain chat model
        rate_limiter: Optional rate limiter
        max_retries: Maximum number of retry attempts

    Returns:
        Korean translation, or None if all retries failed
    """
    for attempt in range(max_retries):
        translation = translate_persona(persona_en, model, rate_limiter)

        if translation:
            return translation

        if attempt < max_retries - 1:
            wait_time = 2 ** attempt  # Exponential backoff
            logger.warning(f"Retry attempt {attempt + 1}/{max_retries} after {wait_time}s")
            time.sleep(wait_time)

    logger.error(f"Translation failed after {max_retries} attempts")
    return None


def load_personas_from_file(file_path: Path) -> list[dict]:
    """
    Load personas from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of persona dictionaries
    """
    personas = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                persona = json.loads(line)
                personas.append(persona)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

    logger.info(f"Loaded {len(personas)} personas from {file_path.name}")
    return personas


def save_translation(
    output_file: Path,
    domain: str,
    persona_en: str,
    persona_kr: str
) -> None:
    """
    Save translated persona to JSONL file.

    Args:
        output_file: Path to output JSONL file
        domain: Domain name
        persona_en: Original English persona
        persona_kr: Korean translation
    """
    translation_record = {
        "domain": domain,
        "persona_en": persona_en,
        "persona_kr": persona_kr
    }

    # Append to file
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(translation_record, ensure_ascii=False) + '\n')


def translate_domain_file(
    input_file: Path,
    output_dir: Path,
    model,
    rate_limiter: Optional[RateLimiter] = None,
    max_retries: int = 3,
    batch_size: int = 10,
    max_concurrency: int = 5
) -> dict:
    """
    Translate all personas in a domain file using LangChain's batch() method.

    Uses model.batch() for efficient parallel processing of translation requests.

    Args:
        input_file: Input JSONL file with English personas
        output_dir: Output directory for Korean translations
        model: LangChain chat model
        rate_limiter: Optional rate limiter
        max_retries: Maximum retry attempts per batch
        batch_size: Number of personas to batch together (default: 10)
        max_concurrency: Maximum concurrent requests in batch (default: 5)

    Returns:
        Dictionary with translation statistics
    """
    # Extract domain name from filename
    domain = input_file.stem  # e.g., "economics.jsonl" -> "economics"

    # Create output file path
    output_file = output_dir / f"{domain}_draft.jsonl"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove existing output file if present (fresh start)
    if output_file.exists():
        logger.warning(f"Removing existing file: {output_file}")
        output_file.unlink()

    # Load personas
    personas = load_personas_from_file(input_file)

    # Track statistics
    stats = {
        "domain": domain,
        "total": len(personas),
        "successful": 0,
        "failed": 0,
        "start_time": time.time(),
        "batch_size": batch_size,
        "max_concurrency": max_concurrency
    }

    # Translate personas in batches
    logger.info(f"Translating {stats['total']} personas for domain: {domain}")
    logger.info(f"Batch size: {batch_size}, Max concurrency: {max_concurrency}")

    num_batches = (len(personas) + batch_size - 1) // batch_size

    with tqdm(total=len(personas), desc=f"Translating {domain}", unit="persona") as pbar:
        for batch_idx in range(num_batches):
            # Extract batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(personas))
            batch_personas = personas[start_idx:end_idx]

            # Extract English persona descriptions
            personas_en = [p.get("persona", "") for p in batch_personas]

            # Filter out empty personas
            valid_indices = [i for i, p in enumerate(personas_en) if p]
            if len(valid_indices) < len(personas_en):
                empty_count = len(personas_en) - len(valid_indices)
                logger.warning(f"Skipping {empty_count} empty personas in batch {batch_idx+1}")
                stats["failed"] += empty_count
                pbar.update(empty_count)

            if not valid_indices:
                continue

            valid_personas_en = [personas_en[i] for i in valid_indices]

            # Create prompts for batch
            prompts = [
                TRANSLATION_PROMPT_TEMPLATE.format(persona_description=persona)
                for persona in valid_personas_en
            ]

            try:
                # Apply rate limiting (one check per batch)
                # Note: model.batch() handles internal rate limiting for concurrent requests
                if rate_limiter:
                    rate_limiter.wait_if_needed()

                # Batch translation using model.batch()
                # Processes multiple prompts with controlled concurrency
                batch_responses = model.batch(
                    prompts,
                    config={"max_concurrency": max_concurrency}
                )

                # Process batch results
                for i, response in enumerate(batch_responses):
                    # Extract translation
                    if hasattr(response, 'content'):
                        translation = response.content.strip()
                    else:
                        translation = str(response).strip()

                    if translation:
                        # Save translation
                        save_translation(
                            output_file=output_file,
                            domain=domain,
                            persona_en=valid_personas_en[i],
                            persona_kr=translation
                        )
                        stats["successful"] += 1
                    else:
                        logger.error(f"Empty translation for persona {start_idx + valid_indices[i] + 1}")
                        stats["failed"] += 1

                pbar.update(len(valid_personas_en))

            except Exception as e:
                logger.error(f"Batch {batch_idx+1} failed: {e}")
                logger.info("Falling back to single-persona mode for this batch")

                # Fallback to single-persona mode
                for i, persona_en in enumerate(valid_personas_en):
                    persona_kr = translate_persona_with_retry(
                        persona_en=persona_en,
                        model=model,
                        rate_limiter=rate_limiter,
                        max_retries=max_retries
                    )

                    if persona_kr:
                        save_translation(
                            output_file=output_file,
                            domain=domain,
                            persona_en=persona_en,
                            persona_kr=persona_kr
                        )
                        stats["successful"] += 1
                    else:
                        logger.error(f"Failed to translate persona after fallback")
                        stats["failed"] += 1

                    pbar.update(1)

    # Calculate elapsed time
    stats["elapsed_time"] = time.time() - stats["start_time"]

    # Calculate success rate
    success_rate = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0.0

    # Log summary
    logger.info(f"""
    Domain: {domain}
    Total: {stats['total']}
    Successful: {stats['successful']}
    Failed: {stats['failed']}
    Success Rate: {success_rate:.1f}%
    Batch Size: {batch_size}
    Max Concurrency: {max_concurrency}
    Elapsed Time: {stats['elapsed_time']:.1f}s
    """)

    return stats


def main() -> None:
    """Main translation pipeline for Stage 1."""
    parser = argparse.ArgumentParser(
        description="Stage 1: Translate personas from English to Korean"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/personas/en/"),
        help="Input directory with English personas"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/personas/kr_draft/"),
        help="Output directory for Korean translation drafts"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="Model name for translation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (lower = more consistent)"
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=60,
        help="Maximum requests per minute (0 = no limit)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts per persona"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Specific domains to translate (default: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of personas to batch together (default: 10)"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Maximum concurrent requests in a batch (default: 5)"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Initialize model
    logger.info("Initializing translation model...")
    model = init_translation_model(
        model_name=args.model,
        temperature=args.temperature
    )

    # Initialize rate limiter
    rate_limiter = None
    if args.rate_limit > 0:
        rate_limiter = RateLimiter(calls_per_minute=args.rate_limit)
        logger.info(f"Rate limiting enabled: {args.rate_limit} requests/minute")

    # Find input files
    if not args.input.exists():
        logger.error(f"Input directory does not exist: {args.input}")
        return

    input_files = sorted(args.input.glob("*.jsonl"))

    if not input_files:
        logger.error(f"No JSONL files found in {args.input}")
        return

    # Filter by specified domains if provided
    if args.domains:
        domain_set = set(args.domains)
        input_files = [f for f in input_files if f.stem in domain_set]
        logger.info(f"Translating specific domains: {args.domains}")

    logger.info(f"Found {len(input_files)} domain files to translate")

    # Translate each domain file
    all_stats = []

    for input_file in input_files:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {input_file.name}")
        logger.info(f"{'='*70}\n")

        stats = translate_domain_file(
            input_file=input_file,
            output_dir=args.output,
            model=model,
            rate_limiter=rate_limiter,
            max_retries=args.max_retries,
            batch_size=args.batch_size,
            max_concurrency=args.max_concurrency
        )

        all_stats.append(stats)

    # Print overall summary
    logger.info(f"\n{'='*70}")
    logger.info("OVERALL SUMMARY")
    logger.info(f"{'='*70}\n")

    total_personas = sum(s["total"] for s in all_stats)
    total_successful = sum(s["successful"] for s in all_stats)
    total_failed = sum(s["failed"] for s in all_stats)
    total_time = sum(s["elapsed_time"] for s in all_stats)

    logger.info(f"Total Personas: {total_personas}")
    logger.info(f"Successful: {total_successful}")
    logger.info(f"Failed: {total_failed}")

    if total_personas > 0:
        success_rate = (total_successful / total_personas) * 100
        avg_time = total_time / total_personas
        logger.info(f"Overall Success Rate: {success_rate:.1f}%")
        logger.info(f"Average Time per Persona: {avg_time:.2f}s")

    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Max Concurrency: {args.max_concurrency}")
    logger.info(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    logger.info(f"\nTranslations saved to: {args.output}")
    logger.info("Stage 1 complete! Proceed to Stage 2 for verification.")


if __name__ == "__main__":
    main()
