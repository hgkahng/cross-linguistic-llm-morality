"""
Stage 4: Back-Translation Verification

This script performs back-translation verification to ensure semantic equivalence
between original English personas and their Korean translations.

Process:
1. Translate Korean back to English
2. Compare back-translated English with original English
3. Evaluate semantic equivalence using LLM
4. Flag divergences (none/minor/major)
5. Reject and regenerate translations with major divergence

Quality assurance:
- Reject translations with "major" divergence
- Flag "minor" divergence cases for manual review
- Regenerate problematic translations
- Final output: data/personas/kr/{domain}.jsonl
"""

import json
import logging
import argparse
import time
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

try:
    from .utils import RateLimiter
    from .prompt_templates import BACKTRANSLATION_PROMPT_TEMPLATE, EQUIVALENCE_CHECK_PROMPT_TEMPLATE
except ImportError:
    from utils import RateLimiter
    from prompt_templates import BACKTRANSLATION_PROMPT_TEMPLATE, EQUIVALENCE_CHECK_PROMPT_TEMPLATE


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EquivalenceCheckResult(BaseModel):
    """Structured output schema for equivalence checking."""
    equivalent: bool = Field(description="Whether descriptions are semantically equivalent")
    divergence_explanation: str = Field(description="Explanation of any meaningful differences")
    severity: Literal["none", "minor", "major"] = Field(description="Severity of divergence")


def init_translation_model(model_name: str = "gemini-2.0-flash", temperature: float = 0.3):
    """
    Initialize LangChain chat model for back-translation.

    Uses lower temperature for consistent back-translation.

    Args:
        model_name: Gemini model name
        temperature: Sampling temperature

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

        logger.info(f"Initialized back-translation model: {model_name} (temperature={temperature})")
        return model

    except ImportError:
        logger.error("langchain not installed. Install with: pip install langchain langchain-google-genai")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise


def init_equivalence_model(model_name: str = "gemini-2.0-flash", temperature: float = 0.5):
    """
    Initialize LangChain chat model for equivalence checking with structured output.

    Args:
        model_name: Gemini model name
        temperature: Sampling temperature

    Returns:
        Initialized chat model with structured output binding
    """
    try:
        from langchain.chat_models import init_chat_model

        base_model = init_chat_model(
            model=model_name,
            model_provider="google_genai",
            temperature=temperature,
        )

        # Bind structured output schema
        model = base_model.with_structured_output(EquivalenceCheckResult)

        logger.info(f"Initialized equivalence model: {model_name} (temperature={temperature})")
        return model

    except ImportError:
        logger.error("langchain not installed. Install with: pip install langchain langchain-google-genai")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise


def backtranslate(
    persona_kr: str,
    model,
    rate_limiter: Optional[RateLimiter] = None
) -> Optional[str]:
    """
    Translate Korean persona back to English.

    Args:
        persona_kr: Korean persona description
        model: LangChain chat model
        rate_limiter: Optional rate limiter

    Returns:
        English back-translation, or None if failed
    """
    try:
        if rate_limiter:
            rate_limiter.wait_if_needed()

        prompt = BACKTRANSLATION_PROMPT_TEMPLATE.format(korean_persona=persona_kr)
        response = model.invoke(prompt)

        if hasattr(response, 'content'):
            translation = response.content.strip()
        else:
            translation = str(response).strip()

        return translation

    except Exception as e:
        logger.error(f"Back-translation failed: {e}")
        return None


def check_equivalence(
    original_en: str,
    backtranslated_en: str,
    model,
    rate_limiter: Optional[RateLimiter] = None
) -> Optional[EquivalenceCheckResult]:
    """
    Check semantic equivalence between original and back-translated English.

    Args:
        original_en: Original English persona
        backtranslated_en: Back-translated English from Korean
        model: LangChain chat model with structured output
        rate_limiter: Optional rate limiter

    Returns:
        EquivalenceCheckResult object, or None if check failed
    """
    try:
        if rate_limiter:
            rate_limiter.wait_if_needed()

        prompt = EQUIVALENCE_CHECK_PROMPT_TEMPLATE.format(
            original_english=original_en,
            backtranslated_english=backtranslated_en
        )

        result = model.invoke(prompt)
        return result

    except Exception as e:
        logger.error(f"Equivalence check failed: {e}")
        return None


def backtranslate_and_verify(
    persona_en: str,
    persona_kr: str,
    backtranslation_model,
    equivalence_model,
    rate_limiter: Optional[RateLimiter] = None
) -> tuple[Optional[str], Optional[EquivalenceCheckResult]]:
    """
    Perform back-translation and equivalence verification.

    Args:
        persona_en: Original English persona
        persona_kr: Korean translation
        backtranslation_model: Model for back-translation
        equivalence_model: Model for equivalence checking
        rate_limiter: Optional rate limiter

    Returns:
        Tuple of (backtranslated_english, equivalence_result)
    """
    # Back-translate Korean to English
    backtranslated_en = backtranslate(
        persona_kr=persona_kr,
        model=backtranslation_model,
        rate_limiter=rate_limiter
    )

    if not backtranslated_en:
        logger.error("Back-translation failed")
        return None, None

    # Check equivalence
    equivalence_result = check_equivalence(
        original_en=persona_en,
        backtranslated_en=backtranslated_en,
        model=equivalence_model,
        rate_limiter=rate_limiter
    )

    if not equivalence_result:
        logger.error("Equivalence check failed")
        return backtranslated_en, None

    return backtranslated_en, equivalence_result


def process_domain_file(
    input_file: Path,
    output_dir: Path,
    backtranslation_model,
    equivalence_model,
    rate_limiter: Optional[RateLimiter] = None,
    save_divergent: bool = True,
    batch_size: int = 10,
    max_concurrency: int = 5
) -> dict:
    """
    Process all verified translations in a domain file with back-translation.

    Uses batch processing for efficient parallel API calls.

    Args:
        input_file: Input JSONL file with verified translations
        output_dir: Output directory for final translations
        backtranslation_model: Model for back-translation
        equivalence_model: Model for equivalence checking
        rate_limiter: Optional rate limiter
        save_divergent: Whether to save divergent cases to separate file
        batch_size: Number of personas to batch together (default: 10)
        max_concurrency: Maximum concurrent requests in batch (default: 5)

    Returns:
        Dictionary with processing statistics
    """
    # Extract domain name (e.g., "economics_verified.jsonl" -> "economics")
    domain = input_file.stem.replace("_verified", "")

    # Create output file paths
    output_file = output_dir / f"{domain}.jsonl"
    divergent_file = output_dir / f"{domain}_divergent.jsonl" if save_divergent else None

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove existing output files
    if output_file.exists():
        logger.warning(f"Removing existing file: {output_file}")
        output_file.unlink()

    if divergent_file and divergent_file.exists():
        logger.warning(f"Removing existing file: {divergent_file}")
        divergent_file.unlink()

    # Load verified translations
    verified = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                verified.append(record)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

    logger.info(f"Loaded {len(verified)} verified translations for domain: {domain}")

    # Track statistics
    stats = {
        "domain": domain,
        "total": len(verified),
        "no_divergence": 0,
        "minor_divergence": 0,
        "major_divergence": 0,
        "failed": 0,
        "start_time": time.time(),
        "batch_size": batch_size,
        "max_concurrency": max_concurrency
    }

    # Process verified translations in batches
    logger.info(f"Processing {len(verified)} personas in batches of {batch_size}")
    num_batches = (len(verified) + batch_size - 1) // batch_size

    with tqdm(total=len(verified), desc=f"Back-translating {domain}", unit="persona") as pbar:
        for batch_idx in range(num_batches):
            # Extract batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(verified))
            batch_records = verified[start_idx:end_idx]

            # Extract English and Korean personas
            personas_en = [r.get("persona_en", "") for r in batch_records]
            personas_kr = [r.get("persona_kr", "") for r in batch_records]

            # Filter out records with missing text
            valid_indices = [i for i in range(len(batch_records)) if personas_en[i] and personas_kr[i]]
            if len(valid_indices) < len(batch_records):
                empty_count = len(batch_records) - len(valid_indices)
                logger.warning(f"Skipping {empty_count} records with missing text in batch {batch_idx+1}")
                stats["failed"] += empty_count
                pbar.update(empty_count)

            if not valid_indices:
                continue

            valid_personas_en = [personas_en[i] for i in valid_indices]
            valid_personas_kr = [personas_kr[i] for i in valid_indices]

            try:
                # Apply rate limiting (one check per batch)
                if rate_limiter:
                    rate_limiter.wait_if_needed()

                # Step 1: Batch back-translate Korean → English
                backtranslation_prompts = [
                    BACKTRANSLATION_PROMPT_TEMPLATE.format(korean_persona=kr)
                    for kr in valid_personas_kr
                ]

                backtranslation_responses = backtranslation_model.batch(
                    backtranslation_prompts,
                    config={"max_concurrency": max_concurrency}
                )

                # Extract back-translated English
                backtranslated_en_list = []
                for response in backtranslation_responses:
                    if hasattr(response, 'content'):
                        translation = response.content.strip()
                    else:
                        translation = str(response).strip()
                    backtranslated_en_list.append(translation)

                # Step 2: Batch equivalence checking
                if rate_limiter:
                    rate_limiter.wait_if_needed()

                equivalence_prompts = [
                    EQUIVALENCE_CHECK_PROMPT_TEMPLATE.format(
                        original_english=valid_personas_en[i],
                        backtranslated_english=backtranslated_en_list[i]
                    )
                    for i in range(len(valid_personas_en))
                ]

                equivalence_results = equivalence_model.batch(
                    equivalence_prompts,
                    config={"max_concurrency": max_concurrency}
                )

                # Process batch results
                for i in range(len(valid_indices)):
                    persona_en = valid_personas_en[i]
                    persona_kr = valid_personas_kr[i]
                    backtranslated_en = backtranslated_en_list[i]
                    equivalence_result = equivalence_results[i]

                    if not backtranslated_en or not equivalence_result:
                        logger.warning("Back-translation verification failed")
                        stats["failed"] += 1
                        pbar.update(1)
                        continue

                    # Create final record with back-translation info
                    final_record = {
                        "domain": domain,
                        "persona_en": persona_en,
                        "persona_kr": persona_kr,
                        "backtranslation": {
                            "backtranslated_en": backtranslated_en,
                            "equivalent": equivalence_result.equivalent,
                            "divergence_explanation": equivalence_result.divergence_explanation,
                            "severity": equivalence_result.severity
                        }
                    }

                    # Update statistics
                    severity = equivalence_result.severity
                    if severity == "none":
                        stats["no_divergence"] += 1
                    elif severity == "minor":
                        stats["minor_divergence"] += 1
                    elif severity == "major":
                        stats["major_divergence"] += 1

                    # Save to appropriate file
                    # Accept translations with "none" or "minor" divergence
                    if severity in ["none", "minor"]:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(final_record, ensure_ascii=False) + '\n')
                    else:
                        # Major divergence - save to divergent file for review
                        if divergent_file:
                            with open(divergent_file, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(final_record, ensure_ascii=False) + '\n')
                        logger.warning(f"Major divergence detected: {equivalence_result.divergence_explanation[:100]}...")

                pbar.update(len(valid_indices))

            except Exception as e:
                logger.error(f"Batch {batch_idx+1} failed: {e}")
                logger.info("Falling back to sequential processing for this batch")

                # Fallback to sequential processing
                for i in valid_indices:
                    persona_en = personas_en[i]
                    persona_kr = personas_kr[i]

                    # Perform back-translation and equivalence check
                    backtranslated_en, equivalence_result = backtranslate_and_verify(
                        persona_en=persona_en,
                        persona_kr=persona_kr,
                        backtranslation_model=backtranslation_model,
                        equivalence_model=equivalence_model,
                        rate_limiter=rate_limiter
                    )

                    if not backtranslated_en or not equivalence_result:
                        logger.warning("Back-translation verification failed")
                        stats["failed"] += 1
                        pbar.update(1)
                        continue

                    # Create final record with back-translation info
                    final_record = {
                        "domain": domain,
                        "persona_en": persona_en,
                        "persona_kr": persona_kr,
                        "backtranslation": {
                            "backtranslated_en": backtranslated_en,
                            "equivalent": equivalence_result.equivalent,
                            "divergence_explanation": equivalence_result.divergence_explanation,
                            "severity": equivalence_result.severity
                        }
                    }

                    # Update statistics
                    severity = equivalence_result.severity
                    if severity == "none":
                        stats["no_divergence"] += 1
                    elif severity == "minor":
                        stats["minor_divergence"] += 1
                    elif severity == "major":
                        stats["major_divergence"] += 1

                    # Save to appropriate file
                    if severity in ["none", "minor"]:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(final_record, ensure_ascii=False) + '\n')
                    else:
                        if divergent_file:
                            with open(divergent_file, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(final_record, ensure_ascii=False) + '\n')
                        logger.warning(f"Major divergence detected: {equivalence_result.divergence_explanation[:100]}...")

                    pbar.update(1)

    # Calculate elapsed time
    stats["elapsed_time"] = time.time() - stats["start_time"]

    # Calculate acceptance rate
    accepted = stats["no_divergence"] + stats["minor_divergence"]
    acceptance_rate = (accepted / stats["total"] * 100) if stats["total"] > 0 else 0.0

    # Log summary
    logger.info(f"""
    Domain: {domain}
    Total: {stats['total']}
    No divergence: {stats['no_divergence']}
    Minor divergence: {stats['minor_divergence']}
    Major divergence: {stats['major_divergence']}
    Failed: {stats['failed']}
    Acceptance rate: {acceptance_rate:.1f}%
    Batch size: {batch_size}
    Max concurrency: {max_concurrency}
    Elapsed time: {stats['elapsed_time']:.1f}s
    """)

    return stats


def main() -> None:
    """Main back-translation verification pipeline for Stage 4."""
    parser = argparse.ArgumentParser(
        description="Stage 4: Back-translation verification of Korean translations"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/personas/kr_verified/"),
        help="Input directory with verified translations"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/personas/kr/"),
        help="Output directory for final translations"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="Model name for back-translation and equivalence checking"
    )
    parser.add_argument(
        "--backtranslation-temperature",
        type=float,
        default=0.3,
        help="Temperature for back-translation (lower = more consistent)"
    )
    parser.add_argument(
        "--equivalence-temperature",
        type=float,
        default=0.5,
        help="Temperature for equivalence checking"
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=60,
        help="Maximum requests per minute (0 = no limit)"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Specific domains to process (default: all)"
    )
    parser.add_argument(
        "--save-divergent",
        action="store_true",
        default=True,
        help="Save divergent cases to separate file for review"
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

    # Initialize models
    logger.info("Initializing back-translation model...")
    backtranslation_model = init_translation_model(
        model_name=args.model,
        temperature=args.backtranslation_temperature
    )

    logger.info("Initializing equivalence checking model (with structured output)...")
    equivalence_model = init_equivalence_model(
        model_name=args.model,
        temperature=args.equivalence_temperature
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

    input_files = sorted(args.input.glob("*_verified.jsonl"))

    if not input_files:
        logger.error(f"No verified JSONL files found in {args.input}")
        return

    # Filter by specified domains if provided
    if args.domains:
        domain_set = set(args.domains)
        input_files = [f for f in input_files if f.stem.replace("_verified", "") in domain_set]
        logger.info(f"Processing specific domains: {args.domains}")

    logger.info(f"Found {len(input_files)} domain files to process")

    # Process each domain file
    all_stats = []

    for input_file in input_files:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {input_file.name}")
        logger.info(f"{'='*70}\n")

        stats = process_domain_file(
            input_file=input_file,
            output_dir=args.output,
            backtranslation_model=backtranslation_model,
            equivalence_model=equivalence_model,
            rate_limiter=rate_limiter,
            save_divergent=args.save_divergent,
            batch_size=args.batch_size,
            max_concurrency=args.max_concurrency
        )

        all_stats.append(stats)

    # Print overall summary
    logger.info(f"\n{'='*70}")
    logger.info("OVERALL SUMMARY")
    logger.info(f"{'='*70}\n")

    total_personas = sum(s["total"] for s in all_stats)
    total_no_div = sum(s["no_divergence"] for s in all_stats)
    total_minor_div = sum(s["minor_divergence"] for s in all_stats)
    total_major_div = sum(s["major_divergence"] for s in all_stats)
    total_failed = sum(s["failed"] for s in all_stats)
    total_time = sum(s["elapsed_time"] for s in all_stats)

    logger.info(f"Total Personas: {total_personas}")
    logger.info(f"No divergence: {total_no_div}")
    logger.info(f"Minor divergence: {total_minor_div}")
    logger.info(f"Major divergence: {total_major_div}")
    logger.info(f"Failed: {total_failed}")

    if total_personas > 0:
        acceptance_rate = ((total_no_div + total_minor_div) / total_personas) * 100
        logger.info(f"Overall acceptance rate: {acceptance_rate:.1f}%")
        logger.info(f"Semantic equivalence (≥95% target): {acceptance_rate >= 95.0}")

    logger.info(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    logger.info(f"\nFinal translations saved to: {args.output}")
    if args.save_divergent:
        logger.info(f"Divergent cases saved to: {args.output}/*_divergent.jsonl")
    logger.info("\nStage 4 complete! All translation stages finished.")
    logger.info("Review any divergent cases and proceed with response generation.")


if __name__ == "__main__":
    main()
