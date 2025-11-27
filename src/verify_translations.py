"""
Stages 2-3: Multi-Instance Verification and Consensus-Based Acceptance

This script implements the verification pipeline for Korean persona translations:
- Stage 2: Multi-instance verification (5 evaluations per translation)
- Stage 3: Consensus-based acceptance (≥3/5 accept votes)

Evaluation criteria:
1. Semantic accuracy and cultural appropriateness
2. Preservation of disciplinary framing and domain terminology
3. Naturalness of Korean formal register
4. Parallel scenario framing and context salience

Acceptance criteria:
- ≥3 out of 5 instances vote "accept"
- Average criteria scores ≥4.0
- Maximum 3 regeneration attempts for rejected translations
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

try:
    from .utils import RateLimiter
    from .prompt_templates import TRANSLATION_PROMPT_TEMPLATE, VERIFICATION_PROMPT_TEMPLATE
except ImportError:
    from utils import RateLimiter
    from prompt_templates import TRANSLATION_PROMPT_TEMPLATE, VERIFICATION_PROMPT_TEMPLATE


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CriteriaScores(BaseModel):
    """Scores for the 4 evaluation criteria."""
    semantic_accuracy: int = Field(ge=1, le=5, description="Semantic accuracy score (1-5)")
    domain_terminology: int = Field(ge=1, le=5, description="Domain terminology score (1-5)")
    naturalness: int = Field(ge=1, le=5, description="Korean naturalness score (1-5)")
    professional_framing: int = Field(ge=1, le=5, description="Professional framing score (1-5)")


class VerificationResult(BaseModel):
    """Structured output schema for translation verification."""
    accept: bool = Field(description="Whether to accept this translation")
    justification: str = Field(description="Brief explanation of the decision")
    criteria_scores: CriteriaScores = Field(description="Scores for evaluation criteria")


def init_verification_model(model_name: str = "gemini-2.0-flash", temperature: float = 0.7):
    """
    Initialize LangChain chat model for verification with structured output.

    Uses temperature=0.7 to reduce determinism across multiple evaluations
    while maintaining coherent reasoning.

    Args:
        model_name: Gemini model name
        temperature: Sampling temperature (0.7 for diverse evaluations)

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
        model = base_model.with_structured_output(VerificationResult)

        logger.info(f"Initialized verification model: {model_name} (temperature={temperature})")
        return model

    except ImportError:
        logger.error("langchain not installed. Install with: pip install langchain langchain-google-genai")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise


def verify_translation_single(
    persona_en: str,
    persona_kr: str,
    model,
    rate_limiter: Optional[RateLimiter] = None
) -> Optional[VerificationResult]:
    """
    Perform a single verification evaluation.

    Args:
        persona_en: English persona description
        persona_kr: Korean translation
        model: LangChain chat model with structured output
        rate_limiter: Optional rate limiter

    Returns:
        VerificationResult object, or None if verification failed
    """
    try:
        # Apply rate limiting
        if rate_limiter:
            rate_limiter.wait_if_needed()

        # Format verification prompt
        prompt = VERIFICATION_PROMPT_TEMPLATE.format(
            english_persona=persona_en,
            korean_persona=persona_kr
        )

        # Invoke model (returns VerificationResult via structured output)
        result = model.invoke(prompt)

        return result

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return None


def verify_translation_multi_instance(
    persona_en: str,
    persona_kr: str,
    model,
    rate_limiter: Optional[RateLimiter] = None,
    num_instances: int = 5,
    max_concurrency: int = 5
) -> list[VerificationResult]:
    """
    Perform multi-instance verification (Stage 2) using batch processing.

    Generates multiple independent evaluations to reduce bias.
    Uses model.batch() for efficient parallel processing.

    Args:
        persona_en: English persona description
        persona_kr: Korean translation
        model: LangChain chat model with structured output
        rate_limiter: Optional rate limiter
        num_instances: Number of independent evaluations (default: 5)
        max_concurrency: Maximum concurrent requests in batch (default: 5)

    Returns:
        List of VerificationResult objects (successful evaluations only)
    """
    # Format verification prompt once
    prompt = VERIFICATION_PROMPT_TEMPLATE.format(
        english_persona=persona_en,
        korean_persona=persona_kr
    )

    # Create batch of identical prompts (temperature=0.7 ensures diversity)
    prompts = [prompt] * num_instances

    try:
        # Apply rate limiting (one check per batch)
        if rate_limiter:
            rate_limiter.wait_if_needed()

        # Batch process all verification instances
        batch_results = model.batch(
            prompts,
            config={"max_concurrency": max_concurrency}
        )

        # Filter out None results
        results = [r for r in batch_results if r is not None]

        if len(results) < num_instances:
            logger.warning(
                f"Only {len(results)}/{num_instances} verification instances succeeded"
            )

        return results

    except Exception as e:
        logger.error(f"Batch verification failed: {e}")
        logger.info("Falling back to sequential verification")

        # Fallback to sequential processing
        results = []
        for i in range(num_instances):
            result = verify_translation_single(
                persona_en=persona_en,
                persona_kr=persona_kr,
                model=model,
                rate_limiter=rate_limiter
            )

            if result:
                results.append(result)
            else:
                logger.warning(f"Verification instance {i+1}/{num_instances} failed")

        return results


def consensus_check(
    verification_results: list[VerificationResult],
    min_accept_votes: int = 3,
    min_avg_score: float = 4.0
) -> tuple[bool, dict]:
    """
    Check if translation passes consensus-based acceptance (Stage 3).

    Acceptance criteria:
    1. ≥3 out of 5 instances vote "accept"
    2. Average criteria scores ≥4.0

    Args:
        verification_results: List of VerificationResult objects
        min_accept_votes: Minimum number of accept votes required
        min_avg_score: Minimum average criteria score required

    Returns:
        Tuple of (passes_consensus, stats_dict)
        - passes_consensus: True if translation is accepted
        - stats_dict: Dictionary with consensus statistics
    """
    if not verification_results:
        return False, {"error": "No verification results available"}

    # Count accept votes
    accept_votes = sum(1 for r in verification_results if r.accept)
    total_votes = len(verification_results)

    # Calculate average criteria scores
    all_scores = []
    for result in verification_results:
        scores = result.criteria_scores
        all_scores.extend([
            scores.semantic_accuracy,
            scores.domain_terminology,
            scores.naturalness,
            scores.professional_framing
        ])

    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

    # Calculate per-criterion averages
    criterion_avgs = {
        "semantic_accuracy": 0.0,
        "domain_terminology": 0.0,
        "naturalness": 0.0,
        "professional_framing": 0.0
    }

    for criterion in criterion_avgs.keys():
        scores = [getattr(r.criteria_scores, criterion) for r in verification_results]
        criterion_avgs[criterion] = sum(scores) / len(scores) if scores else 0.0

    # Determine if consensus is reached
    passes_votes = accept_votes >= min_accept_votes
    passes_score = avg_score >= min_avg_score
    passes_consensus = passes_votes and passes_score

    # Compile statistics
    stats = {
        "total_votes": total_votes,
        "accept_votes": accept_votes,
        "reject_votes": total_votes - accept_votes,
        "acceptance_rate": accept_votes / total_votes if total_votes > 0 else 0.0,
        "avg_score": round(avg_score, 2),
        "criterion_avgs": {k: round(v, 2) for k, v in criterion_avgs.items()},
        "passes_votes": passes_votes,
        "passes_score": passes_score,
        "passes_consensus": passes_consensus,
        "min_accept_votes": min_accept_votes,
        "min_avg_score": min_avg_score
    }

    return passes_consensus, stats


def regenerate_translation(
    persona_en: str,
    model,
    rate_limiter: Optional[RateLimiter] = None
) -> Optional[str]:
    """
    Regenerate Korean translation using the same prompt as Stage 1.

    Used when translation is rejected during verification.

    Args:
        persona_en: English persona description
        model: LangChain chat model (base model, not structured output)
        rate_limiter: Optional rate limiter

    Returns:
        New Korean translation, or None if regeneration failed
    """
    try:
        if rate_limiter:
            rate_limiter.wait_if_needed()

        # Use the same prompt as Stage 1 for consistency
        prompt = TRANSLATION_PROMPT_TEMPLATE.format(persona_description=persona_en)

        response = model.invoke(prompt)

        if hasattr(response, 'content'):
            translation = response.content.strip()
        else:
            translation = str(response).strip()

        return translation

    except Exception as e:
        logger.error(f"Regeneration failed: {e}")
        return None


def verify_and_regenerate(
    persona_en: str,
    persona_kr: str,
    verification_model,
    translation_model,
    rate_limiter: Optional[RateLimiter] = None,
    max_regenerations: int = 3,
    num_verification_instances: int = 5,
    max_concurrency: int = 5
) -> tuple[str, dict]:
    """
    Verify translation and regenerate if rejected (Stages 2-3 combined).

    Args:
        persona_en: English persona description
        persona_kr: Initial Korean translation
        verification_model: Model for verification (with structured output)
        translation_model: Model for regeneration (base model)
        rate_limiter: Optional rate limiter
        max_regenerations: Maximum regeneration attempts
        num_verification_instances: Number of verification instances per attempt
        max_concurrency: Maximum concurrent requests in verification batch

    Returns:
        Tuple of (final_translation, verification_stats)
    """
    current_translation = persona_kr
    all_stats = []

    for attempt in range(max_regenerations + 1):
        # Multi-instance verification
        verification_results = verify_translation_multi_instance(
            persona_en=persona_en,
            persona_kr=current_translation,
            model=verification_model,
            rate_limiter=rate_limiter,
            num_instances=num_verification_instances,
            max_concurrency=max_concurrency
        )

        # Check consensus
        passes, stats = consensus_check(verification_results)
        stats["attempt"] = attempt + 1
        stats["is_regeneration"] = attempt > 0
        all_stats.append(stats)

        if passes:
            logger.debug(f"Translation accepted on attempt {attempt + 1}")
            return current_translation, {
                "accepted": True,
                "attempts": attempt + 1,
                "final_stats": stats,
                "all_attempts": all_stats
            }

        # If rejected and regenerations remain, regenerate
        if attempt < max_regenerations:
            logger.debug(f"Translation rejected. Regenerating (attempt {attempt + 2}/{max_regenerations + 1})")
            new_translation = regenerate_translation(
                persona_en=persona_en,
                model=translation_model,
                rate_limiter=rate_limiter
            )

            if new_translation:
                current_translation = new_translation
            else:
                logger.warning("Regeneration failed, keeping current translation")

    # Max attempts reached without acceptance
    logger.warning(f"Translation failed to pass consensus after {max_regenerations + 1} attempts")
    return current_translation, {
        "accepted": False,
        "attempts": max_regenerations + 1,
        "final_stats": all_stats[-1] if all_stats else {},
        "all_attempts": all_stats
    }


def verify_domain_file(
    input_file: Path,
    output_dir: Path,
    verification_model,
    translation_model,
    rate_limiter: Optional[RateLimiter] = None,
    max_regenerations: int = 3,
    num_verification_instances: int = 5,
    max_concurrency: int = 5
) -> dict:
    """
    Verify all translations in a domain file.

    Args:
        input_file: Input JSONL file with draft translations
        output_dir: Output directory for verified translations
        verification_model: Model for verification
        translation_model: Model for regeneration
        rate_limiter: Optional rate limiter
        max_regenerations: Maximum regeneration attempts per persona
        num_verification_instances: Number of verification instances per attempt
        max_concurrency: Maximum concurrent requests in verification batch

    Returns:
        Dictionary with verification statistics
    """
    # Extract domain name from filename (e.g., "economics_draft.jsonl" -> "economics")
    domain = input_file.stem.replace("_draft", "")

    # Create output file path
    output_file = output_dir / f"{domain}_verified.jsonl"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove existing output file if present
    if output_file.exists():
        logger.warning(f"Removing existing file: {output_file}")
        output_file.unlink()

    # Load draft translations
    drafts = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                draft = json.loads(line)
                drafts.append(draft)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

    logger.info(f"Loaded {len(drafts)} draft translations for domain: {domain}")

    # Track statistics
    stats = {
        "domain": domain,
        "total": len(drafts),
        "accepted_first_pass": 0,
        "accepted_after_regen": 0,
        "failed": 0,
        "avg_attempts": 0.0,
        "avg_acceptance_rate": 0.0,
        "avg_score": 0.0,
        "start_time": time.time()
    }

    total_attempts = 0
    total_acceptance_rate = 0.0
    total_score = 0.0

    # Verify each translation
    with tqdm(total=len(drafts), desc=f"Verifying {domain}", unit="persona") as pbar:
        for draft in drafts:
            persona_en = draft.get("persona_en", "")
            persona_kr = draft.get("persona_kr", "")

            if not persona_en or not persona_kr:
                logger.warning("Skipping persona with missing English or Korean text")
                stats["failed"] += 1
                pbar.update(1)
                continue

            # Verify and regenerate if needed
            final_translation, verification_stats = verify_and_regenerate(
                persona_en=persona_en,
                persona_kr=persona_kr,
                verification_model=verification_model,
                translation_model=translation_model,
                rate_limiter=rate_limiter,
                max_regenerations=max_regenerations,
                num_verification_instances=num_verification_instances,
                max_concurrency=max_concurrency
            )

            # Save verified translation
            verified_record = {
                "domain": domain,
                "persona_en": persona_en,
                "persona_kr": final_translation,
                "verification": verification_stats
            }

            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(verified_record, ensure_ascii=False) + '\n')

            # Update statistics
            if verification_stats["accepted"]:
                if verification_stats["attempts"] == 1:
                    stats["accepted_first_pass"] += 1
                else:
                    stats["accepted_after_regen"] += 1
            else:
                stats["failed"] += 1

            total_attempts += verification_stats["attempts"]
            final_stats = verification_stats.get("final_stats", {})
            total_acceptance_rate += final_stats.get("acceptance_rate", 0.0)
            total_score += final_stats.get("avg_score", 0.0)

            pbar.update(1)

    # Calculate averages
    stats["avg_attempts"] = total_attempts / stats["total"] if stats["total"] > 0 else 0.0
    stats["avg_acceptance_rate"] = total_acceptance_rate / stats["total"] if stats["total"] > 0 else 0.0
    stats["avg_score"] = total_score / stats["total"] if stats["total"] > 0 else 0.0
    stats["elapsed_time"] = time.time() - stats["start_time"]

    # Calculate first-pass acceptance rate
    first_pass_rate = (stats["accepted_first_pass"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
    total_accepted = stats["accepted_first_pass"] + stats["accepted_after_regen"]
    final_acceptance_rate = (total_accepted / stats["total"] * 100) if stats["total"] > 0 else 0.0

    # Log summary
    logger.info(f"""
    Domain: {domain}
    Total: {stats['total']}
    Accepted (1st pass): {stats['accepted_first_pass']} ({first_pass_rate:.1f}%)
    Accepted (after regen): {stats['accepted_after_regen']}
    Final acceptance rate: {final_acceptance_rate:.1f}%
    Failed: {stats['failed']}
    Avg attempts: {stats['avg_attempts']:.2f}
    Avg verification score: {stats['avg_score']:.2f}
    Elapsed time: {stats['elapsed_time']:.1f}s
    """)

    return stats


def main() -> None:
    """Main verification pipeline for Stages 2-3."""
    parser = argparse.ArgumentParser(
        description="Stages 2-3: Verify Korean translations with multi-instance evaluation"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/personas/kr_draft/"),
        help="Input directory with draft translations"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/personas/kr_verified/"),
        help="Output directory for verified translations"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="Model name for verification"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for verification (0.7 for diverse evaluations)"
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=60,
        help="Maximum requests per minute (0 = no limit)"
    )
    parser.add_argument(
        "--max-regenerations",
        type=int,
        default=3,
        help="Maximum regeneration attempts per persona"
    )
    parser.add_argument(
        "--verification-instances",
        type=int,
        default=5,
        help="Number of verification instances per attempt (default: 5)"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Maximum concurrent requests in verification batch (default: 5)"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Specific domains to verify (default: all)"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Initialize models
    logger.info("Initializing verification model (with structured output)...")
    verification_model = init_verification_model(
        model_name=args.model,
        temperature=args.temperature
    )

    logger.info("Initializing translation model (for regeneration)...")
    from langchain.chat_models import init_chat_model
    translation_model = init_chat_model(
        model=args.model,
        model_provider="google_genai",
        temperature=0.3  # Lower temperature for consistent regeneration
    )

    # Initialize rate limiter
    rate_limiter = None
    if args.rate_limit > 0:
        # Adjust for multiple verification instances
        # If 5 instances per persona and 60 calls/min, effective rate is ~12 personas/min
        effective_rate = args.rate_limit
        rate_limiter = RateLimiter(calls_per_minute=effective_rate)
        logger.info(f"Rate limiting enabled: {effective_rate} requests/minute")

    # Find input files
    if not args.input.exists():
        logger.error(f"Input directory does not exist: {args.input}")
        return

    input_files = sorted(args.input.glob("*_draft.jsonl"))

    if not input_files:
        logger.error(f"No draft JSONL files found in {args.input}")
        return

    # Filter by specified domains if provided
    if args.domains:
        domain_set = set(args.domains)
        input_files = [f for f in input_files if f.stem.replace("_draft", "") in domain_set]
        logger.info(f"Verifying specific domains: {args.domains}")

    logger.info(f"Found {len(input_files)} domain files to verify")

    # Verify each domain file
    all_stats = []

    for input_file in input_files:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {input_file.name}")
        logger.info(f"{'='*70}\n")

        stats = verify_domain_file(
            input_file=input_file,
            output_dir=args.output,
            verification_model=verification_model,
            translation_model=translation_model,
            rate_limiter=rate_limiter,
            max_regenerations=args.max_regenerations,
            num_verification_instances=args.verification_instances,
            max_concurrency=args.max_concurrency
        )

        all_stats.append(stats)

    # Print overall summary
    logger.info(f"\n{'='*70}")
    logger.info("OVERALL SUMMARY")
    logger.info(f"{'='*70}\n")

    total_personas = sum(s["total"] for s in all_stats)
    total_first_pass = sum(s["accepted_first_pass"] for s in all_stats)
    total_after_regen = sum(s["accepted_after_regen"] for s in all_stats)
    total_failed = sum(s["failed"] for s in all_stats)
    total_time = sum(s["elapsed_time"] for s in all_stats)

    logger.info(f"Total Personas: {total_personas}")
    logger.info(f"Accepted (1st pass): {total_first_pass}")
    logger.info(f"Accepted (after regen): {total_after_regen}")
    logger.info(f"Failed: {total_failed}")

    if total_personas > 0:
        first_pass_rate = (total_first_pass / total_personas) * 100
        final_acceptance_rate = ((total_first_pass + total_after_regen) / total_personas) * 100
        avg_attempts = sum(s["avg_attempts"] * s["total"] for s in all_stats) / total_personas
        avg_score = sum(s["avg_score"] * s["total"] for s in all_stats) / total_personas

        logger.info(f"First-pass acceptance rate: {first_pass_rate:.1f}%")
        logger.info(f"Final acceptance rate: {final_acceptance_rate:.1f}%")
        logger.info(f"Average attempts per persona: {avg_attempts:.2f}")
        logger.info(f"Average verification score: {avg_score:.2f}")

    logger.info(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    logger.info(f"\nVerified translations saved to: {args.output}")
    logger.info("Stages 2-3 complete! Proceed to Stage 4 for back-translation verification.")


if __name__ == "__main__":
    main()
