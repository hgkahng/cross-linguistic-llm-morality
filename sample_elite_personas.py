"""
Sample elite personas from PersonaHub dataset and organize by domain.

This script streams the elite_persona subset from HuggingFace's PersonaHub dataset
and samples personas from specific domains, saving them to JSONL files.
"""

import json
import logging
import argparse
import time
import signal
import threading
import itertools
from pathlib import Path
from typing import Dict, Optional

from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_NAME = "proj-persona/PersonaHub"
SUBSET_NAME = "elite_persona"
DOMAIN_KEY = "general domain (top 1 percent)"

# Target domains as specified in CLAUDE.md
TARGET_DOMAINS = [
    "Economics", "Law", "Philosophy", "History", "Sociology",
    "Environmental Science", "Mathematics", "Finance",
    "Engineering", "Computer Science"
]

DEFAULT_QUOTA_PER_DOMAIN = 1000
DEFAULT_OUTPUT_DIR = "data/personas/en"
DEFAULT_MAX_ROWS = None  # None = no limit
DEFAULT_SHUFFLE_SEED = 42
DEFAULT_SHUFFLE_BUFFER_SIZE = 10_000  # Smaller buffer for faster iteration
ESTIMATED_TOTAL_ROWS = 370_000_000
PROGRESS_CHECK_INTERVAL = 50_000  # Check progress every N rows
TIMEOUT_SECONDS = 300  # Stop if no new row received for 5 minutes


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_domain_name(domain: str) -> str:
    """
    Normalize domain name to lowercase for matching.

    Args:
        domain: Original domain name

    Returns:
        Lowercase domain name
    """
    return domain.lower().strip()


def match_domain(row_domain: str, target_domains: list) -> Optional[str]:
    """
    Match a row's domain against target domains.

    Args:
        row_domain: Domain string from dataset row
        target_domains: List of target domain names (normalized)

    Returns:
        Matched domain name if found, None otherwise
    """
    normalized_row = normalize_domain_name(row_domain)

    for domain in target_domains:
        if domain in normalized_row:
            return domain

    return None


def save_persona(output_dir: Path, domain: str, persona_data: dict) -> None:
    """
    Save a persona to a JSONL file.

    Args:
        output_dir: Output directory path
        domain: Domain name
        persona_data: Persona data dictionary
    """
    filepath = output_dir / f"{domain}.jsonl"

    with open(filepath, 'a', encoding='utf-8') as f:
        json.dump(persona_data, f, ensure_ascii=False)
        f.write("\n")
        f.flush()  # Ensure data is written to disk immediately


def print_final_statistics(domain_counts: Dict[str, int], quota_per_domain: int) -> None:
    """
    Print final collection statistics.

    Args:
        domain_counts: Dictionary mapping domains to collected counts
        quota_per_domain: Expected quota per domain
    """
    logger.info("\n" + "=" * 60)
    logger.info("FINAL COLLECTION STATISTICS")
    logger.info("=" * 60)

    for domain, count in sorted(domain_counts.items()):
        status = "✓" if count >= quota_per_domain else "✗"
        logger.info(f"{status} {domain:<30} {count:>6,} / {quota_per_domain:,}")

    total_collected = sum(domain_counts.values())
    total_needed = len(domain_counts) * quota_per_domain

    logger.info("=" * 60)
    logger.info(f"Total collected: {total_collected:,} / {total_needed:,}")

    if total_collected >= total_needed:
        logger.info("All quotas successfully filled!")
    else:
        logger.warning(f"Collection incomplete: {total_needed - total_collected:,} personas short")


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def sample_personas(
    output_dir: Path,
    target_domains: list,
    quota_per_domain: int,
    max_rows: Optional[int] = None,
    collect_available: bool = False,
    shuffle_buffer_size: int = DEFAULT_SHUFFLE_BUFFER_SIZE,
    use_take: bool = False,
    timeout_seconds: int = TIMEOUT_SECONDS,
    skip_rows: int = 0,
    watchdog_enabled: bool = True,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED
) -> Dict[str, int]:
    """
    Stream and sample personas from PersonaHub dataset.

    Args:
        output_dir: Directory to save persona files
        target_domains: List of target domain names
        quota_per_domain: Number of personas to collect per domain
        max_rows: Maximum rows to scan (None = no limit)
        collect_available: If True, stop when max_rows reached even if quotas not met
        shuffle_buffer_size: Size of shuffle buffer (smaller = faster but less random)
        use_take: If True, use take() to limit dataset before shuffling
        timeout_seconds: Stop if no new row received for this many seconds
        skip_rows: Skip first N rows (for resuming after a hang)
        watchdog_enabled: Enable watchdog thread to detect hangs

    Returns:
        Dictionary mapping domains to collected counts
    """
    # Normalize target domains for matching
    normalized_domains = [normalize_domain_name(d) for d in target_domains]

    # Initialize counters
    domain_counts = {domain: 0 for domain in normalized_domains}
    total_needed = len(normalized_domains) * quota_per_domain
    total_collected = 0

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.absolute()}")

    # Load dataset in streaming mode
    logger.info(f"Loading dataset: {DATASET_NAME} ({SUBSET_NAME})")
    logger.info("Establishing streaming connection...")

    try:
        dataset = load_dataset(
            DATASET_NAME,
            name=SUBSET_NAME,
            split="train",
            streaming=True
        )
        logger.info("✓ Dataset loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load dataset: {e}")
        raise

    # Apply take() if requested (limits dataset before shuffling)
    if use_take and max_rows:
        logger.info(f"Using take({max_rows:,}) to limit dataset upfront")
        dataset = dataset.take(max_rows)

    # Shuffle to distribute data across shards (unless disabled)
    if shuffle_buffer_size > 0:
        logger.info(f"Applying shuffle with buffer size: {shuffle_buffer_size:,}")
        logger.info("Note: Shuffle may take time to fill buffer before first row appears...")
        dataset = dataset.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_size)
        logger.info("✓ Shuffle configured")
    else:
        logger.info("Shuffling disabled (sequential processing)")

    # Test if iterator works by attempting to peek at first element
    logger.info("Testing iterator connection...")
    try:
        iterator = iter(dataset)
        logger.info("✓ Iterator created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create iterator: {e}")
        raise

    logger.info(f"Targeting {quota_per_domain:,} personas per domain")
    logger.info(f"Total domains: {len(normalized_domains)}")
    logger.info(f"Total needed: {total_needed:,} personas")
    if max_rows:
        logger.info(f"Maximum rows to scan: {max_rows:,}")
    if collect_available:
        logger.info("Mode: Collect as many as available (will stop at max_rows)")
    logger.info("\nStarting collection...\n")

    # Skip rows if resuming
    if skip_rows > 0:
        logger.info(f"Skipping first {skip_rows:,} rows...")
        iterator = iter(list(itertools.islice(iterator, skip_rows, None)))
        logger.info(f"✓ Skipped {skip_rows:,} rows")

    # Watchdog thread to detect hangs
    watchdog_stop = threading.Event()
    watchdog_triggered = threading.Event()
    last_progress = {'rows': 0, 'time': time.time()}

    class WatchdogTimeout(Exception):
        """Raised when watchdog detects a hang"""
        pass

    def watchdog_thread():
        """Monitor progress and forcibly exit if hung"""
        while not watchdog_stop.is_set():
            time.sleep(timeout_seconds)
            if watchdog_stop.is_set():
                break

            current_rows = last_progress['rows']
            elapsed = time.time() - last_progress['time']

            if elapsed >= timeout_seconds:
                logger.error(
                    f"\n{'='*60}\n"
                    f"WATCHDOG: No progress for {elapsed:.0f}s (stuck at row {current_rows:,})\n"
                    f"The dataset stream appears hung. Forcing exit.\n"
                    f"{'='*60}"
                )
                watchdog_triggered.set()

                # Print statistics before exiting
                logger.info("\nFinal statistics before forced exit:")
                for domain, count in sorted(domain_counts.items()):
                    logger.info(f"  {domain:<30} {count:>6,}")
                logger.info(f"\nTotal collected: {sum(domain_counts.values()):,}")

                # Force exit (iterator is hung, can't break out normally)
                logger.warning("Watchdog forcing program exit (iterator hung)...")
                import os
                os._exit(0)  # Forceful exit - iterator is blocking

    if watchdog_enabled:
        watchdog = threading.Thread(target=watchdog_thread, daemon=True)
        watchdog.start()
        logger.info(f"✓ Watchdog enabled (timeout: {timeout_seconds}s)")

    # Process rows with progress bar
    logger.info("Waiting for first row from dataset stream...")
    logger.info("(This may take a while if shuffle buffer needs to fill)")

    pbar = tqdm(
        iterator,
        unit=" rows",
        desc="Scanning dataset",
        total=ESTIMATED_TOTAL_ROWS
    )

    rows_processed = 0
    last_collected = 0
    last_check_row = 0
    last_row_time = time.time()
    first_row_received = False

    try:
        for row in pbar:
            # Check if watchdog triggered
            if watchdog_triggered.is_set():
                logger.warning("Stopping due to watchdog timeout")
                break
            current_time = time.time()

            # Log when first row is received
            if not first_row_received:
                logger.info(f"\n✓ First row received! (waited {current_time - last_row_time:.1f}s)")
                first_row_received = True

            # Timeout detection: check if stuck waiting for next row
            if current_time - last_row_time > timeout_seconds:
                logger.warning(
                    f"\nTimeout: No new row received for {timeout_seconds} seconds. "
                    f"The stream may be stuck. Stopping."
                )
                break

            last_row_time = current_time

            # Update watchdog progress
            last_progress['rows'] = rows_processed
            last_progress['time'] = current_time

            try:
                rows_processed += 1

                # Check max_rows limit
                if max_rows and rows_processed >= max_rows:
                    logger.info(f"\nReached maximum row limit ({max_rows:,}). Stopping.")
                    break

                # Validate row structure
                if not isinstance(row, dict):
                    continue

                # Extract domain
                raw_domain = row.get(DOMAIN_KEY)
                if not raw_domain:
                    continue

                # Match against target domains
                matched_domain = match_domain(raw_domain, normalized_domains)

                if matched_domain and domain_counts[matched_domain] < quota_per_domain:
                    # Save persona data
                    persona_data = {
                        'domain': matched_domain,
                        'persona': row.get('persona', ''),
                    }

                    save_persona(output_dir, matched_domain, persona_data)

                    domain_counts[matched_domain] += 1
                    total_collected += 1

                    # Update progress bar
                    pbar.set_postfix({
                        'collected': f"{total_collected}/{total_needed}",
                        'rows': f"{rows_processed:,}"
                    })

                # Check progress periodically
                if rows_processed - last_check_row >= PROGRESS_CHECK_INTERVAL:
                    collected_since_check = total_collected - last_collected
                    if collected_since_check == 0 and collect_available:
                        logger.info(f"\nNo progress in last {PROGRESS_CHECK_INTERVAL:,} rows. Stopping.")
                        break
                    last_collected = total_collected
                    last_check_row = rows_processed

                # Check stopping condition
                if total_collected >= total_needed:
                    if all(count >= quota_per_domain for count in domain_counts.values()):
                        logger.info("\nAll domain quotas filled! Stopping early.")
                        break

            except Exception as e:
                # Skip problematic rows without crashing
                logger.debug(f"Skipping row {rows_processed} due to error: {e}")
                continue

    except KeyboardInterrupt:
        logger.warning("\nStopped by user (Ctrl+C)")

    finally:
        pbar.close()
        # Stop watchdog thread
        if watchdog_enabled:
            watchdog_stop.set()

    return domain_counts


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Sample elite personas from PersonaHub dataset and organize by domain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--quota",
        type=int,
        default=DEFAULT_QUOTA_PER_DOMAIN,
        metavar="N",
        help="Number of personas to collect per domain"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        metavar="PATH",
        help="Output directory for persona files"
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        metavar="N",
        help="Maximum number of rows to scan (default: no limit)"
    )

    parser.add_argument(
        "--collect-available",
        action="store_true",
        help="Collect as many personas as available within max-rows limit (don't require all quotas to be filled)"
    )

    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=DEFAULT_SHUFFLE_BUFFER_SIZE,
        metavar="N",
        help="Shuffle buffer size (smaller = faster iteration, less random; larger = slower, more random)"
    )

    parser.add_argument(
        "--use-take",
        action="store_true",
        help="Use take() to limit dataset before shuffling (can help avoid shard blocking issues)"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=TIMEOUT_SECONDS,
        metavar="SEC",
        help="Stop if no new row received for this many seconds (detects hung streams)"
    )

    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Skip shuffling (faster but sequential, may exhaust some domains quickly)"
    )

    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Test mode: only fetch first 10 rows to diagnose connection issues"
    )

    parser.add_argument(
        "--skip-rows",
        type=int,
        default=0,
        metavar="N",
        help="Skip first N rows (for resuming after a hang)"
    )

    parser.add_argument(
        "--no-watchdog",
        action="store_true",
        help="Disable watchdog thread (not recommended)"
    )

    return parser.parse_args()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main execution function.
    """
    # Load environment variables
    load_dotenv()

    # Parse command-line arguments
    args = parse_arguments()

    logger.info("=" * 60)
    logger.info("PersonaHub Elite Persona Sampler")
    logger.info("=" * 60)

    try:
        # Test mode: diagnose connection
        if args.test_connection:
            logger.info("\n*** TEST MODE: Attempting to fetch first 10 rows ***\n")

            dataset = load_dataset(
                DATASET_NAME,
                name=SUBSET_NAME,
                split="train",
                streaming=True
            )
            logger.info("Dataset loaded")

            # Try without shuffle first
            logger.info("\nTest 1: No shuffle, take(10)")
            try:
                count = 0
                start = time.time()
                for i, row in enumerate(dataset.take(10)):
                    elapsed = time.time() - start
                    logger.info(f"  Row {i+1}/10 received (after {elapsed:.1f}s)")
                    count += 1
                logger.info(f"✓ Successfully fetched {count} rows without shuffle")
            except Exception as e:
                logger.error(f"✗ Failed: {e}")

            # Try with small shuffle
            logger.info("\nTest 2: Small shuffle (buffer=100), take(10)")
            try:
                dataset_shuffled = dataset.shuffle(seed=42, buffer_size=100)
                count = 0
                start = time.time()
                for i, row in enumerate(dataset_shuffled.take(10)):
                    elapsed = time.time() - start
                    logger.info(f"  Row {i+1}/10 received (after {elapsed:.1f}s)")
                    count += 1
                logger.info(f"✓ Successfully fetched {count} rows with shuffle")
            except Exception as e:
                logger.error(f"✗ Failed: {e}")

            logger.info("\n*** TEST MODE COMPLETE ***")
            return

        # Normal mode: Sample personas
        domain_counts = sample_personas(
            output_dir=Path(args.output_dir),
            target_domains=TARGET_DOMAINS,
            quota_per_domain=args.quota,
            max_rows=args.max_rows,
            collect_available=args.collect_available,
            shuffle_buffer_size=0 if args.no_shuffle else args.shuffle_buffer_size,
            use_take=args.use_take,
            timeout_seconds=args.timeout,
            skip_rows=args.skip_rows,
            watchdog_enabled=not args.no_watchdog
        )

        # Print final statistics
        print_final_statistics(domain_counts, args.quota)

    except Exception as e:
        logger.error(f"✗ Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
