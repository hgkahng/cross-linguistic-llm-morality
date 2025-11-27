#!/usr/bin/env python3
"""
Download JSONL files from HuggingFace PersonaHub ElitePersonas directory.
"""

import os
import argparse
import requests
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import time


def get_file_list(repo_id: str = "proj-persona/PersonaHub",
                  folder: str = "ElitePersonas") -> List[str]:
    """Get list of files in the HuggingFace repository folder."""
    api_url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main/{folder}"

    try:
        response = requests.get(api_url)
        response.raise_for_status()

        files = []
        for item in response.json():
            if item['type'] == 'file' and item['path'].endswith('.jsonl'):
                files.append(item['path'])

        return files
    except requests.exceptions.RequestException as e:
        print(f"Error fetching file list: {e}")
        return []


def download_file(repo_id: str, file_path: str, output_dir: Path,
                  max_retries: int = 3) -> bool:
    """Download a single file from HuggingFace."""
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file_path}"
    filename = os.path.basename(file_path)
    output_path = output_dir / filename

    # Skip if file already exists
    if output_path.exists():
        print(f"Skipping {filename} (already exists)")
        return True

    for attempt in range(max_retries):
        try:
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Get total file size
            total_size = int(response.headers.get('content-length', 0))

            # Download with progress bar
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            print(f"Successfully downloaded {filename}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed for {filename}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to download {filename} after {max_retries} attempts")
                return False
        except Exception as e:
            print(f"Unexpected error downloading {filename}: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Download JSONL files from HuggingFace PersonaHub ElitePersonas"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/elite_personas",
        help="Output directory for downloaded files (default: data/elite_personas)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="proj-persona/PersonaHub",
        help="HuggingFace repository ID (default: proj-persona/PersonaHub)"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="ElitePersonas",
        help="Folder name in the repository (default: ElitePersonas)"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to download (default: all)"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter files by keyword in filename"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Fetching file list from {args.repo_id}/{args.folder}...")

    # Get list of files
    files = get_file_list(args.repo_id, args.folder)

    if not files:
        print("No JSONL files found or error fetching file list")
        return

    print(f"Found {len(files)} JSONL files")

    # Apply filter if specified
    if args.filter:
        files = [f for f in files if args.filter.lower() in f.lower()]
        print(f"After filtering: {len(files)} files match '{args.filter}'")

    # Limit number of files if specified
    if args.max_files:
        files = files[:args.max_files]
        print(f"Limiting to {len(files)} files")

    # Download files
    successful = 0
    failed = 0

    for file_path in files:
        if download_file(args.repo_id, file_path, output_dir):
            successful += 1
        else:
            failed += 1

    # Summary
    print("\n" + "=" * 50)
    print(f"Download complete!")
    print(f"Successfully downloaded: {successful} files")
    if failed > 0:
        print(f"Failed downloads: {failed} files")
    print(f"Files saved to: {output_dir}")


if __name__ == "__main__":
    main()