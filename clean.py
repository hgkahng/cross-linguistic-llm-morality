import json
import argparse
from pathlib import Path


def clean_response_json(input_path: str, output_path: str = None, verbose: bool = False):
    """
    Remove redundant 'existing research' outer key and 'difficulty' inner key.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file (if None, overwrites input)
    """
    # Read the original JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Remove outer 'existing research' key
    if 'existing research' in data:
        data = data['existing research']
    
    # Remove 'difficulty' key from each scenario
    for scenario_key, scenario_data in data.items():
        if 'difficulty' in scenario_data:
            del scenario_data['difficulty']
    
    # Write cleaned data
    output_path = input_path if output_path is None else output_path
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    if verbose:
        print(f"Cleaned JSON saved to: {output_path}")
    
    return data


def clean_all_responses(input_dir='.', pattern='Person_*.json', verbose: bool = False):
    """
    Clean all response JSON files in a directory.
    
    Args:
        input_dir: Directory containing JSON files
        pattern: Glob pattern to match files
    """
    input_path = Path(input_dir)
    json_files = sorted(list(input_path.glob(pattern)))
    
    print(f"Found {len(json_files):,} files matching pattern '{pattern}'")
    
    for json_file in json_files:
        if verbose:
            print(f"\nProcessing: {json_file}")
        clean_response_json(json_file)
    
    print(f"\n✓ All {len(json_files):,} files cleaned!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Temporary file to re-format output structure.")
    parser.add_argument('--directory', type=str, required=True, help="")
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    clean_all_responses(input_dir=args.directory, verbose=args.verbose)
    