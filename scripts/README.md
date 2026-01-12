# Scripts

This directory contains utility scripts, data processing scripts, and other helper scripts that don't belong in the main `src/` package.

## Usage

- Store data processing and transformation scripts
- Keep utility scripts for automation and workflow tasks
- Place one-off analysis scripts
- Include deployment and environment setup scripts

## Best Practices

- Make scripts executable (add shebang line and executable permissions)
- Include clear documentation at the top of each script
- Add command-line argument parsing for flexibility
- Implement logging for better debugging
- Keep scripts focused on a single task
- Use consistent naming conventions
- Consider organizing scripts in subdirectories by purpose

## Example Scripts

- `data_download.py` - Download data from external sources
- `preprocess_data.py` - Clean and prepare data for analysis
- `generate_figures.py` - Create visualizations from results
- `run_experiments.sh` - Execute a series of experiments
- `setup_environment.sh` - Set up development environment
- `deploy_model.py` - Deploy trained models to production

## Difference from `src/`

The `scripts/` directory is for standalone scripts that are used to perform specific tasks, while the `src/` directory contains the core reusable code organized as a Python package. Scripts typically import and use functionality from the `src/` package.

Example:
```python
#!/usr/bin/env python3
"""
Script to preprocess raw data files.

Usage:
    python preprocess_data.py --input data/raw/ --output data/processed/
"""

import argparse
import logging
from pathlib import Path

# Import functionality from the src package
from src.data import preprocessing

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Preprocess raw data files')
    parser.add_argument('--input', required=True, help='Input directory with raw data')
    parser.add_argument('--output', required=True, help='Output directory for processed data')
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Process the data
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logging.info(f"Processing data from {input_dir} to {output_dir}")
    preprocessing.process_files(input_dir, output_dir)
    logging.info("Processing complete")

if __name__ == "__main__":
    main()
```