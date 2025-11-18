#!/usr/bin/env python3
"""
Script to run examples from the root directory
Usage: python run_example.py <example_number>
Example: python run_example.py 1
"""

import sys
import os

if len(sys.argv) < 2:
    print("Usage: python run_example.py <example_number>")
    print("Example: python run_example.py 1")
    print("\nAvailable examples:")
    print("  1 - Classification with Multiple Algorithms")
    print("  2 - Regression Analysis")
    print("  3 - Clustering Analysis")
    print("  4 - Dimensionality Reduction")
    print("  5 - Neural Networks")
    print("  6 - Model Selection and Hyperparameter Tuning")
    print("  7 - Complete ML Pipeline")
    sys.exit(1)

example_num = sys.argv[1]
example_file = f"examples/0{example_num}_*.py"

# Find the example file
import glob
matching_files = glob.glob(example_file)

if not matching_files:
    print(f"Error: Example {example_num} not found")
    sys.exit(1)

example_path = matching_files[0]
print(f"Running {example_path}...\n")

# Execute the example
with open(example_path, 'r') as f:
    code = f.read()
    # Remove the sys.path manipulation from examples
    code = code.replace("sys.path.append('..')", "")
    code = code.replace("sys.path.insert(0, os.path.abspath('..'))", "")
    exec(code)
