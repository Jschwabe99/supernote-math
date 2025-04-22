#!/usr/bin/env python3
"""
Script to run multiple tests on the PosFormer math recognition pipeline.

This script:
1. Tests with simulated handwritten strokes
2. Tests with the full test page of examples
3. Shows recognized equations and their solutions

Usage:
  python run_posformer_tests.py [--no-show] [--model MODEL_PATH]
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run multiple tests on the PosFormer pipeline')
    parser.add_argument('--no-show', action='store_true', help='Do not display results')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    args = parser.parse_args()
    
    # Define paths
    scripts_dir = Path(__file__).resolve().parent
    project_dir = scripts_dir.parent
    
    # Test with simulated strokes
    print("\n======== Testing with simulated strokes ========\n")
    cmd = [
        sys.executable,
        str(scripts_dir / "test_posformer_pipeline.py")
    ]
    if args.model:
        cmd.extend(["--model", args.model])
    if args.no_show:
        cmd.append("--no-show")
    
    try:
        subprocess.run(cmd, cwd=str(project_dir), check=False)
    except Exception as e:
        print(f"Error running test_posformer_pipeline.py: {e}")
    
    # Test with example page
    print("\n======== Testing with full example page ========\n")
    cmd = [
        sys.executable,
        str(scripts_dir / "test_with_examples.py"),
        "--test-page"
    ]
    if args.model:
        cmd.extend(["--model", args.model])
    if args.no_show:
        cmd.append("--no-show")
    
    try:
        subprocess.run(cmd, cwd=str(project_dir), check=False)
    except Exception as e:
        print(f"Error running test_with_examples.py: {e}")
    
    # Display summary
    print("\n======== All tests completed ========\n")
    print("The PosFormer pipeline has been tested with:")
    print("1. Simulated strokes (generated quadratic equation)")
    print("2. Full page of handwritten examples")
    print("\nIf you have a specific image to test, you can run:")
    print(f"  python {scripts_dir}/test_with_examples.py --image /path/to/image.png")
    print("\nTo test with your own handwritten input, you can use the app module directly:")
    print("  from app.pipeline import MathRecognitionPipeline, Stroke")
    print("  pipeline = MathRecognitionPipeline()")
    print("  strokes = [...]  # Your handwritten strokes")
    print("  result = pipeline.process_strokes(strokes)")
    print("  print(result['latex'])")

if __name__ == "__main__":
    main()