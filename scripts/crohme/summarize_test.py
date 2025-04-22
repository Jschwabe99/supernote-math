#!/usr/bin/env python3
"""
Script to summarize test results from CROHME dataset
"""
import os
import re
import glob
import argparse

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Summarize CROHME test results')
    parser.add_argument('--test_dir', type=str, default='assets/crohme_full_2019',
                        help='Directory containing test results')
    parser.add_argument('--sample_size', type=int, default=0,
                        help='Number of samples to process (0 for all available)')
    args = parser.parse_args()
    
    # Get the test results directory
    test_dir = args.test_dir
    if not os.path.isabs(test_dir):
        test_dir = os.path.abspath(test_dir)
    
    # Extract year from directory name if possible
    test_year = 'unknown'
    match = re.search(r'crohme_(\d+)', test_dir)
    if match:
        test_year = match.group(1)
    else:
        match = re.search(r'(\d{4})', test_dir)
        if match:
            test_year = match.group(1)
    
    print(f"Analyzing CROHME {test_year} test results in {test_dir}")
    
    # Find all result files
    result_files = glob.glob(os.path.join(test_dir, "*_results.txt"))
    print(f"Found {len(result_files)} result files")
    
    # Find all processed samples
    processed_files = glob.glob(os.path.join(test_dir, "*_processed.png"))
    print(f"Found {len(processed_files)} processed images")
    
    # Find all comparison images
    comparison_files = glob.glob(os.path.join(test_dir, "*_comparison.png"))
    print(f"Found {len(comparison_files)} comparison images")
    
    # Analyze results
    results = []
    
    # Limit the number of files to analyze if specified
    if args.sample_size > 0 and len(result_files) > args.sample_size:
        result_files = result_files[:args.sample_size]
        print(f"Limiting analysis to {args.sample_size} samples")
    
    for result_file in sorted(result_files):
        sample_id = os.path.basename(result_file).split('_')[2]
        print(f"Processing sample {sample_id}...")
        
        # Read the result file
        with open(result_file, 'r') as f:
            content = f.read()
        
        # Extract ground truth and prediction
        gt_match = re.search(r'GT LaTeX: (.*)', content)
        gt_latex = gt_match.group(1).strip() if gt_match else "Unknown"
        
        pred_match = re.search(r'Result 1 \(score: ([-\d\.]+)\):\n(.*)', content)
        if pred_match:
            score = float(pred_match.group(1))
            pred_latex = pred_match.group(2).strip()
        else:
            score = 0.0
            pred_latex = "No result"
        
        # Check for exact match
        is_match = (gt_latex.strip() == pred_latex.strip())
        
        # Add to results
        results.append({
            'sample_id': sample_id,
            'gt_latex': gt_latex,
            'pred_latex': pred_latex,
            'score': score,
            'match': is_match
        })
    
    # Calculate statistics
    match_count = sum(1 for r in results if r['match'])
    total_count = len(results)
    accuracy = match_count / total_count * 100 if total_count > 0 else 0
    
    print(f"\nStatistics:")
    print(f"- Total samples processed: {total_count}")
    print(f"- Exact matches: {match_count}")
    print(f"- Accuracy: {accuracy:.2f}%")
    
    # Create a markdown summary
    summary_path = os.path.join(test_dir, f"summary_report.md")
    with open(summary_path, 'w') as f:
        f.write(f"# CROHME {test_year} Test Results Summary\n\n")
        f.write(f"## Statistics\n\n")
        f.write(f"- **Total samples processed:** {total_count}\n")
        f.write(f"- **Exact matches:** {match_count}\n")
        f.write(f"- **Accuracy:** {accuracy:.2f}%\n\n")
        
        # Add results table
        f.write("## Results\n\n")
        f.write("| # | Ground Truth | Prediction | Score | Match |\n")
        f.write("|---|-------------|------------|-------|-------|\n")
        
        for i, result in enumerate(results[:50]):  # Show first 50 for readability
            gt_display = result['gt_latex']
            if len(gt_display) > 30:
                gt_display = gt_display[:27] + "..."
            gt_display = gt_display.replace("|", "\\|")
            
            pred_display = result['pred_latex']
            if len(pred_display) > 30:
                pred_display = pred_display[:27] + "..."
            pred_display = pred_display.replace("|", "\\|")
            
            f.write(f"| {result['sample_id']} | `{gt_display}` | `{pred_display}` | {result['score']:.4f} | {result['match']} |\n")
        
        if len(results) > 50:
            f.write(f"\n*Table truncated. Showing 50/{len(results)} results.*\n\n")
        
        # Add comparison examples
        f.write(f"## Sample Comparisons\n\n")
        comparison_images = sorted(comparison_files)[:10]  # First 10 comparisons
        
        for i, img_path in enumerate(comparison_images):
            rel_path = os.path.basename(img_path)
            sample_id = rel_path.split('_')[2]
            
            # Find the corresponding result
            result = next((r for r in results if r['sample_id'] == sample_id), None)
            
            if result:
                f.write(f"### Sample {sample_id}\n\n")
                f.write(f"![Comparison]({rel_path})\n\n")
                f.write(f"**Ground Truth:** `{result['gt_latex']}`\n\n")
                f.write(f"**Prediction:** `{result['pred_latex']}`\n\n")
                f.write("---\n\n")
        
        # Final recommendations
        f.write("## Recommendations\n\n")
        if accuracy < 10:
            f.write("The model performance on CROHME data is poor. This suggests that:\n\n")
            f.write("1. The model was not trained on similar data to CROHME\n")
            f.write("2. The preprocessing pipeline might need further adjustment\n")
            f.write("3. Fine-tuning the model on CROHME data would likely improve performance significantly\n\n")
            f.write("However, the current preprocessing pipeline properly handles the CROHME images\n")
            f.write("without inverting them, which is correct since they're already white on black.\n")
        elif accuracy < 50:
            f.write("The model shows moderate performance on CROHME data. Consider:\n\n")
            f.write("1. Fine-tuning the model on a subset of CROHME data\n")
            f.write("2. Adjusting beam search parameters for better recognition\n")
        else:
            f.write("The model performs well on CROHME data. Further improvements could include:\n\n")
            f.write("1. Ensemble methods with multiple models\n")
            f.write("2. Post-processing to correct common errors\n")
    
    print(f"\nSummary report generated at: {summary_path}")

if __name__ == "__main__":
    main()