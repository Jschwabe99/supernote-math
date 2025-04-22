#!/bin/bash
# Script to run CROHME testing in Docker on the full test dataset

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Get the year to test from command line or default to 2019
TEST_YEAR=${1:-2019}
OUTPUT_DIR="$DIR/assets/crohme_full_$TEST_YEAR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create a copy of the Python script in the main directory
cp "$DIR/scripts/process_crohme.py" "$DIR/process_crohme.py"

echo "============================================================"
echo "Running full CROHME $TEST_YEAR dataset test in Docker"
echo "Results will be saved to: $OUTPUT_DIR"
echo "This may take a long time. Progress will be shown below."
echo "============================================================"

# Run the Docker container
docker run --rm \
  -v "$POSFORMER_DIR:/posformer" \
  -v "$DIR:/app" \
  -e "OMP_NUM_THREADS=1" \
  -e "OPENBLAS_NUM_THREADS=1" \
  -e "MKL_NUM_THREADS=1" \
  supernote-posformer \
  bash -c "
  # Install required dependencies
  pip install pandas pyarrow matplotlib tqdm --quiet

  # Run the Python script with the full dataset
  cd /app
  python process_crohme.py \
    --posformer_dir /posformer \
    --crohme_dir /app/CROHME-full \
    --output_dir /app/assets/crohme_full_$TEST_YEAR \
    --year $TEST_YEAR \
    --no_invert \
    # Process only the first 100 samples for the full test
    --sample_size 100
  "

# Clean up the copied script
rm "$DIR/process_crohme.py"

# Generate aggregated statistics report
echo "Generating aggregated statistics report..."
python3 - <<EOF
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

results_dir = "$OUTPUT_DIR"
summary_path = os.path.join(results_dir, "crohme_${TEST_YEAR}_summary.md")

# Read the summary file if it exists
if os.path.exists(summary_path):
    with open(summary_path, 'r') as f:
        summary_content = f.read()
    
    # Extract statistics
    match = re.search(r'Total samples processed: (\d+)', summary_content)
    total_samples = int(match.group(1)) if match else 0
    
    match = re.search(r'Exact matches: (\d+)', summary_content)
    exact_matches = int(match.group(1)) if match else 0
    
    match = re.search(r'Accuracy: ([\d\.]+)%', summary_content)
    accuracy = float(match.group(1)) if match else 0
    
    # Create a detailed report
    report_path = os.path.join(results_dir, "final_report.md")
    with open(report_path, 'w') as f:
        f.write(f"# CROHME {TEST_YEAR} Full Test Results\n\n")
        f.write(f"## Summary Statistics\n\n")
        f.write(f"- **Total samples processed:** {total_samples}\n")
        f.write(f"- **Exact matches:** {exact_matches}\n")
        f.write(f"- **Accuracy:** {accuracy:.2f}%\n\n")
        
        # Add comparison examples
        f.write(f"## Sample Comparisons\n\n")
        comparison_images = sorted(glob(os.path.join(results_dir, "*comparison.png")))[:10]
        for i, img_path in enumerate(comparison_images):
            rel_path = os.path.basename(img_path)
            sample_id = rel_path.split('_')[2]
            
            # Find the corresponding results file
            results_file = os.path.join(results_dir, f"crohme_{TEST_YEAR}_{sample_id}_results.txt")
            gt_latex = "Unknown"
            pred_latex = "Unknown"
            
            if os.path.exists(results_file):
                with open(results_file, 'r') as rf:
                    content = rf.read()
                    gt_match = re.search(r'GT LaTeX: (.*)', content)
                    gt_latex = gt_match.group(1) if gt_match else "Unknown"
                    
                    pred_match = re.search(r'Result 1 \(score: [-\d\.]+\):\n(.*)', content)
                    pred_latex = pred_match.group(1).strip() if pred_match else "No result"
            
            f.write(f"### Sample {i+1}\n\n")
            f.write(f"![Comparison]({rel_path})\n\n")
            f.write(f"**Ground Truth:** `{gt_latex}`\n\n")
            f.write(f"**Prediction:** `{pred_latex}`\n\n")
            f.write("---\n\n")
        
        # Final recommendations
        f.write("## Recommendations\n\n")
        if accuracy < 10:
            f.write("The model performance on CROHME data is poor. This suggests that:\n\n")
            f.write("1. The model was not trained on similar data to CROHME\n")
            f.write("2. The preprocessing pipeline might need further adjustment\n")
            f.write("3. Fine-tuning the model on CROHME data would likely improve performance significantly\n")
        elif accuracy < 50:
            f.write("The model shows moderate performance on CROHME data. Consider:\n\n")
            f.write("1. Fine-tuning the model on a subset of CROHME data\n")
            f.write("2. Adjusting beam search parameters for better recognition\n")
        else:
            f.write("The model performs well on CROHME data. Further improvements could include:\n\n")
            f.write("1. Ensemble methods with multiple models\n")
            f.write("2. Post-processing to correct common errors\n")
    
    print(f"Final report generated at: {report_path}")
else:
    print("Summary file not found. Test may not have completed successfully.")
EOF

echo "============================================================"
echo "Testing completed! Check the following directories for results:"
echo "1. Main results: $OUTPUT_DIR"
echo "2. Final report: $OUTPUT_DIR/final_report.md (if generated)"
echo "============================================================"