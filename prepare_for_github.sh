#!/bin/bash

# GitHub Repository Preparation Script
# Organizes files into proper structure for GitHub upload

echo "=========================================="
echo "Preparing Repository for GitHub Upload"
echo "=========================================="
echo

# Create directory structure
echo "Creating directory structure..."
mkdir -p src
mkdir -p docs
mkdir -p scripts
mkdir -p results_sample

# Move source files
echo "Moving source files to src/..."
cp rct_ft_single.py src/ 2>/dev/null || echo "  rct_ft_single.py not found"
cp multi_model_runner.py src/ 2>/dev/null || echo "  multi_model_runner.py not found"
cp analyze_results.py src/ 2>/dev/null || echo "  analyze_results.py not found"
cp reevaluate_fixed.py src/ 2>/dev/null || echo "  reevaluate_fixed.py not found"
cp diagnose_models.py src/ 2>/dev/null || echo "  diagnose_models.py not found"
cp fix_evaluation.py src/ 2>/dev/null || echo "  fix_evaluation.py not found"

# Move documentation
echo "Moving documentation to docs/..."
cp ALTERNATIVE_MODELS.md docs/ 2>/dev/null || echo "  ALTERNATIVE_MODELS.md not found"
cp DIAGNOSIS_AND_RESULTS.md docs/ 2>/dev/null || echo "  DIAGNOSIS_AND_RESULTS.md not found"
cp FEASIBILITY_ASSESSMENT.md docs/ 2>/dev/null || echo "  FEASIBILITY_ASSESSMENT.md not found"
cp NEW_CONFIG_SUMMARY.md docs/ 2>/dev/null || echo "  NEW_CONFIG_SUMMARY.md not found"
cp QUICK_START.md docs/ 2>/dev/null || echo "  QUICK_START.md not found"
cp OPTION_A_SAFE_CONFIG.md docs/ 2>/dev/null || echo "  OPTION_A_SAFE_CONFIG.md not found"
cp YOUR_MODEL_CONFIG.md docs/ 2>/dev/null || echo "  YOUR_MODEL_CONFIG.md not found"
cp README_MULTI_MODEL.md docs/ 2>/dev/null || echo "  README_MULTI_MODEL.md not found"
cp FINAL_SUMMARY.md docs/ 2>/dev/null || echo "  FINAL_SUMMARY.md not found"
cp START_HERE.md docs/ 2>/dev/null || echo "  START_HERE.md not found"

# Move scripts
echo "Moving scripts to scripts/..."
cp check_progress.sh scripts/ 2>/dev/null || echo "  check_progress.sh not found"

# Copy sample results (remove sensitive data)
echo "Copying sample results..."
if [ -d "results" ]; then
    # Copy only the structure, not actual results (remove if you want to include results)
    cp results/combined_results_*.json results_sample/ 2>/dev/null || echo "  No combined results found"
    cp results/Llama-3.2-3B*.json results_sample/ 2>/dev/null || echo "  No Llama results found"
    cp results/Mistral-7B*.json results_sample/ 2>/dev/null || echo "  No Mistral results found"
fi

# Create sample data structure (but don't include actual data)
echo "Creating data directory structure..."
mkdir -p data
echo "# Data Directory

Place your clinical dataset here:
- Trial 9/trial9.csv

**Note:** Actual data is not included in this repository for privacy reasons.

## Expected Data Format

CSV file with the following columns:
- X_age_0d: Patient age
- X_country_0d: Country
- Treatment: Treatment type (e.g., Oxytocin)
- X_pulse_0d: Pulse rate
- X_bp_sys_0d: Systolic blood pressure
- X_bp_dia_0d: Diastolic blood pressure
- X_hb_before_delivery_0d: Hemoglobin level
- X_GestationWeeks_0d: Gestational weeks
- X_birth_weight_0d: Birth weight
- YP_manual_removal: Target variable (Yes/No)
" > data/README.md

# Create results README
echo "Creating results README..."
echo "# Results Directory

This directory contains fine-tuning results for each model.

## Files Generated

- \`combined_results_*.json\`: Aggregated results from all models
- \`{ModelName}_*.json\`: Individual model results with metrics
- \`*_FIXED.json\`: Re-evaluated results with robust parsing

## Metrics Included

- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1 Score
- Confusion Matrix (TP, TN, FP, FN)

## Sample Results

See \`results_sample/\` for example output format.
" > results/README.md 2>/dev/null || mkdir -p results && echo "# Results Directory" > results/README.md

echo
echo "=========================================="
echo "Repository Structure Created!"
echo "=========================================="
echo
echo "Directory structure:"
tree -L 2 -I '__pycache__|*.pyc|finetuned-*|.cache' . 2>/dev/null || ls -R .

echo
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo
echo "1. Review the organized structure"
echo "2. Edit README.md with your information:"
echo "   - Your name"
echo "   - Your email"
echo "   - Your institution"
echo "   - GitHub repository URL"
echo
echo "3. Initialize git repository:"
echo "   cd $(pwd)"
echo "   git init"
echo "   git add ."
echo "   git commit -m 'Initial commit: Multi-model LLM comparison'"
echo
echo "4. Create GitHub repository and push:"
echo "   git remote add origin https://github.com/yourusername/clinical-llm-comparison.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo
echo "5. Optional: Add GitHub Actions for CI/CD"
echo "   (See docs for template)"
echo
echo "=========================================="
echo "IMPORTANT: Review .gitignore"
echo "=========================================="
echo
echo "The .gitignore file is set to EXCLUDE:"
echo "  - Actual data files (data/*.csv)"
echo "  - Fine-tuned model weights (finetuned-*/)"
echo "  - Training logs (*.log)"
echo
echo "If you want to include results, edit .gitignore to uncomment:"
echo "  # results/"
echo
echo "=========================================="
