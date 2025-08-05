
# AVSR Comparisons: Evaluation and SNR Analysis

This folder provides a pipeline to evaluate ASR (Automatic Speech Recognition) outputs from different models and analyze their performance across SNR (Signal-to-Noise Ratio) conditions using two main scripts:

1. `evaluate_metrics.py` — Computes evaluation metrics for each model, modality, and SNR.
2. `analyze_snr.py` — Analyzes and visualizes the effect of SNR/SNR Gains on these metrics.

## Directory Structure


**Note:** The folders `AVEC`, `AV_relscore`, and `Auto-AVSR` in this repository contain SNR scores for the LRS3 dataset. However, you can create data in this structure for any other dataset by following the same folder and file organization.

Organize your data as follows:

```
Ablations/AVSR_Comparisons/
    evaluate_metrics.py
    analyze_snr.py
    avec/
        ref.txt
        AO/
            0.0.txt
            -5.0.txt
            ...
        AV/
            0.0.txt
            -5.0.txt
            ...
    AV_relscore/
        ref.txt
        AO/
            ...
        AV/
            ...
    auto-avsr/
        ref.txt
        AO/
            ...
        AV/
            ...
```

Each model folder must have a `ref.txt` (reference transcriptions). Each modality folder (`AO`, `AV`) must have hypothesis files named by SNR (e.g., `0.0.txt`, `-5.0.txt`).

## Step 1: Metric Evaluation (`evaluate_metrics.py`)

This script computes the following metrics for each model, modality, and SNR:
- WER (Word Error Rate)
- CER (Character Error Rate)
- BERTScore (semantic similarity)
- Semantic Similarity (sentence transformer)
- ViPS (Weighted Phonetic Score)

### Usage
1. **Install dependencies:**
   ```bash
   pip install pandas tqdm numpy transformers torch scikit-learn jiwer sentence-transformers
   ```
   Also ensure the `vis_phon` package is available for ViPS metric.

2. **Run the script:**
   ```bash
   python evaluate_metrics.py
   ```

3. **Outputs:**
   - Individual metric CSVs for each SNR in each modality folder (e.g., `avec/AO/metrics_0.0.csv`).
   - A combined `all_metrics.csv` in the main directory with all results.

## Step 2: SNR Analysis and Visualization (`analyze_snr.py`)

This script analyzes the effect of SNR on the computed metrics and generates comprehensive visualizations and summary statistics. It provides detailed insights into how different AVSR models perform across various noise conditions.

### Usage
1. **Ensure `all_metrics.csv` is present** (from the previous step).
2. **Run the script:**
   ```bash
   python analyze_snr.py
   ```

3. **Review Results:**
   - All visualizations are saved in the `snr_analysis/` folder
   - Summary statistics are printed to console, showing:
     - Average SNR gains for each model and metric
     - Best performing models for each metric
     - Detailed performance comparisons

## Workflow Summary

1. Prepare your data and directory structure as above.
2. Run `evaluate_metrics.py` to compute and save all metrics.
3. Run `analyze_snr.py` to analyze and visualize the results.

This pipeline enables robust, reproducible evaluation and comparison of AVSR models under varying noise conditions.
