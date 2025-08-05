# Pearson Correlations: ViPS and Other Metrics

This folder contains Pearson correlation analyses for the following metrics:
- ViPS (Weighted Phonetic Score)
- WER (Word Error Rate)
- CER (Character Error Rate)
- BERTScore (semantic similarity)
- Semantic Similarity (sentence transformer)

Correlations are computed for AV-Hubert outputs:
- AV (Audio-Visual) mode
- Video-only mode
- Two dataset sizes: 30 hours and 433 hours used in finetuning
- Two datasets: LRS2 and LRS3

## Using the Correlation Analysis Script

The `pearson_correlation_visualize.py` script analyzes correlations between ViPS and other metrics from a JSON results file.

### Usage
```bash
python pearson_correlation_visualize.py --json <path_to_results.json> --output <output_directory>
```

### Arguments
- `--json`: Path to the JSON file containing results with metrics
- `--output`: Directory where the output files will be saved (default: 'metric_correlation')

### Outputs
1. `metric_correlation_plots.png`: Individual scatter plots for each metric vs ViPS
2. `combined_correlation_plots.png`: All metrics plotted against ViPS in a single graph
3. `metrics_summary.md`: Summary of all metrics and their correlations with ViPS
4. `metric_correlations.csv`: Raw correlation data in CSV format

### Example
```bash
python pearson_correlation_visualize.py --json ./results.json --output ./correlation_results
```
```bash
python Ablations/pearson_correlations\(ViPS\ and\ Other\ metrics\)/pearson_correlation_visualize.py --json ./avsr_decoded_outputs/AV-Hubert_V_LRS2_vs_LRS3/VSR_lrs3_433h/hypo-244018.json --output ./Ablations/pearson_correlations\(ViPS\ and\ Other\ metrics\)/example_results/V_FT_LRS3_433h_Infer_LRS3
```
```bash
python Ablations/pearson_correlations\(ViPS\ and\ Other\ metrics\)/pearson_correlation_visualize.py --json ./avsr_decoded_outputs/AV-Hubert_AV_LRS2_vs_LRS3/AV_30h_lrs2/hypo-244018.json --output ./Ablations/pearson_correlations\(ViPS\ and\ Other\ metrics\)/example_results/AV_FT_LRS3_30h_Infer_LRS2
```

This allows comparison of how well each metric correlates with human or reference scores across different modalities, dataset sizes, and datasets.
