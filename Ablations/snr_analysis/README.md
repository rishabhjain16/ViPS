# SNR Metric Analysis

This tool analyzes and visualizes the performance of various speech recognition metrics (ViPS, WER, CER, Semantic Similarity, and BERTScore) across different Signal-to-Noise Ratio (SNR) levels.

## Features

- Processes JSON files containing reference-hypothesis pairs for different SNR levels
- Calculates multiple evaluation metrics:
  - ViPS (Visual Phonetic Score)
  - WER (Word Error Rate)
  - CER (Character Error Rate)
  - Semantic Similarity
  - BERTScore
- Generates visualizations:
  - Line plots showing average metric values across SNR levels
  - Combined analysis plots showing correlations and R² values with ViPS
- Saves detailed results and summaries in JSON format

## Usage

```bash
python snr_metric_analysis.py --snr_files FILE1 FILE2 ... --snr_levels SNR1 SNR2 ... [--output_dir DIR] [--weight_method METHOD]
```

### Arguments

- `--snr_files`: List of JSON files containing reference-hypothesis pairs for different SNR levels
- `--snr_levels`: SNR levels corresponding to each file (must match the order of --snr_files)
- `--output_dir`: Directory to save analysis outputs (default: "snr_analysis_output")
- `--weight_method`: Method for phonetic feature weighting (choices: "both", "entropy", "distinctiveness", default: "both")

### Examples

1. Basic example:
```bash
python snr_metric_analysis.py \
  --snr_files data/snr_0db.json data/snr_5db.json data/snr_10db.json \
  --snr_levels 0 5 10 \
  --output_dir analysis_results \
  --weight_method both
```

2. Real-world example with AV-Hubert outputs:
```bash
python Ablations/snr_analysis/snr_metric_analysis.py \
  --snr_files ./avsr_decoded_outputs/AV-Hubert_V_LRS2_vs_LRS3/VSR_lrs3_433h/hypo-244018.json \
              ./avsr_decoded_outputs/AV-Hubert_V_LRS2_vs_LRS3/VSR_lrs3_30h/hypo-244018.json \
  --snr_levels 1 2 \
  --output_dir ./Ablations/snr_analysis/test_example
```

## Input Format

The input JSON files should contain reference-hypothesis pairs in one of these formats:

1. List format:
```json
[
  {
    "reference": "example text one",
    "hypothesis": "example text 1"
  },
  {
    "ref": "example text two",
    "hyp": "example text 2"
  }
]
```

2. Dictionary format:
```json
{
  "reference": ["example text one", "example text two"],
  "hypothesis": ["example text 1", "example text 2"]
}
```

## Output

The script generates:

1. Visualizations:
   - `metric_values_line_plot.png`: Line plot showing average metric values across SNR levels
   - `combined_analysis_plot.png`: Plot showing correlations and R² values between ViPS and other metrics

2. Results:
   - Individual JSON files for each SNR level in the `results` subdirectory
   - `summary_across_snr.json`: Summary statistics across all SNR levels

## Dependencies

- numpy
- matplotlib
- seaborn
- pandas
- tqdm
- ViPS evaluator
