# Calculating ViPS Score on MaFi Dataset 

## Citation

We use the MaFi dataset for our experiments with ViPS as cited below:

> A. Krason, Y. Zhang, H. Man, and G. Vigliocco, "Mouth and facial
informativeness norms for 2276 english words," Behavior Research
Methods, vol. 56, no. 5, pp. 4786–4801, August 2024.

## MaFi Dataset Processing - to create ViPS and other scores

### Overview
`calculate_mafi_parallel.py` is specifically designed to work with the MaFi dataset files:
- `Norms/MaFI_Combined.csv`
- `Behavioural Data/DataComplete_IPA_Dist.xlsx`

The script calculates various metrics (ViPS scores, phonological distances, and optional metrics like WER, CER, BLEU) for the word pairs in these datasets.

### Implementation Note
The parallel processing and checkpointing functionality was implemented with assistance from Claude Sonnet 3.5 (Anthropic's LLM) to address performance challenges with the large MaFi dataset. These optimizations:
- Enable parallel processing across multiple CPU cores
- Provide checkpointing to resume from last saved state
- Significantly reduce processing time for the complete dataset
- Allow recovery from interruptions without data loss

### Usage
The script is designed to work directly with the MaFi dataset files:
```bash
# Process MaFi behavioral data with all metrics
python calculate_mafi_parallel.py "Behavioural Data/DataComplete_IPA_Dist.xlsx" \
    --csv-path "Norms/MaFI_Combined.csv" \
    --all

# Process subset of data for testing (e.g., first 10 rows)
python calculate_mafi_parallel.py "Behavioural Data/DataComplete_IPA_Dist.xlsx" \
    --csv-path "Norms/MaFI_Combined.csv" \
    --limit 10 \
    --all

# Process full dataset with parallel processing and checkpointing
python calculate_mafi_parallel.py "Behavioural Data/DataComplete_IPA_Dist.xlsx" \
    --csv-path "Norms/MaFI_Combined.csv" \
    --all \
    --processes 8 \
    --checkpoint-interval 5000
```

### Command Line Options

Arguments:
- `excel_path`: Path to the input Excel file with Word-Response pairs
- `--output, -o`: Path to save the output Excel file (defaults to input_with_metrics.xlsx in same folder)
- `--csv-path, -c`: Path to a CSV file with Words to calculate average metrics for
- `--csv-output`: Path to save the enhanced CSV file (defaults to input_with_averages.csv in same folder)
- `--limit, -l`: Limit processing to first N rows of Excel file (for testing)
- `--all`: Calculate additional metrics (WER, CER, BLEU, etc.)
- `--no-gpu`: Disable GPU acceleration even if available
- `--checkpoint-interval`: Save checkpoint after processing this many rows (default: 1000)
- `--resume`: Resume from last checkpoint if available
- `--processes, -p`: Number of processes to use for parallel processing (default: number of CPU cores minus 1)

## ViPS Score on MaFi Dataset

Our calculated ViPS score on the MaFi dataset is presented in the `Norms` folder. You can find the results in the following file:

- [MaFI_Combined_with_ViPS_and_other_Scores.csv](../MaFi/Norms/MaFI_Combined_with_ViPS_and_other_Scores.csv)

Behavioural data related to the MaFi dataset is available in the `Behavioural Data` folder. You can find the complete data in the following file:

- [DataComplete_IPA_Dist_with_ViPS.xlsx](/MaFi/Behavioural%20Data/DataComplete_IPA_Dist_with_ViPS.xlsx)


## Plotting MaFi Correlations with ViPS and other Metrics

To generate correlation plots comparing MaFi and ViPS with other metrics (WER, CER, BertScore, Semantic Similarity), use the following command:

```bash
python MaFi/mafi_pearson_visualization_and_comparison.py --csv /path/to/MaFI_Combined_with_ViPS_and_other_Scores.csv --output /path/to/output_directory
```

For example:
```bash
python MaFi/mafi_pearson_visualization_and_comparison.py --csv /home/rishabh/Desktop/Experiments/ViPS/MaFi/Norms/MaFI_Combined_with_ViPS_and_other_Scores.csv --output ./MaFi/pearson_plots_comparison
```

This will generate various visualization plots including:
- Individual correlation plots for each metric
- Combined correlation plots
- Correlation comparison plots
- Metrics summary in both CSV and Markdown formats

