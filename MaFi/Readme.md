
## Citation

We use the MaFi dataset for our experiments with ViPS as cited below:

> A. Krason, Y. Zhang, H. Man, and G. Vigliocco, "Mouth and facial
informativeness norms for 2276 english words," Behavior Research
Methods, vol. 56, no. 5, pp. 4786–4801, August 2024.

## Usage

### Plotting MaFi Correlations

To generate correlation plots comparing MaFi with other metrics (ViPS, WER, CER, BertScore, Semantic Similarity), use the following command:

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

## ViPS Score on MaFi Dataseton

We use the MaFi dataset for our experiments with ViPS as cited below:

> A. Krason, Y. Zhang, H. Man, and G. Vigliocco, “Mouth and facial
informativeness norms for 2276 english words,” Behavior Research
Methods, vol. 56, no. 5, pp. 4786–4801, August 2024.

## ViPS Score on MaFi Dataset

Our calculated ViPS score on the MaFi dataset is presented in the `Norms` folder. You can find the results in the following file:

- [MaFI_Combined_with_ViPS_and_other_Scores.csv](../MaFi/Norms/MaFI_Combined_with_ViPS_and_other_Scores.csv)

## Behavioural Data

Behavioural data related to the MaFi dataset is available in the `Behavioural Data` folder. You can find the complete data in the following file:

- [DataComplete_IPA_Dist_with_ViPS.xlsx](/MaFi/Behavioural%20Data/DataComplete_IPA_Dist_with_ViPS.xlsx)
