import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import argparse
import json
from vips import WeightedLipReadingEvaluator

def create_correlation_plots(df, output_path='correlation_plots.png'):
    """
    Create scatter plots with correlation lines comparing ViPS with other metrics.
    
    Args:
        df: DataFrame with columns for metrics
        output_path: Path to save the plot
    """
    # Set style - use a standard matplotlib style instead of 'seaborn'
    plt.style.use('ggplot')  # Using ggplot which is widely available
    
    # Define metrics to compare against
    comparison_metrics = ['WER', 'CER', 'BertScore', 'SemanticSimilarity']
    
    # Create a 4x1 grid for ViPS metrics
    fig_vips, axes_vips = plt.subplots(4, 1, figsize=(10, 20))
    
    # Color palette
    colors = sns.color_palette("viridis", len(comparison_metrics))
    
    # Create plots for each metric
    for idx, metric in enumerate(comparison_metrics):
        # ViPS plot
        ax = axes_vips[idx]
        
        # Calculate correlation
        pearson_corr, p_value = stats.pearsonr(df['ViPS'], df[metric])
        r_squared = pearson_corr ** 2
        
        # Format p-value for better readability
        if p_value < 0.001:
            p_value_str = "p < 0.001"
        elif p_value < 0.01:
            p_value_str = "p < 0.01"
        elif p_value < 0.05:
            p_value_str = "p < 0.05"
        else:
            p_value_str = f"p = {p_value:.3f}"
        
        # Create scatter plot
        ax.scatter(df['ViPS'], df[metric], alpha=0.6, color=colors[idx], s=40)
        
        # Add correlation line
        z = np.polyfit(df['ViPS'], df[metric], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df['ViPS'].min(), df['ViPS'].max(), 100)
        ax.plot(x_range, p(x_range), color=colors[idx], linewidth=2)
        
        # Add correlation text with more readable p-value
        corr_text = f"r = {pearson_corr:.2f}\nR² = {r_squared:.2f}\n{p_value_str}"
        ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                verticalalignment='top')
        
        # Labels
        ax.set_xlabel('Visual Phoneme Score (ViPS)')
        ax.set_ylabel(metric)
        ax.set_title(f'ViPS vs {metric}')
        ax.grid(True, alpha=0.3)
        

    
    # Add overall title
    fig_vips.suptitle('Correlation Between ViPS and Standard Evaluation Metrics', fontsize=16)
    
    # Save plot
    plt.figure(fig_vips.number)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close('all')
    
    print(f"Correlation plots saved to {output_path}")

def create_combined_correlation_plots(df, output_path='combined_correlation_plots.png'):
    """
    Create a combined plot with all metrics on the same axes for ViPS.
    
    Args:
        df: DataFrame with columns for metrics
        output_path: Path to save the output plot
    """
    # Set style
    plt.style.use('ggplot')
    
    # Define metrics to compare against
    comparison_metrics = ['WER', 'CER', 'BertScore', 'SemanticSimilarity']
    
    # Create plot for ViPS
    fig_vips = plt.figure(figsize=(12, 8))
    ax_vips = fig_vips.add_subplot(111)
    
    # Color palette - different color for each metric
    colors = sns.color_palette("viridis", len(comparison_metrics))
    
    # Define markers for each metric to help distinguish them
    markers = ['o', 's', '^', 'd']
    
    # Plot metrics for ViPS
    for idx, metric in enumerate(comparison_metrics):
        # Calculate correlation and p-value
        pearson_corr, p_value = stats.pearsonr(df['ViPS'], df[metric])
        r_squared = pearson_corr ** 2
        
        # Format p-value for better readability
        if p_value < 0.001:
            p_value_str = "p < 0.001"
        elif p_value < 0.01:
            p_value_str = "p < 0.01"
        elif p_value < 0.05:
            p_value_str = "p < 0.05"
        else:
            p_value_str = f"p = {p_value:.3f}"
        
        # Create scatter plot with unique color and marker
        scatter = ax_vips.scatter(df['ViPS'], df[metric], 
                   alpha=0.6, 
                   color=colors[idx], 
                   s=40, 
                   marker=markers[idx],
                   label=f"{metric} (r={pearson_corr:.2f}, {p_value_str}, R²={r_squared:.2f})")
        
        # Add correlation line
        z = np.polyfit(df['ViPS'], df[metric], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df['ViPS'].min(), df['ViPS'].max(), 100)
        ax_vips.plot(x_range, p(x_range), color=colors[idx], linewidth=2)
    
    # Add legend inside the plot in the upper right corner
    ax_vips.legend(loc='upper right', fontsize=9)
    
    ax_vips.set_xlabel('Visual Phoneme Score (ViPS)', fontsize=12)
    ax_vips.set_ylabel('Metric Value', fontsize=12)
    ax_vips.set_title('ViPS vs All Metrics', fontsize=14)
    ax_vips.grid(True, alpha=0.3)
    

    
    # Save plot
    plt.figure(fig_vips.number)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    
    print(f"Combined correlation plot saved to {output_path}")

def create_metrics_summary_table(df, summary_metrics=None, output_path='metrics_summary.csv', markdown_path='metrics_summary.md'):
    """
    Create a summary table of Pearson correlations between ViPS and other metrics.
    
    Args:
        df: DataFrame with columns for metrics
        summary_metrics: Not used, kept for backward compatibility
        output_path: Path to save the CSV summary
        markdown_path: Path to save a readable Markdown version
    """
    # Define metrics to analyze
    comparison_metrics = ['WER', 'CER', 'BertScore', 'SemanticSimilarity']
    
    # Calculate correlations with ViPS
    correlation_data = []
    
    for metric in comparison_metrics:
        if metric in df.columns and 'ViPS' in df.columns:
            # Correlation with ViPS
            cp_pearson, cp_pvalue = stats.pearsonr(df['ViPS'], df[metric])
            cp_rsquared = cp_pearson ** 2
            
            # Format p-values in a more readable way
            if cp_pvalue < 0.001:
                cp_p_str = "p<0.001"
            elif cp_pvalue < 0.01:
                cp_p_str = "p<0.01"
            elif cp_pvalue < 0.05:
                cp_p_str = "p<0.05"
            else:
                cp_p_str = f"p={cp_pvalue:.3f}"
            
            correlation_data.append({
                'Metric': metric,
                'Pearson': f"{cp_pearson:.4f}",
                'Rsquared': f"{cp_rsquared:.4f}",
                'Pvalue': cp_p_str
            })
    
    correlation_df = pd.DataFrame(correlation_data)
    
    # Save correlation data as CSV
    correlation_df.to_csv(output_path, index=False, float_format='%.4f')
    
    # Create a readable Markdown version
    with open(markdown_path, 'w') as f:
        f.write("# Correlation Analysis with ViPS\n\n")
        f.write("| Metric | Correlation Statistics |\n")
        f.write("|--------|----------------------|\n")
        
        for _, row in correlation_df.iterrows():
            metric = row['Metric']
            corr_entry = f"r={row['Pearson']}, R²={row['Rsquared']}, {row['Pvalue']}"
            f.write(f"| {metric} | {corr_entry} |\n")
    
    print(f"Metrics summary saved to {output_path} and {markdown_path}")
    return correlation_df

def process_json_and_visualize(json_file, output_dir='metric_correlation'):
    """
    Process a JSON file with reference-hypothesis pairs, compute metrics,
    and create correlation visualizations.
    
    Args:
        json_file: Path to JSON file with reference-hypothesis pairs
        output_dir: Directory to save output files
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the evaluator from vis_phon.py
    evaluator = WeightedLipReadingEvaluator()
    
    # Extract pairs from the JSON file using existing function
    print(f"Loading and analyzing data from {json_file}...")
    
    # Analyze the dataset and get results with all metrics
    print(f"Input file: {json_file}")
    results = evaluator.analyze_json_dataset_with_comparisons(
        json_file, 
        output_file=os.path.join(output_dir, 'results.json'),
        include_all_metrics=True
    )

    print("\nRaw results structure:")
    print(json.dumps(results, indent=2)[:1000] if results else "No results")  # First 1000 chars to avoid overflow

    if not results:
        print("Error: Failed to analyze JSON data")
        return

    examples = results.get('results', [])  # Changed from 'examples' to 'results'

    data = []
    
    # Process each example
    for example in examples:
               
            # Get metrics from evaluator results
            # Get metrics directly from example
            metrics = example.get("metrics", {})
            # Only collect examples that have all required metrics
            if all(key in metrics for key in ['wer', 'cer', 'bertscore_f1', 'semantic_similarity']):
                entry = {
                    'ViPS': example.get('vips_score', 0.0),
                    'WER': metrics['wer'],
                    'CER': metrics['cer'],
                    'BertScore': metrics['bertscore_f1'],
                    'SemanticSimilarity': metrics['semantic_similarity']
                }
                data.append(entry)
    
    if not data:
        print("Error: No valid metric data found in results")
        return

    print(f"\nFound {len(data)} valid entries with all required metrics")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Debug: Print the data we got
    print("\nData Statistics:")
    print("-" * 50)
    print(df.describe())
    print("\nNumber of unique values per column:")
    print(df.nunique())
    print("\nFirst few rows:")
    print(df.head())
    print("-" * 50)
    
    # Check if we have variation in the data
    constant_columns = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_columns.append(col)
    
    if constant_columns:
        print("\nWarning: The following columns have constant values:")
        for col in constant_columns:
            print(f"- {col}: {df[col].iloc[0]}")
    
    # Create correlation plots
    output_path = os.path.join(output_dir, 'metric_correlation_plots.png')
    create_correlation_plots(df, output_path)
    
    # Create combined correlation plots (2x1 grid)
    combined_output_path = os.path.join(output_dir, 'combined_correlation_plots.png')
    create_combined_correlation_plots(df, combined_output_path)
    
    # Create metrics summary table using summary metrics if available
    csv_path = os.path.join(output_dir, 'metrics_summary.csv')
    md_path = os.path.join(output_dir, 'metrics_summary.md')
    
    # Create metrics summary table
    create_metrics_summary_table(df, None, csv_path, md_path)
    
    # Show correlation values
    print("\nCorrelation Analysis:")
    print("----------------------------------")
    print("                  | ViPS    ")
    print("----------------------------------")
    for metric in ['WER', 'CER', 'BertScore', 'SemanticSimilarity']:
        vips_corr, _ = stats.pearsonr(df['ViPS'], df[metric])
        print(f"{metric:18} | {vips_corr:.4f}")
    print("----------------------------------")
    
    # Save correlation data as CSV
    csv_path = os.path.join(output_dir, 'metric_correlations.csv')
    df.to_csv(csv_path, index=False)
    print(f"Metric data saved to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze metric correlations from JSON data')
    parser.add_argument('--json', type=str, required=True, help='JSON file with reference-hypothesis pairs')
    parser.add_argument('--output', type=str, default='metric_correlation', help='Output directory')
    args = parser.parse_args()
    
    process_json_and_visualize(args.json, args.output)

if __name__ == "__main__":
    main() 