#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import argparse
import os
from vis_phon import WeightedLipReadingEvaluator

def create_correlation_plots(df, output_path='correlation_plots.png'):
    """
    Create separate scatter plots with correlation lines comparing ViPS and WVS with other metrics.
    
    Args:
        df: DataFrame with columns for metrics
        output_path: Path to save the output plot
    """
    # Set style - use a standard matplotlib style instead of 'seaborn'
    plt.style.use('ggplot')  # Using ggplot which is widely available
    
    # Define metrics to compare against
    comparison_metrics = ['WER', 'CER', 'BertScore', 'SemanticSimilarity']
    
    # Create a 4x1 grid for each metric type (ViPS and WVS)
    fig_vips, axes_vips = plt.subplots(4, 1, figsize=(10, 20))
    fig_wvs, axes_wvs = plt.subplots(4, 1, figsize=(10, 20))
    
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
        
        # WVS plot
        ax = axes_wvs[idx]
        
        # Calculate correlation
        pearson_corr, p_value = stats.pearsonr(df['WVS'], df[metric])
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
        ax.scatter(df['WVS'], df[metric], alpha=0.6, color=colors[idx], s=40)
        
        # Add correlation line
        z = np.polyfit(df['WVS'], df[metric], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df['WVS'].min(), df['WVS'].max(), 100)
        ax.plot(x_range, p(x_range), color=colors[idx], linewidth=2)
        
        # Add correlation text with more readable p-value
        corr_text = f"r = {pearson_corr:.2f}\nR² = {r_squared:.2f}\n{p_value_str}"
        ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                verticalalignment='top')
        
        # Labels
        ax.set_xlabel('Weighted Viseme Score (WVS)')
        ax.set_ylabel(metric)
        ax.set_title(f'WVS vs {metric}')
        ax.grid(True, alpha=0.3)
    
    # Add overall titles
    fig_vips.suptitle('Correlation Between ViPS and Standard Evaluation Metrics', fontsize=16)
    fig_wvs.suptitle('Correlation Between WVS and Standard Evaluation Metrics', fontsize=16)
    
    # Save plots
    vips_output = output_path.replace('.png', '_vips.png')
    wvs_output = output_path.replace('.png', '_wvs.png')
    
    plt.figure(fig_vips.number)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(vips_output, dpi=300, bbox_inches='tight')
    
    plt.figure(fig_wvs.number)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(wvs_output, dpi=300, bbox_inches='tight')
    
    plt.close('all')
    
    print(f"Correlation plots saved to {vips_output} and {wvs_output}")

def create_combined_correlation_plots(df, output_path='combined_correlation_plots.png'):
    """
    Create separate combined plots with all metrics on the same axes for ViPS and WVS.
    
    Args:
        df: DataFrame with columns for metrics
        output_path: Path to save the output plot
    """
    # Set style
    plt.style.use('ggplot')
    
    # Define metrics to compare against
    comparison_metrics = ['WER', 'CER', 'BertScore', 'SemanticSimilarity']
    
    # Create two separate plots - one for ViPS, one for WVS
    fig_vips = plt.figure(figsize=(12, 8))
    fig_wvs = plt.figure(figsize=(12, 8))
    ax_vips = fig_vips.add_subplot(111)
    ax_wvs = fig_wvs.add_subplot(111)
    
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
    
    # Plot metrics for WVS
    for idx, metric in enumerate(comparison_metrics):
        # Calculate correlation and p-value
        pearson_corr, p_value = stats.pearsonr(df['WVS'], df[metric])
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
        scatter = ax_wvs.scatter(df['WVS'], df[metric], 
                   alpha=0.6, 
                   color=colors[idx], 
                   s=40, 
                   marker=markers[idx],
                   label=f"{metric} (r={pearson_corr:.2f}, {p_value_str}, R²={r_squared:.2f})")
        
        # Add correlation line
        z = np.polyfit(df['WVS'], df[metric], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df['WVS'].min(), df['WVS'].max(), 100)
        ax_wvs.plot(x_range, p(x_range), color=colors[idx], linewidth=2)
    
    # Add legend inside the plot in the upper right corner
    ax_wvs.legend(loc='upper right', fontsize=9)
    
    ax_wvs.set_xlabel('Weighted Viseme Score (WVS)', fontsize=12)
    ax_wvs.set_ylabel('Metric Value', fontsize=12)
    ax_wvs.set_title('WVS vs All Metrics', fontsize=14)
    ax_wvs.grid(True, alpha=0.3)
    
    # Save plots
    vips_output = output_path.replace('.png', '_combined_vips.png')
    wvs_output = output_path.replace('.png', '_combined_wvs.png')
    
    # Adjust layout and save ViPS plot
    plt.figure(fig_vips.number)
    plt.tight_layout()
    plt.savefig(vips_output, dpi=300, bbox_inches='tight')
    
    # Adjust layout and save WVS plot
    plt.figure(fig_wvs.number)
    plt.tight_layout()
    plt.savefig(wvs_output, dpi=300, bbox_inches='tight')
    
    plt.close('all')
    
    print(f"Combined correlation plots saved to {vips_output} and {wvs_output}")

def create_metrics_summary_table(df, summary_metrics=None, output_path='metrics_summary.csv', markdown_path='metrics_summary.md'):
    """
    Create a comprehensive metrics summary table with correlation statistics.
    
    Args:
        df: DataFrame with columns for metrics
        summary_metrics: Dictionary of summary metrics from the evaluator (if available)
        output_path: Path to save the CSV summary
        markdown_path: Path to save a readable Markdown version
    """
    # Define metrics to analyze
    comparison_metrics = ['WER', 'CER', 'BertScore', 'SemanticSimilarity']
    
    # Prepare metrics data - either use summary metrics if available or calculate from data
    if summary_metrics:
        # Use the original summary metrics provided by the evaluator
        metrics_dict = {
            'Metric': [],
            'Value': []
        }
        
        # Add ViPS and WVS from summary if available
        if 'weighted' in summary_metrics:
            weighted = summary_metrics['weighted']
            if 'avg_phonetic_score' in weighted:
                metrics_dict['Metric'].append('ViPS')
                metrics_dict['Value'].append(weighted['avg_phonetic_score'])
            if 'avg_phonetically_weighted_viseme_score' in weighted:
                metrics_dict['Metric'].append('WVS')
                metrics_dict['Value'].append(weighted['avg_phonetically_weighted_viseme_score'])
        
        # Add standard metrics if available
        standard_metrics = {
            'word_error_rate': 'WER',
            'character_error_rate': 'CER',
            'bertscore_f1': 'BertScore',
            'semantic_similarity': 'SemanticSimilarity',
            'bleu_score': 'BLEU',
            'meteor_score': 'METEOR',
            'rouge1_score': 'ROUGE-1',
            'rouge2_score': 'ROUGE-2',
            'rougeL_score': 'ROUGE-L'
        }
        
        for key, display_name in standard_metrics.items():
            if key in summary_metrics:
                metrics_dict['Metric'].append(display_name)
                metrics_dict['Value'].append(summary_metrics[key])
    else:
        # Calculate statistics from individual examples
        metrics_dict = {
            'Metric': ['ViPS', 'WVS'] + comparison_metrics,
            'Value': [df['ViPS'].mean(), df['WVS'].mean()] + [df[m].mean() for m in comparison_metrics],
        }
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics_dict)
    
    # Calculate correlations with ViPS (CP - Correlation with Phoneme Score)
    # and with WVS (CV - Correlation with Viseme Score)
    correlation_data = []
    
    for metric in comparison_metrics:
        if metric in df.columns and 'ViPS' in df.columns and 'WVS' in df.columns:
            # Correlation with ViPS
            cp_pearson, cp_pvalue = stats.pearsonr(df['ViPS'], df[metric])
            cp_rsquared = cp_pearson ** 2
            
            # Correlation with WVS
            cv_pearson, cv_pvalue = stats.pearsonr(df['WVS'], df[metric])
            cv_rsquared = cv_pearson ** 2
            
            # Format p-values in a more readable way
            if cp_pvalue < 0.001:
                cp_p_str = "p<0.001"
            elif cp_pvalue < 0.01:
                cp_p_str = "p<0.01"
            elif cp_pvalue < 0.05:
                cp_p_str = "p<0.05"
            else:
                cp_p_str = f"p={cp_pvalue:.3f}"
            
            if cv_pvalue < 0.001:
                cv_p_str = "p<0.001"
            elif cv_pvalue < 0.01:
                cv_p_str = "p<0.01"
            elif cv_pvalue < 0.05:
                cv_p_str = "p<0.05"
            else:
                cv_p_str = f"p={cv_pvalue:.3f}"
            
            correlation_data.append({
                'Metric': metric,
                'CP_Pearson': f"{cp_pearson:.4f}",
                'CP_Rsquared': f"{cp_rsquared:.4f}",
                'CP_Pvalue': cp_p_str,
                'CV_Pearson': f"{cv_pearson:.4f}",
                'CV_Rsquared': f"{cv_rsquared:.4f}",
                'CV_Pvalue': cv_p_str,
            })
    
    correlation_df = pd.DataFrame(correlation_data)
    
    # Save to CSV (combine metrics and correlations)
    full_df = pd.DataFrame({
        'Metric': metrics_df['Metric'],
        'Value': metrics_df['Value']
    })
    
    # Add correlation data for metrics that have it
    for metric in correlation_df['Metric']:
        idx = correlation_df[correlation_df['Metric'] == metric].index[0]
        full_df.loc[full_df['Metric'] == metric, 'CP_Pearson'] = correlation_df.loc[idx, 'CP_Pearson']
        full_df.loc[full_df['Metric'] == metric, 'CP_Rsquared'] = correlation_df.loc[idx, 'CP_Rsquared']
        full_df.loc[full_df['Metric'] == metric, 'CP_Pvalue'] = correlation_df.loc[idx, 'CP_Pvalue']
        full_df.loc[full_df['Metric'] == metric, 'CV_Pearson'] = correlation_df.loc[idx, 'CV_Pearson']
        full_df.loc[full_df['Metric'] == metric, 'CV_Rsquared'] = correlation_df.loc[idx, 'CV_Rsquared']
        full_df.loc[full_df['Metric'] == metric, 'CV_Pvalue'] = correlation_df.loc[idx, 'CV_Pvalue']
    
    full_df.to_csv(output_path, index=False, float_format='%.4f')
    
    # Create a more readable, simplified Markdown version
    with open(markdown_path, 'w') as f:
        f.write("# Metrics Summary\n\n")
        
        # Create a simplified metrics table
        f.write("## Metrics Overview\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        
        for i, metric in enumerate(metrics_df['Metric']):
            value = metrics_df['Value'][i]
            f.write(f"| {metric} | {value:.4f} |\n")
        
        # Create a simplified correlation table
        f.write("\n## Correlation Analysis\n\n")
        f.write("| Metric | Correlation with ViPS | Correlation with WVS |\n")
        f.write("|--------|----------------------|----------------------|\n")
        
        for i, row in correlation_df.iterrows():
            metric = row['Metric']
            cp_entry = f"r={row['CP_Pearson']}, R²={row['CP_Rsquared']}, {row['CP_Pvalue']}"
            cv_entry = f"r={row['CV_Pearson']}, R²={row['CV_Rsquared']}, {row['CV_Pvalue']}"
            f.write(f"| {metric} | {cp_entry} | {cv_entry} |\n")
    
    print(f"Metrics summary saved to {output_path} and {markdown_path}")
    return full_df

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
    results = evaluator.analyze_json_dataset_with_comparisons(
        json_file, 
        output_file=os.path.join(output_dir, 'results.json'),
        include_all_metrics=True
    )
    
    if not results:
        print("Error: Failed to analyze JSON data")
        return
    
    # Extract results and metrics
    all_results = results.get('results', [])
    summary = results.get('summary', {})
    
    # Prepare data for correlation analysis
    data = []
    for result in all_results:
        # Get the weighted metrics
        weighted = result.get('weighted', {})
        standard = result.get('standard', {})
        metrics = result.get('metrics', {})
        
        # Check if we have all the necessary metrics
        if not weighted or not metrics:
            continue
        
        # Extract the data we need
        entry = {
            'ViPS': weighted.get('phonetic_alignment_score', 0.0),
            'WVS': weighted.get('phonetically_weighted_viseme_score', 0.0),
            'WER': metrics.get('wer', 1.0),  # Higher is worse
            'CER': metrics.get('cer', 1.0),  # Higher is worse
            'BertScore': metrics.get('bertscore_f1', 0.0),  # Higher is better
            'SemanticSimilarity': metrics.get('semantic_similarity', 0.0)  # Higher is better
        }
        
        data.append(entry)
    
    if not data:
        print("Error: No valid metric data found in results")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create correlation plots
    output_path = os.path.join(output_dir, 'metric_correlation_plots.png')
    create_correlation_plots(df, output_path)
    
    # Create combined correlation plots (2x1 grid)
    combined_output_path = os.path.join(output_dir, 'combined_correlation_plots.png')
    create_combined_correlation_plots(df, combined_output_path)
    
    # Create metrics summary table using summary metrics if available
    csv_path = os.path.join(output_dir, 'metrics_summary.csv')
    md_path = os.path.join(output_dir, 'metrics_summary.md')
    
    # Extract summary metrics if available
    summary_metrics = None
    if 'additional_metrics' in summary:
        summary_metrics = summary.get('additional_metrics', {})
        # Add weighted metrics to the summary metrics
        if 'weighted' in summary:
            summary_metrics['weighted'] = summary.get('weighted', {})
    
    create_metrics_summary_table(df, summary_metrics, csv_path, md_path)
    
    # Show correlation values
    print("\nCorrelation Analysis:")
    print("----------------------------------")
    print("                  | ViPS    | WVS")
    print("----------------------------------")
    for metric in ['WER', 'CER', 'BertScore', 'SemanticSimilarity']:
        vips_corr, _ = stats.pearsonr(df['ViPS'], df[metric])
        wvs_corr, _ = stats.pearsonr(df['WVS'], df[metric])
        print(f"{metric:18} | {vips_corr:.4f} | {wvs_corr:.4f}")
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