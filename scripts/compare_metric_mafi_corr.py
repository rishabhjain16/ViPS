#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import argparse
import os

def create_correlation_plots(df, output_path='correlation_plots.png'):
    """
    Create separate scatter plots with correlation lines comparing ViPS with other metrics.
    
    Args:
        df: DataFrame with columns for metrics
        output_path: Path to save the output plot
    """
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Define metrics to compare against
    comparison_metrics = ['MaFI', 'word_error_rate_avg', 'character_error_rate_avg', 'bertscore_f1_avg', 'semantic_similarity_avg']
    metric_display_names = {
        'MaFI': 'MaFI Score',
        'word_error_rate_avg': 'WER',
        'character_error_rate_avg': 'CER',
        'bertscore_f1_avg': 'BertScore',
        'semantic_similarity_avg': 'Semantic Similarity'
    }
    
    # Create a 5x1 grid for ViPS
    fig_vips, axes_vips = plt.subplots(5, 1, figsize=(10, 25))
    
    # Color palette
    colors = sns.color_palette("viridis", len(comparison_metrics))
    
    # Create plots for each metric
    for idx, metric in enumerate(comparison_metrics):
        ax = axes_vips[idx]
        
        # Calculate correlation
        pearson_corr, p_value = stats.pearsonr(df['wgt_phonetic_score_avg'], df[metric])
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
        ax.scatter(df['wgt_phonetic_score_avg'], df[metric], alpha=0.6, color=colors[idx], s=40)
        
        # Add correlation line
        z = np.polyfit(df['wgt_phonetic_score_avg'], df[metric], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df['wgt_phonetic_score_avg'].min(), df['wgt_phonetic_score_avg'].max(), 100)
        ax.plot(x_range, p(x_range), color=colors[idx], linewidth=2)
        
        # Add correlation text
        corr_text = f"r = {pearson_corr:.2f}\nR² = {r_squared:.2f}\n{p_value_str}"
        ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                verticalalignment='top', fontsize=14)
        
        # Labels
        ax.set_xlabel('Visemic-Phonetic Score (ViPS)', fontsize=16)
        ax.set_ylabel(metric_display_names[metric], fontsize=16)
        ax.set_title(f'ViPS vs {metric_display_names[metric]}', fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Add overall title
    fig_vips.suptitle('Correlation Between ViPS and Standard Evaluation Metrics', fontsize=20)
    
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
    Also includes a small table of mean scores.
    
    Args:
        df: DataFrame with columns for metrics
        output_path: Path to save the output plot
    """
    # Remove seaborn style
    # plt.style.use('seaborn-v0_8')
    
    # Define metrics to compare against
    comparison_metrics = ['MaFI', 'word_error_rate_avg', 'character_error_rate_avg', 'bertscore_f1_avg', 'semantic_similarity_avg']
    metric_display_names = {
        'MaFI': 'MaFI Score',
        'word_error_rate_avg': 'WER',
        'character_error_rate_avg': 'CER',
        'bertscore_f1_avg': 'BertScore',
        'semantic_similarity_avg': 'Semantic Similarity'
    }
    
    # Create figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    
    # Color palette and markers
    colors = ['#c0392b', '#f39c12', '#45B7D1', '#16a085', '#884ea0']
    markers = ['o', 's', '^', 'd', 'v']
    
    # Create table data
    table_data = []
    headers = ['Metric', 'Scores']
    table_data.append(['ViPS', f"{df['wgt_phonetic_score_avg'].mean():.3f}"])
    for metric in comparison_metrics:
        table_data.append([
            metric_display_names[metric],
            f"{df[metric].mean():.3f}"
        ])
    
    # Create and customize table
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='top',
        bbox=[0.02, 0.85, 0.35, 0.14],
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1.8, 2.2)
    
    # Style the table
    header_color = '#bababa'  # Changed back to original gray
    cell_color = 'white'      # Keep white for cells
    
    for cell in table._cells:
        cell_obj = table._cells[cell]
        cell_obj.set_edgecolor('black')
        cell_obj.set_linewidth(0.25)
        
        if cell[0] == 0:  # Header row
            cell_obj.set_text_props(weight='bold', color='black', fontsize=16)
            cell_obj.set_facecolor(header_color)
        else:
            cell_obj.set_text_props(color='black', fontsize=14, weight='bold')
            cell_obj.set_facecolor(cell_color)
            
        cell_obj.PAD = 0.8
    
    table.PAD = 0.4
    
    # Plot metrics
    for idx, metric in enumerate(comparison_metrics):
        # Calculate correlation
        pearson_corr, p_value = stats.pearsonr(df['wgt_phonetic_score_avg'], df[metric])
        r_squared = pearson_corr ** 2
        
        # Format p-value
        if p_value < 0.001:
            p_value_str = "p < 0.001"
        elif p_value < 0.01:
            p_value_str = "p < 0.01"
        elif p_value < 0.05:
            p_value_str = "p < 0.05"
        else:
            p_value_str = f"p = {p_value:.3f}"
        
        # Create scatter plot
        ax.scatter(df['wgt_phonetic_score_avg'], df[metric], 
                  alpha=0.7,
                  color=colors[idx], 
                  s=60,
                  marker=markers[idx],
                  label=f"{metric_display_names[metric]} (r={pearson_corr:.2f}, {p_value_str}, R²={r_squared:.2f})")
        
        # Add correlation line
        z = np.polyfit(df['wgt_phonetic_score_avg'], df[metric], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df['wgt_phonetic_score_avg'].min(), df['wgt_phonetic_score_avg'].max(), 100)
        ax.plot(x_range, p(x_range), color=colors[idx], linewidth=3)
    
    # Add legend with explicit frame properties
    legend = ax.legend(loc='upper right', fontsize=16,
                      frameon=True,
                      fancybox=False,
                      framealpha=0.8,
                      edgecolor='gray',
                      borderpad=0.7,
                      labelspacing=0.5)
    
    # Set frame properties directly
    frame = legend.get_frame()
    frame.set_linewidth(1.5)
    frame.set_facecolor('white')
    frame.set_edgecolor('gray')
    frame.set_alpha(0.8)
    
    # Make legend text bold
    for text in legend.get_texts():
        text.set_weight('bold')
    
    # Labels
    ax.set_xlabel('Visemic-Phonetic Score (ViPS)', fontsize=17, weight='bold')
    ax.set_ylabel('Metric Value', fontsize=17, weight='bold')
    
    # Make tick labels bold and increase their size
    ax.tick_params(axis='both', which='major', labelsize=15)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_weight('bold')
    
    # Enhanced grid
    ax.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.set_axisbelow(True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    
    print(f"Combined correlation plot with score table saved to {output_path}")

def create_mafi_combined_correlation_plots(df, output_path='mafi_combined_correlation_plots.png'):
    """
    Create a combined plot with all metrics on the same axes for MaFI.
    Also includes a small table of mean scores.
    
    Args:
        df: DataFrame with columns for metrics
        output_path: Path to save the output plot
    """
    # Remove seaborn style
    # plt.style.use('seaborn-v0_8')
    
    # Define metrics to compare against
    comparison_metrics = ['word_error_rate_avg', 'character_error_rate_avg', 'bertscore_f1_avg', 'semantic_similarity_avg', 'wgt_phonetic_score_avg']
    metric_display_names = {
        'word_error_rate_avg': 'WER',
        'character_error_rate_avg': 'CER',
        'bertscore_f1_avg': 'BertScore',
        'semantic_similarity_avg': 'Semantic Similarity',
        'wgt_phonetic_score_avg': 'ViPS'
    }
    
    # Create figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    
    # Color palette and markers
    colors = ['#c0392b', '#f39c12', '#45B7D1', '#16a085', '#884ea0']
    markers = ['o', 's', '^', 'd', 'v']
    
    # Create table data
    table_data = []
    headers = ['Metric', 'Scores']
    table_data.append(['MaFI', f"{df['MaFI'].mean():.3f}"])
    for metric in comparison_metrics:
        table_data.append([
            metric_display_names[metric],
            f"{df[metric].mean():.3f}"
        ])
    
    # Create and customize table
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='top',
        bbox=[0.02, 0.85, 0.35, 0.14],
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1.8, 2.2)
    
    # Style the table
    header_color = '#bababa'  # Changed back to original gray
    cell_color = 'white'      # Keep white for cells
    
    for cell in table._cells:
        cell_obj = table._cells[cell]
        cell_obj.set_edgecolor('black')
        cell_obj.set_linewidth(0.25)
        
        if cell[0] == 0:  # Header row
            cell_obj.set_text_props(weight='bold', color='black', fontsize=16)
            cell_obj.set_facecolor(header_color)
        else:
            cell_obj.set_text_props(color='black', fontsize=14, weight='bold')
            cell_obj.set_facecolor(cell_color)
            
        cell_obj.PAD = 0.8
    
    table.PAD = 0.4
    
    # Plot metrics
    for idx, metric in enumerate(comparison_metrics):
        # Calculate correlation
        pearson_corr, p_value = stats.pearsonr(df['MaFI'], df[metric])
        r_squared = pearson_corr ** 2
        
        # Format p-value
        if p_value < 0.001:
            p_value_str = "p < 0.001"
        elif p_value < 0.01:
            p_value_str = "p < 0.01"
        elif p_value < 0.05:
            p_value_str = "p < 0.05"
        else:
            p_value_str = f"p = {p_value:.3f}"
        
        # Create scatter plot
        ax.scatter(df['MaFI'], df[metric], 
                  alpha=0.7,
                  color=colors[idx], 
                  s=60,
                  marker=markers[idx],
                  label=f"{metric_display_names[metric]} (r={pearson_corr:.2f}, {p_value_str}, R²={r_squared:.2f})")
        
        # Add correlation line
        z = np.polyfit(df['MaFI'], df[metric], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df['MaFI'].min(), df['MaFI'].max(), 100)
        ax.plot(x_range, p(x_range), color=colors[idx], linewidth=3)
    
    # Add legend with explicit frame properties
    legend = ax.legend(loc='upper right', fontsize=16,
                      frameon=True,
                      fancybox=False,
                      framealpha=0.8,
                      edgecolor='gray',
                      borderpad=0.7,
                      labelspacing=0.5)
    
    # Set frame properties directly
    frame = legend.get_frame()
    frame.set_linewidth(1.5)
    frame.set_facecolor('white')
    frame.set_edgecolor('gray')
    frame.set_alpha(0.8)
    
    # Make legend text bold
    for text in legend.get_texts():
        text.set_weight('bold')
    
    # Labels
    ax.set_xlabel('MaFI Score', fontsize=17, weight='bold')
    ax.set_ylabel('Metric Value', fontsize=17, weight='bold')
    
    # Make tick labels bold and increase their size
    ax.tick_params(axis='both', which='major', labelsize=15)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_weight('bold')
    
    # Enhanced grid
    ax.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.set_axisbelow(True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    
    print(f"MaFI combined correlation plot with score table saved to {output_path}")

def create_metrics_summary_table(df, output_path='metrics_summary.csv', markdown_path='metrics_summary.md'):
    """
    Create a comprehensive metrics summary table with correlation statistics.
    
    Args:
        df: DataFrame with columns for metrics
        output_path: Path to save the CSV summary
        markdown_path: Path to save a readable Markdown version
    """
    # Define metrics to analyze
    comparison_metrics = ['MaFI', 'word_error_rate_avg', 'character_error_rate_avg', 'bertscore_f1_avg', 'semantic_similarity_avg']
    metric_display_names = {
        'MaFI': 'MaFI Score',
        'word_error_rate_avg': 'WER',
        'character_error_rate_avg': 'CER',
        'bertscore_f1_avg': 'BertScore',
        'semantic_similarity_avg': 'Semantic Similarity'
    }
    
    # Calculate statistics
    metrics_dict = {
        'Metric': ['ViPS'] + [metric_display_names[m] for m in comparison_metrics],
        'Mean': [df['wgt_phonetic_score_avg'].mean()] + [df[m].mean() for m in comparison_metrics],
        'Std': [df['wgt_phonetic_score_avg'].std()] + [df[m].std() for m in comparison_metrics],
    }
    
    # Calculate correlations with ViPS
    correlation_data = []
    for metric in comparison_metrics:
        pearson_corr, p_value = stats.pearsonr(df['wgt_phonetic_score_avg'], df[metric])
        r_squared = pearson_corr ** 2
        
        if p_value < 0.001:
            p_str = "p<0.001"
        elif p_value < 0.01:
            p_str = "p<0.01"
        elif p_value < 0.05:
            p_str = "p<0.05"
        else:
            p_str = f"p={p_value:.3f}"
        
        correlation_data.append({
            'Metric': metric_display_names[metric],
            'Correlation': f"{pearson_corr:.4f}",
            'R_squared': f"{r_squared:.4f}",
            'P_value': p_str
        })
    
    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_dict)
    correlation_df = pd.DataFrame(correlation_data)
    
    # Save to CSV
    metrics_df.to_csv(output_path, index=False, float_format='%.4f')
    
    # Create Markdown summary
    with open(markdown_path, 'w') as f:
        f.write("# Metrics Summary\n\n")
        
        f.write("## Basic Statistics\n\n")
        f.write("| Metric | Mean | Std |\n")
        f.write("|--------|------|-----|\n")
        for i, metric in enumerate(metrics_df['Metric']):
            mean = metrics_df['Mean'][i]
            std = metrics_df['Std'][i]
            f.write(f"| {metric} | {mean:.4f} | {std:.4f} |\n")
        
        f.write("\n## Correlation with ViPS\n\n")
        f.write("| Metric | Correlation | R² | P-value |\n")
        f.write("|--------|-------------|----|---------|\n")
        for _, row in correlation_df.iterrows():
            f.write(f"| {row['Metric']} | {row['Correlation']} | {row['R_squared']} | {row['P_value']} |\n")
    
    print(f"Metrics summary saved to {output_path} and {markdown_path}")
    return metrics_df, correlation_df

def create_correlation_comparison_plot(df, output_path='correlation_comparison_plot.png'):
    """
    Create a grouped bar plot comparing correlations (r) and R-squared values of ViPS and MaFI with other metrics.
    
    Args:
        df: DataFrame with columns for metrics
        output_path: Path to save the output plot
    """
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Define metrics to compare against
    comparison_metrics = ['word_error_rate_avg', 'character_error_rate_avg', 'bertscore_f1_avg', 'semantic_similarity_avg']
    metric_display_names = {
        'word_error_rate_avg': 'WER',
        'character_error_rate_avg': 'CER',
        'bertscore_f1_avg': 'BertScore',
        'semantic_similarity_avg': 'Semantic Similarity'
    }
    
    # Calculate correlations and R-squared
    correlations = []
    for metric in comparison_metrics:
        # ViPS correlation
        vips_corr, _ = stats.pearsonr(df['wgt_phonetic_score_avg'], df[metric])
        vips_r2 = vips_corr ** 2
        # MaFI correlation
        mafi_corr, _ = stats.pearsonr(df['MaFI'], df[metric])
        mafi_r2 = mafi_corr ** 2
        
        correlations.append({
            'Metric': metric_display_names[metric],
            'ViPS_r': vips_corr,
            'MaFI_r': mafi_corr,
            'ViPS_R2': vips_r2,
            'MaFI_R2': mafi_r2
        })
    
    # Create DataFrame
    corr_df = pd.DataFrame(correlations)
    
    # Create figure with a white background
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 1])
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    
    # Set width of bars
    barWidth = 0.35
    
    # Set positions of the bars on X axis
    x = np.arange(len(corr_df))
    
    # Plot correlation coefficients (r)
    bars1 = ax1.bar(x - barWidth/2, corr_df['ViPS_r'], barWidth, 
                    label='ViPS', color='#1f77b4', edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + barWidth/2, corr_df['MaFI_r'], barWidth, 
                    label='MaFI', color='#ff7f0e', edgecolor='black', linewidth=1)
    
    # Plot R-squared values
    bars3 = ax2.bar(x - barWidth/2, corr_df['ViPS_R2'], barWidth, 
                    label='ViPS', color='#1f77b4', edgecolor='black', linewidth=1)
    bars4 = ax2.bar(x + barWidth/2, corr_df['MaFI_R2'], barWidth, 
                    label='MaFI', color='#ff7f0e', edgecolor='black', linewidth=1)
    
    # Add value labels on top of bars
    def add_value_labels(ax, bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            # For negative values, place label above the bar
            if value < 0:
                y_pos = height - 0.1  # Slightly above the bar
                va = 'top'
            else:
                y_pos = height + 0.02  # Slightly above the bar
                va = 'bottom'
            
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{value:.2f}', ha='center', va=va,
                   fontsize=22, weight='bold')
    
    # Add labels for both plots
    add_value_labels(ax1, bars1, corr_df['ViPS_r'])
    add_value_labels(ax1, bars2, corr_df['MaFI_r'])
    add_value_labels(ax2, bars3, corr_df['ViPS_R2'])
    add_value_labels(ax2, bars4, corr_df['MaFI_R2'])
    
    # Customize the plots
    for ax, title in zip([ax1, ax2], ['Correlation Coefficient (r)', 'R-squared (R²)']):
        ax.set_ylabel(title, fontsize=25, weight='bold', labelpad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(corr_df['Metric'], fontsize=23, weight='bold')
        ax.tick_params(axis='both', which='major', labelsize=23)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_weight('bold')
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.set_axisbelow(True)
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add legend only to the first plot
        if ax == ax1:
            legend = ax.legend(fontsize=23, loc='lower right',
                             frameon=True,
                             fancybox=False,
                             framealpha=0.8,
                             edgecolor='gray',
                             borderpad=0.7,
                             labelspacing=0.5)
            legend.get_frame().set_linewidth(1.5)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('gray')
            legend.get_frame().set_alpha(0.8)
            # Make legend text bold
            for text in legend.get_texts():
                text.set_weight('bold')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close('all')
    
    print(f"Correlation comparison plot saved to {output_path}")

def process_csv_and_visualize(csv_file, output_dir='metric_correlation'):
    """
    Process a CSV file with metrics and create correlation visualizations.
    
    Args:
        csv_file: Path to CSV file with metrics
        output_dir: Directory to save output files
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Create correlation plots
    output_path = os.path.join(output_dir, 'metric_correlation_plots.png')
    create_correlation_plots(df, output_path)
    
    # Create combined correlation plot for ViPS
    combined_output_path = os.path.join(output_dir, 'combined_correlation_plots.png')
    create_combined_correlation_plots(df, combined_output_path)
    
    # Create combined correlation plot for MaFI
    mafi_combined_output_path = os.path.join(output_dir, 'mafi_combined_correlation_plots.png')
    create_mafi_combined_correlation_plots(df, mafi_combined_output_path)
    
    # Create correlation comparison plot
    comparison_output_path = os.path.join(output_dir, 'correlation_comparison_plot.png')
    create_correlation_comparison_plot(df, comparison_output_path)
    
    # Create metrics summary table
    csv_path = os.path.join(output_dir, 'metrics_summary.csv')
    md_path = os.path.join(output_dir, 'metrics_summary.md')
    metrics_df, correlation_df = create_metrics_summary_table(df, csv_path, md_path)
    
    # Print correlation values
    print("\nCorrelation Analysis:")
    print("----------------------------------")
    print("Metric               | Correlation")
    print("----------------------------------")
    for _, row in correlation_df.iterrows():
        metric = row['Metric']
        corr = float(row['Correlation'])
        print(f"{metric:20} | {corr:10.4f}")
    print("----------------------------------")

def main():
    parser = argparse.ArgumentParser(description='Analyze metric correlations from CSV data')
    parser.add_argument('--csv', type=str, required=True, help='CSV file with metrics')
    parser.add_argument('--output', type=str, default='metric_correlation', help='Output directory')
    args = parser.parse_args()
    
    process_csv_and_visualize(args.csv, args.output)

if __name__ == "__main__":
    main() 