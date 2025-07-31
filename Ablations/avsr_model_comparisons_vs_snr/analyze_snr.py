import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def interpolate_snr_level(snr_values, metric_values, target_value):
    """Interpolate the SNR level at which a metric reaches a target value"""
    if len(snr_values) < 2:
        return None
    
    # Create interpolation function
    f = interp1d(metric_values, snr_values, bounds_error=False, fill_value=np.nan)
    return f(target_value)

def calculate_snr_gains(df, reference_snr=0.0):
    """Calculate SNR gains for all metrics using performance at 0dB as reference"""
    models = df['Model'].unique()
    metrics = ['WER', 'CER', 'BERTScore', 'SemanticSimilarity', 'ViPS']
    gains = []
    
    for model in models:
        model_data = df[df['Model'] == model]
        
        # Get reference values at 0dB for AO
        ref_values = {}
        for metric in metrics:
            ref_value = model_data[(model_data['Modality'] == 'AO') & 
                                 (model_data['SNR'] == reference_snr)][metric].iloc[0]
            ref_values[metric] = ref_value
        
        # Calculate gains for AV modality
        for metric in metrics:
            ao_data = model_data[model_data['Modality'] == 'AO']
            av_data = model_data[model_data['Modality'] == 'AV']
            
            # Sort by SNR
            ao_snrs = ao_data['SNR'].values
            ao_values = ao_data[metric].values
            av_snrs = av_data['SNR'].values
            av_values = av_data[metric].values
            
            # Find SNR level where AV achieves the same performance as AO at 0dB
            av_snr = interpolate_snr_level(av_snrs, av_values, ref_values[metric])
            
            if not np.isnan(av_snr):
                gain = reference_snr - av_snr
                gains.append({
                    'Model': model,
                    'Metric': metric,
                    'SNR_Gain': gain,
                    'Reference_Value': ref_values[metric]
                })
    
    return pd.DataFrame(gains)

def plot_combined_metrics(df, output_dir):
    """Create a combined plot with all metrics in subplots"""
    metrics = ['WER', 'CER', 'BERTScore', 'SemanticSimilarity', 'ViPS']
    
    # Create a mapping for display names
    model_display_names = {
        'avec': 'AVEC',
        'AV_relscore': 'AV-RelScore',
        'auto-avsr': 'Auto-AVSR'
    }
    
    models = df['Model'].unique()
    
    # Set style
    sns.set_style("whitegrid")
    
    # Define SNR ticks
    snr_ticks = [-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5]
    
    # Define professional color scheme
    colors = {
        'avec': '#1f77b4',      # Blue
        'AV_relscore': '#2ca02c',  # Green
        'auto-avsr': '#ff7f0e'   # Orange
    }
    
    # Create figure with custom gridspec - reduced height
    fig = plt.figure(figsize=(12, 8))
    
    # Create gridspec with 2 rows, 3 columns
    gs = fig.add_gridspec(2, 3)
    
    # Create axes for plots (2x3 grid)
    axes = []
    for i in range(2):
        for j in range(3):
            if i * 3 + j < len(metrics):  # Only create axes for existing metrics
                ax = fig.add_subplot(gs[i, j])
                axes.append(ax)
    
    for ax, metric in zip(axes, metrics):
        for model in models:
            model_data = df[df['Model'] == model]
            color = colors[model]
            display_name = model_display_names[model]
            
            # Get reference value at 0dB for AO
            ref_value = model_data[(model_data['Modality'] == 'AO') & 
                                 (model_data['SNR'] == 0.0)][metric].iloc[0]
            
            # Plot AO and AV
            ao_data = model_data[model_data['Modality'] == 'AO']
            av_data = model_data[model_data['Modality'] == 'AV']
            
            # Sort data by SNR
            ao_data = ao_data.sort_values('SNR')
            av_data = av_data.sort_values('SNR')
            
            # Plot lines
            ax.plot(ao_data['SNR'], ao_data[metric], '--', color=color, alpha=0.5,
                   label=f'{display_name} (AO)')
            ax.plot(av_data['SNR'], av_data[metric], '-', color=color,
                   label=f'{display_name} (AV)')
            
            # Find SNR gain
            av_snr = interpolate_snr_level(av_data['SNR'].values, 
                                         av_data[metric].values, 
                                         ref_value)
            
            if not np.isnan(av_snr):
                # Calculate y position for label
                y_range = ax.get_ylim()
                label_offset = (y_range[1] - y_range[0]) * 0.02
                
                # Draw horizontal arrow
                ax.annotate('', 
                    xy=(0.0, ref_value),
                    xytext=(av_snr, ref_value),
                    arrowprops=dict(
                        arrowstyle='<->',
                        color='#262626',  # Light black color
                        alpha=0.6,
                        linewidth=1.0
                    ),
                    annotation_clip=True
                )
                
                # Add label above the line without background
                ax.text((av_snr + 0.0) / 2, ref_value + label_offset,
                       f'{0-av_snr:.1f}dB',
                       horizontalalignment='center',
                       verticalalignment='bottom',
                       color='black',
                       fontsize=13)
        
        # Customize subplot - reduced padding and margins
        ax.set_xlabel('SNR (dB)', fontsize=13)
        ax.set_ylabel(metric, fontsize=13)
        ax.set_xticks(snr_ticks)
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.3)
        
        # Reduce margins
        ax.margins(x=0.02)
        
        # Add legend only for BERTScore plot (third plot)
        if metric == 'BERTScore':
            # Create legend handles
            handles = []
            labels = []
            for model in models:
                color = colors[model]
                display_name = model_display_names[model]
                handles.append(plt.Line2D([0], [0], color=color, linestyle='--', alpha=0.5))
                labels.append(f'{display_name} (AO)')
                handles.append(plt.Line2D([0], [0], color=color, linestyle='-'))
                labels.append(f'{display_name} (AV)')
            
            # Add legend below BERTScore plot
            ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.6),
                     ncol=3, fontsize=10, handlelength=1.5, columnspacing=1.0)
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Save plot with tight layout
    plt.savefig(Path(output_dir) / 'combined_metrics.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_snr_curves(df, output_dir):
    """Plot simplified SNR curves with clear gain visualization"""
    metrics = ['WER', 'CER', 'BERTScore', 'SemanticSimilarity', 'ViPS']
    models = df['Model'].unique()
    
    # Create a mapping for display names
    model_display_names = {
        'avec': 'AVEC',
        'AV_relscore': 'AV-RelScore',
        'auto-avsr': 'Auto-AVSR'
    }
    
    # Set style
    sns.set_style("whitegrid")
    
    # Define SNR ticks
    snr_ticks = [-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5]
    
    # Define professional color scheme
    colors = {
        'avec': '#1f77b4',      # Blue
        'AV_relscore': '#2ca02c',  # Green
        'auto-avsr': '#ff7f0e'   # Orange
    }
    
    for metric in metrics:
        # Create figure with custom gridspec
        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
        
        # Create main plot and legend axes
        ax = fig.add_subplot(gs[0])
        legend_ax = fig.add_subplot(gs[1])
        legend_ax.axis('off')
        
        for model in models:
            model_data = df[df['Model'] == model]
            color = colors[model]
            display_name = model_display_names[model]
            
            # Get reference value at 0dB for AO
            ref_value = model_data[(model_data['Modality'] == 'AO') & 
                                 (model_data['SNR'] == 0.0)][metric].iloc[0]
            
            # Plot AO and AV
            ao_data = model_data[model_data['Modality'] == 'AO']
            av_data = model_data[model_data['Modality'] == 'AV']
            
            # Sort data by SNR
            ao_data = ao_data.sort_values('SNR')
            av_data = av_data.sort_values('SNR')
            
            # Plot lines
            ax.plot(ao_data['SNR'], ao_data[metric], '--', color=color, alpha=0.5,
                   label=f'{display_name} (AO)')
            ax.plot(av_data['SNR'], av_data[metric], '-', color=color,
                   label=f'{display_name} (AV)')
            
            # Find SNR gain
            av_snr = interpolate_snr_level(av_data['SNR'].values, 
                                         av_data[metric].values, 
                                         ref_value)
            
            if not np.isnan(av_snr):
                # Calculate y position for label
                y_range = ax.get_ylim()
                label_offset = (y_range[1] - y_range[0]) * 0.02
                
                # Draw horizontal arrow
                ax.annotate('', 
                    xy=(0.0, ref_value),
                    xytext=(av_snr, ref_value),
                    arrowprops=dict(
                        arrowstyle='<->',
                        color='#262626',  # Light black color
                        alpha=0.6,
                        linewidth=1.0
                    ),
                    annotation_clip=True
                )
                
                # Add label above the line without background
                ax.text((av_snr + 0.0) / 2, ref_value + label_offset,
                       f'{0-av_snr:.1f}dB',
                       horizontalalignment='center',
                       verticalalignment='bottom',
                       color='black',
                       fontsize=12)
        
        # Customize plot
        ax.set_title(f'{metric} vs SNR', fontsize=14)
        ax.set_xlabel('SNR (dB)', fontsize=13)
        ax.set_ylabel(metric, fontsize=13)
        ax.set_xticks(snr_ticks)
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.3)
        
        # Create legend handles
        handles = []
        labels = []
        for model in models:
            color = colors[model]
            display_name = model_display_names[model]
            handles.append(plt.Line2D([0], [0], color=color, linestyle='--', alpha=0.5))
            labels.append(f'{display_name} (AO)')
            handles.append(plt.Line2D([0], [0], color=color, linestyle='-'))
            labels.append(f'{display_name} (AV)')
        
        # Add legend at bottom
        legend_ax.legend(handles, labels, loc='center', ncol=len(models),
                        bbox_to_anchor=(0.5, 0.5), fontsize=10)
        
        # Save plot
        plt.savefig(Path(output_dir) / f'snr_curves_{metric}.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()

def plot_snr_gains(gains_df, output_dir):
    """Plot SNR gains comparison"""
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create a mapping for display names
    model_display_names = {
        'avec': 'AVEC',
        'AV_relscore': 'AV-RelScore',
        'auto-avsr': 'Auto-AVSR'
    }
    
    # Update model names in the dataframe
    gains_df['Model'] = gains_df['Model'].map(model_display_names)
    
    # Define professional color scheme for metrics
    metric_colors = {
        'ViPS': 'blue',
        'WER': '#ff7f0e',  # Orange
        'CER': '#2ca02c',  # Green
        'SemanticSimilarity': '#d62728',  # Red
        'BERTScore': '#9467bd'  # Purple
    }
    
    # Create grouped bar plot with custom colors
    ax = sns.barplot(data=gains_df, x='Model', y='SNR_Gain', hue='Metric',
                    palette=metric_colors)
    
    # Add value labels on the bars with black color and smaller font
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, color='black', fontsize=9)
    
    # Customize plot
    plt.title('SNR Gains (Reference: 0dB)', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('SNR Gain (dB)', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Adjust legend
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'snr_gains_comparison.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_metric_values_with_labels(df, output_dir):
    """Plot metric values at each SNR level with value labels on data points"""
    metrics = ['WER', 'CER', 'BERTScore', 'SemanticSimilarity', 'ViPS']
    models = df['Model'].unique()
    
    # Create a mapping for display names
    model_display_names = {
        'avec': 'AVEC',
        'AV_relscore': 'AV-RelScore',
        'auto-avsr': 'Auto-AVSR'
    }
    
    # Set style
    sns.set_style("whitegrid")
    
    # Define SNR ticks
    snr_ticks = [-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5]
    
    # Define professional color scheme
    colors = {
        'avec': '#1f77b4',      # Blue
        'AV_relscore': '#2ca02c',  # Green
        'auto-avsr': '#ff7f0e'   # Orange
    }
    
    for metric in metrics:
        # Create figure with custom gridspec
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
        
        # Create main plot and legend axes
        ax = fig.add_subplot(gs[0])
        legend_ax = fig.add_subplot(gs[1])
        legend_ax.axis('off')
        
        for model in models:
            model_data = df[df['Model'] == model]
            color = colors[model]
            display_name = model_display_names[model]
            
            # Plot AO and AV
            ao_data = model_data[model_data['Modality'] == 'AO']
            av_data = model_data[model_data['Modality'] == 'AV']
            
            # Sort data by SNR
            ao_data = ao_data.sort_values('SNR')
            av_data = av_data.sort_values('SNR')
            
            # Plot lines
            ax.plot(ao_data['SNR'], ao_data[metric], '--', color=color, alpha=0.5,
                   label=f'{display_name} (AO)', marker='o', markersize=4)
            ax.plot(av_data['SNR'], av_data[metric], '-', color=color,
                   label=f'{display_name} (AV)', marker='o', markersize=4)
            
            # Add value labels for AO
            for idx, row in ao_data.iterrows():
                ax.annotate(f'{row[metric]:.2f}', 
                           (row['SNR'], row[metric]),
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center',
                           fontsize=6,
                           color=color,
                           alpha=0.7)
            
            # Add value labels for AV
            for idx, row in av_data.iterrows():
                ax.annotate(f'{row[metric]:.2f}', 
                           (row['SNR'], row[metric]),
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center',
                           fontsize=6,
                           color=color)
        
        # Customize plot
        ax.set_title(f'{metric} Values vs SNR', fontsize=14)
        ax.set_xlabel('SNR (dB)', fontsize=13)
        ax.set_ylabel(metric, fontsize=13)
        ax.set_xticks(snr_ticks)
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.3)
        
        # Create legend handles
        handles = []
        labels = []
        for model in models:
            color = colors[model]
            display_name = model_display_names[model]
            handles.append(plt.Line2D([0], [0], color=color, linestyle='--', alpha=0.5, marker='o', markersize=4))
            labels.append(f'{display_name} (AO)')
            handles.append(plt.Line2D([0], [0], color=color, linestyle='-', marker='o', markersize=4))
            labels.append(f'{display_name} (AV)')
        
        # Add legend at bottom
        legend_ax.legend(handles, labels, loc='center', ncol=len(models),
                        bbox_to_anchor=(0.5, 0.5), fontsize=10)
        
        # Save plot
        plt.savefig(Path(output_dir) / f'metric_values_{metric}.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()

def plot_combined_metrics_with_labels(df, output_dir):
    """Create a combined plot with all metrics in subplots, including value labels"""
    metrics = ['WER', 'CER', 'BERTScore', 'SemanticSimilarity', 'ViPS']
    
    # Create a mapping for display names
    model_display_names = {
        'avec': 'AVEC',
        'AV_relscore': 'AV-RelScore',
        'auto-avsr': 'Auto-AVSR'
    }
    
    models = df['Model'].unique()
    
    # Set style
    sns.set_style("whitegrid")
    
    # Define SNR ticks
    snr_ticks = [-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5]
    
    # Define professional color scheme
    colors = {
        'avec': '#1f77b4',      # Blue
        'AV_relscore': '#2ca02c',  # Green
        'auto-avsr': '#ff7f0e'   # Orange
    }
    
    # Create figure with custom gridspec
    fig = plt.figure(figsize=(12, 8))
    
    # Create gridspec with 2 rows, 3 columns
    gs = fig.add_gridspec(2, 3)
    
    # Create axes for plots (2x3 grid)
    axes = []
    for i in range(2):
        for j in range(3):
            if i * 3 + j < len(metrics):  # Only create axes for existing metrics
                ax = fig.add_subplot(gs[i, j])
                axes.append(ax)
    
    for ax, metric in zip(axes, metrics):
        for model in models:
            model_data = df[df['Model'] == model]
            color = colors[model]
            display_name = model_display_names[model]
            
            # Plot AO and AV
            ao_data = model_data[model_data['Modality'] == 'AO']
            av_data = model_data[model_data['Modality'] == 'AV']
            
            # Sort data by SNR
            ao_data = ao_data.sort_values('SNR')
            av_data = av_data.sort_values('SNR')
            
            # Plot lines with markers
            ax.plot(ao_data['SNR'], ao_data[metric], '--', color=color, alpha=0.5,
                   label=f'{display_name} (AO)', marker='o', markersize=4)
            ax.plot(av_data['SNR'], av_data[metric], '-', color=color,
                   label=f'{display_name} (AV)', marker='o', markersize=4)
            
            # Add value labels for AO (with black text and larger font)
            for idx, row in ao_data.iterrows():
                ax.annotate(f'{row[metric]:.2f}', 
                           (row['SNR'], row[metric]),
                           textcoords="offset points", 
                           xytext=(0,7), 
                           ha='center',
                           fontsize=7,
                           color='black',
                           weight='bold',
                           alpha=0.7,
                           bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
            
            # Add value labels for AV (with black text and larger font)
            for idx, row in av_data.iterrows():
                ax.annotate(f'{row[metric]:.2f}', 
                           (row['SNR'], row[metric]),
                           textcoords="offset points", 
                           xytext=(0,7), 
                           ha='center',
                           fontsize=7,
                           color='black',
                           weight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
        
        # Customize subplot
        ax.set_xlabel('SNR (dB)', fontsize=13)
        ax.set_ylabel(metric, fontsize=13)
        ax.set_xticks(snr_ticks)
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.3)
        
        # Reduce margins
        ax.margins(x=0.02)
    
    # Add a single legend for the entire figure
    # Create legend handles
    handles = []
    labels = []
    for model in models:
        color = colors[model]
        display_name = model_display_names[model]
        handles.append(plt.Line2D([0], [0], color=color, linestyle='--', alpha=0.5, marker='o', markersize=4))
        labels.append(f'{display_name} (AO)')
        handles.append(plt.Line2D([0], [0], color=color, linestyle='-', marker='o', markersize=4))
        labels.append(f'{display_name} (AV)')
    
    # Add legend at the bottom of the figure
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=3, fontsize=12, handlelength=1.5, columnspacing=1.0)
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.15)
    
    # Save plot with tight layout
    plt.savefig(Path(output_dir) / 'combined_metrics_with_labels.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_selected_metrics_vertical(df, output_dir):
    """Create a vertical plot with only ViPS, WER, and Semantic Similarity metrics"""
    # Only include the three requested metrics
    metrics = ['ViPS', 'WER', 'SemanticSimilarity']
    
    # Create a mapping for display names
    model_display_names = {
        'avec': 'AVEC',
        'AV_relscore': 'AV-RelScore',
        'auto-avsr': 'Auto-AVSR'
    }
    
    models = df['Model'].unique()
    
    # Set style
    sns.set_style("whitegrid")
    
    # Define SNR ticks
    snr_ticks = [-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5]
    
    # Define professional color scheme
    colors = {
        'avec': '#1f77b4',      # Blue
        'AV_relscore': '#2ca02c',  # Green
        'auto-avsr': '#ff7f0e'   # Orange
    }
    
    # Create figure with 3 rows, 1 column
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)  # Changed back to sharex=True
    
    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        for model in models:
            model_data = df[df['Model'] == model]
            color = colors[model]
            display_name = model_display_names[model]
            
            # Plot AO and AV
            ao_data = model_data[model_data['Modality'] == 'AO']
            av_data = model_data[model_data['Modality'] == 'AV']
            
            # Sort data by SNR
            ao_data = ao_data.sort_values('SNR')
            av_data = av_data.sort_values('SNR')
            
            # Plot lines with markers
            ax.plot(ao_data['SNR'], ao_data[metric], '--', color=color, alpha=0.5,
                   label=f'{display_name} (AO)', marker='o', markersize=5)
            ax.plot(av_data['SNR'], av_data[metric], '-', color=color,
                   label=f'{display_name} (AV)', marker='o', markersize=5)
            
            # Add value labels for AO with matching color
            for idx, row in ao_data.iterrows():
                ax.annotate(f'{row[metric]:.2f}', 
                           (row['SNR'], row[metric]),
                           textcoords="offset points", 
                           xytext=(0,7), 
                           ha='center',
                           fontsize=8,
                           color=color,
                           weight='bold',
                           alpha=0.7,
                           bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
            
            # Add value labels for AV with matching color
            for idx, row in av_data.iterrows():
                ax.annotate(f'{row[metric]:.2f}', 
                           (row['SNR'], row[metric]),
                           textcoords="offset points", 
                           xytext=(0,7), 
                           ha='center',
                           fontsize=8,
                           color=color,
                           weight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
            
            # Get reference value at 0dB for AO
            ref_value = model_data[(model_data['Modality'] == 'AO') & 
                                 (model_data['SNR'] == 0.0)][metric].iloc[0]
            
            # Find SNR gain
            av_snr = interpolate_snr_level(av_data['SNR'].values, 
                                         av_data[metric].values, 
                                         ref_value)
            
            if not np.isnan(av_snr):
                # Draw horizontal arrow
                ax.annotate('', 
                    xy=(0.0, ref_value),
                    xytext=(av_snr, ref_value),
                    arrowprops=dict(
                        arrowstyle='<->',
                        color='#262626',  # Light black color
                        alpha=0.6,
                        linewidth=1.0
                    ),
                    annotation_clip=True
                )
                
                # Add label above the line without background
                ax.text((av_snr + 0.0) / 2, ref_value + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
                       f'{0-av_snr:.1f}dB',
                       horizontalalignment='center',
                       verticalalignment='bottom',
                       color='black',
                       fontsize=9)
        
        # Customize subplot
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xticks(snr_ticks)
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.3)
        
        # Extend y-axis limits
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.15)  # Add extra padding
        
        # Add title only to the first subplot
        if i == 0:
            ax.set_title('Metric Performance vs SNR', fontsize=14)
        
        # Only add x-label to the bottom subplot
        if i == len(metrics) - 1:
            ax.set_xlabel('SNR (dB)', fontsize=12)
        else:
            # Hide x-axis label for non-bottom subplots
            ax.set_xlabel('')
    
    # Create legend handles
    handles = []
    labels = []
    for model in models:
        color = colors[model]
        display_name = model_display_names[model]
        handles.append(plt.Line2D([0], [0], color=color, linestyle='--', alpha=0.5, marker='o', markersize=5))
        labels.append(f'{display_name} (AO)')
        handles.append(plt.Line2D([0], [0], color=color, linestyle='-', marker='o', markersize=5))
        labels.append(f'{display_name} (AV)')
    
    # Add legend at the bottom of the figure
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01),
              ncol=3, fontsize=10, handlelength=1.5)
    
    # Adjust spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25, bottom=0.1)  # Reduced hspace to bring plots closer
    
    # Save plot
    plt.savefig(Path(output_dir) / 'selected_metrics_vertical.png', 
               bbox_inches='tight', dpi=300)
    plt.close()

def plot_selected_metrics_horizontal(df, output_dir):
    """Create a horizontal plot with only ViPS, WER, and Semantic Similarity metrics"""
    # Only include the three requested metrics
    metrics = ['ViPS', 'WER', 'SemanticSimilarity']
    
    # Create a mapping for display names
    model_display_names = {
        'avec': 'AVEC',
        'AV_relscore': 'AV-RelScore',
        'auto-avsr': 'Auto-AVSR'
    }
    
    # Define the order of models for display
    model_order = ['auto-avsr', 'AV_relscore', 'avec']
    
    # Set style
    sns.set_style("whitegrid")
    
    # Define SNR ticks
    snr_ticks = [-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5]
    
    # Define professional color scheme
    colors = {
        'avec': '#1f77b4',      # Blue
        'AV_relscore': '#2ca02c',  # Green
        'auto-avsr': '#ff7f0e'   # Orange
    }
    
    # Calculate SNR gains for all models and metrics
    snr_gains = {}
    for model in model_order:
        snr_gains[model] = {}
        model_data = df[df['Model'] == model]
        
        for metric in metrics:
            # Get reference value at 0dB for AO
            ref_value = model_data[(model_data['Modality'] == 'AO') & 
                                (model_data['SNR'] == 0.0)][metric].iloc[0]
            
            # Get AV data for interpolation
            av_data = model_data[model_data['Modality'] == 'AV']
            av_data = av_data.sort_values('SNR')
            
            # Find SNR gain
            av_snr = interpolate_snr_level(av_data['SNR'].values, 
                                        av_data[metric].values, 
                                        ref_value)
            
            if not np.isnan(av_snr):
                snr_gains[model][metric] = 0 - av_snr
            else:
                snr_gains[model][metric] = np.nan
    
    # Create figure with 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=False)
    
    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        for model in model_order:
            model_data = df[df['Model'] == model]
            color = colors[model]
            display_name = model_display_names[model]
            
            # Plot AO and AV
            ao_data = model_data[model_data['Modality'] == 'AO']
            av_data = model_data[model_data['Modality'] == 'AV']
            
            # Sort data by SNR
            ao_data = ao_data.sort_values('SNR')
            av_data = av_data.sort_values('SNR')
            
            # Plot lines with markers
            ax.plot(ao_data['SNR'], ao_data[metric], '--', color=color, alpha=0.5,
                   label=f'{display_name} (AO)', marker='o', markersize=7, linewidth=3.0)
            ax.plot(av_data['SNR'], av_data[metric], '-', color=color,
                   label=f'{display_name} (AV)', marker='o', markersize=7, linewidth=3.0)
            
            # Add value labels for AO with model color
            for idx, row in ao_data.iterrows():
                ax.annotate(f'{row[metric]:.2f}', 
                           (row['SNR'], row[metric]),
                           textcoords="offset points", 
                           xytext=(0,7), 
                           ha='center',
                           fontsize=9,
                           color=color,
                           weight='bold',
                           alpha=0.7,
                           bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
            
            # Add value labels for AV with model color
            for idx, row in av_data.iterrows():
                ax.annotate(f'{row[metric]:.2f}', 
                           (row['SNR'], row[metric]),
                           textcoords="offset points", 
                           xytext=(0,7), 
                           ha='center',
                           fontsize=9,
                           color=color,
                           weight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
        
        # Customize subplot
        ax.set_xlabel('SNR (dB)', fontsize=16)
        # Remove y-axis title for all plots
        ax.set_ylabel("")
        ax.set_xticks(snr_ticks)
        ax.tick_params(labelsize=16)
        ax.grid(True, alpha=0.3)
        
        # Only show y-axis tick values for the first plot
        if i != 0:
            ax.set_yticklabels([])
        
        # Set y-axis limits and ticks
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        
        # Add bold subtitle to each subplot
        ax.set_title(metric, fontsize=18, fontweight='bold')
        
        # Add SNR gains in the empty space within the subplot
        if metric == 'WER':
            y_pos = 0.98
            va = 'top'
            y_offset = -0.05
            display_models = model_order
            # Print label first
            ax.text(0.98, y_pos, "SNR Gains (dB):", transform=ax.transAxes, fontsize=11, fontweight='bold',
                    verticalalignment=va, horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))
            for j, model in enumerate(display_models):
                display_name = model_display_names[model]
                model_color = colors[model]
                gain_value = snr_gains[model][metric]
                if not np.isnan(gain_value):
                    current_y = y_pos + (j+1) * y_offset
                    ax.text(0.98, current_y, f"{display_name}: {gain_value:.1f}",
                            transform=ax.transAxes, fontsize=10, fontweight='bold',
                            verticalalignment=va, horizontalalignment='right',
                            color=model_color,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))
        else:
            y_pos = 0.18  # Start higher up to allow space for all values and label
            va = 'top'
            y_offset = -0.05
            display_models = model_order
            # Print label first
            ax.text(0.98, y_pos, "SNR Gains (dB):", transform=ax.transAxes, fontsize=11, fontweight='bold',
                    verticalalignment=va, horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))
            for j, model in enumerate(display_models):
                display_name = model_display_names[model]
                model_color = colors[model]
                gain_value = snr_gains[model][metric]
                if not np.isnan(gain_value):
                    current_y = y_pos + (j+1) * y_offset
                    ax.text(0.98, current_y, f"{display_name}: {gain_value:.1f}",
                            transform=ax.transAxes, fontsize=10, fontweight='bold',
                            verticalalignment=va, horizontalalignment='right',
                            color=model_color,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Remove the main title
    # fig.suptitle('Metric Performance vs SNR', fontsize=14, y=0.98)
    
    # Create legend handles
    handles = []
    labels = []
    for model in model_order:
        color = colors[model]
        display_name = model_display_names[model]
        handles.append(plt.Line2D([0], [0], color=color, linestyle='--', alpha=0.5, marker='o', markersize=7))
        labels.append(f'{display_name} (AO)')
        handles.append(plt.Line2D([0], [0], color=color, linestyle='-', marker='o', markersize=7))
        labels.append(f'{display_name} (AV)')
    
    # Add legend below the plots
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02),
              ncol=3, fontsize=15, handlelength=1.5)
    
    # Adjust spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.03, bottom=0.25, top=0.85)  # Increased bottom margin from 0.2 to 0.25
    
    # Save plot
    plt.savefig(Path(output_dir) / 'selected_metrics_horizontal.png', 
               bbox_inches='tight', dpi=300)
    plt.close()

def plot_selected_metrics_horizontal_no_labels(df, output_dir):
    """Create a horizontal plot with only ViPS, WER, and Semantic Similarity metrics without value labels"""
    # Only include the three requested metrics
    metrics = ['ViPS', 'WER', 'SemanticSimilarity']
    
    # Create a mapping for display names
    model_display_names = {
        'avec': 'AVEC',
        'AV_relscore': 'AV-RelScore',
        'auto-avsr': 'Auto-AVSR'
    }
    
    # Define the order of models for display
    model_order = ['auto-avsr', 'AV_relscore', 'avec']
    
    # Set style
    sns.set_style("whitegrid")
    
    # Define SNR ticks
    snr_ticks = [-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5]
    
    # Define professional color scheme
    colors = {
        'avec': '#1f77b4',      # Blue
        'AV_relscore': '#2ca02c',  # Green
        'auto-avsr': '#ff7f0e'   # Orange
    }
    
    # Calculate SNR gains for all models and metrics
    snr_gains = {}
    for model in model_order:
        snr_gains[model] = {}
        model_data = df[df['Model'] == model]
        
        for metric in metrics:
            # Get reference value at 0dB for AO
            ref_value = model_data[(model_data['Modality'] == 'AO') & 
                                (model_data['SNR'] == 0.0)][metric].iloc[0]
            
            # Get AV data for interpolation
            av_data = model_data[model_data['Modality'] == 'AV']
            av_data = av_data.sort_values('SNR')
            
            # Find SNR gain
            av_snr = interpolate_snr_level(av_data['SNR'].values, 
                                        av_data[metric].values, 
                                        ref_value)
            
            if not np.isnan(av_snr):
                snr_gains[model][metric] = 0 - av_snr
            else:
                snr_gains[model][metric] = np.nan
    
    # Create figure with 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=False)
    
    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        for model in model_order:
            model_data = df[df['Model'] == model]
            color = colors[model]
            display_name = model_display_names[model]
            
            # Plot AO and AV
            ao_data = model_data[model_data['Modality'] == 'AO']
            av_data = model_data[model_data['Modality'] == 'AV']
            
            # Sort data by SNR
            ao_data = ao_data.sort_values('SNR')
            av_data = av_data.sort_values('SNR')
            
            # Plot lines with markers
            ax.plot(ao_data['SNR'], ao_data[metric], '--', color=color, alpha=0.5,
                   label=f'{display_name} (AO)', marker='o', markersize=7, linewidth=3.0)
            ax.plot(av_data['SNR'], av_data[metric], '-', color=color,
                   label=f'{display_name} (AV)', marker='o', markersize=7, linewidth=3.0)
            
            # Get reference value at 0dB for AO
            ref_value = model_data[(model_data['Modality'] == 'AO') & 
                                (model_data['SNR'] == 0.0)][metric].iloc[0]
            
            # Find SNR gain
            av_snr = interpolate_snr_level(av_data['SNR'].values, 
                                        av_data[metric].values, 
                                        ref_value)
            
            if not np.isnan(av_snr):
                # Draw horizontal arrow
                ax.annotate('', 
                    xy=(0.0, ref_value),
                    xytext=(av_snr, ref_value),
                    arrowprops=dict(
                        arrowstyle='<->',
                        color='black',
                        alpha=0.6,
                        linewidth=3.0
                    ),
                    annotation_clip=True
                )
        
        # Customize subplot
        ax.set_xlabel('SNR (dB)', fontsize=16)
        # Remove y-axis title for all plots
        ax.set_ylabel("")
        ax.set_xticks(snr_ticks)
        ax.tick_params(labelsize=16)
        ax.grid(True, alpha=0.3)
        
        # Only show y-axis tick values for the first plot
        if i != 0:
            ax.set_yticklabels([])
        
        # Set y-axis limits and ticks
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        
        # Add bold subtitle to each subplot
        ax.set_title(metric, fontsize=18, fontweight='bold')
        
        # Add SNR gains in the empty space within the subplot
        if metric == 'WER':
            y_pos = 0.98
            va = 'top'
            y_offset = -0.05
            display_models = model_order
            # Print label first
            ax.text(0.98, y_pos, "SNR Gains (dB):", transform=ax.transAxes, fontsize=14, fontweight='bold',
                    verticalalignment=va, horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))
            for j, model in enumerate(display_models):
                display_name = model_display_names[model]
                model_color = colors[model]
                gain_value = snr_gains[model][metric]
                if not np.isnan(gain_value):
                    current_y = y_pos + (j+1) * y_offset
                    ax.text(0.98, current_y, f"{display_name}: {gain_value:.1f}",
                            transform=ax.transAxes, fontsize=13, fontweight='bold',
                            verticalalignment=va, horizontalalignment='right',
                            color=model_color,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))
        else:
            y_pos = 0.18  # Start higher up to allow space for all values and label
            va = 'top'
            y_offset = -0.05
            display_models = model_order
            # Print label first
            ax.text(0.98, y_pos, "SNR Gains (dB):", transform=ax.transAxes, fontsize=14, fontweight='bold',
                    verticalalignment=va, horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))
            for j, model in enumerate(display_models):
                display_name = model_display_names[model]
                model_color = colors[model]
                gain_value = snr_gains[model][metric]
                if not np.isnan(gain_value):
                    current_y = y_pos + (j+1) * y_offset
                    ax.text(0.98, current_y, f"{display_name}: {gain_value:.1f}",
                            transform=ax.transAxes, fontsize=13, fontweight='bold',
                            verticalalignment=va, horizontalalignment='right',
                            color=model_color,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Create legend handles
    handles = []
    labels = []
    for model in model_order:
        color = colors[model]
        display_name = model_display_names[model]
        handles.append(plt.Line2D([0], [0], color=color, linestyle='--', alpha=0.5, marker='o', markersize=7))
        labels.append(f'{display_name} (AO)')
        handles.append(plt.Line2D([0], [0], color=color, linestyle='-', marker='o', markersize=7))
        labels.append(f'{display_name} (AV)')
    
    # Add legend below the plots
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02),
              ncol=3, fontsize=15, handlelength=1.5)
    
    # Adjust spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.03, bottom=0.25, top=0.85)  # Increased bottom margin from 0.2 to 0.25
    
    # Save plot
    plt.savefig(Path(output_dir) / 'selected_metrics_horizontal_no_labels.png', 
               bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Read the metrics data
    df = pd.read_csv('./all_metrics.csv')
    
    # Create output directory
    output_dir = Path('snr_analysis')
    output_dir.mkdir(exist_ok=True)
    
    # Calculate SNR gains
    gains_df = calculate_snr_gains(df)
    
    # Save gains to CSV
    gains_df.to_csv(output_dir / 'snr_gains.csv', index=False)
    
    # Create visualization plots
    plot_snr_curves(df, output_dir)
    plot_snr_gains(gains_df, output_dir)
    plot_combined_metrics(df, output_dir)
    plot_metric_values_with_labels(df, output_dir)
    plot_combined_metrics_with_labels(df, output_dir)
    plot_selected_metrics_vertical(df, output_dir)
    plot_selected_metrics_horizontal(df, output_dir)
    plot_selected_metrics_horizontal_no_labels(df, output_dir)  # Add new plot without labels
    
    # Print summary statistics
    print("\nSNR Gains Summary:")
    print("==================")
    
    # Group by model and metric
    summary = gains_df.groupby(['Model', 'Metric'])['SNR_Gain'].mean().unstack()
    print("\nAverage SNR Gains (dB):")
    print(summary)
    
    # Find best performing model for each metric
    best_models = gains_df.loc[gains_df.groupby('Metric')['SNR_Gain'].idxmax()]
    print("\nBest Performing Models:")
    for _, row in best_models.iterrows():
        print(f"{row['Metric']}: {row['Model']} (Gain: {row['SNR_Gain']:.2f} dB)")

if __name__ == "__main__":
    main() 