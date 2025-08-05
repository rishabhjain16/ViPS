#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from vis_phon import WeightedLipReadingEvaluator

class SNRMetricAnalyzer:
    """Analyzer for comparing lip reading metrics across different SNR levels"""
    
    def __init__(self, evaluator=None, output_dir="snr_analysis_output"):
        """Initialize the analyzer with an evaluator and output directory"""
        self.evaluator = evaluator if evaluator else WeightedLipReadingEvaluator()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.snr_results = {}
        self.combined_data = []
    
    def process_json_file(self, json_path, snr_level):
        """Process a single JSON file for a given SNR level"""
        print(f"\nProcessing SNR level {snr_level} from {json_path}...")
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            pairs = self._extract_pairs(data)
            if not pairs:
                print(f"Warning: No reference-hypothesis pairs found in {json_path}")
                return None
            
            print(f"Found {len(pairs)} reference-hypothesis pairs")
            
            results = []
            total_pairs = len(pairs)
            snr_level_str = str(snr_level)
            
            # Initialize BERTScore
            bertscore_results = None
            try:
                import bert_score
                print("Initializing BERTScore model...")
                references = [ref for ref, _ in pairs]
                hypotheses = [hyp for _, hyp in pairs]
                P, R, F1 = bert_score.score(hypotheses, references, lang="en", rescale_with_baseline=True)
                bertscore_results = {
                    'f1': F1.cpu().numpy()
                }
                print("BERTScore calculation complete.")
            except Exception as e:
                print(f"Could not calculate BERTScore: {e}")
            
            # Calculate semantic similarity
            semantic_similarities = []
            try:
                from sentence_transformers import SentenceTransformer
                print("Calculating semantic similarities...")
                model = SentenceTransformer('all-MiniLM-L6-v2')
                references = [ref for ref, _ in pairs]
                hypotheses = [hyp for _, hyp in pairs]
                ref_embeddings = model.encode(references)
                hyp_embeddings = model.encode(hypotheses)
                
                for i in range(len(references)):
                    ref_emb = ref_embeddings[i]
                    hyp_emb = hyp_embeddings[i]
                    ref_norm = np.linalg.norm(ref_emb)
                    hyp_norm = np.linalg.norm(hyp_emb)
                    
                    if ref_norm > 0 and hyp_norm > 0:
                        similarity = np.dot(ref_emb, hyp_emb) / (ref_norm * hyp_norm)
                        semantic_similarities.append(float(similarity))
                    else:
                        semantic_similarities.append(0.0)
                print("Semantic similarity calculation complete.")
            except Exception as e:
                print(f"Could not calculate semantic similarities: {e}")
                semantic_similarities = [0.0] * len(pairs)
            
            for i, (ref, hyp) in enumerate(pairs):
                if i % 10 == 0:
                    print(f"Processing pair {i+1}/{total_pairs}...")
                
                comparison = self.evaluator.compare_standard_and_weighted(ref, hyp)
                metrics = self._calculate_basic_metrics(ref, hyp)
                
                result = {
                    'reference': ref,
                    'hypothesis': hyp,
                    'snr': snr_level_str,
                    'ViPS': comparison['weighted']['phonetically_weighted_viseme_score'],
                    'wer': metrics['wer'],
                    'cer': metrics['cer'],
                    'semantic_similarity': semantic_similarities[i] if i < len(semantic_similarities) else 0.0
                }
                
                # Add BERTScore if available
                if bertscore_results is not None:
                    result['bertscore'] = float(bertscore_results['f1'][i])
                
                results.append(result)
                self.combined_data.append(result)
            
            summary = self._calculate_summary_stats(results)
            self.snr_results[snr_level_str] = {
                'results': results,
                'summary': summary
            }
            
            print(f"Processed SNR level {snr_level}: {len(results)} pairs")
            return results
            
        except Exception as e:
            print(f"Error processing file {json_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_pairs(self, data):
        """Extract reference-hypothesis pairs from JSON data in various formats"""
        pairs = []
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if 'reference' in item and 'hypothesis' in item:
                        pairs.append((item['reference'], item['hypothesis']))
                    elif 'ref' in item and 'hyp' in item:
                        pairs.append((item['ref'], item['hyp']))
                    elif 'ref' in item and 'hypo' in item:
                        pairs.append((item['ref'], item['hypo']))
                    elif 'ground_truth' in item and 'prediction' in item:
                        pairs.append((item['ground_truth'], item['prediction']))
        
        elif isinstance(data, dict):
            if 'ref' in data and 'hyp' in data and isinstance(data['ref'], list) and isinstance(data['hyp'], list):
                pairs = list(zip(data['ref'], data['hyp']))
            elif 'ref' in data and 'hypo' in data and isinstance(data['ref'], list) and isinstance(data['hypo'], list):
                pairs = list(zip(data['ref'], data['hypo']))
            elif 'reference' in data and 'hypothesis' in data and isinstance(data['reference'], list) and isinstance(data['hypothesis'], list):
                pairs = list(zip(data['reference'], data['hypothesis']))
            elif 'ground_truth' in data and 'prediction' in data and isinstance(data['ground_truth'], list) and isinstance(data['prediction'], list):
                pairs = list(zip(data['ground_truth'], data['prediction']))
        
        return pairs
    
    def _calculate_basic_metrics(self, reference, hypothesis):
        """Calculate basic metrics for a reference-hypothesis pair"""
        import nltk
        from difflib import SequenceMatcher
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        metrics = {}
        
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        if ref_words:
            word_error = nltk.edit_distance(ref_words, hyp_words)
            metrics['wer'] = word_error / len(ref_words)
        else:
            metrics['wer'] = 1.0 if hyp_words else 0.0
        
        if reference:
            char_error = nltk.edit_distance(reference, hypothesis)
            metrics['cer'] = char_error / len(reference)
        else:
            metrics['cer'] = 1.0 if hypothesis else 0.0
        
        metrics['word_similarity'] = SequenceMatcher(None, reference, hypothesis).ratio()
        
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            if ref_words:
                smoothie = SmoothingFunction().method1
                metrics['bleu1'] = sentence_bleu([ref_words], hyp_words, 
                                              weights=(1, 0, 0, 0),
                                              smoothing_function=smoothie)
            else:
                metrics['bleu1'] = 0.0
        except:
            metrics['bleu1'] = 0.0
            
        try:
            from nltk.translate.meteor_score import meteor_score
            if ref_words:
                metrics['meteor'] = meteor_score([ref_words], hyp_words)
            else:
                metrics['meteor'] = 0.0
        except:
            metrics['meteor'] = 0.0
            
        return metrics
    
    def _calculate_summary_stats(self, results):
        """Calculate summary statistics for a set of results"""
        summary = {}
        
        if not results:
            return summary
        
        metrics = ['ViPS', 'wer', 'cer', 'semantic_similarity', 'bertscore']
        
        for metric in metrics:
            values = [r[metric] for r in results if metric in r]
            if values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_median'] = np.median(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_min'] = np.min(values)
                summary[f'{metric}_max'] = np.max(values)
        
        return summary
    
    def create_visualizations(self):
        """Create visualizations comparing metrics across SNR levels"""
        if not self.snr_results:
            print("No results to visualize")
            return
        
        print("\nCreating visualizations...")
        df = pd.DataFrame(self.combined_data)
        
        # Define metrics to compare
        comparison_metrics = [
            ('wer', 'WER'),
            ('cer', 'CER'),
            ('semantic_similarity', 'Semantic Similarity'),
            ('bertscore', 'BERTScore')
        ]
        
        # Only include metrics that exist in the dataframe
        comparison_metrics = [(m, t) for m, t in comparison_metrics if m in df.columns]
        
        # Get SNR levels and sort them numerically
        df['snr_float'] = df['snr'].astype(float)  # Convert to float for proper sorting
        snr_labels = sorted(df['snr'].unique(), key=float)  # Sort numerically
        
        # 1. Create correlation heatmap
        self._create_correlation_heatmap(df, comparison_metrics, snr_labels)
        
        # 2. Create R-squared heatmap
        self._create_r_squared_heatmap(df, comparison_metrics, snr_labels)
        
        # 3. Create line plots for metric values
        self._create_metric_line_plots(df, comparison_metrics, snr_labels)
        
        # 4. Create combined correlation and R-squared plot
        self._create_combined_analysis_plot(df, comparison_metrics, snr_labels)

    def _create_correlation_heatmap(self, df, comparison_metrics, snr_labels):
        """Create correlation heatmap between ViPS and other metrics"""
        correlation_data = []
        
        for snr in snr_labels:
            snr_df = df[df['snr'] == snr]
            for metric, title in comparison_metrics:
                corr = snr_df['ViPS'].corr(snr_df[metric])
                correlation_data.append({
                    'SNR': float(snr),  # Convert to float for proper sorting
                    'Metric': title,
                    'Correlation': corr
                })
        
        corr_df = pd.DataFrame(correlation_data)
        corr_matrix = corr_df.pivot(index='Metric', columns='SNR', values='Correlation')
        # Sort columns numerically
        corr_matrix = corr_matrix.reindex(columns=sorted(corr_matrix.columns, key=float))
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, center=0,
                   fmt='.2f', annot_kws={"size": 15})
        plt.title('Correlation between ViPS and Other Metrics Across SNR Levels', fontsize=19)
        plt.xlabel('SNR Level (dB)', fontsize=18)
        plt.ylabel('Metric', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=19)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'vips_correlation_heatmap.png'), dpi=300)
        plt.close()
            
    def _calculate_r_squared(self, x, y):
        """Safely calculate R-squared value with error handling"""
        try:
            if len(x) <= 1 or len(y) <= 1:
                return np.nan
            
            # Remove any NaN values
            mask = ~(np.isnan(x) | np.isnan(y))
            x = x[mask]
            y = y[mask]
            
            if len(x) <= 1 or len(y) <= 1:
                return np.nan
            
            # Check for constant values
            if np.all(x == x[0]) or np.all(y == y[0]):
                return np.nan
            
            # Calculate R-squared
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            y_pred = p(x)
            
            ss_total = np.sum((y - y.mean())**2)
            if ss_total == 0:
                return np.nan
                
            ss_residual = np.sum((y - y_pred)**2)
            r_squared = 1 - (ss_residual / ss_total)
            
            # Check for valid R-squared value
            if not (0 <= r_squared <= 1):
                return np.nan
                
            return r_squared
            
        except Exception as e:
            print(f"Error calculating R-squared: {e}")
            return np.nan

    def _create_r_squared_heatmap(self, df, comparison_metrics, snr_labels):
        """Create R-squared heatmap between ViPS and other metrics"""
        r_squared_data = []
        
        for snr in snr_labels:
            snr_df = df[df['snr'] == snr]
            for metric, title in comparison_metrics:
                # Calculate R-squared
                x = snr_df[metric].values
                y = snr_df['ViPS'].values
                r_squared = self._calculate_r_squared(x, y)
                
                r_squared_data.append({
                    'SNR': float(snr),  # Convert to float for proper sorting
                        'Metric': title,
                    'R_squared': r_squared
                })
        
        r2_df = pd.DataFrame(r_squared_data)
        r2_matrix = r2_df.pivot(index='Metric', columns='SNR', values='R_squared')
        # Sort columns numerically
        r2_matrix = r2_matrix.reindex(columns=sorted(r2_matrix.columns, key=float))
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(r2_matrix, annot=True, cmap='YlOrRd', vmin=0, vmax=1,
                   fmt='.2f', annot_kws={"size": 15})
        
        plt.title('R² Values between ViPS and Other Metrics Across SNR Levels', fontsize=19)
        plt.xlabel('SNR Level (dB)', fontsize=18)
        plt.ylabel('Metric', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=19)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'vips_r_squared_heatmap.png'), dpi=300)
        plt.close()
        
    def _create_metric_line_plots(self, df, comparison_metrics, snr_labels):
        """Create line plots showing metric values across SNR levels"""
        # Calculate mean values for each metric at each SNR level
        metric_means = df.groupby('snr').agg({
            'ViPS': 'mean',
            'snr_float': 'first',  # Keep the float value for sorting
            **{metric: 'mean' for metric, _ in comparison_metrics}
        }).reset_index()
        
        # Sort by SNR float value
        metric_means = metric_means.sort_values('snr_float')
        
        # Define consistent colors for metrics
        metric_colors = {
            'ViPS': 'blue',
            'WER': '#ff7f0e',  # Orange
            'CER': '#2ca02c',  # Green
            'Semantic Similarity': '#d62728',  # Red
            'BERTScore': '#9467bd'  # Purple
        }
        
        # Create line plot
        plt.figure(figsize=(12, 8))
        
        def add_value_annotations(x, y, metric_name):
            """Helper function to add value annotations"""
            # Fixed positions based on metric
            if metric_name in ['BERTScore', 'WER']:
                y_offset = 8  # Above the line
            else:  # ViPS, Semantic Similarity and CER
                y_offset = -12  # Below the line
            
            plt.annotate(f'{y:.2f}', (x, y), 
                        xytext=(0, y_offset),
                        textcoords='offset points',
                        ha='center',
                        va='bottom' if y_offset > 0 else 'top',
                        fontsize=17,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.3))
        
        # Plot ViPS
        plt.plot(metric_means['snr_float'], metric_means['ViPS'], 'o-', 
                color=metric_colors['ViPS'], label='ViPS', linewidth=2)
        # Add annotations for ViPS
        for x, y in zip(metric_means['snr_float'], metric_means['ViPS']):
            add_value_annotations(x, y, 'ViPS')
        
        # Plot other metrics
        for metric, title in comparison_metrics:
            plt.plot(metric_means['snr_float'], metric_means[metric], 'o-',
                    color=metric_colors[title], label=title, linewidth=2)
            # Add annotations
            for x, y in zip(metric_means['snr_float'], metric_means[metric]):
                add_value_annotations(x, y, title)
        
        plt.title('Average Metric Values Across SNR Levels', fontsize=19)
        plt.xlabel('SNR Level (dB)', fontsize=18)
        plt.ylabel('Score', fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=17)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metric_values_line_plot.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_combined_analysis_plot(self, df, comparison_metrics, snr_labels):
        """Create a combined plot showing correlations and R-squared values"""
        analysis_data = []
        
        # Define consistent colors for metrics
        metric_colors = {
            'WER': '#ff7f0e',  # Orange
            'CER': '#2ca02c',  # Green
            'Semantic Similarity': '#d62728',  # Red
            'BERTScore': '#9467bd'  # Purple
        }
        
        # Calculate correlation and R-squared
        for snr in snr_labels:
            snr_df = df[df['snr'] == snr]
            
            for metric, title in comparison_metrics:
                # Calculate correlation
                corr = snr_df['ViPS'].corr(snr_df[metric])
                
                # Calculate R-squared
                x = snr_df[metric].values
                y = snr_df['ViPS'].values
                r_squared = self._calculate_r_squared(x, y)
                
                analysis_data.append({
                    'SNR': float(snr),
                    'Metric': title,
                    'Correlation': corr,
                    'R_squared': r_squared
                })
        
        analysis_df = pd.DataFrame(analysis_data)
        
        # Sort by SNR values
        analysis_df = analysis_df.sort_values('SNR')
        
        # Create subplot with shared x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
        
        def add_value_annotations(ax, all_points, current_x, current_y, metric_title, is_correlation=False):
            """Helper function to add value annotations, handling overlaps"""
            if is_correlation:
                # Fixed positions for correlation plot based on metric
                if metric_title in ['BERTScore', 'WER']:
                    y_offset = 8  # Above the line
                else:  # Semantic Similarity and CER
                    y_offset = -12  # Below the line
            else:
                # For R-squared plot, keep the dynamic positioning
                nearby_points = [(x, y) for x, y in all_points if abs(x - current_x) < 0.01]
                if len(nearby_points) > 1:
                    y_values = [y for _, y in nearby_points]
                    sorted_y = sorted(y_values)
                    if current_y >= sorted_y[-2]:
                        y_offset = 8
                    else:
                        y_offset = -12
                else:
                    y_offset = 8
            
            ax.annotate(f'{current_y:.2f}', (current_x, current_y), 
                       xytext=(0, y_offset),
                       textcoords='offset points',
                       ha='center',
                       va='bottom' if y_offset > 0 else 'top',
                       fontsize=17,
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.3))
        
        # Plot correlations
        lines = []
        correlation_points = []
        for metric, title in comparison_metrics:
            metric_data = analysis_df[analysis_df['Metric'] == title]
            correlation_points.extend(zip(metric_data['SNR'], metric_data['Correlation']))
        
        for i, (metric, title) in enumerate(comparison_metrics):
            metric_data = analysis_df[analysis_df['Metric'] == title]
            line = ax1.plot(metric_data['SNR'], metric_data['Correlation'], 'o-',
                          label=title, linewidth=2, color=metric_colors[title])[0]
            lines.append(line)
            for x, y in zip(metric_data['SNR'], metric_data['Correlation']):
                add_value_annotations(ax1, correlation_points, x, y, title, is_correlation=True)
        
        ax1.set_title('Pearson Correlation with ViPS', fontsize=19)
        ax1.set_ylabel('Correlation', fontsize=18)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=17)
        ax1.set_ylim(-1.05, 1.05)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        
        # Plot R-squared values
        lines = []
        r_squared_points = []
        for metric, title in comparison_metrics:
            metric_data = analysis_df[analysis_df['Metric'] == title]
            r_squared_points.extend(zip(metric_data['SNR'], metric_data['R_squared']))
        
        for i, (metric, title) in enumerate(comparison_metrics):
            metric_data = analysis_df[analysis_df['Metric'] == title]
            line = ax2.plot(metric_data['SNR'], metric_data['R_squared'], 'o-',
                          label=title, linewidth=2, color=metric_colors[title])[0]
            lines.append(line)
            for x, y in zip(metric_data['SNR'], metric_data['R_squared']):
                add_value_annotations(ax2, r_squared_points, x, y, title, is_correlation=False)
        
        ax2.set_title('R² Values with ViPS', fontsize=19)
        ax2.set_xlabel('SNR Level (dB)', fontsize=18)
        ax2.set_ylabel('R² Value', fontsize=18)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='lower left', bbox_to_anchor=(0.0, 0.0), fontsize=17)
        ax2.set_ylim(-0.05, 1.05)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'combined_analysis_plot.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self):
        """Save all results to JSON files"""
        if not self.snr_results:
            print("No results to save")
            return
        
        print("\nSaving results to files...")
        
        results_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        for snr, results in self.snr_results.items():
            output_file = os.path.join(results_dir, f'results_snr_{snr}.json')
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved results for SNR {snr} to {output_file}")
        
        summary_data = {}
        for snr, results in self.snr_results.items():
            summary_data[snr] = results['summary']
        
        summary_file = os.path.join(self.output_dir, 'summary_across_snr.json')
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Saved summary across SNR levels to {summary_file}")
        
        if self.combined_data:
            csv_file = os.path.join(self.output_dir, 'all_metrics.csv')
            pd.DataFrame(self.combined_data).to_csv(csv_file, index=False)
            print(f"Saved all metrics to {csv_file}")


def main():
    """Main function for running the SNR metric analysis"""
    parser = argparse.ArgumentParser(description='Analyze lip reading metrics across different SNR levels')
    
    parser.add_argument('--snr_files', nargs='+', required=True, 
                        help='List of JSON files with reference-hypothesis pairs for different SNR levels')
    parser.add_argument('--snr_levels', nargs='+', required=True, 
                        help='SNR levels corresponding to each file (must match --snr_files order)')
    parser.add_argument('--output_dir', type=str, default='snr_analysis_output',
                        help='Directory to save analysis outputs')
    parser.add_argument('--weight_method', type=str, choices=['both', 'entropy', 'distinctiveness'], 
                        default='both', help='Method for phonetic feature weighting')
    
    args = parser.parse_args()
    
    if len(args.snr_files) != len(args.snr_levels):
        parser.error("Number of files must match number of SNR levels")
    
    evaluator = WeightedLipReadingEvaluator(weight_method=args.weight_method)
    analyzer = SNRMetricAnalyzer(evaluator=evaluator, output_dir=args.output_dir)
    
    for json_file, snr_level in zip(args.snr_files, args.snr_levels):
        analyzer.process_json_file(json_file, str(snr_level))
    
    analyzer.create_visualizations()
    analyzer.save_results()
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 