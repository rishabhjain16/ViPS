#!/usr/bin/env python3
import os
# Set environment variable to prevent tokenizers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Suppress model loading warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Set up logging to suppress model warnings
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

import argparse
import pandas as pd
import numpy as np
import time
import gc

# Import tqdm for progress bars
try:
    from tqdm import tqdm
    tqdm_available = True
except ImportError:
    print("Warning: tqdm not installed. Install with: pip install tqdm")
    print("Progress bars will not be shown.")
    tqdm_available = False

from vis_phon import WeightedLipReadingEvaluator

# Import dependencies for MaFI calculation
import panphon
import panphon.distance

# Import nltk for WER and CER calculations
try:
    import nltk
    nltk_available = True
except ImportError:
    print("Warning: nltk not installed. Install with: pip install nltk")
    print("Some metrics like WER and CER will not be calculated.")
    nltk_available = False

# Check for GPU availability
import torch
def get_device(use_gpu=True):
    """Get the device to use for computations"""
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if use_gpu and not torch.cuda.is_available():
            print("GPU requested but not available. Using CPU instead.")
        else:
            print("Using CPU for computations")
    return device

# Use phonemizer instead of epitran for consistency
try:
    from phonemizer import phonemize
    from phonemizer.separator import Separator
    phonemizer_available = True
except ImportError:
    print("Warning: phonemizer not installed. Install with: pip install phonemizer")
    print("MaFI scores will not be calculated.")
    phonemizer_available = False

# Calculate WER and CER directly
def calculate_wer_cer(reference, hypothesis):
    """Calculate Word Error Rate and Character Error Rate directly"""
    results = {}
    
    # Handle empty inputs
    if not reference or not hypothesis:
        results['word_error_rate'] = 1.0 if hypothesis and not reference else 0.0 if not hypothesis and not reference else 1.0
        results['character_error_rate'] = 1.0 if hypothesis and not reference else 0.0 if not hypothesis and not reference else 1.0
        return results
        
    if nltk_available:
        # Word Error Rate
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        if ref_words:
            word_error = nltk.edit_distance(ref_words, hyp_words)
            wer = word_error / len(ref_words)
        else:
            wer = 1.0 if hyp_words else 0.0
            
        # Character Error Rate
        if len(reference) > 0:
            char_error = nltk.edit_distance(reference, hypothesis)
            cer = char_error / len(reference)
        else:
            cer = 1.0 if hypothesis else 0.0
            
        results['word_error_rate'] = wer
        results['character_error_rate'] = cer
    else:
        # Fallback calculation if nltk is not available
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        if ref_words:
            # Simple Levenshtein distance implementation for words
            word_error = sum(1 for x, y in zip(ref_words, hyp_words) if x != y)
            word_error += abs(len(ref_words) - len(hyp_words))
            wer = word_error / len(ref_words)
        else:
            wer = 1.0 if hyp_words else 0.0
            
        # Simple character-level calculation
        if reference:
            char_error = sum(1 for x, y in zip(reference, hypothesis) if x != y)
            char_error += abs(len(reference) - len(hypothesis))
            cer = char_error / len(reference)
        else:
            cer = 1.0 if hypothesis else 0.0
            
        results['word_error_rate'] = wer
        results['character_error_rate'] = cer
        
    return results

class MaFICalculator:
    def __init__(self):
        """Initialize the MaFI calculator with necessary tools"""
        self.dst = panphon.distance.Distance()
        
        # Set up phonemizer
        self.phonemizer_available = phonemizer_available
        if self.phonemizer_available:
            self.separator = Separator(word=' ', phone='')
            print("Phonemizer initialized successfully.")
        else:
            raise ImportError("Phonemizer is required for MaFI calculation")
        
        # Prepare regex for text normalization (like vis_phon)
        import re
        self.alphanumeric_pattern = re.compile(r'[^a-zA-Z0-9\s]')
        
    def normalize_text(self, text):
        """Normalize text by removing non-alphanumeric characters and extra whitespace."""
        if not text:
            return ""
        
        # Replace apostrophes with space
        text = text.replace("'", " ")
        
        # Remove all non-alphanumeric characters except spaces
        normalized = self.alphanumeric_pattern.sub('', text)
        
        # Replace multiple spaces with a single space and strip
        normalized = ' '.join(normalized.split())
        
        return normalized
        
    def text_to_ipa(self, text):
        """Convert text to IPA transcription using phonemizer (consistent with vis_phon)"""
        if not self.phonemizer_available:
            raise ImportError("Phonemizer is required for IPA conversion")
            
        # Normalize text first
        normalized_text = self.normalize_text(text)
        if not normalized_text:
            return ""
            
        # Use phonemizer to convert to IPA - specify 'espeak' as string instead of passing backend object
        ipa = phonemize(
            [normalized_text],
            backend='espeak',
            language='en-us',
            separator=self.separator,
            strip=True,
            preserve_punctuation=False,
            with_stress=False
        )[0]
        return ipa
    
    def calculate_phonological_distance(self, word1, word2):
        """Calculate phonological distance between two words"""
        # Always convert to IPA for consistency
        word1_ipa = self.text_to_ipa(word1)
        word2_ipa = self.text_to_ipa(word2)
        
        # Handle empty strings
        if not word1_ipa or not word2_ipa:
            if word1_ipa == word2_ipa:  # Both empty
                return 0.0
            else:  # One is empty, maximum distance
                return 1.0
        
        # Use the jt_weighted_feature_edit_distance_div_maxlen function as in the original MaFI paper
        # This will ensure we get the same range of values (-2.5 to 1) as the original implementation
        distance = self.dst.jt_weighted_feature_edit_distance_div_maxlen(word1_ipa, word2_ipa)
        return distance

def process_excel_with_metrics(excel_path, output_path=None, limit=None, include_additional_metrics=True, use_gpu=True, checkpoint_interval=1000, resume_from_checkpoint=False):
    """Process an Excel file with lip reading data, adding phonetic and viseme metrics."""
    # Set default output path if not provided (same folder as input file)
    if output_path is None:
        base_name = os.path.splitext(excel_path)[0]
        output_path = f"{base_name}_with_metrics.xlsx"

    # Define checkpoint path
    checkpoint_path = f"{os.path.splitext(output_path)[0]}_checkpoint.xlsx"
    
    print(f"Loading data from {excel_path}...")
    
    # Load the Excel file
    original_df = pd.read_excel(excel_path)
    df_to_process = original_df.copy()
    
    # Initialize starting point
    start_index = 0
    
    # Define all possible metric columns
    metric_columns = ['std_viseme_score', 'wgt_viseme_score', 'std_phonetic_score', 'wgt_phonetic_score',
                     'phonological_distance', 'mafi_score']
    
    additional_metric_columns = [
        'word_error_rate', 
        'character_error_rate',
        'word_similarity',
        'meteor_score',
        'rouge1_score',
        'rouge2_score',
        'rougeL_score',
        'sentence_bleu_score',
        'bertscore_precision',
        'bertscore_recall',
        'bertscore_f1',
        'semantic_similarity',
        'semantic_wer'
    ] if include_additional_metrics else []
    
    all_metric_columns = metric_columns + additional_metric_columns
    
    # Initialize metric columns in the dataframe
    for col in all_metric_columns:
        if col not in df_to_process.columns:
            df_to_process[col] = None
    
    # Check if resuming from checkpoint
    if resume_from_checkpoint and os.path.exists(checkpoint_path):
        print(f"Found checkpoint file at {checkpoint_path}")
        try:
            checkpoint_df = pd.read_excel(checkpoint_path)
            
            # Verify the checkpoint file has the required columns
            if not all(col in checkpoint_df.columns for col in ['Word', 'Response'] + metric_columns):
                print("Warning: Checkpoint file is missing required columns. Starting from beginning.")
            else:
                # Find the last processed row by checking all metric columns
                processed_mask = checkpoint_df[metric_columns].notna().any(axis=1)
                
                if processed_mask.any():
                    # Get the last processed index
                    start_index = processed_mask[processed_mask].index[-1] + 1
                    print(f"Resuming from row {start_index}")
                    
                    # First, copy all metric columns from checkpoint to our working dataframe
                    for col in checkpoint_df.columns:
                        if col in all_metric_columns:
                            # Only copy non-null values
                            mask = checkpoint_df[col].notna()
                            df_to_process.loc[mask, col] = checkpoint_df.loc[mask, col]
                    
                    print(f"Restored {start_index} previously processed rows from checkpoint")
                    
                    # Verify the data was properly restored
                    restored_count = df_to_process[metric_columns].notna().any(axis=1).sum()
                    print(f"Verified {restored_count} rows have metrics data")
                else:
                    print("No processed rows found in checkpoint. Starting from beginning.")
        except Exception as e:
            print(f"Error loading checkpoint file: {e}")
            print("Starting from beginning.")
            start_index = 0
    else:
        if resume_from_checkpoint:
            print("No checkpoint file found. Starting from beginning.")
    
    # Apply limit if specified
    if limit and limit > 0 and limit < len(df_to_process):
        print(f"Limiting processing to first {limit} rows for testing")
        df_to_process = df_to_process.iloc[:limit].copy()
    
    # Get device for computations
    device = get_device(use_gpu)
    
    # Initialize the evaluators
    print("Initializing phonetic evaluator...")
    evaluator = WeightedLipReadingEvaluator()
    evaluator.device = device
    
    # Initialize MaFI calculator if possible
    try:
        print("Initializing phonological distance calculator...")
        mafi_calculator = MaFICalculator()
        phonological_distance_available = True
        print("Phonological distance calculator initialized successfully.")
    except Exception as e:
        print(f"Warning: Could not initialize phonological distance calculator: {e}")
        print("Phonological distances will not be calculated.")
        phonological_distance_available = False
    
    # Process each row with a progress bar
    total_rows = len(df_to_process) - start_index
    pbar = tqdm(total=total_rows, desc="Processing words", unit="word", initial=start_index)
    
    last_checkpoint_time = time.time()
    rows_since_checkpoint = 0
    
    try:
        for index in range(start_index, len(df_to_process)):
            pbar.set_description(f"Processing row {index+1}/{len(df_to_process)}")
            
            # Skip if this row already has all metrics calculated
            if df_to_process.loc[index, metric_columns].notna().all():
                pbar.update(1)
                continue
            
            row = df_to_process.iloc[index]
            reference = str(row['Word'])
            hypothesis = str(row['Response'])
            
            # Skip empty values
            if pd.isna(reference) or pd.isna(hypothesis):
                pbar.update(1)
                continue
            
            try:
                # Calculate metrics
                result = evaluator.compare_standard_and_weighted(reference, hypothesis)
                
                # Add metrics to dataframe
                df_to_process.at[index, 'std_viseme_score'] = result['standard']['viseme_score']
                df_to_process.at[index, 'wgt_viseme_score'] = result['weighted']['phonetically_weighted_viseme_score']
                df_to_process.at[index, 'std_phonetic_score'] = result['standard']['phonetic_alignment_score']
                df_to_process.at[index, 'wgt_phonetic_score'] = result['weighted']['phonetic_alignment_score']
                
                # Calculate phonological distance if available
                if phonological_distance_available:
                    try:
                        distance = mafi_calculator.calculate_phonological_distance(reference, hypothesis)
                        df_to_process.at[index, 'phonological_distance'] = distance
                        df_to_process.at[index, 'mafi_score'] = -distance
                    except Exception as e:
                        pbar.write(f"Error calculating phonological distance for row {index}: {e}")
                
                # Calculate additional metrics if requested
                if include_additional_metrics:
                    # Calculate WER and CER
                    if 'word_error_rate' in additional_metric_columns or 'character_error_rate' in additional_metric_columns:
                        try:
                            wer_cer_results = calculate_wer_cer(reference, hypothesis)
                            for metric, value in wer_cer_results.items():
                                if metric in additional_metric_columns:
                                    df_to_process.at[index, metric] = value
                        except Exception as e:
                            pbar.write(f"Error calculating WER/CER for row {index}: {e}")
                    
                    # Calculate other metrics
                    try:
                        example_pair = [(reference, hypothesis)]
                        _, per_example_metrics = evaluator.calculate_additional_metrics(example_pair)
                        
                        if per_example_metrics and len(per_example_metrics) > 0:
                            for metric_name in additional_metric_columns:
                                if metric_name not in ['word_error_rate', 'character_error_rate'] and metric_name in per_example_metrics[0]:
                                    df_to_process.at[index, metric_name] = per_example_metrics[0][metric_name]
                    except Exception as e:
                        pbar.write(f"Error calculating additional metrics for row {index}: {e}")
                
                # Checkpoint logic
                rows_since_checkpoint += 1
                if rows_since_checkpoint >= checkpoint_interval:
                    print(f"\nSaving checkpoint at row {index+1}...")
                    # Verify data before saving checkpoint
                    processed_count = df_to_process[metric_columns].notna().any(axis=1).sum()
                    print(f"Saving checkpoint with {processed_count} processed rows...")
                    df_to_process.to_excel(checkpoint_path, index=False)
                    rows_since_checkpoint = 0
                    last_checkpoint_time = time.time()
                    
            except Exception as e:
                pbar.write(f"Error processing row {index}: {e}")
                # Save checkpoint on error
                print(f"\nError encountered. Saving checkpoint at row {index}...")
                df_to_process.to_excel(checkpoint_path, index=False)
                
            # Update progress bar
            pbar.update(1)
            
            # Free up memory periodically
            if index % 1000 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving checkpoint...")
        df_to_process.to_excel(checkpoint_path, index=False)
        print(f"Checkpoint saved at row {index}. You can resume later using --resume flag.")
        return df_to_process
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Saving checkpoint...")
        df_to_process.to_excel(checkpoint_path, index=False)
        raise
    finally:
        pbar.close()
    
    # Save final results
    print(f"Saving final results to {output_path}...")
    df_to_process.to_excel(output_path, index=False)
    
    # Remove checkpoint file if process completed successfully
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    print(f"Completed! Results saved to {output_path}")
    
    return df_to_process

def create_word_averages_csv(excel_results, csv_path, output_path=None, include_additional_metrics=True):
    """
    Calculate average metrics per word from Excel results and add them to a CSV file.
    
    Args:
        excel_results: DataFrame containing the processed metrics from Excel
        csv_path: Path to the CSV file with words to enhance
        output_path: Path to save the enhanced CSV
        include_additional_metrics: Whether to include additional metrics beyond basic viseme/phonetic scores
    """
    # Set default output path if not provided
    if output_path is None:
        base_name = os.path.splitext(csv_path)[0]
        output_path = f"{base_name}_with_averages.csv"
    
    print(f"\nLoading CSV data from {csv_path}...")
    
    # Load the CSV file
    csv_df = pd.read_csv(csv_path)
    
    # Verify that Word column exists in CSV
    if 'Word' not in csv_df.columns:
        raise ValueError("Column 'Word' not found in the CSV file")
        
    print(f"Calculating average metrics for {len(csv_df)} words...")
    
    # Check which metrics are available in excel_results
    metric_cols = []
    if 'std_viseme_score' in excel_results.columns:
        metric_cols.append('std_viseme_score')
    if 'wgt_viseme_score' in excel_results.columns:
        metric_cols.append('wgt_viseme_score')
    if 'std_phonetic_score' in excel_results.columns:
        metric_cols.append('std_phonetic_score')
    if 'wgt_phonetic_score' in excel_results.columns:
        metric_cols.append('wgt_phonetic_score')
    if 'phonological_distance' in excel_results.columns:
        metric_cols.append('phonological_distance')
    if 'mafi_score' in excel_results.columns:
        metric_cols.append('mafi_score')
    
    # Add additional metrics if requested and available
    additional_metric_cols = []
    if include_additional_metrics:
        possible_additional_metrics = [
            'word_error_rate', 
            'character_error_rate',
            'word_similarity',
            'meteor_score',
            'rouge1_score',
            'rouge2_score',
            'rougeL_score',
            'sentence_bleu_score',
            'bertscore_precision',
            'bertscore_recall',
            'bertscore_f1',
            'semantic_similarity',
            'semantic_wer'
        ]
        
        # Check which additional metrics are available in excel_results
        for col in possible_additional_metrics:
            if col in excel_results.columns and not excel_results[col].isnull().all():
                additional_metric_cols.append(col)
                metric_cols.append(col)
                print(f"Adding average for metric: {col}")
    
    if not metric_cols:
        print("No metric columns found in Excel results")
        return csv_df
    
    # Calculate average metrics per word
    word_averages = {}
    
    # Group by Word and calculate mean
    print("Grouping and calculating averages by word...")
    word_groups = excel_results.groupby('Word')
    
    # Use tqdm to show progress
    pbar = tqdm(total=len(word_groups), desc="Calculating word averages", unit="word")
    for word, group in word_groups:
        word_averages[word] = {}
        
        # Calculate mean for each metric - avoid columns with all NaN values
        for col in metric_cols:
            if not group[col].isnull().all():  # Only if we have non-null values
                word_averages[word][col + '_avg'] = group[col].mean()
        
        pbar.update(1)
    
    pbar.close()
    
    # Add average metrics to CSV DataFrame
    for col in metric_cols:
        csv_df[col + '_avg'] = None
    
    # Fill in the average metric values
    print("Applying averages to CSV dataframe...")
    pbar = tqdm(total=len(csv_df), desc="Updating words in CSV", unit="word")
    for i, row in csv_df.iterrows():
        word = row['Word']
        if word in word_averages:
            for metric, value in word_averages[word].items():
                csv_df.at[i, metric] = value
        pbar.update(1)
    
    pbar.close()
    
    # Save the enhanced CSV
    print(f"Saving enhanced CSV to {output_path}...")
    csv_df.to_csv(output_path, index=False)
    
    print(f"Completed! CSV with average metrics saved to {output_path}")
    
    return csv_df

def main():
    parser = argparse.ArgumentParser(description='Add phonetic and viseme metrics to Excel file and calculate word averages')
    parser.add_argument('excel_path', help='Path to the input Excel file with Word-Response pairs')
    parser.add_argument('--csv-path', '-c', help='Path to a CSV file with Words to calculate average metrics for')
    parser.add_argument('--output', '-o', help='Path to save the output Excel file (defaults to input_with_metrics.xlsx in same folder)')
    parser.add_argument('--csv-output', help='Path to save the enhanced CSV file (defaults to input_with_averages.csv in same folder)')
    parser.add_argument('--limit', '-l', type=int, help='Limit processing to first N rows of Excel file (for testing)')
    parser.add_argument('--no-additional-metrics', action='store_true', help='Skip calculation of additional metrics (WER, CER, BLEU, etc.)')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration even if available')
    parser.add_argument('--checkpoint-interval', type=int, default=500, help='Save checkpoint after processing this many rows')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint if available')

    args = parser.parse_args()
    
    # Determine whether to include additional metrics and use GPU
    include_additional_metrics = not args.no_additional_metrics
    use_gpu = not args.no_gpu
    
    # Show configuration
    print("=== Configuration ===")
    print(f"Input Excel file: {args.excel_path}")
    print(f"Output Excel file: {args.output or 'default'}")
    print(f"CSV input: {args.csv_path or 'none'}")
    print(f"CSV output: {args.csv_output or 'default'}")
    print(f"Row limit: {args.limit or 'none'}")
    print(f"Include additional metrics: {include_additional_metrics}")
    print(f"Use GPU: {use_gpu}")
    print(f"Checkpoint interval: {args.checkpoint_interval}")
    print(f"Resume from checkpoint: {args.resume}")
    print("====================")
    
    # Process Excel file
    excel_results = process_excel_with_metrics(
        args.excel_path,
        args.output,
        args.limit,
        include_additional_metrics=include_additional_metrics,
        use_gpu=use_gpu,
        checkpoint_interval=args.checkpoint_interval,
        resume_from_checkpoint=args.resume
    )
    
    # If CSV path is provided, calculate and add word averages
    if args.csv_path:
        create_word_averages_csv(
            excel_results,
            args.csv_path,
            args.csv_output,
            include_additional_metrics=include_additional_metrics
        )

if __name__ == "__main__":
    main() 