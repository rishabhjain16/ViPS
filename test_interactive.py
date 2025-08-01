#!/usr/bin/env python3
"""
Simple test script to try ViPS in interactive mode
"""

import sys
import os

# Add the current directory to Python path to import vips module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vips import WeightedLipReadingEvaluator

def interactive_test():
    """Interactive mode to test custom examples"""
    
    try:
        # Initialize the evaluator
        print("=== ViPS Interactive Test ===\n")
        print("Initializing ViPS evaluator...")
        evaluator = WeightedLipReadingEvaluator(use_weighted_distance=True, weight_method="both")
        print("✓ Evaluator initialized successfully!\n")
        
        print("Welcome to the ViPS interactive test mode!")
        print("You can enter reference and hypothesis text pairs to compare them.")
        print("Type 'quit' or 'exit' to end the session.\n")
        
        while True:
            print("\n" + "-" * 50)
            reference = input("Enter reference text (or 'quit' to exit): ").strip()
            
            if reference.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using ViPS! Goodbye!")
                break
                
            hypothesis = input("Enter hypothesis text: ").strip()
            
            if not reference or not hypothesis:
                print("Please enter both reference and hypothesis text.")
                continue
            
            print(f"\nAnalyzing:")
            print(f"  Reference:  '{reference}'")
            print(f"  Hypothesis: '{hypothesis}'")
            
            # Calculate scores
            results = evaluator.compare_standard_and_weighted(reference, hypothesis)
            
            # Calculate additional metrics for this single pair
            additional_metrics, _ = evaluator.calculate_additional_metrics([(reference, hypothesis)])
            
            # Display results
            standard_viseme_score = results['standard']['standard_viseme_score']
            standard_phoneme_score = results['standard']['standard_phoneme_score']
            vips_score = results['weighted']['vips_score']

            print("\n" + "=" * 40)
            print(" RESULTS")
            print("=" * 40)
            print(f"{'Metric':<25} {'Score':<10}")
            print(f"{'-'*35}")
            print(f"{'Standard Viseme Score':<25} {standard_viseme_score:<10.3f}")
            print(f"{'Standard Phoneme Score':<25} {standard_phoneme_score:<10.3f}")
            print(f"{'ViPS Score':<25} {vips_score:<10.3f}")
            
            # Show phoneme breakdown
            print(f"\n{'PHONEME BREAKDOWN'}")
            print(f"{'-'*35}")
            print(f"Reference:  {' '.join(results['ref_phonemes'])}")
            print(f"Hypothesis: {' '.join(results['hyp_phonemes'])}")
            
            # Display additional metrics if available
            if additional_metrics:
                print(f"\n{'ADDITIONAL METRICS'}")
                print(f"{'-'*35}")
                for metric_name, metric_value in additional_metrics.items():
                    print(f"{metric_name.replace('_', ' ').title():<25} {metric_value:<10.4f}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Run directly in interactive mode
    interactive_test()
