#!/usr/bin/env python3
"""
Simple test script to try ViPS on a single example
"""

import sys
import os

# Add the current directory to Python path to import vips module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vips import WeightedLipReadingEvaluator

def test_single_example():
    """Test the ViPS system on a single reference-hypothesis pair"""
    
    print("=== ViPS Single Example Test ===\n")
    
    try:
        # Initialize the evaluator
        print("Initializing ViPS evaluator...")
        evaluator = WeightedLipReadingEvaluator(use_weighted_distance=True, weight_method="both")
        print("✓ Evaluator initialized successfully!\n")
        
        # Define a test example
        reference = "hello world"
        hypothesis = "helo word"
        
        print(f"Testing example:")
        print(f"  Reference:  '{reference}'")
        print(f"  Hypothesis: '{hypothesis}'\n")
        
        # Get results using the comparison method
        print("Calculating scores...")
        results = evaluator.compare_standard_and_weighted(reference, hypothesis)
        
        # Extract the three key scores
        standard_viseme_score = results['standard']['standard_viseme_score']
        standard_phoneme_score = results['standard']['standard_phoneme_score']
        vips_score = results['weighted']['vips_score']
        
        # Calculate improvements
        vips_improvement = vips_score - standard_phoneme_score
        
        # Display results
        print("=== RESULTS ===")
        print(f"{'Metric':<25} {'Score':<10} {'Description'}")
        print(f"{'-'*25} {'-'*10} {'-'*30}")
        print(f"{'Standard Viseme Score':<25} {standard_viseme_score:<10.3f} Basic viseme similarity")
        print(f"{'Standard Phoneme Score':<25} {standard_phoneme_score:<10.3f} Basic phoneme similarity")
        print(f"{'ViPS Score':<25} {vips_score:<10.3f} Weighted phoneme similarity")
        print()
        print(f"ViPS Improvement: {vips_improvement:+.3f}")
        if vips_improvement > 0.001:
            print("✓ ViPS shows improvement over standard phoneme scoring!")
        elif vips_improvement < -0.001:
            print("⚠ ViPS shows lower score than standard phoneme scoring")
        else:
            print("= ViPS and standard phoneme scores are essentially equal")
        
        # Show phoneme breakdown
        print(f"\n=== PHONEME BREAKDOWN ===")
        print(f"Reference phonemes:  {' '.join(results['ref_phonemes'])}")
        print(f"Hypothesis phonemes: {' '.join(results['hyp_phonemes'])}")
        
        # Show distances
        print(f"\n=== EDIT DISTANCES ===")
        print(f"Viseme edit distance:  {results['standard']['viseme_edit_distance']:.3f}")
        print(f"Standard phonetic distance: {results['standard']['phonetic_edit_distance']:.3f}")
        print(f"Weighted phonetic distance: {results['weighted']['phonetic_edit_distance']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

def interactive_test():
    """Interactive mode to test custom examples"""
    
    try:
        # Initialize the evaluator
        print("Initializing ViPS evaluator...")
        evaluator = WeightedLipReadingEvaluator(use_weighted_distance=True, weight_method="both")
        print("✓ Evaluator initialized successfully!\n")
        
        while True:
            print("\n=== Interactive ViPS Test ===")
            reference = input("Enter reference text (or 'quit' to exit): ").strip()
            
            if reference.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            hypothesis = input("Enter hypothesis text: ").strip()
            
            if not reference or not hypothesis:
                print("Please enter both reference and hypothesis text.")
                continue
            
            print(f"\nTesting:")
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
            vips_improvement = vips_score - standard_phoneme_score

            print(f"\n{'Metric':<25} {'Score':<10}")
            print(f"{'-'*35}")
            print(f"{'Standard Viseme Score':<25} {standard_viseme_score:<10.3f}")
            print(f"{'Standard Phoneme Score':<25} {standard_phoneme_score:<10.3f}")
            print(f"{'ViPS Score':<25} {vips_score:<10.3f}")
            print(f"{'ViPS Improvement':<25} {vips_improvement:+<10.3f}")
            
            # Display additional metrics if available
            if additional_metrics:
                print(f"\n{'Additional Metrics':<25} {'Score':<10}")
                print(f"{'-'*35}")
                for metric_name, metric_value in additional_metrics.items():
                    print(f"{metric_name.replace('_', ' ').title():<25} {metric_value:<10.4f}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Run predefined example")
    print("2. Interactive mode")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        success = test_single_example()
        if success:
            print("\n✓ Test completed successfully!")
        else:
            print("\n❌ Test failed!")
    elif choice == "2":
        interactive_test()
    else:
        print("Invalid choice. Running predefined example...")
        test_single_example()
