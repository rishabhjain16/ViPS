#!/usr/bin/env python3
import os
import json
import numpy as np
import math
from collections import Counter, defaultdict
from itertools import chain
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Patch
from datetime import datetime
import panphon
import panphon.distance
import panphon.featuretable
from difflib import SequenceMatcher
import nltk
from nltk.translate import meteor_score
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
from sentence_transformers import SentenceTransformer
import bert_score
import csv
import re

class LipReadingEvaluator:
    """Comprehensive lip reading evaluation system with phonetic feature weighting"""
    
    # Class constants
    SILENCE_MARKERS = {'.', ' ', '-', ''}  # Special characters that represent silence
    SCORE_RANGE = (0.0, 1.0)  # Min and max scores for normalization
    UNKNOWN_VISEME = None  # Value for unknown visemes
    MAX_FEATURE_DISTANCE = None  # Will be calculated based on feature set
    
    # Feature value constants
    FEATURE_VALUES = [-1, 0, 1]  # Valid panphon feature values
    
    # Similarity calculation parameters
    SIMILARITY_TRANSFORM = 'quadratic'  # Options: 'linear', 'quadratic', 'cubic'
    
    def __init__(self):
        """Initialize the evaluator with phoneme mappings and converters"""
        
        try:
            # Import phonemizer and set up the backend
            from phonemizer import phonemize
            from phonemizer.separator import Separator
            from phonemizer.backend import EspeakBackend
            
            # Create backend instance
            self.phonemize = phonemize
            self.separator = Separator(word=' ', phone='')
            self.phonemizer_backend = EspeakBackend('en-us', with_stress=False)
        except ImportError:
            print("ERROR: phonemizer library is required for IPA conversion")
            print("Install with: pip install phonemizer")
            raise
        
        try:
            import panphon
            self.ft = panphon.featuretable.FeatureTable()
            self.dst = panphon.distance.Distance()
        except ImportError:
            print("ERROR: panphon library is required for phonetic features")
            print("Install with: pip install panphon")
            raise
        
        # Update regex pattern for text normalization
        import re
        self.alphanumeric_pattern = re.compile(r'[^a-zA-Z0-9\s]')
        
        # Define IPA phoneme-to-viseme mapping
        self.phoneme_to_viseme = {
            # 0: Silence (only empty string and space needed now that punctuation is handled separately)
            '': 0, ' ': 0,
            
            # 1: Open/central vowels (æ, ə, ʌ)
            'æ': 1, 'ə': 1, 'ʌ': 1, 'a': 1, 'ɐ': 1,
            'æː': 1, 'æ̃': 1,  # Add long and nasalized variants
            
            # 2: Open back vowels
            'ɑ': 2, 'ɒ': 2, 'ɑː': 2, 'ɑ̃': 2, 'ɒ̃': 2,  # Add nasalized variants
            
            # 3: Open-mid back rounded
            'ɔ': 3, 'ɔː': 3, 'ɔ̃': 3,  # Add nasalized variant
            
            # 4: Mid vowels
            'ɛ': 4, 'ʊ': 4, 'e': 4, 'ɜ': 4, 'ɜː': 4, 'ɛː': 4, 'eː': 4,
            'ɛ̃': 4, 'ẽ': 4, 'ʊ̃': 4,  # Add nasalized variants
            'ə̃': 4,  # Nasalized schwa
            
            # 5: R-colored vowels
            'ɝ': 5, 'ɚ': 5, 'ɹ̩': 5, 'ɻ': 5, 'r̩': 5,  # Add syllabic r
            'ɝː': 5, 'ɚː': 5,  # Long r-colored vowels
            
            # 6: Close front vowels + /j/
            'i': 6, 'ɪ': 6, 'j': 6, 'iː': 6, 'ɪː': 6, 'y': 6,
            'eɪ': 6, 'ej': 6, 'ɪə': 6, 'iə': 6,
            'ĩ': 6, 'ɪ̃': 6,  # Add nasalized variants
            'ɨ': 6, 'ᵻ': 6,  # Add barred i variants (sometimes used in unstressed positions)
            
            # 7: Close back rounded + /w/
            'u': 7, 'w': 7, 'uː': 7, 'ʍ': 7,
            'ũ': 7, 'w̥': 7,  # Add nasalized u and voiceless w
            
            # 8: Close-mid back rounded
            'o': 8, 'oː': 8, 'oʊ': 8, 'ow': 8, 'õ': 8,  # Add nasalized o
            'əʊ': 8,  # British English variant of /oʊ/
            
            # 9-11: Major diphthongs
            'aʊ': 9, 'aw': 9, 'aːʊ': 9, 'ãʊ̃': 9,  # Add long and nasalized variants
            'ɔɪ': 10, 'oj': 10, 'ɔːɪ': 10, 'ɔ̃ɪ̃': 10,  # Add long and nasalized variants
            'aɪ': 11, 'aj': 11, 'aːɪ': 11, 'ãɪ̃': 11,  # Add long and nasalized variants
            'eə': 4, 'ʊə': 4, 'ɛə': 4, 'ɪə': 4,  # Various centring diphthongs
            
            # 12: Glottal
            'h': 12, 'ʔ': 12, 'ɦ': 12, 'h̃': 12,  # Add breathy/voiced h
            
            # 13: Rhotic approximant
            'ɹ': 13, 'r': 13, 'ɾ': 13, 'ɻ': 13, 'ʀ': 13, 'ʁ': 13,
            'ɹ̥': 13, 'ɹ̩': 13, 'rː': 13,  # Add syllabic and voiceless variants
            
            # 14: Lateral approximant
            'l': 14, 'ɬ': 14, 'ɭ': 14, 'ɫ': 14, 'ʎ': 14, 'l̩': 14,
            'l̥': 14, 'lː': 14,  # Add voiceless and long variants
            
            # 15: Alveolar fricatives
            's': 15, 'z': 15, 'sː': 15, 'zː': 15,  # Add long variants
            
            # 16: Post-alveolar sounds
            'ʃ': 16, 'ʒ': 16, 'tʃ': 16, 'dʒ': 16, 'ʂ': 16, 'ʐ': 16, 'ɕ': 16, 'ʑ': 16,
            'sh': 16, 'zh': 16, 'ch': 16, 'ts': 16, 'dz': 16,  # Add additional affricates
            'ʃː': 16, 'ʒː': 16, 'tʃː': 16, 'dʒː': 16,  # Add long variants
            
            # 17: Voiced dental fricative
            'ð': 17, 'dh': 17, 'ðː': 17,  # Add long variant
            
            # 18: Labiodental fricatives
            'f': 18, 'v': 18, 'ɱ': 18, 'fː': 18, 'vː': 18,  # Add long variants
            
            # 19: Alveolar stops, nasal + voiceless dental
            't': 19, 'd': 19, 'n': 19, 'θ': 19, 'th': 19, 'ɗ': 19, 'n̩': 19,
            'ɳ': 19, 'ṭ': 19, 'ḍ': 19, 
            'tʰ': 19, 'tː': 19, 'dː': 19, 'nː': 19, 'θː': 19,  # Add aspirated and long variants
            'ɾ̃': 19,  # Nasalized flap
            
            # 20: Velar consonants
            'k': 20, 'g': 20, 'ŋ': 20, 'ɲ': 20, 'x': 20, 'ɣ': 20, 'q': 20, 'ɢ': 20,
            'ɡ': 20, 'ng': 20, 'ŋ̍': 20,
            'kʰ': 20, 'kː': 20, 'gː': 20, 'ŋː': 20,  # Add aspirated and long variants
            
            # 21: Bilabial consonants
            'p': 21, 'b': 21, 'm': 21, 'ɓ': 21, 'm̩': 21,
            'pʰ': 21, 'pː': 21, 'bː': 21, 'mː': 21,  # Add aspirated and long variants
            'ʙ': 21,  # Bilabial trill
        }
        
        # Add viseme ID to name mapping for better readability
        self.viseme_id_to_name = {
            0: "SILENCE",
            1: "VOWEL_CENTRAL",
            2: "VOWEL_OPEN_BACK",
            3: "VOWEL_OPEN_MID_BACK",
            4: "VOWEL_MID",
            5: "VOWEL_RHOTIC",
            6: "VOWEL_CLOSE_FRONT",
            7: "VOWEL_CLOSE_BACK",
            8: "VOWEL_MID_BACK",
            9: "DIPHTHONG_AW",
            10: "DIPHTHONG_OY",
            11: "DIPHTHONG_AY",
            12: "CONSONANT_GLOTTAL",
            13: "CONSONANT_RHOTIC",
            14: "CONSONANT_LATERAL",
            15: "CONSONANT_ALVEOLAR_FRICATIVE",
            16: "CONSONANT_POSTALVEOLAR",
            17: "CONSONANT_DENTAL_VOICED",
            18: "CONSONANT_LABIODENTAL",
            19: "CONSONANT_ALVEOLAR_DENTAL",
            20: "CONSONANT_VELAR",
            21: "CONSONANT_BILABIAL"
        }
        
        # Calculate maximum possible feature distance based on panphon features
        if self.MAX_FEATURE_DISTANCE is None:
            # Each feature can contribute a maximum of 1.0 to the distance
            self.MAX_FEATURE_DISTANCE = len(self.ft.names)
    
    def handle_diphthongs(self, phoneme):
        """
        Handle diphthongs by splitting them into components and combining their features.
        
        Args:
            phoneme: The diphthong phoneme
            
        Returns:
            dict: Combined feature dictionary following +>->0 priority
        """
        # Common diphthongs and their components
        diphthong_map = {
            'eɪ': ['e', 'ɪ'],
            'aɪ': ['a', 'ɪ'],
            'ɔɪ': ['ɔ', 'ɪ'],
            'aʊ': ['a', 'ʊ'],
            'oʊ': ['o', 'ʊ'],
            'ɪə': ['ɪ', 'ə'],
            'ʊə': ['ʊ', 'ə'],
            'eə': ['e', 'ə']
        }
        
        # If it's not in our map, check if it's a two-character phoneme
        if phoneme not in diphthong_map and len(phoneme) == 2:
            # Try to split it into component phonemes
            diphthong_map[phoneme] = [phoneme[0], phoneme[1]]
        
        # If we have a mapping for this diphthong
        if phoneme in diphthong_map:
            components = diphthong_map[phoneme]
            
            # Get features for each component
            feature_dicts = []
            for component in components:
                try:
                    feature_vector = self.ft.word_to_vector_list(component, numeric=True)
                    if feature_vector and len(feature_vector) > 0:
                        # Convert to dictionary with feature names
                        feature_dict = {}
                        for i, feature_name in enumerate(self.ft.names):
                            feature_dict[feature_name] = int(feature_vector[0][i])
                        feature_dicts.append(feature_dict)
                except Exception as e:
                    raise ValueError(f"Could not get features for diphthong component '{component}': {e}")
            
            # If we couldn't get features for all components, raise an error
            if len(feature_dicts) != len(components):
                raise ValueError(f"Could not get features for all components of diphthong '{phoneme}'")
            
            # Combine features using priority system: + > - > 0
            combined_features = {}
            for feature_name in self.ft.names:
                values = [fd.get(feature_name, 0) for fd in feature_dicts]
                # Apply priority: + (1) > - (-1) > 0
                if 1 in values:
                    combined_features[feature_name] = 1
                elif -1 in values:
                    combined_features[feature_name] = -1
                else:
                    combined_features[feature_name] = 0
            
            return combined_features
        
        # Not a diphthong we can handle
        raise ValueError(f"Not a recognized diphthong: '{phoneme}'")

    def get_phoneme_features(self, phoneme):
        """
        Get the phonetic features for a given phoneme using panphon.
        Handle diphthongs, special characters, and length markers.
        
        Args:
            phoneme: The phoneme to get features for
            
        Returns:
            dict: A dictionary of phonetic features
        """
        # Handle special characters
        if phoneme in ['.', ' ', '-'] or not phoneme.strip():
            return {'is_silence': 1}
        
        # Handle length markers separately first
        if 'ː' in phoneme:
            # Remove length marker and get features for the base phoneme
            base_phoneme = phoneme.replace('ː', '')
            return self.get_phoneme_features(base_phoneme)
            
        # Handle aspirated and other complex consonants
        if 'ʰ' in phoneme and len(phoneme) > 1:
            # Remove aspiration and get features for the base consonant
            base_phoneme = phoneme.replace('ʰ', '')
            return self.get_phoneme_features(base_phoneme)
        
        # Handle syllabic markers
        if '̩' in phoneme and len(phoneme) > 1:  # Syllabic diacritic
            # Remove syllabic marker and get features for the base consonant
            base_phoneme = phoneme.replace('̩', '')
            return self.get_phoneme_features(base_phoneme)
        
        # Mapping for problematic phonemes
        phoneme_mapping = {
            'ᵻ': 'ɪ',    # Map barred-i to near-close front unrounded vowel
            'ᵿ': 'ʊ',    # Map barred-u to near-close back rounded vowel
            'ɜ˞': 'ɜ',   # Map r-colored mid central vowel
            'ə˞': 'ə',   # Map r-colored schwa to regular schwa
            'ɚ': 'ə',    # Map r-colored schwa to regular schwa
            'ɝ': 'ɜ',    # Map r-colored mid central vowel to regular mid central
            'ɫ': 'l',     # Map velarized l to regular l
            'ʲ': 'j',     # Map palatalization to palatal approximant
            'ʙ': 'b',     # Map bilabial trill to voiced bilabial stop
            'ɓ': 'b',     # Map implosive b to regular b
            'g': 'ɡ',     # Map keyboard g to IPA script g (panphon requirement)
            'ɡː': 'ɡ',    # Map long IPA g to regular IPA g
            'gː': 'ɡ',    # Map long keyboard g to regular IPA g
            'ː': '',      # Remove length marker
            'ˈ': '',      # Remove primary stress
            'ˌ': ''       # Remove secondary stress
        }
        
        # Apply mapping if needed
        if phoneme in phoneme_mapping:
            phoneme = phoneme_mapping[phoneme]
        
        # Check if it's a multi-character phoneme (potential diphthong)
        if len(phoneme) > 1:
            try:
                # Try to handle as diphthong
                return self.handle_diphthongs(phoneme)
            except ValueError:
                # Not a diphthong we can handle, continue with standard processing
                pass
        
        # Standard processing with panphon
        try:
            feature_vector = self.ft.word_to_vector_list(phoneme, numeric=True)
            
            # If panphon returned a valid feature vector
            if feature_vector and len(feature_vector) > 0:
                # Convert to dictionary with feature names
                feature_dict = {}
                for i, feature_name in enumerate(self.ft.names):
                    feature_dict[feature_name] = int(feature_vector[0][i])
                
                return feature_dict
            else:
                raise ValueError(f"Panphon returned no features for '{phoneme}'")
                
        except Exception as e:
            raise ValueError(f"Error getting features for '{phoneme}': {e}")

    def normalize_text(self, text):
        """
        Normalize text by removing non-alphanumeric characters and extra whitespace.
        Replaces apostrophes with blank spaces and preserves spaces for silences.
        
        Args:
            text: Text to normalize
            
        Returns:
            str: Normalized text with non-alphanumeric characters removed
        """
        if not text:
            return ""
        
        # Replace apostrophes with space first
        text = text.replace("'", " ")
        
        # Remove all non-alphanumeric characters except spaces
        normalized = self.alphanumeric_pattern.sub('', text)
        
        # Replace multiple spaces with a single space and strip
        normalized = ' '.join(normalized.split())
        
        return normalized

    def calculate_phoneme_distance(self, phoneme1, phoneme2, use_weights=None):
        """
        Calculate distance between two individual phonemes, with option to use weights.
        This is the primary method for phoneme distance calculation.
        
        Args:
            phoneme1: First phoneme
            phoneme2: Second phoneme
            use_weights: Whether to use weighted features (defaults to self.use_weighted_distance)
            
        Returns:
            float: Distance value based on feature differences. Higher values indicate more different phonemes.
        """
        # If use_weights is not specified, use the class setting
        if use_weights is None:
            use_weights = getattr(self, 'use_weighted_distance', False)
                
        # If phonemes are identical, return 0 distance
        if phoneme1 == phoneme2:
            return 0.0
            
        # Special case: if either is empty or a silence marker
        if (not phoneme1.strip() or phoneme1 in self.SILENCE_MARKERS or
            not phoneme2.strip() or phoneme2 in self.SILENCE_MARKERS):
            return float('inf')  # Maximum possible distance for silence/empty
        
        # Get feature vectors for the phonemes
        feature_dict1 = self.get_phoneme_features(phoneme1)
        feature_dict2 = self.get_phoneme_features(phoneme2)
        
        # Use weighted or standard calculation based on parameter
        if use_weights and hasattr(self, 'feature_weights') and self.feature_weights:
            feature_diffs = []
            
            # Compare each feature with its weight
            for feature_name in self.ft.names:
                if feature_name in self.feature_weights:
                    # Get feature values, defaulting to 0 if missing
                    value1 = feature_dict1.get(feature_name, 0)
                    value2 = feature_dict2.get(feature_name, 0)
                    
                    # Calculate weighted difference for this feature
                    diff = abs(value1 - value2)
                    weight = self.feature_weights[feature_name]
                    feature_diffs.append(diff * weight)
            
            if feature_diffs:
                # Let the natural weighted average emerge
                return sum(feature_diffs) / len(feature_diffs)
            else:
                return float('inf')  # Maximum distance if no features compared
        else:
            # Use standard unweighted calculation
            feature_diffs = []
            
            for feature_name in self.ft.names:
                value1 = feature_dict1.get(feature_name, 0)
                value2 = feature_dict2.get(feature_name, 0)
                feature_diffs.append(abs(value1 - value2))
            
            if feature_diffs:
                return sum(feature_diffs) / len(feature_diffs)
            else:
                return float('inf')  # Maximum distance if no features compared

    def calculate_sequence_alignment(self, seq1, seq2, cost_function=None, allow_weighted=False):
        """
        Unified sequence alignment function for both phonemes and visemes with support for weighted costs.
        
        Parameters:
        - seq1: First sequence (reference)
        - seq2: Second sequence (hypothesis)
        - cost_function: Optional function to calculate substitution cost (default: binary cost)
        - allow_weighted: Whether to allow weighted costs based on similarity matrix (for visemes)
        
        Returns:
        - alignment: List of (operation, seq1_item, seq2_item) tuples
        - edit_distance: Edit distance score
        """
        # Initialize matrix
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill first row and column
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Calculate substitution cost
                if seq1[i-1] == seq2[j-1]:
                    # Exact match
                    cost = 0
                elif cost_function is not None:
                    # Use provided cost function
                    cost = cost_function(seq1[i-1], seq2[j-1])
                elif hasattr(self, 'use_weighted_distance') and self.use_weighted_distance and allow_weighted and hasattr(self, 'viseme_similarity_matrix'):
                    # Use weighted cost from similarity matrix for visemes
                    similarity = self.viseme_similarity_matrix.get((seq1[i-1], seq2[j-1]), 0)
                    cost = 1.0 - similarity
                else:
                    # Standard binary cost
                    cost = 1
                
                dp[i][j] = min(
                    dp[i-1][j] + 1,        # Deletion
                    dp[i][j-1] + 1,        # Insertion
                    dp[i-1][j-1] + cost    # Substitution
                )
        
        # Reconstruct alignment
        alignment = []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                # Calculate the cost that was used
                if seq1[i-1] == seq2[j-1]:
                    match_cost = 0
                elif cost_function is not None:
                    match_cost = cost_function(seq1[i-1], seq2[j-1])
                elif hasattr(self, 'use_weighted_distance') and self.use_weighted_distance and allow_weighted and hasattr(self, 'viseme_similarity_matrix'):
                    similarity = self.viseme_similarity_matrix.get((seq1[i-1], seq2[j-1]), 0)
                    match_cost = 1.0 - similarity
                else:
                    match_cost = 1
                
                if dp[i][j] == dp[i-1][j-1] + match_cost:
                    # Substitution or match
                    if seq1[i-1] == seq2[j-1]:
                        alignment.append(('match', seq1[i-1], seq2[j-1]))
                    else:
                        alignment.append(('substitute', seq1[i-1], seq2[j-1]))
                    i -= 1
                    j -= 1
                    continue
            
            if i > 0 and dp[i][j] == dp[i-1][j] + 1:
                # Deletion
                alignment.append(('delete', seq1[i-1], None))
                i -= 1
            else:
                # Insertion
                alignment.append(('insert', None, seq2[j-1]))
                j -= 1
        
        # Reverse alignment to get correct order
        alignment.reverse()
        
        return alignment, dp[m][n]

    # Primary interface for sequence comparison - uses calculate_sequence_alignment
    def calculate_sequence_distance(self, seq1, seq2, use_weights=None):
        """
        Calculate edit distance between two sequences (phonemes or visemes).
        This is the primary method for sequence comparison.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            use_weights: Whether to use weighted features (defaults to self.use_weighted_distance)
            
        Returns:
            tuple: (alignment, edit_distance)
        """
        # If use_weights is not specified, use the class setting
        if use_weights is None:
            use_weights = getattr(self, 'use_weighted_distance', False)
            
        # Define cost function based on whether we're using weights
        if use_weights:
            cost_function = lambda item1, item2: self.calculate_phoneme_distance(item1, item2, use_weights=True)
        else:
            cost_function = lambda item1, item2: self.calculate_phoneme_distance(item1, item2, use_weights=False)
        
        # Use the unified alignment function with the appropriate cost function
        return self.calculate_sequence_alignment(seq1, seq2, cost_function=cost_function)
    
    # Convenience method for viseme alignment
    def calculate_viseme_alignment(self, seq1, seq2):
        """
        Calculate alignment between two viseme sequences.
        A convenience method that uses calculate_sequence_alignment with viseme-specific settings.
        
        Returns:
        - alignment: List of (operation, seq1_item, seq2_item) tuples
        - edit_distance: Edit distance score
        """
        # Use the unified alignment function with weighted option for visemes
        return self.calculate_sequence_alignment(seq1, seq2, allow_weighted=True)
        
    # ==== LEGACY METHODS (MAINTAINED FOR BACKWARD COMPATIBILITY) ====
    
    def calculate_phonetic_distance(self, phoneme1, phoneme2):
        """
        Legacy method that calls calculate_phoneme_distance with current weighting.
        Maintained for backward compatibility.
        """
        return self.calculate_phoneme_distance(phoneme1, phoneme2)
        
    def calculate_weighted_phonetic_edit_distance(self, phoneme_seq1, phoneme_seq2):
        """
        Legacy method that calls calculate_sequence_distance with weighted=True.
        Maintained for backward compatibility.
        """
        _, edit_distance = self.calculate_sequence_distance(phoneme_seq1, phoneme_seq2, use_weights=True)
        return edit_distance
        
    def calculate_phonetic_alignment(self, seq1, seq2):
        """
        Legacy method that calls calculate_sequence_distance with weighted=False.
        Maintained for backward compatibility.
        """
        return self.calculate_sequence_distance(seq1, seq2, use_weights=False)
        
    def standard_phonetic_distance(self, phoneme1, phoneme2):
        """
        Calculate phonetic distance using the standard (non-weighted) approach.
        This ensures we have a clear distinction between standard and weighted distances.
        
        Args:
            phoneme1: First phoneme
            phoneme2: Second phoneme
            
        Returns:
            float: Distance value between 0.0 (identical) and 1.0 (maximally different)
        """
        # Call the main distance function with use_weights=False
        return self.calculate_phoneme_distance(phoneme1, phoneme2, use_weights=False)

    def text_to_phonemes(self, text):
        """Convert English text to phoneme sequence"""
        # Normalize text first to remove punctuation
        normalized_text = self.normalize_text(text)
        
        # If empty after normalization, return empty list
        if not normalized_text:
            return []
            
        processed_phonemes = []
        
        # Get raw IPA string and split by words
        raw_phonemes = self.phonemizer_backend.phonemize(
            [normalized_text], 
            separator=self.separator
        )[0]
        raw_words = raw_phonemes.split()
        
        # Process word by word
        for word_idx, word in enumerate(raw_words):
            # Add space between words (except before the first word)
            if word_idx > 0:
                processed_phonemes.append(' ')
                
            # Skip empty words
            if not word.strip():
                continue
            
            # Segment the IPA string into individual phonemes
            i = 0
            while i < len(word):
                # Try to match phonemes in mapping (longer sequences first)
                matched = False
                for phoneme_len in [3, 2, 1]:
                    if i + phoneme_len <= len(word):
                        candidate = word[i:i+phoneme_len]
                        if candidate in self.phoneme_to_viseme:
                            processed_phonemes.append(candidate)
                            i += phoneme_len
                            matched = True
                            break
                
                if not matched:
                    # If no match found, use individual character
                    processed_phonemes.append(word[i])
                    i += 1
        
        return processed_phonemes
    
    def evaluate_pair(self, reference, hypothesis):
        """
        Evaluate a single reference-hypothesis pair for phonetic and viseme similarity
        
        Parameters:
        - reference: Reference text (ground truth)
        - hypothesis: Hypothesis text (predicted)
        
        Returns:
        - Dictionary with evaluation results
        """
        # Normalize texts first
        normalized_reference = self.normalize_text(reference)
        normalized_hypothesis = self.normalize_text(hypothesis)
        
        # Convert texts to phoneme sequences
        try:
            ref_phonemes = self.text_to_phonemes(normalized_reference)
            hyp_phonemes = self.text_to_phonemes(normalized_hypothesis)        
            
            # Calculate phonetic alignment and edit distance
            alignment, edit_distance = self.calculate_phonetic_alignment(ref_phonemes, hyp_phonemes)
            
            # Convert phonemes to visemes using most appropriate mapping
            ref_visemes = [self.map_phoneme_to_viseme(p) for p in ref_phonemes]
            hyp_visemes = [self.map_phoneme_to_viseme(p) for p in hyp_phonemes]
            
            # Calculate viseme-level alignment and score
            viseme_alignment, viseme_edit_distance = self.calculate_viseme_alignment(ref_visemes, hyp_visemes)
            
            # Normalize viseme score (lower is better, convert to higher is better)
            max_len = max(len(ref_visemes), len(hyp_visemes))
            if max_len > 0:
                viseme_score = 1.0 - (viseme_edit_distance / max_len)
            else:
                viseme_score = 1.0
            
            # Store original texts and results
            results = {
                'reference': reference,
                'hypothesis': hypothesis,
                'ref_phonemes': ref_phonemes,
                'hyp_phonemes': hyp_phonemes,
                'phonetic_alignment': alignment,
                'phonetic_edit_distance': edit_distance,
                'ref_visemes': ref_visemes,
                'hyp_visemes': hyp_visemes,
                'viseme_alignment': viseme_alignment,
                'viseme_edit_distance': viseme_edit_distance,
                'viseme_score': viseme_score,
            }
            
            return results
            
        except Exception as e:
            print(f"  ERROR in evaluate_pair: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return a minimal result to avoid crashing
            return {
                'reference': reference,
                'hypothesis': hypothesis,
                'error': str(e),
                'viseme_score': 0.0,
                'phonetic_edit_distance': float('inf'),
            }

    def map_phoneme_to_viseme(self, phoneme, default_value=None):
        """
        Map a phoneme to its corresponding viseme using direct mapping.
        
        Args:
            phoneme: The phoneme to map
            default_value: Value to return if phoneme isn't found
            
        Returns:
            int: The viseme ID (0-21) that corresponds to this phoneme,
                 or default_value if not found
        """
        # Try direct lookup first
        if phoneme in self.phoneme_to_viseme:
            return self.phoneme_to_viseme[phoneme]
        
        # Empty string or silence marker is treated as silence
        if not phoneme.strip() or phoneme in self.SILENCE_MARKERS:
            return 0  # silence
        
        # Return default value if phoneme isn't recognized
        return default_value

    def _calculate_similarity_from_distance(self, distance):
        """Convert distance to similarity using configured transformation"""
        if self.SIMILARITY_TRANSFORM == 'linear':
            return 1.0 - distance
        elif self.SIMILARITY_TRANSFORM == 'quadratic':
            return 1.0 - (distance ** 2)
        elif self.SIMILARITY_TRANSFORM == 'cubic':
            return 1.0 - (distance ** 3)
        else:
            raise ValueError(f"Unknown similarity transform: {self.SIMILARITY_TRANSFORM}")

# Enhanced lip reading evaluator with weighted measurements
class WeightedLipReadingEvaluator(LipReadingEvaluator):
    """
    Enhanced lip reading evaluator that uses phonetic feature weighting
    for phoneme and viseme similarity calculations
    """
    
    # Weight calculation parameters
    WEIGHT_METHODS = {
        "both": lambda e, d: e * d,  # Multiply entropy and distinctiveness
        "entropy": lambda e, d: e,    # Use only entropy
        "visual": lambda e, d: d      # Use only visual distinctiveness
    }
    
    def __init__(self, use_weighted_distance=True, weight_method="both"):
        """Initialize weighted evaluator with specified method"""
        super().__init__()
        self.use_weighted_distance = use_weighted_distance
        if weight_method not in self.WEIGHT_METHODS:
            raise ValueError(f"Invalid weight method: {weight_method}. Must be one of {list(self.WEIGHT_METHODS.keys())}")
        self.weight_method = weight_method
        self.feature_weights = {}
        
        # Initialize viseme similarity matrix
        self.viseme_similarity_matrix = {}
        
        # Calculate weights if enabled
        if use_weighted_distance:
            # Debug: Before weight calculation
            print("[Debug] Before weight calculation")
            
            # Calculate weights from scratch
            calculated_weights = self.calculate_phonetic_feature_weights()
            
            # Debug: After weight calculation
            print(f"[Debug] After weight calculation. Got {len(calculated_weights) if calculated_weights else 0} weights")
            
            # Store the weights
            self.feature_weights = calculated_weights
            
            # Debug: After assignment
            print(f"[Debug] After assignment. feature_weights has {len(self.feature_weights)} entries")
            
            # Calculate viseme similarity matrix
            self.viseme_similarity_matrix = self.calculate_viseme_similarity_matrix()
    
    def save_weights_to_file(self, file_path):
        """
        Save computed weights and similarity matrix to file for later use
        
        Parameters:
        - file_path: Path to save JSON file with weights
        """
        try:
            # Convert tuple keys to strings for JSON serialization
            serializable_matrix = {
                f"{k[0]},{k[1]}": v 
                for k, v in self.viseme_similarity_matrix.items()
            }
            
            data = {
                'feature_weights': self.feature_weights,
                'viseme_similarity_matrix': serializable_matrix
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"Saved weights and similarity matrix to {file_path}")
        except Exception as e:
            print(f"Error saving weights to file: {e}")

    def calculate_phonetic_feature_weights(self):
        """Calculate feature weights based on phonetic importance"""
        print(f"Calculating phonetic feature weights using method: {self.weight_method}")
        
        # Get all features from panphon
        all_features = self.ft.names
        self.feature_entropies = {}
        self.visual_distinctiveness = {}
        self.feature_weights = {}
        
        # Generate phoneme inventory
        phoneme_inventory = [p for p in self.phoneme_to_viseme.keys() 
                           if p not in self.SILENCE_MARKERS]
        
        print(f"Using {len(phoneme_inventory)} phonemes from IPA inventory for weight calculation")
        
        # Calculate entropy for each feature
        for feature_idx, feature in enumerate(all_features):
            # Get all values this feature takes across the phoneme inventory
            feature_values = []
            for phoneme in phoneme_inventory:
                try:
                    # Get feature value from panphon directly
                    fv = self.ft.word_to_vector_list(phoneme, numeric=True)
                        
                    if fv and len(fv) > 0:
                        feature_values.append(fv[0][feature_idx])
                except Exception:
                    continue
            
            if not feature_values:
                continue  # Skip if no values available
                
            # Count feature values (should be in FEATURE_VALUES)
            value_counts = Counter(feature_values)
            
            # Calculate entropy
            total = len(feature_values)
            probabilities = [count/total for count in value_counts.values()]
            
            # Using numpy for stability in log calculations
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            
            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(self.FEATURE_VALUES))  # Use defined feature values
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            self.feature_entropies[feature] = normalized_entropy
        
        # Calculate visual distinctiveness for each feature
        total_viseme_classes = len(self.viseme_id_to_name)  # Get from actual viseme class count
        
        for feature_idx, feature in enumerate(all_features):
            # Group phonemes by feature value and viseme class
            feature_value_to_visemes = {}
            
            for phoneme in phoneme_inventory:
                try:
                    # Get feature value from panphon directly
                    fv = self.ft.word_to_vector_list(phoneme, numeric=True)
                    if not fv or len(fv) == 0:
                        continue
                        
                    feature_value = fv[0][feature_idx]
                    viseme = self.map_phoneme_to_viseme(phoneme, self.UNKNOWN_VISEME)
                    
                    if feature_value not in feature_value_to_visemes:
                        feature_value_to_visemes[feature_value] = set()
                    
                    feature_value_to_visemes[feature_value].add(viseme)
                except Exception:
                    continue
            
            # Calculate average visemes per feature value
            total_distinctions = sum(len(viseme_set) for viseme_set in feature_value_to_visemes.values())
            total_values = len(feature_value_to_visemes)
            
            if total_values == 0:
                average_visemes_per_value = 0
            else:
                average_visemes_per_value = total_distinctions / total_values
            
            # More distinctive = less visemes per feature value
            distinctiveness = 1.0 - (average_visemes_per_value / total_viseme_classes)
            # Let natural distinctiveness emerge without artificial minimum
            
            self.visual_distinctiveness[feature] = distinctiveness
        
        # Combine entropy and visual distinctiveness based on selected method
        for feature in all_features:
            if feature in self.feature_entropies and feature in self.visual_distinctiveness:
                # Calculate weight using selected method
                self.feature_weights[feature] = self.WEIGHT_METHODS[self.weight_method](
                    self.feature_entropies[feature],
                    self.visual_distinctiveness[feature]
                )
        
        print("Feature weights calculated successfully")
        
        return self.feature_weights
    
    def save_feature_weights_table(self, file_path):
        """
        Save the feature weights table (entropy, visual distinctiveness, weights) to a text file in a nicely formatted table.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            with open(file_path, 'w') as f:
                # Write header
                f.write("Feature Weights Analysis Table\n")
                f.write("=" * 65 + "\n\n")
                f.write("Weight calculation method: {}\n".format(self.weight_method))
                f.write("Note: Weights are calculated as the product of entropy and visual distinctiveness.\n\n")
                
                # Table header
                f.write("{:<12} | {:<10} | {:<22} | {:<12}\n".format(
                    "Feature", "Entropy", "Visual Distinctiveness", "Weight"))
                f.write("-" * 12 + "-+-" + "-" * 10 + "-+-" + "-" * 22 + "-+-" + "-" * 12 + "\n")
                
                # Sort features by weight for better readability
                sorted_features = sorted(
                    self.feature_weights.keys(),
                    key=lambda f: self.feature_weights[f],
                    reverse=True
                )
                
                # Write each feature's data
                for feature in sorted_features:
                    entropy = self.feature_entropies[feature]
                    distinctiveness = self.visual_distinctiveness[feature]
                    weight = self.feature_weights[feature]
                    
                    f.write("{:<12} | {:<10.3f} | {:<22.3f} | {:<12.3f}\n".format(
                        feature, entropy, distinctiveness, weight))
                
                # Write summary statistics
                f.write("\nSummary Statistics\n")
                f.write("-" * 20 + "\n")
                f.write("Average Entropy: {:.3f}\n".format(
                    sum(self.feature_entropies.values()) / len(self.feature_entropies)))
                f.write("Average Visual Distinctiveness: {:.3f}\n".format(
                    sum(self.visual_distinctiveness.values()) / len(self.visual_distinctiveness)))
                f.write("Average Weight: {:.3f}\n".format(
                    sum(self.feature_weights.values()) / len(self.feature_weights)))
                
            print(f"Saved feature weights table to {file_path}")
        except Exception as e:
            print(f"Error saving feature weights table: {e}")
    
    def calculate_viseme_similarity_matrix(self):
        """
        Calculate similarity between viseme pairs based on their phonetic features.
        Similarity is based directly on feature space distances without normalization.
        
        Returns:
            dict: Mapping of (viseme1, viseme2) tuples to similarity values
        """
        print("Calculating viseme similarity matrix...")
        similarity_matrix = {}
        
        # Get all valid viseme classes (excluding silence markers)
        viseme_classes = sorted(set(
            self.map_phoneme_to_viseme(p) 
            for p in self.phoneme_to_viseme.keys() 
            if p not in self.SILENCE_MARKERS and p
        ))
        print(f"Processing {len(viseme_classes)} viseme classes")
        
        # Get all phonemes for each viseme class
        viseme_to_phonemes = {}
        for viseme in viseme_classes:
            viseme_to_phonemes[viseme] = [
                p for p in self.phoneme_to_viseme.keys()
                if self.map_phoneme_to_viseme(p) == viseme and p not in self.SILENCE_MARKERS
            ]
        
        # Calculate similarity between each pair of viseme classes
        for viseme1 in viseme_classes:
            for viseme2 in viseme_classes:
                # For same viseme class, use the internal similarity of its phonemes
                if viseme1 == viseme2:
                    phonemes = viseme_to_phonemes[viseme1]
                    if len(phonemes) <= 1:
                        # Single phoneme or empty viseme class
                        similarity_matrix[(viseme1, viseme2)] = 1.0
                    else:
                        # Calculate average distance between all phonemes in this viseme
                        distances = []
                        for i, p1 in enumerate(phonemes):
                            for p2 in phonemes[i+1:]:  # Compare each pair once
                                try:
                                    distance = self.calculate_phoneme_distance(p1, p2, use_weights=True)
                                    if distance != float('inf'):
                                        distances.append(distance)
                                except Exception as e:
                                    print(f"Error comparing '{p1}' and '{p2}': {e}")
                        
                        if distances:
                            # Use exponential decay to convert distance to similarity
                            # exp(-x) gives 1.0 for x=0 and approaches 0 as x increases
                            avg_distance = sum(distances) / len(distances)
                            similarity_matrix[(viseme1, viseme2)] = math.exp(-avg_distance)
                        else:
                            similarity_matrix[(viseme1, viseme2)] = 1.0
                    continue
                
                # Skip if already calculated (symmetrical)
                if (viseme1, viseme2) in similarity_matrix or (viseme2, viseme1) in similarity_matrix:
                    continue
                
                phonemes1 = viseme_to_phonemes.get(viseme1, [])
                phonemes2 = viseme_to_phonemes.get(viseme2, [])
                
                if not phonemes1 or not phonemes2:
                    # No phonemes to compare
                    similarity = 0.0
                else:
                    # Calculate distances between all phoneme pairs
                    distances = []
                    for p1 in phonemes1:
                        for p2 in phonemes2:
                            try:
                                distance = self.calculate_phoneme_distance(p1, p2, use_weights=True)
                                if distance != float('inf'):
                                    distances.append(distance)
                            except Exception as e:
                                print(f"Error comparing '{p1}' and '{p2}': {e}")
                    
                    if distances:
                        # Convert average distance to similarity using exponential decay
                        avg_distance = sum(distances) / len(distances)
                        similarity = math.exp(-avg_distance)
                    else:
                        similarity = 0.0
                
                # Store in matrix (symmetrical)
                similarity_matrix[(viseme1, viseme2)] = similarity
                similarity_matrix[(viseme2, viseme1)] = similarity
        
        self.viseme_similarity_matrix = similarity_matrix
        print("Viseme similarity matrix calculated successfully")
        
        return similarity_matrix
    
    def calculate_phonetic_distance(self, phoneme1, phoneme2):
        """
        Override parent method to use weighted phonetic distance calculation.
        Maintained for backward compatibility.
        
        Args:
            phoneme1: First phoneme
            phoneme2: Second phoneme
            
        Returns:
            float: Weighted distance value between 0.0 (identical) and 1.0 (maximally different)
        """
        # Use the main calculate_phoneme_distance method with current weighting
        return self.calculate_phoneme_distance(phoneme1, phoneme2)
    
    def calculate_weighted_phonetic_edit_distance(self, phoneme_seq1, phoneme_seq2):
        """
        Calculate phonetic edit distance between two phoneme sequences using weighted distance.
        
        Args:
            phoneme_seq1: First phoneme sequence
            phoneme_seq2: Second phoneme sequence
            
        Returns:
            float: Weighted edit distance value
        """
        # Define cost function that uses our class's weighted distance calculation
        def phonetic_weighted_cost(p1, p2):
            return self.calculate_phoneme_distance(p1, p2, use_weights=True)
        
        # Use the unified alignment function with our weighted phonetic distance
        _, edit_distance = self.calculate_sequence_alignment(
            phoneme_seq1, phoneme_seq2, 
            cost_function=phonetic_weighted_cost
        )
        
        return edit_distance
    
    def evaluate_pair(self, reference, hypothesis):
        """
        Override parent method to evaluate a reference-hypothesis pair using weighted distance
        
        Args:
            reference: Reference text (ground truth)
            hypothesis: Hypothesis text (predicted)
            
        Returns:
            dict: Evaluation results
        """
        # Normalize texts first
        normalized_reference = self.normalize_text(reference)
        normalized_hypothesis = self.normalize_text(hypothesis)
        
        # Convert text to phonemes using normalized text
        ref_phonemes = self.text_to_phonemes(normalized_reference)
        hyp_phonemes = self.text_to_phonemes(normalized_hypothesis)
        
        # Convert phonemes to visemes
        ref_visemes = [self.map_phoneme_to_viseme(p) for p in ref_phonemes]
        hyp_visemes = [self.map_phoneme_to_viseme(p) for p in hyp_phonemes]
        
        # Calculate phonetic distances using appropriate method based on weighting setting
        if self.use_weighted_distance:
            # For weighted approach, use weighted sequence distance
            _, phonetic_distance = self.calculate_sequence_distance(ref_phonemes, hyp_phonemes, use_weights=True)
            phonetic_alignment = None  # We don't need the alignment for the weighted version
        else:
            # For standard approach, use standard sequence distance
            phonetic_alignment, phonetic_distance = self.calculate_sequence_distance(ref_phonemes, hyp_phonemes, use_weights=False)
        
        # Calculate viseme alignment
        alignment, edit_distance = self.calculate_viseme_alignment(ref_visemes, hyp_visemes)
        
        # Calculate normalized scores (0-1, higher is better)
        max_viseme_len = max(len(ref_visemes), len(hyp_visemes))
        max_phoneme_len = max(len(ref_phonemes), len(hyp_phonemes))
        
        # Standard viseme score (always calculated the same way)
        viseme_score = 1.0 - (edit_distance / max_viseme_len if max_viseme_len > 0 else 0.0)
        
        # ViPS score (weighted or standard phonetic alignment score)
        vips_score = 1.0 - (phonetic_distance / max_phoneme_len if max_phoneme_len > 0 else 0.0)
        
        # Return simplified results
        results = {
            'ref_phonemes': ref_phonemes,
            'hyp_phonemes': hyp_phonemes,
            'ref_visemes': ref_visemes,
            'hyp_visemes': hyp_visemes,
            'phonetic_edit_distance': phonetic_distance,
            'viseme_edit_distance': edit_distance,
            'viseme_score': viseme_score,
            'vips_score': vips_score
        }
        
        return results
    
    def compare_standard_and_weighted(self, reference, hypothesis):
        """
        Evaluate a single reference-hypothesis pair with detailed metrics using both
        standard and weighted approaches for comparison.
        
        Parameters:
        - reference: Reference text (ground truth)
        - hypothesis: Hypothesis text (predicted)
        
        Returns:
        - Dictionary with evaluation results for both approaches
        """
        # Store original weighted distance setting (but preserve IPA setting)
        original_setting = self.use_weighted_distance
        
        # Normalize texts for consistency
        normalized_reference = self.normalize_text(reference)
        normalized_hypothesis = self.normalize_text(hypothesis)
        
        try:
            # Evaluate with standard distance (no weights)
            self.use_weighted_distance = False
            standard_results = self.evaluate_pair(normalized_reference, normalized_hypothesis)
            
            # Evaluate with weighted distance
            self.use_weighted_distance = True
            weighted_results = self.evaluate_pair(normalized_reference, normalized_hypothesis)
            
            # Restore original setting
            self.use_weighted_distance = original_setting
            
            # Combine results
            combined_results = {
                'reference': reference,
                'hypothesis': hypothesis,
                'ref_phonemes': standard_results.get('ref_phonemes', []),
                'hyp_phonemes': standard_results.get('hyp_phonemes', []),
                'standard': {
                    'phonetic_edit_distance': standard_results.get('phonetic_edit_distance', float('inf')),
                    'standard_phoneme_score': standard_results.get('vips_score', 0.0),
                    'viseme_edit_distance': standard_results.get('viseme_edit_distance', float('inf')),
                    'standard_viseme_score': standard_results.get('viseme_score', 0.0),
                },
                'weighted': {
                    'phonetic_edit_distance': weighted_results.get('phonetic_edit_distance', float('inf')),
                    'vips_score': weighted_results.get('vips_score', 0.0),
                    'viseme_edit_distance': weighted_results.get('viseme_edit_distance', float('inf')),
                    'viseme_score': weighted_results.get('viseme_score', 0.0),
                }
            }
            
            return combined_results
            
        except Exception as e:
            print(f"ERROR in compare_standard_and_weighted: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Restore original setting
            self.use_weighted_distance = original_setting
            
            # Return a minimal result
            return {
                'reference': reference,
                'hypothesis': hypothesis,
                'error': str(e),
                'standard': {'standard_viseme_score': 0.0, 'standard_phoneme_score': 0.0},
                'weighted': {'viseme_score': 0.0, 'vips_score': 0.0}
            }
    
    def calculate_additional_metrics(self, example_pairs):
        """
        Calculate additional metrics for a list of reference-hypothesis pairs
        
        Parameters:
        - example_pairs: List of (reference, hypothesis) tuples
        
        Returns:
        - Dictionary with additional metrics (WER, CER, etc.)
        - List of per-example metrics
        """
        # Import numpy at the beginning to ensure it's available for all calculations
        import numpy as np
        
        # Check if we have valid input
        if not example_pairs or not isinstance(example_pairs, list):
            print("Warning: No valid example pairs provided for metric calculation")
            return {}, []
            
        # Extract references and hypotheses
        references = [ref for ref, _ in example_pairs]
        hypotheses = [hyp for _, hyp in example_pairs]
        
        metrics = {}
        per_example_metrics = []
        
        # Initialize per-example metric storage
        example_metrics_data = [{'reference': ref, 'hypothesis': hyp} for ref, hyp in example_pairs]
                
        # Character Error Rate (CER)
        cer_values = []
        try:
            import nltk
            
            # Calculate per-example CER
            for ref, hyp in example_pairs:
                if len(ref) > 0:
                    char_error = nltk.edit_distance(ref, hyp)
                    cer = char_error / len(ref)
                    cer_values.append(cer)
                else:
                    cer_values.append(1.0 if hyp else 0.0)
            
            # Calculate corpus-level CER
            total_chars = sum(len(ref) for ref in references)
            char_errors = sum(nltk.edit_distance(ref, hyp) for ref, hyp in example_pairs)
            cer = char_errors / total_chars if total_chars > 0 else 0
            metrics['character_error_rate'] = cer
            
            # Add to per-example metrics
            for i, metrics_dict in enumerate(example_metrics_data):
                metrics_dict['cer'] = cer_values[i]
                
        except Exception as e:
            print(f"Warning: Could not calculate CER: {e}")
            
        # Word Error Rate (WER)
        wer_values = []
        try:
            import nltk
            
            # Calculate per-example WER
            for ref, hyp in example_pairs:
                ref_words = ref.split()
                hyp_words = hyp.split()
                if ref_words:
                    word_error = nltk.edit_distance(ref_words, hyp_words)
                    wer = word_error / len(ref_words)
                    wer_values.append(wer)
                else:
                    wer_values.append(1.0 if hyp_words else 0.0)
            
            # Calculate corpus-level WER
            ref_words = [ref.split() for ref in references]
            hyp_words = [hyp.split() for hyp in hypotheses]
            total_words = sum(len(words) for words in ref_words)
            word_errors = sum(nltk.edit_distance(ref, hyp) for ref, hyp in zip(ref_words, hyp_words))
            wer = word_errors / total_words if total_words > 0 else 0
            metrics['word_error_rate'] = wer
            
            # Add to per-example metrics
            for i, metrics_dict in enumerate(example_metrics_data):
                if i < len(wer_values):
                    metrics_dict['wer'] = wer_values[i]
                
        except Exception as e:
            print(f"Warning: Could not calculate WER: {e}")
        
        # Word similarity using SequenceMatcher
        word_similarities = []
        try:
            from difflib import SequenceMatcher
            
            # Calculate per-example word similarity
            for ref, hyp in example_pairs:
                similarity = SequenceMatcher(None, ref, hyp).ratio()
                word_similarities.append(similarity)
            
            # Calculate average word similarity
            metrics['word_similarity'] = np.mean(word_similarities)
            
            # Add to per-example metrics
            for i, metrics_dict in enumerate(example_metrics_data):
                if i < len(word_similarities):
                    metrics_dict['word_similarity'] = word_similarities[i]
                
        except Exception as e:
            print(f"Warning: Could not calculate word similarity: {e}")
        
        # METEOR score
        meteor_scores = []
        try:
            from nltk.translate import meteor_score
            
            # Ensure necessary NLTK data is available
            try:
                import nltk
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)
            
            # Calculate per-example METEOR scores
            for ref, hyp in example_pairs:
                ref_tokens = [ref.split()]  # METEOR expects a list of reference tokenized sentences
                hyp_tokens = hyp.split()
                score = meteor_score.meteor_score(ref_tokens, hyp_tokens)
                meteor_scores.append(score)
            
            # Calculate average METEOR score
            metrics['meteor_score'] = np.mean(meteor_scores)
            
            # Add to per-example metrics
            for i, metrics_dict in enumerate(example_metrics_data):
                if i < len(meteor_scores):
                    metrics_dict['meteor_score'] = meteor_scores[i]
                
        except Exception as e:
            print(f"Warning: Could not calculate METEOR score: {e}")
        
        # ROUGE scores
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        try:
            from rouge_score import rouge_scorer
            
            # Initialize ROUGE scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            # Calculate per-example ROUGE scores
            for ref, hyp in example_pairs:
                scores = scorer.score(ref, hyp)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            
            # Calculate average ROUGE scores
            metrics['rouge1_score'] = np.mean(rouge1_scores)
            metrics['rouge2_score'] = np.mean(rouge2_scores)
            metrics['rougeL_score'] = np.mean(rougeL_scores)
            
            # Add to per-example metrics
            for i, metrics_dict in enumerate(example_metrics_data):
                if i < len(rouge1_scores):
                    metrics_dict['rouge1_score'] = rouge1_scores[i]
                    metrics_dict['rouge2_score'] = rouge2_scores[i]
                    metrics_dict['rougeL_score'] = rougeL_scores[i]
                
        except Exception as e:
            print(f"Warning: Could not calculate ROUGE scores: {e}")
        
        # BLEU scores
        sentence_bleu_scores = []
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            # Get smoothing function for BLEU
            smoothie = SmoothingFunction().method1
            
            # Calculate per-example sentence BLEU scores
            for ref, hyp in example_pairs:
                ref_tokens = [ref.split()]  # BLEU expects a list of reference tokenized sentences
                hyp_tokens = hyp.split()
                
                # Skip empty sequences
                if not hyp_tokens or not ref_tokens[0]:
                    sentence_bleu_scores.append(0.0)
                    continue
                
                # Use smoothing function to handle cases with no n-gram overlaps
                score = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothie)
                sentence_bleu_scores.append(score)
            
            # Calculate average sentence BLEU score
            metrics['sentence_bleu_score'] = np.mean(sentence_bleu_scores) if sentence_bleu_scores else 0
            
            # Add to per-example metrics
            for i, metrics_dict in enumerate(example_metrics_data):
                if i < len(sentence_bleu_scores):
                    metrics_dict['sentence_bleu_score'] = sentence_bleu_scores[i]
                
        except Exception as e:
            print(f"Warning: Could not calculate BLEU scores: {e}")
        
        # BERTScore
        bertscore_p = []
        bertscore_r = []
        bertscore_f1 = []
        try:
            import torch
            import bert_score
            
            # Check if we have valid inputs
            if hypotheses and references:
                # Calculate BERTScores
                P, R, F1 = bert_score.score(hypotheses, references, lang="en", verbose=False)
                
                # Convert to Python floats
                bertscore_p = [p.item() for p in P]
                bertscore_r = [r.item() for r in R]
                bertscore_f1 = [f1.item() for f1 in F1]
                
                # Calculate average BERTScores
                metrics['bertscore_precision'] = float(torch.mean(P).item())
                metrics['bertscore_recall'] = float(torch.mean(R).item())
                metrics['bertscore_f1'] = float(torch.mean(F1).item())
                
                # Add to per-example metrics
                for i, metrics_dict in enumerate(example_metrics_data):
                    if i < len(bertscore_p):
                        metrics_dict['bertscore_precision'] = bertscore_p[i]
                        metrics_dict['bertscore_recall'] = bertscore_r[i]
                        metrics_dict['bertscore_f1'] = bertscore_f1[i]
        except Exception as e:
            print(f"Warning: Could not calculate BERTScore: {e}")
        
        # Semantic similarity
        semantic_similarities = []
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Check for valid inputs
            if not references or not hypotheses:
                raise ValueError("Empty references or hypotheses lists")
                
            # Use a suitable sentence transformer model
            model_name = 'all-MiniLM-L6-v2'  # Consider making this configurable
            semantic_model = SentenceTransformer(model_name)
            
            # Encode all texts to get embeddings
            ref_embeddings = semantic_model.encode(references)
            hyp_embeddings = semantic_model.encode(hypotheses)
            
            # Calculate per-example cosine similarities
            for i in range(len(references)):
                ref_emb = ref_embeddings[i]
                hyp_emb = hyp_embeddings[i]
                # Ensure vectors are normalized to prevent numerical issues
                ref_norm = np.linalg.norm(ref_emb)
                hyp_norm = np.linalg.norm(hyp_emb)
                
                if ref_norm > 0 and hyp_norm > 0:
                    similarity = np.dot(ref_emb, hyp_emb) / (ref_norm * hyp_norm)
                    semantic_similarities.append(float(similarity))
                else:
                    semantic_similarities.append(0.0)
            
            # Calculate average semantic similarity
            metrics['semantic_similarity'] = float(np.mean(semantic_similarities))
            
            # Calculate semantic WER as 1 - semantic similarity (simplified)
            semantic_wer = 1.0 - metrics['semantic_similarity']
            metrics['semantic_wer'] = float(semantic_wer)
            
            # Add to per-example metrics
            for i, metrics_dict in enumerate(example_metrics_data):
                if i < len(semantic_similarities):
                    metrics_dict['semantic_similarity'] = semantic_similarities[i]
                    metrics_dict['semantic_wer'] = 1.0 - semantic_similarities[i]
        except Exception as e:
            print(f"Warning: Could not calculate semantic metrics: {e}")
        
        # Create per-example metrics list
        per_example_metrics = example_metrics_data
        
        return metrics, per_example_metrics
    
    def save_additional_metrics_to_file(self, metrics, file_path=None):
        """
        Save additional metrics to a separate file
        
        Parameters:
        - metrics: Dictionary with metrics to save
        - file_path: Path to save file (default: metrics.txt in current directory)
        """
        if not metrics:
            print("No metrics to save")
            return
            
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f'metrics_{timestamp}.txt'
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            with open(file_path, 'w') as f:
                f.write("=== ADDITIONAL METRICS ===\n\n")
                for metric, value in sorted(metrics.items()):
                    f.write(f"{metric}: {value:.4f}\n")
            
            print(f"Saved additional metrics to {file_path}")
        except Exception as e:
            print(f"Error saving additional metrics: {e}")

    def analyze_json_dataset_with_comparisons(self, json_file, output_file=None, max_samples=None, include_all_metrics=False, export_csv=False):
        """
        Analyze a dataset with both standard and weighted approaches for comparison
        
        Parameters:
        - json_file: Path to JSON file with reference-hypothesis pairs
        - output_file: Path to save analysis results (optional)
        - max_samples: Maximum number of samples to process (default: all)
        - include_all_metrics: Whether to include additional metrics
        - export_csv: Whether to export results to CSV
        
        Returns:
        - Dictionary with analysis results
        """
        try:
            # Load JSON data
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            print(f"Loaded data from {json_file}")
            
            # Create pairs for analysis based on the JSON structure
            example_pairs = []
            
            # Special case: Dictionary with "ref"/"hypo" lists
            if isinstance(data, dict) and 'ref' in data and 'hypo' in data:
                if isinstance(data['ref'], list) and isinstance(data['hypo'], list):
                    # Pair up matching indices from ref and hypo lists
                    num_pairs = min(len(data['ref']), len(data['hypo']))
                    print(f"Found parallel ref/hypo lists with {num_pairs} pairs")
                    
                    for i in range(num_pairs):
                        example_pairs.append((data['ref'][i], data['hypo'][i]))
            
            # If we don't have pairs yet, try other formats
            if not example_pairs:
                # Try standard list of dicts format
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            if 'reference' in item and 'hypothesis' in item:
                                example_pairs.append((item['reference'], item['hypothesis']))
                            elif 'ref' in item and 'hyp' in item:
                                example_pairs.append((item['ref'], item['hyp']))
                            elif 'ground_truth' in item and 'prediction' in item:
                                example_pairs.append((item['ground_truth'], item['prediction']))
                        elif isinstance(item, list) and len(item) >= 2:
                            example_pairs.append((item[0], item[1]))
                
                # Try top-level dict format
                elif isinstance(data, dict):
                    if 'reference' in data and 'hypothesis' in data:
                        example_pairs.append((data['reference'], data['hypothesis']))
                    elif 'ref' in data and 'hyp' in data:
                        example_pairs.append((data['ref'], data['hyp']))
            
            if not example_pairs:
                print("Could not extract reference-hypothesis pairs from the JSON file.")
                return None
            
            # Limit number of samples if specified
            if max_samples is not None and len(example_pairs) > max_samples:
                print(f"Limiting analysis to {max_samples} samples (out of {len(example_pairs)} total)")
                example_pairs = example_pairs[:max_samples]
            
            print(f"Processing {len(example_pairs)} reference-hypothesis pairs")
            
            # Process all examples with both approaches
            all_results = []
            
            # Add progress reporting
            print("Starting evaluation with both standard and weighted approaches...")
            total_examples = len(example_pairs)
            
            for i, (ref, hyp) in enumerate(example_pairs):
                if i % 10 == 0 or i == total_examples - 1:
                    print(f"Evaluating example {i+1}/{total_examples} ({(i+1)/total_examples*100:.1f}%)")
                
                # Evaluate with both approaches
                results = self.compare_standard_and_weighted(ref, hyp)
                all_results.append(results)
            
            # Calculate summary statistics with simplified format
            summary = {
                'num_examples': len(all_results),
                'standard_viseme_score': np.mean([r['standard']['standard_viseme_score'] for r in all_results]),
                'standard_phoneme_score': np.mean([r['standard']['standard_phoneme_score'] for r in all_results]),
                'vips_score': np.mean([r['weighted']['vips_score'] for r in all_results]),
            }
            
            # Add additional metrics if requested
            per_example_metrics = []
            if include_all_metrics:
                print("\nCalculating additional metrics...")
                additional_metrics, per_example_metrics = self.calculate_additional_metrics(example_pairs)
                summary['additional_metrics'] = additional_metrics
            
            # Print summary
            print("\n=== ANALYSIS SUMMARY ===")
            print(f"Total examples: {summary['num_examples']}")
            print(f"Standard Viseme Score: {summary['standard_viseme_score']:.3f}")
            print(f"Standard Phoneme Score: {summary['standard_phoneme_score']:.3f}")
            print(f"ViPS Score: {summary['vips_score']:.3f}")
            
            # Print additional metrics if requested
            if include_all_metrics and 'additional_metrics' in summary:
                print("\nAdditional Metrics:")
                for metric, value in summary['additional_metrics'].items():
                    print(f"  {metric}: {value:.4f}")
            
            # Save results if output file specified
            if output_file:
                # Create directory if needed
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                
                results_to_save = {
                    'summary': summary,
                    'examples': all_results
                }
                
                # Add per-example metrics if calculated
                if per_example_metrics:
                    # Merge the per-example metrics with the existing results
                    for i, (result, metrics) in enumerate(zip(all_results, per_example_metrics)):
                        if i < len(all_results):
                            # Add additional metrics to each example
                            result['metrics'] = metrics
                            
                            # Add backward compatibility mapping for 'phonemes'
                            if 'ref_phonemes' in result and 'phonemes' not in result:
                                result['phonemes'] = result['ref_phonemes']
                
                with open(output_file, 'w') as f:
                    json.dump(results_to_save, f, indent=2)
                
                print(f"\nSaved analysis results to {output_file}")
            
            # Export to CSV if requested
            if export_csv:
                csv_file = os.path.join(os.path.dirname(os.path.abspath(output_file if output_file else 'results.json')), 'results.csv')
                self.export_results_to_csv(all_results, per_example_metrics, csv_file)
                print(f"Saved detailed results to CSV: {csv_file}")
            
            return {
                'summary': summary,
                'results': all_results
            }
            
        except Exception as e:
            print(f"Error analyzing dataset: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def export_results_to_csv(self, results, additional_metrics=None, csv_file='results.csv'):
        """
        Export evaluation results to a CSV file with proper column separation
        
        Parameters:
        - results: List of evaluation result dictionaries
        - additional_metrics: List of additional metrics dictionaries (optional)
        - csv_file: Path to save the CSV file
        """
        import csv
        
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(os.path.abspath(csv_file)), exist_ok=True)
            
            # Define CSV columns - only include necessary columns
            columns = [
                'reference', 
                'hypothesis',
                'ref_phonemes',
                'hyp_phonemes',
                'standard_viseme_score',
                'standard_phoneme_score',
                'viseme_score',
                'vips_score',
                'viseme_score_difference',
                'vips_score_difference'
            ]
            
            # Add additional metric columns if available
            additional_metric_keys = []
            if additional_metrics and len(additional_metrics) > 0:
                for metrics in additional_metrics:
                    if metrics:  # Check if metrics exist for this sample
                        for key in metrics.keys():
                            # Only add metrics that aren't already included and aren't reference/hypothesis
                            if key not in ['reference', 'hypothesis'] and key not in columns and key not in additional_metric_keys:
                                additional_metric_keys.append(key)
                
                # Sort additional metrics alphabetically for consistency
                additional_metric_keys.sort()
                columns.extend(additional_metric_keys)
            
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                
                # Write each result row
                for i, result in enumerate(results):
                    # Make sure we have valid data for this result
                    if not isinstance(result, dict) or 'standard' not in result or 'weighted' not in result:
                        continue
                        
                    # Basic row information
                    row = {
                        'reference': result.get('reference', ''),
                        'hypothesis': result.get('hypothesis', ''),
                        'ref_phonemes': ' '.join(result.get('ref_phonemes', [])) if result.get('ref_phonemes') else '',
                        'hyp_phonemes': ' '.join(result.get('hyp_phonemes', [])) if result.get('hyp_phonemes') else ''
                    }
                    
                    # Standard metrics
                    std = result.get('standard', {})
                    row['standard_viseme_score'] = round(std.get('standard_viseme_score', 0), 4) if 'standard_viseme_score' in std else ''
                    row['standard_phoneme_score'] = round(std.get('standard_phoneme_score', 0), 4) if 'standard_phoneme_score' in std else ''
                    
                    # Weighted metrics
                    wgt = result.get('weighted', {})
                    row['viseme_score'] = round(wgt.get('viseme_score', 0), 4) if 'viseme_score' in wgt else ''
                    row['vips_score'] = round(wgt.get('vips_score', 0), 4) if 'vips_score' in wgt else ''
                    
                    # Calculate score differences
                    if row['viseme_score'] and row['standard_viseme_score']:
                        row['viseme_score_difference'] = round(row['viseme_score'] - row['standard_viseme_score'], 4)
                    else:
                        row['viseme_score_difference'] = ''
                        
                    if row['vips_score'] and row['standard_phoneme_score']:
                        row['vips_score_difference'] = round(row['vips_score'] - row['standard_phoneme_score'], 4)
                    else:
                        row['vips_score_difference'] = ''
                    
                    # Add additional metrics if available
                    if additional_metrics and i < len(additional_metrics) and additional_metrics[i]:
                        metric_data = additional_metrics[i]
                        for key in additional_metric_keys:
                            value = metric_data.get(key, '')
                            # Format numeric values
                            if isinstance(value, (int, float)):
                                row[key] = round(value, 4)
                            else:
                                row[key] = value
                    
                    writer.writerow(row)
                    
            print(f"Successfully exported {len(results)} results to {csv_file}")
            
        except Exception as e:
            print(f"Error exporting results to CSV: {e}")
            import traceback
            traceback.print_exc()

    def visualize_results(self, analysis_results, output_dir):
        """Create visualizations for the analysis results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all results from the analysis
        all_results = analysis_results.get('results', [])
        summary = analysis_results.get('summary', {})
        
        if not all_results:
            print("No results to visualize")
            return
        
        print(f"Creating visualizations in {output_dir}...")
        
        # Create confusion matrices
        self._plot_confusion_matrices(all_results, output_dir)
        
        print("Visualizations complete")
    
    def _plot_confusion_matrices(self, all_results, output_dir):
        """Plot viseme confusion matrices for standard and weighted approaches"""
        # Collect viseme substitutions from alignments
        std_substitutions = []
        wgt_substitutions = []
        
        for result in all_results:
            ref_phonemes = result.get('ref_phonemes', [])
            hyp_phonemes = result.get('hyp_phonemes', [])
            
            # Ensure we have phonemes for processing
            if not ref_phonemes:
                ref_phonemes = self.text_to_phonemes(result['reference'])
            if not hyp_phonemes:
                hyp_phonemes = self.text_to_phonemes(result['hypothesis'])
            
            # Convert phonemes to visemes - using consistent phoneme-to-viseme mapping
            ref_visemes = [self.map_phoneme_to_viseme(p) for p in ref_phonemes]
            hyp_visemes = [self.map_phoneme_to_viseme(p) for p in hyp_phonemes]
            
            # Get alignment and find substitutions
            # First for standard approach
            self.use_weighted_distance = False
            alignment, _ = self.calculate_viseme_alignment(ref_visemes, hyp_visemes)
            for op, ref_v, hyp_v in alignment:
                if op == 'substitute':
                    std_substitutions.append((ref_v, hyp_v))
            
            # Then for weighted approach
            self.use_weighted_distance = True
            alignment, _ = self.calculate_viseme_alignment(ref_visemes, hyp_visemes)
            for op, ref_v, hyp_v in alignment:
                if op == 'substitute':
                    wgt_substitutions.append((ref_v, hyp_v))
        
        # Restore original setting
        self.use_weighted_distance = getattr(self, 'use_weighted_distance', True)
        
        # Get unique viseme classes from the data
        all_visemes = list(range(22))  # We use 22 viseme classes (0-21)
        
        # Use the viseme_id_to_name from the parent class for better readability
        if hasattr(self, 'viseme_id_to_name'):
            viseme_names = self.viseme_id_to_name
        else:
            # Fallback to basic naming
            viseme_names = {v: f"Viseme {v}" for v in all_visemes}
        
        # Create shortened labels for the axis to avoid overlap
        viseme_labels = [f"{v}" for v in all_visemes]
        
        # Create confusion matrices
        if std_substitutions and all_visemes:
            # Standard approach matrix   
            std_true, std_pred = zip(*std_substitutions)
            std_cm = confusion_matrix(std_true, std_pred, labels=all_visemes)
            
            # Normalize safely - avoid division by zero
            row_sums = std_cm.sum(axis=1)
            std_cm_norm = np.zeros_like(std_cm, dtype=float)
            for i in range(len(row_sums)):
                if row_sums[i] > 0:  # Only normalize if row sum is positive
                    std_cm_norm[i] = std_cm[i] / row_sums[i]
            
            # Weighted approach matrix
            wgt_true, wgt_pred = zip(*wgt_substitutions)
            wgt_cm = confusion_matrix(wgt_true, wgt_pred, labels=all_visemes)
            
            # Normalize safely - avoid division by zero
            row_sums = wgt_cm.sum(axis=1)
            wgt_cm_norm = np.zeros_like(wgt_cm, dtype=float)
            for i in range(len(row_sums)):
                if row_sums[i] > 0:  # Only normalize if row sum is positive
                    wgt_cm_norm[i] = wgt_cm[i] / row_sums[i]
            
            # Add diagnostic information about viseme class distribution
            missing_std_visemes = [v for i, v in enumerate(all_visemes) if row_sums[i] == 0]
            if missing_std_visemes:
                print(f"Note: Some viseme classes were not found in reference data: {missing_std_visemes}")
                print("This is normal if your dataset doesn't contain examples of these viseme classes.")
                missing_names = [self.viseme_id_to_name.get(v, f"Class {v}") for v in missing_std_visemes]
                print(f"Missing classes: {', '.join(missing_names)}")
            
            # Calculate difference matrix
            diff_cm = wgt_cm_norm - std_cm_norm
            
            # Define colormaps
            std_cmap = plt.cm.Blues
            wgt_cmap = plt.cm.Blues
            diff_cmap = sns.diverging_palette(240, 10, as_cmap=True)
            
            # Helper function to create and save a confusion matrix heatmap
            def create_heatmap(matrix, title, filename, cmap, vmin=0, vmax=1, center=None):
                plt.figure(figsize=(16, 12))
                
                # Split figure for main plot and legend
                gs = plt.GridSpec(1, 2, width_ratios=[3, 1])
                
                # Main heatmap axis
                ax_heatmap = plt.subplot(gs[0])
                ax_legend = plt.subplot(gs[1])
                
                # Create the heatmap
                sns.heatmap(matrix, annot=True, fmt='.2f', cmap=cmap, square=True, 
                           vmin=vmin, vmax=vmax, center=center,
                           annot_kws={"size": 8}, xticklabels=viseme_labels, 
                           yticklabels=viseme_labels, ax=ax_heatmap)
                
                # Set labels
                ax_heatmap.set_title(title)
                ax_heatmap.set_xlabel('Predicted Viseme')
                ax_heatmap.set_ylabel('True Viseme')
                
                # Add legend in the right panel
                ax_legend.axis('off')
                legend_text = "Viseme Reference:\n\n"
                for v in all_visemes:
                    legend_text += f"{v}: {viseme_names.get(v, 'Unknown')}\n"
                ax_legend.text(0, 0.5, legend_text, va='center', fontsize=10)
                
                # Save plot
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, filename), dpi=300)
                plt.close()
            
            # Create the three different confusion matrix visualizations
            create_heatmap(
                std_cm_norm, 
                'Standard Approach - Viseme Confusion Matrix',
                'standard_confusion_matrix.png',
                std_cmap
            )
            
            create_heatmap(
                wgt_cm_norm, 
                'Weighted Approach - Viseme Confusion Matrix',
                'weighted_confusion_matrix.png',
                wgt_cmap
            )
            
            create_heatmap(
                diff_cm, 
                'Difference (Weighted - Standard) Confusion Matrix',
                'difference_confusion_matrix.png',
                diff_cmap,
                vmin=-0.5,
                vmax=0.5,
                center=0
            )
    
    def standard_phonetic_distance(self, phoneme1, phoneme2):
        """
        Calculate phonetic distance using the standard (non-weighted) approach.
        This ensures we have a clear distinction between standard and weighted distances.
        
        Args:
            phoneme1: First phoneme
            phoneme2: Second phoneme
            
        Returns:
            float: Distance value between 0.0 (identical) and 1.0 (maximally different)
        """
        # Call the main distance function with use_weights=False
        return self.calculate_phoneme_distance(phoneme1, phoneme2, use_weights=False)

def save_examples_to_csv(examples, evaluator, filename="phonetic_examples.csv"):
    """
    Save the comparison examples to a CSV file with additional analysis columns
    
    Args:
        examples: List of example tuples (reference, hypothesis, type, description)
        evaluator: LipReadingEvaluator instance to calculate scores
        filename: Path to save the CSV file
    """
    import csv
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['reference', 'hypothesis', 'confusion_type', 'description', 
                         'substituted_phonemes', 'articulatory_difference',
                         'standard_viseme_score', 'standard_phoneme_score', 'vips_score',
                         'viseme_diff', 'vips_diff']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for item in examples:
                # Unpack values from the example tuple
                if len(item) >= 4:
                    ref, hyp, confusion_type, description = item[:4]
                    
                    # Add optional fields if available
                    substituted_phonemes = item[4] if len(item) > 4 else ""
                    articulatory_difference = item[5] if len(item) > 5 else ""
                else:
                    # Backward compatibility for simple (ref, hyp) tuples
                    ref, hyp = item[:2]
                    confusion_type = ""
                    description = ""
                    substituted_phonemes = ""
                    articulatory_difference = ""
                
                # Calculate scores
                results = evaluator.compare_standard_and_weighted(ref, hyp)
                std_viseme = results['standard']['standard_viseme_score']
                std_phonetic = results['standard']['standard_phoneme_score']
                vips_score = results['weighted']['vips_score']
                
                # Calculate differences
                vips_diff = vips_score - std_phonetic
                
                # Write to CSV
                writer.writerow({
                    'reference': ref,
                    'hypothesis': hyp,
                    'confusion_type': confusion_type,
                    'description': description,
                    'substituted_phonemes': substituted_phonemes,
                    'articulatory_difference': articulatory_difference,
                    'standard_viseme_score': f"{std_viseme:.3f}",
                    'standard_phoneme_score': f"{std_phonetic:.3f}",
                    'vips_score': f"{vips_score:.3f}",
                    'viseme_diff': "0.000",  # No longer calculated
                    'vips_diff': f"{vips_diff:.3f}"
                })
                
        print(f"Saved {len(examples)} examples with scores to {filename}")
    except Exception as e:
        print(f"Error saving examples to CSV: {e}")

def main():
    """Main function for demonstrating and comparing different alignment approaches"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phonetically-Weighted Viseme Scoring')
    parser.add_argument('--json', type=str, help='JSON file with reference-hypothesis pairs')
    parser.add_argument('--weights', type=str, help='Path to save computed weights (will not load from this file)')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save all outputs (default: current directory)')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to process')
    parser.add_argument('-all', action='store_true', help='Calculate all metrics including WER, CER, and others')
    parser.add_argument('--csv', action='store_true', help='Export results to CSV file')
    parser.add_argument('--save-examples', action='store_true', help='Save examples to CSV file')
    parser.add_argument('--save-text', action='store_true', help='Save examples to readable text file')
    parser.add_argument('--weight-method', type=str, choices=['both', 'entropy', 'visual'], default='both',
                        help='Method to calculate feature weights: both (default), entropy-only, or visual-only')
    parser.add_argument('--compare-methods', action='store_true', help='Compare results using different weight methods')
    args = parser.parse_args()
    
    # Set up output directory - only create viseme_output if save_dir is not specified
    if args.save_dir is None:
        args.save_dir = '.'  # Use current directory as default
    
    # Only create directory if it's not the current directory
    if args.save_dir != '.':
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Define output paths
    if args.weights:
        weights_path = args.weights
    else:
        method_suffix = f"_{args.weight_method}" if args.weight_method != "both" else ""
        weights_path = os.path.join(args.save_dir, f'viseme_weights{method_suffix}.json')
    
    results_path = os.path.join(args.save_dir, f'results{method_suffix}.json')
    metrics_path = os.path.join(args.save_dir, f'metrics{method_suffix}.txt')
    examples_path = os.path.join(args.save_dir, f'phonetic_examples{method_suffix}.csv')
    comparison_text_path = os.path.join(args.save_dir, f'phonetic_comparisons{method_suffix}.txt')
    
    if args.compare_methods:
        # Run analysis with all three weight methods and compare
        print("\n=== COMPARING DIFFERENT WEIGHT METHODS ===")
        methods = ['both', 'entropy', 'distinctiveness']
        method_results = {}
        example_pairs = []
        
        # Load pairs from JSON if provided, or use sample examples
        if args.json:
            try:
                with open(args.json, 'r') as f:
                    data = json.load(f)
                
                # Extract pairs depending on the format
                if isinstance(data, dict) and 'ref' in data and 'hypo' in data:
                    pairs = list(zip(data['ref'], data['hypo']))
                elif isinstance(data, list):
                    pairs = [(item.get('reference', item.get('ref', '')), 
                             item.get('hypothesis', item.get('hyp', '')))
                             for item in data if isinstance(item, dict)]
                
                if pairs:
                    example_pairs = pairs[:20]  # Limit to 20 examples
            except Exception as e:
                print(f"Error loading JSON: {e}")
                example_pairs = []
        
        # If no pairs from JSON, use sample examples
        if not example_pairs:
            example_pairs = [
                ("Hello world", "Hello word"),
                ("Please pass the salt", "Please pass the fault"),
                ("Bring me a glass of water", "Bring me a glass of vodka"),
                ("Meet me at five", "Meet me at nine"),
                ("I enjoy boys playing outside", "I enjoy toys playing outside"),
            ]
        
        # Compare results for each method
        print(f"\nAnalyzing {len(example_pairs)} example pairs with three different weight methods")
        
        for method in methods:
            print(f"\n--- EVALUATING WITH {method.upper()} WEIGHTS ---")
            evaluator = WeightedLipReadingEvaluator(weight_method=method)
            
            # Save the weights
            method_suffix = f"_{method}" if method != "both" else ""
            method_weights_path = os.path.join(args.save_dir, f'viseme_weights{method_suffix}.json')
            evaluator.save_weights_to_file(method_weights_path)
            
            # Calculate scores for all examples
            scores = []
            for ref, hyp in example_pairs:
                result = evaluator.compare_standard_and_weighted(ref, hyp)
                scores.append({
                    'reference': ref,
                    'hypothesis': hyp,
                    'standard_score': result['standard']['standard_viseme_score'],
                    'weighted_score': result['weighted']['viseme_score'],
                    'improvement': result['weighted']['viseme_score'] - result['standard']['standard_viseme_score']
                })
            
            # Calculate average scores
            avg_standard = sum(s['standard_score'] for s in scores) / len(scores)
            avg_weighted = sum(s['weighted_score'] for s in scores) / len(scores)
            avg_improvement = sum(s['improvement'] for s in scores) / len(scores)
            
            method_results[method] = {
                'scores': scores,
                'avg_standard': avg_standard,
                'avg_weighted': avg_weighted,
                'avg_improvement': avg_improvement
            }
            
            print(f"Average standard score: {avg_standard:.3f}")
            print(f"Average weighted score: {avg_weighted:.3f}")
            print(f"Average improvement: {avg_improvement:.3f}")
            
            # Save detailed results to JSON
            details_path = os.path.join(args.save_dir, f'method_comparison_{method}.json')
            with open(details_path, 'w') as f:
                json.dump({'scores': scores, 'summary': {
                    'avg_standard': avg_standard,
                    'avg_weighted': avg_weighted,
                    'avg_improvement': avg_improvement
                }}, f, indent=2)
        
        # Summarize and compare
        print("\n=== WEIGHT METHOD COMPARISON SUMMARY ===")
        print("Method       | Std Score | Weighted Score | Improvement")
        print("-------------|-----------|---------------|------------")
        for method in methods:
            res = method_results[method]
            print(f"{method:12} | {res['avg_standard']:.3f}    | {res['avg_weighted']:.3f}       | {res['avg_improvement']:.3f}")
        
        # Save overall comparison to JSON
        comparison_path = os.path.join(args.save_dir, 'weight_method_comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump({method: {
                'avg_standard': res['avg_standard'],
                'avg_weighted': res['avg_weighted'],
                'avg_improvement': res['avg_improvement']
            } for method, res in method_results.items()}, f, indent=2)
        
        print(f"\nSaved comparison results to {comparison_path}")
        print("Individual method results saved to method_comparison_*.json files")
        
        return
    
    # Create weighted evaluator with the specified weight method
    evaluator = WeightedLipReadingEvaluator(weight_method=args.weight_method)
    
    # Debug: Check feature weights
    print(f"Initial feature weights: {len(evaluator.feature_weights) if hasattr(evaluator, 'feature_weights') and evaluator.feature_weights else 'Not set'}")
    
    # Print phoneme conversion approach
    print("\n=== USING IPA CONVERSION ===")
    print(f"Using weight method: {args.weight_method}")
    
    print("Note: Weights are now always calculated from scratch")
    
    # Quick conversion test with a sentence containing 'oy' diphthong
    test_sentence = "I enjoy toys and boys playing outside."
    phonemes = evaluator.text_to_phonemes(test_sentence)
    print(f"Test conversion: '{test_sentence}'")
    print(f"Phonemes: {' '.join(phonemes)}")

    # Add detailed debugging of each phoneme and its viseme mapping
    print("\nDetailed phoneme-to-viseme mapping:")
    for p in phonemes:
        viseme_id = evaluator.map_phoneme_to_viseme(p)
        if viseme_id == 0:
            print(f"  '{p}' -> {viseme_id} (SILENCE)")
        else:
            print(f"  '{p}' -> {viseme_id} ({evaluator.viseme_id_to_name.get(viseme_id, 'Unknown')})")

    # Show all visemes
    visemes = [evaluator.map_phoneme_to_viseme(p) for p in phonemes]
    print(f"Visemes: {' '.join(str(v) for v in visemes)}")

    # Count silence visemes
    silence_count = sum(1 for v in visemes if v == 0)
    print(f"Silence visemes: {silence_count} out of {len(visemes)} ({silence_count/len(visemes)*100:.1f}%)")
    print()
    
    # Save weights
    if hasattr(evaluator, 'feature_weights') and evaluator.feature_weights:
        print(f"Feature weights before saving: {len(evaluator.feature_weights)} entries")
        evaluator.save_weights_to_file(weights_path)
        print(f"Saved computed weights to {weights_path}")
    else:
        print(f"No feature weights available to save. Type: {type(evaluator.feature_weights)}")
        print(f"Feature weights content: {evaluator.feature_weights}")
    
    # Process JSON file if provided
    if args.json:
        results = evaluator.analyze_json_dataset_with_comparisons(
            args.json, 
            output_file=results_path,
            max_samples=args.max_samples,
            include_all_metrics=args.all,
            export_csv=args.csv
        )
        
        if results:
            # If all metrics were requested, save them
            if args.all and 'summary' in results and 'additional_metrics' in results['summary']:
                evaluator.save_additional_metrics_to_file(
                    results['summary']['additional_metrics'], 
                    file_path=metrics_path
                )
            
            print("\nGenerating visualizations...")
            evaluator.visualize_results(results, output_dir=args.save_dir)
            print(f"\nAnalysis complete! Using IPA conversion with {args.weight_method} weighting.")
            print(f"All outputs saved to: {args.save_dir}")


if __name__ == "__main__":
    main() 