# Ensure parent directory is in sys.path before any other imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from vips import WeightedLipReadingEvaluator
import pandas as pd
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics import accuracy_score
import jiwer
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr
import pickle
import logging
import warnings
# Suppress unnecessary warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Suppress debug messages from other loggers
for logger_name in ['transformers', 'torch', 'tensorflow']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

def get_cached_evaluator(cache_dir='cache'):
    """Get evaluator from cache or create new one"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'evaluator.pkl')
    
    if os.path.exists(cache_file):
        logging.info("Loading cached evaluator...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    logging.info("Initializing new evaluator (this may take a moment)...")
    # Temporarily suppress logging during evaluator initialization
    log_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.WARNING)
    evaluator = WeightedLipReadingEvaluator()
    logging.getLogger().setLevel(log_level)
    
    logging.info("Caching evaluator for future use...")
    with open(cache_file, 'wb') as f:
        pickle.dump(evaluator, f)
    
    return evaluator

def calculate_wer(reference, hypothesis):
    return jiwer.wer(reference, hypothesis)

def calculate_cer(reference, hypothesis):
    return jiwer.cer(reference, hypothesis)

def get_bert_model():
    """Cache BERT model and tokenizer"""
    if not hasattr(get_bert_model, 'model'):
        logging.info("Loading BERT model...")
        get_bert_model.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        get_bert_model.model = AutoModel.from_pretrained("bert-base-uncased")
        get_bert_model.model.eval()
    return get_bert_model.model, get_bert_model.tokenizer

def get_sentence_transformer():
    """Cache sentence transformer model"""
    if not hasattr(get_sentence_transformer, 'model'):
        logging.info("Loading Sentence Transformer model...")
        get_sentence_transformer.model = SentenceTransformer('all-MiniLM-L6-v2')
    return get_sentence_transformer.model

def calculate_bert_score(references, hypotheses, batch_size=32):
    model, tokenizer = get_bert_model()
    scores = []
    
    for i in range(0, len(references), batch_size):
        batch_refs = references[i:i + batch_size]
        batch_hyps = hypotheses[i:i + batch_size]
        
        # Tokenize and encode
        ref_tokens = tokenizer(batch_refs, return_tensors="pt", padding=True, truncation=True)
        hyp_tokens = tokenizer(batch_hyps, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            ref_outputs = model(**ref_tokens)
            hyp_outputs = model(**hyp_tokens)
            
            # Get embeddings from last hidden state
            ref_emb = ref_outputs.last_hidden_state.mean(dim=1)
            hyp_emb = hyp_outputs.last_hidden_state.mean(dim=1)
            
            # Calculate cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(ref_emb, hyp_emb)
            scores.extend(cos_sim.tolist())
    
    return np.mean(scores)

def calculate_semantic_similarity(references, hypotheses, batch_size=32):
    model = get_sentence_transformer()
    scores = []
    
    for i in range(0, len(references), batch_size):
        batch_refs = references[i:i + batch_size]
        batch_hyps = hypotheses[i:i + batch_size]
        
        # Encode sentences
        ref_embeddings = model.encode(batch_refs, show_progress_bar=False)
        hyp_embeddings = model.encode(batch_hyps, show_progress_bar=False)
        
        # Calculate cosine similarities
        for ref_emb, hyp_emb in zip(ref_embeddings, hyp_embeddings):
            similarity = np.dot(ref_emb, hyp_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(hyp_emb))
            scores.append(similarity)
    
    return np.mean(scores)

def process_snr_file(ref_path, hyp_path, evaluator):
    logging.info(f"Processing {os.path.basename(hyp_path)}...")
    
    with open(ref_path, 'r') as f:
        references = f.readlines()
    with open(hyp_path, 'r') as f:
        hypotheses = f.readlines()
        
    references = [ref.strip() for ref in references]
    hypotheses = [hyp.strip() for hyp in hypotheses]
    
    # Calculate metrics
    metrics = {}
    
    logging.info("Calculating WER and CER...")
    metrics['WER'] = calculate_wer(references, hypotheses)
    metrics['CER'] = calculate_cer(references, hypotheses)
    
    logging.info("Calculating BERTScore...")
    metrics['BERTScore'] = calculate_bert_score(references, hypotheses)
    
    logging.info("Calculating Semantic Similarity...")
    metrics['SemanticSimilarity'] = calculate_semantic_similarity(references, hypotheses)
    
    logging.info("Calculating ViPS...")
    vips_scores = []
    for ref, hyp in tqdm(zip(references, hypotheses), total=len(references), desc="ViPS Progress"):
        try:
            score = evaluator.evaluate_pair(ref, hyp)['vips_score']
            vips_scores.append(score)
        except Exception as e:
            logging.warning(f"Error calculating ViPS for ref: '{ref}', hyp: '{hyp}'. Error: {str(e)}")
            vips_scores.append(0.0)
    
    metrics['ViPS'] = np.mean(vips_scores)
    
    return metrics

def main():
    # Initialize the evaluator with caching
    evaluator = get_cached_evaluator()
    
    # Get list of model directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dirs = ['AVEC', 'AV_Relscore', 'Auto-AVSR']
    
    # Store all results
    all_results = []
    
    # Process each model
    for model in tqdm(model_dirs, desc="Processing models"):
        logging.info(f"\nProcessing model: {model}")
        model_path = os.path.join(base_dir, model)
        ref_path = os.path.join(model_path, 'ref.txt')
        
        if not os.path.exists(ref_path):
            logging.warning(f"Reference file not found: {ref_path}")
            continue
        
        # Process each modality
        for modality in ['AO', 'AV']:
            modality_path = os.path.join(model_path, modality)
            
            if not os.path.exists(modality_path):
                logging.warning(f"Modality path not found: {modality_path}")
                continue
            
            logging.info(f"Processing {modality} modality...")
            
            # Process each SNR file
            snr_files = [f for f in os.listdir(modality_path) if f.endswith('.txt')]
            
            for snr_file in tqdm(snr_files, desc=f"Processing {modality} SNR files"):
                try:
                    snr = float(snr_file.replace('.txt', ''))
                    hyp_path = os.path.join(modality_path, snr_file)
                    
                    # Calculate metrics
                    metrics = process_snr_file(ref_path, hyp_path, evaluator)
                    
                    # Add to results
                    result = {
                        'Model': model,
                        'Modality': modality,
                        'SNR': snr,
                        **metrics
                    }
                    all_results.append(result)
                    
                    # Save individual results
                    df = pd.DataFrame([result])
                    csv_path = os.path.join(modality_path, f'metrics_{snr}.csv')
                    df.to_csv(csv_path, index=False)
                    logging.info(f"Saved results to {csv_path}")
                    
                except Exception as e:
                    logging.error(f"Error processing {snr_file}: {str(e)}")
    
    # Save combined results
    df_all = pd.DataFrame(all_results)
    output_path = os.path.join(base_dir, 'all_metrics.csv')
    df_all.to_csv(output_path, index=False)
    logging.info(f"\nSaved combined results to {output_path}")

if __name__ == "__main__":
    main() 