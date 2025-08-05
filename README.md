# ViPS : Visemic-Phonetic Scoring for Audio-Visual Speech Recognition Evaluation

ViPS is a unified audio-visual analysis framework that evaluates the similarity between reference and hypothesis transcriptions by integrating phonetic and visual speech dimensions through feature-based weighting. The feature-specific weights are based on entropy and visual distinctiveness. It builds on articulatory feature-based phonetic edit distance (PED), which uses phonetic features to compute linguistically grounded substitution costs. We adapt the classic dynamic programming algorithm for sequence alignment incorporating our feature-weighted phoneme distance as the substitution cost. This cost reflects the articulatory similarity between phonemes, assigning lower values to pairs that share perceptually salient features. The score ranges from 0 to 1, with higher values indicating greater similarity.

## Installation

### Setting up the Environment

1. Clone the repository:
```bash
git clone https://github.com/rishabhjain16/ViPS.git
cd ViPS
```

2. Create and activate a conda environment:
```bash
# Create new environment with Python 3.10
conda create -n vips python=3.10
conda activate vips
```

3. Install basic dependencies:
```bash
# Install PyTorch (adjust cuda version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# Install espeak (required for phonemizer)
# For Ubuntu/Debian:
sudo apt-get install espeak-ng
# For macOS:
brew install espeak
```

### Verifying Installation

To verify the installation:
```bash
# Activate the environment if not already active
conda activate vips

# Run a simple test
python vips.py --help
```

If you see the help message, the installation was successful.

### Troubleshooting

If you encounter issues:
1. Make sure espeak is installed and accessible from command line
2. Verify that all dependencies are installed: `pip freeze`
3. Check if CUDA is properly configured (if using GPU)
4. For phonemizer issues, try reinstalling: `pip install phonemizer --force-reinstall`

## ViPS Metric Script Usage (`vips.py`)

The `vips.py` script provides a comprehensive evaluation framework for analyzing audio-visual speech recognition (AVSR) outputs using the ViPS metric and related phonetic/visemic metrics.

### Main Features
- Three-level Scoring System:
  - Standard Visemic Score (unweighted viseme-based evaluation)
  - Standard Phonetic Score (unweighted phoneme-based evaluation)
  - ViPS Score (weighted phonetic evaluation incorporating visual distinctiveness)
- Comprehensive Metrics Suite:
  - Core metrics: WER, CER
  - Advanced metrics: BLEU, ROUGE, METEOR
  - Deep learning-based metrics: BERTScore, Semantic Similarity
- Feature Weight Analysis:
  - Entropy-based weighting
  - Visual distinctiveness weighting
  - Combined weighting approach

### Quick Start

#### Basic Usage
```bash
python vips.py --json path/to/data.json --save_dir results_dir
```
This computes all three scores (Standard Visemic, Standard Phonetic, and ViPS) for all pairs in the JSON file.

#### Advanced Analysis
```bash
python vips.py --json path/to/data.json --save_dir results_dir --all --weight-method both
```
Includes additional metrics and uses both entropy and visual distinctiveness for weighting.

#### Feature Weight Analysis
```bash
python vips.py --json path/to/data.json --save_dir results_dir --weight-method entropy
```
Uses entropy-only weighting for feature analysis.

### Command Line Arguments
- `--json`: Path to JSON file containing reference-hypothesis pairs
- `--save_dir`: Directory to save outputs (default: current directory)
- `--all`: Calculate additional metrics (WER, CER, BLEU, ROUGE, etc.)
- `--save-text`: Save detailed analysis in human-readable format
- `--weight-method`: Feature weighting method (`both`, `entropy`, `visual`)

### Output Files
1. `results.json`: Complete evaluation results including:
   - Summary statistics
   - Per-example scores
   - Additional metrics (if requested)
2. `vips_weights.json`: Feature weights and analysis details:
   - Feature entropies
   - Visual distinctiveness scores


### Input JSON Format
The script accepts multiple JSON formats:

1. List of Dictionaries:
```json
[
    {
        "reference": "example one",
        "hypothesis": "example won"
    },
    {
        "reference": "test two",
        "hypothesis": "test too"
    }
]
```

2. Parallel Lists:
```json
{
    "ref": ["example one", "test two"],
    "hypo": ["example won", "test too"]
}
```

3. Single Pair:
```json
{
    "reference": "example text",
    "hypothesis": "example test"
}
```


### Notes
- The script requires dependencies listed in `requirements.txt` (see installation instructions above).
- Input JSON can be a list of dicts or a dict with `ref` and `hypo` lists.
- For custom analyses, see the script's help: `python vips.py --help`

### Additional Scripts and Analyses
ViPS is used in various ways throughout this repository for different types of analyses. Each directory contains its own scripts and README with detailed instructions:

- `MaFi/`: Scripts for calculating MaFi scores and correlations with ViPS
- `Ablations/`: Analysis of AVSR model comparisons and SNR impacts
- `Additional_llm_exp/`: Comparisons between AV-Hubert and VSP-LLM models
- `avsr_decoded_outputs/`: Processing and evaluation of AVSR model outputs

Each directory includes specific documentation and example usage in its README file.

