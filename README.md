# ViPS : Visemic-Phonetic Scoring for Audio-Visual Speech Recognition Evaluation

ViPS is a unified audio-visual analysis framework that evaluates the similarity between reference and hypothesis transcriptions by integrating phonetic and visual speech dimensions through feature-based weighting. The feature-specific weights are based on entropy and visual distinctiveness. It builds on articulatory feature-based phonetic edit distance (PED), which uses phonetic features to compute linguistically grounded substitution costs. We adapt the classic dynamic programming algorithm for sequence alignment incorporating our feature-weighted phoneme distance as the substitution cost. This cost reflects the articulatory similarity between phonemes, assigning lower values to pairs that share perceptually salient features. The score ranges from 0 to 1, with higher values indicating greater similarity.




## ViPS Metric Script Usage (`vips.py`)

The `vips.py` script provides a comprehensive set of tools for evaluating, analyzing, and visualizing audio-visual speech recognition (AVSR) outputs using the ViPS metric and related phonetic/visemic metrics.

### Main Capabilities
- Compute ViPS and other metrics (WER, CER, BERTScore, Semantic Similarity) for reference/hypothesis pairs
- Compare standard and weighted phonetic/visemic scoring
- Analyze results from JSON datasets
- Export results to CSV or text
- Visualize confusion matrices and metric distributions
- Save and load feature weights

### Example Usage

#### 1. Analyze a JSON file of reference/hypothesis pairs
```bash
python vips.py --json path/to/data.json --save_dir results_dir
```
This will compute ViPS and related metrics for all pairs in the JSON file and save results in `results_dir`.

#### 2. Compare standard and weighted scoring approaches
```bash
python vips.py --json path/to/data.json --compare --save_dir results_dir
```
Adds a detailed comparison of standard (unweighted) and weighted ViPS metrics for each pair.

#### 3. Export results to CSV
```bash
python vips.py --json path/to/data.json --csv --save_dir results_dir
```
Saves detailed per-example results in CSV format for further analysis.

#### 4. Save feature weights to a file
```bash
python vips.py --json path/to/data.json --weights path/to/weights.json --save_dir results_dir
```
Saves the computed feature weights and viseme similarity matrix for reuse or inspection.

#### 5. Visualize confusion matrices and metric distributions
```bash
python vips.py --json path/to/data.json --save_dir results_dir
# After running, check the results_dir for PNG plots and summary files.
```

#### 6. Run ablation or method comparison
```bash
python vips.py --json path/to/data.json --compare-methods --save_dir results_dir
```
Compares different feature weighting methods (entropy, distinctiveness, both).

#### 7. Save example analyses to text or CSV
```bash
python vips.py --json path/to/data.json --save-examples --save-text --save_dir results_dir
```
Saves detailed example-by-example analysis in both text and CSV formats.

### Arguments
- `--json`: Path to a JSON file with reference/hypothesis pairs (required for most analyses)
- `--weights`: Path to save computed feature weights (optional)
- `--save_dir`: Directory to save all outputs (default: `viseme_output`)
- `--max_samples`: Limit the number of samples processed (optional)
- `--compare`: Run detailed comparison of standard vs. weighted scoring
- `--csv`: Export results to CSV
- `--save-examples`: Save example analyses to CSV
- `--save-text`: Save example analyses to text
- `--weight-method`: Choose feature weighting method (`both`, `entropy`, `distinctiveness`)
- `--compare-methods`: Compare results using different weighting methods

### Output Files
- `results.json`: Full results for all pairs
- `metrics.txt`: Summary of computed metrics
- `phonetic_examples.csv`: Per-example analysis
- `phonetic_comparisons.txt`: Readable text report of example analyses
- `feature_weights.json`/`feature_weights_table.txt`: Feature weights and analysis
- PNG plots: Confusion matrices and metric visualizations

### Notes
- The script requires dependencies listed in `requirements.txt` (see installation instructions above).
- Input JSON can be a list of dicts or a dict with `ref` and `hypo` lists.
- For custom analyses, see the script's help: `python vips.py --help`


#### Note on Terminology in Folders and Ablations

In some added folders and ablation experiments, you may find references to ViPS as "WVS" or "WPS." These were earlier names used before we finalized the term "ViPS," while we were exploring different ablation settings and experimental variations. All such references correspond to the current ViPS metric.

**Weighted Phonetic Score (WPS):**
The Weighted Phonetic Score quantifies the similarity between reference and hypothesis transcriptions by considering the articulatory and phonetic features of each phoneme. 

**Weighted Visemic Score (WVS):**
It is slightly different from WPS as the Weighted Visemic Score measures the similarity between reference and hypothesis transcriptions based on visual speech features (visemes). which were mapped using WPS as done in ViPS.
