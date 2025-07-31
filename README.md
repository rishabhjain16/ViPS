# ViPS : Visemic-Phonetic Scoring for Audio-Visual Speech Recognition Evaluation

ViPS is a unified audio-visual analysis framework that evaluates the similarity between reference and hypothesis transcriptions by integrating phonetic and visual speech dimensions through feature-based weighting. The feature-specific weights are based on entropy and visual distinctiveness. It builds on articulatory feature-based phonetic edit distance (PED), which uses phonetic features to compute linguistically grounded substitution costs. We adapt the classic dynamic programming algorithm for sequence alignment incorporating our feature-weighted phoneme distance as the substitution cost. This cost reflects the articulatory similarity between phonemes, assigning lower values to pairs that share perceptually salient features. The score ranges from 0 to 1, with higher values indicating greater similarity.



### Code: 
Code will be provided in coming months.


#### Note on Terminology in Folders and Ablations

In some added folders and ablation experiments, you may find references to ViPS as "WVS" or "WPS." These were earlier names used before we finalized the term "ViPS," while we were exploring different ablation settings and experimental variations. All such references correspond to the current ViPS metric.

**Weighted Phonetic Score (WPS):**
The Weighted Phonetic Score quantifies the similarity between reference and hypothesis transcriptions by considering the articulatory and phonetic features of each phoneme. 

**Weighted Visemic Score (WVS):**
It is slightly different from WPS as the Weighted Visemic Score measures the similarity between reference and hypothesis transcriptions based on visual speech features (visemes). which were mapped using WPS as done in ViPS.
