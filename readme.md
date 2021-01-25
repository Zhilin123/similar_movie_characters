## Identifying similar movie characters

Ever had the feeling that a character in a movie feels pretty much the same as another character in another movie? Wouldn't it be amazing if we could find more of our favorite character roles, understand what these recurring themes are and perhaps enjoy our binge-watching a little more?
This code does exactly that.

In this task, we try to predict movie characters that are most likely from a common trope (theme in cinematic speak) as another character.

1. ```prepare_dataset/``` : Shows how concise descriptions of characters in tropes are downloaded from allthetropes.org and post-processed. A prepared version of the dataset is available [here](https://drive.google.com/file/d/1cF_NMP6yPGyGDQEwmkb8-cHKJB359Xka/view?usp=sharing).

2. ```get_candidate_embeddings.py```: Demonstrates how paragraph embeddings can be obtained based on these descriptions of movie characters.

3. ```generate_candidates_for_refinement.py```: Approximately identifies candidates are likely to be similar to a movie character (using their paragraph-level text embeddings)

4. ```generate_cosine_like_grid_for_siamesebert.py```: Helper script (to ```generate_candidates_for_refinement.py```) to generate candidates using the SiameseBERT model.

5. ```ccm_training.py```: Trains a Character Comparison Model to more precisely determine whether two movie characters are similar.

6. ```siamesebert_training.py```: Training a baseline model in identifying similarity between movie characters

7. ```siamesebert_model.py```: The model architecture of the baseline model (trained in ```siamesebert_training.py```)

8. ```eval/compare_overlap_with_exhaustive_comparison.py```: Determine which approach in selecting candidates overlaps most with using CCM to exhaustive compare all possible character-pairs(for a tiny number of characters).

9. ```eval/evaluat*```: Uses automated metrics (Recall @ k, normalized Discounted Cumulative Gain @ k and Mean Reciprocal Rank) to compare the performance between baseline models and our Select-and-Refine models.
