"""
For those clustering that has higher or lower entropies, show their top words.
"""
import json
import numpy as np
from collections import Counter

def calculate_entropy_numpy(probabilities):
    probabilities = np.array(probabilities)  # Ensure input is a NumPy array
    
    # Input validation
    if not np.issubdtype(probabilities.dtype, np.number):
        raise ValueError("All probabilities must be numbers")
    
    # Probability range check
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be in the range [0, 1]")
    
    # Sum of probabilities check
    if not np.isclose(np.sum(probabilities), 1, rtol=1e-9):
        raise ValueError("Sum of probabilities must be close to 1")
    
    # Calculate entropy using vectorized operations, safely handling zero probabilities
    entropy = -np.sum(np.where(probabilities > 0, probabilities * np.log2(probabilities), 0))
    
    return entropy

with open("InstructEmbedBench/datasets/few_event/converted_data.json", 'r') as f:
# with open("InstructEmbedBench/results/goalex_feedbacks_0-gpt-4-feedbacks_v3_converted.json", 'r') as f:
# with open("InstructEmbedBench/results/goalex_rate_my_prof_0-gpt-4-ratemyprof_v1_converted.json", 'r') as f:
    dataset = json.load(f)
    dataset_tuple = []
    for k in dataset:
        for d in dataset[k]:
            dataset_tuple.append((d, k))

file_name = "Llama-2-7b-hf-qa-ml=20_last_ppt_layer_32_event-type_seed=42"
with open(f"propose_results/{file_name}.json", 'r') as f:
    clustering = json.load(f)

components = {}
for c in clustering:
    c_list = [dataset_tuple[t[1]][1] for t in clustering[c]]
    components[c] = dict(Counter(c_list))

with open(f"propose_results/{file_name}_analyze.json", 'r') as f:
    analyze = json.load(f)
    topwords_tfidf = analyze["generation_tfidf"]
    topwords_decode = analyze["contrastive_decoding"]

gathered_clusters = []
for c in components:
    props = np.asarray([components[c][co] for co in components[c]], dtype=np.float64)
    props /= props.sum()
    gathered_clusters.append({
        "components": components[c],
        "topwords_tfidf": topwords_tfidf[c],
        "topwords_decode": topwords_decode[c],
        "entropy": float(calculate_entropy_numpy(props))
    })
gathered_clusters.sort(key=lambda x: x['entropy'])

with open(f"propose_results/{file_name}_gather.json", 'w') as f:
    json.dump(gathered_clusters, f, indent=4)