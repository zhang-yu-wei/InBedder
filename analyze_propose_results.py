import os
import json
import torch
import numpy as np
from safetensors import safe_open

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM

file_to_analyze = "Llama-2-7b-hf-qa-ml=20_fst_gen_layer_32_event-type_seed=42"

with open(f"propose_results/{file_to_analyze}.json", 'r') as f:
    data = json.load(f)

generations = {k: [d[0].split("### Response:")[1] for d in data[k]] for k in data}
documents = {k: ' '.join(generations[k]) for k in generations} # each document is the summation of all the words from a cluster
doc_keys = list(documents.keys())
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform([documents[k] for k in doc_keys])
feature_names = vectorizer.get_feature_names_out()

avg_features = features.A.mean(0)
cluster_keywords = {}
for i, k in enumerate(doc_keys):
    feat = features[i, :].A.squeeze() - avg_features
    cluster_keywords[k] = feature_names[feat.argsort()[::-1][:20]].tolist()

all_embeddings = {}
with safe_open(os.path.join(f"propose_results/{file_to_analyze}", "embeds.safetensors"), framework="pt", device="cpu") as f:
    for k in f.keys():
        all_embeddings[k] = f.get_tensor(k)

embeddings = all_embeddings["last_ppt_layer_32"]

v_dataset = embeddings.mean(0)
v_cluster = {k: torch.stack([embeddings[int(d[1]), :] for d in data[k]]).mean(0) for k in data}

v_cluster_proc = {k: v_cluster[k] - v_dataset for k in v_cluster}

tokenizer = AutoTokenizer.from_pretrained("alpaca_train/checkpoints/qa_Llama-2-7b-hf", padding_side='left', truncation_side="left")

model = AutoModelForCausalLM.from_pretrained("alpaca_train/checkpoints/qa_Llama-2-7b-hf",
                                            torch_dtype=torch.bfloat16,
                                            low_cpu_mem_usage=True,
                                            trust_remote_code=True,
                                            use_flash_attention_2=True,
                                            device_map="cpu") # flash attention 2 is a speed-up implementation of attention module

probs = {k: model.lm_head(v_cluster_proc[k]) for k in v_cluster_proc}
top_token_ids = {k: torch.topk(probs[k], k=20).indices for k in probs}
top_tokens = {k: tokenizer.convert_ids_to_tokens(top_token_ids[k]) for k in probs}

with open(f"propose_results/{file_to_analyze}_analyze.json", 'w') as f:
    json.dump({
        "generation_tfidf": cluster_keywords,
        "contrastive_decoding": top_tokens,
        }, f, indent=4)