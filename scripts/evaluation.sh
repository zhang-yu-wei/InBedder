#!/bin/bash

export CUDA_VISIBLE_DEVICES=[CUDA_VISIBLE_DEVICES]
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# ===== pre-trained lms =====
# python evaluation.py --config_file configs/causallm_llama-2-7b-chat-e5-large-v2-pretrain.json --last_layer_only False --overwrite_results True
# python evaluation.py --config_file configs/causallm_llama-2-7b-chat-e5-large-v2-pretrain-multiple.json --last_layer_only True --overwrite_results True --output_value "re_encode"
# python evaluation.py --config_file configs/causallm_llama-2-7b-chat-e5-large-v2-pretrain-filter.json --last_layer_only True --overwrite_results True
# python evaluation.py --config_file configs/causallm_llama-2-13b-chat-e5-large-v2-pretrain.json --last_layer_only True --overwrite_results True
# python evaluation.py --config_file configs/causallm_llama-2-13b-chat-e5-large-v2-pretrain-filter.json --last_layer_only True --overwrite_results True
# python evaluation.py --config_file configs/seq2seqlm_flan-t5-xl-e5-large-v2-pretrain.json --last_layer_only True --overwrite_results True
# python evaluation.py --config_file configs/seq2seqlm_flan-t5-xl-e5-large-v2-pretrain-filter.json --last_layer_only True --overwrite_results True

# ===== alpaca lms =====
# python evaluation.py --config_file configs/causallm_opt-350m-alpaca.json --last_layer_only True --overwrite_results True
# python evaluation.py --config_file configs/causallm_opt-1.3b-alpaca.json --last_layer_only True --overwrite_results True
# python evaluation.py --config_file configs/causallm_opt-2.7b-alpaca.json --last_layer_only True --overwrite_results True
# python evaluation.py --config_file configs/maskedlm_roberta-large-alpaca.json --last_layer_only True --overwrite_results True

# ===== qa lms =====
# python evaluation.py --config_file configs/causallm_opt-350m-qa.json --last_layer_only True --overwrite_results False
# python evaluation.py --config_file configs/causallm_opt-1.3b-qa.json --last_layer_only True --overwrite_results False
# python evaluation.py --config_file configs/causallm_opt-2.7b-qa.json --last_layer_only True --overwrite_results False
# python evaluation.py --config_file configs/maskedlm_roberta-large-qa.json --last_layer_only True --overwrite_results False
# python evaluation.py --config_file configs/causallm_llama-2-7b-qa-ml=3.json --last_layer_only False --overwrite_results True
# python evaluation.py --config_file configs/causallm_llama-2-7b-e5-large-v2-qa-ml=3-multiple.json --last_layer_only True --overwrite_results True --output_value "re_encode"

# ===== qa noprocess =====
# python evaluation.py --config_file configs/causallm_llama-2-7b-qa-noprocess.json --last_layer_only True --overwrite_results True

# ===== embedder =====
# python evaluation.py --config_file configs/sentence-transformers_e5-large-v2.json --overwrite_results True
# python evaluation.py --config_file configs/sentence-transformers_instructor-large.json --overwrite_results True

# ===== robustness =====
# python robust_evaluation.py --config_file configs/causallm_llama-2-7b-qa-ml=3.json --last_layer_only True --overwrite_results True
# python robust_evaluation.py --config_file configs/sentence-transformers_instructor-large.json --overwrite_results True
# python robust_evaluation.py --config_file configs/causallm_llama-2-7b-chat-e5-large-v2-pretrain.json --last_layer_only True --overwrite_results True