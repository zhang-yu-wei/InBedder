#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# ===== pre-trained lms =====
# python evaluation.py --config_file configs/causallm_llama-2-7b-chat-pretrain.json --last_layer_only True --overwrite_results False
# python evaluation.py --config_file configs/causallm_llama-2-7b-chat-pretrain-whiten.json --overwrite_results True
# python evaluation.py --config_file configs/causallm_llama-2-7b-chat-pretrain-filter.json --last_layer_only True --overwrite_results True
# python evaluation.py --config_file configs/causallm_llama-2-7b-chat-e5-large-v2-pretrain.json --last_layer_only True --overwrite_results True
# python evaluation.py --config_file configs/causallm_llama-2-7b-chat-e5-large-v2-pretrain-filter.json --last_layer_only True --overwrite_results True
# python evaluation.py --config_file configs/causallm_llama-2-7b-chat-e5-large-v2-pretrain-ppl=07.json --last_layer_only True --overwrite_results False
# python evaluation.py --config_file configs/causallm_llama-2-13b-chat-pretrain.json --last_layer_only True --overwrite_results False
# python evaluation.py --config_file configs/causallm_llama-2-13b-chat-e5-large-v2-pretrain.json --last_layer_only True --overwrite_results True
# python evaluation.py --config_file configs/causallm_llama-2-13b-chat-pretrain-whiten.json --overwrite_results True
# python evaluation.py --config_file configs/causallm_llama-2-13b-chat-pretrain-filter.json --last_layer_only True --overwrite_results True
# python evaluation.py --config_file configs/causallm_llama-2-13b-chat-e5-large-v2-pretrain-filter.json --last_layer_only True --overwrite_results True
# python evaluation.py --config_file configs/causallm_opt-350m-pretrain.json --overwrite_results True
# python evaluation.py --config_file configs/maskedlm_roberta-large-pretrain.json --overwrite_results True
# python evaluation.py --config_file configs/seq2seqlm_t5-v1_1-large-pretrain.json --overwrite_results True
# python evaluation.py --config_file configs/seq2seqlm_flan-t5-large-pretrain.json --last_layer_only True --overwrite_results False
# python evaluation.py --config_file configs/seq2seqlm_flan-t5-xl-e5-large-v2-pretrain.json --last_layer_only True --overwrite_results True
# python evaluation.py --config_file configs/seq2seqlm_flan-t5-large-e5-large-v2-pretrain-filter.json --last_layer_only True --overwrite_results False
# python evaluation.py --config_file configs/seq2seqlm_flan-t5-xl-e5-large-v2-pretrain-filter.json --last_layer_only True --overwrite_results True

# ===== alpaca lms =====
# python evaluation.py --config_file configs/causallm_opt-350m-alpaca.json --last_layer_only True --overwrite_results True
# python evaluation.py --config_file configs/causallm_opt-350m-e5-large-v2-alpaca.json --overwrite_results True
# python evaluation.py --config_file configs/causallm_opt-1.3b-alpaca.json --last_layer_only True --overwrite_results True
# python evaluation.py --config_file configs/causallm_opt-2.7b-alpaca.json --last_layer_only False --overwrite_results False
# python evaluation.py --config_file configs/maskedlm_roberta-large-alpaca.json --last_layer_only True --overwrite_results True
# python evaluation.py --config_file configs/seq2seqlm_t5-v1_1-large-alpaca.json --overwrite_results True
# python evaluation.py --config_file configs/seq2seqlm_flan-t5-large-alpaca.json --overwrite_results True

# ===== qa lms =====
# python evaluation.py --config_file configs/causallm_opt-350m-qa.json --last_layer_only True --overwrite_results False
# python evaluation.py --config_file configs/causallm_opt-350m-e5-large-v2-qa.json --overwrite_results True
# python evaluation.py --config_file configs/causallm_opt-1.3b-qa.json --last_layer_only True --overwrite_results False
# python evaluation.py --config_file configs/causallm_opt-2.7b-qa.json --last_layer_only False --overwrite_results False
# python evaluation.py --config_file configs/causallm_opt-6.7b-qa.json --last_layer_only True --overwrite_results False
# python evaluation.py --config_file configs/maskedlm_roberta-large-qa.json --last_layer_only True --overwrite_results False
# python evaluation.py --config_file configs/maskedlm_roberta-large-e5-large-v2-qa.json --overwrite_results True
# python evaluation.py --config_file configs/maskedlm_roberta-large-qa-ml=10.json --overwrite_results True
# python evaluation.py --config_file configs/seq2seqlm_t5-v1_1-large-qa.json --last_layer_only True --overwrite_results True
# python evaluation.py --config_file configs/seq2seqlm_flan-t5-large-qa.json --last_layer_only True --overwrite_results False
# python evaluation.py --config_file configs/seq2seqlm_flan-t5-xl-qa.json --last_layer_only True --overwrite_results True
# python evaluation.py --config_file configs/causallm_llama-2-7b-qa.json --last_layer_only True --overwrite_results True
# python evaluation.py --config_file configs/causallm_llama-2-7b-qa-ml=3.json --last_layer_only True --output_value avg_gen_layer_32 --overwrite_results True
# python evaluation.py --config_file configs/peftcausallm_llama-2-7b-qa.json --last_layer_only True --overwrite_results False
# python evaluation.py --config_file configs/peftcausallm_llama-2-7b-qa-2e-4.json --last_layer_only True --overwrite_results False
# python evaluation.py --config_file configs/bowcelm_opt-350m-qa.json --overwrite_results True

# ===== qa noprocess =====
# python evaluation.py --config_file configs/causallm_llama-2-7b-qa-noprocess.json --last_layer_only True --overwrite_results False

# ===== qa+compression lms =====
# python evaluation.py --config_file configs/causallm_opt-350m-qa+compression.json --overwrite_results True
# python evaluation.py --config_file configs/causallm_opt-1.3b-qa+compression.json --overwrite_results False
# python evaluation.py --config_file configs/maskedlm_roberta-large-qa+compression.json --overwrite_results True
# python evaluation.py --config_file configs/seq2seqlm_flan-t5-large-qa+compression.json --overwrite_results True

# ===== alpaca+qa lms =====
# python evaluation.py --config_file configs/causallm_opt-350m-alpaca+qa.json --overwrite_results True
# python evaluation.py --config_file configs/maskedlm_roberta-large-alpaca+qa.json --overwrite_results True
# python evaluation.py --config_file configs/seq2seqlm_flan-t5-large-alpaca+qa.json --overwrite_results True

# ===== embedder =====
# python evaluation.py --config_file configs/sentence-transformers_e5-large-v2.json --overwrite_results False
# python evaluation.py --config_file configs/sentence-transformers_instructor-large.json --overwrite_results False

# ===== robustness =====
# python robust_evaluation.py --config_file configs/causallm_llama-2-7b-qa.json --last_layer_only True --overwrite_results False
# python robust_evaluation.py --config_file configs/sentence-transformers_instructor-large.json --overwrite_results False
# python robust_evaluation.py --config_file configs/causallm_llama-2-7b-chat-e5-large-v2-pretrain.json --last_layer_only True --overwrite_results False