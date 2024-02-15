#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# ===== qa lms =====
# python propose.py --config_file propose_configs/causallm_opt-350m-qa-ratemyprof-likeness.json
# python propose.py --config_file propose_configs/causallm_llama-2-7b-qa-ratemyprof-likeness.json --use_cache True
# python propose.py --config_file propose_configs/causallm_llama-2-7b-qa-ratemyprof-aspect.json --use_cache True
# python propose.py --config_file propose_configs/causallm_llama-2-7b-qa-ratemyprof-detailed.json --use_cache True
# python propose.py --config_file propose_configs/causallm_llama-2-7b-qa-ethos-reason.json --use_cache False
# python propose.py --config_file propose_configs/causallm_llama-2-7b-qa-ethos-hate.json --use_cache False
# python propose.py --config_file propose_configs/causallm_llama-2-7b-qa-feedback-reason.json --use_cache True
# python propose.py --config_file propose_configs/causallm_llama-2-7b-qa-feedback-detailed.json --use_cache False
# python propose.py --config_file propose_configs/causallm_llama-2-7b-qa-feedback-likeness.json --use_cache False
# python propose.py --config_file propose_configs/causallm_llama-2-7b-qa-yelp-likeness.json --use_cache True
# python propose.py --config_file propose_configs/causallm_llama-2-7b-qa-yelp-product.json --use_cache True
# python propose.py --config_file propose_configs/causallm_llama-2-7b-qa-yelp-aspect.json --use_cache True
# python propose.py --config_file propose_configs/causallm_llama-2-7b-qa-yelp-detailed.json --use_cache False
# python propose.py --config_file propose_configs/causallm_llama-2-7b-qa-fewrel.json --use_cache True
# python propose.py --config_file propose_configs/causallm_llama-2-7b-qa-fewnerd.json --use_cache True
python propose.py --config_file propose_configs/causallm_llama-2-7b-qa-fewevent.json --use_cache True