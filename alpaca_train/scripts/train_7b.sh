#!/bin/bash

# training large models
# "facebook/opt-6.7b"

# YOU NEED TO REQUEST FOR ACCESS FIRST, SEE https://huggingface.co/meta-llama/Llama-2-7b-hf
for model_name_or_path in "/data/shared/llama-hf/llama-2-7b-hf"
do
    
    if [[ $model_name_or_path == *"/"* ]]; then
        IFS='/' read -ra model_name <<< "$model_name_or_path"
        model_name=${model_name[4]}
    else
        model_name=$model_name_or_path
    fi
    echo $model_name
    
    learning_rate=2e-5

    export CUDA_VISIBLE_DEVICES=[CUDA_VISIBLE_DEVICES]

    # ===== alpaca =====
    # torchrun --nproc_per_node=2 --master_port=1234 train.py \
    #     --model_name_or_path ${model_name_or_path} \
    #     --data_path "data/alpaca_data_processed.json" \
    #     --output_dir "checkpoints/alpaca_${model_name}" \
    #     --num_train_epochs 3 \
    #     --per_device_train_batch_size 16 \
    #     --gradient_accumulation_steps 2 \
    #     --evaluation_strategy "no" \
    #     --save_strategy "steps" \
    #     --save_steps 2000 \
    #     --save_total_limit 1 \
    #     --learning_rate $learning_rate \
    #     --weight_decay 0. \
    #     --warmup_ratio 0.03 \
    #     --lr_scheduler_type "cosine" \
    #     --logging_steps 1 \
    #     --bf16 True \
    #     --overwrite_output_dir True \
    #     --deepspeed "./configs/default_offload_opt_param.json" \
    #     --run_name "${model_name}-alpaca"

    # ===== qa =====
    torchrun --nproc_per_node=4 --master_port=1234 train.py \
        --model_name_or_path ${model_name_or_path} \
        --data_path "KomeijiForce/Inbedder-Pretrain-Data" \
        --output_dir "checkpoints/qa_${model_name}" \
        --num_train_epochs 1 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 2000 \
        --save_total_limit 1 \
        --learning_rate $learning_rate \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --bf16 True \
        --overwrite_output_dir True \
        --deepspeed "./configs/default_offload_opt_param.json" \
        --run_name "${model_name}-qa"

    # ===== qa ablation =====
    # torchrun --nproc_per_node=4 --master_port=1234 train.py \
    #     --model_name_or_path ${model_name_or_path} \
    #     --data_path "data/qa_collection.json" \
    #     --output_dir "checkpoints/qa_noprocess_${model_name}" \
    #     --num_train_epochs 1 \
    #     --per_device_train_batch_size 8 \
    #     --gradient_accumulation_steps 2 \
    #     --evaluation_strategy "no" \
    #     --save_strategy "steps" \
    #     --save_steps 2000 \
    #     --save_total_limit 1 \
    #     --learning_rate $learning_rate \
    #     --weight_decay 0. \
    #     --warmup_ratio 0.03 \
    #     --lr_scheduler_type "cosine" \
    #     --logging_steps 1 \
    #     --bf16 True \
    #     --overwrite_output_dir True \
    #     --deepspeed "./configs/default_offload_opt_param.json" \
    #     --run_name "${model_name}-qa-noprocess"
done