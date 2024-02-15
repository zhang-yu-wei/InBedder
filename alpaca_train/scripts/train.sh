#!/bin/bash

# training small models
# 355M
# model_name_or_path="roberta-large"
# 350M
# model_name_or_path="facebook/opt-350m"
# 738M
# model_name_or_path="google/t5-v1_1-large"
# 783M
# model_name_or_path="google/flan-t5-large"

# for model_name_or_path in "roberta-large" "facebook/opt-350m" "google/flan-t5-large" "facebook/opt-1.3b"
for model_name_or_path in "google/t5-v1_1-large"
do
    
    if [[ $model_name_or_path == *"/"* ]]; then
        IFS='/' read -ra model_name <<< "$model_name_or_path"
        model_name=${model_name[1]}
    else
        model_name=$model_name_or_path
    fi

    if [[ $model_name_or_path == *"t5-large"* ]]; then
        train_batch_size=8
        gradient_accumulation_steps=8
        learning_rate=1e-4
    
    elif [[ $model_name_or_path == *"1.3b"* ]]; then
        train_batch_size=8
        gradient_accumulation_steps=8
        learning_rate=2e-5
    else
        train_batch_size=16
        gradient_accumulation_steps=4
        learning_rate=2e-5
    fi

    export CUDA_VISIBLE_DEVICES=2

    # ===== alpaca =====
    # python train.py \
    #     --model_name_or_path ${model_name_or_path} \
    #     --data_path "data/alpaca_data_processed.json" \
    #     --output_dir "checkpoints/alpaca_${model_name}" \
    #     --num_train_epochs 3 \
    #     --per_device_train_batch_size $train_batch_size \
    #     --gradient_accumulation_steps $gradient_accumulation_steps \
    #     --evaluation_strategy "no" \
    #     --save_strategy "steps" \
    #     --save_steps 2000 \
    #     --save_total_limit 1 \
    #     --learning_rate $learning_rate \
    #     --weight_decay 0. \
    #     --warmup_ratio 0.03 \
    #     --lr_scheduler_type "cosine" \
    #     --logging_steps 1 \
    #     --tf32 True \
    #     --overwrite_output_dir True \
    #     --run_name "${model_name}-alpaca"

    # ===== qa =====
    python train.py \
        --model_name_or_path ${model_name_or_path} \
        --data_path "data/qa_collection_processed.json" \
        --output_dir "checkpoints/qa_${model_name}" \
        --num_train_epochs 1 \
        --per_device_train_batch_size $train_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 2000 \
        --save_total_limit 1 \
        --learning_rate $learning_rate \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --overwrite_output_dir True \
        --run_name "${model_name}-qa"

    # ===== alpaca+qa =====
    # python train.py \
    #     --model_name_or_path ${model_name_or_path} \
    #     --data_path "data/alpaca+qa_data_processed.json" \
    #     --output_dir "checkpoints/alpaca+qa_${model_name}" \
    #     --num_train_epochs 1 \
    #     --per_device_train_batch_size $train_batch_size \
    #     --gradient_accumulation_steps $gradient_accumulation_steps \
    #     --evaluation_strategy "no" \
    #     --save_strategy "steps" \
    #     --save_steps 2000 \
    #     --save_total_limit 1 \
    #     --learning_rate $learning_rate \
    #     --weight_decay 0. \
    #     --warmup_ratio 0.03 \
    #     --lr_scheduler_type "cosine" \
    #     --logging_steps 1 \
    #     --tf32 True \
    #     --overwrite_output_dir True \
    #     --run_name "${model_name}-alpaca+qa"

    # ===== qa+compression =====
    # python train.py \
    #     --model_name_or_path ${model_name_or_path} \
    #     --data_path "data/qa+compression_data.json" \
    #     --output_dir "checkpoints/qa+compression_${model_name}" \
    #     --num_train_epochs 1 \
    #     --per_device_train_batch_size $train_batch_size \
    #     --gradient_accumulation_steps $gradient_accumulation_steps \
    #     --evaluation_strategy "no" \
    #     --save_strategy "steps" \
    #     --save_steps 2000 \
    #     --save_total_limit 1 \
    #     --learning_rate $learning_rate \
    #     --weight_decay 0. \
    #     --warmup_ratio 0.03 \
    #     --lr_scheduler_type "cosine" \
    #     --logging_steps 1 \
    #     --tf32 True \
    #     --overwrite_output_dir True \
    #     --run_name "${model_name}-qa+compression"
done