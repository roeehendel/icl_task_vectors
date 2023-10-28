#!/bin/bash

# This script is used to convert the original LLAMA model weights to the HuggingFace format.
# Also, it can be used to convert the regular model weights to the vicuna weights.

convert_llama_to_huggingface() {
    local input_dir=$1
    local model_size=$2
    local output_dir=$3
    
    python $TRANSFORMER_DIR/src/transformers/models/llama/convert_llama_weights_to_hf.py \
        --input_dir $input_dir --model_size $model_size --output_dir $output_dir
}

convert_huggingface_to_vicuna() {
    local base_model_path=$1
    local target_model_path=$2
    local delta_path=$3
    
    python -m fastchat.model.apply_delta \
        --base-model-path $base_model_path \
        --target-model-path $target_model_path \
        --delta-path $delta_path
}

case $1 in
    "llama_to_huggingface_7B")
        convert_llama_to_huggingface "$LLAMA_DIR" "7B" "$LLAMA_DIR/huggingface/7B"
        ;;
    "llama_to_huggingface_13B")
        convert_llama_to_huggingface "$LLAMA_DIR" "13B" "$LLAMA_DIR/huggingface/13B"
        ;;
    "llama_to_huggingface_30B")
        convert_llama_to_huggingface "$LLAMA_DIR" "30B" "$LLAMA_DIR/huggingface/30B"
        ;;
    "llama_to_huggingface_65B")
        convert_llama_to_huggingface "$LLAMA_DIR" "65B" "$LLAMA_DIR/huggingface/65B"
        ;;
    "huggingface_to_vicuna_7B")
        convert_huggingface_to_vicuna "$LLAMA_DIR/huggingface/7B" "$LLAMA_DIR/vicuna/7B" "lmsys/vicuna-7b-delta-v1.1"
        ;;
    "huggingface_to_vicuna_13B")
        convert_huggingface_to_vicuna "$LLAMA_DIR/huggingface/13B" "$LLAMA_DIR/vicuna/13B" "lmsys/vicuna-13b-delta-v1.1"
        ;;
    "huggingface_to_stable_vicuna_13B")
        convert_huggingface_to_vicuna "$LLAMA_DIR/huggingface/13B" "$LLAMA_DIR/stable-vicuna/13B" "CarperAI/stable-vicuna-13b-delta"
        ;;
    *)
        echo "Invalid conversion option. Please provide a valid conversion."
        exit 1
        ;;
esac
