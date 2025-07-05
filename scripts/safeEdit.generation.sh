export CUDA_VISIBLE_DEVICES=0

ROOT=/root/autodl-tmp/Detoxify
EXP_ROOT=$ROOT/results

model_path=/root/autodl-tmp/checkpoints/llava-v1.5-7b #e.g. Llama-2-7b-chat-hf

python3  $ROOT/scripts/safeEdit_generation.py \
    --model-path $model_path  \
    --output-file $EXP_ROOT/safeEdit_generation/llava-v1.5-7b_30sample.jsonl \
    --fewshot-prompting True