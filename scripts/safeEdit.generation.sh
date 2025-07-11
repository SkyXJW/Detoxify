export CUDA_VISIBLE_DEVICES=0

ROOT=/home/xjg/myTruthX
EXP_ROOT=$ROOT/results

model_path=/home/xjg/checkpoints/mistral-7b-v0.1 #e.g. Llama-2-7b-chat-hf

python3  $ROOT/scripts/safeEdit_generation.py \
    --model-path $model_path  \
    --output-file $EXP_ROOT/safeEdit_generation/mistral-7b-v0.1_500sample.jsonl \
    # --fewshot-prompting True