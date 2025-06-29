export CUDA_VISIBLE_DEVICES=7

ROOT=/home/xjg/myTruthX
EXP_ROOT=$ROOT/results
model_path=/home/xjg/checkpoints/llava-v1.5-7b # e.g. Llama-2-7b-chat-hf

# two-fold validation
truthx_model1=truthx_models/llava-v1.5-7b/llava_truthx_model_100epoch.pt
truthx_model2=truthx_models/llava-v1.5-7b/llava_truthx_model_100epoch.pt

strength=1.0
layers=10

python3  $ROOT/scripts/safeEdit_generation_truthx.py \
    --model-path $model_path \
    --truthx-model $truthx_model1 \
    --truthx-model2 $truthx_model2 \
    --data-yaml data/truthx/TruthfulQA/truthfulqa_data_fold1.yaml \
    --edit-strength $strength --top-layers $layers  \
    --fewshot-prompting True \
    --output-file $EXP_ROOT/safeEdit_generation_truthx/llava-v1.5-7b.jsonl \
    # --two-fold False
    # --output-file $EXP_ROOT/safeEdit_generation_truthx/test.jsonl