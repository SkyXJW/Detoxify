export CUDA_VISIBLE_DEVICES=4

ROOT=/home/xjg/myTruthX
EXP_ROOT=$ROOT/results
model_path=/home/xjg/checkpoints/mistral-7b-v0.1 # e.g. Llama-2-7b-chat-hf

# two-fold validation
# 实际实验过程中并没有用到双重验证，这里提供的truthx_model1、truthx_model2仅仅是为了附和已有的代码参数
truthx_model1=/home/xjg/myTruthX/truthx_models/mistral-7b-v0.1/mistral_truthx_model_100epoch_500sample.pt
truthx_model2=/home/xjg/myTruthX/truthx_models/mistral-7b-v0.1/mistral_truthx_model_100epoch_500sample.pt

strength=1.0
layers=10

python3  $ROOT/scripts/safeEdit_generation_truthx.py \
    --model-path $model_path \
    --truthx-model $truthx_model1 \
    --truthx-model2 $truthx_model2 \
    --data-yaml data/truthx/TruthfulQA/truthfulqa_data_fold1.yaml \
    --edit-strength $strength --top-layers $layers  \
    --output-file $EXP_ROOT/safeEdit_generation_truthx/mistral-7b-v0.1_500sample_DS_1.5delta.jsonl \
    # --fewshot-prompting True \
    # --two-fold False
    # --output-file $EXP_ROOT/safeEdit_generation_truthx/test.jsonl