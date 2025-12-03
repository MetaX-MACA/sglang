python fused_moe_auto_tune_int4.py \
    --model /model/llm/DeepSeek/DeepSeek-R1-awq \
    --tp-size 8 \
    --n-share-experts-fusion 0 \
    --dtype int4_w4a16 \
    --stage stage1 \
    --tune

python fused_moe_auto_tune_int4.py \
    --model /model/llm/DeepSeek/DeepSeek-R1-awq \
    --tp-size 8 \
    --n-share-experts-fusion 0 \
    --dtype int4_w4a16 \
    --stage stage2 \
    --tune