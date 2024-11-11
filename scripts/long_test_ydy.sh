# model e.g.: meta-llama/Llama-2-7b-hf
# bash scripts/long_test.sh

# mistral-community/Mistral-7B-Instruct-v0.3
# NousResearch/Llama-2-7b-chat-hf
# lmsys/longchat-7b-v1.5-32k
# NousResearch/Meta-Llama-3-8B-Instruct

gpuid=1
model=NousResearch/Llama-2-7b-chat-hf
method=kivi
e=0

CUDA_VISIBLE_DEVICES=$gpuid python pred_long_bench_ydy.py --model_name_or_path $model \
    --cache_dir /root/KIVI-lmeval/cached_models \
    --method $method\
    --e ${e}
