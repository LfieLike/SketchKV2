export HF_ENDPOINT=https://hf-mirror.com
bash scripts/long_test.sh 1 16 16 666 666  NousResearch/Llama-2-7b-chat-hf

python eval_long_bench.py Llama-2-7b-hf_4096_16bits_group32_residual32