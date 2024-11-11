#meta-llama/Llama-2-7b-hf
#huggyllama/llama-7b

gpuid=$1
k_bits=$2
v_bits=$3
group_size=$4
residual_length=$5
tasks=$6
model=$7

model_name="${model#*/}"
echo "$model_name"
CUDA_VISIBLE_DEVICES=$gpuid python run_lm_eval_harness.py --model_name_or_path $model \
    --tasks $tasks \
    --cache_dir ./cached_models \
    --k_bits $k_bits \
    --v_bits $v_bits \
    --group_size $group_size \
    --residual_length $residual_length 

#  accelerate launch -m  lmeval_test.sh 0,1,2,3 2 2 32 32 gsm8k /root/model/Llama-2-7b-hf

python run_lm_eval_harness.py --model_name_or_path /root/model/Llama-2-7b-hf \
    --tasks gsm8k \
    --cache_dir ./cached_models \
    --k_bits 2 \
    --v_bits 2 \
    --group_size 32 \
    --residual_length 32 