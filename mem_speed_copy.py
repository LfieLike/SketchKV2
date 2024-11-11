import torch
import os
from models.modeling_llama import LlamaForCausalLM
from transformers import LlamaConfig, AutoTokenizer
import time

K_BITS = 2
V_BITS = 2
GROUP_SIZE = 32
RESIDUAL_LENGTH = 128
BATCH_SIZE = 96
PATH_TO_YOUR_SAVE_DIR = './cached_models'

model_name_or_path = '/root/model/Llama-2-7b-hf'
config = LlamaConfig.from_pretrained(model_name_or_path)
config.k_bits = K_BITS # current support 2/4 bit for KV Cache
config.v_bits = V_BITS # current support 2/4 bit for KV Cache
config.group_size = GROUP_SIZE
config.residual_length = RESIDUAL_LENGTH # the number of recent fp16 tokens
CACHE_DIR = PATH_TO_YOUR_SAVE_DIR

if K_BITS < 16 and V_BITS < 16:
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
else:
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, 
    use_fast=False, 
    trust_remote_code=True, 
    tokenizer_type='llama')

model.cuda().eval()

context = []
batch_size = BATCH_SIZE
prompt_lenth = 160
output_length = 20
num_repeats = 3
for _ in range(batch_size):
    string = 't,' * (prompt_lenth // 2)
    context.append(string[:-1])
inputs = tokenizer(context, return_tensors="pt").to('cuda')
print("1111",torch.cuda.max_memory_allocated()/ 1024 ** 3,"GB")
input_ids = inputs['input_ids']
print(f"bs: {batch_size}, seqlen: {input_ids.shape[1]}+{output_length}\nmodel:{model_name_or_path}")
torch.cuda.reset_peak_memory_stats()
print("3333",torch.cuda.max_memory_allocated()/ 1024 ** 3,"GB")
with torch.no_grad():
    print("6666",torch.cuda.max_memory_allocated()/ 1024 ** 3,"GB")
    torch.cuda.synchronize()
    print("5555",torch.cuda.max_memory_allocated()/ 1024 ** 3,"GB")
    st = time.time()
    for i in range(num_repeats):
        print("4444",torch.cuda.max_memory_allocated()/ 1024 ** 3,"GB")
        outputs = model.generate(**inputs, max_new_tokens=output_length)
        print("2222",torch.cuda.max_memory_allocated()/ 1024 ** 3,"GB")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    current_memory = torch.cuda.memory_allocated()
    print("current_memory",current_memory)
    print(f'used time: {(time.time() - st) / num_repeats * 1000} ms')
    used_mem = torch.cuda.max_memory_allocated()
    print(f'peak mem: {used_mem / 1024 ** 3} GB')