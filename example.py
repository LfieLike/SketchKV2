# LLaMA model with KIVI
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
import warnings
warnings.filterwarnings("ignore")
import torch
import random
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset

# For reproducibility
random.seed(0)
torch.manual_seed(0)

config = LlamaConfig.from_pretrained("/root/model/Llama-2-7b-hf")


model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path='/root/model/Llama-2-7b-hf',
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
).cuda()

enc = AutoTokenizer.from_pretrained(
    '/root/model/Llama-2-7b-hf', 
    use_fast=False, 
    trust_remote_code=True, 
    tokenizer_type='llama')

dataset = load_dataset('gsm8k', 'main')

prompt = ''
for i in range(5):
    prompt += 'Question: ' + dataset['train'][i]['question'] + '\nAnswer: ' + dataset['train'][i]['answer'] + '\n'
prompt += "Question: John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?"
inputs = enc(prompt, return_tensors="pt").input_ids.cuda()

output = model.generate(inputs, max_new_tokens=96)

config_str = f"# prompt tokens: {inputs.shape[1]} # generate tokens: {output.shape[1]}"

print(prompt + "\n" + "=" * 10 + f'\n{config_str}\n' + "=" * 10 + "\nKiVi Output:")
print(enc.decode(output[0].tolist()[inputs.shape[1]:], skip_special_tokens=True))