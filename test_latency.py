import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import numpy as np
from models.modeling_llama import LlamaForCausalLM
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig,LlamaConfig
from transformers.generation.utils import GenerationConfig
import torch.multiprocessing as mp
import pandas as pd
import tqdm
import torch
from loguru import logger
import json
from torch.profiler import profile, record_function, ProfilerActivity
# NOTE: We use Llama2-7b to benchmark the latency.

def main():
    os.environ["SUBVEC"] = "2"
    os.environ["SUBBITS"] = "6"
    os.environ["MODE"] = "off"

    model_path = "NousResearch/Llama-2-7b-chat-hf"
    # model_path = "./pqcache/llama-32k"
    # model_path = "./pqcache/Mistral-32k"
    # print(model_path)

    file_name = "passkey_examples.jsonl"
    df = pd.read_json(file_name, lines=True) # 读取文件

    # if config.compressor == "pq_search":
    #     initialize_objects(config, model=model_path)
    
    config = LlamaConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, 
                                        use_fast=False, 
                                        trust_remote_code=True, 
                                        tokenizer_type='llama')
    # from transformers import LlamaForCausalLM
    # model = LlamaForCausalLM.from_pretrained(
    #     pretrained_model_name_or_path=model_path,
    #     cache_dir="./cached_models",
    #     config=config,
    #     torch_dtype=torch.half,
    #     low_cpu_mem_usage=True,
    #     use_flash_attention_2=True,
    #     device_map="auto",
    # )
    # config.k_bits = 2 # KiVi currently support 2/4 K/V bits
    # config.v_bits = 2
    # config.group_size = 32 
    # config.residual_length = 128 # corresponding to the number of recent fp16 tokens
    # config.use_flash = True # use flash-attention with KiVi for long context inference

    # model = LlamaForCausalLM_KIVI.from_pretrained(
    #     pretrained_model_name_or_path=model_path,
    #     config=config,
    #     cache_dir="./cached_models",
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch.float16,
    # ).cuda()
    
    from models.modeling_llama_V2 import LlamaForCausalLM
    from models.cache_utils_V2 import SkectchCache,pruneCache

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        cache_dir="./cached_models",
        # config=config,
        torch_dtype=torch.half,
        low_cpu_mem_usage=True,
        use_flash_attention_2=True,
        device_map="auto",
    )
    model = model.half().eval()

    print("We are loading model", model_path)
    peak_allocated_memory = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved_memory = torch.cuda.max_memory_reserved() / 1024**2

    print(f"New peak allocated memory: {peak_allocated_memory:.2f} MB")
    print(f"New peak reserved memory: {peak_reserved_memory:.2f} MB")

    for line in open(file_name, "r"):
        example = json.loads(line)
        prompt_postfix = "What is the pass key? The pass key is "
        prompt = example["input"]+example["input"] + prompt_postfix
        input_ids = tokenizer([prompt]*1, return_tensors="pt").input_ids.cuda()
        print("maxlen",len(input_ids[0]))
        gen_max_token = 20
        for idx in range(5):
            for seqlen in tqdm.tqdm([2000,4000,6000,8000,12000,16000]):
                # 预热
                output = model.generate(
                                input_ids=input_ids[:, :seqlen],
                                attention_mask=None,
                                pad_token_id=tokenizer.eos_token_id,
                                max_new_tokens=1, 
                                num_beams=1,
                                do_sample=False,
                                temperature=1.0,
                                past_key_values = SkectchCache(device=model.device)
                            )[0]
            # for seqlen in [2000, 4000]:
                print("fuck",len(input_ids[0]))
                begin = time.perf_counter()
                for i in range(5):
                    # past_key_values = SkectchCache()
                    torch.cuda.reset_peak_memory_stats()
                    output = model.generate(
                                input_ids=input_ids[:, :seqlen],
                                attention_mask=None,
                                pad_token_id=tokenizer.eos_token_id,
                                max_new_tokens=1, 
                                num_beams=1,
                                do_sample=False,
                                temperature=1.0,
                                past_key_values = SkectchCache(device=model.device)
                            )[0]
                print(f"{output.flatten()[-1]} \r")
                end = time.perf_counter()
                ttft = (end - begin)/5
                
                time.sleep(2)
                
                begin = time.perf_counter()
                for i in range(5):
                    output = model.generate(
                                input_ids=input_ids[:, 40:seqlen+40],
                                attention_mask=None,
                                pad_token_id=tokenizer.eos_token_id,
                                max_new_tokens=2, 
                                num_beams=1,
                                do_sample=False,
                                temperature=1.0,
                                past_key_values = SkectchCache(device=model.device)
                            )[0]
                print(f"{output.flatten()[-1]} \r")
                end = time.perf_counter()
                tt2t = (end - begin)/5
                
                time.sleep(2)

                begin = time.perf_counter()
                for i in range(5):
                    output = model.generate(
                                input_ids=input_ids[:, 40:seqlen+40],
                                attention_mask=None,
                                pad_token_id=tokenizer.eos_token_id,
                                max_new_tokens=gen_max_token, 
                                num_beams=1,
                                do_sample=False,
                                temperature=1.0,
                                past_key_values = SkectchCache(device=model.device)
                            )[0]
                print(f"{output.flatten()[-1]}")
                print("len",len(output))
                end = time.perf_counter()
                decoding_elapsed = (end - begin)/5 - ttft
                
                
                # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                #     with record_function("LlamaAttention"):
                #         outputs = model.generate(
                #             input_ids=input_ids[:, :seqlen],
                #             attention_mask=None,
                #             pad_token_id=tokenizer.eos_token_id,
                #             max_new_tokens=1, 
                #             num_beams=1,
                #             do_sample=False,
                #             temperature=1.0,past_key_values = SkectchCache(device=model.device)
                #         )[0]
                # prof.export_chrome_trace(f"{seqlen}trace.json")
                # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
                # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
                
                
                print(f"Given input len is:{seqlen}, gen seq_len:{gen_max_token},"
                        f"ttft is {ttft},"
                        f"tt2t is {tt2t},"
                        f"decoding elasped:{decoding_elapsed},"
                        f"{decoding_elapsed / (gen_max_token - 1)} per decoding token.")
    del model
    logger.info(f"del objects done.")   
    exit()

if __name__ == "__main__":
    main()


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import time
# import numpy as np
# from models.modeling_llama import LlamaForCausalLM
# from models.llama_kivi import LlamaForCausalLM_KIVI
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig,LlamaConfig
# from transformers.generation.utils import GenerationConfig
# import torch.multiprocessing as mp
# import pandas as pd
# import tqdm
# import torch
# from loguru import logger
# import json
# # NOTE: We use Llama2-7b to benchmark the latency.

# def main():
#     os.environ["SUBVEC"] = "2"
#     os.environ["SUBBITS"] = "6"
#     os.environ["MODE"] = "off"

#     model_path = "NousResearch/Llama-2-7b-chat-hf"
#     # model_path = "./pqcache/llama-32k"
#     # model_path = "./pqcache/Mistral-32k"
#     # print(model_path)

#     file_name = "passkey_examples.jsonl"
#     df = pd.read_json(file_name, lines=True) # 读取文件

#     # if config.compressor == "pq_search":
#     #     initialize_objects(config, model=model_path)
    
#     config = LlamaConfig.from_pretrained(model_path)
#     tokenizer = AutoTokenizer.from_pretrained(model_path, 
#                                         use_fast=False, 
#                                         trust_remote_code=True, 
#                                         tokenizer_type='llama')
#     from transformers import LlamaForCausalLM
#     model = LlamaForCausalLM.from_pretrained(
#         pretrained_model_name_or_path=model_path,
#         cache_dir="./cached_models",
#         config=config,
#         torch_dtype=torch.half,
#         low_cpu_mem_usage=True,
#         use_flash_attention_2=True,
#         device_map="auto",
#     )
# config.k_bits = 2 # KiVi currently support 2/4 K/V bits
# config.v_bits = 2
# config.group_size = 32 
# config.residual_length = 32 # corresponding to the number of recent fp16 tokens
# config.use_flash = True # use flash-attention with KiVi for long context inference
# # CACHE_DIR = "/scratch/cached_model"

# model = LlamaForCausalLM_KIVI.from_pretrained(
#     pretrained_model_name_or_path="lmsys/vicuna-7b-v1.5-16k",
#     config=config,
#     cache_dir="./cached_models",
#     low_cpu_mem_usage=True,
#     torch_dtype=torch.float16,
# ).cuda()
#     # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast = True, config = config)
#     # model = VQLlamaForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True)
#     model = model.half().eval()

#     print("We are loading model", model_path)


#     for line in open(file_name, "r"):
#         example = json.loads(line)
#         prompt_postfix = "What is the pass key? The pass key is "
#         prompt = example["input"] + prompt_postfix
#         input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
#         print("maxlen",len(input_ids[0]))
#         gen_max_token = 20
#         for idx in range(5):
#             for seqlen in tqdm.tqdm([2000, 4000, 8000, 16000]):
#             # for seqlen in [2000, 4000]:
#                 begin = time.perf_counter()
#                 output = model.generate(
#                             input_ids=input_ids[:, :seqlen],
#                             attention_mask=None,
#                             pad_token_id=tokenizer.eos_token_id,
#                             max_new_tokens=1, 
#                             num_beams=1,
#                             do_sample=False,
#                             temperature=1.0,
#                         )[0]
#                 print(f"{output.flatten()[-1]} \r")
#                 end = time.perf_counter()
#                 ttft = end - begin
                
#                 time.sleep(2)
                
#                 begin = time.perf_counter()
#                 output = model.generate(
#                             input_ids=input_ids[:, :seqlen],
#                             attention_mask=None,
#                             pad_token_id=tokenizer.eos_token_id,
#                             max_new_tokens=2, 
#                             num_beams=1,
#                             do_sample=False,
#                             temperature=1.0,
#                         )[0]
#                 print(f"{output.flatten()[-1]} \r")
#                 end = time.perf_counter()
#                 tt2t = end - begin
                
#                 time.sleep(2)

#                 begin = time.perf_counter()
#                 output = model.generate(
#                             input_ids=input_ids[:, :seqlen],
#                             attention_mask=None,
#                             pad_token_id=tokenizer.eos_token_id,
#                             max_new_tokens=gen_max_token, 
#                             num_beams=1,
#                             do_sample=False,
#                             temperature=1.0,
#                         )[0]
#                 print(f"{output.flatten()[-1]}")
#                 end = time.perf_counter()
#                 decoding_elapsed = end - begin - ttft
#                 print(f"Given input len is:{seqlen}, gen seq_len:{gen_max_token},"
#                         f"ttft is {ttft},"
#                         f"tt2t is {tt2t},"
#                         f"decoding elasped:{decoding_elapsed},"
#                         f"{decoding_elapsed / (gen_max_token - 1)} per decoding token.")
#     del model
#     logger.info(f"del objects done.")   
#     exit()

# if __name__ == "__main__":
#     main()
