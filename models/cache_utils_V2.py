from typing import Any, Dict, List, Optional, Tuple
import torch.nn.functional as F
import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from quant.myquant import ontbitquant,onebitgemv,extract_sign_and_compress,my_scatter_add,mydequant,my_resduialquant,my_value_quant,my_value_dequant
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple
import torch.nn.functional as F
import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from quant.myquant import ontbitquant,onebitgemv,extract_sign_and_compress,my_scatter_add,mydequant,my_resduialquant,my_value_quant,my_value_dequant
# from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
batch_size = 1
def get_tensor_mem(tensor):
    tensor_memory_size = tensor.element_size() * tensor.numel()

    # 转换为 MB
    tensor_memory_size_mb = tensor_memory_size / 1024**2
    print("fuck",tensor_memory_size_mb)
    return tensor_memory_size_mb

class KVquant_unit:
    def __init__(self):
        len = 16050
        self.quantized_tensor_KV = torch.empty((batch_size, 32, len, 128),device='cpu', dtype=torch.uint8 ,pin_memory=True)
        self.device = 'cpu'
        self.bit_num = 4
        self.head_dim = 128
        self.quant_param = torch.empty((batch_size, 32, len, 4),device='cpu', dtype=torch.float,pin_memory=True)
        self.first = True
        self.quantized_tensor_KV_gpu = None
        self.quant_param_gpu = None
        self.token_len = 0
        self.quantized_tensor_KV_cpu = None
        self.quant_param_cpu = None
        self.prepare_future =None
        self.event = torch.cuda.Event()
    def cat_new_cache(self,quantized_kv,quant_param):
        # print(quantized_kv)
        if self.first:
            self.token_len = quantized_kv.shape[-2]
            self.quantized_tensor_KV[:,:,:self.token_len,:].copy_(quantized_kv,non_blocking=True)
            self.quant_param[:,:,:self.token_len,:].copy_(quant_param,non_blocking=True)
            self.first = False
        else:
            self.token_len+=1
            self.quantized_tensor_KV[:,:,self.token_len-1:self.token_len,:].copy_(quantized_kv,non_blocking=True)
            self.quant_param[:,:,self.token_len-1:self.token_len,:].copy_(quant_param,non_blocking=True)
        # self.event.record()
        # if self.quant_param is None:
        #     self.quant_param = quant_param
        # else:
        #     self.quant_param = torch.cat((self.quant_param,quant_param),dim=-2)
    def prepare_for_next_gen_in_cpu(self,indices,stream):
        with torch.cuda.stream(stream):
            # self.event.synchronize()
            indices = indices.cpu()
            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, self.quantized_tensor_KV.shape[-1])
            # print(self.quantized_tensor_KV)
            self.quantized_tensor_KV_cpu = torch.gather(self.quantized_tensor_KV, 2, indices_expanded).contiguous().pin_memory()
            self.quant_param_cpu = torch.gather(self.quant_param, 2, indices.unsqueeze(-1).expand(-1,-1,-1,4)).contiguous().pin_memory()
    def select(self,indices):
        # indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        # # print(self.quantized_tensor_KV)
        # quantized_tensor_slip = torch.gather(self.quantized_tensor_KV_gpu, 2, indices_expanded).contiguous()
        # quant_param = torch.gather(self.quant_param_gpu, 2, indices.unsqueeze(-1).expand(-1,-1,-1,4)).contiguous()
        # print(quantized_tensor_slip,quant_param)
        # quant_param= torch.cat((zero_points_slip_K.view(-1,1),scales_slip_K.view(-1,1),zero_points_slip_V.view(-1,1),scales_slip_V.view(-1,1)),dim=-1).reshape(batch_size, head_size, num_elements*4)
        return self.quantized_tensor_KV_gpu,self.quant_param_gpu
    def prefetch(self,device):
        self.device = device
        if self.prepare_future is None:
            return
        self.prepare_future.result()
        self.quantized_tensor_KV_gpu = self.quantized_tensor_KV_cpu.to(device=device, non_blocking=True)
        # .to(device=self.device, non_blocking=non_blocking)
        # self.quantized_tensor_KV_gpu = self.quantized_tensor_KV[:, :,:self.token_len, :].to(device, non_blocking=non_blocking)
        self.quant_param_gpu = self.quant_param_cpu.to(device=device, non_blocking=True)
    def evcit(self):
        if self.quantized_tensor_KV is None:
            return
        self.quantized_tensor_KV_gpu =None
        self.quant_param_gpu = None
    def get_cpu_memory(self):
        memory_size = self.quantized_tensor_KV.element_size() * self.quantized_tensor_KV.numel() +self.quant_param.element_size() * self.quant_param.numel()
        # print(f"Memory size of the tensor: {memory_size / 1024**2:.2f} MB")
class quant_unit_V2:
    def __init__(self):
        self.quantized_tensor = None
        self.zero_points = None
        self.scales = None
        self.quant_param = None
        self.device = 'cpu'
        self.head_dim = 128
    def cat_new_cache_V2(self,quantized_tensor,quant_param):
        # print(unpact_tensor-quantized_tensor)
        if self.quantized_tensor is None:
            # self.head_dim = quantized_tensor.shape[-1]
            # quantized_tensor = self.pact_to_int8(quantized_tensor)
            self.quantized_tensor = quantized_tensor
            self.device = quantized_tensor.device
        else:
            # quantized_tensor = self.pact_to_int8(quantized_tensor)
            self.quantized_tensor = torch.cat((self.quantized_tensor,quantized_tensor),dim=-2)
        if self.quant_param is None:
            self.quant_param = quant_param
        else:
            self.quant_param = torch.cat((self.quant_param,quant_param),dim=-2)
class outline_quant_unit:
    def __init__(self,scales,zero_points,num_bits=8):
        self.qmin = 0.
        self.qmax = 2.**num_bits - 1.
        self.scales = scales
        self.zero_points = zero_points
class SkectchCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self,
                 layernum=32
                 ,buff_shape=(batch_size,32,16050,128),dtype=torch.float16,
                 device='cpu') -> None:
        super().__init__()
        self.layernum=layernum
        self.key_outline = [torch.empty(0) for _ in range(layernum)]
        self.key_cache_full = [torch.empty(0) for _ in range(layernum)]
        self.key_cache_index = [torch.empty(0) for _ in range(layernum)]
        self.key_cache_means = [torch.empty(0) for _ in range(layernum)]
        self.KV_cache_quant_unit = [KVquant_unit() for _ in range(layernum)]
        self.key_cache_channel_zp = {_:torch.empty(0) for _ in range(layernum)}
        self.key_cache_channel_scale = {_:torch.empty(0) for _ in range(layernum)}
        self.value_cache_full = {_:torch.empty(0) for _ in range(layernum)}
        self.value_cache_max = {_:torch.empty(0) for _ in range(layernum)}

        # self.value_cache_quant_unit = [quant_unit() for _ in range(layernum)]
        self.value_cache_quant_unit_V2 = [quant_unit_V2() for _ in range(layernum)]
        self.value_sink = {_:torch.empty(0) for _ in range(layernum)}
        self.key_sink = {_:torch.empty(0) for _ in range(layernum)}
        self.key_sign_new = {_:torch.empty(0) for _ in range(layernum)}
        self.attn_cache = {_:torch.empty(0) for _ in range(layernum)}
        self.outline_quant = {_:None for _ in range(layernum)}
        self.skectch_seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.window_size = 128
        self.sink_size = 4
        self.imp_rate = 0.3
        self.prefetch_stream={_:torch.cuda.Stream() for _ in range(layernum)}
        self.key_cache_temp=torch.empty(buff_shape,device=device, dtype=dtype)
        self.value_cache_temp=torch.empty(buff_shape,device=device, dtype=dtype)
        self.device = device
        torch._dynamo.mark_static_address(self.key_cache_temp)
        torch._dynamo.mark_static_address(self.value_cache_temp)
        # print(self.value_cache_temp)
        self.default_stream = torch.cuda.default_stream()
        self.topK = 12
        self.executor = ThreadPoolExecutor(max_workers=layernum+1)
    def resize_buff(self,device,num_key_value_heads=32,dtype=torch.float16,shapes=None):
        if self.key_cache_temp is not None:
            return
        shape=(1,num_key_value_heads,3250,128)
        # for i in range(2):
    def get_total_memory(self,layer_num):
        totla_size = 0
        for i in range(layer_num):
            totla_size += get_tensor_mem(self.key_outline[i])
            totla_size += get_tensor_mem(self.key_cache_full[i])
            totla_size += get_tensor_mem(self.key_cache_index[i])
            totla_size += get_tensor_mem(self.key_cache_means[i])
            totla_size += get_tensor_mem(self.key_cache_channel_zp[i])
            totla_size += get_tensor_mem(self.key_cache_channel_scale[i])
            totla_size += get_tensor_mem(self.value_cache_full[i])
            totla_size += get_tensor_mem(self.value_cache_max[i])
            totla_size += get_tensor_mem(self.value_sink[i])
            totla_size += get_tensor_mem(self.key_sign_new[i])
            totla_size += get_tensor_mem(self.attn_cache[i])
            totla_size += get_tensor_mem(self.outline_quant[i].scales)
            totla_size += get_tensor_mem(self.outline_quant[i].zero_points)
            totla_size += get_tensor_mem(self.KV_cache_quant_unit[i].quant_param)
            totla_size += get_tensor_mem(self.KV_cache_quant_unit[i].quantized_tensor_KV)
            totla_size += get_tensor_mem(self.value_cache_quant_unit_V2[i].quant_param)
            totla_size += get_tensor_mem(self.value_cache_quant_unit_V2[i].quantized_tensor)
        # totla_size+=get_tensor_mem(self.key_cache_temp)*2
        print(f"Tensor 显存占用: {totla_size:.2f} MB")
    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache_full[layer_idx], self.value_cache_full[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache_full[layer_idx], self.value_cache_full[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache_full)   
    def prefetch_layer(self,layer_idx:int,non_blocking=True):
        with torch.cuda.stream(self.prefetch_stream[layer_idx]):
            self.KV_cache_quant_unit[layer_idx].prefetch(device = self.device)
            # self.value_cache_quant_unit[layer_idx].prefetch()
    def evict_previous_layer(self, layer_idx: int,non_blocking=True):
        # with torch.cuda.stream(self.prefetch_stream[layer_idx]):
        self.KV_cache_quant_unit[layer_idx].evcit()
        # self.value_cache_quant_unit[layer_idx].evcit()
    def get_kv(self,
        layer_idx: int
    )-> Tuple[torch.Tensor, torch.Tensor]:
        device = self.attn_cache[layer_idx].device
        windows_size = self.window_size

        token_num = self.key_sign_new[layer_idx].shape[-2]
        
        mydequant(
            self.key_sign_new[layer_idx].contiguous(),
            self.key_outline[layer_idx].contiguous(),
            self.key_cache_temp,
            self.key_cache_means[layer_idx].contiguous(),
            self.key_cache_index[layer_idx],
            self.outline_quant[layer_idx].zero_points,
            self.outline_quant[layer_idx].scales,
            self.topK
        )
        # torch.cuda.default_stream().synchronize()
        key_value_quantized_data,quant_param = self.KV_cache_quant_unit[layer_idx].select(self.attn_cache[layer_idx])
        my_value_dequant(self.value_cache_quant_unit_V2[layer_idx].quantized_tensor,
                         self.value_cache_temp,
                         self.value_cache_max[layer_idx],
                         self.value_cache_quant_unit_V2[layer_idx].quant_param)
        # print((test_quant_value-quant_value).abs().mean())
        my_scatter_add(
            key_value_quantized_data.contiguous(),
            self.key_cache_temp,
            self.value_cache_temp,
            quant_param.contiguous(),
            self.attn_cache[layer_idx].int(),
            key_value_quantized_data.shape[-2],
            self.value_cache_quant_unit_V2[layer_idx].quantized_tensor.shape[-2]
        )
        
        self.key_cache_temp[:,:,:self.sink_size,:].copy_(self.key_sink[layer_idx],non_blocking=True)
        self.key_cache_temp[:,:,token_num-windows_size:token_num,:].copy_(self.key_cache_full[layer_idx][:,:,-windows_size:,:],non_blocking=True)
        self.value_cache_temp[:,:,:self.sink_size,:].copy_(self.value_sink[layer_idx],non_blocking=True)
        self.value_cache_temp[:,:,token_num-windows_size:token_num,:].copy_(self.value_cache_full[layer_idx][:,:,-windows_size:,:],non_blocking=True)
    def update_new(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        query_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self.skectch_seen_tokens += key_states.shape[-2]
        # get_tensor_mem(key_states)
        # Update the cache
        self.evict_previous_layer((layer_idx-1)%self.layernum,non_blocking=True)
        self.prefetch_layer((layer_idx + 1) % self.layernum,value_states.device)
        # value_states = value_states.contiguous()
        # print("fuck")
        if key_states.shape[-2] != 1:
            self.evict_previous_layer((layer_idx-1)%self.layernum,non_blocking=True)
            self.resize_buff(device = key_states.device)
            means = key_states.abs().mean(dim=-2,keepdim=True)
            self.key_cache_means[layer_idx] = means
            abs_matrix = torch.abs(key_states)
            abs_sum1 = abs_matrix.mean(dim=(2),keepdim=True)
            sorted_indices = torch.argsort(abs_sum1, dim=-1, descending=True)
            maxvalue = value_states.abs().max(dim=-2,keepdim=True)[0].contiguous()
            self.value_cache_max[layer_idx] = maxvalue
            self.key_cache_index[layer_idx] = sorted_indices[...,:self.topK].contiguous().to(torch.uint8)
            qmin = 0.
            qmax = 2.**8 - 1.
            min_val = key_states.min(dim=-2)[0]
            max_val = key_states.max(dim=-2)[0]
            outlier_scale = (max_val - min_val) / (qmax - qmin)
            outlier_scale[outlier_scale==0]=1
            outlier_zp = (-min_val / outlier_scale)
            outlier_zp = outlier_zp.round().half()
            self.key_cache_channel_zp[layer_idx] = outlier_zp.contiguous()
            self.key_cache_channel_scale[layer_idx] = outlier_scale.contiguous()
            test_key_states = key_states.clone().contiguous()
            key_quant1, outlier_quant,my_outlier_zp_quant,my_outlier_scale_quant,my_resduial  = ontbitquant(
                test_key_states.contiguous(),
                self.key_cache_means[layer_idx].contiguous(),
                self.key_cache_index[layer_idx].contiguous(),
                self.key_cache_channel_zp[layer_idx].contiguous(),
                self.key_cache_channel_scale[layer_idx].contiguous(),
                self.topK
            )
            # torch.cuda.current_stream().synchronize()
            self.key_sign_new[layer_idx]=key_quant1
            # get_tensor_mem(key_quant1)
            score =  onebitgemv(key_quant1,query_states[:,:,-1,:]).unsqueeze(dim=-2).to(query_states.device)
            hlaf_len = int(key_states.shape[-2]*self.imp_rate)
            _,sort_index = torch.topk(score[...,:-self.window_size],k=hlaf_len,dim = -1,sorted=False)
            sort_index = sort_index.squeeze(dim=-2)
            self.attn_cache[layer_idx] = sort_index
            # get_tensor_mem(sort_index)
            # 不能直接保存切片，必须克隆一份，不然pytorch的内存管理机制会保留整个key_states，我真是日乐购
            self.key_cache_full[layer_idx]=(key_states[:,:,-self.window_size:,:]).clone()
            self.value_cache_full[layer_idx]=(value_states[:,:,-self.window_size:,:]).clone()
            self.key_sink[layer_idx] = key_states[:,:,:self.sink_size,:].clone()
            self.value_sink[layer_idx] = value_states[:,:,:self.sink_size,:].clone()
            # print("fuck",key_states.shape)
            value_quant_param, value_quant,my_value_resduial = my_value_quant(value_states,maxvalue)
            quant_param, key_value_quant = my_resduialquant(my_resduial,my_value_resduial)
            self.KV_cache_quant_unit[layer_idx].cat_new_cache(key_value_quant,quant_param)
            self.value_cache_quant_unit_V2[layer_idx].cat_new_cache_V2(value_quant,value_quant_param)
            self.outline_quant[layer_idx] = outline_quant_unit(my_outlier_scale_quant.unsqueeze(dim=-2),my_outlier_zp_quant.unsqueeze(dim=-2))
            self.key_outline[layer_idx] = outlier_quant
            
            self.KV_cache_quant_unit[layer_idx].prepare_future = self.executor.submit(self.KV_cache_quant_unit[layer_idx].prepare_for_next_gen_in_cpu,
                                                                                 self.attn_cache[layer_idx],self.prefetch_stream[layer_idx])
            # if layer_idx==31:
            # self.get_total_memory(1)
            return key_states,value_states
        else:
            self.get_kv(layer_idx=(layer_idx)%self.layernum)

            value_quant_param, value_quant,my_value_resduial = my_value_quant(value_states,self.value_cache_max[layer_idx])
            self.value_cache_quant_unit_V2[layer_idx].cat_new_cache_V2(value_quant,value_quant_param)
            key_quant1, outlier_quant,my_outlier_zp_quant,my_outlier_scale_quant,my_resduial  = ontbitquant(
                key_states.contiguous(),
                self.key_cache_means[layer_idx].contiguous(),
                self.key_cache_index[layer_idx],
                self.key_cache_channel_zp[layer_idx],
                self.key_cache_channel_scale[layer_idx],
                self.topK
            )
            quant_param, key_value_quant = my_resduialquant(my_resduial.contiguous(),my_value_resduial.contiguous())
            self.key_sign_new[layer_idx] = torch.cat((self.key_sign_new[layer_idx],key_quant1),dim=-2)
            self.KV_cache_quant_unit[layer_idx].cat_new_cache(key_value_quant,quant_param)
            self.key_outline[layer_idx] = torch.cat((self.key_outline[layer_idx],outlier_quant),dim=-2) 
        my_score =  onebitgemv(self.key_sign_new[layer_idx],query_states[:,:,-1,:]).unsqueeze(dim=-2).to(query_states.device)
        hlaf_len = int(self.skectch_seen_tokens*self.imp_rate)
        _,sort_index = torch.topk(my_score[...,:-self.window_size],k=hlaf_len,dim = -1,sorted=False)
        sort_index = sort_index.squeeze(dim=-2)
        self.attn_cache[layer_idx] = sort_index 
        self.key_cache_temp[:,:,self.skectch_seen_tokens-1:self.skectch_seen_tokens,:].copy_(key_states)
        self.value_cache_temp[:,:,self.skectch_seen_tokens-1:self.skectch_seen_tokens,:].copy_(value_states)
        self.key_cache_full[layer_idx].copy_(self.key_cache_temp[:,:,self.skectch_seen_tokens-self.window_size:self.skectch_seen_tokens,:])
        self.value_cache_full[layer_idx].copy_(self.value_cache_temp[:,:,self.skectch_seen_tokens-self.window_size:self.skectch_seen_tokens,:])
        self.KV_cache_quant_unit[layer_idx].prepare_future = self.executor.submit(self.KV_cache_quant_unit[layer_idx].prepare_for_next_gen_in_cpu,
                                                                                 self.attn_cache[layer_idx],self.prefetch_stream[layer_idx])
        return self.key_cache_temp[:,:,:self.skectch_seen_tokens,:],self.value_cache_temp[:,:,:self.skectch_seen_tokens,:]
        # return self.get_kv(layer_idx=layer_idx)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache_full) <= layer_idx:
            return 0
        return self.skectch_seen_tokens

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        print("fuck")
        for layer_idx in range(len(self.key_cache_full)):
            device = self.key_cache_full[layer_idx].device
            self.key_cache_full[layer_idx] = self.key_cache_full[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache_full[layer_idx].device
            self.value_cache_full[layer_idx] = self.value_cache_full[layer_idx].index_select(0, beam_idx.to(device))
def mock_prune(key_states,topk,layer_idx = None):
    W_metric = key_states.abs()
    # 在分数张量的最后一维进行排序并获取前 k 个值及其索引
    top_scores, top_indices = torch.topk(W_metric, topk, dim=-1)
    # 创建一个零张量用于存储结果
    result = torch.zeros_like(key_states)
    # 使用 scatter 将值填入结果张量
    result = result.scatter_(3, top_indices, key_states.gather(3, top_indices))
    hit_num = torch.zeros_like(key_states)
    hit_num = hit_num.scatter_(3, top_indices, torch.ones_like(key_states).gather(3, top_indices))
    hit_num = hit_num.sum(dim=-2)
    if layer_idx is not None:
        display = key_states.abs().sum(dim=-2)
        display = display/display.max(dim=-1, keepdim=True)[0]
        # 关闭所有matplotlib图形窗口
        plt.close('all')
        # 使用独立的figure对象
        plt.figure()
        plt.imshow(display.squeeze().cpu().numpy(), cmap='hot', interpolation='nearest')
        plt.colorbar()  # 添加颜色条
        plt.title('Visualization of hit_num')
        plt.xlabel('X-axis label')
        plt.ylabel('Y-axis label')
        # 保存图像
        plt.savefig(str(layer_idx)+'hit_num_visualization.png')
    return result
class pruneCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self,device = None) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.pruneCache_seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.prune_sorce = None
        self.sink_value: List[torch.Tensor] = []
        self.sink_key: List[torch.Tensor] = []
        self.neaest_key: List[torch.Tensor] = []
        self.neaest_value: List[torch.Tensor] = []
        self.topk = 64
        self.neaset_len = 128
        self.sink_size = 4
    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        query_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self.pruneCache_seen_tokens += key_states.shape[-2]

        # Update the cache
        # print("fuck")
        if len(self.key_cache) <= layer_idx:
            self.prune_sorce = torch.norm(query_states, p=2, dim=-2,keepdim=True)
            self.sink_key.append(key_states[:,:,:4,:].clone())
            self.sink_value.append(value_states[:,:,:4,:].clone())
            self.neaest_key.append(key_states[:,:,-self.neaset_len:,:].clone())
            self.neaest_value.append(value_states[:,:,-self.neaset_len:,:].clone())
            # 使用 scatter 将值填入结果张量
            result_key = mock_prune(key_states, 64)
            result_value = mock_prune(value_states, 64,layer_idx=layer_idx)
            # print((result - key_states).sum())
            self.key_cache.append(result_key)
            self.value_cache.append(result_value)
            return key_states,value_states
        else:
            result_key = mock_prune(key_states, 64)
            result_value = mock_prune(value_states, 64)
            # print((result - key_states).sum())
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], result_key], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], result_value], dim=-2)
            self.neaest_key[layer_idx] = torch.cat([self.neaest_key[layer_idx][:,:,1-self.neaset_len:,:], key_states], dim=-2)
            self.neaest_value[layer_idx] = torch.cat([self.neaest_value[layer_idx][:,:,1-self.neaset_len:,:], value_states], dim=-2)
        result_key = torch.cat([self.sink_key[layer_idx],self.key_cache[layer_idx][:,:,self.sink_size:-self.neaset_len,:],self.neaest_key[layer_idx]],dim=-2)
        result_value = torch.cat([self.sink_value[layer_idx],self.value_cache[layer_idx][:,:,self.sink_size:-self.neaset_len,:],self.neaest_value[layer_idx]],dim=-2)
        return result_key, result_value
    def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
        thres_cumsum = sum_before * alpha 
        sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
        thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
        W_mask = (W_metric <= thres)
        cur_sparsity = (W_mask==True).sum() / W_mask.numel()
        return W_mask, cur_sparsity
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

