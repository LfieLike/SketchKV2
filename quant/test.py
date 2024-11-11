# import torch
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
# import numpy as np
# import random
# # import ipdb
# import math
# import os
# import triton
# from new_pack import quant_and_pack_vcache, unpack_and_dequant_kcache, triton_quantize_and_pack_along_last_dim, unpack_and_dequant_vcache, quant_and_pack_kcache
# from matmul import triton_bmm_fA_qB_outer
# from timeit_v2 import py_benchmark


# def set_seed(seed):
# 	np.random.seed(seed)
# 	torch.random.manual_seed(seed)
# 	random.seed(seed)
	

# def test_vcache():
# 	torch.manual_seed(0)
# 	np.random.seed(0)
# 	random.seed(0)
# 	B, nh, T, hd = 555, 32, 433, 128
# 	v = torch.randn((B, nh, T, hd), device='cuda', dtype=torch.float16)
# 	group_size = 64
# 	for bits in [2, 4, 8]:
# 		code, scale, mn = triton_quantize_and_pack_along_last_dim(v, group_size, bits)
# 		# print(f'bit {bits}, scale.shape: {scale.shape}')
# 		# print(f'bit {bits}, code.shape: {code.shape}')
# 		dequant_v = unpack_and_dequant_vcache(code, scale.unsqueeze(-1), mn.unsqueeze(-1), group_size, bits)
# 		assert not dequant_v.isnan().any()
# 		gap = (dequant_v - v) / v
# 		gap = torch.nan_to_num(gap)
# 		print(f'bit {bits}, mean v rel arr: {torch.mean(torch.abs(gap))}')


# def test_kcache():
# 	torch.manual_seed(0)
# 	np.random.seed(0)
# 	random.seed(0)
# 	BS, nh, T, D = 11, 32, 4096, 128
# 	k = torch.randn((BS, nh, T, D), device='cuda', dtype=torch.float16)
# 	group_size = 64
# 	for bits in [2, 4, 8]:
# 		code, scale, mn = triton_quantize_and_pack_along_last_dim(k.transpose(2, 3).contiguous(), 
# 															group_size, 
# 															bits)
# 		dequant_k = unpack_and_dequant_vcache(code, scale.unsqueeze(-1), mn.unsqueeze(-1), group_size, bits)
# 		assert not dequant_k.isnan().any()
# 		gap = (dequant_k.transpose(2, 3) - k) / k
# 		gap = torch.nan_to_num(gap)
# 		print(f'bit {bits}, k mean rel arr: {torch.mean(torch.abs(gap))}')


# def test_bmm_speed():
# 	BS, nh, T, D = 64, 32, 512, 128
# 	bits = 2
# 	key_state = torch.randn((BS, nh, T, D), device='cuda', dtype=torch.float16)
# 	val_state = torch.randn((BS, nh, T, D), device='cuda', dtype=torch.float16)
# 	group_size = 64
# 	query_len = 1
# 	query_state = torch.randn((BS, nh, query_len, D), device='cuda', dtype=torch.float16)

# 	# quantiles = [0.5, 0.2, 0.8]
# 	# ms, min_ms, max_ms = triton.testing.do_bench(
# 	# 	lambda: triton_quantize_and_pack_along_last_dim(key_state.transpose(2,3).contiguous(), 
# 	# 											  group_size, bits), quantiles=quantiles)
# 	# print(f'batch size {BS} nh {nh} seqlen {T} quant and pack pytorch impl: {ms * 1000: .2f} ms')
# 	code, scale, mn = triton_quantize_and_pack_along_last_dim(
# 		key_state.transpose(2,3).contiguous(), group_size, bits)
# 	code = code.contiguous()
# 	scale = scale.contiguous()
# 	mn = mn.contiguous()

# 	stmt = "triton_quantize_and_pack_along_last_dim(key_state.transpose(2,3).contiguous(), group_size, bits)"
# 	t_triton_quant = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=3,
#                                      setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
# 	print(f'our triton quant & pack impl: {t_triton_quant * 1000} ms')
# 	stmt = "quant_and_pack_kcache(key_state, group_size, bits)"
# 	t_quant = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=3,
#                                      setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
# 	print(f'vanilla pytorch quant & pack impl: {t_quant * 1000} ms')
# 	stmt = 'triton_bmm_fA_qB_outer(group_size, query_state, code, scale, mn, bits)'
# 	t_qk = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=3,
#                                      setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
# 	print(f'batch size {BS} seqlen {T} our fused batch qk impl: {t_qk * 1000: .2f} ms')
# 	stmt = 'torch.matmul(query_state, key_state.transpose(2, 3))'
# 	t_qk_ref = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=3,
#                                      setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
# 	print(f'batch size {BS} seqlen {T} pytorch batch qk impl: {t_qk_ref * 1000: .2f} ms')
# 	attn_weight = torch.randn((BS, nh, query_len, T), device='cuda', dtype=torch.float16)
# 	code, scale, mn = triton_quantize_and_pack_along_last_dim(
# 		val_state, group_size, bits)
# 	stmt = 'triton_bmm_fA_qB_outer(group_size, attn_weight, code, scale, mn, bits)'
# 	t_av = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=3,
#                                      setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
# 	print(f'batch size {BS} seqlen {T} our fused batch av impl: {t_av * 1000: .2f} ms')
# 	stmt = 'torch.matmul(attn_weight, val_state)'
# 	t_av_ref = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=3,
#                                      setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
# 	print(f'batch size {BS} seqlen {T} pytorch batch av impl: {t_av_ref * 1000: .2f} ms')

# 	# _code, _scale, _mn = quant_and_pack_kcache(
# 	# 	key_state, group_size, bits)
# 	# _code = _code.transpose(2, 3)
# 	# _scale = _scale.squeeze(-2).transpose(2,3)
# 	# _mn = _mn.squeeze(-2).transpose(2,3)
# 	# print(_code.shape, code.shape, _code.dtype, code.dtype)
# 	# print(_scale.shape, scale.shape, _scale.dtype, scale.dtype)

# 	# our_out = triton_bmm_fA_qB_outer(group_size, query_state, code, scale, mn, bits)
# 	# ref_out = torch.matmul(query_state, key_state.transpose(2, 3))
# 	# gap = (our_out - ref_out) / ref_out
# 	# gap = torch.nan_to_num(gap)
# 	# err = torch.mean(torch.abs(gap)).item()
# 	# print(f'bits {bits}, err: {err}')
# 	# ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_bmm_fA_qB_outer(group_size, query_state, code, scale, mn, bits), quantiles=quantiles)
# 	# print(f'batch size {BS} seqlen {T} our fused batch matmul impl: {ms * 1000: .2f} ms')
# 	# ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(query_state, key_state.transpose(2, 3)), quantiles=quantiles)
# 	# print(f'batch size {BS} seqlen {T} pytorch batch matmul impl: {ms * 1000: .2f} ms')


# def test_streaming_kvcache():
# 	BS, nh, T, D = 1, 32, 340, 128
# 	our_attn_output = None
# 	group_size = 64
# 	query_len = 1
# 	bits = 2
# 	key_states = torch.randn((BS, nh, T, D), device='cuda', dtype=torch.float16)
# 	value_states =  torch.randn((BS, nh, T, D), device='cuda', dtype=torch.float16)
# 	key_states_quant = key_states[:, :, :-(key_states.shape[-2] % group_size), :].contiguous()
# 	key_states_full = key_states[:, :, -(key_states.shape[-2] % group_size):, :].contiguous()
# 	value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(value_states, 
# 																					group_size,
# 																					bits)
# 	key_states_quant_trans, key_scale_trans, key_mn_trans = triton_quantize_and_pack_along_last_dim(key_states_quant.transpose(2, 3).contiguous(),
# 																								group_size, bits)
# 	for i in range(16):
# 		if our_attn_output is None:
# 			query_states = torch.randn((BS, nh, query_len, D), device='cuda', dtype=torch.float16)
# 		else:
# 			query_states = our_attn_output
# 		key_states_new = torch.randn((BS, nh, query_len, D), device='cuda', dtype=torch.float16)
# 		value_states_new =  torch.randn((BS, nh, query_len, D), device='cuda', dtype=torch.float16)
# 		att_qkquant = triton_bmm_fA_qB_outer(group_size, query_states, key_states_quant_trans, 
# 										key_scale_trans, key_mn_trans, bits)
# 		key_states_full = torch.cat([key_states_full, key_states_new], dim=2)
# 		att_qkfull = torch.matmul(query_states, key_states_full.transpose(2, 3))
# 		our_att_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(D)
# 		our_att_weights = torch.softmax(our_att_weights, dim=-1)
# 		value_states_quant_new, scale, mn = triton_quantize_and_pack_along_last_dim(value_states_new, 
# 																					group_size, 
# 																					bits)
# 		value_states_quant = torch.cat([value_states_quant, value_states_quant_new], dim=2)
# 		value_scale = torch.cat([value_scale, scale], dim=2)
# 		value_mn = torch.cat([value_mn, mn], dim=2)
# 		our_attn_output = triton_bmm_fA_qB_outer(group_size, our_att_weights, value_states_quant, 
# 										value_scale, value_mn, bits)
# 		# ===
# 		key_states = torch.cat([key_states, key_states_new], dim=2)
# 		value_states = torch.cat([value_states, value_states_new], dim=2)
# 		ref_att_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(D)
# 		ref_att_weights = torch.softmax(ref_att_weights, dim=-1)
# 		ref_att_out = torch.matmul(ref_att_weights, value_states)
# 		att_weight_gap = (ref_att_weights - our_att_weights) / ref_att_weights
# 		print(f'i {i} bit {bits}, mean att weight rel arr: {torch.mean(torch.abs(att_weight_gap))}')
# 		att_out_gap = (ref_att_out - our_attn_output) / ref_att_out
# 		print(f'i {i} bit {bits}, mean att out rel arr: {torch.mean(torch.abs(att_out_gap))}')


# def test_4d_qmatmul():
# 	torch.manual_seed(0)
# 	np.random.seed(0)
# 	random.seed(0)
# 	query_len = 1
# 	BS, nh, T, D = 16, 32, 1024, 128
# 	group_size = 64
# 	# k = torch.randn((BS, nh, T, D), device='cuda', dtype=torch.float16)
# 	# query_state = torch.randn((BS, nh, query_len, D), device='cuda', dtype=torch.float16)
# 	k = torch.randint(10, (BS, nh, T, D), device='cuda').to(torch.float16)
# 	query_state = torch.randint(5, (BS, nh, query_len, D), device='cuda').to(torch.float16)
# 	for bits in [8, 4, 2]:
# 		# code.shape == BS, nh, T // feat_per_int, D
# 		# scale, mn.shape == BS, nh, ng, 1, D
# 		code, scale, mn = quant_and_pack_kcache(k, group_size, bits)
# 		dequant_k = unpack_and_dequant_kcache(code, scale, mn, group_size, bits)
# 		# BS, nh, D, T // feat_per_int
# 		code = code.transpose(2, 3)
# 		# BS, nh, D, T // group_size
# 		scale = scale.view(BS, nh, -1, D).transpose(2, 3)
# 		mn = mn.view(BS, nh, -1, D).transpose(2, 3)
# 		our_out = triton_bmm_fA_qB_outer(group_size, query_state, code, scale, mn, bits)
# 		ref_out = torch.matmul(query_state, k.transpose(2, 3))
# 		# ref_out = torch.matmul(query_state, k.transpose(2, 3))
# 		assert not our_out.isnan().any() 
# 		assert not ref_out.isnan().any() 
# 		gap = (our_out - ref_out) / ref_out
# 		gap = torch.nan_to_num(gap)
# 		err = torch.mean(torch.abs(gap)).item()
# 		print(f'bits {bits}, err: {err}')


# if __name__ == '__main__':
# 	set_seed(114514)
# 	# test_kcache()
# 	# test_vcache()
# 	# test_4d_qmatmul()
# 	# test_streaming_kvcache()
# 	test_bmm_speed()
import triton
import triton.language as tl
import random
import numpy as np
import torch
from functools import wraps
def gpu_timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 创建 CUDA 事件以记录时间
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # 同步以确保之前的操作完成
        torch.cuda.synchronize()

        # 记录起始时间
        start_event.record()

        # 调用实际的函数
        result = func(*args, **kwargs)

        # 记录结束时间
        end_event.record()

        # 同步确保计算完成
        torch.cuda.synchronize()

        # 计算经过的时间（以毫秒为单位）
        elapsed_time_ms = start_event.elapsed_time(end_event)

        print(f"{func.__name__} executed in: {elapsed_time_ms:.3f} ms")
        return result

    return wrapper
@triton.jit
def _pack_along_last_dim(
	bits: tl.constexpr,
	intensor_ptr,
	code_ptr,
	N,
	num_feats: tl.constexpr,
	feat_per_int: tl.constexpr,
	BLOCK_SIZE_N: tl.constexpr
):
	num_int_per_y_dim = num_feats // feat_per_int
	bid = tl.program_id(axis=0)
	yid = tl.program_id(axis=1)
	offs_N = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	block_start = intensor_ptr + offs_N * num_feats + yid * feat_per_int # offset of the first element at current tile
	packed = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32)
	for i in range(feat_per_int):
		ptr = block_start + i
		element = tl.load(ptr, mask=offs_N<N, other=0.)
		element = element << (i * bits)
		# Combine the value using bitwise OR
		packed = packed | element
	tl.store(code_ptr + offs_N * num_int_per_y_dim + yid, packed, mask=offs_N < N)



@triton.jit
def _minmax_along_last_dim(
	x_ptr,
	mn_ptr, mx_ptr,
	total_elements: tl.constexpr, 
	N: tl.constexpr,
	num_groups: tl.constexpr, 
	group_size: tl.constexpr,
	BLOCK_SIZE_N: tl.constexpr
):
	bid = tl.program_id(axis=0)
	offsets_b = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	offsets = offsets_b[:, None] * group_size + tl.arange(0, group_size)[None, :]
	mask = offsets < total_elements
	x = tl.load(x_ptr + offsets, mask=mask)
	mx_val = tl.max(x, axis=1)
	mn_val = tl.min(x, axis=1)
	# tl.device_print('shape', mn_val[:, None].shape)
	tl.store(mn_ptr+offsets_b, mn_val, mask=offsets_b<N*num_groups)
	tl.store(mx_ptr+offsets_b, mx_val, mask=offsets_b<N*num_groups)	
 
@gpu_timer
def triton_quantize_and_pack_along_last_dim(data: torch.Tensor, group_size: int, bit: int):
	assert len(data.shape) == 4
	shape = data.shape
	B, nh, D, T = shape
	# ================== Get Scale & Zeros ===============
	assert T % group_size == 0
	num_groups = T // group_size
	new_shape = (B * nh * D, num_groups, group_size)
	scale_mn_shape = B, nh, D, num_groups
	# Quantize
	data = data.reshape(new_shape)
	mx = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
	mn = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
	BLOCK_SIZE_N = 128
	grid = lambda meta: (triton.cdiv(data.shape[0]*data.shape[1], BLOCK_SIZE_N),)
	_minmax_along_last_dim[grid](data, mn, mx,
							 data.numel(), data.shape[0], num_groups, group_size,
							 BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8) 
	# mn = torch.min(data, dim=-1, keepdim=True)[0].squeeze(-1)
	# mx = torch.max(data, dim=-1, keepdim=True)[0].squeeze(-1)
	scale = (mx - mn) / (2 ** bit - 1)
	data = data - mn.unsqueeze(-1)
	data.div_(scale.unsqueeze(-1))
	data = data.clamp_(0, 2 ** bit - 1).round_().to(torch.int32)
	data = data.view(-1, T)
	feat_per_int = 32 // bit
	packshape = (np.prod(shape[:-1]), shape[-1] // feat_per_int,)
	code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
	grid = lambda meta: (triton.cdiv(data.shape[0], BLOCK_SIZE_N), data.shape[1] // feat_per_int,)
	_pack_along_last_dim[grid](bit, data, code, data.shape[0], 
								data.shape[1], feat_per_int, 
								BLOCK_SIZE_N=BLOCK_SIZE_N, 
								num_warps=8)
	return code.view(B, nh, D, -1), scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)



# Test case
def test_triton_quantize_and_pack_along_last_dim():
    # Parameters
    B, nh, D, T = 1, 32, 4096, 128  # Example dimensions
    group_size = 128
    bit = 4

    # Generate random data
    data = torch.randn((B, nh, D, T), device='cuda')

    # Call the function
    for i in range(10):
	    code, scale, mn = triton_quantize_and_pack_along_last_dim(data, group_size, bit)

    # Print results
    print("Quantized and Packed Code:")
    print(code.shape)
    print("Scale:")
    print(scale.shape)
    print("Min Values:")
    print(mn.shape)

    # Additional checks can be added here to validate the output
    # For example, check if the shape of the output is as expected
    assert code.shape == (B, nh, D, T // (32 // bit))
    assert scale.shape == (B, nh, D, T // group_size)
    assert mn.shape == (B, nh, D, T // group_size)

    print("Test passed!")

# Run the test case
if __name__ == "__main__":
    test_triton_quantize_and_pack_along_last_dim()