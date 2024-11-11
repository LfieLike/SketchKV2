#include <torch/extension.h>
#include <math_constants.h>
#include <cmath>
#define NUM_PER_THREAD 8
#define WARP_SIZE 256
#define WARPS_PER_BLOCK 128
#define EMB_DIM 128
__inline__ __device__
float reduce16(float sum) {
    // mask全设置为0x0000FFFF 结果也对，不知道为啥，反正结果是对的，那就不管了。
    unsigned int mask = ((threadIdx.y*16+threadIdx.x) % 32 < 16) ? 0x0000FFFF : 0xFFFF0000;

    // 使用 __shfl_down_sync 在 16 个线程范围内进行归约
    sum += __shfl_down_sync(mask, sum, 8);  // 将前 8 个线程的值加到后 8 个线程上
    sum += __shfl_down_sync(mask, sum, 4);  // 将前 4 个线程的值加到后 4 个线程上
    sum += __shfl_down_sync(mask, sum, 2);  // 将前 2 个线程的值加到后 2 个线程上
    sum += __shfl_down_sync(mask, sum, 1);  // 将前 1 个线程的值加到后 1 个线程上

    return sum;  // 最终每组的第 0 号线程保存归约结果
}
template <typename T>
__device__ float convert_to_float(T value) {
    // Return 0 by default, indicating misuse if not specialized correctly.
    return 0.0f;
}

template <>
__device__ float convert_to_float<c10::Half>(c10::Half value) {
    return __half2float(value);
}

template <>
__device__ float convert_to_float<float>(float value) {
    return value;
}

template <>
__device__ float convert_to_float<at::BFloat16>(at::BFloat16 value) {
    return static_cast<float>(value);
}

template <typename T>
__device__ T convert_from_float(float value) {
    // Return 0 by default, indicating misuse if not specialized correctly.
    return static_cast<T>(0);
}
template <>
__device__ uint8_t convert_from_float<uint8_t>(float value) {
    return static_cast<uint8_t>(value);
}
template <>
__device__ c10::Half convert_from_float<c10::Half>(float value) {
    return __float2half(value);
}

template <>
__device__ float convert_from_float<float>(float value) {
    return value;
}

template <>
__device__ at::BFloat16 convert_from_float<at::BFloat16>(float value) {
    return static_cast<at::BFloat16>(value);
}



template<typename T>
__global__ void quantize_with_outliers_kernel(
    // 1bit 压缩的keystates，8个一组
    uint8_t* key_states,
    // 8bit 压缩的keyststes outlier ，
    uint8_t* outlier_key_states,
    c10::Half* quant_outlier_zp,
    // outlier量化所需要的zeropoint的
    c10::Half* quant_outlier_scale, 
    // outlier的量化的scale
    uint8_t* outlier_idx,
    // 每一个维度scale
    int outlier_num,
    // query向量
    c10::Half* query_states,
    // 最终结果
    c10::Half* res,
    // outlier channel的数目
    int batch_size, int head_size, int len
    ) {
    size_t batch_id = blockIdx.x;
    size_t head_id = blockIdx.y;
    size_t pro_id = blockIdx.z;

    // batch_id和head_id不能*NUM_PER_THREAD*WARPS_PER_BLOCK 来表示位移，因为可能会有越界的线程。
    int base_index = (batch_id * head_size * len * EMB_DIM) 
                   + (head_id * len * EMB_DIM) 
                   + (pro_id * NUM_PER_THREAD*WARPS_PER_BLOCK);
    // warp排序是列主序,获取当前线程在block中的id.
    size_t th_id = threadIdx.y*16+threadIdx.x;
    int sub_th_id = threadIdx.x;
    // 首先需要知道当前线程处理哪一个向量, 向量长度除每个线程处理的元素数量，得到当前处理的向量的id
    int thread_num_per_emb = EMB_DIM/NUM_PER_THREAD;
    int vec_id = threadIdx.y;
    // 获取当前线程处理的vec的起始位置
    int base_index_key = base_index+vec_id*128;
    // 判断边界 numProjBlocks个block要处理len*EMBDIM个元素
    if(pro_id * NUM_PER_THREAD*WARPS_PER_BLOCK + vec_id*128>=len*EMB_DIM){
        return;
    }
    // 获取当前线程计算的偏移量，每个线程计算NUM_PER_THREAD个元素，每thread_num_per_emb个线程计算计算一个
    // 每个线程处理NUM_PER_THREAD个元素，也就是每个线程读取一个uint8_t
    uint8_t onebit_key = key_states[base_index/8 + th_id];
    // 获取当前线程处理的query位置，每个head有一个对应的128维的query, 还要加上偏移量，每个线程需要读取8个元素
    int query_index = batch_id*gridDim.y*EMB_DIM + head_id*EMB_DIM+ th_id*NUM_PER_THREAD%EMB_DIM;
    // printf("vec_id:%d,query_index:%d,base_index:%d\n",vec_id,query_index,base_index);
    float sub_res = 0;
    #pragma unroll
    for(int shift_i=0;shift_i<8;++shift_i){
        float sub_query = convert_to_float(query_states[query_index+shift_i]);
        sub_res +=  ((onebit_key>> shift_i)&1 ? sub_query:-sub_query);
    }

    // 处理outlier，每个线程处理一个outlier，首先需要计算读取的数据的位置
    if(sub_th_id<outlier_num){
        // 首先获取outlier key的指针位置， 每个vec有outlier个outlier base_index/128*outlier_num  +  vec_id*outlier_num
        int shift1 = base_index/128*outlier_num  +  vec_id*outlier_num + sub_th_id;
        // printf("vec_id:%d,sub_th_id:%d,shift1:%d\n",vec_id,sub_th_id,shift1);
        float sub_outlier_key = (float)(outlier_key_states[shift1]);
        // 获取量化参数的指针,每个head有一个对应的outlier_num维的量化参数
        int shift2 = batch_id*gridDim.y*outlier_num + head_id*outlier_num +sub_th_id;
        int sub_outlier_idx = static_cast<int>(outlier_idx[shift2]);
        float zp = __half2float(quant_outlier_zp[shift2]);
        float scale = __half2float(quant_outlier_scale[shift2]);
        int query_shfit = batch_id*gridDim.y*EMB_DIM + head_id*EMB_DIM;
        float value = (sub_outlier_key-zp)*scale;
        sub_res += value*__half2float(query_states[query_shfit+sub_outlier_idx]);
    }

    // 硬编码一下,每16个线程
    sub_res = reduce16(sub_res);
    if(sub_th_id==0){
        // 获取写入的位置 每个head有 len个元素需要写入， 所以先要获取当前处理的是len中的哪一行
        int cur_len = (pro_id * NUM_PER_THREAD*WARPS_PER_BLOCK + th_id*NUM_PER_THREAD)/EMB_DIM;
        // 当前head的初始写入位置为 batch_id * head_id * len
        int base_index_res = (batch_id*head_size+head_id)*len;
        res[cur_len+base_index_res] = __float2half(sub_res);
    }
    return;
}


torch::TensorOptions getOptionsForType(const std::type_info& typeInfo) {
    if (typeInfo == typeid(c10::Half)) {
        return torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kHalf);
    } else if (typeInfo == typeid(float)) {
        return torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat);
    } else if (typeInfo == typeid(at::BFloat16)) {
        return torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kBFloat16);
    } else {
        // Default case for unexpected types
        throw std::runtime_error("Unsupported type for tensor options.");
    }
}

template <typename T>
torch::Tensor MyScoreCudaTemplate(
    torch::Tensor key_states,
    torch::Tensor outlier_quant,
    torch::Tensor outlier_idx,
    torch::Tensor outlier_zp,
    torch::Tensor outlier_scale,
    torch::Tensor query_states,
    int outlier_num
    ) {

    auto options = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kUInt8);
    auto options_uint32 = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kInt32);
    auto options_outlier_norm = getOptionsForType(typeid(T));

    int batch = key_states.size(0);
    int head = key_states.size(1);
    int len = key_states.size(2);
    

    auto device = key_states.device();

    // 使用 key_states 的设备信息来设置 res 的设备
    auto res = torch::zeros({batch, head, len}, options_outlier_norm.device(device)).contiguous();
    // warp_size 表示一个warp中的线程数，一般为32  WARPS_PER_BLOCK 表示一个block中warp的数量， 一个block最大的线程数是1024，因此WARPS_PER_BLOCK的最大值为32
    // 每个线程处理8个元素，每16个线程处理一个向量的点积,一个warp处理两个向量的点积
    int numProjBlocks = (len*EMB_DIM+(NUM_PER_THREAD*WARPS_PER_BLOCK)-1) / (NUM_PER_THREAD*WARPS_PER_BLOCK);
    dim3 numBlocks(batch , head, numProjBlocks);
    dim3 threadsPerBlockDim(16,8);



//     Compiler hints for using L2 Persistent Cache
    quantize_with_outliers_kernel<c10::Half><<<numBlocks, threadsPerBlockDim, 0>>>(
    key_states.data_ptr<uint8_t>(),
    outlier_quant.data_ptr<uint8_t>(),
    outlier_zp.data_ptr<c10::Half>(),
    outlier_scale.data_ptr<c10::Half>(),
    outlier_idx.data_ptr<uint8_t>(),
    outlier_num,
    query_states.data_ptr<c10::Half>(),
    res.data_ptr<c10::Half>(),
    batch,head,len);
                                                         // Remove any persistent lines in L2

    return res;
}
    // torch::Tensor key_states,
    // torch::Tensor outlier_quant,
    // torch::Tensor outlier_idx,
    // torch::Tensor outlier_zp,
    // torch::Tensor outlier_scale,
    // torch::Tensor query_states,
    // int outlier_num
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("My_score_half_half", &MyScoreCudaTemplate<c10::Half>, "Quantize using Half precision",
    py::arg("key_states"),
    py::arg("outlier_quant"),
    py::arg("outlier_idx"),
    py::arg("outlier_zp"),
    py::arg("outlier_scale"),
    py::arg("res"),
    py::arg("outlier_num"));

    // m.def("My_score_half_float", &MyScoreCudaTemplate<c10::Half>, "Quantize using Half to Float precision",
    // py::arg("key_states"),
    // py::arg("outlier_quant"),
    // py::arg("outlier_idx"),
    // py::arg("outlier_zp"),
    // py::arg("outlier_scale"),
    // py::arg("res"),
    // py::arg("outlier_num"));

    // m.def("My_score_float_float", &MyScoreCudaTemplate<float>, "Quantize using Float precision",
    // py::arg("key_states"),
    // py::arg("outlier_quant"),
    // py::arg("outlier_idx"),
    // py::arg("outlier_zp"),
    // py::arg("outlier_scale"),
    // py::arg("res"),
    // py::arg("outlier_num"));

    // m.def("My_score_bf16_bf16", &MyScoreCudaTemplate<at::BFloat16>, "Quantize using BF16 precision",
    // py::arg("key_states"),
    // py::arg("outlier_quant"),
    // py::arg("outlier_idx"),
    // py::arg("outlier_zp"),
    // py::arg("outlier_scale"),
    // py::arg("res"),
    // py::arg("outlier_num"));

    // m.def("My_score_bf16_float", &MyScoreCudaTemplate<at::BFloat16>, "Quantize using BF16 to Float precision",
    // py::arg("key_states"),
    // py::arg("outlier_quant"),
    // py::arg("outlier_idx"),
    // py::arg("outlier_zp"),
    // py::arg("outlier_scale"),
    // py::arg("res"),
    // py::arg("outlier_num"));
}
