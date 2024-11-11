#include <torch/extension.h>
#include <vector>
#include <thread>

void gather_thread(const torch::Tensor& input, const torch::Tensor& index, torch::Tensor& output, int start, int end) {
    for (int i = start; i < end; ++i) {
        output[i] = input[index[i]];
    }
}

torch::Tensor threaded_gather(const torch::Tensor& input, const torch::Tensor& index) {
    auto output = torch::empty(index.sizes(), input.options());
    int num_threads = std::thread::hardware_concurrency();
    int chunk_size = index.size(0) / num_threads;
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? index.size(0) : start + chunk_size;
        threads.emplace_back(gather_thread, std::cref(input), std::cref(index), std::ref(output), start, end);
    }

    for (auto& t : threads) {
        t.join();
    }

    return output;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("threaded_gather", &threaded_gather, "Multi-threaded gather operation");
}