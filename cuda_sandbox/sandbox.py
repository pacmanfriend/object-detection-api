import time

import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule


def show_cuda_info():
    print(f"Устройство: {pycuda.autoinit.device.name()}")
    print(f"Вычислительная способность: {pycuda.autoinit.device.compute_capability()}")
    print(f"Общая память: {pycuda.autoinit.device.total_memory() // 1024 // 1024} MB")


def multiply_by_scalar():
    kernel_code = """
    __global__ void multiply_by_scalar(float *array, float scalar, int n) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {
            array[idx] *= scalar;
        }
    }
    """

    # Компиляция ядра
    mod = SourceModule(kernel_code)
    multiply_kernel = mod.get_function("multiply_by_scalar")

    a = np.random.randn(100000).astype(np.float32)
    gpu_a = gpuarray.to_gpu(a)

    block_size = 32
    grid_size = (100 + block_size - 1) // block_size

    multiply_kernel(gpu_a, np.float32(2.0), np.int32(100000), block=(block_size, 1, 1), grid=(grid_size, 1))


if __name__ == '__main__':
    show_cuda_info()

    start = time.perf_counter()
    multiply_by_scalar()
    end = time.perf_counter()
    print(f"{end - start} seconds")
