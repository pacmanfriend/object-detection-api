// kernels.cu
extern "C" {

__global__ void conv2d(float* input, float* kernel, float* output,
                       int in_channels, int out_channels,
                       int in_size, int kernel_size, int stride, int padding) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int ch_out = blockIdx.z;

    if (out_x >= (in_size + 2 * padding - kernel_size) / stride + 1 ||
        out_y >= (in_size + 2 * padding - kernel_size) / stride + 1 ||
        ch_out >= out_channels)
        return;

    int out_idx = ch_out * ((in_size + 2 * padding - kernel_size) / stride + 1) *
                          ((in_size + 2 * padding - kernel_size) / stride + 1) +
                  out_y * ((in_size + 2 * padding - kernel_size) / stride + 1) + out_x;

    float sum = 0.0f;
    for (int ch_in = 0; ch_in < in_channels; ++ch_in) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_x = out_x * stride + kx - padding;
                int in_y = out_y * stride + ky - padding;

                if (in_x >= 0 && in_x < in_size && in_y >= 0 && in_y < in_size) {
                    int in_idx = ch_in * in_size * in_size + in_y * in_size + in_x;
                    int ker_idx = ch_out * in_channels * kernel_size * kernel_size + ch_in * kernel_size * kernel_size + ky * kernel_size + kx;
                    sum += input[in_idx] * kernel[ker_idx];
                }
            }
        }
    }
    output[out_idx] = sum;
}

__global__ void relu_activation(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        data[idx] = data[idx] > 0 ? data[idx] : 0;
}

}