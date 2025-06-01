class ReLU:
    """ReLU активация"""

    def __init__(self):
        self.input_cache = None

    def forward(self, x, kernels):
        self.input_cache = x

        output = gpuarray.empty_like(x)
        size = x.size

        block_size = 256
        grid_size = (size + block_size - 1) // block_size

        kernels['relu'](
            x, output, np.int32(size),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )

        return output

    def backward(self, grad_output, kernels):
        grad_input = gpuarray.empty_like(self.input_cache)
        size = self.input_cache.size

        block_size = 256
        grid_size = (size + block_size - 1) // block_size

        kernels['relu_backward'](
            grad_output, self.input_cache, grad_input, np.int32(size),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )

        return grad_inputimport
        numpy as np


import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from PIL import Image
import json


class CudaKernels:
    """CUDA ядра для операций нейронной сети"""

    def __init__(self):
        self.kernels = self._compile_kernels()

    def _compile_kernels(self):
        cuda_code = """
        // Forward pass kernels
        __global__ void conv2d_kernel(float* input, float* weight, float* output, float* bias,
                                    int batch_size, int in_channels, int out_channels,
                                    int in_height, int in_width, int out_height, int out_width,
                                    int kernel_size, int stride, int padding) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_outputs = batch_size * out_channels * out_height * out_width;

            if (idx >= total_outputs) return;

            int b = idx / (out_channels * out_height * out_width);
            int remainder = idx % (out_channels * out_height * out_width);
            int oc = remainder / (out_height * out_width);
            remainder = remainder % (out_height * out_width);
            int oh = remainder / out_width;
            int ow = remainder % out_width;

            float sum = 0.0f;

            for (int ic = 0; ic < in_channels; ic++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int ih = oh * stride - padding + kh;
                        int iw = ow * stride - padding + kw;

                        if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                            int input_idx = b * in_channels * in_height * in_width +
                                          ic * in_height * in_width + ih * in_width + iw;
                            int weight_idx = oc * in_channels * kernel_size * kernel_size +
                                           ic * kernel_size * kernel_size + kh * kernel_size + kw;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }

            if (bias != NULL) {
                sum += bias[oc];
            }

            output[idx] = sum;
        }

        // Batch Normalization kernel
        __global__ void batch_norm_kernel(float* input, float* output, float* gamma, float* beta,
                                        float* running_mean, float* running_var,
                                        int batch_size, int channels, int height, int width,
                                        float eps) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_elements = batch_size * channels * height * width;

            if (idx >= total_elements) return;

            int b = idx / (channels * height * width);
            int remainder = idx % (channels * height * width);
            int c = remainder / (height * width);

            float normalized = (input[idx] - running_mean[c]) / sqrtf(running_var[c] + eps);
            output[idx] = gamma[c] * normalized + beta[c];
        }

        // ReLU activation kernel
        __global__ void relu_kernel(float* input, float* output, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                output[idx] = fmaxf(0.0f, input[idx]);
            }
        }

        // Max pooling kernel
        __global__ void max_pool_kernel(float* input, float* output,
                                      int batch_size, int channels, int in_height, int in_width,
                                      int out_height, int out_width, int kernel_size, int stride) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_outputs = batch_size * channels * out_height * out_width;

            if (idx >= total_outputs) return;

            int b = idx / (channels * out_height * out_width);
            int remainder = idx % (channels * out_height * out_width);
            int c = remainder / (out_height * out_width);
            remainder = remainder % (out_height * out_width);
            int oh = remainder / out_width;
            int ow = remainder % out_width;

            float max_val = -INFINITY;

            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int ih = oh * stride + kh;
                    int iw = ow * stride + kw;

                    if (ih < in_height && iw < in_width) {
                        int input_idx = b * channels * in_height * in_width +
                                      c * in_height * in_width + ih * in_width + iw;
                        max_val = fmaxf(max_val, input[input_idx]);
                    }
                }
            }

            output[idx] = max_val;
        }

        // Average pooling kernel
        __global__ void avg_pool_kernel(float* input, float* output,
                                      int batch_size, int channels, int in_height, int in_width,
                                      int kernel_size) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_outputs = batch_size * channels;

            if (idx >= total_outputs) return;

            int b = idx / channels;
            int c = idx % channels;

            float sum = 0.0f;
            int count = 0;

            for (int h = 0; h < in_height; h++) {
                for (int w = 0; w < in_width; w++) {
                    int input_idx = b * channels * in_height * in_width +
                                  c * in_height * in_width + h * in_width + w;
                    sum += input[input_idx];
                    count++;
                }
            }

            output[idx] = sum / count;
        }

        // Element-wise addition kernel
        __global__ void add_kernel(float* a, float* b, float* output, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                output[idx] = a[idx] + b[idx];
            }
        }

        // Fully connected layer kernel
        __global__ void linear_kernel(float* input, float* weight, float* output, float* bias,
                                    int batch_size, int input_size, int output_size) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_outputs = batch_size * output_size;

            if (idx >= total_outputs) return;

            int b = idx / output_size;
            int o = idx % output_size;

            float sum = 0.0f;
            for (int i = 0; i < input_size; i++) {
                int input_idx = b * input_size + i;
                int weight_idx = o * input_size + i;
                sum += input[input_idx] * weight[weight_idx];
            }

            if (bias != NULL) {
                sum += bias[o];
            }

            output[idx] = sum;
        }

        // Backward pass kernels

        // Convolution backward input gradient
        __global__ void conv2d_backward_input(float* grad_output, float* weight, float* grad_input,
                                            int batch_size, int in_channels, int out_channels,
                                            int in_height, int in_width, int out_height, int out_width,
                                            int kernel_size, int stride, int padding) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_inputs = batch_size * in_channels * in_height * in_width;

            if (idx >= total_inputs) return;

            int b = idx / (in_channels * in_height * in_width);
            int remainder = idx % (in_channels * in_height * in_width);
            int ic = remainder / (in_height * in_width);
            remainder = remainder % (in_height * in_width);
            int ih = remainder / in_width;
            int iw = remainder % in_width;

            float sum = 0.0f;

            for (int oc = 0; oc < out_channels; oc++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int oh = (ih + padding - kh) / stride;
                        int ow = (iw + padding - kw) / stride;

                        if (oh >= 0 && oh < out_height && ow >= 0 && ow < out_width &&
                            (ih + padding - kh) % stride == 0 && (iw + padding - kw) % stride == 0) {

                            int grad_output_idx = b * out_channels * out_height * out_width +
                                                oc * out_height * out_width + oh * out_width + ow;
                            int weight_idx = oc * in_channels * kernel_size * kernel_size +
                                           ic * kernel_size * kernel_size + kh * kernel_size + kw;
                            sum += grad_output[grad_output_idx] * weight[weight_idx];
                        }
                    }
                }
            }

            grad_input[idx] = sum;
        }

        // Convolution backward weight gradient
        __global__ void conv2d_backward_weight(float* input, float* grad_output, float* grad_weight,
                                             int batch_size, int in_channels, int out_channels,
                                             int in_height, int in_width, int out_height, int out_width,
                                             int kernel_size, int stride, int padding) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_weights = out_channels * in_channels * kernel_size * kernel_size;

            if (idx >= total_weights) return;

            int oc = idx / (in_channels * kernel_size * kernel_size);
            int remainder = idx % (in_channels * kernel_size * kernel_size);
            int ic = remainder / (kernel_size * kernel_size);
            remainder = remainder % (kernel_size * kernel_size);
            int kh = remainder / kernel_size;
            int kw = remainder % kernel_size;

            float sum = 0.0f;

            for (int b = 0; b < batch_size; b++) {
                for (int oh = 0; oh < out_height; oh++) {
                    for (int ow = 0; ow < out_width; ow++) {
                        int ih = oh * stride - padding + kh;
                        int iw = ow * stride - padding + kw;

                        if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                            int input_idx = b * in_channels * in_height * in_width +
                                          ic * in_height * in_width + ih * in_width + iw;
                            int grad_output_idx = b * out_channels * out_height * out_width +
                                                oc * out_height * out_width + oh * out_width + ow;
                            sum += input[input_idx] * grad_output[grad_output_idx];
                        }
                    }
                }
            }

            grad_weight[idx] = sum;
        }

        // Convolution backward bias gradient
        __global__ void conv2d_backward_bias(float* grad_output, float* grad_bias,
                                           int batch_size, int out_channels, int out_height, int out_width) {

            int oc = blockIdx.x * blockDim.x + threadIdx.x;
            if (oc >= out_channels) return;

            float sum = 0.0f;

            for (int b = 0; b < batch_size; b++) {
                for (int h = 0; h < out_height; h++) {
                    for (int w = 0; w < out_width; w++) {
                        int idx = b * out_channels * out_height * out_width +
                                oc * out_height * out_width + h * out_width + w;
                        sum += grad_output[idx];
                    }
                }
            }

            grad_bias[oc] = sum;
        }

        // ReLU backward
        __global__ void relu_backward_kernel(float* grad_output, float* input, float* grad_input, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
            }
        }

        // Batch norm backward
        __global__ void batch_norm_backward_kernel(float* grad_output, float* input, float* gamma,
                                                 float* running_mean, float* running_var,
                                                 float* grad_input, float* grad_gamma, float* grad_beta,
                                                 int batch_size, int channels, int height, int width,
                                                 float eps) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_elements = batch_size * channels * height * width;

            if (idx >= total_elements) return;

            int c = (idx / (height * width)) % channels;

            // Simplified backward pass
            float var_eps = running_var[c] + eps;
            float inv_std = 1.0f / sqrtf(var_eps);

            grad_input[idx] = grad_output[idx] * gamma[c] * inv_std;

            // Note: grad_gamma and grad_beta would need reduction across batch/spatial dims
            // This is simplified for demonstration
        }

        // Max pooling backward
        __global__ void max_pool_backward_kernel(float* grad_output, float* input, float* grad_input,
                                               int* max_indices, int batch_size, int channels,
                                               int in_height, int in_width, int out_height, int out_width,
                                               int kernel_size, int stride) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_outputs = batch_size * channels * out_height * out_width;

            if (idx >= total_outputs) return;

            int b = idx / (channels * out_height * out_width);
            int remainder = idx % (channels * out_height * out_width);
            int c = remainder / (out_height * out_width);
            remainder = remainder % (out_height * out_width);
            int oh = remainder / out_width;
            int ow = remainder % out_width;

            float max_val = -INFINITY;
            int max_ih = -1, max_iw = -1;

            // Find max position
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int ih = oh * stride + kh;
                    int iw = ow * stride + kw;

                    if (ih < in_height && iw < in_width) {
                        int input_idx = b * channels * in_height * in_width +
                                      c * in_height * in_width + ih * in_width + iw;
                        if (input[input_idx] > max_val) {
                            max_val = input[input_idx];
                            max_ih = ih;
                            max_iw = iw;
                        }
                    }
                }
            }

            // Set gradient only for max position
            if (max_ih >= 0 && max_iw >= 0) {
                int grad_input_idx = b * channels * in_height * in_width +
                                   c * in_height * in_width + max_ih * in_width + max_iw;
                atomicAdd(&grad_input[grad_input_idx], grad_output[idx]);
            }
        }

        // Linear backward input gradient
        __global__ void linear_backward_input(float* grad_output, float* weight, float* grad_input,
                                            int batch_size, int input_size, int output_size) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_inputs = batch_size * input_size;

            if (idx >= total_inputs) return;

            int b = idx / input_size;
            int i = idx % input_size;

            float sum = 0.0f;
            for (int o = 0; o < output_size; o++) {
                int grad_output_idx = b * output_size + o;
                int weight_idx = o * input_size + i;
                sum += grad_output[grad_output_idx] * weight[weight_idx];
            }

            grad_input[idx] = sum;
        }

        // Linear backward weight gradient
        __global__ void linear_backward_weight(float* input, float* grad_output, float* grad_weight,
                                             int batch_size, int input_size, int output_size) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_weights = output_size * input_size;

            if (idx >= total_weights) return;

            int o = idx / input_size;
            int i = idx % input_size;

            float sum = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                int input_idx = b * input_size + i;
                int grad_output_idx = b * output_size + o;
                sum += input[input_idx] * grad_output[grad_output_idx];
            }

            grad_weight[idx] = sum;
        }

        // Linear backward bias gradient
        __global__ void linear_backward_bias(float* grad_output, float* grad_bias,
                                           int batch_size, int output_size) {

            int o = blockIdx.x * blockDim.x + threadIdx.x;
            if (o >= output_size) return;

            float sum = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                sum += grad_output[b * output_size + o];
            }

            grad_bias[o] = sum;
        }

        // Cross entropy loss
        __global__ void cross_entropy_loss_kernel(float* logits, int* targets, float* loss,
                                                int batch_size, int num_classes) {

            int b = blockIdx.x * blockDim.x + threadIdx.x;
            if (b >= batch_size) return;

            // Find max for numerical stability
            float max_logit = -INFINITY;
            for (int c = 0; c < num_classes; c++) {
                max_logit = fmaxf(max_logit, logits[b * num_classes + c]);
            }

            // Compute softmax denominator
            float sum_exp = 0.0f;
            for (int c = 0; c < num_classes; c++) {
                sum_exp += expf(logits[b * num_classes + c] - max_logit);
            }

            // Compute loss
            int target = targets[b];
            float log_prob = logits[b * num_classes + target] - max_logit - logf(sum_exp);
            loss[b] = -log_prob;
        }

        // Cross entropy gradient
        __global__ void cross_entropy_grad_kernel(float* logits, int* targets, float* grad,
                                                int batch_size, int num_classes) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_elements = batch_size * num_classes;

            if (idx >= total_elements) return;

            int b = idx / num_classes;
            int c = idx % num_classes;

            // Find max for numerical stability
            float max_logit = -INFINITY;
            for (int i = 0; i < num_classes; i++) {
                max_logit = fmaxf(max_logit, logits[b * num_classes + i]);
            }

            // Compute softmax
            float sum_exp = 0.0f;
            for (int i = 0; i < num_classes; i++) {
                sum_exp += expf(logits[b * num_classes + i] - max_logit);
            }

            float softmax = expf(logits[idx] - max_logit) / sum_exp;

            // Gradient: softmax - one_hot
            grad[idx] = softmax - (c == targets[b] ? 1.0f : 0.0f);
        }

        // SGD optimizer update
        __global__ void sgd_update_kernel(float* params, float* grads, float* momentum,
                                        float lr, float momentum_factor, float weight_decay,
                                        int size) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return;

            // Apply weight decay
            if (weight_decay > 0) {
                grads[idx] += weight_decay * params[idx];
            }

            // Update momentum
            if (momentum != NULL) {
                momentum[idx] = momentum_factor * momentum[idx] + grads[idx];
                params[idx] -= lr * momentum[idx];
            } else {
                params[idx] -= lr * grads[idx];
            }
        }
        """

        mod = SourceModule(cuda_code)
        return {
            'conv2d': mod.get_function("conv2d_kernel"),
            'batch_norm': mod.get_function("batch_norm_kernel"),
            'relu': mod.get_function("relu_kernel"),
            'max_pool': mod.get_function("max_pool_kernel"),
            'avg_pool': mod.get_function("avg_pool_kernel"),
            'add': mod.get_function("add_kernel"),
            'linear': mod.get_function("linear_kernel"),
            # Backward pass kernels
            'conv2d_backward_input': mod.get_function("conv2d_backward_input"),
            'conv2d_backward_weight': mod.get_function("conv2d_backward_weight"),
            'conv2d_backward_bias': mod.get_function("conv2d_backward_bias"),
            'relu_backward': mod.get_function("relu_backward_kernel"),
            'batch_norm_backward': mod.get_function("batch_norm_backward_kernel"),
            'max_pool_backward': mod.get_function("max_pool_backward_kernel"),
            'linear_backward_input': mod.get_function("linear_backward_input"),
            'linear_backward_weight': mod.get_function("linear_backward_weight"),
            'linear_backward_bias': mod.get_function("linear_backward_bias"),
            # Loss and optimizer kernels
            'cross_entropy_loss': mod.get_function("cross_entropy_loss_kernel"),
            'cross_entropy_grad': mod.get_function("cross_entropy_grad_kernel"),
            'sgd_update': mod.get_function("sgd_update_kernel")
        }


class Conv2d:
    """Сверточный слой"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = bias

        # Инициализация весов (He initialization)
        fan_in = in_channels * kernel_size * kernel_size
        std = np.sqrt(2.0 / fan_in)
        self.weight = gpuarray.to_gpu(
            np.random.normal(0, std, (out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32)
        )

        # Gradients for weights
        self.grad_weight = gpuarray.zeros_like(self.weight)

        if bias:
            self.bias = gpuarray.to_gpu(np.zeros(out_channels, dtype=np.float32))
            self.grad_bias = gpuarray.zeros_like(self.bias)
        else:
            self.bias = None
            self.grad_bias = None

        # Store input for backward pass
        self.input_cache = None

    def forward(self, x, kernels):
        # Store input for backward pass
        self.input_cache = x

        batch_size, in_channels, in_height, in_width = x.shape

        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        output = gpuarray.empty((batch_size, self.out_channels, out_height, out_width), dtype=np.float32)

        total_outputs = batch_size * self.out_channels * out_height * out_width
        block_size = 256
        grid_size = (total_outputs + block_size - 1) // block_size

        bias_ptr = self.bias.ptr if self.bias is not None else 0

        kernels['conv2d'](
            x, self.weight, output, bias_ptr,
            np.int32(batch_size), np.int32(in_channels), np.int32(self.out_channels),
            np.int32(in_height), np.int32(in_width), np.int32(out_height), np.int32(out_width),
            np.int32(self.kernel_size), np.int32(self.stride), np.int32(self.padding),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )

        return output

    def backward(self, grad_output, kernels):
        batch_size, in_channels, in_height, in_width = self.input_cache.shape
        out_height = grad_output.shape[2]
        out_width = grad_output.shape[3]

        # Gradient w.r.t. input
        grad_input = gpuarray.zeros_like(self.input_cache)

        total_inputs = batch_size * in_channels * in_height * in_width
        block_size = 256
        grid_size = (total_inputs + block_size - 1) // block_size

        kernels['conv2d_backward_input'](
            grad_output, self.weight, grad_input,
            np.int32(batch_size), np.int32(in_channels), np.int32(self.out_channels),
            np.int32(in_height), np.int32(in_width), np.int32(out_height), np.int32(out_width),
            np.int32(self.kernel_size), np.int32(self.stride), np.int32(self.padding),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )

        # Gradient w.r.t. weight
        total_weights = self.out_channels * in_channels * self.kernel_size * self.kernel_size
        grid_size = (total_weights + block_size - 1) // block_size

        kernels['conv2d_backward_weight'](
            self.input_cache, grad_output, self.grad_weight,
            np.int32(batch_size), np.int32(in_channels), np.int32(self.out_channels),
            np.int32(in_height), np.int32(in_width), np.int32(out_height), np.int32(out_width),
            np.int32(self.kernel_size), np.int32(self.stride), np.int32(self.padding),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )

        # Gradient w.r.t. bias
        if self.bias is not None:
            grid_size = (self.out_channels + block_size - 1) // block_size
            kernels['conv2d_backward_bias'](
                grad_output, self.grad_bias,
                np.int32(batch_size), np.int32(self.out_channels), np.int32(out_height), np.int32(out_width),
                block=(block_size, 1, 1), grid=(grid_size, 1)
            )

        return grad_input


class BatchNorm2d:
    """Batch Normalization слой"""

    def __init__(self, num_features, eps=1e-5):
        self.num_features = num_features
        self.eps = eps

        self.gamma = gpuarray.to_gpu(np.ones(num_features, dtype=np.float32))
        self.beta = gpuarray.to_gpu(np.zeros(num_features, dtype=np.float32))
        self.running_mean = gpuarray.to_gpu(np.zeros(num_features, dtype=np.float32))
        self.running_var = gpuarray.to_gpu(np.ones(num_features, dtype=np.float32))

        # Gradients
        self.grad_gamma = gpuarray.zeros_like(self.gamma)
        self.grad_beta = gpuarray.zeros_like(self.beta)

        # Cache for backward pass
        self.input_cache = None

    def forward(self, x, kernels):
        self.input_cache = x

        batch_size, channels, height, width = x.shape
        output = gpuarray.empty_like(x)

        total_elements = batch_size * channels * height * width
        block_size = 256
        grid_size = (total_elements + block_size - 1) // block_size

        kernels['batch_norm'](
            x, output, self.gamma, self.beta, self.running_mean, self.running_var,
            np.int32(batch_size), np.int32(channels), np.int32(height), np.int32(width),
            np.float32(self.eps),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )

        return output

    def backward(self, grad_output, kernels):
        batch_size, channels, height, width = self.input_cache.shape
        grad_input = gpuarray.empty_like(self.input_cache)

        total_elements = batch_size * channels * height * width
        block_size = 256
        grid_size = (total_elements + block_size - 1) // block_size

        kernels['batch_norm_backward'](
            grad_output, self.input_cache, self.gamma,
            self.running_mean, self.running_var,
            grad_input, self.grad_gamma, self.grad_beta,
            np.int32(batch_size), np.int32(channels), np.int32(height), np.int32(width),
            np.float32(self.eps),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )

        return grad_input


class ReLU:
    """ReLU активация"""

    def forward(self, x, kernels):
        output = gpuarray.empty_like(x)
        size = x.size

        block_size = 256
        grid_size = (size + block_size - 1) // block_size

        kernels['relu'](
            x, output, np.int32(size),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )

        return output


class MaxPool2d:
    """Max Pooling слой"""

    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.input_cache = None

    def forward(self, x, kernels):
        self.input_cache = x

        batch_size, channels, in_height, in_width = x.shape

        out_height = (in_height - self.kernel_size) // self.stride + 1
        out_width = (in_width - self.kernel_size) // self.stride + 1

        output = gpuarray.empty((batch_size, channels, out_height, out_width), dtype=np.float32)

        total_outputs = batch_size * channels * out_height * out_width
        block_size = 256
        grid_size = (total_outputs + block_size - 1) // block_size

        kernels['max_pool'](
            x, output,
            np.int32(batch_size), np.int32(channels), np.int32(in_height), np.int32(in_width),
            np.int32(out_height), np.int32(out_width), np.int32(self.kernel_size), np.int32(self.stride),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )

        return output

    def backward(self, grad_output, kernels):
        batch_size, channels, in_height, in_width = self.input_cache.shape
        out_height, out_width = grad_output.shape[2], grad_output.shape[3]

        grad_input = gpuarray.zeros_like(self.input_cache)

        total_outputs = batch_size * channels * out_height * out_width
        block_size = 256
        grid_size = (total_outputs + block_size - 1) // block_size

        # Create dummy max_indices (not used in simplified version)
        max_indices = gpuarray.zeros((total_outputs,), dtype=np.int32)

        kernels['max_pool_backward'](
            grad_output, self.input_cache, grad_input, max_indices,
            np.int32(batch_size), np.int32(channels),
            np.int32(in_height), np.int32(in_width), np.int32(out_height), np.int32(out_width),
            np.int32(self.kernel_size), np.int32(self.stride),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )

        return grad_input

    def forward(self, x, kernels):
        batch_size, channels, in_height, in_width = x.shape

        out_height = (in_height - self.kernel_size) // self.stride + 1
        out_width = (in_width - self.kernel_size) // self.stride + 1

        output = gpuarray.empty((batch_size, channels, out_height, out_width), dtype=np.float32)

        total_outputs = batch_size * channels * out_height * out_width
        block_size = 256
        grid_size = (total_outputs + block_size - 1) // block_size

        kernels['max_pool'](
            x, output,
            np.int32(batch_size), np.int32(channels), np.int32(in_height), np.int32(in_width),
            np.int32(out_height), np.int32(out_width), np.int32(self.kernel_size), np.int32(self.stride),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )

        return output


class AdaptiveAvgPool2d:
    """Adaptive Average Pooling слой"""

    def __init__(self, output_size):
        self.output_size = output_size

    def forward(self, x, kernels):
        batch_size, channels, in_height, in_width = x.shape
        output = gpuarray.empty((batch_size, channels, self.output_size, self.output_size), dtype=np.float32)

        # Простая реализация для output_size = 1
        if self.output_size == 1:
            total_outputs = batch_size * channels
            block_size = 256
            grid_size = (total_outputs + block_size - 1) // block_size

            kernels['avg_pool'](
                x, output,
                np.int32(batch_size), np.int32(channels), np.int32(in_height), np.int32(in_width),
                np.int32(in_height),
                block=(block_size, 1, 1), grid=(grid_size, 1)
            )

        return output


class Linear:
    """Полносвязный слой"""

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        # Инициализация весов
        std = np.sqrt(2.0 / in_features)
        self.weight = gpuarray.to_gpu(
            np.random.normal(0, std, (out_features, in_features)).astype(np.float32)
        )

        if bias:
            self.bias = gpuarray.to_gpu(np.zeros(out_features, dtype=np.float32))
        else:
            self.bias = None

    def forward(self, x, kernels):
        # Flatten input if needed
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)

        batch_size, input_size = x.shape
        output = gpuarray.empty((batch_size, self.out_features), dtype=np.float32)

        total_outputs = batch_size * self.out_features
        block_size = 256
        grid_size = (total_outputs + block_size - 1) // block_size

        bias_ptr = self.bias.ptr if self.bias is not None else 0

        kernels['linear'](
            x, self.weight, output, bias_ptr,
            np.int32(batch_size), np.int32(input_size), np.int32(self.out_features),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )

        return output


class BasicBlock:
    """Базовый блок ResNet"""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        self.conv1 = Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(out_channels)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x, kernels):
        identity = x

        out = self.conv1.forward(x, kernels)
        out = self.bn1.forward(out, kernels)
        out = self.relu.forward(out, kernels)

        out = self.conv2.forward(out, kernels)
        out = self.bn2.forward(out, kernels)

        if self.downsample is not None:
            identity = self.downsample.forward(x, kernels)

        # Element-wise addition
        output = gpuarray.empty_like(out)
        size = out.size
        block_size = 256
        grid_size = (size + block_size - 1) // block_size

        kernels['add'](
            out, identity, output, np.int32(size),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )

        out = self.relu.forward(output, kernels)
        return out


class Downsample:
    """Downsample блок для ResNet"""

    def __init__(self, in_channels, out_channels, stride):
        self.conv = Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        self.bn = BatchNorm2d(out_channels)

    def forward(self, x, kernels):
        x = self.conv.forward(x, kernels)
        x = self.bn.forward(x, kernels)
        return x


class ResNet18:
    """ResNet-18 архитектура"""

    def __init__(self, num_classes=1000):
        self.kernels = CudaKernels().kernels

        # Входной слой
        self.conv1 = Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU()
        self.maxpool = MaxPool2d(3, stride=2)

        # ResNet блоки
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Финальные слои
        self.avgpool = AdaptiveAvgPool2d(1)
        self.fc = Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []

        # Первый блок может изменить размерность
        downsample = None if in_channels == out_channels and stride == 1 else Downsample(in_channels, out_channels,
                                                                                         stride)
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))

        # Остальные блоки
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return layers

    def forward(self, x):
        # Входной блок
        x = self.conv1.forward(x, self.kernels)
        x = self.bn1.forward(x, self.kernels)
        x = self.relu.forward(x, self.kernels)
        x = self.maxpool.forward(x, self.kernels)

        # ResNet слои
        for block in self.layer1:
            x = block.forward(x, self.kernels)

        for block in self.layer2:
            x = block.forward(x, self.kernels)

        for block in self.layer3:
            x = block.forward(x, self.kernels)

        for block in self.layer4:
            x = block.forward(x, self.kernels)

        # Финальные слои
        x = self.avgpool.forward(x, self.kernels)
        x = self.fc.forward(x, self.kernels)

        return x

    def predict(self, image_path):
        """Предсказание для одного изображения"""
        # Загрузка и предобработка изображения
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0

        # ImageNet нормализация
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        for i in range(3):
            img_array[:, :, i] = (img_array[:, :, i] - mean[i]) / std[i]

        # Изменение формата с HWC на CHW и добавление batch dimension
        img_array = img_array.transpose(2, 0, 1)
        img_array = img_array[np.newaxis, :]  # Добавляем batch dimension

        # Копирование на GPU
        x = gpuarray.to_gpu(img_array)

        # Прямой проход
        output = self.forward(x)

        # Получение результата с GPU
        result = output.get()

        # Применение softmax
        exp_scores = np.exp(result - np.max(result, axis=1, keepdims=True))
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probabilities[0]


# Пример использования
def main():
    """Пример использования ResNet-18"""

    # Создание модели
    model = ResNet18(num_classes=1000)

    print("ResNet-18 модель создана успешно!")
    print(f"Общее количество параметров: ~11M")

    # Тестовый тензор
    batch_size = 1
    test_input = gpuarray.to_gpu(
        np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
    )

    print(f"Входной тензор: {test_input.shape}")

    # Прямой проход
    output = model.forward(test_input)

    print(f"Выходной тензор: {output.shape}")
    print("Прямой проход выполнен успешно!")

    # Пример предсказания (раскомментируйте если есть изображение)
    # probabilities = model.predict("path/to/your/image.jpg")
    # top5_indices = np.argsort(probabilities)[-5:][::-1]
    # print("Top 5 предсказаний:")
    # for i, idx in enumerate(top5_indices):
    #     print(f"{i+1}. Класс {idx}: {probabilities[idx]:.4f}")


if __name__ == "__main__":
    main()