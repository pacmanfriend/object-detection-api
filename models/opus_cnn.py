import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pycuda.cumath as cumath
from pycuda.elementwise import ElementwiseKernel
import cv2
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
import queue

# CUDA kernels для прямого прохода
forward_kernels = """
__global__ void conv2d_forward(
    float *input, float *weight, float *bias, float *output,
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int kernel_size,
    int out_height, int out_width, int stride, int padding
) {
    int b = blockIdx.z;
    int out_c = blockIdx.y;
    int out_y = blockIdx.x * blockDim.x + threadIdx.x;
    int out_x = threadIdx.y;

    if (b >= batch_size || out_c >= out_channels || 
        out_y >= out_height || out_x >= out_width) return;

    float sum = 0.0f;

    for (int in_c = 0; in_c < in_channels; in_c++) {
        for (int k_y = 0; k_y < kernel_size; k_y++) {
            for (int k_x = 0; k_x < kernel_size; k_x++) {
                int in_y = out_y * stride - padding + k_y;
                int in_x = out_x * stride - padding + k_x;

                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    int input_idx = ((b * in_channels + in_c) * in_height + in_y) * in_width + in_x;
                    int weight_idx = ((out_c * in_channels + in_c) * kernel_size + k_y) * kernel_size + k_x;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    sum += bias[out_c];
    int output_idx = ((b * out_channels + out_c) * out_height + out_y) * out_width + out_x;
    output[output_idx] = sum;
}

__global__ void maxpool2d_forward(
    float *input, float *output, int *indices,
    int batch_size, int channels, int in_height, int in_width,
    int pool_size, int stride, int out_height, int out_width
) {
    int b = blockIdx.z;
    int c = blockIdx.y;
    int out_y = blockIdx.x * blockDim.x + threadIdx.x;
    int out_x = threadIdx.y;

    if (b >= batch_size || c >= channels || 
        out_y >= out_height || out_x >= out_width) return;

    float max_val = -1e30f;
    int max_idx = -1;

    for (int p_y = 0; p_y < pool_size; p_y++) {
        for (int p_x = 0; p_x < pool_size; p_x++) {
            int in_y = out_y * stride + p_y;
            int in_x = out_x * stride + p_x;

            if (in_y < in_height && in_x < in_width) {
                int input_idx = ((b * channels + c) * in_height + in_y) * in_width + in_x;
                if (input[input_idx] > max_val) {
                    max_val = input[input_idx];
                    max_idx = input_idx;
                }
            }
        }
    }

    int output_idx = ((b * channels + c) * out_height + out_y) * out_width + out_x;
    output[output_idx] = max_val;
    indices[output_idx] = max_idx;
}

__global__ void batchnorm_forward_train(
    float *input, float *output, float *gamma, float *beta,
    float *mean, float *var, float *running_mean, float *running_var,
    int batch_size, int channels, int spatial_size, 
    float eps, float momentum
) {
    int c = blockIdx.x;
    if (c >= channels) return;

    // Вычисление среднего
    float sum = 0.0f;
    int total_elements = batch_size * spatial_size;

    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < spatial_size; s++) {
            int idx = (b * channels + c) * spatial_size + s;
            sum += input[idx];
        }
    }

    float batch_mean = sum / total_elements;
    mean[c] = batch_mean;

    // Вычисление дисперсии
    float var_sum = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < spatial_size; s++) {
            int idx = (b * channels + c) * spatial_size + s;
            float diff = input[idx] - batch_mean;
            var_sum += diff * diff;
        }
    }

    float batch_var = var_sum / total_elements;
    var[c] = batch_var;

    // Обновление running statistics
    running_mean[c] = momentum * running_mean[c] + (1 - momentum) * batch_mean;
    running_var[c] = momentum * running_var[c] + (1 - momentum) * batch_var;

    // Нормализация и масштабирование
    float std = sqrtf(batch_var + eps);

    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < spatial_size; s++) {
            int idx = (b * channels + c) * spatial_size + s;
            float normalized = (input[idx] - batch_mean) / std;
            output[idx] = gamma[c] * normalized + beta[c];
        }
    }
}

__global__ void cross_entropy_loss(
    float *predictions, int *labels, float *loss,
    int batch_size, int num_classes
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    int label = labels[b];
    float pred = predictions[b * num_classes + label];
    loss[b] = -logf(fmaxf(pred, 1e-7f));
}
"""

# CUDA kernels для обратного прохода
backward_kernels = """
__global__ void conv2d_backward_input(
    float *grad_output, float *weight, float *grad_input,
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int kernel_size,
    int out_height, int out_width, int stride, int padding
) {
    int b = blockIdx.z;
    int in_c = blockIdx.y;
    int in_y = blockIdx.x * blockDim.x + threadIdx.x;
    int in_x = threadIdx.y;

    if (b >= batch_size || in_c >= in_channels || 
        in_y >= in_height || in_x >= in_width) return;

    float sum = 0.0f;

    for (int out_c = 0; out_c < out_channels; out_c++) {
        for (int k_y = 0; k_y < kernel_size; k_y++) {
            for (int k_x = 0; k_x < kernel_size; k_x++) {
                int out_y = (in_y + padding - k_y);
                int out_x = (in_x + padding - k_x);

                if (out_y % stride == 0 && out_x % stride == 0) {
                    out_y /= stride;
                    out_x /= stride;

                    if (out_y >= 0 && out_y < out_height && 
                        out_x >= 0 && out_x < out_width) {
                        int grad_idx = ((b * out_channels + out_c) * out_height + out_y) * out_width + out_x;
                        int weight_idx = ((out_c * in_channels + in_c) * kernel_size + k_y) * kernel_size + k_x;
                        sum += grad_output[grad_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    int input_idx = ((b * in_channels + in_c) * in_height + in_y) * in_width + in_x;
    grad_input[input_idx] = sum;
}

__global__ void conv2d_backward_weight(
    float *input, float *grad_output, float *grad_weight,
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int kernel_size,
    int out_height, int out_width, int stride, int padding
) {
    int out_c = blockIdx.z;
    int in_c = blockIdx.y;
    int k_y = blockIdx.x * blockDim.x + threadIdx.x;
    int k_x = threadIdx.y;

    if (out_c >= out_channels || in_c >= in_channels || 
        k_y >= kernel_size || k_x >= kernel_size) return;

    float sum = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        for (int out_y = 0; out_y < out_height; out_y++) {
            for (int out_x = 0; out_x < out_width; out_x++) {
                int in_y = out_y * stride - padding + k_y;
                int in_x = out_x * stride - padding + k_x;

                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    int input_idx = ((b * in_channels + in_c) * in_height + in_y) * in_width + in_x;
                    int grad_idx = ((b * out_channels + out_c) * out_height + out_y) * out_width + out_x;
                    sum += input[input_idx] * grad_output[grad_idx];
                }
            }
        }
    }

    int weight_idx = ((out_c * in_channels + in_c) * kernel_size + k_y) * kernel_size + k_x;
    grad_weight[weight_idx] = sum;
}

__global__ void conv2d_backward_bias(
    float *grad_output, float *grad_bias,
    int batch_size, int out_channels, int out_height, int out_width
) {
    int out_c = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_c >= out_channels) return;

    float sum = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        for (int y = 0; y < out_height; y++) {
            for (int x = 0; x < out_width; x++) {
                int idx = ((b * out_channels + out_c) * out_height + y) * out_width + x;
                sum += grad_output[idx];
            }
        }
    }

    grad_bias[out_c] = sum;
}

__global__ void maxpool2d_backward(
    float *grad_output, int *indices, float *grad_input,
    int total_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    int input_idx = indices[idx];
    if (input_idx >= 0) {
        atomicAdd(&grad_input[input_idx], grad_output[idx]);
    }
}

__global__ void batchnorm_backward(
    float *grad_output, float *input, float *gamma,
    float *mean, float *var, float *grad_input,
    float *grad_gamma, float *grad_beta,
    int batch_size, int channels, int spatial_size, float eps
) {
    int c = blockIdx.x;
    if (c >= channels) return;

    float batch_mean = mean[c];
    float batch_var = var[c];
    float std = sqrtf(batch_var + eps);
    int total_elements = batch_size * spatial_size;

    // Вычисление градиентов для gamma и beta
    float dgamma = 0.0f;
    float dbeta = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < spatial_size; s++) {
            int idx = (b * channels + c) * spatial_size + s;
            float normalized = (input[idx] - batch_mean) / std;
            dgamma += grad_output[idx] * normalized;
            dbeta += grad_output[idx];
        }
    }

    grad_gamma[c] = dgamma;
    grad_beta[c] = dbeta;

    // Вычисление градиента по входу
    float dvar = 0.0f;
    float dmean = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < spatial_size; s++) {
            int idx = (b * channels + c) * spatial_size + s;
            //float normalized = (input[idx] - batch_mean) / std;
            dvar += grad_output[idx] * gamma[c] * (input[idx] - batch_mean) * -0.5f * powf(std, -3.0f);
            dmean += grad_output[idx] * gamma[c] * -1.0f / std;
        }
    }

    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < spatial_size; s++) {
            int idx = (b * channels + c) * spatial_size + s;
            grad_input[idx] = grad_output[idx] * gamma[c] / std + 
                             dvar * 2.0f * (input[idx] - batch_mean) / total_elements + 
                             dmean / total_elements;
        }
    }
}

__global__ void softmax_cross_entropy_backward(
    float *predictions, int *labels, float *grad_output,
    int batch_size, int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / num_classes;
    int c = idx % num_classes;

    if (b >= batch_size) return;

    float grad = predictions[idx];
    if (c == labels[b]) {
        grad -= 1.0f;
    }
    grad_output[idx] = grad / batch_size;
}
"""

# CUDA kernels для оптимизатора
optimizer_kernels = """
__global__ void sgd_momentum_update(
    float *param, float *grad, float *momentum,
    float lr, float momentum_coef, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    momentum[idx] = momentum_coef * momentum[idx] + grad[idx];
    param[idx] -= lr * momentum[idx];
}

__global__ void adam_update(
    float *param, float *grad, float *m, float *v,
    float lr, float beta1, float beta2, float eps,
    int t, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Обновление моментов
    m[idx] = beta1 * m[idx] + (1 - beta1) * grad[idx];
    v[idx] = beta2 * v[idx] + (1 - beta2) * grad[idx] * grad[idx];

    // Коррекция смещения
    float m_hat = m[idx] / (1 - powf(beta1, t));
    float v_hat = v[idx] / (1 - powf(beta2, t));

    // Обновление параметров
    param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
}
"""

# Компиляция всех kernels
mod_forward = SourceModule(forward_kernels)
mod_backward = SourceModule(backward_kernels)
mod_optimizer = SourceModule(optimizer_kernels)

# Получение функций
conv2d_forward = mod_forward.get_function("conv2d_forward")
maxpool2d_forward = mod_forward.get_function("maxpool2d_forward")
batchnorm_forward_train = mod_forward.get_function("batchnorm_forward_train")
cross_entropy_loss_func = mod_forward.get_function("cross_entropy_loss")

conv2d_backward_input = mod_backward.get_function("conv2d_backward_input")
conv2d_backward_weight = mod_backward.get_function("conv2d_backward_weight")
conv2d_backward_bias = mod_backward.get_function("conv2d_backward_bias")
maxpool2d_backward = mod_backward.get_function("maxpool2d_backward")
batchnorm_backward = mod_backward.get_function("batchnorm_backward")
softmax_ce_backward = mod_backward.get_function("softmax_cross_entropy_backward")

sgd_momentum_update = mod_optimizer.get_function("sgd_momentum_update")
adam_update = mod_optimizer.get_function("adam_update")

# Активации
relu_forward = ElementwiseKernel(
    "float *input, float *output",
    "output[i] = fmaxf(0.0f, input[i])",
    "relu_forward"
)

relu_backward = ElementwiseKernel(
    "float *grad_output, float *input, float *grad_input",
    "grad_input[i] = input[i] > 0 ? grad_output[i] : 0.0f",
    "relu_backward"
)


# Базовый класс для слоев
class Layer:
    def __init__(self):
        self.training = True
        self.params = {}
        self.grads = {}
        self.cache = {}

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def get_params(self):
        return self.params

    def get_grads(self):
        return self.grads


class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Инициализация весов
        fan_in = in_channels * kernel_size * kernel_size
        std = np.sqrt(2.0 / fan_in)

        self.params['weight'] = gpuarray.to_gpu(
            np.random.normal(0, std, (out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32)
        )
        self.params['bias'] = gpuarray.zeros(out_channels, dtype=np.float32)

        # Градиенты
        self.grads['weight'] = gpuarray.zeros_like(self.params['weight'])
        self.grads['bias'] = gpuarray.zeros_like(self.params['bias'])

    def forward(self, x):
        self.cache['input'] = x
        batch_size, _, in_height, in_width = x.shape
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        output = gpuarray.zeros((batch_size, self.out_channels, out_height, out_width), dtype=np.float32)

        block = (16, 16, 1)
        grid = ((out_height + block[0] - 1) // block[0], self.out_channels, batch_size)

        conv2d_forward(
            x, self.params['weight'], self.params['bias'], output,
            np.int32(batch_size), np.int32(self.in_channels), np.int32(self.out_channels),
            np.int32(in_height), np.int32(in_width), np.int32(self.kernel_size),
            np.int32(out_height), np.int32(out_width),
            np.int32(self.stride), np.int32(self.padding),
            block=block, grid=grid
        )

        return output

    def backward(self, grad_output):
        x = self.cache['input']
        batch_size, _, in_height, in_width = x.shape
        _, _, out_height, out_width = grad_output.shape

        # Градиент по входу
        grad_input = gpuarray.zeros_like(x)
        block = (16, 16, 1)
        grid = ((in_height + block[0] - 1) // block[0], self.in_channels, batch_size)

        conv2d_backward_input(
            grad_output, self.params['weight'], grad_input,
            np.int32(batch_size), np.int32(self.in_channels), np.int32(self.out_channels),
            np.int32(in_height), np.int32(in_width), np.int32(self.kernel_size),
            np.int32(out_height), np.int32(out_width),
            np.int32(self.stride), np.int32(self.padding),
            block=block, grid=grid
        )

        # Градиент по весам
        self.grads['weight'].fill(0)
        block = (self.kernel_size, self.kernel_size, 1)
        grid = (1, self.in_channels, self.out_channels)

        conv2d_backward_weight(
            x, grad_output, self.grads['weight'],
            np.int32(batch_size), np.int32(self.in_channels), np.int32(self.out_channels),
            np.int32(in_height), np.int32(in_width), np.int32(self.kernel_size),
            np.int32(out_height), np.int32(out_width),
            np.int32(self.stride), np.int32(self.padding),
            block=block, grid=grid
        )

        # Градиент по bias
        block = (256, 1, 1)
        grid = ((self.out_channels + block[0] - 1) // block[0], 1, 1)

        conv2d_backward_bias(
            grad_output, self.grads['bias'],
            np.int32(batch_size), np.int32(self.out_channels),
            np.int32(out_height), np.int32(out_width),
            block=block, grid=grid
        )

        return grad_input


class MaxPool2d(Layer):
    def __init__(self, pool_size, stride=None):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size

    def forward(self, x):
        batch_size, channels, in_height, in_width = x.shape
        out_height = (in_height - self.pool_size) // self.stride + 1
        out_width = (in_width - self.pool_size) // self.stride + 1

        output = gpuarray.zeros((batch_size, channels, out_height, out_width), dtype=np.float32)
        indices = gpuarray.zeros((batch_size, channels, out_height, out_width), dtype=np.int32)

        self.cache['indices'] = indices
        self.cache['input_shape'] = x.shape

        block = (16, 16, 1)
        grid = ((out_height + block[0] - 1) // block[0], channels, batch_size)

        maxpool2d_forward(
            x, output, indices,
            np.int32(batch_size), np.int32(channels),
            np.int32(in_height), np.int32(in_width),
            np.int32(self.pool_size), np.int32(self.stride),
            np.int32(out_height), np.int32(out_width),
            block=block, grid=grid
        )

        return output

    def backward(self, grad_output):
        indices = self.cache['indices']
        input_shape = self.cache['input_shape']

        grad_input = gpuarray.zeros(input_shape, dtype=np.float32)

        total_size = grad_output.size
        block = (256, 1, 1)
        grid = ((total_size + block[0] - 1) // block[0], 1, 1)

        maxpool2d_backward(
            grad_output, indices, grad_input,
            np.int32(total_size),
            block=block, grid=grid
        )

        return grad_input


class BatchNorm2d(Layer):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.params['gamma'] = gpuarray.ones(num_features, dtype=np.float32)
        self.params['beta'] = gpuarray.zeros(num_features, dtype=np.float32)

        self.grads['gamma'] = gpuarray.zeros_like(self.params['gamma'])
        self.grads['beta'] = gpuarray.zeros_like(self.params['beta'])

        self.running_mean = gpuarray.zeros(num_features, dtype=np.float32)
        self.running_var = gpuarray.ones(num_features, dtype=np.float32)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        spatial_size = height * width

        output = gpuarray.zeros_like(x)

        if self.training:
            mean = gpuarray.zeros(channels, dtype=np.float32)
            var = gpuarray.zeros(channels, dtype=np.float32)

            self.cache['input'] = x
            self.cache['mean'] = mean
            self.cache['var'] = var

            block = (1, 1, 1)
            grid = (channels, 1, 1)

            batchnorm_forward_train(
                x, output, self.params['gamma'], self.params['beta'],
                mean, var, self.running_mean, self.running_var,
                np.int32(batch_size), np.int32(channels),
                np.int32(spatial_size), np.float32(self.eps),
                np.float32(self.momentum),
                block=block, grid=grid
            )
        else:
            # Inference mode - использовать running statistics
            # Упрощенная версия для примера
            pass

        return output

    def backward(self, grad_output):
        x = self.cache['input']
        mean = self.cache['mean']
        var = self.cache['var']

        batch_size, channels, height, width = x.shape
        spatial_size = height * width

        grad_input = gpuarray.zeros_like(x)

        block = (1, 1, 1)
        grid = (channels, 1, 1)

        batchnorm_backward(
            grad_output, x, self.params['gamma'],
            mean, var, grad_input,
            self.grads['gamma'], self.grads['beta'],
            np.int32(batch_size), np.int32(channels),
            np.int32(spatial_size), np.float32(self.eps),
            block=block, grid=grid
        )

        return grad_input


class ReLU(Layer):
    def forward(self, x):
        self.cache['input'] = x
        output = gpuarray.zeros_like(x)
        relu_forward(x, output)
        return output

    def backward(self, grad_output):
        x = self.cache['input']
        grad_input = gpuarray.zeros_like(x)
        relu_backward(grad_output, x, grad_input)
        return grad_input


class Linear(Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Xavier initialization
        std = np.sqrt(2.0 / (in_features + out_features))
        self.params['weight'] = gpuarray.to_gpu(
            np.random.normal(0, std, (out_features, in_features)).astype(np.float32)
        )
        self.params['bias'] = gpuarray.zeros(out_features, dtype=np.float32)

        self.grads['weight'] = gpuarray.zeros_like(self.params['weight'])
        self.grads['bias'] = gpuarray.zeros_like(self.params['bias'])

    def forward(self, x):
        self.cache['input'] = x
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)

        output = gpuarray.zeros((batch_size, self.out_features), dtype=np.float32)

        # Простое матричное умножение без cuBLAS для стабильности
        # output = x_flat @ weight.T + bias
        weight_t = self.params['weight'].reshape(self.out_features, self.in_features).T

        for i in range(batch_size):
            output[i] = gpuarray.dot(x_flat[i], weight_t) + self.params['bias']

        return output

    def backward(self, grad_output):
        x = self.cache['input']
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)

        # Градиент по весам: grad_w = x.T @ grad_output
        for i in range(batch_size):
            self.grads['weight'] += gpuarray.outer(grad_output[i], x_flat[i])

        # Градиент по bias
        self.grads['bias'] = gpuarray.sum(grad_output, axis=0)

        # Градиент по входу: grad_x = grad_output @ weight
        grad_input_flat = gpuarray.zeros((batch_size, self.in_features), dtype=np.float32)

        for i in range(batch_size):
            grad_input_flat[i] = gpuarray.dot(grad_output[i], self.params['weight'])

        # Восстанавливаем форму
        grad_input = grad_input_flat.reshape(x.shape)

        return grad_input


class GlobalAvgPool2d(Layer):
    def forward(self, x):
        self.cache['input_shape'] = x.shape
        batch_size, channels, height, width = x.shape
        spatial_size = height * width

        # Reshape и усреднение
        x_reshaped = x.reshape(batch_size, channels, spatial_size)
        output = gpuarray.zeros((batch_size, channels), dtype=np.float32)

        for b in range(batch_size):
            for c in range(channels):
                output[b, c] = gpuarray.sum(x_reshaped[b, c]) / spatial_size

        return output

    def backward(self, grad_output):
        input_shape = self.cache['input_shape']
        batch_size, channels, height, width = input_shape
        spatial_size = height * width

        # Распределяем градиент равномерно
        grad_output_expanded = grad_output.reshape(batch_size, channels, 1, 1)
        grad_input = gpuarray.empty(input_shape, dtype=np.float32)

        # Используем broadcasting
        for b in range(batch_size):
            for c in range(channels):
                grad_input[b, c, :, :] = grad_output[b, c] / spatial_size

        return grad_input


# Оптимизаторы
class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param_dict in self.params:
            for grad in param_dict.values():
                grad.fill(0)


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum

        # Инициализация моментов
        self.velocities = []
        for layer in params:
            vel_dict = {}
            for name, param in layer.get_params().items():
                vel_dict[name] = gpuarray.zeros_like(param)
            self.velocities.append(vel_dict)

    def step(self):
        for layer in self.params:
            param_dict = layer.get_params()
            grad_dict = layer.get_grads()
            vel_dict = self.velocities[self.params.index(layer)]

            for name in param_dict:
                param = param_dict[name]
                grad = grad_dict[name]
                vel = vel_dict[name]

                size = param.size
                block = (256, 1, 1)
                grid = ((size + block[0] - 1) // block[0], 1, 1)

                sgd_momentum_update(
                    param, grad, vel,
                    np.float32(self.lr), np.float32(self.momentum),
                    np.int32(size),
                    block=block, grid=grid
                )


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        # Инициализация моментов
        self.m = []
        self.v = []
        for layer in params:
            m_dict = {}
            v_dict = {}
            for name, param in layer.get_params().items():
                m_dict[name] = gpuarray.zeros_like(param)
                v_dict[name] = gpuarray.zeros_like(param)
            self.m.append(m_dict)
            self.v.append(v_dict)

    def step(self):
        self.t += 1

        for i, layer in enumerate(self.params):
            param_dict = layer.get_params()
            grad_dict = layer.get_grads()
            m_dict = self.m[i]
            v_dict = self.v[i]

            for name in param_dict:
                param = param_dict[name]
                grad = grad_dict[name]
                m = m_dict[name]
                v = v_dict[name]

                size = param.size
                block = (256, 1, 1)
                grid = ((size + block[0] - 1) // block[0], 1, 1)

                adam_update(
                    param, grad, m, v,
                    np.float32(self.lr), np.float32(self.beta1),
                    np.float32(self.beta2), np.float32(self.eps),
                    np.int32(self.t), np.int32(size),
                    block=block, grid=grid
                )


# Data loader для ImageNet
class ImageNetDataLoader:
    def __init__(self, data_dir, batch_size, split='train', num_workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.split = split
        self.num_workers = num_workers

        # Загрузка списка файлов
        self.image_paths = []
        self.labels = []

        # Здесь должна быть загрузка реальных путей и меток
        # Для примера создаем синтетические данные
        self.num_samples = 1000  # В реальности ~1.2M для train
        self.num_classes = 1000

        # ImageNet statistics
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        self.queue = queue.Queue(maxsize=10)
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def __len__(self):
        return self.num_samples // self.batch_size

    def preprocess_image(self, img_path):
        """Предобработка изображения"""
        # В реальности здесь загрузка изображения
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Для демонстрации создаем случайное изображение
        img = np.random.rand(224, 224, 3).astype(np.float32)

        # Нормализация
        img = (img - self.mean) / self.std

        # Transpose to CHW
        img = np.transpose(img, (2, 0, 1))

        return img

    def load_batch(self, indices):
        """Загрузка батча"""
        batch_images = []
        batch_labels = []

        for idx in indices:
            # В реальности загрузка из файла
            img = self.preprocess_image(f"dummy_path_{idx}.jpg")
            label = np.random.randint(0, self.num_classes)

            batch_images.append(img)
            batch_labels.append(label)

        images = np.stack(batch_images)
        labels = np.array(batch_labels)

        return gpuarray.to_gpu(images), gpuarray.to_gpu(labels.astype(np.int32))

    def __iter__(self):
        indices = np.random.permutation(self.num_samples)

        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]

            if len(batch_indices) < self.batch_size:
                continue

            yield self.load_batch(batch_indices)


# Модель CNN для ImageNet
class ImageNetCNN:
    def __init__(self, num_classes=1000):
        # Создание слоев
        self.layers = []

        # Начальные слои
        self.layers.append(Conv2d(3, 64, kernel_size=7, stride=2, padding=3))
        self.layers.append(BatchNorm2d(64))
        self.layers.append(ReLU())
        self.layers.append(MaxPool2d(3, stride=2))

        # Основные блоки
        self.layers.append(Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
        self.layers.append(BatchNorm2d(128))
        self.layers.append(ReLU())

        self.layers.append(Conv2d(128, 256, kernel_size=3, stride=2, padding=1))
        self.layers.append(BatchNorm2d(256))
        self.layers.append(ReLU())

        self.layers.append(Conv2d(256, 512, kernel_size=3, stride=2, padding=1))
        self.layers.append(BatchNorm2d(512))
        self.layers.append(ReLU())

        # Классификатор
        self.layers.append(GlobalAvgPool2d())
        self.layers.append(Linear(512, num_classes))

        self.num_classes = num_classes

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def train(self):
        for layer in self.layers:
            layer.training = True

    def eval(self):
        for layer in self.layers:
            layer.training = False

    def get_params(self):
        params = []
        for layer in self.layers:
            if layer.params:
                params.append(layer)
        return params


# Функция потерь
class CrossEntropyLoss:
    def __init__(self):
        self.cache = {}

    def forward(self, predictions, labels):
        batch_size = predictions.shape[0]

        # Softmax
        from pycuda.compiler import SourceModule
        softmax_mod = SourceModule("""
        __global__ void softmax_forward(
            float *input, float *output, int batch_size, int num_classes
        ) {
            int b = blockIdx.x * blockDim.x + threadIdx.x;
            if (b >= batch_size) return;

            float max_val = -1e30f;
            for (int i = 0; i < num_classes; i++) {
                max_val = fmaxf(max_val, input[b * num_classes + i]);
            }

            float sum = 0.0f;
            for (int i = 0; i < num_classes; i++) {
                float exp_val = expf(input[b * num_classes + i] - max_val);
                output[b * num_classes + i] = exp_val;
                sum += exp_val;
            }

            for (int i = 0; i < num_classes; i++) {
                output[b * num_classes + i] /= sum;
            }
        }
        """)

        softmax_func = softmax_mod.get_function("softmax_forward")

        softmax_output = gpuarray.zeros_like(predictions)

        block = (256, 1, 1)
        grid = ((batch_size + block[0] - 1) // block[0], 1, 1)

        softmax_func(
            predictions, softmax_output,
            np.int32(batch_size), np.int32(predictions.shape[1]),
            block=block, grid=grid
        )

        self.cache['softmax'] = softmax_output
        self.cache['labels'] = labels

        # Cross entropy loss
        losses = gpuarray.zeros(batch_size, dtype=np.float32)

        cross_entropy_loss_func(
            softmax_output, labels, losses,
            np.int32(batch_size), np.int32(predictions.shape[1]),
            block=block, grid=grid
        )

        return gpuarray.sum(losses) / batch_size

    def backward(self):
        softmax = self.cache['softmax']
        labels = self.cache['labels']

        batch_size, num_classes = softmax.shape
        grad_output = gpuarray.zeros_like(softmax)

        block = (256, 1, 1)
        grid = ((batch_size * num_classes + block[0] - 1) // block[0], 1, 1)

        softmax_ce_backward(
            softmax, labels, grad_output,
            np.int32(batch_size), np.int32(num_classes),
            block=block, grid=grid
        )

        return grad_output


# Основная функция обучения
def train_model(model, train_loader, val_loader, num_epochs=90,
                initial_lr=0.1, momentum=0.9, weight_decay=1e-4):
    """Обучение модели"""

    # Оптимизатор
    optimizer = SGD(model.get_params(), lr=initial_lr, momentum=momentum)
    # optimizer = Adam(model.get_params(), lr=0.001)

    # Loss function
    criterion = CrossEntropyLoss()

    # Learning rate scheduler
    def adjust_learning_rate(optimizer, epoch):
        """Уменьшение learning rate каждые 30 эпох"""
        lr = initial_lr * (0.1 ** (epoch // 30))
        optimizer.lr = lr
        return lr

    # Обучение
    for epoch in range(num_epochs):
        model.train()

        # Настройка learning rate
        current_lr = adjust_learning_rate(optimizer, epoch)
        print(f"\nEpoch {epoch + 1}/{num_epochs}, LR: {current_lr:.6f}")

        epoch_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model.forward(images)
            loss = criterion.forward(outputs, labels)

            # Backward pass
            grad_output = criterion.backward()
            model.backward(grad_output)

            # Обновление весов
            optimizer.step()
            optimizer.zero_grad()

            # Статистика
            epoch_loss += loss.get()

            # Подсчет accuracy
            predictions = outputs.get()
            pred_labels = np.argmax(predictions, axis=1)
            true_labels = labels.get()

            correct += np.sum(pred_labels == true_labels)
            total += labels.shape[0]

            if batch_idx % 10 == 0:
                print(f"Batch [{batch_idx}/{len(train_loader)}], "
                      f"Loss: {loss.get():.4f}, "
                      f"Acc: {100 * correct / total:.2f}%")

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        print(f"Training Loss: {epoch_loss / len(train_loader):.4f}")
        print(f"Training Accuracy: {100 * correct / total:.2f}%")

        # Валидация
        if val_loader and (epoch + 1) % 5 == 0:
            val_loss, val_acc = validate_model(model, val_loader, criterion)
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_acc:.2f}%")


def validate_model(model, val_loader, criterion):
    """Валидация модели"""
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with cuda.Context.get_current() as ctx:
        for images, labels in val_loader:
            outputs = model.forward(images)
            loss = criterion.forward(outputs, labels)

            val_loss += loss.get()

            predictions = outputs.get()
            pred_labels = np.argmax(predictions, axis=1)
            true_labels = labels.get()

            correct += np.sum(pred_labels == true_labels)
            total += labels.shape[0]

    return val_loss / len(val_loader), 100 * correct / total


# Сохранение и загрузка модели
def save_model(model, filepath):
    """Сохранение весов модели"""
    state_dict = {}

    for i, layer in enumerate(model.layers):
        if layer.params:
            for name, param in layer.params.items():
                key = f"layer_{i}_{name}"
                state_dict[key] = param.get()

    np.savez(filepath, **state_dict)
    print(f"Model saved to {filepath}")


def load_model(model, filepath):
    """Загрузка весов модели"""
    state_dict = np.load(filepath)

    for i, layer in enumerate(model.layers):
        if layer.params:
            for name, param in layer.params.items():
                key = f"layer_{i}_{name}"
                if key in state_dict:
                    param.set(state_dict[key])

    print(f"Model loaded from {filepath}")


# Пример использования
def main():
    # Параметры
    batch_size = 32
    num_epochs = 90
    data_dir = r"D:\Python projects\imagenet\ILSVRC\Data\CLS-LOC"

    # Создание модели
    print("Creating model...")
    model = ImageNetCNN(num_classes=1000)

    # Создание data loaders
    print("Creating data loaders...")
    train_loader = ImageNetDataLoader(data_dir, batch_size, split='train')
    val_loader = ImageNetDataLoader(data_dir, batch_size, split='val')

    # Обучение
    print("Starting training...")
    train_model(model, train_loader, val_loader,
                num_epochs=num_epochs, initial_lr=0.1)

    # Сохранение модели
    save_model(model, "imagenet_model.npz")


if __name__ == "__main__":
    main()
