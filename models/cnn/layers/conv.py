import time
from timeit import default_timer as timer

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray


class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Инициализация весов и смещений
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * 0.01
        self.bias = np.zeros(out_channels).astype(np.float32)

        # Градиенты
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

        # Компиляция CUDA кернелов
        self._compile_kernels()

    def _compile_kernels(self):
        """Компиляция CUDA кернелов для прямого и обратного прохода"""

        # Кернел для прямого прохода
        forward_kernel = """
        __global__ void conv_forward(float* input, float* weights, float* bias, float* output,
                                   int batch_size, int in_channels, int out_channels,
                                   int input_height, int input_width,
                                   int output_height, int output_width,
                                   int kernel_size, int stride, int padding) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_outputs = batch_size * out_channels * output_height * output_width;

            if (idx >= total_outputs) return;

            // Декодирование индексов
            int batch = idx / (out_channels * output_height * output_width);
            int remainder = idx % (out_channels * output_height * output_width);
            int out_ch = remainder / (output_height * output_width);
            remainder = remainder % (output_height * output_width);
            int out_h = remainder / output_width;
            int out_w = remainder % output_width;

            float sum = 0.0f;

            // Свертка
            for (int in_ch = 0; in_ch < in_channels; in_ch++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int in_h = out_h * stride - padding + kh;
                        int in_w = out_w * stride - padding + kw;

                        if (in_h >= 0 && in_h < input_height && 
                            in_w >= 0 && in_w < input_width) {

                            int input_idx = batch * in_channels * input_height * input_width +
                                          in_ch * input_height * input_width +
                                          in_h * input_width + in_w;

                            int weight_idx = out_ch * in_channels * kernel_size * kernel_size +
                                           in_ch * kernel_size * kernel_size +
                                           kh * kernel_size + kw;

                            sum += input[input_idx] * weights[weight_idx];
                        }
                    }
                }
            }

            output[idx] = sum + bias[out_ch];
        }
        """

        # Кернел для вычисления градиентов по входу
        backward_input_kernel = """
        __global__ void conv_backward_input(float* grad_output, float* weights, float* grad_input,
                                          int batch_size, int in_channels, int out_channels,
                                          int input_height, int input_width,
                                          int output_height, int output_width,
                                          int kernel_size, int stride, int padding) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_inputs = batch_size * in_channels * input_height * input_width;

            if (idx >= total_inputs) return;

            // Декодирование индексов
            int batch = idx / (in_channels * input_height * input_width);
            int remainder = idx % (in_channels * input_height * input_width);
            int in_ch = remainder / (input_height * input_width);
            remainder = remainder % (input_height * input_width);
            int in_h = remainder / input_width;
            int in_w = remainder % input_width;

            float sum = 0.0f;

            // Обратная свертка
            for (int out_ch = 0; out_ch < out_channels; out_ch++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int out_h = (in_h + padding - kh);
                        int out_w = (in_w + padding - kw);

                        if (out_h % stride == 0 && out_w % stride == 0) {
                            out_h /= stride;
                            out_w /= stride;

                            if (out_h >= 0 && out_h < output_height &&
                                out_w >= 0 && out_w < output_width) {

                                int grad_output_idx = batch * out_channels * output_height * output_width +
                                                    out_ch * output_height * output_width +
                                                    out_h * output_width + out_w;

                                int weight_idx = out_ch * in_channels * kernel_size * kernel_size +
                                               in_ch * kernel_size * kernel_size +
                                               kh * kernel_size + kw;

                                sum += grad_output[grad_output_idx] * weights[weight_idx];
                            }
                        }
                    }
                }
            }

            grad_input[idx] = sum;
        }
        """

        # Кернел для вычисления градиентов по весам
        backward_weights_kernel = """
        __global__ void conv_backward_weights(float* input, float* grad_output, float* grad_weights,
                                            int batch_size, int in_channels, int out_channels,
                                            int input_height, int input_width,
                                            int output_height, int output_width,
                                            int kernel_size, int stride, int padding) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_weights = out_channels * in_channels * kernel_size * kernel_size;

            if (idx >= total_weights) return;

            // Декодирование индексов
            int out_ch = idx / (in_channels * kernel_size * kernel_size);
            int remainder = idx % (in_channels * kernel_size * kernel_size);
            int in_ch = remainder / (kernel_size * kernel_size);
            remainder = remainder % (kernel_size * kernel_size);
            int kh = remainder / kernel_size;
            int kw = remainder % kernel_size;

            float sum = 0.0f;

            // Вычисление градиента по весам
            for (int batch = 0; batch < batch_size; batch++) {
                for (int out_h = 0; out_h < output_height; out_h++) {
                    for (int out_w = 0; out_w < output_width; out_w++) {
                        int in_h = out_h * stride - padding + kh;
                        int in_w = out_w * stride - padding + kw;

                        if (in_h >= 0 && in_h < input_height &&
                            in_w >= 0 && in_w < input_width) {

                            int input_idx = batch * in_channels * input_height * input_width +
                                          in_ch * input_height * input_width +
                                          in_h * input_width + in_w;

                            int grad_output_idx = batch * out_channels * output_height * output_width +
                                                out_ch * output_height * output_width +
                                                out_h * output_width + out_w;

                            sum += input[input_idx] * grad_output[grad_output_idx];
                        }
                    }
                }
            }

            grad_weights[idx] = sum;
        }
        """

        # Кернел для вычисления градиентов по смещениям
        backward_bias_kernel = """
        __global__ void conv_backward_bias(float* grad_output, float* grad_bias,
                                         int batch_size, int out_channels,
                                         int output_height, int output_width) {

            int out_ch = blockIdx.x * blockDim.x + threadIdx.x;

            if (out_ch >= out_channels) return;

            float sum = 0.0f;

            for (int batch = 0; batch < batch_size; batch++) {
                for (int h = 0; h < output_height; h++) {
                    for (int w = 0; w < output_width; w++) {
                        int idx = batch * out_channels * output_height * output_width +
                                out_ch * output_height * output_width +
                                h * output_width + w;
                        sum += grad_output[idx];
                    }
                }
            }

            grad_bias[out_ch] = sum;
        }
        """

        # Компиляция модулей
        self.mod_forward = SourceModule(forward_kernel)
        self.mod_backward_input = SourceModule(backward_input_kernel)
        self.mod_backward_weights = SourceModule(backward_weights_kernel)
        self.mod_backward_bias = SourceModule(backward_bias_kernel)

        # Получение функций
        self.conv_forward = self.mod_forward.get_function("conv_forward")
        self.conv_backward_input = self.mod_backward_input.get_function("conv_backward_input")
        self.conv_backward_weights = self.mod_backward_weights.get_function("conv_backward_weights")
        self.conv_backward_bias = self.mod_backward_bias.get_function("conv_backward_bias")

    def forward(self, input_data):
        """Прямой проход"""
        batch_size, in_channels, input_height, input_width = input_data.shape

        # Вычисление размеров выхода
        output_height = (input_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (input_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Сохранение входа для обратного прохода
        self.input_cache = input_data.copy()

        # Подготовка данных на GPU
        input_gpu = gpuarray.to_gpu(input_data.astype(np.float32))
        weights_gpu = gpuarray.to_gpu(self.weights)
        bias_gpu = gpuarray.to_gpu(self.bias)
        output_gpu = gpuarray.zeros((batch_size, self.out_channels, output_height, output_width), dtype=np.float32)

        # Настройка блоков и сетки
        total_outputs = batch_size * self.out_channels * output_height * output_width
        # block_size = 16
        # grid_size = (total_outputs + block_size - 1) // block_size

        block = (16, 16, 4)
        grid = ((output_width + block[0] - 1) // block[0], (output_height + block[1] - 1) // block[1], 1)

        # Запуск кернела
        self.conv_forward(
            input_gpu, weights_gpu, bias_gpu, output_gpu,
            np.int32(batch_size), np.int32(in_channels), np.int32(self.out_channels),
            np.int32(input_height), np.int32(input_width),
            np.int32(output_height), np.int32(output_width),
            np.int32(self.kernel_size), np.int32(self.stride), np.int32(self.padding),
            block=block, grid=grid)

        return output_gpu.get()

    def backward(self, grad_output):
        """Обратный проход"""
        batch_size, out_channels, output_height, output_width = grad_output.shape
        input_shape = self.input_cache.shape

        # GPU данные
        grad_output_gpu = gpuarray.to_gpu(grad_output.astype(np.float32))
        weights_gpu = gpuarray.to_gpu(self.weights)
        input_gpu = gpuarray.to_gpu(self.input_cache)

        # 1. Градиент по входу
        grad_input_gpu = gpuarray.zeros(input_shape, dtype=np.float32)

        total_inputs = np.prod(input_shape)
        block_size = 512
        grid_size = (total_inputs + block_size - 1) // block_size

        self.conv_backward_input(
            grad_output_gpu, weights_gpu, grad_input_gpu,
            np.int32(batch_size), np.int32(self.in_channels), np.int32(self.out_channels),
            np.int32(input_shape[2]), np.int32(input_shape[3]),
            np.int32(output_height), np.int32(output_width),
            np.int32(self.kernel_size), np.int32(self.stride), np.int32(self.padding),
            block=(int(block_size), 1, 1), grid=(int(grid_size), 1)
        )

        # 2. Градиент по весам
        grad_weights_gpu = gpuarray.zeros(self.weights.shape, dtype=np.float32)

        total_weights = np.prod(self.weights.shape)
        grid_size = (total_weights + block_size - 1) // block_size

        self.conv_backward_weights(
            input_gpu, grad_output_gpu, grad_weights_gpu,
            np.int32(batch_size), np.int32(self.in_channels), np.int32(self.out_channels),
            np.int32(input_shape[2]), np.int32(input_shape[3]),
            np.int32(output_height), np.int32(output_width),
            np.int32(self.kernel_size), np.int32(self.stride), np.int32(self.padding),
            block=(int(block_size), 1, 1), grid=(int(grid_size), 1, 1)
        )

        # 3. Градиент по смещениям
        grad_bias_gpu = gpuarray.zeros(self.bias.shape, dtype=np.float32)

        grid_size = (self.out_channels + block_size - 1) // block_size

        self.conv_backward_bias(
            grad_output_gpu, grad_bias_gpu,
            np.int32(batch_size), np.int32(self.out_channels),
            np.int32(output_height), np.int32(output_width),
            block=(int(block_size), 1, 1), grid=(int(grid_size), 1)
        )

        # Сохранение градиентов
        self.grad_weights = grad_weights_gpu.get()
        self.grad_bias = grad_bias_gpu.get()

        return grad_input_gpu.get()

    def update_parameters(self, learning_rate=0.001):
        """Обновление параметров"""
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias


# Пример использования и тестирование
if __name__ == "__main__":
    # Создание слоя
    conv_layer = ConvLayer(in_channels=3, out_channels=128, kernel_size=5, stride=1, padding=1)

    # Тестовые данные
    batch_size = 16
    input_data = np.random.randn(batch_size, 3, 1024, 1024).astype(np.float32)

    print("Тестирование сверточного слоя на PyCUDA:")
    print(f"Вход: {input_data.shape}")

    start = timer()
    # Прямой проход
    output = conv_layer.forward(input_data)
    print(f"Выход: {output.shape}")

    end = timer()
    print(f"Время: {end - start}")

    # Обратный проход
    grad_output = np.random.randn(*output.shape).astype(np.float32)
    grad_input = conv_layer.backward(grad_output)



    print(f"Градиент по входу: {grad_input.shape}")
    print(f"Градиент по весам: {conv_layer.grad_weights.shape}")
    print(f"Градиент по смещениям: {conv_layer.grad_bias.shape}")

    # Обновление параметров
    conv_layer.update_parameters(learning_rate=0.001)
    print("Параметры обновлены!")

    print("\nТест пройден успешно!")
