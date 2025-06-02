import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from timeit import default_timer as timer


class MaxPool2D:
    def __init__(self, pool_size=(2, 2), stride=None):
        """
        MaxPool2D слой на PyCUDA

        Args:
            pool_size: размер окна пулинга (height, width)
            stride: шаг пулинга, если None то равен pool_size
        """
        self.pool_h, self.pool_w = pool_size
        self.stride_h, self.stride_w = stride if stride else pool_size

        # Компилируем CUDA кернелы
        self._compile_kernels()

        # Буферы для хранения индексов максимальных элементов
        self.max_indices = None
        self.input_shape = None

    def _compile_kernels(self):
        """Компилирует CUDA кернелы для прямого и обратного прохода"""

        # Кернел для прямого прохода
        forward_kernel = """
        __global__ void maxpool_forward(
            float* input, float* output, int* indices,
            int batch_size, int channels, int input_h, int input_w,
            int output_h, int output_w, int pool_h, int pool_w,
            int stride_h, int stride_w
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_outputs = batch_size * channels * output_h * output_w;

            if (idx >= total_outputs) return;

            // Вычисляем координаты в выходном тензоре
            int w_out = idx % output_w;
            int h_out = (idx / output_w) % output_h;
            int c = (idx / (output_w * output_h)) % channels;
            int n = idx / (output_w * output_h * channels);

            // Вычисляем область входного тензора для пулинга
            int h_start = h_out * stride_h;
            int w_start = w_out * stride_w;
            int h_end = min(h_start + pool_h, input_h);
            int w_end = min(w_start + pool_w, input_w);

            float max_val = -3.402823466e+38F; // -FLT_MAX
            int max_idx = -1;

            // Находим максимальное значение в окне пулинга
            for (int h = h_start; h < h_end; h++) {
                for (int w = w_start; w < w_end; w++) {
                    int input_idx = ((n * channels + c) * input_h + h) * input_w + w;
                    if (input[input_idx] > max_val) {
                        max_val = input[input_idx];
                        max_idx = input_idx;
                    }
                }
            }

            output[idx] = max_val;
            indices[idx] = max_idx;
        }
        """

        # Кернел для обратного прохода
        backward_kernel = """
        __global__ void maxpool_backward(
            float* grad_input, float* grad_output, int* indices,
            int batch_size, int channels, int input_h, int input_w,
            int output_h, int output_w
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_outputs = batch_size * channels * output_h * output_w;

            if (idx >= total_outputs) return;

            int max_idx = indices[idx];
            if (max_idx >= 0) {
                atomicAdd(&grad_input[max_idx], grad_output[idx]);
            }
        }
        """

        # Компилируем модуль
        mod = SourceModule(forward_kernel + backward_kernel)
        self.forward_func = mod.get_function("maxpool_forward")
        self.backward_func = mod.get_function("maxpool_backward")

    def forward(self, x):
        """
        Прямой проход MaxPool

        Args:
            x: входной тензор размером (batch_size, channels, height, width)

        Returns:
            output: выходной тензор после пулинга
        """
        batch_size, channels, input_h, input_w = x.shape
        self.input_shape = x.shape

        # Вычисляем размеры выходного тензора
        output_h = (input_h - self.pool_h) // self.stride_h + 1
        output_w = (input_w - self.pool_w) // self.stride_w + 1

        # Создаем выходные массивы на GPU
        output_shape = (batch_size, channels, output_h, output_w)
        output = gpuarray.zeros(output_shape, dtype=np.float32)

        # Массив для хранения индексов максимальных элементов
        self.max_indices = gpuarray.zeros(output_shape, dtype=np.int32)

        # Конвертируем входной тензор в GPU array если нужно
        if isinstance(x, np.ndarray):
            x_gpu = gpuarray.to_gpu(x.astype(np.float32))
        else:
            x_gpu = x

        # Параметры запуска кернела
        total_outputs = batch_size * channels * output_h * output_w
        block_size = 256
        grid_size = (total_outputs + block_size - 1) // block_size

        block = (32, 8, 4)
        grid = ((output_w + block[0] - 1) // block[0], (output_h + block[1] - 1) // block[1], 1)

        # Запускаем кернел прямого прохода
        self.forward_func(
            x_gpu, output, self.max_indices,
            np.int32(batch_size), np.int32(channels),
            np.int32(input_h), np.int32(input_w),
            np.int32(output_h), np.int32(output_w),
            np.int32(self.pool_h), np.int32(self.pool_w),
            np.int32(self.stride_h), np.int32(self.stride_w),
            block=block, grid=grid
        )

        return output

    def backward(self, grad_output):
        """
        Обратный проход MaxPool

        Args:
            grad_output: градиенты от следующего слоя

        Returns:
            grad_input: градиенты для входного тензора
        """
        if self.max_indices is None or self.input_shape is None:
            raise ValueError("Сначала выполните прямой проход!")

        batch_size, channels, input_h, input_w = self.input_shape
        output_h, output_w = grad_output.shape[2], grad_output.shape[3]

        # Создаем массив градиентов для входа
        grad_input = gpuarray.zeros(self.input_shape, dtype=np.float32)

        # Конвертируем градиенты в GPU array если нужно
        if isinstance(grad_output, np.ndarray):
            grad_output_gpu = gpuarray.to_gpu(grad_output.astype(np.float32))
        else:
            grad_output_gpu = grad_output

        # Параметры запуска кернела
        total_outputs = batch_size * channels * output_h * output_w
        block_size = 256
        grid_size = (total_outputs + block_size - 1) // block_size

        block = (256, 1, 1)
        grid = ((output_w + block[0] - 1) // block[0], (output_h + block[1] - 1) // block[1], 1)

        # Запускаем кернел обратного прохода
        self.backward_func(
            grad_input, grad_output_gpu, self.max_indices,
            np.int32(batch_size), np.int32(channels),
            np.int32(input_h), np.int32(input_w),
            np.int32(output_h), np.int32(output_w),
            block=block, grid=grid
        )

        return grad_input


# Пример использования
def test_maxpool():
    """Тестирование MaxPool слоя"""

    # Создаем тестовые данные
    batch_size, channels, height, width = 32, 256, 56, 56
    x = np.random.randn(batch_size, channels, height, width).astype(np.float32)

    print(f"Входной тензор: {x.shape}")
    print(f"Примерные значения входа:\n{x[0, 0, :4, :4]}\n")

    # Создаем MaxPool слой
    maxpool = MaxPool2D(pool_size=(2, 2), stride=(2, 2))

    # Прямой проход

    start = timer()
    output = maxpool.forward(x)
    end = timer()
    print(f"Время: {end - start}")

    output_cpu = output.get()  # Копируем результат с GPU на CPU

    print(f"Выходной тензор: {output_cpu.shape}")
    print(f"Результат MaxPool:\n{output_cpu[0, 0, :2, :2]}\n")

    # Тестируем обратный проход
    grad_output = np.ones_like(output_cpu)
    grad_input = maxpool.backward(grad_output)
    grad_input_cpu = grad_input.get()

    print(f"Градиенты входа: {grad_input_cpu.shape}")
    print(f"Примерные градиенты:\n{grad_input_cpu[0, 0, :4, :4]}\n")

    # Проверяем корректность: градиенты должны быть только в позициях максимумов
    print("Проверка корректности обратного прохода:")
    print(f"Сумма градиентов на входе: {np.sum(grad_input_cpu)}")
    print(f"Сумма градиентов на выходе: {np.sum(grad_output)}")
    print(f"Количество ненулевых градиентов: {np.count_nonzero(grad_input_cpu)}")


if __name__ == "__main__":
    test_maxpool()
