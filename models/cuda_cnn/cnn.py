import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from PIL import Image
import os
import time

# Параметры сети
INPUT_SIZE = 224  # Размер входного изображения (224x224)
NUM_CLASSES = 1000  # Количество классов в ImageNet
BATCH_SIZE = 1  # Размер батча


# Класс для слоя свертки
class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Инициализация весов (фильтров) и смещений
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * scale
        self.biases = np.zeros(out_channels, dtype=np.float32)

        # Выделение памяти на GPU
        self.weights_gpu = cuda.mem_alloc(self.weights.nbytes)
        self.biases_gpu = cuda.mem_alloc(self.biases.nbytes)

        # Копирование данных на GPU
        cuda.memcpy_htod(self.weights_gpu, self.weights)
        cuda.memcpy_htod(self.biases_gpu, self.biases)

        # Компиляция CUDA ядра для свертки
        self.kernel_code = """
        __global__ void conv_forward(float *input, float *weights, float *biases, float *output, 
                                    int in_channels, int out_channels, int input_size, int kernel_size, 
                                    int stride, int padding, int output_size) {
            int out_x = blockIdx.x * blockDim.x + threadIdx.x;
            int out_y = blockIdx.y * blockDim.y + threadIdx.y;
            int out_z = blockIdx.z * blockDim.z + threadIdx.z;

            if (out_x >= output_size || out_y >= output_size || out_z >= out_channels) return;

            float sum = 0.0f;

            for (int in_c = 0; in_c < in_channels; in_c++) {
                for (int k_x = 0; k_x < kernel_size; k_x++) {
                    for (int k_y = 0; k_y < kernel_size; k_y++) {
                        int in_x = out_x * stride + k_x - padding;
                        int in_y = out_y * stride + k_y - padding;

                        if (in_x >= 0 && in_x < input_size && in_y >= 0 && in_y < input_size) {
                            int input_idx = in_c * input_size * input_size + in_y * input_size + in_x;
                            int weights_idx = out_z * in_channels * kernel_size * kernel_size + 
                                            in_c * kernel_size * kernel_size + k_y * kernel_size + k_x;
                            sum += input[input_idx] * weights[weights_idx];
                        }
                    }
                }
            }

            int output_idx = out_z * output_size * output_size + out_y * output_size + out_x;
            output[output_idx] = sum + biases[out_z];
        }
        """
        self.mod = SourceModule(self.kernel_code)
        self.conv_forward = self.mod.get_function("conv_forward")

    def forward(self, input_gpu, input_size, output_size):
        # Выделение памяти для вывода
        output_gpu = cuda.mem_alloc(self.out_channels * output_size * output_size * 4)

        # Запуск ядра
        block = (8, 8, 4)
        grid = (
            (output_size + block[0] - 1) // block[0],
            (output_size + block[1] - 1) // block[1],
            (self.out_channels + block[2] - 1) // block[2]
        )

        self.conv_forward(
            input_gpu, self.weights_gpu, self.biases_gpu, output_gpu,
            np.int32(self.in_channels), np.int32(self.out_channels),
            np.int32(input_size), np.int32(self.kernel_size),
            np.int32(self.stride), np.int32(self.padding),
            np.int32(output_size),
            block=block, grid=grid
        )

        return output_gpu


# Класс для MaxPool слоя
class MaxPoolLayer:
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

        # Компиляция CUDA ядра для MaxPool
        self.kernel_code = """
        __global__ void maxpool_forward(float *input, float *output, 
                                       int channels, int input_size, 
                                       int kernel_size, int stride, 
                                       int output_size) {
            int out_x = blockIdx.x * blockDim.x + threadIdx.x;
            int out_y = blockIdx.y * blockDim.y + threadIdx.y;
            int out_c = blockIdx.z * blockDim.z + threadIdx.z;

            if (out_x >= output_size || out_y >= output_size || out_c >= channels) return;

            float max_val = -FLT_MAX;

            for (int k_x = 0; k_x < kernel_size; k_x++) {
                for (int k_y = 0; k_y < kernel_size; k_y++) {
                    int in_x = out_x * stride + k_x;
                    int in_y = out_y * stride + k_y;

                    if (in_x < input_size && in_y < input_size) {
                        int input_idx = out_c * input_size * input_size + in_y * input_size + in_x;
                        max_val = max(max_val, input[input_idx]);
                    }
                }
            }

            int output_idx = out_c * output_size * output_size + out_y * output_size + out_x;
            output[output_idx] = max_val;
        }
        """
        self.mod = SourceModule(self.kernel_code)
        self.maxpool_forward = self.mod.get_function("maxpool_forward")

    def forward(self, input_gpu, input_size, channels):
        output_size = (input_size - self.kernel_size) // self.stride + 1
        output_gpu = cuda.mem_alloc(channels * output_size * output_size * 4)

        block = (8, 8, 4)
        grid = (
            (output_size + block[0] - 1) // block[0],
            (output_size + block[1] - 1) // block[1],
            (channels + block[2] - 1) // block[2]
        )

        self.maxpool_forward(
            input_gpu, output_gpu,
            np.int32(channels), np.int32(input_size),
            np.int32(self.kernel_size), np.int32(self.stride),
            np.int32(output_size),
            block=block, grid=grid
        )

        return output_gpu, output_size


# Класс для полносвязного слоя
class FCLayer:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # Инициализация весов и смещений
        scale = np.sqrt(2.0 / in_features)
        self.weights = np.random.randn(out_features, in_features).astype(np.float32) * scale
        self.biases = np.zeros(out_features, dtype=np.float32)

        # Выделение памяти на GPU
        self.weights_gpu = cuda.mem_alloc(self.weights.nbytes)
        self.biases_gpu = cuda.mem_alloc(self.biases.nbytes)

        # Копирование данных на GPU
        cuda.memcpy_htod(self.weights_gpu, self.weights)
        cuda.memcpy_htod(self.biases_gpu, self.biases)

        # Компиляция CUDA ядра для матричного умножения
        self.kernel_code = """
        __global__ void fc_forward(float *input, float *weights, float *biases, float *output, 
                                  int in_features, int out_features) {
            int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (out_idx >= out_features) return;

            float sum = 0.0f;

            for (int in_idx = 0; in_idx < in_features; in_idx++) {
                sum += input[in_idx] * weights[out_idx * in_features + in_idx];
            }

            output[out_idx] = sum + biases[out_idx];
        }
        """
        self.mod = SourceModule(self.kernel_code)
        self.fc_forward = self.mod.get_function("fc_forward")

    def forward(self, input_gpu):
        output_gpu = cuda.mem_alloc(self.out_features * 4)

        block = 256
        grid = (self.out_features + block - 1) // block

        self.fc_forward(
            input_gpu, self.weights_gpu, self.biases_gpu, output_gpu,
            np.int32(self.in_features), np.int32(self.out_features),
            block=(block, 1, 1), grid=(grid, 1)
        )

        return output_gpu


# Класс для ReLU активации
class ReLU:
    def __init__(self):
        # Компиляция CUDA ядра для ReLU
        self.kernel_code = """
        __global__ void relu_forward(float *input, float *output, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx >= size) return;

            output[idx] = fmaxf(0.0f, input[idx]);
        }
        """
        self.mod = SourceModule(self.kernel_code)
        self.relu_forward = self.mod.get_function("relu_forward")

    def forward(self, input_gpu, size):
        output_gpu = cuda.mem_alloc(size * 4)

        block = 256
        grid = (size + block - 1) // block

        self.relu_forward(
            input_gpu, output_gpu, np.int32(size),
            block=(block, 1, 1), grid=(grid, 1)
        )

        return output_gpu


# Простая CNN архитектура (упрощенный вариант ResNet)
class SimpleCNN:
    def __init__(self):
        # Создание слоев
        self.conv1 = ConvLayer(3, 64, 7, stride=2, padding=3)
        self.relu1 = ReLU()
        self.pool1 = MaxPoolLayer(3, stride=2)

        self.conv2 = ConvLayer(64, 128, 3, stride=1, padding=1)
        self.relu2 = ReLU()

        self.conv3 = ConvLayer(128, 256, 3, stride=1, padding=1)
        self.relu3 = ReLU()

        self.conv4 = ConvLayer(256, 512, 3, stride=1, padding=1)
        self.relu4 = ReLU()

        self.pool2 = MaxPoolLayer(7, stride=1)

        self.fc = FCLayer(512, NUM_CLASSES)

    def forward(self, input_gpu):
        # Первый блок
        x = self.conv1.forward(input_gpu, INPUT_SIZE, INPUT_SIZE // 2)
        x_size = INPUT_SIZE // 2
        channels = 64

        x = self.relu1.forward(x, 64 * x_size * x_size)
        x, x_size = self.pool1.forward(x, x_size, channels)

        # Второй блок
        x = self.conv2.forward(x, x_size, x_size)
        x = self.relu2.forward(x, 128 * x_size * x_size)

        # Третий блок
        x = self.conv3.forward(x, x_size, x_size)
        x = self.relu3.forward(x, 256 * x_size * x_size)

        # Четвертый блок
        x = self.conv4.forward(x, x_size, x_size)
        x = self.relu4.forward(x, 512 * x_size * x_size)

        # Финал
        x, _ = self.pool2.forward(x, x_size, 512)

        # Полносвязный слой
        x = self.fc.forward(x)

        return x


# Функция для загрузки и предобработки изображения
def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize((INPUT_SIZE, INPUT_SIZE))
    img = np.array(img).astype(np.float32)

    # Нормализация (примерные значения для ImageNet)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img / 255.0 - mean) / std

    # Преобразование в формат CHW
    img = np.transpose(img, (2, 0, 1))

    # Выделение памяти на GPU и копирование данных
    img_gpu = cuda.mem_alloc(img.nbytes)
    cuda.memcpy_htod(img_gpu, img)

    return img_gpu


# Функция для получения предсказания
def predict(image_path):
    # Загрузка модели
    model = SimpleCNN()

    # Загрузка и предобработка изображения
    input_gpu = load_image(image_path)

    # Прямой проход
    output_gpu = model.forward(input_gpu)

    # Копирование результатов обратно на CPU
    output = np.empty(NUM_CLASSES, dtype=np.float32)
    cuda.memcpy_dtoh(output, output_gpu)

    # Получение топ-5 предсказаний
    top5 = np.argsort(output)[-5:][::-1]

    return top5, output[top5]


# Пример использования
if __name__ == "__main__":
    image_path = "example.jpg"  # Замените на путь к вашему изображению

    if not os.path.exists(image_path):
        print(f"Файл {image_path} не найден")
    else:
        start_time = time.time()
        top5_classes, top5_probs = predict(image_path)
        end_time = time.time()

        print(f"Время выполнения: {end_time - start_time:.2f} секунд")
        print("Топ-5 предсказаний:")

        # Здесь должен быть ваш словарь с соответствием классов ImageNet
        # Для примера просто выведем номера классов
        for i, (class_idx, prob) in enumerate(zip(top5_classes, top5_probs)):
            print(f"{i + 1}. Класс {class_idx}: {prob:.4f}")
