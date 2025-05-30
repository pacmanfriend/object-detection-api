import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Гиперпараметры
BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 0.0001
NUM_CLASSES = 2  # ImageNet содержит 1000 классов


# Простая CNN архитектура
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 14 * 14, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Тренировочный цикл
def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 9:  # Каждые 10 батчей выводим лог
            print(f'Epoch [{epoch}], Step [{batch_idx + 1}], Loss: {loss.item():.4f}')

    print(f'Epoch [{epoch}] завершён. Средний loss: {running_loss / len(train_loader):.4f}')


# Валидационный цикл
def validate(model, device, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Точность на валидации: {100 * correct / total:.2f}%')


def run_simple_cnn():
    # Проверяем доступность CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Используется устройство: {device}")

    # Преобразования изображений
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Сформировали преобразования изображений")

    # Загрузка данных (замените путь на свой)
    train_dataset = datasets.ImageFolder(root=r'D:\Python projects\imagenet2\data\train',
                                         transform=transform)
    val_dataset = datasets.ImageFolder(root=r'D:\Python projects\imagenet2\data\val', transform=transform)

    print("Загрузили датасеты")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    print("Загрузили лоадеры")

    # Инициализация модели, функции потерь и оптимизатора
    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Проиниализировали модель")

    # Основной цикл обучения
    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, criterion, optimizer, epoch)
        validate(model, device, val_loader)


if __name__ == '__main__':
    run_simple_cnn()
