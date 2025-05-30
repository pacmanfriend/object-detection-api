import os
import xml.etree.ElementTree as ET
import shutil

# Пути к данным
val_images_dir = r'D:\Python projects\imagenet\ILSVRC\Data\CLS-LOC\val'  # где лежат .JPEG файлы
val_annotations_dir = r'D:\Python projects\imagenet\ILSVRC\Annotations\CLS-LOC\val'  # где лежат .xml файлы

# Проходим по всем .xml файлам
for xml_file in os.listdir(val_annotations_dir):
    if not xml_file.endswith('.xml'):
        continue

    # Полный путь к XML
    xml_path = os.path.join(val_annotations_dir, xml_file)

    # Парсим XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Получаем имя изображения и ID класса
    image_name = root.find('filename').text + '.JPEG'
    class_id = root.find('object/name').text

    # Пути
    src_image_path = os.path.join(val_images_dir, image_name)
    dest_class_dir = os.path.join(val_images_dir, class_id)
    dest_image_path = os.path.join(dest_class_dir, image_name)

    # Создаем папку класса, если не существует
    os.makedirs(dest_class_dir, exist_ok=True)

    # Перемещаем изображение
    if os.path.exists(src_image_path):
        shutil.move(src_image_path, dest_image_path)
        print(f"Moved {image_name} -> {class_id}/")
    else:
        print(f"Файл {image_name} не найден!")

print("✅ Валидационные данные организованы по классам.")