import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import os
import sys
import torch
from torchvision import transforms
import parser
import commons
from network import GeoLocalizationNet
import warnings
warnings.filterwarnings('ignore')

# Импортируем rerun
import rerun as rr

# Устанавливаем устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

######################################### SETUP #########################################
args = parser.parse_arguments()

# Добавляем отсутствующий атрибут registers
if not hasattr(args, 'registers'):
    args.registers = False

# Устанавливаем устройство
args.device = device

commons.make_deterministic(args.seed)

# Инициализируем rerun
rr.init("selavpr_visualization", spawn=True)

t = transforms.Compose([transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def show_feature_map_with_rerun(imgpath, conv_features, original_img):
    """Визуализация feature map с использованием rerun"""
    
    # Логируем оригинальное изображение
    rr.log_image("original/image", np.array(original_img))
    
    # Подготовка feature map для визуализации
    heat = conv_features.squeeze(0)
    heat_mean = torch.mean(heat, dim=0)
    heatmap = heat_mean.detach().cpu().numpy()
    
    # Нормализуем heatmap
    heatmap = -heatmap  # Инвертируем если нужно
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    
    # Resize heatmap к размеру оригинального изображения
    heatmap_resized = cv2.resize(heatmap, (original_img.size[0], original_img.size[1]))
    
    # Логируем heatmap
    rr.log_image("features/heatmap", heatmap_resized)
    rr.log_histogram("features/heatmap_distribution", heatmap_resized.flatten())
    
    # Логируем отдельные каналы feature map (первые 16)
    num_channels_to_show = min(16, conv_features.shape[1])
    for i in range(num_channels_to_show):
        channel_feature = conv_features[0, i].detach().cpu().numpy()
        channel_resized = cv2.resize(channel_feature, (original_img.size[0], original_img.size[1]))
        rr.log_image(f"features/channel_{i:02d}", channel_resized)
    
    # Создаем наложенное изображение
    heatmap_vis = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
    
    # Конвертируем оригинальное изображение в BGR для OpenCV
    original_cv = np.array(original_img)[:, :, ::-1]  # RGB to BGR
    
    # Наложение heatmap на оригинальное изображение
    superimposed = cv2.addWeighted(original_cv, 0.7, heatmap_colored, 0.3, 0)
    
    # Логируем наложенное изображение
    rr.log_image("superimposed/result", superimposed)
    
    # Логируем метаданные
    rr.log_text_entry("metadata/image_path", imgpath)
    rr.log_text_entry("metadata/feature_shape", str(conv_features.shape))
    rr.log_scalar("metadata/heatmap_min", float(np.min(heatmap)))
    rr.log_scalar("metadata/heatmap_max", float(np.max(heatmap)))
    rr.log_scalar("metadata/heatmap_mean", float(np.mean(heatmap)))
    
    # Традиционная визуализация (сохраняем для совместимости)
    cv2.imwrite('heatmap_original.jpg', heatmap_colored)
    cv2.imwrite('heatmap_result.jpg', superimposed)
    
    print("Визуализация завершена! Откройте Rerun viewer для просмотра результатов.")

# Убедитесь, что путь к изображению правильный
imgpath = "image/img_pair/img0.jpg"
if not os.path.exists(imgpath):
    # Создаем тестовое изображение, если его нет
    os.makedirs("./image/img_pair", exist_ok=True)
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite(imgpath, test_img)
    print(f"Created test image at {imgpath}")

# Загружаем и обрабатываем изображение
original_img = Image.open(imgpath)
print(f"Изображение загружено: {original_img.size}")

img_tensor = t(original_img).unsqueeze(0).to(device)
print(f"Тензор изображения: {img_tensor.shape}")

##### Загружаем модель
model = GeoLocalizationNet(args)
model = model.to(device)

# Загружаем веса модели
if args.resume is not None:
    state_dict = torch.load(args.resume, map_location=device)
    
    # Обрабатываем разные форматы state_dict
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    
    # Удаляем "module." из ключей если нужно
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    print("Модель успешно загружена")
else:
    print("Warning: No model path provided")

model.eval()

##### Извлекаем features
with torch.no_grad():
    features = model.backbone(img_tensor)

# Обработка features в зависимости от структуры
if isinstance(features, dict) and "x_norm_patchtokens" in features:
    features_processed = features["x_norm_patchtokens"].view(-1, 16, 16, 1024).permute(0, 3, 1, 2)
    print(f"Processed features shape: {features_processed.shape}")
else:
    features_processed = features
    print(f"Features shape: {features_processed.shape}")

# Визуализация с rerun
show_feature_map_with_rerun(imgpath, features_processed, original_img)

print("Готово! Rerun записывает данные...")