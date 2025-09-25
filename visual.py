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

# Инициализируем rerun с явными параметрами
try:
    rr.init("selavpr_visualization", spawn=True)
    print("Rerun инициализирован успешно")
except Exception as e:
    print(f"Ошибка инициализации Rerun: {e}")
    # Пробуем альтернативный способ
    rr.init("selavpr_visualization", spawn=False)

t = transforms.Compose([transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def show_feature_map_with_rerun(imgpath, conv_features, original_img, mode="mean", channel_idx=0, save_dir="./outputs"):
    """
    Визуализация feature map с использованием rerun и сохранением на диск.

    Аргументы:
        imgpath: путь к исходному изображению
        conv_features: тензор features [B, C, H, W]
        original_img: PIL.Image (оригинал)
        mode: "mean" | "channel" | "grid"
        channel_idx: индекс канала для mode="channel"
        save_dir: папка для сохранения результатов
    """

    os.makedirs(save_dir, exist_ok=True)

    try:
        rr.log("original/image", rr.Image(np.array(original_img)))
    except Exception as e:
        print(f"Ошибка при логировании изображения: {e}")
        return

    if len(conv_features.shape) == 4:
        heat = conv_features.squeeze(0)  # [C, H, W]
    else:
        print(f"Неожиданная размерность features: {conv_features.shape}")
        return

    try:
        if mode == "mean":
            heatmap = torch.mean(heat, dim=0).detach().cpu().numpy()

        elif mode == "channel":
            channel_idx = min(channel_idx, heat.shape[0] - 1)
            heatmap = heat[channel_idx].detach().cpu().numpy()

        elif mode == "grid":
            num_channels = min(16, heat.shape[0])
            rows, cols = 4, 4
            channel_imgs = []
            for i in range(num_channels):
                ch = heat[i].detach().cpu().numpy()
                ch = (ch - np.min(ch)) / (np.max(ch) - np.min(ch) + 1e-8)
                ch = cv2.resize(ch, (original_img.size[0] // cols, original_img.size[1] // rows))
                channel_imgs.append(ch)
            grid = np.zeros((rows * channel_imgs[0].shape[0], cols * channel_imgs[0].shape[1]))
            for i in range(rows):
                for j in range(cols):
                    idx = i * cols + j
                    if idx < len(channel_imgs):
                        grid[i*channel_imgs[0].shape[0]:(i+1)*channel_imgs[0].shape[0],
                             j*channel_imgs[0].shape[1]:(j+1)*channel_imgs[0].shape[1]] = channel_imgs[idx]
            heatmap = grid

        else:
            print(f"Неизвестный режим визуализации: {mode}")
            return

        # нормализация
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        heatmap_resized = cv2.resize(heatmap, (original_img.size[0], original_img.size[1]))

        # --- логирование ---
        rr.log("features/heatmap", rr.Image(heatmap_resized))

        # --- сохранение ---
        heatmap_gray = (255 * heatmap_resized).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, "heatmap.png"), heatmap_gray)

        heatmap_colored = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_dir, "heatmap_colored.png"), heatmap_colored)

        original_cv = np.array(original_img)[:, :, ::-1]  # RGB→BGR
        superimposed = cv2.addWeighted(original_cv, 0.7, heatmap_colored, 0.3, 0)
        cv2.imwrite(os.path.join(save_dir, "superimposed.png"), superimposed)

        rr.log("superimposed/result", rr.Image(superimposed))

        print(f"Heatmap сохранён в папку: {os.path.abspath(save_dir)}")

    except Exception as e:
        print(f"Ошибка при обработке feature maps: {e}")
        import traceback
        traceback.print_exc()


# ФИКСИРУЕМ ПУТЬ К ИЗОБРАЖЕНИЮ
# Используем абсолютный путь или корректный относительный
imgpath = "./image/img_pair/img0.jpg"  # Добавим ./ в начало


# Создаем директорию если не существует
os.makedirs(os.path.dirname(imgpath), exist_ok=True)

if not os.path.exists(imgpath):
    # Создаем тестовое изображение, если его нет
    print(f"Создаем тестовое изображение: {imgpath}")
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # Сохраняем как RGB
    test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(imgpath, test_img_rgb)

# Проверяем существование файла
if not os.path.exists(imgpath):
    print(f"ОШИБКА: Файл {imgpath} не существует!")
    sys.exit(1)

print(f"Путь к изображению: {os.path.abspath(imgpath)}")

try:
    # Загружаем и обрабатываем изображение
    original_img = Image.open(imgpath).convert('RGB')
    print(f"Изображение загружено: {original_img.size}")
except Exception as e:
    print(f"Ошибка загрузки изображения: {e}")
    sys.exit(1)

try:
    img_tensor = t(original_img).unsqueeze(0).to(device)
    print(f"Тензор изображения: {img_tensor.shape}")
except Exception as e:
    print(f"Ошибка преобразования изображения: {e}")
    sys.exit(1)

##### Загружаем модель
try:
    model = GeoLocalizationNet(args)
    model = model.to(device)

    # Загружаем веса модели
    if args.resume is not None:
        print(f"Загружаем веса из: {args.resume}")
        state_dict = torch.load(args.resume, map_location=device, weights_only=False)
        
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

    print(f"Raw features type: {type(features)}")
    
    # Обработка features в зависимости от структуры
    if isinstance(features, dict):
        print("Features are dictionary, keys:", features.keys())
        if "x_norm_patchtokens" in features:
            features_processed = features["x_norm_patchtokens"]
            print(f"Patch tokens shape: {features_processed.shape}")
            # Преобразуем в изображение-like формат
            if len(features_processed.shape) == 3:
                # [batch, tokens, features] -> [batch, features, sqrt(tokens), sqrt(tokens)]
                batch_size, num_tokens, feat_dim = features_processed.shape
                grid_size = int(np.sqrt(num_tokens))
                if grid_size * grid_size == num_tokens:
                    features_processed = features_processed.view(batch_size, grid_size, grid_size, feat_dim)
                    features_processed = features_processed.permute(0, 3, 1, 2)
                    print(f"Processed features shape: {features_processed.shape}")
                else:
                    print(f"Не могу преобразовать {num_tokens} токенов в квадратную сетку")
                    # Используем первый канал как есть
                    features_processed = features_processed.unsqueeze(-1).unsqueeze(-1)
            else:
                features_processed = features_processed.unsqueeze(-1).unsqueeze(-1)
        else:
            # Берем первый доступный тензор
            for key, value in features.items():
                if torch.is_tensor(value):
                    features_processed = value
                    print(f"Using tensor from key '{key}': {features_processed.shape}")
                    break
    else:
        features_processed = features
        print(f"Features are tensor: {features_processed.shape}")

    # Визуализация с rerun
    show_feature_map_with_rerun(imgpath, features_processed, original_img, mode="mean")

except Exception as e:
    print(f"Ошибка при работе с моделью: {e}")
    import traceback
    traceback.print_exc()

print("Готово! Проверьте Rerun viewer.")