import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import os
import sys
import torch
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')

# Убираем привязку к GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Закомментировать эту строку

######################################### SETUP #########################################
# Создаем простой парсер аргументов для CPU
class Args:
    def __init__(self):
        self.device = torch.device('cpu')  # Принудительно используем CPU
        self.seed = 42
        self.resume = r"models\kaggle\working\sela_output\2025-09-20_07-55-58\best_model.pth"

        # Добавьте другие необходимые аргументы здесь

# Создаем объект args
args = args = parser.parse_arguments()
commons.make_deterministic(args.seed)


# Имитируем функцию make_deterministic
def make_deterministic(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

make_deterministic(args.seed)

# Трансформации
t = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def show_feature_map(imgpath, conv_features):
    # Загружаем изображение с обработкой ошибок
    try:
        img = Image.open(imgpath).convert('RGB')
    except Exception as e:
        print(f"Ошибка загрузки изображения: {e}")
        # Создаем тестовое изображение если нужно
        img = Image.new('RGB', (224, 224), color='red')
    
    heat = conv_features.squeeze(0)
    heat_mean = torch.mean(heat, dim=0)
    heatmap = heat_mean.detach().cpu().numpy()
    print(f"Heatmap shape: {heatmap.shape}")
    
    heatmap = -heatmap  # Инвертируем если нужно
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)) 
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Сохраняем heatmap
    cv2.imwrite('heatmap_original.jpg', heatmap)

    # Визуализация
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(heatmap)
    plt.title('Heatmap')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    superimg = heatmap * 0.6 + np.array(img)[:, :, ::-1]
    plt.imshow(superimg[:, :, ::-1])
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    cv2.imwrite('heatmap_result.jpg', superimg)

# Загрузка изображения с проверкой
imgpath = r"C:\Users\Егор\VsCode project\SelaVPR\image\img_pair\img0.jpg"
if not os.path.exists(imgpath):
    print(f"Изображение не найдено: {imgpath}")
    # Создаем тестовое изображение
    os.makedirs("./image/img_pair", exist_ok=True)
    test_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    Image.fromarray(test_img).save(imgpath)
    print(f"Создано тестовое изображение: {imgpath}")

try:
    img = Image.open(imgpath).convert('RGB')
    img_tensor = t(img).unsqueeze(0).to(args.device)
    print(f"Изображение загружено: {img.size} -> Тензор: {img_tensor.shape}")
except Exception as e:
    print(f"Ошибка обработки изображения: {e}")
    sys.exit(1)

##### Load trained model and extract feature map
try:
    # Импортируем необходимые модули
    import commons
    from network import GeoLocalizationNet
    
    model = GeoLocalizationNet(args)
    model = model.to(args.device)
    
    # Для CPU не используем DataParallel
    # model = torch.nn.DataParallel(model)  # Закомментировать для CPU
    
    if args.resume is not None and os.path.exists(args.resume):
        print(f"Загружаем модель из: {args.resume}")
        # Для CPU используем map_location='cpu'
        state_dict = torch.load(args.resume, map_location='cpu')
        
        # Проверяем структуру файла
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        
        # Удаляем префикс 'module.' если он есть (для моделей, обученных с DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Удаляем 'module.'
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        print("Модель успешно загружена!")
    else:
        print(f"Файл модели не найден: {args.resume}")
        print("Будет использована случайно инициализированная модель")
    
    # Извлекаем фичи
    model.eval()
    with torch.no_grad():
        feature = model.backbone(img_tensor)  # Для модели без DataParallel
    
    print(f"Feature keys: {feature.keys()}")
    
    # Обрабатываем фичи
    if "x_norm_patchtokens" in feature:
        feature_processed = feature["x_norm_patchtokens"].view(-1, 16, 16, 1024).permute(0, 3, 1, 2)
        print(f"Processed feature shape: {feature_processed.shape}")
        
        show_feature_map(imgpath, feature_processed)
    else:
        print("Ключ 'x_norm_patchtokens' не найден в feature")
        print("Доступные ключи:", list(feature.keys()))
        
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что файлы commons.py и network.py находятся в текущей директории")
except Exception as e:
    print(f"Ошибка при работе модели: {e}")
    import traceback
    traceback.print_exc()
