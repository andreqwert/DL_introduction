import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import os, shutil
from tqdm import tqdm
from scipy import spatial

root_dir = '/home/user/Desktop/imgs/'   # путь до папки с исходными изображениями
model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
model.eval();


def calculate_mse(imageA, imageB):
    return torch.sum((imageA - imageB) ** 2)


def get_unsimilar_image_paths_MSE(img_paths, features_all_images, mse_thr=200, top=-1):
    """
    Найти непохожие друг на друга картинки.
    ВХОД:
    img_paths: полный путь до картинок;
    features_all_images: список с embedding'ом картинок (соответствующих img_paths);
    mse_thr: порог схожести для пары изображений. Если он ниже чем, то изображения между собой очень схожи;
    top: вывести top изображений похожих на данное изображение. Если -1, то считаются все.

    ВЫХОД: 
    img_paths: пути максимально непохожих друг на друга кратинок"""

    d = {}
    for imgA_path, i in zip(img_paths, range(len(features_all_images))):
        MSE_paths = []   # список с путями до из-й с превышенным MSE
        for imgB_path, j in zip(img_paths, range(len(features_all_images))):
            if imgA_path != imgB_path:   # не сравниваем из-е с самим собой
                mse = calculate_mse(features_all_images[i], features_all_images[j]).cpu().detach().numpy()
                print(imgB_path, mse)
                if mse < mse_thr:
                    MSE_paths.append(imgB_path)
                d[f'{imgA_path}'] = MSE_paths[:top]   # получаем словарь вида: <путь до из-я>: <top_максимально_похожих_на_него>

    similar_image_paths = np.array(sum(d.values(), []))   # словарь --> список из ключей словаря
    img_paths = [x for x in img_paths if x not in similar_image_paths]   # из всех путей до картинок удаляем похожие между собой
    return img_paths





img_paths, features_all_images = [], []
for img_path in tqdm(os.listdir(root_dir)[:72]):
    img_paths.append(root_dir + img_path)
    input_image = Image.open(root_dir + img_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(672),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
        features = output.flatten()
        features_all_images.append(features)


img_paths = get_unsimilar_image_paths_MSE(img_paths, features_all_images)
for f in img_paths:   # копируем в отдельную папку
    shutil.copy(f, '/home/user/Desktop/unsimilar/')