import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import glob
import os
import cv2
from utils import save_Dataset

# Carregar modelo pré-treinado
modelo = models.resnet50(pretrained=True)

# Remover a última camada (Global Average Pooling permanece)
# No ResNet, a última camada é a 'fc'. Nós a substituímos por uma Identidade.
modelo.fc = nn.Identity()
modelo.eval() # Modo de inferência

def extrair_embedding(img_processada):
    # A imagem vinda do seu tratamento_avancado (OpenCV) precisa virar Tensor
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    tensor = preprocess(img_processada).unsqueeze(0) # Adiciona dimensão de batch
    
    with torch.no_grad():
        features = modelo(tensor)
    
    return features.numpy().flatten() # Vetor de 2048 dimensões

if __name__ == "__main__":
    PASTA_INPUT = "data/stage2"
    imagens = glob.glob(os.path.join(PASTA_INPUT, "*"))
    
    data_embeddings = []
    data_names = []
    
    for imgPath in imagens:
        img_original = cv2.imread(imgPath)
        imgName = os.path.basename(imgPath)
        
        embedding = extrair_embedding(img_original)
        
        data_embeddings.append(embedding)
        data_names.append(imgPath.split("_")[-1].split(".")[0])
    
    save_Dataset(data_embeddings)
    save_Dataset(data_names, "nomes.npy")