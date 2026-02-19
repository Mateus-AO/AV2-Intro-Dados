import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os

def tratamento_avancado(caminho_imagem):
    # 1. Carregar com OpenCV (em Grayscale direto)
    img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
    
    if img is None: return None

    # --- PASSO A: CLAHE (Melhorar Contraste Local) ---
    # clipLimit=2.0 impede que fique com ruído demais
    # tileGridSize=(8,8) divide a imagem em pedacinhos para ajustar a luz localmente
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)

    # --- PASSO 2: FILTRO DE MEDIANA (O "Exterminador de Pelos") ---
    # Diferente do Gaussian, a Mediana substitui o pixel pelo valor central da vizinhança.
    # Isso remove fios isolados de pelo sem borrar as bordas principais.
    # O kernel deve ser ímpar (5 ou 7 são ideais para fotos de rosto).
    img_median = cv2.medianBlur(img_clahe, 5)

    # 3. FILTRO BILATERAL (O Substituto do Blur + Sharpen)
    # Parâmetros explicados abaixo:
    # d=9: Diâmetro da vizinhança (olha 9 pixels em volta).
    # sigmaColor=75: Quão diferente a cor tem que ser para manter a borda.
    #    (Alto = borra pelos com cores parecidas, mas mantém contraste forte).
    # sigmaSpace=75: O quanto o blur se espalha geograficamente.
    
    img_bilateral = cv2.bilateralFilter(img_median, d=9, sigmaColor=75, sigmaSpace=75)

    # --- PASSO 4: FECHAMENTO MORFOLÓGICO (Unificação de Estrutura) ---
    # Substituímos a Erosão pura pelo Closing (Dilatação seguida de Erosão).
    # Isso "fecha" os espaços pretos entre os pelos, criando uma massa sólida.
    # É excelente para dar ao animal uma aparência de "estátua" ou "pele lisa".
    kernel = np.ones((3,3), np.uint8)
    img_final = cv2.morphologyEx(img_bilateral, cv2.MORPH_CLOSE, kernel)

    # Neutralizando texturas de pelagem agressivamente
    # img_final = cv2.bilateralFilter(img_final, d=15, sigmaColor=150, sigmaSpace=75)
    img_final = cv2.bilateralFilter(img_final, d=12, sigmaColor=100, sigmaSpace=75)
    
    # --- CONVERSÃO PARA IA ---
    # O OpenCV usa NumPy, mas a IA (PyTorch) quer PIL Image e RGB
    # Transformamos de volta para RGB (copiando os canais cinzas)
    img_final = cv2.cvtColor(img_final, cv2.COLOR_GRAY2RGB)
    
    return img_final

def ProcessImageStage2(imgPath, TAMANHO_FINAL=224, PASTA_SAIDA="data/stage2"):
    img = tratamento_avancado(imgPath)
    
    final = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # final = img
    h,w = final.shape
    dimFactor = TAMANHO_FINAL/max(w,h)
    
    hborder = TAMANHO_FINAL - h*dimFactor
    wborder = TAMANHO_FINAL - w*dimFactor
    
    final = cv2.resize(final, (int(w/2), int(h/2)))
    final = cv2.resize(final, (int(w), int(h)))
    
    final = cv2.copyMakeBorder(
        final, 
        int(hborder//2), int((hborder+1)//2), int(wborder//2), int((wborder+1)//2), 
        cv2.BORDER_CONSTANT, 
        value=[0] # Preto
    )
    
    img_name = os.path.basename(imgPath)
    cv2.imwrite(os.path.join(PASTA_SAIDA, f"PROCESSADO_{img_name}"), final)

# Exemplo de uso no loop:
# imagem_pil = tratamento_avancado("leao.jpg")
# tensor = preprocessamento_ia(imagem_pil)