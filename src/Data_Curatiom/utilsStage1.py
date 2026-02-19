import cv2
import numpy as np
import os

def rotacionar_imagem(imagem, angulo):
    """
    Gira a imagem ao redor do centro sem cortar os cantos (mantém a escala).
    """
    (h, w) = imagem.shape[:2]
    centro = (w // 2, h // 2)

    # Cria a matriz de rotação
    M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    
    # Executa a rotação (o fundo vira preto, o que é ótimo para o padding)
    rotacionada = cv2.warpAffine(imagem, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return rotacionada

def addGuideLines(img):
    grid = img.copy()
    h, w = grid.shape[:2]
    cx, cy = w // 2, h // 2

    # Cor: (Blue, Green, Red)
    # verde_neon = (0, 255, 0)
    azul_claro = (255, 0, 155)
    
    # 2. Linhas Auxiliares (Para ajudar a ver a inclinação)
    offset = 40 # Distância das linhas auxiliares
    for i in range(0, h//2, offset):
    # Horizontais
        cv2.line(grid, (0, cy - i), (w, cy - i), azul_claro, 1) 
        cv2.line(grid, (0, cy + i), (w, cy + i), azul_claro, 1)
    # Verticais
    for i in range(0, w//2, offset):
        cv2.line(grid, (cx - i, 0), (cx - i, h), azul_claro, 1)
        cv2.line(grid, (cx + i, 0), (cx + i, h), azul_claro, 1)

    return grid


def reziseWithFactor(img_original):
    fator_escala = 1
    if img_original.shape[0] > 400:
        fator_escala = 400 / img_original.shape[0]
        img_original = cv2.resize(img_original, (0,0), fx=fator_escala, fy=fator_escala)
    return img_original


def ProcessResizeStage(img_original, stages, nome_arquivo):
    angulo_atual = 0
    img_rotate_section = addGuideLines(img_original.copy())
    while stages[0]:
        # Mostra a imagem rotacionada
        cv2.imshow("Editor (A/D gira, ENTER seleciona)", img_rotate_section)
        
        # Espera uma tecla
        k = cv2.waitKey(0)

        if k == ord('a'): # Esquerda
            angulo_atual += 5
            img_rotate_section = addGuideLines(rotacionar_imagem(img_original, angulo_atual))
        
        elif k == ord('d'): # Direita
            angulo_atual -= 5
            img_rotate_section = addGuideLines(rotacionar_imagem(img_original, angulo_atual))
        
        elif k == 13 or k == 32: # ENTER ou ESPAÇO
            stages[1] = True
            stages[0] = False # Sai do loop de rotação
            
        elif k == ord('c'): # Pular
            stages[0] = False
            stages[1] = False
            print(f"Pulado: {nome_arquivo}")
            
        elif k == 27: # ESC
            exit()

    return (rotacionar_imagem(img_original, angulo_atual), angulo_atual)

def ProcessSelectionStage(img_original, stages, nome_arquivo, PASTA_SAIDA, TAMANHO_FINAL=224):
    img_display, angulo_atual = ProcessResizeStage(img_original, stages, nome_arquivo)
    
    if stages[1]:
        cv2.destroyWindow("Editor (A/D gira, ENTER seleciona)")
            
            # Abre a ferramenta de seleção na imagem JÁ ROTACIONADA
        r = cv2.selectROI("Selecione o Rosto", img_display, showCrosshair=True)
        cv2.destroyWindow("Selecione o Rosto") # Fecha logo após selecionar
        
        # Se selecionou algo válido
        if r[2] > 0 and r[3] > 0:
            x, y, w, h = int(r[0]), int(r[1]), int(r[2]), int(r[3])
            
            # Recorta quadrado
            corte = img_display[y:y+h, x:x+w]
            
            try:
                # Redimensiona para 224x224
                dimFactor = TAMANHO_FINAL/max(w,h)
                final = cv2.resize(corte, (int(w*dimFactor), int(h*dimFactor)))
                
                # Salva
                cv2.imwrite(os.path.join(PASTA_SAIDA, f"TRATADO_{nome_arquivo}"), final)
                print(f"[OK] Salvo: TRATADO_{nome_arquivo} (Rotacionado {angulo_atual}°), {dimFactor*h}, {dimFactor*w}")
                return True
            except Exception:
                print(f"[ERRO] Falha ao salvar {nome_arquivo}, {Exception}")
    return False