import cv2
import os
import glob
import numpy as np
import argparse
from utilsStage1 import *
from utilsStage2 import *

# --- CONFIGURAÇÕES ---
TAMANHO_FINAL = 224             

def runStage2(imgs, PASTA_SAIDA):
    for imgPath in imgs:
        ProcessImageStage2(imgPath, PASTA_SAIDA=PASTA_SAIDA)
        print(f"[OK] Salvo: PROCESSADO_{imgPath} em {PASTA_SAIDA}")

def runStage1(imgs, PASTA_SAIDA, animal, skips = False):
    print(f"Encontradas {len(imgs)} images.")
    print("--- CONTROLES ---")
    print("[A] / [D] : Rotacionar Esquerda/Direita")
    print("[ENTER]   : Confirmar rotação e Selecionar Rosto")
    print("[C]       : Pular imagem")
    print("[ESC]     : Sair")

    renames = []

    for i, caminho_img in enumerate(imgs):
        nome_arquivo = f"{animal}{"0"+str(i) if i < 10 else i}.{os.path.basename(caminho_img).split(".")[-1]}"
        if nome_arquivo == os.path.basename(caminho_img) and skips:
            continue
        
        img_original = cv2.imread(caminho_img)
        if img_original is None: continue

        img_original = reziseWithFactor(img_original)
        if ProcessSelectionStage(img_original, [True, False], nome_arquivo, PASTA_SAIDA) and nome_arquivo != os.path.basename(caminho_img):
            print(caminho_img, nome_arquivo)
            renames.append((caminho_img, i))
    
    for i,j in renames:
        os.rename(i, os.path.join(os.path.dirname(i), f"{animal}{"0"+str(j) if j < 10 else j}.{os.path.basename(i).split(".")[-1]}"))

    cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", help="Select [1] Stage 1 (Image Selection), [2] Stage 2 (Image Processing)", type=int)
    parser.add_argument("--animal", help="Input animal folders name", required=False)
    parser.add_argument("--Ipath", help="Input path", required=False)
    parser.add_argument("--Opath", help="Output path", required=False)
    parser.add_argument("--skip", help="Skip already selected ones", default=False)
    
    args = parser.parse_args()
    

    PASTA_ENTRADA = f'raw/{args.animal}' if args.stage == 1 else 'data/stage1'
    PASTA_SAIDA = 'data/stage1' if args.stage == 1 else 'data/stage2'
    
    if args.Ipath: PASTA_ENTRADA = args.Ipath
    if args.Opath: PASTA_SAIDA   = args.Opath
    
    if not os.path.exists(PASTA_SAIDA):
        os.makedirs(PASTA_SAIDA)

    imagens = glob.glob(os.path.join(PASTA_ENTRADA, "*"))
    
    if args.stage == 1: runStage1(imagens, PASTA_SAIDA, args.animal, skips=args.skip)
    else: runStage2(imagens, PASTA_SAIDA)