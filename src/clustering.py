from sklearn.cluster import KMeans
import seaborn as sns
from utils import load_Dataset
from sklearn.metrics.pairwise import cosine_distances
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from Data_Curatiom.utilsStage2 import ProcessImageStage2
from neural_network import extrair_embedding
from PCA import usePCA
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
import pandas as pd
import argparse


def FindViableNClusters(features_animais_pca):
    inercias = []
    K_range = range(1, 20)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(features_animais_pca)
        inercias.append(km.inertia_)

    plt.plot(K_range, inercias, 'bx-')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inércia')
    plt.title('Método do Cotovelo')
    plt.show()

def PlotSimilarityMatrix(features, metric):
    plt.figure(figsize=(6,5))
    sns.heatmap(
        metric(features),
        cmap="viridis",
        vmin=0, vmax=1,      # escala entre 0 e 1
        xticklabels=1,      # mostra a cada 10 pontos para evitar poluição visual
        yticklabels=1,
        square=True          # força a aparencia de quadrado
    )

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Elbow", "-E", action="store_true", help="Use essa flag para mostrar o gráfico da inercia / gráfico do cotovelo")
    parser.add_argument("--Similarity", "-S", action="store_true", help="Use essa flag para mostrar a matrix de similaridade")
    parser.add_argument("--Dendrogram", "-D", action="store_true")
    
    args = parser.parse_args()
    
    n_clusters = 12
    qntPca = 7
    
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    features_pca = load_Dataset("features_pca.npy")
    names = load_Dataset("nomes.npy")

    # myImg = ProcessImageStage2("test/imgsStage2/PROCESSADO_TRATADO_None00.jpg", PASTA_SAIDA="test")
    imgPath = "test/stage2/PROCESSADO_TRATADO_None07.jpg"
    img = cv2.imread(imgPath)
    
    Human = extrair_embedding(img)
    Human = usePCA(Human.reshape(1, -1))

    fullData = np.vstack([features_pca, Human])[:, :qntPca]
    fullNames = np.append(names, "Zueverton")
    
    TrainData = features_pca[:, :qntPca]
    HumanFit = Human[:, :qntPca]
    print(HumanFit.shape)
    
    # cluster = AgglomerativeClustering(n_clusters, metric=cosine_distances,  linkage='average')
    # cluster.fit(fullData)
    
    if args.Elbow: FindViableNClusters(TrainData)
    if args.Similarity: PlotSimilarityMatrix(fullData, cosine_distances)
    
    if args.Dendrogram:
        Z = linkage(fullData, method="average", metric="cosine")
        
        plt.figure(figsize=(12,6))
        dendrogram(Z, labels=fullNames, leaf_rotation=90)
        plt.title("Dendrograma Facial: Onde o Humano se ramifica?")
        plt.show()

    
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="tied", random_state=42, reg_covar=1e-5, n_init = 10)
    gmm.fit(TrainData)
    probs_zuzu = gmm.predict_proba(HumanFit)[0]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(TrainData)
    print(np.unique(kmeans.labels_, return_counts=True))

    for i, prob in enumerate(probs_zuzu):
        if prob > 0.01: # Mostrar apenas clusters com mais de 1% de chance
            print(f"Cluster {i}: {prob*100:.2f}%")

    labels_treino = gmm.predict(TrainData.astype(np.float64))
    mapeamento = pd.DataFrame({
        'Animal': names,
        'Cluster': labels_treino
    })

    # 3. Ver quem são os "representantes" de cada cluster
    for i in range(gmm.n_components):
        integrantes = mapeamento[mapeamento['Cluster'] == i]['Animal'].tolist()
        print(f"\n--- CLUSTER {i} ---")
        print(f"Total de animais: {len(integrantes)}")
        print(f"Exemplos: {integrantes}") # Mostra os 5 primeiros nomes
    
    # 1. Pegue as densidades de log (em vez das probabilidades prontas)
    log_probs = gmm.score_samples(HumanFit.reshape(1, -1)) # Verossimilhança total
    # O GMM não dá o log por cluster facilmente, então usamos as distâncias ponderadas:
    weighted_log_probs = gmm._estimate_weighted_log_prob(HumanFit.reshape(1, -1))[0]

    # 2. Aplique uma "Temperatura" (T > 1 suaviza, T < 1 polariza)
    T = 5.0 
    exp_probs = np.exp(weighted_log_probs / T)
    porcentagens = (exp_probs / np.sum(exp_probs)) * 100

    for i, p in enumerate(porcentagens):
        print(f"Cluster {i}: {p:.2f}%")
