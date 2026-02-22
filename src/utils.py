import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# --- SAVE ---
def save_Dataset(embeddings, arquivo_features="features.npy"):
    if not os.path.exists("data/features"):
        os.makedirs("data/features")
    
    np.save(os.path.join("data/features", arquivo_features), embeddings)
    print("[Save] Banco de dados vetorial salvo")

# --- LOAD ---
def load_Dataset(arquivo_features="features.npy"):
    features = np.load(os.path.join("data/features", arquivo_features))
    return features



def FindViableNClusters(features_animais_pca, range_):
    inercias = []
    K_range = range_

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(features_animais_pca)
        inercias.append(km.inertia_)

    plt.plot(K_range, inercias, 'bx-')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inércia')
    plt.title('Método do Cotovelo')
    plt.xticks(K_range)
    plt.show()

def PlotSimilarityMatrix(features, metric):
    plt.figure(figsize=(6,5))
    sns.heatmap(
        metric(features),
        cmap="viridis",
        vmin=0, vmax=1,      # escala entre 0 e 1
        xticklabels=10,      # mostra a cada x pontos
        yticklabels=10,

        square=True          # força aparencia de quadrado
    )

    plt.show()