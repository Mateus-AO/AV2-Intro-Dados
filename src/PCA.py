from utils import *
import numpy as np
from sklearn.decomposition import PCA
import joblib

def usePCA(data):
    model_path = "data/model_pca.pkl"
    
    if os.path.exists(model_path):
        pca = joblib.load(model_path)
        return pca.transform(data)

    # 2. Inicializar o PCA
    # 'n_components=0.95' significa: "mantenha componentes suficientes para 
    # explicar 95% da variância dos dados". Geralmente reduz de 2048 para ~100.
    pca = PCA(n_components=0.70)
    data_reduzida = pca.fit_transform(data)
    joblib.dump(pca, model_path)
    
    return data_reduzida


if __name__ == "__main__":
    features = load_Dataset()
    
    features_reduzida = usePCA(features)
    
    print(f"Dimensões originais: {features.shape[1]}")
    print(f"Dimensões após PCA: {features_reduzida.shape[1]}")
    
    save_Dataset(features_reduzida, "features_pca.npy")