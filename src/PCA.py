from utils import *
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import argparse

def usePCA(data, reset = False):
    model_path = "data/model_pca.pkl"
    
    if os.path.exists(model_path) and not reset:
        model = joblib.load(model_path)
        scaler = model['scaler']
        pca = model['pca']
        
        data = scaler.transform(data)
        return pca.transform(data)

    # 2. Inicializar o PCA
    # 'n_components=0.95' significa: "mantenha componentes suficientes para 
    # explicar 95% da variância dos dados". Geralmente reduz de 2048 para ~100.
    pca = PCA(n_components=0.70)
    scaler = StandardScaler()
    
    data = scaler.fit_transform(data)
    data_reduzida = pca.fit_transform(data)
    
    model =  {
        'scaler': scaler,
        'pca': pca
    }
    joblib.dump(model, model_path)
    
    return data_reduzida


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", '-r', action="store_true", help="Para resetar o arquivo model_pca.pkl")
    args = parser.parse_args()
    
    features = load_Dataset()
    
    features_reduzida = usePCA(features, args.reset)
    
    print(f"Dimensões originais: {features.shape[1]}")
    print(f"Dimensões após PCA: {features_reduzida.shape[1]}")
    
    save_Dataset(features_reduzida, "features_pca.npy")