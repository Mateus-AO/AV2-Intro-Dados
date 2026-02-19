import numpy as np
import os

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