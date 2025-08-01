import pickle

def load_pca_model(path):
    with open(path, "rb") as f:
        pca = pickle.load(f)
    print("[INFO] Loaded trained PCA for 128D reduction.")
    return pca

def reduce_embedding(pca, embedding_512):
    return pca.transform([embedding_512])[0]
