import os
import numpy as np
from itertools import combinations
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from modules.embedding_extractor import extract_embedding
from modules.pca_reducer import load_pca_model, reduce_embedding
from modules.hashing import load_projection_matrix, compute_hash_from_embedding

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load models
pca = load_pca_model("pca_512_to_128.pkl")
proj_matrix = load_projection_matrix("neuralhash_128x96_seed1.dat")

# === Utility Functions ===
def compute_hash_and_embedding(image_path):
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    emb_512 = extract_embedding(img)
    if emb_512 is None:
        return None, None
    emb_128 = reduce_embedding(pca, emb_512)
    binary_hash = compute_hash_from_embedding(emb_128, proj_matrix)
    hash_str = ''.join(map(str, binary_hash.tolist()))
    return hash_str, emb_512

def hamming_similarity(h1, h2):
    dissimilar_bits = sum(c1 != c2 for c1, c2 in zip(h1, h2))
    return 1 - (dissimilar_bits / 96)

def cosine_similarity(e1, e2):
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))

# === Evaluation ===
def evaluate_dataset(dataset_path, similarity_threshold=0.75):
    folders = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])
    persons = {}

    # Precompute embeddings for all images
    print("📥 Loading and computing embeddings...")
    for person in tqdm(folders, desc="Processing Folders", unit="folder", ncols=100, mininterval=1.0):
        folder_path = os.path.join(dataset_path, person)
        persons[person] = []
        for img_file in sorted(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, img_file)
            h, e = compute_hash_and_embedding(img_path)
            if h is not None:
                persons[person].append((h, e))

    y_true = []
    y_pred = []

   # --- True Positive Check (same person) ---
    print("🔍 Checking same-person pairs...")
    for person, data in tqdm(persons.items(), desc="Same-person eval", unit="person", ncols=100, mininterval=1.0):
        if len(data) < 2:
            continue  # skip if folder has 0 or 1 image
        h_ref, e_ref = data[0]  # first image as reference
        for h, e in data[1:]:   # compare all other images to first image
            hs = hamming_similarity(h_ref, h)
            cs = cosine_similarity(e_ref, e)
            sim = max(hs, cs)
            y_true.append(1)  # same person
            y_pred.append(1 if sim >= similarity_threshold else 0)

    # --- False Positive Check (different person) ---
    print("🚫 Checking different-person pairs...")
    for p1, p2 in tqdm(combinations(folders, 2), desc="Diff-person eval", unit="pair", ncols=100, mininterval=1.0):
        if persons[p1] and persons[p2]:
            h1, e1 = persons[p1][0]  # first image of p1
            h2, e2 = persons[p2][0]  # first image of p2
            hs = hamming_similarity(h1, h2)
            cs = cosine_similarity(e1, e2)
            sim = max(hs, cs)
            y_true.append(0)  # different person
            y_pred.append(1 if sim >= similarity_threshold else 0)

    # --- Metrics ---
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n📊 Threshold: {similarity_threshold}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall:    {recall:.4f}")
    print(f"✅ F1-score:  {f1:.4f}")

    return precision, recall, f1

if __name__ == '__main__':
    dataset_path = "C:\\Users\\ASUS\\Desktop\\a"
    evaluate_dataset(dataset_path, similarity_threshold=0.75)
