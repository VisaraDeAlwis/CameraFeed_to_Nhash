import csv
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from modules.embedding_extractor import extract_embedding
from modules.pca_reducer import load_pca_model, reduce_embedding
from modules.hashing import load_projection_matrix, compute_hash_from_embedding
from scipy.spatial.distance import cosine
import cv2

# Load PCA and projection matrix
pca = load_pca_model("pca_512_to_128.pkl")
proj = load_projection_matrix("neuralhash_128x96_seed1.dat")

# Load CSV
pairs = []
with open("test_pairs.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pairs.append((row["img1"], row["img2"], int(row["label"])))

cosine_similarities = []
hamming_distances = []
true_labels = []

for img1_path, img2_path, label in pairs:
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print(f"[WARNING] Skipping missing image(s): {img1_path}, {img2_path}")
        continue

    emb1 = extract_embedding(img1)
    emb2 = extract_embedding(img2)

    if emb1 is None or emb2 is None:
        print(f"[WARNING] Skipping undetected face(s): {img1_path}, {img2_path}")
        continue

    # Cosine similarity (higher = more similar)
    cos_sim = 1 - cosine(emb1, emb2)
    cosine_similarities.append(cos_sim)

    # Hash comparison (lower = more similar)
    hash1 = compute_hash_from_embedding(reduce_embedding(pca, emb1), proj)
    hash2 = compute_hash_from_embedding(reduce_embedding(pca, emb2), proj)
    hamming = np.sum(hash1 != hash2)
    hamming_distances.append(hamming)

    true_labels.append(label)

# -----------------------------
# Evaluate cosine similarity
# -----------------------------
cosine_preds = [1 if sim > 0.5 else 0 for sim in cosine_similarities]  # Adjust threshold as needed
print("\n📊 Cosine Similarity Metrics:")
print("Accuracy:", accuracy_score(true_labels, cosine_preds))
print("Precision:", precision_score(true_labels, cosine_preds))
print("Recall:", recall_score(true_labels, cosine_preds))
print("F1 Score:", f1_score(true_labels, cosine_preds))
print("Confusion Matrix:\n", confusion_matrix(true_labels, cosine_preds))
