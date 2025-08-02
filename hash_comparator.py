import cv2
from modules.embedding_extractor import extract_embedding
from modules.pca_reducer import load_pca_model, reduce_embedding
from modules.hashing import load_projection_matrix, compute_hash_from_embedding

def hamming_distance(hash1, hash2):
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

# === Load Models ===
pca = load_pca_model("pca_512_to_128.pkl")
proj_matrix = load_projection_matrix("neuralhash_128x96_seed1.dat")

# === Input image paths ===
img1_path = "uploads/5.jpg"
img2_path = "uploads/6.jpg"

# === Load images using OpenCV ===
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

if img1 is None or img2 is None:
    print("[ERROR] One or both images not found.")
    exit()

# === Extract 512D embeddings from InsightFace ===
emb1 = extract_embedding(img1)
emb2 = extract_embedding(img2)

if emb1 is not None and emb2 is not None:
    emb1_128 = reduce_embedding(pca, emb1)
    emb2_128 = reduce_embedding(pca, emb2)

    hash1 = compute_hash_from_embedding(emb1_128, proj_matrix)
    hash2 = compute_hash_from_embedding(emb2_128, proj_matrix)

    print(f"\n🧬 Hash 1: {hash1}")
    print(f"🧬 Hash 2: {hash2}")

    dist = hamming_distance(hash1, hash2)
    print(f"\n📏 Hamming Distance: {dist}")
else:
    print("[ERROR] Could not extract embeddings from one or both images.")
