import os
import json
import random
import warnings
from tqdm import tqdm
import random
import cv2

from modules.embedding_extractor import extract_embedding
from modules.pca_reducer import load_pca_model, reduce_embedding
from modules.hashing import load_projection_matrix, compute_hash_from_embedding

# --- Suppress warnings ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load models
pca = load_pca_model("pca_512_to_128.pkl")
proj_matrix = load_projection_matrix("neuralhash_128x96_seed1.dat")

def compute_hash_and_embedding(image_path):
    img = cv2.imread(image_path)
    emb_512 = extract_embedding(img)
    if emb_512 is None:
        return None, None
    emb_128 = reduce_embedding(pca, emb_512)
    binary_hash = compute_hash_from_embedding(emb_128, proj_matrix)
    hash_str = ''.join(map(str, binary_hash.tolist()))
    return hash_str, emb_128  # store reduced embedding only

dataset_path = "C:\\Users\\ASUS\\Desktop\\a"
reduced_data = {}

folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

for folder in tqdm(folders, desc="Processing Folders", ncols=100, unit="folder"):
    folder_path = os.path.join(dataset_path, folder)
    all_images = sorted(os.listdir(folder_path))
    selected_images = random.sample(all_images, min(50, len(all_images)))  # pick up to 50 images
    reduced_data[folder] = []

    for img_file in tqdm(selected_images, desc=f"Images in {folder}", leave=False, ncols=100, unit="img"):
        img_path = os.path.join(folder_path, img_file)
        h, e = compute_hash_and_embedding(img_path)
        if h is not None:
            reduced_data[folder].append({
                "filename": img_file,
                "hash": h,
                "embedding": e.tolist()  # store as list for JSON
            })

# Save to JSON
with open("reduced_dataset.json", "w") as f:
    json.dump(reduced_data, f)

print("✅ Done! JSON saved as 'reduced_dataset.json'.")
