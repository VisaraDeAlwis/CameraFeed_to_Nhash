import json
import numpy as np
from itertools import combinations
from tqdm import tqdm
import warnings
import csv

warnings.simplefilter(action='ignore', category=FutureWarning)

def hamming_similarity(h1, h2):
    return 1 - sum(c1 != c2 for c1, c2 in zip(h1, h2)) / 96

def cosine_similarity(e1, e2):
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))

# Load embeddings
with open("reduced_dataset.json", "r") as f:
    data = json.load(f)

# Precompute first embeddings per folder for faster computation
precomputed = {}
for person, imgs in data.items():
    if imgs:
        h_ref = imgs[0]["hash"]
        e_ref = np.array(imgs[0]["embedding"])
        precomputed[person] = {"hash": h_ref, "embedding": e_ref}

# Compute all genuine and impostor pairs
genuine_pairs = []
impostor_pairs = []

# Same-person pairs
for person, imgs in tqdm(data.items(), desc="Collecting same-person pairs", ncols=100, unit="person"):
    if len(imgs) < 2:
        continue
    h_ref = imgs[0]["hash"]
    e_ref = np.array(imgs[0]["embedding"])
    for img in imgs[1:]:
        h = img["hash"]
        e = np.array(img["embedding"])
        genuine_pairs.append((h_ref, e_ref, h, e))

# Different-person pairs (first image only)
persons = list(precomputed.keys())
for p1, p2 in tqdm(combinations(persons, 2), desc="Collecting different-person pairs", ncols=100, unit="pair"):
    h1, e1 = precomputed[p1]["hash"], precomputed[p1]["embedding"]
    h2, e2 = precomputed[p2]["hash"], precomputed[p2]["embedding"]
    impostor_pairs.append((h1, e1, h2, e2))

# Sweep thresholds
thresholds = np.arange(0.50, 0.91, 0.01)
results = []

for t in tqdm(thresholds, desc="Evaluating thresholds", ncols=100):
    tp = 0  # True positives (genuine accepted)
    fp = 0  # False positives (impostor accepted)
    
    # Genuine pairs
    for h1, e1, h2, e2 in genuine_pairs:
        sim = max(hamming_similarity(h1, h2), cosine_similarity(e1, e2))
        if sim >= t:
            tp += 1
    
    # Impostor pairs
    for h1, e1, h2, e2 in impostor_pairs:
        sim = max(hamming_similarity(h1, h2), cosine_similarity(e1, e2))
        if sim >= t:
            fp += 1
    
    gar = tp / len(genuine_pairs) if genuine_pairs else 0
    far = fp / len(impostor_pairs) if impostor_pairs else 0
    results.append({
        "Threshold": round(t, 2),
        "True_Positive": tp,
        "False_Positive": fp,
        "GAR": gar,
        "FAR": far
    })

# Save results to CSV
csv_file = "evaluation_metrics.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Threshold", "True_Positive", "False_Positive", "GAR", "FAR"])
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Evaluation complete! Results saved to {csv_file}")
