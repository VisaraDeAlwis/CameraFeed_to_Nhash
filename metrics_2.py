# File: 1_generate_and_analyze_metrics_hamming_only.py

import json
import numpy as np
from itertools import combinations
from tqdm import tqdm
import warnings
import csv

warnings.simplefilter(action='ignore', category=FutureWarning)

def hamming_similarity(h1, h2):
    """Calculates the Hamming similarity between two binary hash strings."""
    dissimilar_bits = sum(c1 != c2 for c1, c2 in zip(h1, h2))
    return 1 - (dissimilar_bits / 96)

# --- 1. Load and Pre-process Data ---
print("📥 Loading pre-computed embeddings from reduced_dataset.json...")
with open("reduced_dataset.json", "r") as f:
    data = json.load(f)

# --- 2. Collect Genuine and Impostor Pairs ---
# This separation is key: we compute pairs only once.
genuine_scores = []
impostor_scores = []

# Collect scores for same-person pairs
print("🔍 Calculating scores for genuine (same-person) pairs...")
for person, imgs in tqdm(data.items(), ncols=100, unit="person"):
    if len(imgs) < 2:
        continue
    # Create all combinations of images for a single person
    for img1_data, img2_data in combinations(imgs, 2):
        # --- MODIFIED LINE: Only using hash for similarity ---
        h1 = img1_data["hash"]
        h2 = img2_data["hash"]
        sim = hamming_similarity(h1, h2)
        genuine_scores.append(sim)

# Collect scores for different-person pairs
print("🚫 Calculating scores for impostor (different-person) pairs...")
persons = list(data.keys())
for p1, p2 in tqdm(combinations(persons, 2), ncols=100, unit="pair", total=len(persons)*(len(persons)-1)//2):
    if data[p1] and data[p2]:
        # Use the first image of each person for the impostor check
        # --- MODIFIED LINE: Only using hash for similarity ---
        h1 = data[p1][0]["hash"]
        h2 = data[p2][0]["hash"]
        sim = hamming_similarity(h1, h2)
        impostor_scores.append(sim)

# --- 3. Sweep Thresholds and Calculate Metrics ---
thresholds = np.arange(0.5, 1.00, 0.01)
results = []
num_genuine = len(genuine_scores)
num_impostor = len(impostor_scores)

print(f"\nTotal Genuine Pairs Found: {num_genuine}")
print(f"Total Impostor Pairs Found: {num_impostor}\n")

print("📈 Evaluating thresholds to calculate metrics...")
for t in tqdm(thresholds, ncols=100):
    # Calculate the raw counts of TP, FP, FN, TN
    tp = sum(1 for score in genuine_scores if score >= t)
    fp = sum(1 for score in impostor_scores if score >= t)
    fn = num_genuine - tp
    tn = num_impostor - fp
    
    # Calculate the rates
    gar = tp / num_genuine if num_genuine else 0
    far = fp / num_impostor if num_impostor else 0
    frr = fn / num_genuine if num_genuine else 0
    
    results.append({
        "Threshold": round(t, 2), "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "GAR_TPR": gar, "FAR_FPR": far, "FRR": frr
    })

# --- 4. Save Results to CSV ---
csv_file = "evaluation_metrics_detailed_hamming_only.csv"
fieldnames = ["Threshold", "TP", "FN", "FP", "TN", "GAR_TPR", "FAR_FPR", "FRR"]

with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Evaluation complete! Detailed results saved to {csv_file}")

# --- 5. Analyze and Recommend Best Thresholds ---
# This final analysis section remains the same, but will now operate on the Hamming-only results.

# 1. Best for Convenience (Lowest FRR)
best_convenience = sorted(results, key=lambda x: (x['FRR'], x['FAR_FPR']))[0]

# 2. Best for Protection (Lowest FAR)
best_protection = sorted(results, key=lambda x: (x['FAR_FPR'], x['FRR']))[0]

# 3. Balanced Point (Closest to Equal Error Rate)
balanced = min(results, key=lambda x: abs(x['FAR_FPR'] - x['FRR']))

print("\n" + "="*50)
print("RECOMMENDED THRESHOLDS (HAMMING SIMILARITY ONLY)")
print("="*50)

print("\n1. BEST FOR CONVENIENCE (Lowest False Rejections)")
print(f"   - Recommended Threshold: {best_convenience['Threshold']:.2f}")
print(f"   - At this threshold:")
print(f"     - FRR (Inconvenience): {best_convenience['FRR']:.2%}")
print(f"     - FAR (Security Risk): {best_convenience['FAR_FPR']:.2%}")
print("   - Use Case: Low-security applications where user experience is the top priority.")

print("\n2. BEST FOR PROTECTION (Lowest False Acceptances)")
print(f"   - Recommended Threshold: {best_protection['Threshold']:.2f}")
print(f"   - At this threshold:")
print(f"     - FAR (Security Risk): {best_protection['FAR_FPR']:.4%}")
print(f"     - FRR (Inconvenience): {best_protection['FRR']:.2%}")
print("   - Use Case: High-security applications where preventing unauthorized access is critical.")

print("\n3. BALANCED PERFORMANCE (Equal Error Rate)")
print(f"   - Recommended Threshold: {balanced['Threshold']:.2f}")
print(f"   - At this threshold:")
print(f"     - FRR (Inconvenience): {balanced['FRR']:.2%}")
print(f"     - FAR (Security Risk): {balanced['FAR_FPR']:.2%}")
print("   - Use Case: General-purpose applications that need a fair compromise between security and usability.")
print("="*50)