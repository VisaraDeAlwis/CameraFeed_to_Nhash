import json
from sklearn.metrics.pairwise import cosine_similarity

def load_known_embeddings(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    embeddings = data["embeddings"]
    labels = data["labels"]
    print(f"[INFO] Loaded {len(labels)} known embeddings.")
    return embeddings, labels

def compute_similarity(test_emb, known_embs, known_labels, threshold=0.5):
    max_score = -1
    matched_label = "Unknown"
    for emb, label in zip(known_embs, known_labels):
        score = cosine_similarity([test_emb], [emb])[0][0]
        if score > max_score:
            max_score = score
            matched_label = label
    return (matched_label if max_score >= threshold else "Unknown", max_score)
