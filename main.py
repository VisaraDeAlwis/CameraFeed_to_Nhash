import cv2
import os
import time
import warnings
from modules.embedding_extractor import extract_embedding
from modules.utils import load_known_embeddings, compute_similarity
from modules.pca_reducer import load_pca_model, reduce_embedding
from modules.hashing import load_projection_matrix, compute_hash_from_embedding

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Function to check if image is blurry
def is_blurry(image, threshold=50.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold  # True = blurry

# Load models
pca = load_pca_model("pca_512_to_128.pkl")
proj_matrix = load_projection_matrix("neuralhash_128x96_seed1.dat")
known_embeddings, known_labels = load_known_embeddings("face_embeddings_128.json")

# Create directory
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

print("[INFO] Starting video capture. Press 'q' to quit.")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    # Check if frame is blurry
    if is_blurry(frame):
        print("[WARNING] Skipped blurry frame.")
        continue

    # Extract embedding and bounding box directly
    embedding_512, bbox = extract_embedding(frame, return_bbox=True)

    if embedding_512 is not None and bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save image
        existing = [f for f in os.listdir(save_dir) if f.startswith("capture_") and f.endswith(".jpg")]
        numbers = [int(f.split('_')[1].split('.')[0]) for f in existing if f.split('_')[1].split('.')[0].isdigit()]
        next_id = max(numbers, default=0) + 1
        filename = f"capture_{next_id}.jpg"
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, frame)
        print(f"[INFO] Face detected, saved image: {filename}")

        # Reduce and hash
        embedding_128 = reduce_embedding(pca, embedding_512)
        binary_hash = compute_hash_from_embedding(embedding_128, proj_matrix)
        print("🔐 96-bit Hash:", "".join(map(str, binary_hash.tolist())))

        time.sleep(0.5)

    cv2.imshow("Live Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
