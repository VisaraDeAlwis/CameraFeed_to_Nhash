import cv2
import os
import time
import uuid
import warnings
from modules.face_detector import detect_largest_face
from modules.embedding_extractor import extract_embedding
from modules.utils import load_known_embeddings, compute_similarity
from modules.pca_reducer import load_pca_model, reduce_embedding
from modules.hashing import load_projection_matrix, compute_hash_from_embedding

proj_matrix = load_projection_matrix("neuralhash_128x96_seed1.dat")

warnings.filterwarnings("ignore", category=FutureWarning)

# Load PCA model and known embeddings
pca = load_pca_model("pca_512_to_128.pkl")
known_embeddings, known_labels = load_known_embeddings("face_embeddings_128.json")

save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

print("[INFO] Starting video capture. Press 'q' to quit.")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    largest_face = detect_largest_face(frame)

    if largest_face is not None:
        x, y, w, h = largest_face['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Frame", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        existing = [f for f in os.listdir(save_dir) if f.startswith("capture_") and f.endswith(".jpg")]
        numbers = [int(f.split('_')[1].split('.')[0]) for f in existing if f.split('_')[1].split('.')[0].isdigit()]
        next_id = max(numbers, default=0) + 1
        filename = f"capture_{next_id}.jpg"

        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, frame)
        print(f"[INFO] Face detected, saved image: {filename}")

        embedding_512 = extract_embedding(save_path)
        if embedding_512 is not None:
            embedding_128 = reduce_embedding(pca, embedding_512)

            binary_hash = compute_hash_from_embedding(embedding_128, proj_matrix)
            print("🔐 96-bit Hash:", "".join(map(str, binary_hash.tolist())))
            
        else:
            print("No embedding extracted from the image.")

        time.sleep(0.5)

    cv2.imshow("Live Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
