import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
from modules.embedding_extractor import extract_embedding
from modules.pca_reducer import load_pca_model, reduce_embedding
from modules.hashing import load_projection_matrix, compute_hash_from_embedding

# Load models
pca = load_pca_model("pca_512_to_128.pkl")
proj_matrix = load_projection_matrix("neuralhash_128x96_seed1.dat")

# Storage
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)
upload_hashes = [None, None]
upload_embeddings = [None, None]

class FaceHashApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Hash Comparator")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")

        tk.Label(root, text="🔐 Face Recognition Hash App", font=("Helvetica", 20, "bold"), bg="#f0f0f0").pack(pady=10)

        self.image_frame = tk.Frame(root, bg="#f0f0f0")
        self.image_frame.pack(pady=10)

        self.image_label1 = tk.Label(self.image_frame)
        self.image_label1.grid(row=0, column=0, padx=10)

        self.image_label2 = tk.Label(self.image_frame)
        self.image_label2.grid(row=0, column=1, padx=10)

        self.buttons_frame = tk.Frame(root, bg="#f0f0f0")
        self.buttons_frame.pack(pady=10)

        tk.Button(self.buttons_frame, text="📸 Capture from Webcam", width=25, command=self.capture_face).grid(row=0, column=0, padx=10, pady=5)
        tk.Button(self.buttons_frame, text="📤 Upload Image 1", width=25, command=lambda: self.upload_image(0)).grid(row=0, column=1, padx=10, pady=5)
        tk.Button(self.buttons_frame, text="📤 Upload Image 2", width=25, command=lambda: self.upload_image(1)).grid(row=0, column=2, padx=10, pady=5)
        tk.Button(self.buttons_frame, text="🔎 Compare Hashes", width=25, command=self.compare_hashes).grid(row=1, column=1, padx=10, pady=5)

        self.hash_label1 = tk.Label(root, text="Hash 1: ---", bg="#f0f0f0", font=("Courier", 10))
        self.hash_label1.pack(pady=2)

        self.hash_label2 = tk.Label(root, text="Hash 2: ---", bg="#f0f0f0", font=("Courier", 10))
        self.hash_label2.pack(pady=2)

        self.result_label = tk.Label(root, text="", font=("Helvetica", 12, "bold"), fg="blue", bg="#f0f0f0")
        self.result_label.pack(pady=10)

    def display_image(self, img, target_label):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((350, 280))
        img = ImageTk.PhotoImage(img)
        target_label.configure(image=img)
        target_label.image = img

    def compute_hash_and_embedding(self, image):
        emb_512 = extract_embedding(image)
        if emb_512 is not None:
            emb_128 = reduce_embedding(pca, emb_512)
            binary_hash = compute_hash_from_embedding(emb_128, proj_matrix)
            hash_str = ''.join(map(str, binary_hash.tolist()))
            return hash_str, emb_512
        return None, None

    def capture_face(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            filename = os.path.join(save_dir, "capture_temp.jpg")
            cv2.imwrite(filename, frame)
            self.display_image(frame, self.image_label1)
            hash_str, emb = self.compute_hash_and_embedding(frame)
            if hash_str:
                upload_hashes[0] = hash_str
                upload_embeddings[0] = emb
                self.hash_label1.config(text=f"Hash 1: {hash_str}")
        else:
            messagebox.showerror("Error", "Failed to capture image")

    def upload_image(self, slot):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if path:
            img = cv2.imread(path)
            label = self.image_label1 if slot == 0 else self.image_label2
            self.display_image(img, label)
            hash_str, emb = self.compute_hash_and_embedding(img)
            if hash_str:
                upload_hashes[slot] = hash_str
                upload_embeddings[slot] = emb
                if slot == 0:
                    self.hash_label1.config(text=f"Hash 1: {hash_str}")
                else:
                    self.hash_label2.config(text=f"Hash 2: {hash_str}")

    def compare_hashes(self):
        h1, h2 = upload_hashes
        e1, e2 = upload_embeddings
        if h1 and h2 and e1 is not None and e2 is not None:
            hamming_dist = sum(c1 != c2 for c1, c2 in zip(h1, h2))
            cosine_sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
            self.result_label.config(
                text=f"🧮 Hamming Distance: {hamming_dist}\n📐 Cosine Similarity: {cosine_sim:.4f}"
            )
        else:
            self.result_label.config(text="⚠️ Please provide both valid hashes and embeddings.")

if __name__ == '__main__':
    root = tk.Tk()
    app = FaceHashApp(root)
    root.mainloop()
