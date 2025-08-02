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

captured_hashes = []
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

class FaceHashApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Hash App")
        self.root.geometry("800x600")

        self.label = tk.Label(root, text="Face Recognition Hash Generator", font=("Helvetica", 16))
        self.label.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.capture_button = tk.Button(root, text="Capture from Webcam", command=self.capture_face)
        self.capture_button.pack(pady=10)

        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.hash_label = tk.Label(root, text="Hash will appear here")
        self.hash_label.pack(pady=5)

        self.hamming_button = tk.Button(root, text="Compare Last Two Hashes", command=self.compare_hashes)
        self.hamming_button.pack(pady=10)

        self.hamming_result = tk.Label(root, text="")
        self.hamming_result.pack(pady=5)

    def display_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((400, 300))
        img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img)
        self.image_label.image = img

    def compute_and_display_hash(self, image):
        emb_512 = extract_embedding(image)
        if emb_512 is not None:
            emb_128 = reduce_embedding(pca, emb_512)
            binary_hash = compute_hash_from_embedding(emb_128, proj_matrix)
            hash_str = ''.join(map(str, binary_hash.tolist()))
            captured_hashes.append(hash_str)
            self.hash_label.config(text=f"🔐 Hash: {hash_str}")
        else:
            self.hash_label.config(text="[ERROR] No face detected.")

    def capture_face(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            filename = os.path.join(save_dir, "capture_temp.jpg")
            cv2.imwrite(filename, frame)
            self.display_image(frame)
            self.compute_and_display_hash(frame)
        else:
            messagebox.showerror("Error", "Failed to capture image")

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if path:
            img = cv2.imread(path)
            self.display_image(img)
            self.compute_and_display_hash(img)

    def compare_hashes(self):
        if len(captured_hashes) >= 2:
            h1 = captured_hashes[-1]
            h2 = captured_hashes[-2]
            dist = sum(c1 != c2 for c1, c2 in zip(h1, h2))
            self.hamming_result.config(text=f"🧮 Hamming Distance: {dist}")
        else:
            self.hamming_result.config(text="Need at least two hashes to compare")


if __name__ == '__main__':
    root = tk.Tk()
    app = FaceHashApp(root)
    root.mainloop()
