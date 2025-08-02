from insightface.app import FaceAnalysis

# Initialize once
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)  # Use 0 for GPU, -1 for CPU

def detect_largest_face(frame):
    faces = app.get(frame)
    if not faces:
        return None

    # Select the largest face (by bounding box area)
    largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    return largest_face
