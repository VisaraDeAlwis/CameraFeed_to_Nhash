from modules.face_detector import detect_largest_face

def extract_embedding(frame):
    face = detect_largest_face(frame)
    if face is None:
        return None
    return face.embedding  # Returns 512D embedding
