from modules.face_detector import detect_largest_face

def extract_embedding(frame, return_bbox=False):
    face = detect_largest_face(frame)
    if face is None:
        return (None, None) if return_bbox else None
    embedding = face.embedding
    bbox = face.bbox.astype(int) if return_bbox else None
    return (embedding, bbox) if return_bbox else embedding