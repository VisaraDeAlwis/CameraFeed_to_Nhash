import insightface
import cv2

# Initialize InsightFace model (includes detection + alignment)
model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0)

def extract_embedding(image_path):
    img = cv2.imread(image_path)
    faces = model.get(img)
    if len(faces) > 0:
        return faces[0].embedding
    return None
