import insightface

# Load InsightFace model only once
model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0)

def extract_embedding(image):
    faces = model.get(image)
    if len(faces) > 0:
        return faces[0].embedding
    return None
