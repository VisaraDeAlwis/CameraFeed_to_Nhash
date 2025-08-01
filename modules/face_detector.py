from mtcnn import MTCNN
import cv2

detector = MTCNN()

def detect_largest_face(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)
    if not faces:
        return None
    largest_face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
    return largest_face
