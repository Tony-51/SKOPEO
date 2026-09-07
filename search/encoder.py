import os
from functools import lru_cache

import cv2
import numpy as np


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "face_recognition_sface_2021dec.onnx",
)


@lru_cache(maxsize=1)
def load_model():
    """
    Load SFace exactly once per Python process.

    The previous implementation created a new ONNX model for every face
    embedding. With ~167 web candidates this caused a large amount of
    unnecessary model initialization overhead.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"SFace model not found at: {MODEL_PATH}"
        )

    return cv2.FaceRecognizerSF.create(MODEL_PATH, "")


def generate_embedding(image, face_box):
    """Generate an SFace embedding using the cached model."""
    model = load_model()

    face = np.asarray(face_box, dtype=np.float32)

    aligned_face = model.alignCrop(image, face)
    embedding = model.feature(aligned_face)

    return embedding
