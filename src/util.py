import os
import numpy as np


def extract_label(img_path):
    filename, _ = os.path.splitext(os.path.basename(img_path))

    # 1__M_Left_index_finger_CR.BMP
    # 1__M_Left_index_finger.BMP
    subject_id, etc = filename.split("__")
    gender, lr, finger = etc.split("_")[:3]

    gender = 0 if gender == "M" else 1
    lr = 0 if lr == "Left" else 1

    if finger == "thumb":
        finger = 0
    elif finger == "index":
        finger = 1
    elif finger == "middle":
        finger = 2
    elif finger == "ring":
        finger = 3
    elif finger == "little":
        finger = 4

    return np.array([subject_id, gender, lr, finger], dtype=np.uint16)
