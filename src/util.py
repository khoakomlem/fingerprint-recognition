import os
import numpy as np
import cv2
import base64
import pathlib
import fingerprint_enhancer


def resolve(*args):
    return str(pathlib.Path(os.path.join(*args)).resolve())


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


def format_image_prediction(img):
    img = resize_and_pad_image(img, (90, 90))
    return img.reshape((1, 90, 90, 1)).astype(np.float32) / 255.0


def combine_label(tup):
    gender = ["M", "F"][tup[1]]
    lr = ["Left", "Right"][tup[2]]
    finger = ["thumb", "index", "middle", "ring", "little"][tup[3]]
    return f"{tup[0]}__{gender}_{lr}_{finger}_finger"


def base64_to_image(base64_str):
    # Decode the base64 string to binary data
    image_data = base64.b64decode(base64_str)

    # Convert binary data to a NumPy array
    np_arr = np.frombuffer(image_data, np.uint8)

    # Decode the NumPy array to an image
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    return img


# ---- GABOR UTILS ----
def apply_sliding_window_on_1_channel(img, kernel):
    return cv2.filter2D(src=img, ddepth=-1, kernel=kernel)


def generate_gabor_bank(
    num_kernels, ksize=(15, 15), sigma=3, lambd=7.3, gamma=0.25, psi=0
):
    bank = []
    theta = 0
    step = np.pi / num_kernels
    for idx in range(num_kernels):
        theta = idx * step
        kernel = cv2.getGaborKernel(
            ksize=ksize, sigma=sigma, theta=theta, lambd=lambd, gamma=gamma, psi=psi
        )
        bank.append(kernel)
    return bank


def scale_image(img, scale):
    return cv2.resize(img, (0, 0), fx=scale, fy=scale)


def resize_and_pad_image(image, target_size=(90, 90)):
    old_size = image.shape[:2]  # old_size is in (height, width) format
    ratio = float(target_size[0]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # Resize the image with the new size
    resized_image = cv2.resize(
        image, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA
    )

    # Compute padding
    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # Pad the image to the target size with white background
    color = [255, 255, 255]  # White padding
    padded_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return padded_image


def Gabor(img):
    target_gabor_h = 824
    target_train_size = (90, 90)

    img = scale_image(img, img.shape[0] / target_gabor_h)

    gabor_bank = generate_gabor_bank(num_kernels=16)
    avg_out = np.zeros(img.shape)

    for idx, kernel in enumerate(gabor_bank):
        res = apply_sliding_window_on_1_channel(img, kernel)
        avg_out += res

    avg_out = avg_out / len(gabor_bank)
    avg_out = avg_out.astype(np.uint8)
    # avg_out = cv2.equalizeHist(avg_out)
    # avg_out[avg_out < 130] = 0
    # avg_out[avg_out > 130] = 255
    avg_out = resize_and_pad_image(avg_out, target_size=target_train_size)
    # avg_out[avg_out < 130] = 0
    # avg_out[avg_out > 130] = 255
    return avg_out
