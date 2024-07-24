import os
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from ThreadPool import ThreadPool
from util import extract_label


def scale_to_0_255(img):
    min_val = np.min(img)
    max_val = np.max(img)
    new_img = (img - min_val) / (max_val - min_val)  # 0-1
    new_img *= 255
    return new_img


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


def Gabor(img):
    img = cv2.resize(img, (7 * 96, 7 * 103))
    gabor_bank = generate_gabor_bank(num_kernels=16)
    avg_out = np.zeros(img.shape)

    for idx, kernel in enumerate(gabor_bank):
        res = apply_sliding_window_on_1_channel(img, kernel)
        avg_out += res

    avg_out = avg_out / len(gabor_bank)
    avg_out = avg_out.astype(np.uint8)
    avg_out = cv2.resize(avg_out, (90, 90))
    return avg_out


def process_image(index, img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = Gabor(img)
    img = img[:, :, np.newaxis]
    return [index, img, extract_label(img_path)]


# Config
DATASET_RAW_DIR = "./dataset-raw"
DATASET_DIR = "./dataset-enhanced"
THREAD_COUNT = os.cpu_count()
DATASET = [
    "Real",
    "Altered/Altered-Easy",
    "Altered/Altered-Medium",
    "Altered/Altered-Hard",
]

# Create thread pool with number of threads equal to CPU cores count
thread_pool = ThreadPool(num_threads=THREAD_COUNT, verbose=True)

# Read image list & init variables
paths_dict = {
    dataset: sorted(glob.glob(os.path.join(f"{DATASET_RAW_DIR}/", dataset, "*.BMP")))
    for dataset in DATASET
}
imgs_dict = {
    dataset: np.empty((len(paths_dict[dataset]), 90, 90, 1), dtype=np.uint8)
    for dataset in DATASET
}
labels_dict = {
    dataset: np.empty((len(paths_dict[dataset]), 4), dtype=np.uint16)
    for dataset in DATASET
}

for dataset in DATASET:
    real_len = len(paths_dict[dataset])
    print(
        f'Generating dataset-original "{dataset}" with {real_len}/{len(paths_dict[dataset])} images'
    )
    thread_pool.reset(real_len)

    for index, img_path in enumerate(paths_dict[dataset][:real_len]):
        thread_pool.add_task(process_image, index, img_path)

    thread_pool.wait_completion()

    for result in list(thread_pool.results.queue):
        index, img, label = result
        imgs_dict[dataset][index] = img
        labels_dict[dataset][index] = label

    os.makedirs(f"{DATASET_DIR}/{dataset}", exist_ok=True)
    np.savez_compressed(f"{DATASET_DIR}/{dataset}/imgs.npz", imgs_dict[dataset])
    np.save(f"{DATASET_DIR}/{dataset}/labels.npy", labels_dict[dataset])
