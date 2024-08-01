import os
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from ThreadPool import ThreadPool
from util import extract_label
import fingerprint_enhancer
from util import resize_and_pad_image


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


def process_image(img_path, index):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if np.mean(img) > 170:
        img = cv2.equalizeHist(img)
    img = cv2.resize(img, (2 * 96, 2 * 103))
    img = fingerprint_enhancer.enhance_Fingerprint(img)
    img = cv2.bitwise_not(img)
    img = resize_and_pad_image(img, (90, 90))
    img[img > 200] = 255
    img[img < 100] = 0
    img = img[:, :, np.newaxis]
    return img, extract_label(img_path), index


# Config
DATASET_RAW_DIR = "./dataset-raw"
DATASET_DIR = "./dataset-enhanced"
THREAD_COUNT = 24 or os.cpu_count()
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
result_dict = {dataset: [] for dataset in DATASET}

for dataset in DATASET:
    data = np.load(f"{DATASET_DIR}/{dataset}/data.npz", allow_pickle=True)["arr_0"]
    plt.figure(figsize=(16, 8))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(data[i][0], cmap="gray")
        plt.title(data[i][1])
        # plt.axis("off")
    plt.suptitle(f"Dataset: {dataset}")
    plt.show()
input("Press Enter to continue...")
exit()


for dataset in DATASET:
    # real_len = 50
    real_len = len(paths_dict[dataset])
    print(
        f'Generating dataset-original "{dataset}" with {real_len}/{len(paths_dict[dataset])} images'
    )
    thread_pool.reset(real_len)

    for index, img_path in enumerate(paths_dict[dataset][:real_len]):
        thread_pool.add_task(process_image, img_path, index)

    thread_pool.wait_completion()

    for result in list(thread_pool.results.queue):
        img, label, index = result
        result_dict[dataset].append(np.array([img, label], dtype=object))

    os.makedirs(f"{DATASET_DIR}/{dataset}", exist_ok=True)
    np.savez_compressed(f"{DATASET_DIR}/{dataset}/data.npz", result_dict[dataset])
    print(f"Saved dataset-enhanced/{dataset}/data.npz")
