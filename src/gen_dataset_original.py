import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob, os
from ThreadPool import ThreadPool


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


def process_image(index, img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (90, 90))
    img = img[:, :, np.newaxis]
    return [index, img, extract_label(img_path)]


# Config
DATASET_RAW_DIR = "./dataset-raw"
DATASET_DIR = "./dataset-original"
THREAD_COUNT = os.cpu_count()
DATASET = [
    "Real",
    "Altered/Altered-Easy",
    "Altered/Altered-Medium",
    "Altered/Altered-Hard",
]

# Create thread pool with number of threads equal to CPU cores count
thread_pool = ThreadPool(num_threads=THREAD_COUNT, verbose=False)

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
