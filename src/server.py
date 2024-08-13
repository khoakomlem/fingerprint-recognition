import http.server
import socketserver
import json
import numpy as np
import cv2
import base64
from util import (
    Gabor,
    resolve,
    extract_label,
    combine_label,
    format_image_prediction,
    resize_and_pad_image,
    scale_image,
)
import os
import tensorflow as tf
import traceback
import matplotlib.pyplot as plt
import fingerprint_enhancer
import json

PORT = 8000

PROJECT_DIR = resolve(
    os.path.dirname(os.path.realpath(__file__)), ".."
)  # up one level from src
print("PROJECT_DIR", PROJECT_DIR)  # must the root of the project

fingerprints = []
fingerprint_database = []
current_id = 0
model = None


def get_id():
    global current_id
    current_id += 1
    return int(current_id)


def init_database():
    global current_id
    global fingerprint_database

    fingerprint_paths = sorted(resolve("database/fingerprints").glob("*.bmp"))
    fingerprint_database = json.load(
        open(resolve(PROJECT_DIR, "database/database.json"))
    )

    for i, path in enumerate(fingerprint_paths):
        filename = path.stem
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = format_image_prediction(img)
        label = extract_label(filename)
        fingerprints.append((label, img))
        current_id = max(current_id, label[0])
    print(
        f"Database initialized with {len(fingerprints)} fingerprints, next id: {current_id + 1}"
    )


def process_image(img):
    target_gabor_h = 400
    print(img.shape, "scale", target_gabor_h / img.shape[0])
    img = scale_image(img, target_gabor_h / img.shape[0])

    if np.mean(img) > 200:
        img = cv2.equalizeHist(img)
        img = fingerprint_enhancer.enhance_Fingerprint(img)
        img = cv2.bitwise_not(img)

    img[img > 200] = 255
    img[img < 100] = 0
    return img


def init_model():
    global model
    MODEL_PATH = resolve(PROJECT_DIR, "result/finalized_model2.h5")
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()
    print(f"Model loaded from {MODEL_PATH}")


def save_database():
    json.dump(
        fingerprint_database,
        open(resolve(PROJECT_DIR, "database/database.json"), "w", encoding="utf-8"),
        indent=4,
    )
    print("Database saved")


init_database()
init_model()


def base64_to_image(base64_str):
    image_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    return img


def register_fingerprint(data):
    for it in fingerprint_database:
        if it["name"] == data["name"]:
            return f"Name {data['name']} already registered!"

    img = base64_to_image(data["fingerprint"])
    id = get_id()
    label_text = combine_label((id, data["gender"], data["lr"], data["finger"]))
    img = process_image(img)
    path = "database/fingerprints/" + f"{label_text}.bmp"
    print("Saving fingerprint to", path)
    cv2.imwrite(path, img)
    fingerprints.append((extract_label(label_text), format_image_prediction(img)))
    fingerprint_database.append(
        {
            "id": int(id),
            "name": str(data["name"]),
            "label": label_text,
        }
    )
    save_database()
    return f"register success with id: {id}"


def compare_fingerprint(fingerprint1, fingerprint2, label2):
    pred = model.predict([fingerprint1, fingerprint2])
    return float(np.max(pred)), label2


def test(img):
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 1, 1)
    plt.title("Input: ")
    plt.imshow(img.squeeze(), cmap="gray")
    img = cv2.resize(img, (2 * 96, 2 * 103))
    if np.mean(img) > 200:
        img = cv2.equalizeHist(img)
    img = cv2.resize(img, (2 * 96, 2 * 103))
    img = fingerprint_enhancer.enhance_Fingerprint(img)
    img = cv2.bitwise_not(img)
    img = resize_and_pad_image(img, (90, 90))

    # plt.figure(figsize=(16, 8))
    # plt.subplot(2, 1, 1)
    # plt.title("Input: ")
    # plt.imshow(img.squeeze(), cmap="gray")
    img[img > 200] = 255
    img[img < 100] = 0
    plt.subplot(2, 1, 2)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.waitforbuttonpress()


def find_fingerprint(data):
    img = base64_to_image(data["fingerprint"])
    img = process_image(img)
    img = format_image_prediction(img)

    print(img.shape)

    best_score = 0
    id_match = 0

    for label, fingerprint in fingerprints:
        score, lbl = compare_fingerprint(img, fingerprint, label)
        if score > best_score:
            best_score = score
            id_match = lbl[0]

    if id_match == 0:
        return {
            "id": 0,
            "score": 0,
            "label": "Unknown",
        }

    return {
        "id": int(id_match),
        "name": [f["name"] for f in fingerprint_database if f["id"] == id_match][0],
        "score": best_score,
        "label": combine_label((id_match, label[1], label[2], label[3])),
    }


def delete_fingerprint(data):
    print("delete_fingerprint", data)
    # id = data["id"]
    name = data["name"]
    found_database = [f for f in fingerprint_database if f["name"] == name]
    id = found_database[0]["id"]
    found = [f for f in fingerprints if f[0][0] == id]
    if len(found) == 0:
        return f"No fingerprint found with id: {id}"
    fingerprints[:] = [f for f in fingerprints if f[0][0] != id]
    fingerprint_database[:] = [f for f in fingerprint_database if f["id"] != id]
    os.remove(
        resolve(
            PROJECT_DIR, "database/fingerprints", f"{combine_label(found[0][0])}.bmp"
        )
    )
    save_database()
    return f"Delete success with id: {id}"


class Response:
    def __init__(self, code, data, message):
        self.code = code
        self.data = data
        if code == 200:
            self.status = "ok"
        else:
            self.status = "error"
        self.message = message

    def json(self):
        return json.dumps(
            {
                "status": self.status,
                "code": self.code,
                "data": self.data,
                "message": self.message,
            }
        ).encode("utf-8")


class SimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        data = self.parse_body()
        try:
            if self.path == "/register":
                self.response(200, register_fingerprint(data))
            elif self.path == "/find":
                self.response(200, find_fingerprint(data))
            elif self.path == "/delete":
                self.response(200, delete_fingerprint(data))
            else:
                self.response(404, None, "Not Found")
        except Exception as e:
            traceback.print_exc()
            self.response(500, None, repr(e))

    def do_GET(self):
        if self.path == "/":
            print(fingerprint_database)
            self.response(200, fingerprint_database)

        else:
            self.send_response(404)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"404 Not Found")

    def response(self, code, data, message=""):
        response = Response(code, data, message)
        self.send_response(code)
        self.send_header("Content-type", "text/json")
        self.end_headers()
        self.wfile.write(response.json())

    def parse_body(self):
        content_length = int(self.headers["Content-Length"])  # Get the size of data
        post_data = self.rfile.read(content_length)  # Get the data
        try:
            return json.loads(post_data.decode("utf-8"))
        except:
            return {}

    def parse_formdata(self):
        form_data = {}
        for item in self.rfile.read().decode("utf-8").split("&"):
            key, value = item.split("=")
            form_data[key] = value
        return form_data

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "x-api-key, Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "x-api-key, Content-Type")
        self.end_headers()


with socketserver.TCPServer(("", PORT), SimpleHTTPRequestHandler) as httpd:
    print(f"Serving at port {PORT}")
    httpd.serve_forever()
