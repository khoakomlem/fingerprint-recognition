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

# Project directory
PROJECT_DIR = resolve(
    os.path.dirname(os.path.realpath(__file__)), ".."
)  # up one level from src
print("PROJECT_DIR", PROJECT_DIR)  # must the root of the project


# Class house
class House:
    def __init__(self, id: str, name: str, fingerprints: list[np.ndarray]):
        self.id = id
        self.name = name
        self.fingerprints = fingerprints


# Generate i
current_id = 0


def get_id():
    global current_id
    current_id += 1
    return str(current_id)


model = None
house_map: dict[str, House] = {}


# Init database
def init_database():
    global current_id
    json_database = json.load(open(resolve(PROJECT_DIR, "database/database.json")))
    house_map.clear()
    for json_house in json_database:
        house_map[json_house["id"]] = House(json_house["id"], json_house["name"], [])
        house_map[json_house["id"]].fingerprints = [
            cv2.imread(
                str(resolve(PROJECT_DIR, "database/fingerprints", filename)),
                cv2.IMREAD_GRAYSCALE,
            )
            for i, filename in enumerate(json_house["fingerprints"])
        ]
        house_map[json_house["id"]].fingerprints = [
            format_image_prediction(fingerprint)
            for fingerprint in house_map[json_house["id"]].fingerprints
        ]
        current_id = max(current_id, int(json_house["id"]))

    print(f"Database initialized!")


def process_image(img):
    target_gabor_h = 300
    print(img.shape, "scale", target_gabor_h / img.shape[0])
    img = scale_image(img, target_gabor_h / img.shape[0])

    # if np.mean(img) > 200:
    img = cv2.equalizeHist(img)
    img = fingerprint_enhancer.enhance_Fingerprint(img)
    img = cv2.bitwise_not(img)

    img[img > 200] = 255
    img[img < 100] = 0
    return img


def init_model():
    global model
    MODEL_PATH = resolve(PROJECT_DIR, "result/finalized_model22.h5")
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()
    print(f"Model loaded from {MODEL_PATH}")


def serialize_database() -> list[dict]:
    json_database = []
    for house in house_map.values():
        json_database.append(
            {
                "id": house.id,
                "name": house.name,
                "fingerprints": [
                    f"{house.id}-{i}.bmp" for i in range(len(house.fingerprints))
                ],
            }
        )
    return json_database


def save_database():
    json_database = serialize_database()
    json.dump(
        json_database,
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
    name, fingerprints = data["name"], data["fingerprints"]
    id = get_id()
    imgs = []
    for fingerprint in fingerprints:
        img = base64_to_image(fingerprint)
        img = process_image(img)
        filename = f"{id}-{len(imgs)}.bmp"
        cv2.imwrite(resolve(PROJECT_DIR, "database/fingerprints", filename), img)
        img = format_image_prediction(img)
        imgs.append(img)
    house_map[id] = House(id, name, imgs)
    save_database()
    return {
        "id": id,
        "name": name,
        "fingerprints": [f"{id}-{i}.bmp" for i in range(len(imgs))],
    }


def compare_fingerprint(fingerprint1, fingerprint2) -> float:
    pred = model.predict([fingerprint1, fingerprint2])
    return float(np.max(pred))


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

    best_score = 0
    id_match = "0"
    filename_match = "unknown.bmp"

    for house in house_map.values():
        for i, fingerprint in enumerate(house.fingerprints):
            score = compare_fingerprint(img, fingerprint)
            if score > best_score:
                best_score = score
                id_match = house.id
                filename_match = f"{id_match}-{i}.bmp"

    if id_match == "0":
        return {
            "id": id_match,
            "score": 0,
            "name": "Unknown",
            "filename": filename_match,
        }

    return {
        "id": id_match,
        "score": best_score,
        "name": house_map[id_match].name,
        "filename": filename_match,
    }


def delete_fingerprint(data):
    print("delete_fingerprint", data)
    id = data["id"]
    found = [(k, v) for k, v in house_map.items() if k == id]
    if len(found) == 0:
        return f"House with id: {id} not found"
    house = found[0][1]

    for i, fingerprint in enumerate(house.fingerprints):
        os.remove(resolve(PROJECT_DIR, "database/fingerprints", f"{id}-{i}.bmp"))
    del house_map[id]

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
            print(serialize_database())
            self.response(200, serialize_database())

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
