import http.server
import socketserver
import json
import numpy as np
import cv2
import base64
from util import (
    resolve,
    format_image_prediction,
    scale_image,
)
import os
import tensorflow as tf
import traceback
import matplotlib.pyplot as plt
import fingerprint_enhancer
import json
from uuid import uuid4
from glob import glob

PORT = 8000
MIN_SCORE = 0.94
DEFAULT_MODEL = "gabor15epochs"

# Project directory
PROJECT_DIR = resolve(
    os.path.dirname(os.path.realpath(__file__)), ".."
)  # up one level from src
print("PROJECT_DIR", PROJECT_DIR)  # must the root of the project


# Class house
class House:
    def __init__(self, id: str, name: str, fingerprint_ids: list[str]):
        self.id = id
        self.name = name
        self.fingerprint_ids = fingerprint_ids


def get_id():
    return uuid4().hex


MODELPATHS = glob(resolve(PROJECT_DIR, "database/models/*.h5"))
FILTERS = ["none", "gabor"]

model_map: dict[str, tf.keras.Model] = {}
house_map: dict[str, House] = {}
fingerprint_map: dict[str, np.ndarray] = {}


# Init database
def init_database():
    json_database = json.load(open(resolve(PROJECT_DIR, "database/database.json")))
    house_map.clear()
    for json_house in json_database:
        house_map[json_house["id"]] = House(json_house["id"], json_house["name"], [])
        house_map[json_house["id"]].fingerprint_ids = json_house["fingerprint_ids"]
        for fingerprint_id in json_house["fingerprint_ids"]:
            img = cv2.imread(
                resolve(PROJECT_DIR, "database/fingerprints", f"{fingerprint_id}.bmp"),
                cv2.IMREAD_GRAYSCALE,
            )
            img = format_image_prediction(img)
            fingerprint_map[fingerprint_id] = img

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
    for model_path in MODELPATHS:
        model = tf.keras.models.load_model(model_path)
        modelname = os.path.basename(model_path).split(".")[0]
        model_map[modelname] = model
        print(f"Model loaded from {model_path} with name {modelname}")


def serialize_database() -> list[dict]:
    json_database = []
    for house in house_map.values():
        json_database.append(
            {
                "id": house.id,
                "name": house.name,
                "fingerprint_ids": house.fingerprint_ids,
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


def is_unique_fingerprint(modelname: str, fingerprint: np.ndarray) -> bool:
    img = format_image_prediction(fingerprint)
    for fingerprint_id in fingerprint_map.keys():
        if (
            compare_fingerprint(
                model_map[modelname], img, fingerprint_map[fingerprint_id]
            )
            > MIN_SCORE
        ):
            return False
    return True


def register_house(data):
    (name, fingerprints, modelname, filtername) = (
        data["name"],
        data["fingerprints"],
        data["model"],
        data["filter"],
    )
    fingerprint_ids = []
    imgs = []
    for fingerprint in fingerprints:
        img = base64_to_image(fingerprint)
        if filtername == "gabor":
            img = process_image(img)
        if not is_unique_fingerprint(modelname, img):
            raise Exception("Fingerprint already exists")
        imgs.append(img)

    for i, img in enumerate(imgs):
        id = get_id()
        filename = f"{id}.bmp"
        cv2.imwrite(resolve(PROJECT_DIR, "database/fingerprints", filename), img)
        img = format_image_prediction(img)
        fingerprint_map[id] = img
        fingerprint_ids.append(id)

    house_map[id] = House(id, name, fingerprint_ids)
    save_database()
    return {
        "id": id,
        "name": name,
        "fingerprint_ids": fingerprint_ids,
    }


def compare_fingerprint(
    model: tf.keras.Model, fingerprint1: np.ndarray, fingerprint2: np.ndarray
) -> float:
    pred = model.predict([fingerprint1, fingerprint2])
    return float(np.max(pred))


def find_fingerprint(data):
    img, modelname, filtername = (
        base64_to_image(data["fingerprint"]),
        data["model"],
        data["filter"],
    )
    if filtername == "gabor":
        img = process_image(img)
    img = format_image_prediction(img)

    best_score: float = 0
    id_match = "0"
    fingerprint_match = "unknown"

    for house in house_map.values():
        for i, fingerprint_id in enumerate(house.fingerprint_ids):
            score = compare_fingerprint(
                model_map[modelname], img, fingerprint_map[fingerprint_id]
            )
            if score > best_score:
                best_score = score
                id_match = house.id
                fingerprint_match = fingerprint_id

    if id_match == "0":
        return {
            "id": id_match,
            "score": 0,
            "name": "Unknown",
            "fingerprint_id": fingerprint_match,
        }

    return {
        "id": id_match,
        "score": best_score,
        "name": house_map[id_match].name,
        "fingerprint_id": fingerprint_match,
    }


def delete_house(data):
    print("delete_house", data)
    id = data["id"]
    found = [(k, v) for k, v in house_map.items() if k == id]
    if len(found) == 0:
        return f"House with id: {id} not found"
    house = found[0][1]

    for i, fingerprint in enumerate(house.fingerprint_ids):
        fingerprint_map.pop(fingerprint)
        os.remove(resolve(PROJECT_DIR, "database/fingerprints", f"{fingerprint}.bmp"))
    house_map.pop(id)

    save_database()
    return f"Delete success with id: {id}"


def update_house(data):
    print("update_house", data)
    id, name, fingerprints = data["id"], data["name"], data["fingerprints"]

    if id not in house_map:
        return f"House with id: {id} not found"
    house = house_map[id]

    # remove old fingerprints
    for fingerprint_id in house.fingerprint_ids:
        os.remove(
            resolve(PROJECT_DIR, "database/fingerprints", f"{fingerprint_id}.bmp")
        )
        fingerprint_map.pop(fingerprint_id)

    house.name = name
    house.fingerprint_ids = []

    for fingerprint in fingerprints:
        id = get_id()
        img = base64_to_image(fingerprint)
        img = process_image(img)
        filename = f"{id}.bmp"
        cv2.imwrite(resolve(PROJECT_DIR, "database/fingerprints", filename), img)
        img = format_image_prediction(img)
        fingerprint_map[id] = img
        house.fingerprint_ids.append(id)

    save_database()
    return {
        "id": id,
        "name": name,
        "fingerprint_ids": house.fingerprint_ids,
    }


def get_fingerprint(data):
    print("get_fingerprint", data)
    fingerprint_id = data["fingerprint_id"]
    img = cv2.imread(
        resolve(PROJECT_DIR, "database/fingerprints", f"{fingerprint_id}.bmp"),
        cv2.IMREAD_GRAYSCALE,
    )
    _, buffer = cv2.imencode(".bmp", img)
    return base64.b64encode(buffer).decode("utf-8")


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
                self.response(200, register_house(data))
            elif self.path == "/find":
                self.response(200, find_fingerprint(data))
            elif self.path == "/delete":
                self.response(200, delete_house(data))
            elif self.path == "/update":
                self.response(200, update_house(data))
            else:
                self.response(404, None, "Not Found")
        except Exception as e:
            traceback.print_exc()
            self.response(500, None, repr(e))

    def do_GET(self):
        if self.path == "/":
            self.response(200, serialize_database())
        elif self.path == "/models":
            modelnames = [os.path.basename(model).split(".")[0] for model in MODELPATHS]
            self.response(200, modelnames)
        elif self.path.startswith("/fingerprint"):
            # print(self.parse_formdata())
            self.response(200, get_fingerprint(self.parse_query()))
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

    def parse_query(self):
        form_data = {}
        for item in self.path.split("?")[1].split("&"):
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
