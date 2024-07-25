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
)
import os
import tensorflow as tf
import traceback

PORT = 8000

PROJECT_DIR = resolve(
    os.path.dirname(os.path.realpath(__file__)), ".."
)  # up one level from src
print("PROJECT_DIR", PROJECT_DIR)  # must the root of the project

fingerprint_database = []
current_id = 0
model = None


def get_id():
    global current_id
    current_id += 1
    return current_id


def init_database():
    global current_id
    fingerprint_paths = sorted(resolve("database/fingerprints").glob("*.bmp"))
    for i, path in enumerate(fingerprint_paths):
        filename = path.stem
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = Gabor(img)
        img = format_image_prediction(img)
        label = extract_label(filename)
        fingerprint_database.append((label, img))
        current_id = max(current_id, label[0])
    print(
        f"Database initialized with {len(fingerprint_database)} fingerprints, next id: {current_id + 1}"
    )


def init_model():
    global model
    MODEL_PATH = resolve(PROJECT_DIR, "result/finalized_model.h5")
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()
    print(f"Model loaded from {MODEL_PATH}")


init_database()
init_model()


def base64_to_image(base64_str):
    image_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    return img


def register_fingerprint(data):
    img = base64_to_image(data["fingerprint"])
    id = get_id()
    label_text = combine_label((id, data["gender"], data["lr"], data["finger"]))
    cv2.imwrite(resolve(PROJECT_DIR, "database/fingerprints", f"{label_text}.bmp"), img)
    fingerprint_database.append(
        (extract_label(label_text), format_image_prediction(Gabor(img)))
    )
    return f"register success with id: {id}"


def compare_fingerprint(fingerprint1, fingerprint2, label2):
    pred = model.predict([fingerprint1, fingerprint2])
    return round(np.max(pred), 5), label2


def find_fingerprint(data):
    img = base64_to_image(data["fingerprint"])
    img = Gabor(img)
    img = format_image_prediction(img)

    best_score = 0
    id_match = 0

    for label, fingerprint in fingerprint_database:
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
        "score": int(best_score * 10**5),
        "label": combine_label((id_match, label[1], label[2], label[3])),
    }


def delete_fingerprint(data):
    print("delete_fingerprint", data)
    id = data["id"]
    found = [f for f in fingerprint_database if f[0][0] == id]
    if len(found) == 0:
        return f"No fingerprint found with id: {id}"
    fingerprint_database[:] = [f for f in fingerprint_database if f[0][0] != id]
    os.remove(
        resolve(
            PROJECT_DIR, "database/fingerprints", f"{combine_label(found[0][0])}.bmp"
        )
    )
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
    # def do_GET(self):
    #     self.response(200)

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
            self.response(200, None, repr(e))

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


with socketserver.TCPServer(("", PORT), SimpleHTTPRequestHandler) as httpd:
    print(f"Serving at port {PORT}")
    httpd.serve_forever()
