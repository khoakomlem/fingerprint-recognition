import http.server
import socketserver
import json


PORT = 8000


def register_fingerprint(data):
    return "register success"


def compare_fingerprint(data):
    return 0.8


def delete_fingerprint(data):
    return "delete success"


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
                # "message": self.message,
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
            elif self.path == "/compare":
                self.response(200, compare_fingerprint(data))
            elif self.path == "/delete":
                self.response(200, delete_fingerprint(data))
            else:
                self.response(404, None, "Not Found")
        except:
            self.response(500, None, "Internal Server Error")
        print(f"Received data: {data} at {self.path}")

    def response(self, code, data, message=""):
        self.send_response(code)
        self.send_header("Content-type", "text/json")
        self.end_headers()
        response = Response(code, data, message)
        self.wfile.write(response.json())

    def parse_body(self):
        content_length = int(self.headers["Content-Length"])  # Get the size of data
        post_data = self.rfile.read(content_length)  # Get the data
        try:
            return json.loads(post_data.decode("utf-8"))
        except:
            return {}


with socketserver.TCPServer(("", PORT), SimpleHTTPRequestHandler) as httpd:
    print(f"Serving at port {PORT}")
    httpd.serve_forever()
