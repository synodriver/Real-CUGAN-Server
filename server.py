from numpy import frombuffer, uint8
from upcunet_v3 import RealWaifuUpScaler
from cv2 import imencode, imdecode, IMREAD_UNCHANGED
from flask import Flask, request
from gevent import pywsgi
from urllib.request import unquote
from sys import argv
from urllib3 import PoolManager
from time import time
from hashlib import md5
from os import mkdir
from os.path import exists
from threading import Lock

app = Flask(__name__)
pool = PoolManager()
ups = {}

def get_arg(key: str) -> str:
    return request.args.get(key)

last_req_time = 0
def clear_pool() -> None:
    global pool, last_req_time
    if time() - last_req_time > 60:
        pool.clear()
        last_req_time = time()

# frame, result is all cv2 image
def calc(model: str, scale: int, tile: int, frame):
    m = f"{model}_{tile}"
    if m in ups: m = ups[m]
    else:
        ups[m] = RealWaifuUpScaler(scale, model, half=False, device="cpu:0")
        m = ups[m]
    img = m(frame, tile_mode=tile)[:, :, ::-1]
    del frame
    return img

# data is image data, result is cv2 image
# data will be deleted
def calcdata(model: str, scale: int, tile: int, data: bytes):
    umat = frombuffer(data, uint8)
    del data
    frame = imdecode(umat, IMREAD_UNCHANGED)[:, :, [2, 1, 0]]
    del umat
    return calc(model, scale, tile, frame)

MODEL_LIST = ["conservative", "no-denoise", "denoise1x", "denoise2x", "denoise3x"]

@app.route("/scale", methods=['GET', 'POST'])
def scale():
    model = get_arg("model")
    scale = get_arg("scale")
    tile = get_arg("tile")
    print(model, scale, tile)

    if model == None: model = "no-denoise"
    if scale == None: scale = "2"
    if tile == None: tile = "2"
    scale = int(scale)
    tile = int(tile)

    if model not in MODEL_LIST: return "400 BAD REQUEST: no such model", 400
    if scale not in [2, 3, 4]: return "400 BAD REQUEST: no such scale", 400
    if tile not in range(9): return "400 BAD REQUEST: no such tile", 400

    model = f"weights_v3/up{scale}x-latest-{model}.pth"
    if not exists(model): return "400 BAD REQUEST: no such model", 400

    if request.method == 'GET':
        url = get_arg("url")
        if url == None: return "400 BAD REQUEST: no url", 400
        url = unquote(url)
        global pool
        clear_pool()
        r = pool.request('GET', url)
        data = r.data
        r.release_conn()
        del r
    else:
        data = request.get_data(as_text=False)
    if not len(data): return "400 BAD REQUEST: zero data len", 400
    m = "tmp/"+md5(data+f"{model}_{tile}".encode()).hexdigest()
    if exists(m):
        with open(m, "rb") as f: data = f.read()
    else:
        _, data = imencode(".webp", calcdata(model, scale, tile, data))
        data = data.tobytes()
        if not len(data): return "500 Internal Server Error: zero output data len", 500
        with open(m, "wb") as f: f.write(data)
    return data, 200, {"Content-Type": "image/webp", "Content-Length": len(data)}

def handle_client():
    global app
    host = argv[1]
    port = int(argv[2])
    print("Starting SC at:", host, port)
    pywsgi.WSGIServer((host, port), app).serve_forever()

if __name__ == "__main__":
    if not exists("tmp"): mkdir("tmp", 0o755)
    if len(argv) == 3: handle_client()
    else: print("Usage: <host> <port>")