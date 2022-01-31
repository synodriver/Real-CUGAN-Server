from sklearn.preprocessing import scale
from upcunet_v3 import RealWaifuUpScaler
from cv2 import imread, imencode
from flask import Flask, request
from gevent import pywsgi
from urllib.request import unquote, quote
from sys import argv
from urllib3 import PoolManager
from time import time
from io import BytesIO
from hashlib import md5
from os import mkdir
from os.path import exists

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
    model = f"weights_v3/up{scale}x-latest-{model}.pth"
    m = f"{model}_{tile}"
    if m in ups: m = ups[m]
    else:
        ups[m] = RealWaifuUpScaler(scale, model, half=False, device="cpu:0")
        m = ups[m]
    return m(frame, tile_mode=tile)[:, :, ::-1]

# path is input image path, result is cv2 image
def calcpath(model: str, scale: int, tile: int, path: str):
    return calc(model, scale, tile, imread(path)[:, :, [2, 1, 0]])

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
    if request.method == 'GET':
        url = get_arg("url")
        if url == None: return "400 BAD REQUEST: no url", 400
        url = unquote(url)
        global pool
        clear_pool()
        r = pool.request('GET', url)
        data = r.data
        r.release_conn()
    else:
        data = request.get_data()
    if not len(data): return "400 BAD REQUEST: zero data len", 400
    m = "tmp/"+md5(data+model.encode()+f"{scale}_{tile}".encode()).hexdigest()
    if exists(m):
        with open(m, "rb") as f:
            data = f.read()
            f.close()
    else:
        with open(m, "wb") as f:
            f.write(data)
            f.close()
        img = calcpath(model, scale, tile, m)
        _, data = imencode(".webp", img)
        data = data.tobytes()
        if not len(data): return "400 BAD REQUEST: zero output data len", 400
        with open(m, "wb") as f:
            f.write(data)
            f.close()
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