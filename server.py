from hashlib import md5
from io import BytesIO
from os.path import exists, abspath, dirname, join
from typing import Optional

from cv2 import IMREAD_UNCHANGED, imdecode, imencode

from fastapi import FastAPI, Query, File, UploadFile
from fastapi.responses import UJSONResponse, StreamingResponse
import aiohttp

from numpy import frombuffer, uint8

from .upcunet_v3 import RealWaifuUpScaler

app = FastAPI(title=__name__)

ups = {}

# def get_arg(key: str) -> str:
#     return request.args.get(key)


last_req_time = 0


# def clear_pool() -> None:
#     global pool, last_req_time
#     if time() - last_req_time > 60:
#         pool.clear()
#         last_req_time = time()


# frame, result is all cv2 image
def calc(model: str, scale: int, tile: int, frame):
    m = f"{model}_{tile}"
    if m in ups:
        m = ups[m]
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
CURRENT_FOLDER = abspath(dirname(__file__))


@app.get("/scale")
async def scale(
    model: Optional[str] = Query("no-denoise", description="which model to use"),
    scale: Optional[int] = Query(2, description="scale size 2-4"),
    tile: Optional[int] = Query(2, description="tile 0-9"),
    url: Optional[str] = Query(None, description="url of a picture"),
):
    # model = get_arg("model")
    # scale = get_arg("scale")
    # tile = get_arg("tile")
    # print(model, scale, tile)
    #
    # if model is None:
    #     model = "no-denoise"
    # if scale is None:
    #     scale = 2
    # if tile is None:
    #     tile = 2

    if model not in MODEL_LIST:
        return UJSONResponse({"status": "no such model"}, 400)
    if scale not in [2, 3, 4]:
        return UJSONResponse({"status": "no such scale"}, 400)
    if tile not in range(9):
        return UJSONResponse({"status": "no such tile"}, 400)

    model = join(CURRENT_FOLDER, "weights_v3", f"up{scale}x-latest-{model}.pth")
    if not exists(model):
        return UJSONResponse({"status": "no such model"}, 400)

    if url is None:
        return UJSONResponse({"status": "no url"}, 400)
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.read()
    if not data:
        return UJSONResponse({"status": "url resp no data"}, 400)
    path = join(
        CURRENT_FOLDER, "tmp", md5(data + f"{model}_{tile}".encode()).hexdigest()
    )  # todo 加个选项控制缓存路径
    if exists(path):
        with open(path, "rb") as f:
            data = f.read()
    else:
        _, data = imencode(
            ".webp", calcdata(model, scale, tile, data)
        )  # todo 这里改成传memoryview?
        data = data.tobytes()
        if not data:
            return UJSONResponse({"status": "zero output data len"}, 500)
        with open(path, "wb") as f:
            f.write(data)
    return StreamingResponse(BytesIO(data), 200, {"Content-Type": "image/webp"})
    # return data, 200, {"Content-Type": "image/webp", "Content-Length": len(data)}
    # if request.method == "GET":
    #     url = get_arg("url")
    #     if url == None:
    #         return "400 BAD REQUEST: no url", 400
    #     url = unquote(url)
    #     global pool
    #     clear_pool()
    #     r = pool.request("GET", url)
    #     data = r.data
    #     r.release_conn()
    #     del r
    # else:
    #     data = request.get_data(as_text=False)
    # if not len(data):
    #     return "400 BAD REQUEST: zero data len", 400
    # m = "tmp/" + md5(data + f"{model}_{tile}".encode()).hexdigest()
    # if exists(m):
    #     with open(m, "rb") as f:
    #         data = f.read()
    # else:
    #     _, data = imencode(".webp", calcdata(model, scale, tile, data))
    #     data = data.tobytes()
    #     if not len(data):
    #         return "500 Internal Server Error: zero output data len", 500
    #     with open(m, "wb") as f:
    #         f.write(data)
    # return data, 200, {"Content-Type": "image/webp", "Content-Length": len(data)}


@app.post("/scale")
async def scale_(
    model: Optional[str] = Query("no-denoise", description="which model to use"),
    scale: Optional[int] = Query(2, description="scale size 2-4"),
    tile: Optional[int] = Query(2, description="tile 0-9"),
    file: UploadFile = File(..., description="a picture"),
):
    if model not in MODEL_LIST:
        return UJSONResponse({"status": "no such model"}, 400)
    if scale not in [2, 3, 4]:
        return UJSONResponse({"status": "no such scale"}, 400)
    if tile not in range(9):
        return UJSONResponse({"status": "no such tile"}, 400)

    model = join(CURRENT_FOLDER, "weights_v3", f"up{scale}x-latest-{model}.pth")
    if not exists(model):
        return UJSONResponse({"status": "no such model"}, 400)

    data = await file.read()
    if not data:
        return UJSONResponse({"status": "url resp no data"}, 400)
    path = join(
        CURRENT_FOLDER, "tmp", md5(data + f"{model}_{tile}".encode()).hexdigest()
    )  # todo 加个选项控制缓存路径
    if exists(path):
        with open(path, "rb") as f:
            data = f.read()
    else:
        _, data = imencode(".webp", calcdata(model, scale, tile, data))
        data = data.tobytes()
        if not data:
            return UJSONResponse({"status": "zero output data len"}, 500)
        with open(path, "wb") as f:
            f.write(data)
    # print(len(data))
    return StreamingResponse(BytesIO(data), 200, {"Content-Type": "image/webp"})


# def handle_client():
#     global app
#     host = argv[1]
#     port = int(argv[2])
#     print("Starting SC at:", host, port)
#     pywsgi.WSGIServer((host, port), app).serve_forever()


# if __name__ == "__main__":
#     if not exists("tmp"):
#         mkdir("tmp", 0o755)
#     if len(argv) == 3:
#         handle_client()
#     else:
#         print("Usage: <host> <port>")
