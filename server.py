import time
from hashlib import md5
from io import BytesIO
from os.path import abspath, dirname, exists, join
from typing import Optional

import aiohttp
from cv2 import IMREAD_UNCHANGED, imdecode, imencode
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import StreamingResponse, UJSONResponse
from numpy import frombuffer, uint8

from .upcunet_v3 import RealWaifuUpScaler
from .utils import run_sync

app = FastAPI(title="图片放大器")

ups = {}

last_req_time = 0


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
@run_sync
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
    format: Optional[str] = Query("png", description="response format"),
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
            f".{format}", await calcdata(model, scale, tile, data)
        )  # todo 这里改成传memoryview?
        data = data.tobytes()
        if not data:
            return UJSONResponse({"status": "zero output data len"}, 500)
        with open(path, "wb") as f:
            f.write(data)
    return StreamingResponse(BytesIO(data), 200, {"Content-Type": f"image/{format}"})


@app.post("/scale")
async def scale_(
    model: Optional[str] = Query("no-denoise", description="which model to use"),
    scale: Optional[int] = Query(2, description="scale size 2-4"),
    tile: Optional[int] = Query(2, description="tile 0-9"),
    file: UploadFile = File(..., description="a picture"),
    format: Optional[str] = Query("png", description="response format"),
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
        _, data = imencode(f".{format}", await calcdata(model, scale, tile, data))
        data = data.tobytes()
        if not data:
            return UJSONResponse({"status": "zero output data len"}, 500)
        with open(path, "wb") as f:
            f.write(data)
    # print(len(data))
    return StreamingResponse(BytesIO(data), 200, {"Content-Type": f"image/{format}"})


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
