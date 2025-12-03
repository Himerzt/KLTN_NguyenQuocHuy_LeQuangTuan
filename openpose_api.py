# openpose_api.py
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import cv2
import os, importlib.util

app = FastAPI(title="OpenPose API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====== OPENPOSE SETUP ======
pyd_path = r"D:\KLTN_NguyenQuocHuy_LeQuangTuan\openpose\bin\python\openpose\Release\pyopenpose.cp37-win_amd64.pyd"
dll_dir1 = r"D:\KLTN_NguyenQuocHuy_LeQuangTuan\openpose\x64\Release"
dll_dir2 = r"D:\KLTN_NguyenQuocHuy_LeQuangTuan\openpose\bin"


# thêm đường dẫn chứa DLL
os.environ['PATH'] += ";" + dll_dir1 + ";" + dll_dir2

# nạp file .pyd thủ công
spec = importlib.util.spec_from_file_location("pyopenpose", pyd_path)
pyopenpose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pyopenpose)
op = pyopenpose

print("✅ OpenPose imported successfully (manual load)!")

# ====== KHỞI TẠO OPENPOSE ======
params = {
    "model_folder": r"D:\KLTN_NguyenQuocHuy_LeQuangTuan\openpose\models",
    "net_resolution": "-1x192",
    "model_pose": "BODY_25",
    "render_pose": 0,           
    "disable_blending": True,
    "logging_level": 3
}

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


@app.get("/")
def home():
    return {"message": "OpenPose API is running"}


@app.post("/extract-keypoints")
async def extract_keypoints(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    datum = op.Datum()
    datum.cvInputData = img
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    if datum.poseKeypoints is None:
        return {
            "message": "No keypoints detected",
            "num_people": 0,
            "keypoints": []
        }

    all_keypoints = []
    num_people = int(datum.poseKeypoints.shape[0])

    # Ngưỡng confidence – dưới ngưỡng coi như "missing"
    CONF_THRESH = 0.1

    for i in range(num_people):
        kp = datum.poseKeypoints[i] 
        kp = kp.astype(np.float32)

        # Với joint có conf < thresh → set về (0,0,0)
        low_conf = kp[:, 2] < CONF_THRESH
        kp[low_conf] = 0.0

        all_keypoints.append(kp.tolist())

    return {
        "message": "Keypoints extracted successfully",
        "num_people": num_people,
        "keypoints": all_keypoints
    }



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
