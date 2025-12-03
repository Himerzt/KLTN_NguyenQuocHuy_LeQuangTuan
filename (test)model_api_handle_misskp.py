# model_api.py
import asyncio
import base64
import time
import cv2
import numpy as np
import aiohttp
import uvicorn
from ultralytics import YOLO 
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input,  Dense, Dropout, LSTM)
import requests
from fastapi import File, UploadFile, Form, BackgroundTasks

model_yolo = YOLO("yolov8m.pt")

DEBUG = False

NUM_FRAMES = 12           # Số frame liên tiếp đưa vào LSTM mỗi lần dự đoán  
IMG_SIZE = 224            # Kích thước ảnh đầu vào (224x224)  
NUM_JOINTS = 25           # Số khớp (keypoints) mà OpenPose trích xuất
NUM_FEATURES = NUM_JOINTS * 3 # Mỗi khớp có (x, y, c) → tổng 75 giá trị  

ACTIONS = ["CLIMB", "FALL", "LIEDOWN", "SIT", "STAND"]

MODEL_WEIGHTS = r"kp_handle_miss.weights.h5"
OPENPOSE_URL = "http://127.0.0.1:8001/extract-keypoints"

PROCESS_FPS = 10           # Số frame xử lý mỗi giây (10 fps)  
PROCESS_INTERVAL = 1.0 / PROCESS_FPS  # Thời gian giữa 2 lần xử lý (~0.1 giây)  
FRAME_QUEUE_MAXSIZE = 1    # Hàng đợi frame tối đa (tránh trễ xử lý)
NUM_CLASSES = len(ACTIONS)

TELEGRAM_BOT_TOKEN = "8464653213:AAHbmJ9sEUuhcIvUTY1vaMiFxIDkG2Wa5Z8"
TELEGRAM_CHAT_ID = "5855449751"

# ================== BUILD MODEL ==================
kp_input = Input(shape=(NUM_FRAMES, 75), name="keypoints")

xk = LSTM(128, return_sequences=True, name="kp_lstm1")(kp_input)
xk = Dropout(0.3, name="kp_dropout1")(xk)

xk = LSTM(128, name="kp_lstm2")(xk)
xk = Dropout(0.3, name="kp_dropout2")(xk)

xk = Dense(128, activation="relu", name="kp_dense")(xk)
xk = Dropout(0.3, name="kp_dropout3")(xk)

out = Dense(NUM_CLASSES, activation="softmax", name="cls_head")(xk)

model = Model(inputs=kp_input, outputs=out, name="kp_only_model")

model.load_weights(MODEL_WEIGHTS)
print("LSTM model loaded successfully")

# ================== APP INIT ==================
app = FastAPI(title="Action Recognition API (YOLO + OpenPose + LSTM)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frame_buffer = {}   # Lưu tạm các frame video theo từng người (ID riêng)  
kp_buffer = {}      # Lưu tạm keypoints (tọa độ khớp) tương ứng mỗi người 

aiohttp_session = None

person_history = {} 


@app.on_event("startup")
async def startup_event():
    global aiohttp_session
    aiohttp_session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=20, force_close=False))
    print("aiohttp session created")


@app.on_event("shutdown")
async def shutdown_event():
    global aiohttp_session
    if aiohttp_session:
        await aiohttp_session.close()
        print("aiohttp session closed")


async def call_openpose_async(cropped_frame):
    """Send cropped frame to OpenPose API and return keypoints."""
    global aiohttp_session
    try:
        _, jpg = cv2.imencode(".jpg", cropped_frame)
        form = aiohttp.FormData()
        form.add_field("file", jpg.tobytes(), filename="frame.jpg", content_type="image/jpeg")
        async with aiohttp_session.post(OPENPOSE_URL, data=form, timeout=10) as resp:
            res = await resp.json()
            return res.get("keypoints", [])
    except Exception as e:
        print("OpenPose call failed:", e)
        return []


async def do_model_predict(frames_list, kps_list):
    kp_np = np.array(kps_list)[np.newaxis, ...].astype(np.float32)
    print(f"[DEBUG] kp_np.shape={kp_np.shape}")

    try:
        preds = await asyncio.to_thread(model.predict, kp_np)
        idx = int(np.argmax(preds))
        return ACTIONS[idx], float(np.max(preds))
    except Exception as e:
        print("Predict error:", e)
        return "ERROR", 0.0


# ================== MAIN WEBSOCKET ==================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("📡 Client connected")

    frame_queue = asyncio.Queue(maxsize=FRAME_QUEUE_MAXSIZE)

    
        
    async def receiver():
        """Receive frames from frontend."""
        try:
            while True:
                data = await ws.receive_text()
                if not data:
                    continue
                try:
                    if frame_queue.full():
                        _ = frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                await frame_queue.put(data)
        except WebSocketDisconnect:
            return
        except Exception as e:
            print("Receiver error:", e)
            return

    
    # Lưu vị trí bbox cũ cho mỗi ID để làm mượt di chuyển
    miss_count = {}
    openpose_tasks = {}  # lưu task async OpenPose cho mỗi pid
    last_kps = {}        # lưu keypoints cuối cùng cho mỗi pid (fallback)
    async def processor():
        """Process latest frame: YOLO ↔ OpenPose ↔ LSTM (multi-person, parallel)."""
        try:
            while True:
                try:
                    # Lấy frame mới nhất từ queue 
                    data = await asyncio.wait_for(frame_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                try:
                    if "," in data:
                        _, encoded = data.split(",", 1)
                    else:
                        encoded = data
                    img_bytes = base64.b64decode(encoded)
                    np_img = np.frombuffer(img_bytes, np.uint8)
                    if np_img.size == 0: continue
                    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                    if frame is None: continue
                except Exception as e:
                    print("decode fail:", e)
                    continue


                t1 = time.time()

                # YOLO detection
                results = model_yolo.track(
                    source=frame,
                    persist=True,
                    stream=False,
                    verbose=False,
                    tracker="botsort.yaml"
                )

                tracked_persons = []
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and boxes.data.numel() > 0:
                        for box in boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            if cls == 0 and conf > 0.15:  # chỉ lấy người
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                track_id = int(box.id[0]) if box.id is not None else -1
                                tracked_persons.append({
                                    "id": track_id,
                                    "bbox": [x1, y1, x2, y2]
                                })
                print(f"🔍 YOLO detections: {len(tracked_persons)} người được phát hiện")



                active_ids = {f"person_{p['id']}" for p in tracked_persons}

                ids_to_remove = set()

                for old_id in list(person_history.keys()):
                    last_seen = person_history[old_id][-1].get("last_seen", 0)

                    # cập nhật miss_count
                    if old_id not in active_ids:
                        miss_count[old_id] = miss_count.get(old_id, 0) + 1
                    else:
                        miss_count[old_id] = 0

                    # chỉ xóa nếu người đó mất > 40 frame liên tiếp
                    if miss_count[old_id] > 40:
                        print(f"🧹 Xóa lịch sử và buffer cho ID: {old_id} (mất > 40 frame)")
                        ids_to_remove.add(old_id)
                        miss_count.pop(old_id, None)
                    
                # thực hiện xóa
                for old_id in ids_to_remove:
                    person_history.pop(old_id, None)
                    frame_buffer.pop(old_id, None)
                    kp_buffer.pop(old_id, None)
                    if "history_kps" in globals():
                        history_kps.pop(old_id, None)

                t2 = time.time()
                print(f"YOLO+DeepSORT+Cleanup: {(t2 - t1)*1000:.1f} ms")


                if not tracked_persons:
                    await ws.send_json({"predictions": {}, "ts": time.time()})
                    await asyncio.sleep(PROCESS_INTERVAL)
                    continue
                
                # Khởi tạo predictions tạm thời và gửi về frontend ngay (cho phản hồi nhanh)
                predictions = {}
                for d in tracked_persons:
                    pid = f"person_{d['id']}"
                    x1, y1, x2, y2 = d["bbox"]
                    pad_x = int(0.2 * (x2 - x1))
                    pad_y = int(0.2 * (y2 - y1))
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(frame.shape[1]-1, x2 + pad_x)
                    y2 = min(frame.shape[0]-1, y2 + pad_y)
                    if pid in person_history and len(person_history[pid]) > 0:
                        last = person_history[pid][-1]
                        predictions[pid] = {
                            "bbox": [x1, y1, x2, y2],
                            "action": last["action"],
                            "prob": last["prob"]
                        }
                    else:
                        predictions[pid] = {
                            "bbox": [x1, y1, x2, y2],
                            "action": "DETECTING",
                            "prob": 0.0
                        }
                await ws.send_json({"predictions": predictions, "ts": time.time()})


                total_t1 = time.time()

                # ================= Xử lý tất cả người song song (OpenPose + LSTM) =================
                async def process_person(det):
                    pid = f"person_{det['id']}"
                    x1, y1, x2, y2 = det["bbox"]

                    pad_x, pad_y = int(0.2 * (x2 - x1)), int(0.2 * (y2 - y1))
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(frame.shape[1] - 1, x2 + pad_x)
                    y2 = min(frame.shape[0] - 1, y2 + pad_y)

                    cropped = frame[y1:y2, x1:x2]
                    if cropped.size == 0:
                        return None

                    try:
                        cropped_224 = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
                    except Exception as e:
                        print("Resize failed:", e)
                        return None
                    
                    h = w = IMG_SIZE


                    # Gọi OpenPose async nếu chưa có task
                    if pid not in openpose_tasks:
                        openpose_tasks[pid] = asyncio.create_task(call_openpose_async(cropped_224))

                    # Lấy kết quả keypoints nếu task xong
                    if openpose_tasks[pid].done():
                        res = openpose_tasks[pid].result()
                        del openpose_tasks[pid]
                        if isinstance(res, list) and len(res) > 0:
                            arr = np.array(res[0], dtype=np.float32)
                            if arr.ndim == 1 and arr.size >= 75:
                                arr = arr.reshape(-1,3)#[:,:2] 
                            # last_kps[pid] = arr.copy()
                        else:
                            arr = last_kps.get(pid, None)
                    else:
                        arr = last_kps.get(pid, None)

                    if arr is None:
                        return None

                    person_kps = arr.copy()

                    # invalid = (0,0) hoặc NaN
                    invalid_mask = (
                        ((person_kps[:,0]==0) & (person_kps[:,1]==0)) |
                        np.isnan(person_kps[:,0]) |
                        np.isnan(person_kps[:,1])
                    )

                    if pid in last_kps:
                        prev_kps = last_kps[pid]

                        valid_prev = ~(
                            ((prev_kps[:,0]==0) & (prev_kps[:,1]==0)) |
                            np.isnan(prev_kps[:,0]) |
                            np.isnan(prev_kps[:,1])
                        )

                        # 1.1. Copy từ frame trước cho những điểm miss
                        fill_mask = invalid_mask & valid_prev
                        person_kps[fill_mask] = prev_kps[fill_mask]

                        # cập nhật lại invalid sau khi copy
                        invalid_mask = (
                            ((person_kps[:,0]==0) & (person_kps[:,1]==0)) |
                            np.isnan(person_kps[:,0]) |
                            np.isnan(person_kps[:,1])
                        )

                        # 1.2. Nội suy theo hàng xóm
                        JOINT_NEIGHBORS = {
                            1: [0, 2], 2: [1, 3], 3: [2, 4],
                            5: [1, 6], 6: [5, 7], 7: [6, 8],
                            9: [8, 10], 10: [9, 11],
                            12: [11, 13], 13: [12, 14],
                            15: [0, 16], 16: [15, 17],
                            18: [17, 19], 19: [18, 20],
                            21: [0, 22], 22: [21, 23],
                            24: [23, 1],
                        }

                        for j in np.where(invalid_mask)[0]:
                            if j in JOINT_NEIGHBORS:
                                valid_refs = [r for r in JOINT_NEIGHBORS[j]
                                            if not invalid_mask[r]
                                            and not np.isnan(person_kps[r,0])
                                            and not np.isnan(person_kps[r,1])]
                                if valid_refs:
                                    person_kps[j] = np.mean(person_kps[valid_refs], axis=0)
                                    invalid_mask[j] = False

                        # cập nhật lại mask lần nữa
                        invalid_mask = (
                            ((person_kps[:,0]==0) & (person_kps[:,1]==0)) |
                            np.isnan(person_kps[:,0]) |
                            np.isnan(person_kps[:,1])
                        )

                        # 1.3. Nội suy theo cụm (block interpolation)
                        JOINT_GROUPS = {
                            "head":      [0,1,15,16,17,18],
                            "right_arm": [2,3,4],
                            "left_arm":  [5,6,7],
                            "spine":     [1,8],
                            "right_leg": [9,10,11,22,23,24],  # hông → gối → cổ chân → bàn chân phải
                            "left_leg":  [12,13,14,19,20,21], # hông → gối → cổ chân → bàn chân trái
                        }

                        prev_valid_mask = ~(
                            ((prev_kps[:,0]==0) & (prev_kps[:,1]==0)) |
                            np.isnan(prev_kps[:,0]) |
                            np.isnan(prev_kps[:,1])
                            )


                        if np.any(prev_valid_mask):
                            for group, joints in JOINT_GROUPS.items():
                                group_invalid = invalid_mask[joints]
                                group_valid = ~group_invalid


                                # Nếu toàn bộ group mất thì bỏ qua (không có thông tin để nội suy)
                                if not np.any(group_valid):
                                    continue

                                # Lấp vào các khớp invalid trong cùng group
                                for j in joints:
                                    if invalid_mask[j]:
                                        neighbors = [person_kps[k] for k in joints if not invalid_mask[k]]
                                        if len(neighbors) > 0:
                                            person_kps[j] = np.mean(neighbors, axis=0)
                                            invalid_mask[j] = False

                        invalid_mask = (
                            ((person_kps[:,0]==0) & (person_kps[:,1]==0)) |
                            np.isnan(person_kps[:,0]) |
                            np.isnan(person_kps[:,1])
                        )
                        
                        # 1.4. Quán tính (prev - prevprev) cho điểm vẫn còn miss
                        fix_mask = invalid_mask & valid_prev
                        if np.any(fix_mask):
                            prevprev = last_kps.get(f"{pid}_prevprev", prev_kps)
                            vel = prev_kps - prevprev
                            vel_norm = np.linalg.norm(vel, axis=1)

                            safe = vel_norm < 0.2 * max(w, h)
                            mask = fix_mask & safe

                            person_kps[mask] = prev_kps[mask] + vel[mask]


                        # 1.5. Làm mượt toàn khung xương theo prev (EMA)
                        alpha = 0.3 if np.any(invalid_mask) else 0.5
                        person_kps = alpha * person_kps + (1 - alpha) * prev_kps

                    # 2. Fallback
                    invalid_mask = (
                        ((person_kps[:,0]==0) & (person_kps[:,1]==0)) |
                        np.isnan(person_kps[:,0]) |
                        np.isnan(person_kps[:,1])
                    )


                    if np.any(invalid_mask):
                        if pid in last_kps:
                            prev_kps = last_kps[pid]
                            valid_prev = ~(
                                (prev_kps[:,0]==0) | (prev_kps[:,1]==0) |
                                np.isnan(prev_kps[:,0]) | np.isnan(prev_kps[:,1])
                            )
                            if np.any(valid_prev):
                                copy_mask = invalid_mask & valid_prev
                                person_kps[copy_mask] = prev_kps[copy_mask]
                                invalid_mask = (
                                    ((person_kps[:,0]==0) & (person_kps[:,1]==0)) |
                                    np.isnan(person_kps[:,0]) | np.isnan(person_kps[:,1])
                            )

                        # Nếu sau khi copy từ prev mà vẫn còn invalid
                        if np.any(invalid_mask):
                            valid_mask = ~invalid_mask

                            # Nếu không còn điểm hợp lệ nào → bỏ frame
                            if not np.any(valid_mask):
                                return None

                            invalid_ratio = invalid_mask.sum() / person_kps.shape[0]

                            
                            # quá 50–60% khớp bị lỗi → bỏ luôn
                            if invalid_ratio > 0.4:
                                return None

                            # # còn lại: lỗi ít → dùng mean để lấp
                            # mean_xy = np.mean(person_kps[valid_mask, :2], axis=0)
                            # person_kps[invalid_mask, 0] = mean_xy[0]
                            # person_kps[invalid_mask, 1] = mean_xy[1]

                    # 3. Re-center skeleton sau khi đã fill tất cả invalid
                    # TÙY VÀO ĐỘ ỔN ĐỊNH CỦA KHUNG XƯƠNG
                    # invalid_ratio = invalid_mask.mean()
                    # if invalid_ratio > 0.2 and pid in last_kps:
                    #     spine = person_kps[[1, 8]]
                    #     prev_spine = last_kps[pid][[1, 8]]

                    #     if not (np.any(np.isnan(spine)) or np.any(np.isnan(prev_spine))):
                    #         center = spine.mean(axis=0)
                    #         prev_center = prev_spine.mean(axis=0)
                    #         shift = np.linalg.norm(center - prev_center)

                    #         # chỉ recenter khi shift không bất thường
                    #         if shift < 0.2 * min(w, h):  
                    #             person_kps += (prev_center - center)

                    # Lưu lịch sử keypoints (giữ tối đa 5 frame)
                    if "history_kps" not in globals():
                        global history_kps
                        history_kps = {}

                    if pid not in history_kps:
                        history_kps[pid] = []

                    # Chỉ lưu keypoints hợp lệ (không NaN)
                    if not np.any(np.isnan(person_kps)):
                        history_kps[pid].append(person_kps.copy())
                        if len(history_kps[pid]) > 5:  # chỉ giữ 5 frame gần nhất
                            history_kps[pid].pop(0)    

                    if pid in last_kps:
                        last_kps[f"{pid}_prevprev"] = last_kps[pid].copy()
                    last_kps[pid] = person_kps.copy()

                    if np.isnan(person_kps).any():
                        # đưa các NaN còn sót về 0.0 (coi như điểm mất)
                        person_kps = np.nan_to_num(person_kps, nan=0.0)

                    # ==== DEBUG ====
                    if DEBUG:
                        debug_img = cropped_224.copy()
                        for (x, y, c) in person_kps.astype(int):
                            cv2.circle(debug_img, (x, y), 3, (0, 255, 0), -1)
                        cv2.imshow(f"debug_{pid}", debug_img)
                        cv2.waitKey(1)

                    # Normalize keypoints cho model (trong phạm vi bbox)
                    kp_norm = person_kps.copy()
                    kp_norm[:, 0] = np.clip(kp_norm[:, 0] / w, 0.0, 1.0)
                    kp_norm[:, 1] = np.clip(kp_norm[:, 1] / h, 0.0, 1.0)


                    kp_norm_xy = kp_norm[:, :2].copy().flatten()


                    if person_kps.shape[1] == 2:
                        conf_col = np.ones((person_kps.shape[0], 1), dtype=np.float32)
                        person_kps = np.concatenate([person_kps, conf_col], axis=1)

                    # Flatten keypoints và lưu buffer
                    keypoints_flat = kp_norm.flatten()

                    last_kps[pid] = person_kps.copy()

                    # nếu thiếu keypoints
                    if keypoints_flat.size < NUM_FEATURES:
                        keypoints_flat = np.pad(keypoints_flat, (0, NUM_FEATURES - keypoints_flat.size))
                    elif keypoints_flat.size > NUM_FEATURES:
                        keypoints_flat = keypoints_flat[:NUM_FEATURES]
                        
                    frame_buffer.setdefault(pid, []).append(cropped_224)

                    # kiểm tra shape cho chắc
                    if cropped_224.shape != (IMG_SIZE, IMG_SIZE, 3):
                        print("Skipping frame due to wrong shape:", cropped_224.shape)
                        return None


                    kp_buffer.setdefault(pid, []).append(keypoints_flat)
                    if len(frame_buffer[pid]) > NUM_FRAMES:
                        frame_buffer[pid].pop(0)
                        kp_buffer[pid].pop(0)


                    #####DEBUG#####
                    # print(f"len(keypoints_flat)={len(keypoints_flat)}, has_nan={np.isnan(keypoints_flat).any()}")
                    # print(f"bbox=({x1},{y1},{x2},{y2}), crop size={cropped.shape}, kp_minmax=({person_kps[:,0].min():.1f},{person_kps[:,0].max():.1f})")
                    # print(f"[DEBUG] PID={pid}: crop_shape={cropped.shape}, kps_minmax=({arr[:,0].min():.1f},{arr[:,0].max():.1f}), ({arr[:,1].min():.1f},{arr[:,1].max():.1f})")
                    
                        
                    # ===== Dự đoán với LSTM =====
                    if len(frame_buffer[pid]) >= NUM_FRAMES:
                        # lấy đúng 12 frame
                        frames_list = frame_buffer[pid][-NUM_FRAMES:]
                        kps_list    = kp_buffer[pid][-NUM_FRAMES:]

                        # lọc lại cho chắc (đề phòng frame lỗi shape / kp thiếu)
                        frames_list = [f for f in frames_list if f.shape == (IMG_SIZE, IMG_SIZE, 3)]
                        kps_list    = [k for k in kps_list if k.size == NUM_FEATURES]

                        # nếu còn < NUM_FRAMES thì bỏ
                        if len(frames_list) < NUM_FRAMES or len(kps_list) < NUM_FRAMES:
                            pass
                        else:
                            action, prob = await do_model_predict(frames_list, kps_list)
                            if action != "ERROR" and prob > 0.65:
                                entry = {
                                    "time": time.time(),
                                    "bbox": [x1, y1, x2, y2],
                                    "action": action,
                                    "prob": round(prob, 3),
                                    "keypoints": kp_norm_xy.tolist(),
                                    "last_seen": time.time()
                                }
                                person_history.setdefault(pid, []).append(entry)
                                if len(person_history[pid]) > 200:
                                    person_history[pid] = person_history[pid][-200:]
                                return {
                                    "pid": pid,
                                    "bbox": [x1, y1, x2, y2],
                                    "action": action,
                                    "prob": round(prob, 3),
                                    "time": time.strftime("%H:%M:%S", time.localtime()),
                                    "keypoints": kp_norm_xy.tolist()
                                }

                    # Nếu chưa đủ frame hoặc chưa predict mới, dùng action cuối cùng nếu có
                    if pid in person_history and len(person_history[pid]) > 0:
                        last = person_history[pid][-1]
                        return {
                            "pid": pid,
                            "bbox": [x1, y1, x2, y2],
                            "action": last["action"],
                            "prob": last["prob"],
                            "time": time.strftime("%H:%M:%S", time.localtime()),
                            "keypoints": kp_norm_xy.tolist(),
                        }

                    return None


                # chạy tất cả người song song 
                person_tasks = [asyncio.create_task(process_person(det)) for det in tracked_persons]
                results = await asyncio.gather(*person_tasks)

                predictions = {r["pid"]: {"bbox": r["bbox"], "action": r["action"], "prob": r["prob"], "keypoints": r.get("keypoints", None)} for r in results if r}
                
                total_t2 = time.time()
                print(f"Multi-person total time: {(total_t2 - total_t1)*1000:.1f} ms")

                # Gửi kết quả cuối cùng (cập nhật nếu có dự đoán mới từ LSTM)
                try:
                    if predictions:
                        await ws.send_json({"predictions": predictions, "ts": time.time()})
 
                except Exception as e:
                    print("send fail:", e)
                    return

                await asyncio.sleep(PROCESS_INTERVAL)

        except WebSocketDisconnect:
            return
        except asyncio.CancelledError:
            return
        except Exception as e:
            print("Processor unexpected error:", e)
            return


    recv_task = asyncio.create_task(receiver())
    proc_task = asyncio.create_task(processor())

    done, pending = await asyncio.wait([recv_task, proc_task], return_when=asyncio.FIRST_COMPLETED)
    for p in pending:
        p.cancel()

    print("Client disconnected")
    try:
        await ws.close()
    except Exception:
        pass
    
# ================== ALERT FALL API ==================
@app.post("/alert-fall")
async def alert_fall(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    message: str = Form("Phát hiện hành động nguy hiểm từ hệ thống!")
):
    """
    Nhận ảnh + message từ frontend và gửi sang Telegram ở background,
    trả response cho frontend NGAY LẬP TỨC, không block event loop.
    """
    try:
        file_bytes = await image.read()

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

        files = {
            "photo": ("fall_frame.jpg", file_bytes, image.content_type)
        }
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": message
        }

        def send_telegram():
            try:
                resp = requests.post(url, data=data, files=files, timeout=10)
                if resp.status_code != 200:
                    print("Telegram response:", resp.text)
                else:
                    print("Sent alert to Telegram")
            except Exception as e:
                print("Telegram send error:", e)

        # chạy gửi Telegram ở background, không chặn request chính
        background_tasks.add_task(send_telegram)

        # Trả về ngay cho frontend
        return {"ok": True}
    except Exception as e:
        print("alert_fall error:", e)
        return {"ok": False, "error": str(e)}

    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)



