import cv2
import os
import shutil
from ultralytics import YOLO

# Load model YOLO
model = YOLO("yolov8m.pt")
 
# Folder gốc chứa nhiều chuỗi hành động
input_root = r"D:\test\LIEDOWN\images"
output_root = r"D:\test\LIEDOWN\images_resized"
os.makedirs(output_root, exist_ok=True)
resize_size = (224, 224)

def add_padding_and_crop(image, box, padding_ratio=0.2):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)

    x1_new = max(0, x1 - pad_x)
    y1_new = max(0, y1 - pad_y)
    x2_new = min(w, x2 + pad_x)
    y2_new = min(h, y2 + pad_y)

    cropped = image[y1_new:y2_new, x1_new:x2_new]
    return cropped


for folder in os.listdir(input_root):
    input_dir = os.path.join(input_root, folder)
    if not os.path.isdir(input_dir):
        continue

    output_dir = os.path.join(output_root, folder)
    os.makedirs(output_dir, exist_ok=True)

    counter = 1
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(input_dir, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # Detect YOLO
            results = model(img)[0]

            # Lọc tất cả box là người
            person_boxes = [
                (box.xyxy[0], box.conf[0].item())
                for box in results.boxes if int(box.cls[0].item()) == 0
            ]

            if len(person_boxes) == 0:
                continue

            # 🔹 Chọn box người có confidence cao nhất
            best_box, best_conf = max(person_boxes, key=lambda x: x[1])

            if best_conf < 0.5:  # confidence thấp thì bỏ
                continue

            cropped = add_padding_and_crop(img, best_box)
            if cropped.size == 0:
                continue

            cropped_resized = cv2.resize(cropped, resize_size)
            save_path = os.path.join(output_dir, f"{folder}_{counter:05d}.jpg")
            cv2.imwrite(save_path, cropped_resized)
            counter += 1


    # Sau khi xử lý xong folder, kiểm tra số lượng ảnh
    saved_images = len([f for f in os.listdir(output_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    if saved_images < 10:
        shutil.rmtree(output_dir)
        print(f"Xóa {output_dir} vì chỉ có {saved_images} ảnh")
    else:
        print(f"Giữ {output_dir} với {saved_images} ảnh")
