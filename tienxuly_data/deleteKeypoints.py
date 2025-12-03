import glob
import os
import json

# Thư mục gốc chứa các action (WALK, CLIMB, FALL, ...)
root_dir = r"D:\xuly\data_case_3_224"

deleted_json_only = 0
deleted_pairs = 0

for action in os.listdir(root_dir):
    action_path = os.path.join(root_dir, action)
    if not os.path.isdir(action_path):
        continue

    img_root = os.path.join(action_path, "images_resized")
    kp_root  = os.path.join(action_path, "keypoints")

    if not os.path.exists(img_root) or not os.path.exists(kp_root):
        continue

    for subfolder in os.listdir(kp_root):
        kp_sub = os.path.join(kp_root, subfolder)
        img_sub = os.path.join(img_root, subfolder)

        if not os.path.exists(kp_sub) or not os.path.exists(img_sub):
            continue

        for json_file in os.listdir(kp_sub):
            if not json_file.endswith(".json"):
                continue

            json_path = os.path.join(kp_sub, json_file)
            base = os.path.splitext(json_file)[0]
            img_jpg = os.path.join(img_sub, base + ".jpg")
            img_png = os.path.join(img_sub, base + ".png")

            # Trường hợp 1: JSON rỗng (không có người)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "people" in data and len(data["people"]) == 0:
                    os.remove(json_path)
                    if os.path.exists(img_jpg):
                        os.remove(img_jpg)
                    elif os.path.exists(img_png):
                        os.remove(img_png)
                    deleted_pairs += 1
                    print(f"Đã xóa cặp rỗng: {json_path}")
                    continue
            except Exception as e:
                print(f"Lỗi khi đọc {json_path}: {e}")
                continue

            

print(f"Tổng cộng đã xóa {deleted_pairs} cặp (ảnh + JSON rỗng).")
print(f"Tổng cộng đã xóa {deleted_json_only} JSON không có ảnh tương ứng.")
