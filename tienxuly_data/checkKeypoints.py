import os
import json
import numpy as np

# Danh sách các thư mục gốc cần kiểm tra
base_dirs = [
    r"D:\xuly\data_case_3_224\CLIMB\keypoints",
    r"D:\xuly\data_case_3_224\FALL\keypoints",
    r"D:\xuly\data_case_3_224\SIT\keypoints",
    r"D:\xuly\data_case_3_224\STAND\keypoints",
    r"D:\xuly\data_case_3_224\WALK\keypoints"
]

def analyze_folder(base_dir):
    total_files = 0
    total_persons = 0
    total_detected = 0
    total_undetected = 0
    total_keypoints = 0

    folders_checked = 0
    folders_with_people = 0
    folders_without_people = 0

    for root, dirs, files in os.walk(base_dir):
        if files:
            folders_checked += 1
            has_people = False

            for file in files:
                if file.endswith(".json"):
                    total_files += 1
                    path = os.path.join(root, file)

                    with open(path, "r") as f:
                        data = json.load(f)

                    people = data.get("people", [])
                    if not people:
                        continue

                    has_people = True

                    for person in people:
                        keypoints = person.get("pose_keypoints_2d", [])
                        keypoints = np.array(keypoints).reshape((-1, 3))  # (25,3)

                        detected = np.sum(keypoints[:, 2] > 0)
                        undetected = keypoints.shape[0] - detected

                        total_persons += 1
                        total_detected += detected
                        total_undetected += undetected
                        total_keypoints += keypoints.shape[0]

            if has_people:
                folders_with_people += 1
            else:
                folders_without_people += 1

    return {
        "folder": base_dir,
        "files": total_files,
        "persons": total_persons,
        "detected": total_detected,
        "undetected": total_undetected,
        "folders_checked": folders_checked,
        "folders_with_people": folders_with_people,
        "folders_without_people": folders_without_people,
        "avg_detected": (total_detected / total_persons) if total_persons > 0 else 0,
        "avg_total": (total_keypoints / total_persons) if total_persons > 0 else 0
    }

# ---- Chạy cho từng thư mục ----
for folder in base_dirs:
    stats = analyze_folder(folder)
    print(f"\nDataset: {stats['folder']}")
    print(f"Tổng số file JSON: {stats['files']}")
    print(f"Tổng số người detect được: {stats['persons']}")
    print(f"Tổng keypoints detect được: {stats['detected']}")
    print(f"Tổng keypoints không detect được: {stats['undetected']}")
    if stats['persons'] > 0:
        print(f"Trung bình mỗi người detect được: {stats['avg_detected']:.2f} / {stats['avg_total']:.0f} keypoints")

    print(f"Số folder con: {stats['folders_checked']}")
    print(f"Số folder có người detect được: {stats['folders_with_people']}")
    print(f"Số folder không detect được ai: {stats['folders_without_people']}")
