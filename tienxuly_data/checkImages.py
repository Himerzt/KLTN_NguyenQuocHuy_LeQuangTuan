import os

# Danh sách các thư mục gốc chứa ảnh
base_dirs = [
    r"D:\xuly\data_case2_3_224\CLIMB\images",
    r"D:\xuly\data_case2_3_224\FALL\images",
    r"D:\xuly\data_case2_3_224\SIT\images",
    r"D:\xuly\data_case2_3_224\STAND\images",
    r"D:\xuly\data_case2_3_224\WALK\images"
]

# Các phần mở rộng ảnh thường gặp
image_exts = (".jpg", ".jpeg", ".png", ".bmp")

def count_images(base_dir):
    total_images = 0
    folders_checked = 0

    for root, dirs, files in os.walk(base_dir):
        if files:
            folders_checked += 1
            for file in files:
                if file.lower().endswith(image_exts):
                    total_images += 1

    return {
        "folder": base_dir,
        "total_images": total_images,
        "folders_checked": folders_checked
    }

# ---- Chạy cho từng thư mục ----
grand_total = 0
for folder in base_dirs:
    stats = count_images(folder)
    print(f"\nDataset: {stats['folder']}")
    print(f"Tổng số ảnh: {stats['total_images']}")
    print(f"Số folder con: {stats['folders_checked']}")
    grand_total += stats['total_images']

print(f"\n>>> Tổng số ảnh trong toàn bộ dataset: {grand_total}")
