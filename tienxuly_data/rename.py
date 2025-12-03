import os

# Thư mục gốc chứa các folder sequence
root_dir = "C:/Users/ADMIN/Downloads/11 - 3/FALL/images"

# Lấy danh sách folder, sort để giữ thứ tự
folders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])

# Rename folder và ảnh bên trong
for idx, folder in enumerate(folders, start=500):
    old_folder_path = os.path.join(root_dir, folder)
    new_folder_name = f"fall{idx}"
    new_folder_path = os.path.join(root_dir, new_folder_name)

    # Đổi tên folder
    os.rename(old_folder_path, new_folder_path)
    print(f"Đổi {folder} → {new_folder_name}")

    # Rename ảnh trong folder đó
    images = sorted([f for f in os.listdir(new_folder_path) if f.lower().endswith((".jpg", ".png"))])
    for img_idx, img_name in enumerate(images, start=1):
        old_img_path = os.path.join(new_folder_path, img_name)
        new_img_name = f"{new_folder_name}_{img_idx:05d}.jpg"
        new_img_path = os.path.join(new_folder_path, new_img_name)

        os.rename(old_img_path, new_img_path)

    print(f"  → Đã rename {len(images)} ảnh trong {new_folder_name}")
