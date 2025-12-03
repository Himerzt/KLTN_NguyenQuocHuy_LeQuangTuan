import os
import shutil

# Đường dẫn gốc
images_dir = r"D:\xuly\data_case_3_224\LIEDOWN\images_resized"
keypoints_dir = r"D:\xuly\data_case_3_224\LIEDOWN\keypoints"

# Lấy danh sách thư mục con ở mỗi bên
images_subfolders = {f for f in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, f))}
keypoints_subfolders = {f for f in os.listdir(keypoints_dir) if os.path.isdir(os.path.join(keypoints_dir, f))}

# Tìm các folder chỉ có 1 bên
only_in_images = images_subfolders - keypoints_subfolders
only_in_keypoints = keypoints_subfolders - images_subfolders

print(f"Có {len(only_in_images)} thư mục chỉ có bên images_resized:", only_in_images)
print(f"Có {len(only_in_keypoints)} thư mục chỉ có bên keypoints:", only_in_keypoints)

# Xác nhận trước khi xóa
confirm = input("Bạn có chắc muốn xóa các thư mục KHÔNG có cặp tương ứng không? (y/n): ")
if confirm.lower() == 'y':
    for folder in only_in_images:
        path = os.path.join(images_dir, folder)
        shutil.rmtree(path, ignore_errors=True)
        print(f"Đã xóa bên images_resized: {folder}")

    for folder in only_in_keypoints:
        path = os.path.join(keypoints_dir, folder)
        shutil.rmtree(path, ignore_errors=True)
        print(f"Đã xóa bên keypoints: {folder}")

    print("✅ Đã giữ lại những folder có mặt ở cả 2 bên.")
else:
    print("❌ Hủy thao tác.")
