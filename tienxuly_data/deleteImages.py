import os

# Thư mục gốc chứa các action (WALK, CLIMB, FALL, ...)
root_dir = r"D:\xuly\data_case_3_224"

deleted = 0

for action in os.listdir(root_dir):
    action_path = os.path.join(root_dir, action)
    if not os.path.isdir(action_path):
        continue

    img_root = os.path.join(action_path, "images_resized")
    kp_root  = os.path.join(action_path, "keypoints")

    if not os.path.exists(img_root) or not os.path.exists(kp_root):
        continue

    # duyệt các folder nhỏ bên trong
    for subfolder in os.listdir(img_root):
        img_sub = os.path.join(img_root, subfolder)
        kp_sub  = os.path.join(kp_root, subfolder)

        if not os.path.exists(img_sub) or not os.path.exists(kp_sub):
            continue

        # lấy danh sách ảnh
        img_files = {os.path.splitext(f)[0] for f in os.listdir(img_sub) 
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))}

        # lấy danh sách keypoints (bỏ "_keypoints")
        kp_files = {os.path.splitext(f)[0].replace("_keypoints", "") 
                    for f in os.listdir(kp_sub) if f.endswith(".json")}

        # ảnh không có keypoint
        to_delete = img_files - kp_files

        if to_delete:
            print(f"\n {action}/{subfolder}: Xóa {len(to_delete)} ảnh không có keypoint")
        
        for img_name in to_delete:
            for ext in [".jpg", ".png", ".jpeg"]:
                img_path = os.path.join(img_sub, img_name + ext)
                if os.path.exists(img_path):
                    os.remove(img_path)
                    deleted += 1
                    print("  Đã xóa:", img_path)

print(f"\n Tổng cộng đã xóa {deleted} ảnh không có file keypoint tương ứng.")
