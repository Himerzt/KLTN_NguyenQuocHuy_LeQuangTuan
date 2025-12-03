import os

root_img = r"D:\xuly\data_case_3_224\LIEDOWN\images_resized"
root_kp  = r"D:\xuly\data_case_3_224\LIEDOWN\keypoints"

for folder in os.listdir(root_img):
    img_folder = os.path.join(root_img, folder)
    kp_folder  = os.path.join(root_kp, folder)

    if not os.path.isdir(img_folder) or not os.path.isdir(kp_folder):
        continue

    num_imgs = len([f for f in os.listdir(img_folder) if f.endswith(".jpg")])
    num_kps  = len([f for f in os.listdir(kp_folder) if f.endswith(".json")])

    if num_imgs != num_kps:
        print(f"Lỗi: {folder} → {num_imgs} ảnh, {num_kps} keypoints")
