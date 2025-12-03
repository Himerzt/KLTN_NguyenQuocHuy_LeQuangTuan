import os
import shutil  # để xóa cả folder

base_dir = r"D:\xuly\data_case_3_224"  # thư mục gốc chứa WALK, CLIMB, FALL, ...
min_count = 5  # ngưỡng tối thiểu

summary = {}
deleted_seqs = []

for action in os.listdir(base_dir):
    action_path = os.path.join(base_dir, action)
    if not os.path.isdir(action_path):
        continue

    img_dir = os.path.join(action_path, "images_resized")
    kp_dir  = os.path.join(action_path, "keypoints")

    action_stats = {}
    total_imgs_action = 0
    total_kps_action  = 0
    kept_seq_count = 0
    deleted_seq_count = 0

    if os.path.exists(img_dir) and os.path.exists(kp_dir):
        for seq in os.listdir(img_dir):
            seq_img_path = os.path.join(img_dir, seq)
            seq_kp_path  = os.path.join(kp_dir, seq)

            if not os.path.isdir(seq_img_path):
                continue

            num_imgs = len([f for f in os.listdir(seq_img_path) if f.lower().endswith((".png", ".jpg"))])
            num_kps  = len([f for f in os.listdir(seq_kp_path)]) if os.path.exists(seq_kp_path) else 0

            # kiểm tra điều kiện xóa
            if num_imgs < min_count or num_kps < min_count:
                print(f"Xóa {action}/{seq} ({num_imgs} ảnh, {num_kps} keypoints)")
                if os.path.exists(seq_img_path):
                    shutil.rmtree(seq_img_path)
                if os.path.exists(seq_kp_path):
                    shutil.rmtree(seq_kp_path)
                deleted_seqs.append((action, seq, num_imgs, num_kps))
                deleted_seq_count += 1
                continue

            # giữ lại
            action_stats[seq] = {"images": num_imgs, "keypoints": num_kps}
            total_imgs_action += num_imgs
            total_kps_action  += num_kps
            kept_seq_count += 1

    summary[action] = {
        "sequences": action_stats,
        "total_images": total_imgs_action,
        "total_keypoints": total_kps_action,
        "kept_sequences": kept_seq_count,
        "deleted_sequences": deleted_seq_count
    }

# In kết quả
grand_total_imgs = 0
grand_total_kps  = 0
grand_total_kept = 0
grand_total_deleted = 0

for action, stats in summary.items():
    print(f"\n=== {action} ===")
    print(f"Số sequence giữ lại: {stats['kept_sequences']}")
    print(f"Số sequence bị xóa : {stats['deleted_sequences']}")
    print(f"Tổng {action}: {stats['total_images']} images, {stats['total_keypoints']} keypoints")

    grand_total_imgs += stats["total_images"]
    grand_total_kps  += stats["total_keypoints"]
    grand_total_kept += stats["kept_sequences"]
    grand_total_deleted += stats["deleted_sequences"]

print("\n=== Tổng kết toàn bộ dataset sau khi lọc ===")
print(f"Tổng số sequence giữ lại: {grand_total_kept}")
print(f"Tổng số sequence bị xóa : {grand_total_deleted}")
print(f"Tổng số ảnh: {grand_total_imgs}")
print(f"Tổng số keypoints: {grand_total_kps}")

print("\n=== Các sequence đã bị xóa ===")
for action, seq, imgs, kps in deleted_seqs:
    print(f"{action}/{seq}: {imgs} ảnh, {kps} keypoints")
