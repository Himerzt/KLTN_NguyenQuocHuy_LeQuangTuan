import os
import shutil
import re

source_dir = r"D:\2"                 # thư mục chứa ảnh gốc
out_base = r"D:\11-26\LIEDOWN\images"    # nơi lưu các folder "sit800", "sit801", ...
batch_size = 30                      # tối đa 30 ảnh / folder
min_len = 24                         # tối thiểu 24 ảnh / folder
start_index = 812                    # sit800 trở đi

# Thư mục lưu các sequence ngắn (<24 ảnh) nếu bạn muốn giữ lại
short_base = os.path.join(out_base, "_short")
os.makedirs(out_base, exist_ok=True)
os.makedirs(short_base, exist_ok=True)

def extract_index(filename: str):
    """
    Lấy số frame từ tên file.
    Ví dụ: frame_00001181.jpg -> 1181
    """
    m = re.search(r"(\d+)", filename)
    if m:
        return int(m.group(1))
    return None

# Lấy danh sách ảnh và sort theo số frame
files = [
    f for f in os.listdir(source_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# Bỏ những file không parse được số
files_with_idx = []
for f in files:
    idx = extract_index(f)
    if idx is not None:
        files_with_idx.append((f, idx))

files_with_idx.sort(key=lambda x: x[1])  # sort theo index frame tăng dần

folder_idx = start_index
short_idx = 0

current_group = []      # list tên file trong group hiện tại
prev_idx = None         # frame index trước đó

def save_group(group_files, folder_idx):
    """Lưu 1 group hợp lệ (>=24 & <=30 ảnh) vào sit{folder_idx}."""
    folder_name = f"liedown{folder_idx}"
    folder_path = os.path.join(out_base, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    for fname in group_files:
        src_path = os.path.join(source_dir, fname)
        dst_path = os.path.join(folder_path, fname)
        shutil.copy(src_path, dst_path)

    print(f"✅ Lưu {len(group_files)} ảnh vào {folder_name}")

def save_short_group(group_files, short_idx):
    """Lưu group không đủ min_len vào thư mục _short cho đỡ mất ảnh."""
    if not group_files:
        return
    folder_name = f"short_{short_idx}_len{len(group_files)}"
    folder_path = os.path.join(short_base, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    for fname in group_files:
        src_path = os.path.join(source_dir, fname)
        dst_path = os.path.join(folder_path, fname)
        shutil.copy(src_path, dst_path)

    print(f"⚠️ Group chỉ có {len(group_files)} ảnh -> lưu tạm vào {folder_name}")

for fname, idx in files_with_idx:
    if prev_idx is None:
        # bắt đầu group mới
        current_group = [fname]
        prev_idx = idx
        continue

    # Kiểm tra có còn liên tiếp và chưa vượt quá 30 ảnh không
    if idx == prev_idx + 1 and len(current_group) < batch_size:
        current_group.append(fname)
        prev_idx = idx

        # Nếu đã đủ 30 ảnh -> chốt group
        if len(current_group) == batch_size:
            save_group(current_group, folder_idx)
            folder_idx += 1
            current_group = []
            prev_idx = None  # reset, frame sau sẽ mở group mới
    else:
        # Bị đứt khúc hoặc đã full 30 rồi
        if len(current_group) >= min_len:
            # đủ điều kiện 24–30 -> lưu group chính
            save_group(current_group, folder_idx)
            folder_idx += 1
        else:
            # không đủ 24 -> cho qua thư mục short
            save_short_group(current_group, short_idx)
            short_idx += 1

        # bắt đầu group mới với frame hiện tại
        current_group = [fname]
        prev_idx = idx

# Flush group cuối cùng
if current_group:
    if len(current_group) >= min_len:
        save_group(current_group, folder_idx)
    else:
        save_short_group(current_group, short_idx)

print("Done! 🎉")
