import cv2
from pathlib import Path

count = 0  
step = 5
c = 1
for i in range(9, 10):
    video_path = Path(f"D:/7277372611424.mp4")    

    output_dir = video_path.parent / f"{c}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
    else:
        print(f"Đang trích xuất frame từ video: {video_path.name}")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Chỉ lưu khi frame_idx chia hết cho step
            if frame_idx % step == 0:
                # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                frame_path = output_dir / f"frame_{count:008}.jpg"
                cv2.imwrite(str(frame_path), frame)
                count += 1

            frame_idx += 1

        cap.release()
        print(f"Đã trích xuất {count} frame → {output_dir}")
        c+=1
