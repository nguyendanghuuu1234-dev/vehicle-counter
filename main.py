

from ultralytics import YOLO
import cv2
import os

# ══════════════════════════════════════════════════════════
# CẤU HÌNH - Chỉnh sửa tại đây
# ══════════════════════════════════════════════════════════
VIDEO_INPUT     = "plate_test.mp4"
CONF_THRESH     = 0.4
VEHICLE_CLASSES = [2, 3, 5, 7]  # 2=car, 3=motorcycle, 5=bus, 7=truck

CLASS_NAMES = {2: "Car", 3: "Motorbike", 5: "Bus", 7: "Truck"}

CLASS_COLORS = {
    2: (0, 255, 0),     # Xanh lá  - Car
    3: (255, 165, 0),  # Cam      - Motorbike
    5: (0, 0, 255),    # Đỏ       - Bus
    7: (255, 0, 255),  # Tím      - Truck
}


# ══════════════════════════════════════════════════════════
# BƯỚC 1: Khởi tạo
# ══════════════════════════════════════════════════════════
print("=" * 50)
print("  Khởi động chương trình đếm xe...")
print("=" * 50)

print("\n[1/3] Đang load model YOLO...")
model = YOLO("yolov8n.pt")
print("      ✓ Model sẵn sàng!")

# ══════════════════════════════════════════════════════════
# BƯỚC 2: Mở video
# ══════════════════════════════════════════════════════════
print(f"\n[2/3] Đang mở video: {VIDEO_INPUT}")
cap = cv2.VideoCapture(VIDEO_INPUT)

if not cap.isOpened():
    print(f"  ✗ Lỗi: Không mở được video '{VIDEO_INPUT}'")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS)
width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"      ✓ Video: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")

# ══════════════════════════════════════════════════════════
# BƯỚC 3: Xử lý từng frame
# ══════════════════════════════════════════════════════════
print(f"\n[3/3] Đang xử lý video...")

seen_ids    = set()
frame_count = 0

LINE_Y = 250
# crossline_id = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 30 == 0:
        pct = frame_count / total_frames * 100
        print(f"      Frame {frame_count}/{total_frames} ({pct:.0f}%) | Đã đếm: {len(seen_ids)} xe")

    results = model.track(
        frame,
        persist = True,
        conf    = CONF_THRESH,
        classes = VEHICLE_CLASSES,
        verbose = False,
    )

    result = results[0]

    if result.boxes is not None and result.boxes.id is not None:
        boxes       = result.boxes.xyxy.cpu().numpy()
        track_ids   = result.boxes.id.cpu().numpy().astype(int)
        class_ids   = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

    
        # vòng lặp qua tất cả các đối tượng được phát hiện và theo dõi trong frame
        for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confidences): 
            x1, y1, x2, y2 = map(int, box)


            seen_ids.add(track_id)
            # cy = (y1 + y2) // 2
            # if abs(cy - LINE_Y) < 15 and track_id not in crossline_id:
            #     crossline_id.add(track_id)
            
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            label = CLASS_NAMES.get(class_id, "Vehicle")
            text  = f"#{track_id} {label} {conf:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    
            cv2.putText(frame, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            
            
    # Hiển thị tổng xe (chữ, không có nền)
    cv2.putText(frame, f"TONG XE: {len(seen_ids)}", (14, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.putText(frame, f"ROI: ({ROI_X1}, {ROI_Y1})", (14, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    cv2.imshow("Vehicle Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ══════════════════════════════════════════════════════════
# BƯỚC 4: Kết quả
# ══════════════════════════════════════════════════════════
cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 50)
print("  ✓ Xử lý hoàn tất!")
print("=" * 50)
print(f"\n  📊 KẾT QUẢ CUỐI:")
print(f"│  Tổng xe đếm được : {len(seen_ids):>5}│")
print("=" * 50)