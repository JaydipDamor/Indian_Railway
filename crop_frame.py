import cv2
import os

# === CONFIGURATION ===
video_path = r"D:\Users\DELL\Downloads\Indain_railway_Part-2\4_video.mp4"             # Path to your CCTV video
output_folder = 'cropped_frames'           # Folder to save cropped images
fps_to_extract = 2                         # Frames per second to extract
roi = (120, 360, 200, 150)  # (x, y, width, height) - Adjust this rectangle as needed

# === PREPARE OUTPUT FOLDER ===
os.makedirs(output_folder, exist_ok=True)

# === OPEN VIDEO ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(video_fps // fps_to_extract)

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        x, y, w, h = roi
        cropped = frame[y:y+h, x:x+w]
        filename = f"frame_{saved_count:04d}.jpg"
        cv2.imwrite(os.path.join(output_folder, filename), cropped)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Done. Saved {saved_count} cropped frames to '{output_folder}' folder.")
