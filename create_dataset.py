import os
import pickle
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# Load the YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')

DATA_DIR = './data'
data = []
labels = []
skipped = 0

# Iterate over each class folder
for dir_ in tqdm(os.listdir(DATA_DIR), desc="Processing Classes"):
    dir_path = os.path.join(DATA_DIR, dir_)
    for img_file in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_file)
        data_aux = []

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not read image: {img_path}")
            skipped += 1
            continue

        results = model(img, verbose=False)

        # Extract keypoints if available
        keypoints_tensor = results[0].keypoints.xy if results and results[0].keypoints is not None else None

        if keypoints_tensor is not None and len(keypoints_tensor) > 0:
            keypoints = keypoints_tensor[0].cpu().numpy()

            x_ = keypoints[:, 0].tolist()
            y_ = keypoints[:, 1].tolist()

            for x, y in zip(x_, y_):
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)
        else:
            print(f"⚠️ No keypoints found in image: {img_path}")
            skipped += 1

# Save the processed data
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"\n✅ Dataset creation complete.")
print(f"📦 Total samples saved: {len(data)}")
print(f"⚠️ Skipped {skipped} image(s) due to missing keypoints.")
