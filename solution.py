import os
import sys
import time
import numpy as np
import pandas as pd
import cv2
from ultralytics import RTDETR
import torch
from typing import List, Union
from tqdm import tqdm

model = RTDETR("best.pt")
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def predict(images: Union[List[np.ndarray], np.ndarray]) -> List[List[dict]]:
    if isinstance(images, np.ndarray):
        images = [images]

    all_predictions = []

    for img in images:
        h_img, w_img = img.shape[:2]

        results = model.predict(
            source=img,
            imgsz=928,
            verbose=False,
            device=DEVICE
        )[0]

        image_preds = []

        if hasattr(results, 'boxes') and results.boxes is not None:
            for box in results.boxes:
                xc, yc, w, h = box.xywhn[0].tolist()
                conf = float(box.conf[0])
                label = int(box.cls[0])

                if label == 0:
                    image_preds.append({
                        'xc': xc,
                        'yc': yc,
                        'w': w,
                        'h': h,
                        'label': label,
                        'score': conf,
                        'w_img': w_img,
                        'h_img': h_img
                    })

        if not image_preds:
            image_preds.append({
                'xc': np.nan,
                'yc': np.nan,
                'w': np.nan,
                'h': np.nan,
                'label': 0,
                'score': np.nan,
                'w_img': w_img,
                'h_img': h_img
            })

        all_predictions.append(image_preds)

    return all_predictions


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python solution.py <input_image_directory> <output_csv_path>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_csv_path = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        sys.exit(1)

    results_data = []
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_name in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_dir, image_name)
        image_id = os.path.splitext(image_name)[0]

        image_np = cv2.imread(image_path)
        if image_np is None:
            continue

        start_time = time.time()
        predictions_for_image = predict(image_np)[0]
        elapsed_time = time.time() - start_time

        for res in predictions_for_image:
            results_data.append({
                'image_id': image_id,
                'xc': res['xc'],
                'yc': res['yc'],
                'w': res['w'],
                'h': res['h'],
                'label': res['label'],
                'score': res['score'],
                'time_spent': round(elapsed_time, 4),
                'w_img': res['w_img'],
                'h_img': res['h_img']
            })

    COLUMNS = ['image_id', 'xc', 'yc', 'w', 'h', 'label', 'score', 'time_spent', 'w_img', 'h_img']
    results_df = pd.DataFrame(results_data, columns=COLUMNS)
    results_df.to_csv(output_csv_path, index=False, na_rep='NaN')

    print(f"Results saved to {output_csv_path}")


