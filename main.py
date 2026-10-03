import os
import cv2
import numpy as np
from functions import (
    find_polygon_center, save_object, load_object,
    is_point_in_polygon, get_label_name,
    VEHICLE_CLASSES, OCCUPANCY_THRESHOLD, DEFAULT_FPS,
    FRAME_WIDTH, FRAME_HEIGHT
)
from ultralytics import YOLO
from shapely.geometry import Polygon as ShapelyPolygon
import logging

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "Models/yolov8m mAp 48/weights/best.pt")
VIDEO_PATH = os.environ.get("VIDEO_PATH", "Media/parking_crop_loop.mp4")
DEVICE = os.environ.get("YOLO_DEVICE", "cpu")

# Load model
try:
    model = YOLO(MODEL_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Load saved polygon data
polygon_data = load_object()
points = []  # Points for the polygon currently being drawn


def draw_polygon(event, x, y, flags, param):
    """Mouse callback: left-click adds a point to the current polygon."""
    if event == cv2.EVENT_LBUTTONUP:
        points.append((x, y))


cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    logger.error(f"Cannot open video: {VIDEO_PATH}")
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

cv2.namedWindow("image")
cv2.setMouseCallback("image", draw_polygon)

logger.info("Controls: Left-click=add point | S=save polygon | R=remove last | Q=quit")

while True:
    ret, frame = cap.read()
    if not ret:
        # Loop video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    mask_occupied = np.zeros_like(frame)
    mask_free = np.zeros_like(frame)

    results = model(frame, device=DEVICE)[0]
    occupied_indices = set()

    for detection in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        label_name = get_label_name(class_id)

        if label_name not in VEHICLE_CLASSES:
            continue

        car_polygon = [
            (int(x1), int(y1)), (int(x1), int(y2)),
            (int(x2), int(y2)), (int(x2), int(y1))
        ]
        car_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        car_shapely = ShapelyPolygon(car_polygon)

        for idx, slot in enumerate(polygon_data):
            if idx in occupied_indices:
                continue

            slot_center = find_polygon_center(slot)

            # Method 1: center-point check (fast)
            if is_point_in_polygon(slot_center, car_polygon) or is_point_in_polygon(car_center, slot):
                occupied_indices.add(idx)
                continue

            # Method 2: intersection area check (robust)
            try:
                slot_shapely = ShapelyPolygon(slot)
                if not slot_shapely.is_valid:
                    slot_shapely = slot_shapely.buffer(0)
                intersection_area = car_shapely.intersection(slot_shapely).area
                slot_area = slot_shapely.area
                if slot_area > 0 and (intersection_area / slot_area) > OCCUPANCY_THRESHOLD:
                    occupied_indices.add(idx)
            except Exception as e:
                logger.debug(f"Intersection error for slot {idx}: {e}")

    # Draw overlays
    for idx in occupied_indices:
        cv2.fillPoly(mask_occupied, [np.array(polygon_data[idx])], (0, 0, 255))

    for idx, slot in enumerate(polygon_data):
        if idx not in occupied_indices:
            cv2.fillPoly(mask_free, [np.array(slot)], (0, 255, 255))

    frame = cv2.addWeighted(mask_occupied, 0.2, frame, 1, 0)
    frame = cv2.addWeighted(mask_free, 0.2, frame, 1, 0)

    # HUD text
    cv2.putText(frame, f'Total: {len(polygon_data)}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (8, 210, 255), 2, cv2.LINE_4)
    cv2.putText(frame, f'Free: {len(polygon_data) - len(occupied_indices)}', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (8, 210, 90), 3, cv2.LINE_4)

    # Draw current in-progress polygon points
    for x, y in points:
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], (255, 255, 0), 1)

    cv2.imshow("image", frame)

    key = cv2.waitKey(1)

    if key in (ord('s'), ord('S')):
        if len(points) >= 3:
            polygon_data.append(list(points))
            points.clear()
            save_object(polygon_data)
            logger.info(f"Polygon saved. Total: {len(polygon_data)}")
        else:
            logger.warning("Need at least 3 points to save a polygon.")

    elif key in (ord('r'), ord('R')):
        if points:
            points.pop()
        elif polygon_data:
            polygon_data.pop()
            save_object(polygon_data)
            logger.info(f"Last polygon removed. Total: {len(polygon_data)}")

    elif key & 0xFF in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()
