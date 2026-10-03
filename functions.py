import os
import pickle
import logging
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

# Constants
VEHICLE_LABELS = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}

VEHICLE_CLASSES = {"bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"}

OCCUPANCY_THRESHOLD = 0.3  # 30% intersection to mark slot as occupied
DEFAULT_FPS = 30
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


def find_polygon_center(points):
    """Find the centroid of a polygon given its points."""
    if not points:
        return (0, 0)
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    return (int(sum(x_coords) / len(points)), int(sum(y_coords) / len(points)))


def is_point_in_polygon(point, polygon):
    """Ray-casting algorithm to check if a point is inside a polygon."""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def box_center(x1, y1, x2, y2):
    """Find the center of a bounding box."""
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def get_label_name(n):
    """Return the label name for a given class index."""
    return VEHICLE_LABELS.get(int(n), "unknown")


def compute_iou(polygon_1, polygon_2):
    """Calculate the Intersection over Union (IoU) of two polygons."""
    p1 = Polygon(polygon_1)
    p2 = Polygon(polygon_2)
    intersect = p1.intersection(p2).area
    union = p1.union(p2).area
    if union == 0:
        return 0.0
    return intersect / union


def save_object(polygon_data):
    """Save the polygon list to disk using pickle."""
    os.makedirs("object", exist_ok=True)
    with open("object/poligon.obj", "wb") as f:
        pickle.dump(polygon_data, f)


def load_object():
    """Load the polygon list from disk. Returns empty list if file missing."""
    try:
        with open("object/poligon.obj", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        logger.info("No polygon file found, starting fresh.")
        save_object([])
        return []
    except Exception as e:
        logger.error(f"Failed to load polygon file: {e}")
        save_object([])
        return []


def get_video_time(frame_number, fps):
    """Convert a frame number to HH:MM:SS string."""
    if fps <= 0:
        return "00:00:00"
    total_seconds = int(frame_number / fps)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
