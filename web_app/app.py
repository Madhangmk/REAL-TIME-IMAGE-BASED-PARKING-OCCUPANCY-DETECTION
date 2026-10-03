import os
import json
import cv2
import numpy as np
import base64
import sqlite3
import hashlib
import time
import logging
import threading
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from ultralytics import YOLO
from shapely.geometry import Polygon as ShapelyPolygon

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App & Config
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # disable static file caching

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
OBJECT_DIR = os.path.join(BASE_DIR, 'object')
DATABASE_DIR = os.path.join(BASE_DIR, 'database')

# Secret key: use env var in production, fallback for dev only
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(32))

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

# Detection constants
OCCUPANCY_THRESHOLD = float(os.environ.get('OCCUPANCY_THRESHOLD', '0.3'))
DEFAULT_FPS = 30
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
DEVICE = os.environ.get('YOLO_DEVICE', 'cpu')  # set to '0' for GPU

VEHICLE_CLASSES = {"bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"}
VEHICLE_LABELS = {
    0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
    5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor",
}

# Create directories
for d in (UPLOAD_FOLDER, OBJECT_DIR, DATABASE_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Password hashing (SHA-256 + per-user salt stored in DB)
# ---------------------------------------------------------------------------
def hash_password(password, salt=None):
    """Hash password with SHA-256 + salt. Returns (hashed, salt)."""
    if salt is None:
        salt = os.urandom(16).hex()
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return hashed, salt

def verify_password(password, stored_hash, salt):
    """Verify a password against stored hash and salt."""
    hashed, _ = hash_password(password, salt)
    return hashed == stored_hash

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def get_db():
    """Return a new SQLite connection."""
    return sqlite3.connect(os.path.join(DATABASE_DIR, 'users.db'))

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email    TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        salt     TEXT NOT NULL DEFAULT '',
        role     TEXT NOT NULL DEFAULT 'user'
    )''')

    # Add salt column if upgrading from old schema
    try:
        c.execute("SELECT salt FROM users LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE users ADD COLUMN salt TEXT NOT NULL DEFAULT ''")

    # Seed default accounts only if table is empty
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        for uname, email, pwd, role in [
            ('admin', 'admin@smartparking.com', 'admin123', 'admin'),
            ('user',  'user@smartparking.com',  'user123',  'user'),
        ]:
            h, s = hash_password(pwd)
            c.execute(
                "INSERT INTO users (username, email, password, salt, role) VALUES (?,?,?,?,?)",
                (uname, email, h, s, role)
            )
    conn.commit()
    conn.close()
    logger.info("Database initialised.")

init_db()

# ---------------------------------------------------------------------------
# Occupancy history (persisted to JSON)
# ---------------------------------------------------------------------------
HISTORY_FILE = os.path.join(OBJECT_DIR, 'occupancy_history.json')

def load_history():
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_history(entry):
    """Append one occupancy snapshot to the history file (max 1000 entries)."""
    history = load_history()
    history.append(entry)
    if len(history) > 1000:
        history = history[-1000:]
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

# ---------------------------------------------------------------------------
# YOLO model
# ---------------------------------------------------------------------------
model_path = os.path.join(
    os.path.dirname(BASE_DIR), "Models", "yolov8m mAp 48", "weights", "best.pt"
)
try:
    model = YOLO(model_path)
    logger.info(f"YOLO model loaded from {model_path} (device={DEVICE})")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    raise

# ---------------------------------------------------------------------------
# Thread-safe in-memory state
# ---------------------------------------------------------------------------
_state_lock = threading.Lock()
_state = {
    'video_states': {},          # {filename: {'last_frame': int}}
    'processed_cache': {},       # {cache_key: {'data': ..., 'timestamp': float}}
    'current_admin_video': None,
    'slot_occupancy_times': {},  # {filename: {slot_idx_str: time_str}}
}

def get_state(key):
    with _state_lock:
        return _state[key]

def set_state(key, value):
    with _state_lock:
        _state[key] = value

def update_state(key, subkey, value):
    with _state_lock:
        _state[key][subkey] = value

# ---------------------------------------------------------------------------
# Polygon storage
# ---------------------------------------------------------------------------
POLYGONS_FILE = os.path.join(OBJECT_DIR, "polygons_data.json")

def load_polygons_data():
    try:
        with open(POLYGONS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_polygons_data(data):
    with open(POLYGONS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def get_video_polygons(filename):
    return load_polygons_data().get(filename, [])

def save_video_polygons(filename, polygons):
    data = load_polygons_data()
    data[filename] = polygons
    save_polygons_data(data)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_polygons_hash(polygons):
    return hashlib.md5(json.dumps(polygons, sort_keys=True).encode()).hexdigest()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_label_name(n):
    return VEHICLE_LABELS.get(int(n), "unknown")

def find_polygon_center(points):
    if not points:
        return (0, 0)
    return (
        int(sum(p[0] for p in points) / len(points)),
        int(sum(p[1] for p in points) / len(points)),
    )

def is_point_in_polygon(point, polygon):
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

def get_video_time(frame_number, fps):
    if fps <= 0:
        return "00:00:00"
    total_seconds = int(frame_number / fps)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def get_video_fps(filepath):
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    cap.release()
    return fps

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            if request.is_json:
                return jsonify({'error': 'Authentication required'}), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'admin':
            if request.is_json:
                return jsonify({'error': 'Admin access required'}), 403
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------
def process_frame(frame, polygon_data, filename=None, frame_number=0, show_boxes=False):
    """Run YOLO detection and return annotated frame + occupancy data."""
    if not polygon_data:
        polygon_data = []

    mask_occupied = np.zeros_like(frame)
    mask_free = np.zeros_like(frame)

    results = model(frame, device=DEVICE)[0]
    occupied_indices = set()
    slot_vehicle_map = {}

    for detection in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        label_name = get_label_name(int(class_id))

        if label_name not in VEHICLE_CLASSES:
            continue

        if show_boxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_name} {score:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        car_polygon = [
            (int(x1), int(y1)), (int(x1), int(y2)),
            (int(x2), int(y2)), (int(x2), int(y1))
        ]
        car_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        car_shapely = ShapelyPolygon(car_polygon)

        for idx, poly in enumerate(polygon_data):
            if idx in occupied_indices:
                continue

            slot_center = find_polygon_center(poly)
            if is_point_in_polygon(slot_center, car_polygon) or is_point_in_polygon(car_center, poly):
                occupied_indices.add(idx)
                slot_vehicle_map[str(idx)] = label_name
                continue

            try:
                slot_shapely = ShapelyPolygon(poly)
                if not slot_shapely.is_valid:
                    slot_shapely = slot_shapely.buffer(0)
                intersection_area = car_shapely.intersection(slot_shapely).area
                slot_area = slot_shapely.area
                if slot_area > 0 and (intersection_area / slot_area) > OCCUPANCY_THRESHOLD:
                    occupied_indices.add(idx)
                    slot_vehicle_map[str(idx)] = label_name
            except Exception as e:
                logger.debug(f"Intersection error slot {idx}: {e}")

    # Occupancy time tracking (thread-safe)
    fps = DEFAULT_FPS
    if filename:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            fps = get_video_fps(filepath)

    v_time = get_video_time(frame_number, fps)

    if filename:
        with _state_lock:
            times = _state['slot_occupancy_times'].setdefault(filename, {})
            for idx in occupied_indices:
                times.setdefault(str(idx), v_time)
            for idx_str in list(times.keys()):
                if int(idx_str) not in occupied_indices:
                    del times[idx_str]

    # Draw overlays
    for idx in occupied_indices:
        cv2.fillPoly(mask_occupied, [np.array(polygon_data[idx])], (0, 0, 255))
    for idx, poly in enumerate(polygon_data):
        if idx not in occupied_indices:
            cv2.fillPoly(mask_free, [np.array(poly)], (0, 255, 255))

    frame = cv2.addWeighted(mask_occupied, 0.2, frame, 1, 0)
    frame = cv2.addWeighted(mask_free, 0.2, frame, 1, 0)

    with _state_lock:
        times = _state['slot_occupancy_times'].get(filename, {}) if filename else {}

    return frame, len(polygon_data) - len(occupied_indices), list(occupied_indices), slot_vehicle_map, dict(times)

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
CACHE_MAX = 100
CACHE_EVICT = 20

def cache_put(key, data):
    with _state_lock:
        _state['processed_cache'][key] = {'data': data, 'timestamp': time.time()}
        if len(_state['processed_cache']) > CACHE_MAX:
            sorted_keys = sorted(
                _state['processed_cache'],
                key=lambda k: _state['processed_cache'][k]['timestamp']
            )
            for k in sorted_keys[:CACHE_EVICT]:
                del _state['processed_cache'][k]

def cache_get(key):
    with _state_lock:
        entry = _state['processed_cache'].get(key)
        if entry:
            entry['timestamp'] = time.time()
            return entry['data']
    return None

@app.after_request
def add_no_cache(response):
    """Prevent browser from caching HTML pages."""
    if 'text/html' in response.content_type:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

# ---------------------------------------------------------------------------
# Routes — Auth
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            return render_template('login.html', error='Username and password required', page='login')

        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT id, username, role, password, salt FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()

        if user and verify_password(password, user[3], user[4]):
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['role'] = user[2]
            logger.info(f"User '{username}' logged in.")
            return redirect(url_for('dashboard'))

        logger.warning(f"Failed login attempt for '{username}'.")
        return render_template('login.html', error='Invalid username or password', page='login')

    return render_template('login.html', page='login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
        role = request.form.get('role', 'user')

        if password != confirm:
            return render_template('login.html', error='Passwords do not match', page='register')
        if len(password) < 6:
            return render_template('login.html', error='Password must be at least 6 characters', page='register')
        if role not in ('user', 'admin'):
            role = 'user'

        try:
            h, s = hash_password(password)
            conn = get_db()
            c = conn.cursor()
            c.execute(
                "INSERT INTO users (username, email, password, salt, role) VALUES (?,?,?,?,?)",
                (username, email, h, s, role)
            )
            conn.commit()
            conn.close()
            logger.info(f"New user registered: '{username}' ({role})")
            return render_template('login.html', success='Registration successful! Please login.', page='login')
        except sqlite3.IntegrityError:
            return render_template('login.html', error='Username or email already exists', page='register')

    return render_template('login.html', page='register')

@app.route('/logout')
def logout():
    username = session.get('username', 'unknown')
    session.clear()
    logger.info(f"User '{username}' logged out.")
    return redirect(url_for('login'))

# ---------------------------------------------------------------------------
# Routes — Dashboard
# ---------------------------------------------------------------------------
@app.route('/dashboard')
@login_required
def dashboard():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE id = ?", (session['user_id'],))
    result = c.fetchone()
    conn.close()
    role = result[0] if result else 'user'
    session['role'] = role
    if role == 'admin':
        return render_template('index.html', logged_in=True, username=session.get('username'), role='admin')
    return render_template('user_dashboard.html', logged_in=True, username=session.get('username'), role='user')

@app.route('/user-dashboard')
@login_required
def user_dashboard():
    return render_template('user_dashboard.html', logged_in=True, username=session.get('username'))

@app.route('/admin-dashboard')
@login_required
def admin_dashboard():
    return render_template('index.html', logged_in=True, username=session.get('username'), role='admin')

# ---------------------------------------------------------------------------
# Routes — Video
# ---------------------------------------------------------------------------
@app.route('/upload', methods=['POST'])
@login_required
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid or unsupported file type'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Verify it's a real video
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        cap.release()
        os.remove(filepath)
        return jsonify({'error': 'File is not a valid video'}), 400
    cap.release()

    with _state_lock:
        _state['current_admin_video'] = filename
    logger.info(f"Video uploaded: {filename}")
    return jsonify({'success': True, 'filename': filename, 'url': f'/static/uploads/{filename}'})

@app.route('/get-video-info', methods=['GET'])
@login_required
def get_video_info():
    filename = request.args.get('filename', '')
    if not filename:
        return jsonify({'error': 'Filename required'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    cap = cv2.VideoCapture(filepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    cap.release()
    return jsonify({'total_frames': total_frames, 'fps': fps})

@app.route('/get-videos', methods=['GET'])
@login_required
def get_videos():
    uploads_dir = app.config['UPLOAD_FOLDER']
    videos = [
        f for f in os.listdir(uploads_dir)
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ] if os.path.exists(uploads_dir) else []
    return jsonify({'videos': sorted(videos)})

# ---------------------------------------------------------------------------
# Routes — Polygons
# ---------------------------------------------------------------------------
@app.route('/save-polygons', methods=['POST'])
@login_required
def save_polygons():
    data = request.json or {}
    filename = data.get('filename', '').strip()
    points = data.get('points', [])

    if not filename:
        return jsonify({'error': 'Filename required'}), 400
    if not points or len(points) < 3:
        return jsonify({'error': 'Need at least 3 points'}), 400

    polygon_data = get_video_polygons(filename)
    polygon_data.append(points)
    save_video_polygons(filename, polygon_data)
    return jsonify({'success': True, 'count': len(polygon_data)})

@app.route('/get-polygons', methods=['GET'])
@login_required
def get_polygons():
    filename = request.args.get('filename', '').strip()
    polygons = get_video_polygons(filename) if filename else []
    return jsonify({'polygons': polygons})

@app.route('/clear-polygons', methods=['POST'])
@login_required
def clear_polygons():
    data = request.json or {}
    filename = data.get('filename', '').strip()
    if filename:
        save_video_polygons(filename, [])
    return jsonify({'success': True})

# ---------------------------------------------------------------------------
# Routes — Frame processing
# ---------------------------------------------------------------------------
@app.route('/process-frame', methods=['POST'])
@login_required
def process_frame_route():
    data = request.json or {}
    filename = data.get('filename', '').strip()
    frame_number = int(data.get('frame_number', 0))

    if not filename:
        return jsonify({'error': 'Filename required'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    cap = cv2.VideoCapture(filepath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({'error': 'Failed to read frame'}), 400

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    polygon_data = get_video_polygons(filename)

    with _state_lock:
        _state['current_admin_video'] = filename
        _state['video_states'].setdefault(filename, {})['last_frame'] = frame_number

    # Admin view (with bounding boxes)
    processed_admin, free_count, occupied_indices, slot_vehicle_map, occupancy_times = \
        process_frame(frame.copy(), polygon_data, filename, frame_number, show_boxes=True)

    # User/cache view (clean)
    processed_user, _, _, _, _ = \
        process_frame(frame.copy(), polygon_data, filename, frame_number, show_boxes=False)

    _, buf_admin = cv2.imencode('.jpg', processed_admin, [cv2.IMWRITE_JPEG_QUALITY, 85])
    _, buf_user = cv2.imencode('.jpg', processed_user, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_b64_admin = base64.b64encode(buf_admin).decode('utf-8')
    frame_b64_user = base64.b64encode(buf_user).decode('utf-8')

    total = len(polygon_data)
    slots_status = [i not in occupied_indices for i in range(total)]

    user_result = {
        'total': total,
        'free': free_count,
        'occupied': len(occupied_indices),
        'slots_status': slots_status,
        'slot_vehicles': slot_vehicle_map,
        'occupancy_times': occupancy_times,
        'video': filename,
        'frame': frame_b64_user,
    }
    cache_key = f"{filename}_{frame_number}_{get_polygons_hash(polygon_data)}"
    cache_put(cache_key, user_result)

    # Save occupancy snapshot to history
    save_history({
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'video': filename,
        'frame': frame_number,
        'total': total,
        'free': free_count,
        'occupied': len(occupied_indices),
    })

    return jsonify({
        'frame': frame_b64_admin,
        'free': free_count,
        'occupied': list(occupied_indices),
        'total': total,
        'slot_vehicles': slot_vehicle_map,
        'occupancy_times': occupancy_times,
    })

# ---------------------------------------------------------------------------
# Routes — Stats API
# ---------------------------------------------------------------------------
@app.route('/api/parking-stats')
@login_required
def get_parking_stats():
    filename_param = request.args.get('filename', '').strip()

    with _state_lock:
        current_video = _state['current_admin_video']
        video_states = dict(_state['video_states'])

    latest_video = filename_param or current_video

    if not latest_video:
        return jsonify({
            'total': 0, 'free': 0, 'occupied': 0,
            'slots_status': [], 'video': None,
            'message': 'No video active. Ask admin to process a video first.'
        })

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(latest_video))
    if not os.path.exists(filepath):
        return jsonify({'total': 0, 'free': 0, 'occupied': 0, 'slots_status': [], 'video': None})

    polygon_data = get_video_polygons(latest_video)
    frame_number = video_states.get(latest_video, {}).get('last_frame', 0)
    cache_key = f"{latest_video}_{frame_number}_{get_polygons_hash(polygon_data)}"

    cached = cache_get(cache_key)
    if cached:
        return jsonify(cached)

    cap = cv2.VideoCapture(filepath)
    if frame_number > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({'total': len(polygon_data), 'free': 0, 'occupied': 0,
                        'slots_status': [], 'video': latest_video})

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    processed, free_count, occupied_indices, slot_vehicle_map, occupancy_times = \
        process_frame(frame, polygon_data, latest_video, frame_number, show_boxes=False)

    total = len(polygon_data)
    _, buf = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_b64 = base64.b64encode(buf).decode('utf-8')

    result = {
        'total': total,
        'free': free_count,
        'occupied': len(occupied_indices),
        'slots_status': [i not in occupied_indices for i in range(total)],
        'slot_vehicles': slot_vehicle_map,
        'occupancy_times': occupancy_times,
        'video': latest_video,
        'frame': frame_b64,
    }
    cache_put(cache_key, result)
    return jsonify(result)

@app.route('/api/occupancy-history')
@login_required
def occupancy_history():
    """Return the last N occupancy snapshots."""
    limit = min(int(request.args.get('limit', 50)), 500)
    history = load_history()
    return jsonify({'history': history[-limit:]})

@app.route('/api/analytics')
@login_required
def analytics():
    """Return aggregated analytics: hourly averages and per-video totals."""
    history = load_history()
    if not history:
        return jsonify({'hourly': [], 'videos': {}})

    # Hourly occupancy rate buckets
    hourly = {}
    video_totals = {}

    for entry in history:
        total = entry.get('total', 0)
        if total == 0:
            continue
        occupied = entry.get('occupied', 0)
        rate = round(occupied / total * 100, 1)

        # Parse hour from timestamp "YYYY-MM-DD HH:MM:SS"
        ts = entry.get('timestamp', '')
        hour = ts[11:13] if len(ts) >= 13 else '00'
        if hour not in hourly:
            hourly[hour] = {'rates': [], 'label': f"{hour}:00"}
        hourly[hour]['rates'].append(rate)

        # Per-video stats
        vid = entry.get('video', 'unknown')
        if vid not in video_totals:
            video_totals[vid] = {'samples': 0, 'total_rate': 0.0, 'peak': 0.0}
        video_totals[vid]['samples'] += 1
        video_totals[vid]['total_rate'] += rate
        if rate > video_totals[vid]['peak']:
            video_totals[vid]['peak'] = rate

    hourly_out = sorted([
        {
            'hour': h,
            'label': v['label'],
            'avg_occupancy': round(sum(v['rates']) / len(v['rates']), 1),
            'samples': len(v['rates'])
        }
        for h, v in hourly.items()
    ], key=lambda x: x['hour'])

    videos_out = {
        vid: {
            'avg_occupancy': round(d['total_rate'] / d['samples'], 1),
            'peak_occupancy': d['peak'],
            'samples': d['samples']
        }
        for vid, d in video_totals.items()
    }

    return jsonify({'hourly': hourly_out, 'videos': videos_out})

@app.route('/api/clear-system', methods=['POST'])
@admin_required
def clear_system():
    with _state_lock:
        _state['current_admin_video'] = None
        _state['video_states'] = {}
        _state['processed_cache'] = {}
        _state['slot_occupancy_times'] = {}
    logger.info("System state cleared by admin.")
    return jsonify({'success': True, 'message': 'System state cleared.'})

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    logger.info(f"Starting Smart Parking app on port {port} (debug={debug})")
    app.run(debug=debug, host='0.0.0.0', port=port)
