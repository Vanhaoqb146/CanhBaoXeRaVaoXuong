import cv2
import json
import math
import os
import threading
import time
import collections
from datetime import datetime

import numpy as np
from ultralytics import YOLO

from zalo_utils_refactored import send_alert


# Load config
try:
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
except Exception as e:
    print(f"Lỗi đọc file cấu hình: {e}")
    exit(1)

RTSP_URL = config["rtsp_url"]
TARGET_CLASSES = config.get("target_classes", [1, 2, 3, 5, 7])
DAYS_TO_KEEP = int(config.get("days_to_keep_images", 7))

# Runtime tuning (có thể thêm vào config.json khi cần tinh chỉnh)
MODEL_PATH = config.get("model_path", "yolov8s.pt")
TRACKER_CONFIG = config.get("tracker_config", "tracker_bytetrack.yaml")
MODEL_INPUT_SIZE = int(config.get("model_input_size", 640))
DETECTION_CONF = float(config.get("detection_conf", 0.25))
FRAME_SKIP = max(1, int(config.get("frame_skip", 1)))
DEBUG_MODE = bool(config.get("debug_mode", False))
DEBUG_LOG_INTERVAL = max(1, int(config.get("debug_log_interval", 30)))  # Log mỗi N frame
DAYTIME_ONLY = bool(config.get("daytime_only", True))
ENABLE_NIGHT_ENHANCE = bool(config.get("enable_night_enhance", not DAYTIME_ONLY))
NIGHT_BRIGHTNESS_THRESHOLD = float(config.get("night_brightness_threshold", 80))
NIGHT_CONF_BOOST = float(config.get("night_conf_boost", 0.15))
CLAHE_CLIP_LIMIT = float(config.get("clahe_clip_limit", 3.0))
CLAHE_TILE_GRID = int(config.get("clahe_tile_grid", 8))

# Bộ lọc ban đêm & bbox tối thiểu
MIN_BOX_AREA_DAY = float(config.get("min_box_area_day", 800.0))
MIN_BOX_AREA_NIGHT = float(config.get("min_box_area_night", 1500.0))
MIN_CONSECUTIVE_DETECTIONS = max(1, int(config.get("min_consecutive_detections", 3)))


def _load_class_float_map(config_key):
    raw = config.get(config_key, {})
    if not isinstance(raw, dict):
        return {}

    parsed = {}
    for class_id, value in raw.items():
        try:
            parsed[int(class_id)] = float(value)
        except (TypeError, ValueError):
            print(f"⚠️ Bỏ qua cấu hình {config_key}.{class_id} không hợp lệ: {value}")
    return parsed


def _load_class_int_map(config_key):
    raw = config.get(config_key, {})
    if not isinstance(raw, dict):
        return {}

    parsed = {}
    for class_id, value in raw.items():
        try:
            parsed[int(class_id)] = max(1, int(value))
        except (TypeError, ValueError):
            print(f"⚠️ Bỏ qua cấu hình {config_key}.{class_id} không hợp lệ: {value}")
    return parsed


CLASS_MIN_CONF = _load_class_float_map("class_min_conf")
CLASS_MIN_BOX_AREA_DAY = _load_class_float_map("class_min_box_area_day")
CLASS_MIN_BOX_AREA_NIGHT = _load_class_float_map("class_min_box_area_night")
CLASS_MIN_CONSECUTIVE_DETECTIONS = _load_class_int_map("class_min_consecutive_detections")
ALERT_MIN_CONF = _load_class_float_map("alert_min_conf")
FAST_VEHICLE_CLASS_IDS = {
    int(class_id) for class_id in config.get("fast_vehicle_classes", [5, 7])
}
FAST_VEHICLE_MIN_MOVE = float(config.get("fast_vehicle_min_move", 18.0))
FAST_VEHICLE_OBSERVE_TIME = float(config.get("fast_vehicle_observe_time", 0.0))

GLOBAL_COOLDOWN_SEC = float(config.get("global_cooldown_sec", 180))
ALERT_COOLDOWN_SEC = float(config.get("alert_cooldown_sec", min(GLOBAL_COOLDOWN_SEC, 10.0)))
GENERIC_ALERT_COOLDOWN_SEC = float(
    config.get("generic_alert_cooldown_sec", min(ALERT_COOLDOWN_SEC, 30.0))
)
OBSERVATION_TIME_SEC = float(config.get("observation_time_sec", 1.0))
MIN_INSIDE_FRAMES = max(1, int(config.get("min_inside_frames", 3)))
INNER_WINDOW_SIZE = max(MIN_INSIDE_FRAMES, int(config.get("inner_window_size", MIN_INSIDE_FRAMES + 4)))

MIN_MOTION_SAMPLES = max(3, int(config.get("min_motion_samples", 5)))
MIN_DELTA_Y = float(config.get("min_delta_y", 10.0))
MIN_TOTAL_MOVE = float(config.get("min_total_move", 15.0))
MIN_DELTA_X = float(config.get("min_delta_x", 12.0))
MIN_TOTAL_MOVE_X = float(config.get("min_total_move_x", 15.0))
DIRECTION_VOTE_RATIO = float(config.get("direction_vote_ratio", 0.65))
ENABLE_X_DIRECTION_FALLBACK = bool(config.get("enable_x_direction_fallback", True))
# Camera hiện tại: xe VÀO thường đi sang trái (X giảm), xe RA thường đi sang phải (X tăng)
IN_X_SIGN = -1 if int(config.get("in_x_sign", -1)) < 0 else 1
FLOW_X_WEIGHT = float(config.get("flow_x_weight", float(IN_X_SIGN)))
FLOW_Y_WEIGHT = float(config.get("flow_y_weight", 1.0))
FLOW_VECTOR_NORM = math.hypot(FLOW_X_WEIGHT, FLOW_Y_WEIGHT) or 1.0
FLOW_X_WEIGHT /= FLOW_VECTOR_NORM
FLOW_Y_WEIGHT /= FLOW_VECTOR_NORM
MIN_FLOW_DELTA = float(config.get("min_flow_delta", max(MIN_DELTA_Y, MIN_DELTA_X)))
MIN_FLOW_TOTAL_MOVE = float(
    config.get("min_flow_total_move", max(MIN_TOTAL_MOVE, MIN_TOTAL_MOVE_X))
)
EXIT_GATE_SCALE = float(config.get("exit_gate_scale", 1.28))
EXIT_GATE_TOP_MARGIN = float(config.get("exit_gate_top_margin", 80.0))
EXIT_GATE_X_MARGIN = float(config.get("exit_gate_x_margin", 120.0))
EXIT_GATE_MIN_FLOW_DELTA = float(config.get("exit_gate_min_flow_delta", MIN_FLOW_DELTA))

# Gate Line Crossing (Virtual Tripwire)
LINE_CROSSING_ENABLED = bool(config.get("line_crossing_enabled", True))
GATE_LINE_ZONE = config.get("gate_line_zone", [])  # [{"x":...,"y":...}, {"x":...,"y":...}]
GATE_LINE_MIN_CROSS_DISTANCE = float(config.get("gate_line_min_cross_distance", 10.0))
GATE_LINE_SIDE_TOLERANCE = float(config.get("gate_line_side_tolerance", 3.0))
GATE_LINE_DEBUG_DRAW = bool(config.get("gate_line_debug_draw", False))
GATE_LINE_REQUIRE_CONTEXT = bool(config.get("gate_line_require_context", True))
GATE_LINE_IN_SNAPSHOT_DELAY_SEC = float(config.get("gate_line_in_snapshot_delay_sec", 0.6))
GATE_LINE_IN_MIN_AFTER_CROSS_MOVE = float(config.get("gate_line_in_min_after_cross_move", 15.0))
GATE_LINE_OUT_MIN_AFTER_CROSS_MOVE = float(config.get("gate_line_out_min_after_cross_move", 8.0))

INNER_POLYGON_SCALE = float(config.get("inner_polygon_scale", 0.93))
MIN_OUTER_RATIO = float(config.get("min_outer_ratio", 0.70))
MIN_INNER_RATIO = float(config.get("min_inner_ratio", 0.50))
STALE_TRACK_SEC = float(config.get("stale_track_sec", 3.0))
MAX_OBSERVATION_SEC = float(config.get("max_observation_sec", 5.0))
STATIONARY_MIN_ENTRY_DELTA = float(config.get("stationary_min_entry_delta", 30.0))
STATIONARY_MIN_ENTRY_DELTA_X = float(
    config.get("stationary_min_entry_delta_x", STATIONARY_MIN_ENTRY_DELTA)
)
STATIONARY_MIN_FLOW_DELTA = float(
    config.get(
        "stationary_min_flow_delta",
        max(STATIONARY_MIN_ENTRY_DELTA, STATIONARY_MIN_ENTRY_DELTA_X),
    )
)
STATIONARY_DISPLACEMENT_THRESHOLD = float(config.get("stationary_displacement_threshold", 50.0))
STATIONARY_TOTAL_MOVE_THRESHOLD = float(config.get("stationary_total_move_threshold", 55.0))
STATIONARY_OSCILLATION_RATIO = float(config.get("stationary_oscillation_ratio", 0.30))
MAX_STOP_COUNT = max(1, int(config.get("max_stop_count", 2)))
INSTANT_LINE_CROSS_THRESHOLD = float(config.get("instant_line_cross_threshold", 25.0))
ANCHOR_Y_RATIO = float(config.get("anchor_y_ratio", 0.80))
SNAPSHOT_AREA_GAIN = float(config.get("snapshot_area_gain", 1.0))
SNAPSHOT_CENTER_WEIGHT = float(config.get("snapshot_center_weight", 0.35))
SNAPSHOT_ZONE_BONUS = float(config.get("snapshot_zone_bonus", 0.12))

# Xe tải cỡ lớn: bbox chiếm >= X% diện tích frame → dùng multi-anchor
LARGE_VEHICLE_AREA_RATIO = float(config.get("large_vehicle_area_ratio", 0.25))
LARGE_VEHICLE_EDGE_FRAGMENT_Y = float(config.get("large_vehicle_edge_fragment_y", 80.0))
# Snapshot hold: xe lớn RA tiếp tục gom frame thêm N giây sau khi classify hướng
# để đảm bảo chụp được lúc xe đã ra xa → thấy rõ biển số + hàng hóa
LARGE_VEHICLE_SNAPSHOT_HOLD_SEC = float(config.get("large_vehicle_snapshot_hold_sec", 1.5))

# Anti-spam: Spatial dedup + Global direction cooldown
SPATIAL_DEDUP_DISTANCE = float(config.get("spatial_dedup_distance", 100.0))
SPATIAL_DEDUP_SEC = float(config.get("spatial_dedup_sec", 10.0))
GLOBAL_DIRECTION_COOLDOWN_SEC = float(
    config.get("global_direction_cooldown_sec", max(ALERT_COOLDOWN_SEC * 2, 10.0))
)

# Giờ hoạt động (theo giờ Việt Nam UTC+7)
OPERATING_HOUR_START = int(config.get("operating_hour_start", 5))   # 5h sáng
OPERATING_HOUR_END = int(config.get("operating_hour_end", 19))     # 19h tối

# Đọc polygon
poly_config = config.get("polygon_zone", [])
if len(poly_config) < 3:
    print("Lỗi cấu hình: polygon_zone cần ít nhất 3 điểm")
    exit(1)

POLYGON_PTS = np.array([[pt["x"], pt["y"]] for pt in poly_config], np.int32)

if not os.path.exists("alerts"):
    os.makedirs("alerts")

import zoneinfo
VN_TZ = zoneinfo.ZoneInfo("Asia/Ho_Chi_Minh")

def is_within_operating_hours():
    """Kiểm tra có đang trong giờ hoạt động không (theo giờ VN)."""
    now_vn = datetime.now(VN_TZ)
    return OPERATING_HOUR_START <= now_vn.hour < OPERATING_HOUR_END


def build_scaled_polygon(pts: np.ndarray, scale: float) -> np.ndarray:
    pts_float = pts.astype(np.float32)
    centroid = np.mean(pts_float, axis=0)
    scaled = centroid + (pts_float - centroid) * scale
    return scaled.astype(np.int32)


def build_inner_polygon(pts: np.ndarray, scale: float) -> np.ndarray:
    """Thu nhỏ polygon về phía tâm để giảm false positive ở mép cổng."""
    if scale >= 0.999:
        return pts.copy()

    scale = max(0.30, min(scale, 0.999))
    pts_float = pts.astype(np.float32)
    centroid = np.mean(pts_float, axis=0)
    inner = centroid + (pts_float - centroid) * scale
    return inner.astype(np.int32)


def build_gate_line(pts: np.ndarray, edge_ratio: float, offset: float, length_scale: float):
    """Tự suy ra line gần miệng cổng từ top-vertex của polygon."""
    pts_float = pts.astype(np.float32)
    if len(pts_float) < 3:
        return None

    edge_ratio = max(0.20, min(edge_ratio, 0.75))
    length_scale = max(0.60, float(length_scale))

    top_idx = int(np.argmin(pts_float[:, 1]))
    prev_idx = (top_idx - 1) % len(pts_float)
    next_idx = (top_idx + 1) % len(pts_float)

    top_pt = pts_float[top_idx]
    prev_pt = pts_float[prev_idx]
    next_pt = pts_float[next_idx]

    left_pt = top_pt + (prev_pt - top_pt) * edge_ratio
    right_pt = top_pt + (next_pt - top_pt) * edge_ratio

    center = (left_pt + right_pt) / 2.0
    line_vec = right_pt - left_pt
    line_norm = np.linalg.norm(line_vec)
    if line_norm < 1e-6:
        return None

    half_vec = (line_vec * length_scale) / 2.0
    centroid = np.mean(pts_float, axis=0)
    inward = centroid - top_pt
    inward_norm = np.linalg.norm(inward)
    if inward_norm < 1e-6:
        inward_unit = np.array([0.0, 1.0], dtype=np.float32)
    else:
        inward_unit = inward / inward_norm

    offset_center = center + inward_unit * float(offset)
    start = offset_center - half_vec
    end = offset_center + half_vec
    return start.astype(np.float32), end.astype(np.float32), inward_unit.astype(np.float32), offset_center.astype(np.float32)


def signed_distance_to_line(point, line_start, line_end):
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    denom = math.hypot(x2 - x1, y2 - y1)
    if denom < 1e-6:
        return 0.0
    return ((px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)) / denom


def get_line_side(point, line_start, line_end, tolerance=0.0):
    dist = signed_distance_to_line(point, line_start, line_end)
    if dist > tolerance:
        return 1, dist
    if dist < -tolerance:
        return -1, dist
    return 0, dist


def detect_line_crossing(prev_point, curr_point, line_start, line_end, tolerance, min_cross_distance):
    prev_side, prev_dist = get_line_side(prev_point, line_start, line_end, tolerance)
    curr_side, curr_dist = get_line_side(curr_point, line_start, line_end, tolerance)
    moved = math.hypot(curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
    crossed = (
        prev_side != 0
        and curr_side != 0
        and prev_side != curr_side
        and moved >= min_cross_distance
        and abs(prev_dist - curr_dist) >= min_cross_distance
    )
    return crossed, prev_side, curr_side, prev_dist, curr_dist, moved


def create_observation(class_name, current_time, anchor_x, anchor_y, frame, inside_outer, inside_inner, box, start_reason, is_large=False):
    max_samples = 100
    return {
        "class_name": class_name,
        "first_time": current_time,
        "last_seen": current_time,
        "first_x": anchor_x,
        "last_x": anchor_x,
        "first_y": anchor_y,
        "last_y": anchor_y,
        "x_samples": collections.deque([anchor_x], maxlen=max_samples),
        "y_samples": collections.deque([anchor_y], maxlen=max_samples),
        "in_outer": collections.deque([1 if inside_outer else 0], maxlen=max_samples),
        "in_inner": collections.deque([1 if inside_inner else 0], maxlen=max_samples),
        "last_frame": frame.copy(),
        "last_box": list(box),
        "best_snapshot_frame": frame.copy(),
        "best_snapshot_score": float("-inf"),
        "best_snapshot_center_dist": float("inf"),
        "best_snapshot_area": 0.0,
        "start_reason": start_reason,
        "stop_alert_sent": False,
        "stop_count": 0,
        "line_direction": None,
        "line_cross_time": 0.0,
        "line_cross_anchor": None,
        "line_cross_distance": 0.0,
        "line_triggered": False,
        "is_large_vehicle": is_large,
        "class_id": None,
        "max_conf": 0.0,
        "max_conf_area": 0.0,
        # Snapshot hold: chờ thêm sau khi classify hướng để chụp frame tốt hơn
        "snapshot_hold_direction": None,   # Hướng đã xác định (chờ chụp)
        "snapshot_hold_started_at": None,   # Thời điểm bắt đầu hold
        "snapshot_hold_stats": None,        # Stats classify direction lúc trigger
        "snapshot_hold_best_frame": None,   # Frame tốt nhất trong hold period
        "snapshot_hold_best_score": float("-inf"),  # Score cao nhất trong hold
    }


def update_observation_conf(obs, class_id, conf_val, box_area):
    obs["class_id"] = class_id
    if conf_val >= obs.get("max_conf", 0.0):
        obs["max_conf"] = conf_val
        obs["max_conf_area"] = box_area


def passes_alert_conf_gate(obs):
    class_id = obs.get("class_id")
    min_conf = ALERT_MIN_CONF.get(class_id)
    if min_conf is None:
        return True
    return obs.get("max_conf", 0.0) >= min_conf


def prepend_observation_entry(obs, entry_x, entry_y, inside_outer, inside_inner):
    """Thêm vị trí frame đầu vào observation cho xe chạy nhanh chỉ còn 2 frame."""
    if math.hypot(obs["first_x"] - entry_x, obs["first_y"] - entry_y) < 1e-3:
        return

    obs["first_x"] = entry_x
    obs["first_y"] = entry_y
    obs["x_samples"].appendleft(entry_x)
    obs["y_samples"].appendleft(entry_y)
    obs["in_outer"].appendleft(1 if inside_outer else 0)
    obs["in_inner"].appendleft(1 if inside_inner else 0)


def get_anchor_point(box):
    """Dùng điểm ở 80% chiều cao box thay vì chân xe, tránh anchor rơi ngoài polygon."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, y1 + (y2 - y1) * ANCHOR_Y_RATIO


def get_multi_anchors(box):
    """Trả về nhiều anchor points cho xe lớn:
    - bottom-anchor (80% height - mặc định)
    - center (50% height)
    - top-center (30% height)
    Giúp xe tải cỡ lớn không bị miss khi single anchor rơi ngoài polygon.
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    return [
        (cx, y1 + (y2 - y1) * ANCHOR_Y_RATIO),  # bottom anchor
        (cx, y1 + (y2 - y1) * 0.50),              # center
        (cx, y1 + (y2 - y1) * 0.30),              # top-center
    ]


def is_large_vehicle(box, frame_w, frame_h):
    """Kiểm tra xe có phải cỡ lớn (bbox chiếm >= LARGE_VEHICLE_AREA_RATIO diện tích frame)."""
    x1, y1, x2, y2 = box
    box_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_w * frame_h
    return (box_area / max(frame_area, 1)) >= LARGE_VEHICLE_AREA_RATIO


def flow_progress(x_value, y_value):
    return x_value * FLOW_X_WEIGHT + y_value * FLOW_Y_WEIGHT


def update_snapshot_candidate(obs, frame, box, inside_outer, inside_inner):
    x1, y1, x2, y2 = box
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    box_area = box_w * box_h

    frame_h, frame_w = frame.shape[:2]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    norm_dx = (center_x - frame_w / 2.0) / max(frame_w / 2.0, 1.0)
    norm_dy = (center_y - frame_h / 2.0) / max(frame_h / 2.0, 1.0)
    center_dist = math.hypot(norm_dx, norm_dy)

    zone_bonus = 0.0
    if inside_outer:
        zone_bonus += SNAPSHOT_ZONE_BONUS
    if inside_inner:
        zone_bonus += SNAPSHOT_ZONE_BONUS

    snapshot_score = box_area * SNAPSHOT_AREA_GAIN * (1.0 + zone_bonus) - (
        center_dist * SNAPSHOT_CENTER_WEIGHT * box_area
    )

    best_score = obs.get("best_snapshot_score", float("-inf"))
    best_center_dist = obs.get("best_snapshot_center_dist", float("inf"))
    if (
        snapshot_score > best_score * 1.02
        or (abs(snapshot_score - best_score) <= max(abs(best_score) * 0.02, 50.0) and center_dist < best_center_dist)
    ):
        obs["best_snapshot_frame"] = frame.copy()
        obs["best_snapshot_score"] = snapshot_score
        obs["best_snapshot_center_dist"] = center_dist
        obs["best_snapshot_area"] = box_area

    obs["last_box"] = [x1, y1, x2, y2]


def get_snapshot_frame(obs, fallback_frame, direction=None):
    # Ưu tiên 1: frame từ snapshot hold (đã gom sau classify hướng)
    hold_frame = obs.get("snapshot_hold_best_frame")
    if hold_frame is not None:
        return hold_frame

    # Ưu tiên 2: xe lớn RA → lấy frame cuối (đã ra xa hơn)
    if direction == "RA KHỎI XƯỞNG" and obs.get("is_large_vehicle"):
        snapshot_frame = obs.get("last_frame")
        if snapshot_frame is not None:
            return snapshot_frame

    snapshot_frame = obs.get("best_snapshot_frame")
    if snapshot_frame is None:
        snapshot_frame = obs.get("last_frame")
    if snapshot_frame is None:
        snapshot_frame = fallback_frame
    return snapshot_frame


def update_snapshot_hold_candidate(obs, frame, box):
    """
    Scoring cho snapshot hold period (xe lớn RA).

    Tiêu chí:
    - Ưu tiên bbox NHỎ hơn (xe đã ra xa → thấy rõ toàn thân + biển số)
    - Nhưng bbox phải còn đủ lớn (≥ 5% frame) để ảnh rõ ràng
    - Trừ điểm nặng nếu bbox sát mép frame (bị cắt biển số)
    - Bonus nếu bbox gần trung tâm frame
    """
    x1, y1, x2, y2 = box
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    box_area = box_w * box_h

    frame_h, frame_w = frame.shape[:2]
    frame_area = frame_w * frame_h
    area_ratio = box_area / max(frame_area, 1)

    # Bỏ qua nếu xe quá nhỏ (đã ra hết, chỉ còn 1 chấm)
    if area_ratio < 0.05:
        return

    # Score = ưu tiên area_ratio NHỎ nhưng hợp lý (0.15-0.35 là lý tưởng)
    # Dùng: score = khoảng cách đến ideal_ratio càng nhỏ càng tốt
    ideal_ratio = 0.25  # lý tưởng: xe chiếm 25% frame
    ratio_penalty = abs(area_ratio - ideal_ratio)

    # Center bonus: xe ở gần trung tâm frame = thấy rõ hơn
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    norm_dx = (center_x - frame_w / 2.0) / max(frame_w / 2.0, 1.0)
    norm_dy = (center_y - frame_h / 2.0) / max(frame_h / 2.0, 1.0)
    center_dist = math.hypot(norm_dx, norm_dy)

    # Edge penalty: mạnh — nếu bbox sát mép → biển số bị cắt
    edge_margin = 10  # px
    edge_hits = 0
    if x1 <= edge_margin:
        edge_hits += 1
    if y1 <= edge_margin:
        edge_hits += 1
    if frame_w - x2 <= edge_margin:
        edge_hits += 1
    if frame_h - y2 <= edge_margin:
        edge_hits += 1

    # Score tổng: thấp hơn = tốt hơn → chuyển sang cao hơn = tốt hơn
    score = (1.0 - ratio_penalty) - (center_dist * 0.3) - (edge_hits * 0.25)

    best_score = obs.get("snapshot_hold_best_score", float("-inf"))
    if score > best_score:
        obs["snapshot_hold_best_frame"] = frame.copy()
        obs["snapshot_hold_best_score"] = score


def reset_observation_window(obs, current_time, current_x, current_y, inside_outer, inside_inner):
    max_samples = obs["x_samples"].maxlen or 100
    obs["first_time"] = current_time
    obs["last_seen"] = current_time
    obs["first_x"] = current_x
    obs["last_x"] = current_x
    obs["first_y"] = current_y
    obs["last_y"] = current_y
    obs["x_samples"] = collections.deque([current_x], maxlen=max_samples)
    obs["y_samples"] = collections.deque([current_y], maxlen=max_samples)
    obs["in_outer"] = collections.deque([1 if inside_outer else 0], maxlen=max_samples)
    obs["in_inner"] = collections.deque([1 if inside_inner else 0], maxlen=max_samples)


def classify_direction(x_samples, y_samples, min_samples=None):
    """
    Phân loại hướng từ chuỗi Y:
    - net_dy: độ dời ròng
    - total_move: tổng độ dời tuyệt đối
    - vote ratio: tỉ lệ bước cùng chiều
    """
    required_samples = MIN_MOTION_SAMPLES if min_samples is None else max(2, int(min_samples))
    if len(y_samples) < required_samples or len(x_samples) < required_samples:
        return None, {
            "method": "insufficient_samples",
            "net_flow": 0.0,
            "total_flow": 0.0,
            "in_ratio": 0.0,
            "out_ratio": 0.0,
            "net_dx": 0.0,
            "total_move_x": 0.0,
            "left_ratio": 0.0,
            "right_ratio": 0.0,
            "net_dy": 0.0,
            "total_move": 0.0,
            "up_ratio": 0.0,
            "down_ratio": 0.0,
        }

    xs = np.asarray(x_samples, dtype=np.float32)
    ys = np.asarray(y_samples, dtype=np.float32)
    flow_samples = xs * FLOW_X_WEIGHT + ys * FLOW_Y_WEIGHT
    step_x = np.diff(xs)
    step = np.diff(ys)
    step_flow = np.diff(flow_samples)

    if step.size == 0 or step_x.size == 0 or step_flow.size == 0:
        return None, {
            "method": "empty_step",
            "net_flow": 0.0,
            "total_flow": 0.0,
            "in_ratio": 0.0,
            "out_ratio": 0.0,
            "net_dx": 0.0,
            "total_move_x": 0.0,
            "left_ratio": 0.0,
            "right_ratio": 0.0,
            "net_dy": 0.0,
            "total_move": 0.0,
            "up_ratio": 0.0,
            "down_ratio": 0.0,
        }

    net_flow = float(flow_samples[-1] - flow_samples[0])
    total_flow = float(np.sum(np.abs(step_flow)))
    net_dx = float(xs[-1] - xs[0])
    total_move_x = float(np.sum(np.abs(step_x)))
    net_dy = float(ys[-1] - ys[0])
    total_move = float(np.sum(np.abs(step)))

    # Lọc nhiễu: chỉ tính vote trên bước di chuyển thực sự (>1px theo vector)
    # Ngăn các frame đứng yên làm loãng tỉ lệ vote
    NOISE_THRESHOLD = 1.0
    significant_mask = np.sqrt(step_x * step_x + step * step) > NOISE_THRESHOLD

    if np.any(significant_mask):
        sig_steps_flow = step_flow[significant_mask]
        sig_steps_x = step_x[significant_mask]
        sig_steps = step[significant_mask]
        in_ratio = float(np.mean(sig_steps_flow > 0))
        out_ratio = float(np.mean(sig_steps_flow < 0))
        right_ratio = float(np.mean(sig_steps_x > 0))
        left_ratio = float(np.mean(sig_steps_x < 0))
        down_ratio = float(np.mean(sig_steps > 0))
        up_ratio = float(np.mean(sig_steps < 0))
    else:
        in_ratio = 0.0
        out_ratio = 0.0
        right_ratio = 0.0
        left_ratio = 0.0
        down_ratio = 0.0
        up_ratio = 0.0

    stats = {
        "method": "none",
        "net_flow": net_flow,
        "total_flow": total_flow,
        "in_ratio": in_ratio,
        "out_ratio": out_ratio,
        "net_dx": net_dx,
        "total_move_x": total_move_x,
        "left_ratio": left_ratio,
        "right_ratio": right_ratio,
        "net_dy": net_dy,
        "total_move": total_move,
        "up_ratio": up_ratio,
        "down_ratio": down_ratio,
    }

    # Ưu tiên trục Y (giữ tương thích logic cũ)
    if abs(net_flow) >= MIN_FLOW_DELTA and total_flow >= MIN_FLOW_TOTAL_MOVE:
        if net_flow > 0 and in_ratio >= DIRECTION_VOTE_RATIO:
            stats["method"] = "flow_axis"
            return "VÀO XƯỞNG", stats

        if net_flow < 0 and out_ratio >= DIRECTION_VOTE_RATIO:
            stats["method"] = "flow_axis"
            return "RA KHỎI XƯỞNG", stats

    if abs(net_dy) >= MIN_DELTA_Y and total_move >= MIN_TOTAL_MOVE:
        # Ảnh camera này: Y tăng = xe đi xuống (vào sâu xưởng), Y giảm = đi lên (ra cổng)
        if net_dy > 0 and down_ratio >= DIRECTION_VOTE_RATIO:
            stats["method"] = "y_axis"
            return "VÀO XƯỞNG", stats

        if net_dy < 0 and up_ratio >= DIRECTION_VOTE_RATIO:
            stats["method"] = "y_axis"
            return "RA KHỎI XƯỞNG", stats

    # Fallback trục X: xử lý xe đi chéo/ra nhanh, Y thay đổi rất ít
    if ENABLE_X_DIRECTION_FALLBACK and abs(net_dx) >= MIN_DELTA_X and total_move_x >= MIN_TOTAL_MOVE_X:
        if IN_X_SIGN < 0:
            if net_dx < 0 and left_ratio >= DIRECTION_VOTE_RATIO:
                stats["method"] = "x_axis"
                return "VÀO XƯỞNG", stats
            if net_dx > 0 and right_ratio >= DIRECTION_VOTE_RATIO:
                stats["method"] = "x_axis"
                return "RA KHỎI XƯỞNG", stats
        else:
            if net_dx > 0 and right_ratio >= DIRECTION_VOTE_RATIO:
                stats["method"] = "x_axis"
                return "VÀO XƯỞNG", stats
            if net_dx < 0 and left_ratio >= DIRECTION_VOTE_RATIO:
                stats["method"] = "x_axis"
                return "RA KHỎI XƯỞNG", stats

    return None, stats


def cleanup_images():
    while True:
        try:
            now = time.time()
            cutoff = now - DAYS_TO_KEEP * 86400
            for filename in os.listdir("alerts"):
                filepath = os.path.join("alerts", filename)
                if os.path.isfile(filepath) and os.path.getctime(filepath) < cutoff:
                    os.remove(filepath)
                    print(f"🧹 Dọn rác: Đã xóa {filepath}")
        except Exception as e:
            print(f"⚠️ Cleanup Error: {e}")
        time.sleep(3600)


# ============================================================
# THREADED FRAME GRABBER - Luôn giữ frame mới nhất, giảm lag
# ============================================================
class FrameGrabber:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.running = True
        self.connected = False
        self.cap = None

    def start(self):
        self.thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.thread.start()
        return self

    def _grab_loop(self):
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    print("🔄 Đang kết nối tới RTSP...")
                    self.cap = cv2.VideoCapture(self.rtsp_url)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if not self.cap.isOpened():
                        print("❌ Không thể kết nối. Đợi 5 giây...")
                        self.connected = False
                        time.sleep(5)
                        continue
                    print("✅ Đã kết nối Camera thành công!")
                    self.connected = True

                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.ret = True
                        self.frame = frame
                else:
                    print("⚠️ Mất kết nối camera. Đang thử lại...")
                    self.cap.release()
                    self.cap = None
                    self.connected = False
                    time.sleep(3)
            except Exception as e:
                print(f"⚠️ FrameGrabber Error: {e}")
                if self.cap:
                    self.cap.release()
                self.cap = None
                self.connected = False
                time.sleep(3)

    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


def main():
    threading.Thread(target=cleanup_images, daemon=True).start()

    print(f"⏳ Đang tải mô hình {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    inner_polygon_pts = build_inner_polygon(POLYGON_PTS, INNER_POLYGON_SCALE)
    exit_gate_pts = build_scaled_polygon(POLYGON_PTS, max(1.0, EXIT_GATE_SCALE))
    exit_gate_top_x = float(POLYGON_PTS[np.argmin(POLYGON_PTS[:, 1])][0])
    exit_gate_y_limit = float(np.min(POLYGON_PTS[:, 1])) + EXIT_GATE_TOP_MARGIN

    # Gate Line Crossing initialization — đọc 2 điểm từ config
    gate_line = None
    gate_line_start = np.array([0.0, 0.0], dtype=np.float32)
    gate_line_end = np.array([0.0, 0.0], dtype=np.float32)
    gate_line_inside_side = 0

    if LINE_CROSSING_ENABLED and len(GATE_LINE_ZONE) >= 2:
        gate_line_start = np.array(
            [float(GATE_LINE_ZONE[0]["x"]), float(GATE_LINE_ZONE[0]["y"])], dtype=np.float32
        )
        gate_line_end = np.array(
            [float(GATE_LINE_ZONE[1]["x"]), float(GATE_LINE_ZONE[1]["y"])], dtype=np.float32
        )
        gate_line = (gate_line_start, gate_line_end)
        # Xác định side nào là "trong xưởng" bằng centroid
        centroid = np.mean(POLYGON_PTS.astype(np.float32), axis=0)
        _, inside_dist = get_line_side(
            (float(centroid[0]), float(centroid[1])),
            gate_line_start, gate_line_end, 0.0
        )
        gate_line_inside_side = 1 if inside_dist > 0 else -1
        print(
            f"🚧 Gate line: ({gate_line_start[0]:.0f},{gate_line_start[1]:.0f}) → "
            f"({gate_line_end[0]:.0f},{gate_line_end[1]:.0f}), inside_side={gate_line_inside_side}"
        )
    elif LINE_CROSSING_ENABLED:
        print("⚠️ gate_line_zone chưa cấu hình đủ 2 điểm! Line crossing sẽ bị tắt.")

    print(
        "🧭 Rule:"
        f" inner_scale={INNER_POLYGON_SCALE},"
        f" min_inside_frames={MIN_INSIDE_FRAMES}(window={INNER_WINDOW_SIZE}),"
        f" obs={OBSERVATION_TIME_SEC}s,"
        f" max_obs={MAX_OBSERVATION_SEC}s,"
        f" alert_cooldown={ALERT_COOLDOWN_SEC}s(direction),"
        f" generic_cd={GENERIC_ALERT_COOLDOWN_SEC}s,"
        f" flow=({FLOW_X_WEIGHT:.2f},{FLOW_Y_WEIGHT:.2f}),"
        f" exit_gate_scale={EXIT_GATE_SCALE},"
        f" exit_gate_x_margin={EXIT_GATE_X_MARGIN},"
        f" gate_line={LINE_CROSSING_ENABLED},"
        f" x_fallback={ENABLE_X_DIRECTION_FALLBACK},"
        f" daytime_only={DAYTIME_ONLY},"
        f" night_enhance={ENABLE_NIGHT_ENHANCE},"
        f" night_conf_boost={NIGHT_CONF_BOOST},"
        f" min_box_area={MIN_BOX_AREA_DAY}/{MIN_BOX_AREA_NIGHT},"
        f" min_consec_det={MIN_CONSECUTIVE_DETECTIONS},"
        f" oscillation_ratio={STATIONARY_OSCILLATION_RATIO},"
        f" max_stop_count={MAX_STOP_COUNT},"
        f" operating_hours={OPERATING_HOUR_START}h-{OPERATING_HOUR_END}h"
    )

    # Cooldown theo từng loại cảnh báo để không chặn chéo VÀO/RA
    last_alert_time_by_label = {}
    # Global direction cooldown (bất kể tracker ID)
    last_alert_time_by_direction = {}  # {"VÀO XƯỞNG": timestamp, "RA KHỎI XƯỞNG": timestamp}
    # Spatial dedup: lưu vị trí alert gần đây để chặn cùng chỗ + cùng hướng
    recent_alert_positions = []  # [(direction, x, y, timestamp), ...]
    effective_stale_track_sec = max(STALE_TRACK_SEC, MAX_OBSERVATION_SEC + 0.5)
    if effective_stale_track_sec > STALE_TRACK_SEC:
        pass  # print(
#             f"⚙️ Tự nâng stale_track_sec từ {STALE_TRACK_SEC:.1f}s "
#             f"lên {effective_stale_track_sec:.1f}s để không hủy trước timeout fallback"
#         )

    # Theo dõi ổn định mỗi ID
    known_ids = set()
    tracker_last_seen = {} # {tid: timestamp}
    inside_history = {}  # {tracker_id: deque of 1/0 for inner-zone sliding window}
    completed_cycle_ids = set()  # ID đã xử lý cảnh báo trong 1 chu kỳ còn ở trong vùng
    vehicle_first_x = {}     # {tid: X khi xe đầu tiên xuất hiện} - dùng cho timeout fallback
    vehicle_first_y = {}     # {tid: Y khi xe đầu tiên xuất hiện} - dùng cho timeout fallback

    # Quan sát chuyển động nhiều frame trước khi kết luận
    pending_observation = {}
    # pending_observation[tid] = {
    #   "class_name": str,
    #   "first_time": float,
    #   "last_seen": float,
    #   "first_x": float,
    #   "last_x": float,
    #   "first_y": float,
    #   "last_y": float,
    #   "x_samples": deque[float],     # Sliding window 100 frame
    #   "y_samples": deque[float],     # Sliding window 30 frame
    #   "in_outer": deque[int],         # 1/0 mỗi frame
    #   "in_inner": deque[int],         # 1/0 mỗi frame
    # }
    gate_track_state = {}  # Lưu side/anchor theo line crossing cho từng tracker
    track_detection_count = {}  # {Đếm số frame liên tiếp detect được mỗi ID}
    completed_stop_position = {}  # {tid: (x, y)} - vị trí khi xe bị đánh dấu completed do DẪNH DỮNG
    is_night_mode = False  # Flag đánh dấu đang ở chế độ ban đêm

    print("🟩 Hệ thống sẵn sàng!")
    print("🎥 Bắt đầu kết nối Camera...")

    grabber = FrameGrabber(RTSP_URL).start()

    frame_count = 0
    processed_count = 0
    fps_start_time = time.time()

    while not grabber.connected:
        time.sleep(0.5)

    while True:
        try:
            ret, frame = grabber.get_frame()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1
            current_time = time.time()

            # Kiểm tra giờ hoạt động — nghỉ ngoài khung 4h-22h
            if not is_within_operating_hours():
                if frame_count % 1800 == 1:  # Log 1 lần mỗi ~1 phút
                    now_vn = datetime.now(VN_TZ)
                    pass  # print(
#                         f"💤 Ngoài giờ hoạt động ({OPERATING_HOUR_START}h-{OPERATING_HOUR_END}h), "
#                         f"hiện tại {now_vn.strftime('%H:%M')} — tạm nghỉ"
#                     )
                time.sleep(30)
                continue

            if FRAME_SKIP > 1 and (frame_count % FRAME_SKIP != 0):
                continue

            processed_count += 1

            if current_time - fps_start_time >= 300:
                elapsed = current_time - fps_start_time
                fps = processed_count / elapsed if elapsed > 0 else 0
                pass  # print(f"⚡ FPS xử lý trung bình: {fps:.1f} (5 phút qua)")
                fps_start_time = current_time
                processed_count = 0

            resized_frame = cv2.resize(frame, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))

            # Debug: log tình trạng frame định kỳ
            if DEBUG_MODE and (frame_count % (DEBUG_LOG_INTERVAL * 10) == 0):
                now_vn = datetime.now(VN_TZ)
                n_pending = len(pending_observation)
                n_completed = len(completed_cycle_ids)
                n_tracking = len(known_ids)
                pass  # print(
#                     f"\n{'='*60}"
#                     f"\n📊 [{now_vn.strftime('%H:%M:%S')}] Frame #{frame_count}"
#                     f" | Tracking: {n_tracking} xe"
#                     f" | Pending: {n_pending}"
#                     f" | Completed: {n_completed}"
#                     f" | Night: {is_night_mode}"
#                     f"\n{'='*60}"
#                 )

            if LINE_CROSSING_ENABLED and gate_line is not None and GATE_LINE_DEBUG_DRAW:
                cv2.line(
                    resized_frame,
                    tuple(np.round(gate_line_start).astype(int)),
                    tuple(np.round(gate_line_end).astype(int)),
                    (255, 255, 0),
                    2,
                )

            inference_frame = resized_frame
            # Detect chế độ ban đêm (ảnh tối / IR)
            effective_conf = DETECTION_CONF
            gray_check = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            brightness = gray_check.mean()
            is_night_mode = brightness < NIGHT_BRIGHTNESS_THRESHOLD

            if is_night_mode:
                # Night confidence boost — tăng conf để giảm false positive ban đêm
                effective_conf = min(DETECTION_CONF + NIGHT_CONF_BOOST, 0.50)
                if ENABLE_NIGHT_ENHANCE:
                    # Dùng CLAHE tăng tương phản trên kênh L (LAB)
                    lab = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2LAB)
                    l_ch, a_ch, b_ch = cv2.split(lab)
                    clahe = cv2.createCLAHE(
                        clipLimit=CLAHE_CLIP_LIMIT,
                        tileGridSize=(CLAHE_TILE_GRID, CLAHE_TILE_GRID),
                    )
                    l_ch = clahe.apply(l_ch)
                    lab = cv2.merge([l_ch, a_ch, b_ch])
                    inference_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            results = model.track(
                inference_frame,
                persist=True,
                classes=TARGET_CLASSES,
                verbose=False,
                conf=effective_conf,
                tracker=TRACKER_CONFIG,
            )

            has_boxes = results is not None and len(results[0].boxes) > 0
            has_ids = has_boxes and results[0].boxes.id is not None

            current_active_ids = set()
            # Ngưỡng bbox area tối thiểu theo chế độ sáng/tối
            min_box_area = MIN_BOX_AREA_NIGHT if is_night_mode else MIN_BOX_AREA_DAY

            if has_ids:
                boxes = results[0].boxes
                for j in range(len(boxes)):
                    tracker_id = int(boxes.id[j])

                    box = boxes.xyxy[j].tolist()
                    x1_box, y1_box, x2_box, y2_box = box
                    box_area_check = (x2_box - x1_box) * (y2_box - y1_box)
                    class_id = int(boxes.cls[j])
                    class_name = results[0].names.get(class_id, "?")
                    conf_val = float(boxes.conf[j]) if hasattr(boxes, 'conf') else 0.0

                    class_min_conf = CLASS_MIN_CONF.get(class_id)
                    if class_min_conf is not None and conf_val < class_min_conf:
                        if DEBUG_MODE:
                            pass  # print(
#                                 f"  ❌ SKIP ID:{tracker_id} [{class_name}] conf thấp "
#                                 f"({conf_val:.2f} < {class_min_conf:.2f})"
#                             )
                        continue

                    class_area_map = CLASS_MIN_BOX_AREA_NIGHT if is_night_mode else CLASS_MIN_BOX_AREA_DAY
                    class_min_box_area = max(min_box_area, class_area_map.get(class_id, min_box_area))

                    # Lọc bbox quá nhỏ (noise / false positive)
                    if box_area_check < class_min_box_area:
                        if DEBUG_MODE:
                            pass  # print(
#                                 f"  ❌ SKIP ID:{tracker_id} bbox quá nhỏ "
#                                 f"[{class_name}] ({box_area_check:.0f} < {class_min_box_area:.0f}px²)"
#                             )
                        continue

                    current_active_ids.add(tracker_id)
                    anchor_x, anchor_y = get_anchor_point(box)

                    # Consecutive detection filter — xe phải được detect đủ frame
                    track_detection_count[tracker_id] = track_detection_count.get(tracker_id, 0) + 1
                    min_consecutive_needed = CLASS_MIN_CONSECUTIVE_DETECTIONS.get(
                        class_id,
                        MIN_CONSECUTIVE_DETECTIONS,
                    )
                    if track_detection_count[tracker_id] < min_consecutive_needed:
                        if DEBUG_MODE:
                            pass  # print(
#                                 f"  ⏳ ID:{tracker_id} [{class_name}] chờ detect "
#                                 f"{track_detection_count[tracker_id]}/{min_consecutive_needed} frame "
#                                 f"conf={conf_val:.2f} area={box_area_check:.0f}px "
#                                 f"pos=({anchor_x:.0f},{anchor_y:.0f})"
#                             )
                        # Vẫn track nhưng chưa đủ frame để bắt đầu observation
                        if tracker_id not in known_ids:
                            known_ids.add(tracker_id)
                            vehicle_first_x[tracker_id] = anchor_x
                            vehicle_first_y[tracker_id] = anchor_y
                        continue

                    if tracker_id not in known_ids:
                        known_ids.add(tracker_id)
                        vehicle_first_x[tracker_id] = anchor_x
                        vehicle_first_y[tracker_id] = anchor_y

                    # Xe tải cỡ lớn: dùng multi-anchor để không bỏ sót
                    large_vehicle = is_large_vehicle(box, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)

                    inside_outer = cv2.pointPolygonTest(
                        POLYGON_PTS,
                        (anchor_x, anchor_y),
                        False,
                    ) >= 0
                    inside_inner = cv2.pointPolygonTest(
                        inner_polygon_pts,
                        (anchor_x, anchor_y),
                        False,
                    ) >= 0
                    inside_exit_gate = (
                        (
                            cv2.pointPolygonTest(exit_gate_pts, (anchor_x, anchor_y), False) >= 0
                            or abs(anchor_x - exit_gate_top_x) <= EXIT_GATE_X_MARGIN
                        )
                        and anchor_y <= exit_gate_y_limit
                    )

                    # Multi-anchor fallback cho xe lớn: nếu single anchor miss,
                    # thử các anchor khác (center, top-center)
                    if large_vehicle and not inside_outer:
                        for alt_ax, alt_ay in get_multi_anchors(box):
                            if cv2.pointPolygonTest(POLYGON_PTS, (alt_ax, alt_ay), False) >= 0:
                                inside_outer = True
                                if DEBUG_MODE:
                                    pass  # print(
#                                         f"  🚛 ID:{tracker_id} XE LỚN multi-anchor hit outer "
#                                         f"alt=({alt_ax:.0f},{alt_ay:.0f})"
#                                     )
                                break
                    if large_vehicle and not inside_inner and inside_outer:
                        for alt_ax, alt_ay in get_multi_anchors(box):
                            if cv2.pointPolygonTest(inner_polygon_pts, (alt_ax, alt_ay), False) >= 0:
                                inside_inner = True
                                break
                    if large_vehicle and not inside_exit_gate:
                        for alt_ax, alt_ay in get_multi_anchors(box):
                            chk = (
                                (
                                    cv2.pointPolygonTest(exit_gate_pts, (alt_ax, alt_ay), False) >= 0
                                    or abs(alt_ax - exit_gate_top_x) <= EXIT_GATE_X_MARGIN
                                )
                                and alt_ay <= exit_gate_y_limit
                            )
                            if chk:
                                inside_exit_gate = True
                                if DEBUG_MODE:
                                    pass  # print(
#                                         f"  🚛 ID:{tracker_id} XE LỚN multi-anchor hit exit_gate "
#                                         f"alt=({alt_ax:.0f},{alt_ay:.0f})"
#                                     )
                                break

                    if tracker_id not in inside_history:
                        inside_history[tracker_id] = collections.deque(maxlen=INNER_WINDOW_SIZE)
                    inside_history[tracker_id].append(1 if inside_inner else 0)

                    log_interval = 5 if large_vehicle else DEBUG_LOG_INTERVAL
                    if DEBUG_MODE and (frame_count % log_interval == 0):
                        conf_val = float(boxes.conf[j]) if hasattr(boxes, 'conf') else 0.0
                        zones = []
                        if inside_outer: zones.append("outer")
                        if inside_inner: zones.append("inner")
                        if inside_exit_gate: zones.append("exit_gate")
                        zone_str = ",".join(zones) if zones else "none"
                        obs_status = "obs" if tracker_id in pending_observation else (
                            "done" if tracker_id in completed_cycle_ids else "idle"
                        )
                        
                        frame_area = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE
                        area_ratio = box_area_check / max(frame_area, 1)
                        size_tag = "LỚN" if large_vehicle else "thường"
                        
                        pass  # print(
#                             f"  🚗 ID:{tracker_id} [{class_name} - {size_tag}] "
#                             f"pos=({anchor_x:.0f},{anchor_y:.0f}) "
#                             f"conf={conf_val:.2f} "
#                             f"area_ratio={area_ratio:.2f} ({box_area_check:.0f}px) "
#                             f"zone=[{zone_str}] status={obs_status}"
#                         )

                    # Xe đã cảnh báo nhưng đang đi ra khỏi vùng -> cho phép chu kỳ cảnh báo mới
                    if not inside_outer and tracker_id in completed_cycle_ids:
                        completed_cycle_ids.discard(tracker_id)
                        completed_stop_position.pop(tracker_id, None)

                    track_state = gate_track_state.setdefault(
                        tracker_id,
                        {
                            "prev_anchor": None,
                            "prev_side": 0,
                            "has_been_inside_outer": False,
                            "has_been_near_gate": False,
                            "line_cross_direction": None,
                            "line_cross_time": 0.0,
                        },
                    )
                    track_state["has_been_inside_outer"] = track_state["has_been_inside_outer"] or inside_outer
                    track_state["has_been_near_gate"] = track_state["has_been_near_gate"] or inside_exit_gate

                    line_direction = None
                    if LINE_CROSSING_ENABLED and gate_line is not None:
                        curr_anchor = (anchor_x, anchor_y)
                        if track_state["prev_anchor"] is not None:
                            crossed, prev_side, curr_side, prev_dist, curr_dist, cross_move = detect_line_crossing(
                                track_state["prev_anchor"],
                                curr_anchor,
                                gate_line_start,
                                gate_line_end,
                                GATE_LINE_SIDE_TOLERANCE,
                                GATE_LINE_MIN_CROSS_DISTANCE,
                            )
                            if crossed:
                                context_ok = True
                                if GATE_LINE_REQUIRE_CONTEXT:
                                    context_ok = (
                                        inside_outer
                                        or inside_exit_gate
                                        or track_state["has_been_inside_outer"]
                                        or track_state["has_been_near_gate"]
                                    )
                                if context_ok and gate_line_inside_side != 0:
                                    if curr_side == gate_line_inside_side and prev_side != gate_line_inside_side:
                                        line_direction = "VÀO XƯỞNG"
                                    elif prev_side == gate_line_inside_side and curr_side != gate_line_inside_side:
                                        line_direction = "RA KHỎI XƯỞNG"

                                    if line_direction is not None:
                                        track_state["line_cross_direction"] = line_direction
                                        track_state["line_cross_time"] = current_time
                        else:
                            curr_side, _ = get_line_side(
                                curr_anchor,
                                gate_line_start,
                                gate_line_end,
                                GATE_LINE_SIDE_TOLERANCE,
                            )
                            track_state["prev_side"] = curr_side
                        track_state["prev_anchor"] = curr_anchor

                    # Nếu đang quan sát thì cập nhật mẫu chuyển động liên tục
                    if tracker_id in pending_observation:
                        obs = pending_observation[tracker_id]
                        obs["last_seen"] = current_time
                        obs["last_x"] = anchor_x
                        obs["last_y"] = anchor_y
                        obs["x_samples"].append(anchor_x)
                        obs["y_samples"].append(anchor_y)
                        obs["in_outer"].append(1 if inside_outer else 0)
                        obs["in_inner"].append(1 if inside_inner else 0)
                        obs["last_frame"] = frame.copy()
                        update_observation_conf(obs, class_id, conf_val, box_area_check)
                        update_snapshot_candidate(obs, frame, box, inside_outer, inside_inner)
                        # Snapshot hold: nếu đã vào hold mode, tiếp tục gom frame tốt
                        if obs.get("snapshot_hold_started_at") is not None:
                            update_snapshot_hold_candidate(obs, frame, box)
                    else:
                        if tracker_id in completed_cycle_ids:
                            # Cho phép tái quan sát nếu xe đã dừng rồi bắt đầu di chuyển lại
                            stop_pos = completed_stop_position.get(tracker_id)
                            if stop_pos is not None:
                                move_from_stop = math.hypot(
                                    anchor_x - stop_pos[0],
                                    anchor_y - stop_pos[1],
                                )
                                if move_from_stop >= STATIONARY_DISPLACEMENT_THRESHOLD * 0.6:
                                    # Xe đã di chuyển đủ xa từ vị trí dừng → cho phép quan sát lại
                                    completed_cycle_ids.discard(tracker_id)
                                    completed_stop_position.pop(tracker_id, None)
                                    pass  # print(
#                                         f"🔄 ID:{tracker_id} xe di chuyển lại từ vị trí dừng "
#                                         f"(move={move_from_stop:.0f}px) → tái quan sát"
#                                     )
                                else:
                                    continue
                            else:
                                continue

                        inner_count = sum(inside_history.get(tracker_id, []))
                        start_reason = None
                        entry_x = vehicle_first_x.get(tracker_id, anchor_x)
                        entry_y = vehicle_first_y.get(tracker_id, anchor_y)
                        entry_move = math.hypot(anchor_x - entry_x, anchor_y - entry_y)
                        if line_direction is not None:
                            start_reason = "line_gate"
                        elif inner_count >= MIN_INSIDE_FRAMES:
                            start_reason = "inner"
                        elif inside_exit_gate and not inside_outer:
                            start_reason = "exit_gate"
                        elif (
                            class_id in FAST_VEHICLE_CLASS_IDS
                            and not large_vehicle
                            and (inside_inner or inside_outer or inside_exit_gate)
                            and entry_move >= FAST_VEHICLE_MIN_MOVE
                        ):
                            start_reason = "fast_vehicle"
                            if DEBUG_MODE:
                                pass  # print(
#                                     f"  ⚡ ID:{tracker_id} [{class_name}] fast trigger "
#                                     f"move={entry_move:.1f}px"
#                                 )
                        elif large_vehicle and (inside_outer or inside_exit_gate):
                            # Xe tải cỡ lớn: chỉ cần 1 anchor hit zone là đủ trigger
                            start_reason = "large_vehicle"
                            if DEBUG_MODE:
                                pass  # print(
#                                     f"  🚛 ID:{tracker_id} XE LỚN trigger observation "
#                                     f"outer={inside_outer} exit_gate={inside_exit_gate}"
#                                 )

                        if start_reason is not None:
                            if DEBUG_MODE:
                                pass  # print(
#                                     f"  🟢 ID:{tracker_id} BắT ĐẦU QUAN SÁT [{start_reason}] "
#                                     f"tại ({anchor_x:.0f},{anchor_y:.0f})"
#                                 )
                            pending_observation[tracker_id] = create_observation(
                                class_name,
                                current_time,
                                anchor_x,
                                anchor_y,
                                frame,
                                inside_outer,
                                inside_inner,
                                box,
                                start_reason,
                                is_large=large_vehicle,
                            )
                            if start_reason == "fast_vehicle":
                                prepend_observation_entry(
                                    pending_observation[tracker_id],
                                    entry_x,
                                    entry_y,
                                    inside_outer,
                                    inside_inner,
                                )
                            update_observation_conf(
                                pending_observation[tracker_id],
                                class_id,
                                conf_val,
                                box_area_check,
                            )
                            update_snapshot_candidate(
                                pending_observation[tracker_id],
                                frame,
                                box,
                                inside_outer,
                                inside_inner,
                            )

                    if line_direction is not None:
                        obs = pending_observation.get(tracker_id)
                        if obs is None:
                            pending_observation[tracker_id] = create_observation(
                                class_name,
                                current_time,
                                anchor_x,
                                anchor_y,
                                frame,
                                inside_outer,
                                inside_inner,
                                box,
                                "line_gate",
                                is_large=large_vehicle,
                            )
                            obs = pending_observation[tracker_id]
                            update_observation_conf(obs, class_id, conf_val, box_area_check)
                            update_snapshot_candidate(obs, frame, box, inside_outer, inside_inner)
                        obs["line_direction"] = line_direction
                        obs["line_cross_time"] = current_time
                        obs["line_cross_anchor"] = (anchor_x, anchor_y)
                        obs["line_cross_distance"] = math.hypot(
                            anchor_x - obs["first_x"],
                            anchor_y - obs["first_y"],
                        )
                        obs["line_triggered"] = True

            # Cập nhật thời điểm nhìn thấy cuối cùng cho tất cả ID active
            for tid in current_active_ids:
                tracker_last_seen[tid] = current_time

            # track_detection_count được dọn dẹp bởi stale track cleanup bên dưới

            # Dọn dẹp trạng thái của các xe đã mất dấu quá lâu (chống chớp nháy đứt quãng 1-2 frame)
            stale_delay = effective_stale_track_sec
            force_evaluate_tids = set()
            for tid in list(tracker_last_seen.keys()):
                if tid not in current_active_ids and (current_time - tracker_last_seen[tid] > stale_delay):
                    tracker_last_seen.pop(tid, None)
                    known_ids.discard(tid)
                    inside_history.pop(tid, None)
                    completed_cycle_ids.discard(tid)
                    track_detection_count.pop(tid, None)
                    if tid in pending_observation:
                        force_evaluate_tids.add(tid)
                    else:
                        vehicle_first_x.pop(tid, None)
                        vehicle_first_y.pop(tid, None)
                        gate_track_state.pop(tid, None)
                        completed_stop_position.pop(tid, None)

            # Đánh giá các xe đã quan sát đủ thời gian
            for tid in list(pending_observation.keys()):
                obs = pending_observation[tid]
                elapsed = current_time - obs["first_time"]
                start_reason = obs.get("start_reason", "inner")
                obs_is_large = obs.get("is_large_vehicle", False)
                
                # Cần đủ list samples mới quyết định
                # Line crossing cho phép ít samples hơn (xe RA nhanh chỉ xuất hiện 1-2 frame)
                # Xe lớn dễ bị mất track → giảm min samples xuống 2
                # Xe lớn đã timeout: cho phép 1 sample (sẽ dùng timeout fallback)
                if start_reason in ("line_gate", "fast_vehicle") or obs_is_large:
                    min_samples_needed = 2
                else:
                    min_samples_needed = MIN_MOTION_SAMPLES
                timed_out = elapsed >= MAX_OBSERVATION_SEC
                if obs_is_large and timed_out:
                    min_samples_needed = 1
                if len(obs["y_samples"]) < min_samples_needed or len(obs["x_samples"]) < min_samples_needed:
                    if tid not in force_evaluate_tids:
                        continue

                # Thời gian quan sát tối thiểu: đợi xe đi sâu hơn trước khi đánh giá
                # Line-gate không cần đợi (xe đã ở ranh giới cổng)
                # Xe lớn: giảm thời gian chờ vì hay bị mất track nhanh
                MIN_OBSERVE_TIME = 2.0  # giây
                LARGE_OBSERVE_TIME = 1.0  # giây — xe lớn cần phản ứng nhanh hơn
                if start_reason == "fast_vehicle":
                    effective_observe_time = FAST_VEHICLE_OBSERVE_TIME
                else:
                    effective_observe_time = LARGE_OBSERVE_TIME if obs_is_large else MIN_OBSERVE_TIME
                if start_reason != "line_gate" and elapsed < effective_observe_time:
                    if tid not in force_evaluate_tids:
                        continue

                outer_ratio = sum(obs["in_outer"]) / len(obs["in_outer"]) if len(obs["in_outer"]) > 0 else 0
                inner_ratio = sum(obs["in_inner"]) / len(obs["in_inner"]) if len(obs["in_inner"]) > 0 else 0

                # === SNAPSHOT HOLD: xe lớn RA đang chờ gom frame tốt hơn ===
                if obs.get("snapshot_hold_started_at") is not None:
                    hold_elapsed = current_time - obs["snapshot_hold_started_at"]
                    is_force = tid in force_evaluate_tids
                    if hold_elapsed < LARGE_VEHICLE_SNAPSHOT_HOLD_SEC and not is_force:
                        continue
                    huong_di = obs["snapshot_hold_direction"]
                    stats = obs["snapshot_hold_stats"] or {}
                    is_timeout = False
                    if DEBUG_MODE:
                        hold_score = obs.get("snapshot_hold_best_score", float("-inf"))
                        has_hold_frame = obs.get("snapshot_hold_best_frame") is not None
                        pass  # print(
#                             f"  📷 ID:{tid} Snapshot hold hoàn tất ({hold_elapsed:.1f}s) "
#                             f"hold_frame={'CÓ' if has_hold_frame else 'KHÔNG'} "
#                             f"hold_score={hold_score:.3f}"
#                         )
                else:
                    huong_di = None
                    stats = {
                        "method": "none",
                        "net_flow": 0.0, "total_flow": 0.0,
                        "in_ratio": 0.0, "out_ratio": 0.0,
                        "net_dx": 0.0, "total_move_x": 0.0,
                        "left_ratio": 0.0, "right_ratio": 0.0,
                        "net_dy": 0.0, "total_move": 0.0,
                        "up_ratio": 0.0, "down_ratio": 0.0,
                    }
                    if obs.get("line_direction"):
                        line_direction = obs["line_direction"]
                        since_line = current_time - obs.get("line_cross_time", current_time)
                        cross_anchor = obs.get("line_cross_anchor") or (obs["first_x"], obs["first_y"])
                        post_cross_move = math.hypot(
                            obs["last_x"] - cross_anchor[0],
                            obs["last_y"] - cross_anchor[1],
                        )
                        if line_direction == "VÀO XƯỞNG":
                            if (
                                since_line < GATE_LINE_IN_SNAPSHOT_DELAY_SEC
                                and post_cross_move < GATE_LINE_IN_MIN_AFTER_CROSS_MOVE
                                and tid not in force_evaluate_tids
                            ):
                                continue
                        elif line_direction == "RA KHỎI XƯỞNG":
                            out_min_move = GATE_LINE_OUT_MIN_AFTER_CROSS_MOVE
                            if obs_is_large:
                                out_min_move = max(out_min_move, 25.0)
                            if post_cross_move < out_min_move and tid not in force_evaluate_tids:
                                continue
                        huong_di = line_direction
                        stats = {
                            "method": "line_crossing",
                            "line_post_move": post_cross_move,
                            "line_since": since_line,
                            "line_cross_distance": obs.get("line_cross_distance", 0.0),
                            "net_flow": 0.0, "total_flow": 0.0,
                            "in_ratio": 0.0, "out_ratio": 0.0,
                            "net_dx": obs["last_x"] - obs["first_x"],
                            "total_move_x": 0.0,
                            "left_ratio": 0.0, "right_ratio": 0.0,
                            "net_dy": obs["last_y"] - obs["first_y"],
                            "total_move": 0.0,
                            "up_ratio": 0.0, "down_ratio": 0.0,
                        }
                    else:
                        huong_di, stats = classify_direction(
                            list(obs["x_samples"]),
                            list(obs["y_samples"]),
                            min_samples=min_samples_needed,
                        )
                        if DEBUG_MODE and huong_di is not None:
                            pass  # print(
#                                 f"  🧭 ID:{tid} classify_direction [{stats.get('method')}] "
#                                 f"→ {huong_di} | flow={stats.get('net_flow',0):.1f} "
#                                 f"dy={stats.get('net_dy',0):.1f} dx={stats.get('net_dx',0):.1f} "
#                                 f"samples={len(obs['y_samples'])}"
#                             )

                    is_timeout = False
                    if huong_di is None:
                        if elapsed < MAX_OBSERVATION_SEC and tid not in force_evaluate_tids:
                            continue

                        is_timeout = True
                        if DEBUG_MODE:
                            pass  # print(
#                                 f"  ⏰ ID:{tid} TIMEOUT ({elapsed:.1f}s) "
#                                 f"samples={len(obs['y_samples'])} reason={start_reason}"
#                             )
                        entry_x = vehicle_first_x.get(tid, obs["first_x"])
                        entry_y = vehicle_first_y.get(tid, obs["first_y"])
                        current_x = obs["last_x"]
                        current_y = obs["last_y"]
                        delta_x = current_x - entry_x
                        delta = current_y - entry_y
                        delta_flow = flow_progress(current_x, current_y) - flow_progress(entry_x, entry_y)

                        displacement = math.hypot(delta_x, delta)
                        obs_total_move = 0.0
                        if len(obs["x_samples"]) >= 2 and len(obs["y_samples"]) >= 2:
                            xs_arr = np.asarray(obs["x_samples"], dtype=np.float32)
                            ys_arr = np.asarray(obs["y_samples"], dtype=np.float32)
                            obs_total_move = float(
                                np.sum(np.abs(np.diff(xs_arr))) + np.sum(np.abs(np.diff(ys_arr)))
                            )
                        obs_total_move = max(obs_total_move, displacement)
                        displacement_ratio = displacement / max(obs_total_move, 1.0)

                        if DEBUG_MODE:
                            pass  # print(
#                                 f"    📍 disp={displacement:.1f}px total_move={obs_total_move:.1f}px "
#                                 f"ratio={displacement_ratio:.2f} "
#                                 f"d_flow={delta_flow:.1f} dx={delta_x:.1f} dy={delta:.1f}"
#                             )

                        if obs_is_large:
                            if displacement_ratio < STATIONARY_OSCILLATION_RATIO and obs_total_move > 5.0:
                                huong_di = "ĐANG DỪNG"
                            elif displacement < STATIONARY_DISPLACEMENT_THRESHOLD and obs_total_move < STATIONARY_TOTAL_MOVE_THRESHOLD:
                                huong_di = "ĐANG DỪNG"
                            elif delta_flow >= STATIONARY_MIN_FLOW_DELTA:
                                huong_di = "VÀO XƯỞNG"
                            elif delta_flow <= -STATIONARY_MIN_FLOW_DELTA:
                                huong_di = "RA KHỎI XƯỞNG"
                            elif delta > STATIONARY_MIN_ENTRY_DELTA:
                                huong_di = "VÀO XƯỞNG"
                            elif delta < -STATIONARY_MIN_ENTRY_DELTA:
                                huong_di = "RA KHỎI XƯỞNG"
                            elif ENABLE_X_DIRECTION_FALLBACK and abs(delta_x) >= STATIONARY_MIN_ENTRY_DELTA_X:
                                if IN_X_SIGN < 0:
                                    huong_di = "VÀO XƯỞNG" if delta_x < 0 else "RA KHỎI XƯỞNG"
                                else:
                                    huong_di = "VÀO XƯỞNG" if delta_x > 0 else "RA KHỎI XƯỞNG"
                            else:
                                huong_di = "ĐANG DỪNG"
                        else:
                            if displacement_ratio < STATIONARY_OSCILLATION_RATIO and obs_total_move > 5.0:
                                huong_di = "ĐANG DỪNG"
                            elif displacement < STATIONARY_DISPLACEMENT_THRESHOLD or obs_total_move < STATIONARY_TOTAL_MOVE_THRESHOLD:
                                huong_di = "ĐANG DỪNG"
                            elif delta_flow >= STATIONARY_MIN_FLOW_DELTA:
                                huong_di = "VÀO XƯỞNG"
                            elif delta_flow <= -STATIONARY_MIN_FLOW_DELTA:
                                huong_di = "RA KHỎI XƯỞNG"
                            elif delta > STATIONARY_MIN_ENTRY_DELTA:
                                huong_di = "VÀO XƯỞNG"
                            elif delta < -STATIONARY_MIN_ENTRY_DELTA:
                                huong_di = "RA KHỎI XƯỞNG"
                            elif ENABLE_X_DIRECTION_FALLBACK and abs(delta_x) >= STATIONARY_MIN_ENTRY_DELTA_X:
                                if IN_X_SIGN < 0:
                                    huong_di = "VÀO XƯỞNG" if delta_x < 0 else "RA KHỎI XƯỞNG"
                                else:
                                    huong_di = "VÀO XƯỞNG" if delta_x > 0 else "RA KHỎI XƯỞNG"
                            else:
                                huong_di = "ĐANG DỪNG"

                        stats = {
                            "method": "timeout",
                            "entry_x": entry_x, "entry_y": entry_y,
                            "current_x": current_x, "current_y": current_y,
                            "delta_x": delta_x, "delta": delta,
                            "delta_flow": delta_flow,
                        }

                    # === XE LỚN RA: TRIGGER SNAPSHOT HOLD thay vì gửi ngay ===
                    if (
                        obs_is_large
                        and huong_di == "RA KHỎI XƯỞNG"
                        and obs.get("snapshot_hold_started_at") is None
                    ):
                        obs["snapshot_hold_direction"] = huong_di
                        obs["snapshot_hold_started_at"] = current_time
                        obs["snapshot_hold_stats"] = stats
                        last_box = obs.get("last_box")
                        if last_box and obs.get("last_frame") is not None:
                            update_snapshot_hold_candidate(obs, obs["last_frame"], last_box)
                        pass  # print(
#                             f"  📷 ID:{tid} XE LỚN RA → snapshot hold "
#                             f"{LARGE_VEHICLE_SNAPSHOT_HOLD_SEC}s "
#                             f"(chờ chụp frame rõ biển số)"
#                         )
                        continue

                # Log thông tin
                flow_signal = stats.get("delta_flow", stats.get("net_flow", 0.0))
                if start_reason in ("exit_gate", "large_vehicle"):
                    if huong_di != "RA KHỎI XƯỞNG" and start_reason == "exit_gate":
                        del pending_observation[tid]
                        inside_history.pop(tid, None)
                        continue
                    if start_reason == "exit_gate" and abs(flow_signal) < EXIT_GATE_MIN_FLOW_DELTA:
                        del pending_observation[tid]
                        inside_history.pop(tid, None)
                        continue

                # Log thông tin chỉ cho alerts thực sự (VÀO/RA)
                if not is_timeout and huong_di in ("VÀO XƯỞNG", "RA KHỎI XƯỞNG"):
                    pass  # print(
#                         f"📍 ID:{tid} [{stats.get('method', 'n/a')}] "
#                         f"→ {huong_di}"
#                     )

                # Kiểm tra ratio CHỈ cho timeout fallback
                # Fast path: xe đã được xác nhận ở inner zone (inside_consecutive >= MIN_INSIDE_FRAMES)
                #            VÀ có hướng rõ ràng từ movement → không cần check lại ratio
                # Timeout path: cần kiểm tra xe có thực sự ở trong vùng không
                if is_timeout:
                    # Xe lớn và exit_gate: giảm yêu cầu ratio vì bbox lớn hay tràn ra ngoài polygon
                    if start_reason in ("exit_gate", "large_vehicle"):
                        min_outer_ratio = 0.0
                        min_inner_ratio = 0.0
                    else:
                        min_outer_ratio = MIN_OUTER_RATIO
                        min_inner_ratio = MIN_INNER_RATIO * 0.5
                    if outer_ratio < min_outer_ratio or inner_ratio < min_inner_ratio:
                        del pending_observation[tid]
                        inside_history.pop(tid, None)
                        continue

                # Cooldown theo từng label (VÀO / RA / generic), không chặn chéo hướng
                if huong_di == "ĐANG DỪNG":
                    obs["stop_count"] = obs.get("stop_count", 0) + 1

                    # Max stop count — sau N lần ĐANG DỪNG, xóa observation vĩnh viễn
                    if obs["stop_count"] >= MAX_STOP_COUNT or tid in force_evaluate_tids:
                        if obs["stop_count"] <= 1:
                            pass  # print(
#                                 f"🔇 ID:{tid} XE {huong_di} tại "
#                                 f"({obs['last_x']:.0f},{obs['last_y']:.0f}) → chỉ log"
#                             )
                        else:
                            pass  # print(
#                                 f"🔇 ID:{tid} XE {huong_di} lần {obs['stop_count']}/{MAX_STOP_COUNT}"
#                                 f" → dừng theo dõi"
#                             )
                        completed_cycle_ids.add(tid)
                        # Lưu vị trí dừng để cho phép tái quan sát nếu xe di chuyển lại
                        completed_stop_position[tid] = (obs["last_x"], obs["last_y"])
                        del pending_observation[tid]
                        inside_history.pop(tid, None)
                        continue

                    # Chỉ log console, KHÔNG gửi Zalo — tránh spam khi xe đỗ
                    pass  # print(
#                         f"🔇 ID:{tid} XE {huong_di} lần {obs['stop_count']}/{MAX_STOP_COUNT} tại "
#                         f"({obs['last_x']:.0f},{obs['last_y']:.0f}) → reset window, chờ tiếp"
#                     )

                    obs["stop_alert_sent"] = True
                    vehicle_first_x[tid] = obs["last_x"]
                    vehicle_first_y[tid] = obs["last_y"]
                    reset_observation_window(
                        obs,
                        current_time,
                        obs["last_x"],
                        obs["last_y"],
                        bool(obs["in_outer"] and obs["in_outer"][-1]),
                        bool(obs["in_inner"] and obs["in_inner"][-1]),
                    )
                    continue

                # --- Edge Fragment Filter (Chống bóng ma ở rìa màn hình) ---
                # Nếu xe xuất hiện lần đầu ở tít lề ngoài (y < 150) và được nhận diện là RA KHỎI XƯỞNG,
                # đó chắc chắn là xe cũ đang đi ra bị đứt ID, chặn ngay lập tức.
                # NGOẠI TRỪ xe tải cỡ lớn (bbox chiếm >25% frame) — xe thật sự đang ra.
                edge_y_threshold = LARGE_VEHICLE_EDGE_FRAGMENT_Y if start_reason == "large_vehicle" else 150.0
                if huong_di == "RA KHỎI XƯỞNG" and obs["first_y"] < edge_y_threshold and start_reason != "large_vehicle":
                    pass  # print(
#                         f"🚫 ID:{tid} XE RA KHỎI XƯỞNG bị chặn vì xuất hiện ở mép ảnh "
#                         f"(first_y={obs['first_y']:.0f}px < {edge_y_threshold:.0f}px) → bóng ma đứt ID"
#                     )
                    completed_cycle_ids.add(tid)
                    del pending_observation[tid]
                    inside_history.pop(tid, None)
                    continue

                alert_key = f"{huong_di}:{tid}"
                last_sent = last_alert_time_by_label.get(alert_key, 0.0)
                since_last = current_time - last_sent
                if since_last <= ALERT_COOLDOWN_SEC:
                    remain = ALERT_COOLDOWN_SEC - since_last
                    pass  # print(
#                         f"⏳ ID:{tid} XE {huong_di} đang cooldown "
#                         f"còn {remain:.1f}s → BỎ QUA"
#                     )
                    completed_cycle_ids.add(tid)
                    del pending_observation[tid]
                    inside_history.pop(tid, None)
                    continue

                # --- Anti-spam Layer 2: Global direction cooldown (bất kể tracker ID) ---
                # Chỉ block nếu alert mới GẦN vị trí alert trước (cùng xe bị đổi ID)
                # 2 xe khác nhau ở vị trí khác nhau sẽ không bị chặn
                last_dir_sent = last_alert_time_by_direction.get(huong_di, 0.0)
                since_dir = current_time - last_dir_sent
                if since_dir <= GLOBAL_DIRECTION_COOLDOWN_SEC:
                    # Kiểm tra khoảng cách với alert trước cùng hướng
                    alert_x_check, alert_y_check = obs["last_x"], obs["last_y"]
                    near_prev_alert = False
                    for prev_dir, prev_x, prev_y, prev_t in recent_alert_positions:
                        if prev_dir == huong_di and (current_time - prev_t) <= GLOBAL_DIRECTION_COOLDOWN_SEC:
                            dist_to_prev = math.hypot(alert_x_check - prev_x, alert_y_check - prev_y)
                            if dist_to_prev < SPATIAL_DEDUP_DISTANCE * 1.5:
                                near_prev_alert = True
                                break
                    if near_prev_alert:
                        remain_dir = GLOBAL_DIRECTION_COOLDOWN_SEC - since_dir
                        pass  # print(
#                             f"⏳ ID:{tid} XE {huong_di} global direction cooldown "
#                             f"còn {remain_dir:.1f}s (gần vị trí alert trước) → BỎ QUA"
#                         )
                        completed_cycle_ids.add(tid)
                        del pending_observation[tid]
                        inside_history.pop(tid, None)
                        continue
                    else:
                        pass  # print(
#                             f"✅ ID:{tid} XE {huong_di} trong cooldown nhưng vị trí khác → cho phép"
#                         )

                # --- Anti-spam Layer 3: Spatial dedup (cùng vị trí + cùng hướng) ---
                alert_x, alert_y = obs["last_x"], obs["last_y"]
                # Dọn entries cũ
                recent_alert_positions[:] = [
                    entry for entry in recent_alert_positions
                    if current_time - entry[3] <= SPATIAL_DEDUP_SEC
                ]
                spatial_dup = False
                for prev_dir, prev_x, prev_y, prev_t in recent_alert_positions:
                    if prev_dir == huong_di:
                        dist = math.hypot(alert_x - prev_x, alert_y - prev_y)
                        if dist < SPATIAL_DEDUP_DISTANCE:
                            spatial_dup = True
                            pass  # print(
#                                 f"🔕 ID:{tid} XE {huong_di} spatial dedup "
#                                 f"(dist={dist:.0f}px < {SPATIAL_DEDUP_DISTANCE:.0f}px, "
#                                 f"since={current_time - prev_t:.1f}s) → BỎ QUA"
#                             )
                            break
                if spatial_dup:
                    completed_cycle_ids.add(tid)
                    del pending_observation[tid]
                    inside_history.pop(tid, None)
                    continue

                # Tất cả anti-spam checks đã pass — gửi cảnh báo
                # Xe chạy nhanh (fast_vehicle) và xe lớn chỉ có 2-3 frame detect
                # → conf tự nhiên thấp → bypass conf gate để không chặn nhầm
                conf_gate_bypass = start_reason in ("fast_vehicle", "large_vehicle")
                if not conf_gate_bypass and not passes_alert_conf_gate(obs):
                    min_conf = ALERT_MIN_CONF.get(obs.get("class_id"), 0.0)
                    pass  # print(
#                         f"🚫 ID:{tid} XE {huong_di} bị chặn trước khi gửi "
#                         f"vì confidence cao nhất chỉ {obs.get('max_conf', 0.0):.2f} "
#                         f"< {min_conf:.2f}"
#                     )
                    completed_cycle_ids.add(tid)
                    del pending_observation[tid]
                    inside_history.pop(tid, None)
                    continue

                last_alert_time_by_label[alert_key] = current_time
                last_alert_time_by_direction[huong_di] = current_time
                recent_alert_positions.append((huong_di, alert_x, alert_y, current_time))

                now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"alerts/{now_str}_{tid}.jpg"
                snapshot_frame = get_snapshot_frame(obs, frame, huong_di)
                cv2.imwrite(filename, snapshot_frame)
                pass  # print(f"📸 XE {huong_di} - Đã chụp: {filename}")

                threading.Thread(
                    target=send_alert,
                    args=(filename, f"🚨 CẢNH BÁO: XE {huong_di}"),
                    daemon=True,
                ).start()

                completed_cycle_ids.add(tid)
                del pending_observation[tid]

            # Dọn dẹp tài nguyên của các xe đã force evaluate (mất track)
            for tid in force_evaluate_tids:
                vehicle_first_x.pop(tid, None)
                vehicle_first_y.pop(tid, None)
                gate_track_state.pop(tid, None)

        except Exception as e:
            print(f"⚠️ System Error Loop: {e}")
            import traceback

            traceback.print_exc()
            time.sleep(2)


if __name__ == "__main__":
    main()
