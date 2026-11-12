import os
import time
import sqlite3
import threading
from datetime import datetime, timezone

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "eyecare.db")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

# ---------------- Existing EyeCare algorithm configuration ----------------
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_CENTER = 468

EAR_THRESHOLD = 0.21

# Blink duration limits in seconds
MIN_BLINK_DURATION = 0.05
MAX_BLINK_DURATION = 0.60
OBSERVATION_PERIOD_SECONDS = 60
KNOWN_IPD_CM = 6.3
CALIBRATION_DISTANCE_CM = 50.0
UNSAFE_DISTANCE_CM = 40.0

HEALTHY_BLINK_RATE = 17.0
IDEAL_DISTANCE_CM = 60.0
MAX_SESSION_MINUTES = 120.0

W_BLINK = 0.30
W_CLOSURE = 0.25
W_DISTANCE = 0.25
W_SCREEN_TIME = 0.20

ESI_ALERT_THRESHOLD = 60.0

mp_face_mesh = mp.solutions.face_mesh
processor = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

lock = threading.Lock()
state = {
    "active": False,
    "session_start": None,
    "last_frame_time": None,
    "frame_counter": 0,
    "blink_count": 0,
    "eyes_closed_start": None,
    "total_eye_closure_time": 0.0,
    "focal_length": None,
    "last_pixel_ipd": None,
    "last_ear": None,
    "last_distance": None,
    "last_seen": None,
    "last_alert_time": 0.0,
    "time_at_unsafe_distance": 0.0,
}


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                blink_count INTEGER NOT NULL,
                average_blink_rate REAL NOT NULL,
                final_esi REAL NOT NULL,
                average_distance REAL,
                unsafe_distance_seconds REAL NOT NULL
            )
        """)
        con.commit()


def euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float)))


def eye_aspect_ratio(landmarks, eye_indices, width, height):
    pts = [(landmarks[i].x * width, landmarks[i].y * height) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = pts
    vertical = euclidean(p2, p6) + euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    return vertical / (2.0 * horizontal) if horizontal else 0.0


def get_pixel_ipd(landmarks, width, height):
    left = landmarks[LEFT_IRIS_CENTER]
    right = landmarks[RIGHT_IRIS_CENTER]
    return euclidean(
        (left.x * width, left.y * height),
        (right.x * width, right.y * height),
    )



def compute_esi(blink_rate, closure_ratio, distance_cm, session_minutes):

    # Initial observation period: 60 seconds
    OBSERVATION_PERIOD_SECONDS = 60

    observation_complete = (
    session_minutes * 60 >= OBSERVATION_PERIOD_SECONDS
    ) 

    # Blink factor
    # Do not penalize zero blinks during initial observation.
    if not observation_complete:
        blink_factor = 0.0

    elif blink_rate >= HEALTHY_BLINK_RATE:
        blink_factor = 0.0

    else:
        blink_factor = min(
            100.0,
            (1.0 - blink_rate / HEALTHY_BLINK_RATE) * 100.0
        )

    # Eye closure factor
    closure_factor = min(
        100.0,
        max(0.0, (closure_ratio - 0.05) / 0.20 * 100.0)
    )

    # Viewing distance factor
    if distance_cm is None or distance_cm >= IDEAL_DISTANCE_CM:
        distance_factor = 0.0
    else:
        distance_factor = min(
            100.0,
            max(0.0, (1.0 - distance_cm / IDEAL_DISTANCE_CM) * 100.0)
        )

    # Screen time factor
    screen_time_factor = min(
        100.0,
        (session_minutes / MAX_SESSION_MINUTES) * 100.0
    )

    # Weighted ESI
    score = (
        W_BLINK * blink_factor
        + W_CLOSURE * closure_factor
        + W_DISTANCE * distance_factor
        + W_SCREEN_TIME * screen_time_factor
    )

    return round(score, 1), {
        "blink": round(blink_factor, 1),
        "closure": round(closure_factor, 1),
        "distance": round(distance_factor, 1),
        "screen_time": round(screen_time_factor, 1)
    }



def recommendation(esi):
    if esi < 30:
        return "Your measured indicators currently suggest low strain. Keep taking regular breaks."
    if esi < ESI_ALERT_THRESHOLD:
        return "Some strain indicators are increasing. Relax your eyes and blink naturally."
    return "High strain indicators detected. Take a break and follow the 20-20-20 rule."


def snapshot():
    now = time.time()
    started = state["session_start"] or now
    elapsed = max(0.0, now - started)
    minutes = max(elapsed / 60.0, 1e-6)
    blink_rate = state["blink_count"] / minutes
    closed_total = state["total_eye_closure_time"]
    if state["eyes_closed_start"] is not None:
        closed_total += max(0.0, now - state["eyes_closed_start"])
    closure_ratio = min(1.0, closed_total / max(elapsed, 1e-6))
    esi, factors = compute_esi(
        blink_rate, closure_ratio, state["last_distance"], elapsed / 60.0
    )
    distance = state["last_distance"]
    observation_elapsed = min(
    elapsed, OBSERVATION_PERIOD_SECONDS
    )

    observation_remaining = max(
    0, OBSERVATION_PERIOD_SECONDS - elapsed
    )

    observation_complete = (
    elapsed >= OBSERVATION_PERIOD_SECONDS
    )
    return {
        "active": state["active"],
        "elapsed_seconds": int(elapsed),
        "elapsed_display": f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}",
        "blink_count": state["blink_count"],
        "blink_rate": round(blink_rate, 1),
        "ear": round(state["last_ear"], 3) if state["last_ear"] is not None else None,
        "distance_cm": round(distance, 1) if distance is not None else None,
        "calibrated": state["focal_length"] is not None,
        "esi": esi,
        "level": "Low" if esi < 30 else ("Moderate" if esi < 60 else "High"),
        "recommendation": recommendation(esi),
        "factors": factors,
        "unsafe_distance_seconds": round(state["time_at_unsafe_distance"], 1),
        "face_detected": state["last_seen"] is not None and (now - state["last_seen"] < 2.0),
        "observation_complete": observation_complete,
        "observation_elapsed": round(observation_elapsed, 1),
        "observation_remaining": round(observation_remaining, 1),
        "observation_progress": round(
        (observation_elapsed / OBSERVATION_PERIOD_SECONDS) * 100, 1),
    }


def reset_state():
    now = time.time()
    state.update({
        "active": True,
        "session_start": now,
        "last_frame_time": now,
        "frame_counter": 0,
        "blink_count": 0,
        "eyes_closed_start": None,
        "total_eye_closure_time": 0.0,
        "focal_length": None,
        "last_pixel_ipd": None,
        "last_ear": None,
        "last_distance": None,
        "last_seen": None,
        "last_alert_time": 0.0,
        "time_at_unsafe_distance": 0.0,
    })


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/history")
def history():
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM sessions ORDER BY id DESC LIMIT 10"
        ).fetchall()
    return jsonify([dict(row) for row in rows])


@app.post("/api/start")
def start_session():
    with lock:
        reset_state()
    return jsonify({"ok": True, "message": "Monitoring session started."})


@app.post("/api/frame")
def process_frame():
    if not state["active"]:
        return jsonify({"ok": False, "error": "Start a monitoring session first."}), 400

    uploaded = request.files.get("frame")
    if uploaded is None:
        return jsonify({"ok": False, "error": "No frame was received."}), 400

    raw = np.frombuffer(uploaded.read(), np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"ok": False, "error": "Could not decode webcam frame."}), 400

    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with lock:
        now = time.time()
        previous = state["last_frame_time"] or now
        dt = min(max(now - previous, 0.0), 1.0)
        state["last_frame_time"] = now

        result = processor.process(rgb)
        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_EAR, width, height)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_EAR, width, height)
            avg_ear = (left_ear + right_ear) / 2.0
            state["last_ear"] = avg_ear
            state["last_seen"] = now

            eyes_closed = avg_ear < EAR_THRESHOLD
            if eyes_closed:

               if state["eyes_closed_start"] is None:
                  state["eyes_closed_start"] = now

            else:
                if state["eyes_closed_start"] is not None:
                    closure_duration = now - state["eyes_closed_start"]
                    # Count only closures within the normal blink-duration window
                    if MIN_BLINK_DURATION <= closure_duration <= MAX_BLINK_DURATION:
                         state["blink_count"] += 1
                    state["total_eye_closure_time"] += closure_duration
                    state["eyes_closed_start"] = None


            pixel_ipd = get_pixel_ipd(landmarks, width, height)
            state["last_pixel_ipd"] = pixel_ipd
            if state["focal_length"] is not None and pixel_ipd > 0:
                distance = (KNOWN_IPD_CM * state["focal_length"]) / pixel_ipd
                state["last_distance"] = distance
                if distance < UNSAFE_DISTANCE_CM:
                    state["time_at_unsafe_distance"] += dt
            else:
                state["last_distance"] = None
        else:
            state["last_ear"] = None
            state["last_distance"] = None

        data = snapshot()

    return jsonify({"ok": True, **data})


@app.post("/api/calibrate")
def calibrate():
    with lock:
        pixel_ipd = state["last_pixel_ipd"]
        if pixel_ipd is None or pixel_ipd <= 0:
            return jsonify({"ok": False, "error": "No face/iris detected. Look at the camera and try again."}), 400
        state["focal_length"] = (pixel_ipd * CALIBRATION_DISTANCE_CM) / KNOWN_IPD_CM
    return jsonify({
        "ok": True,
        "message": f"Calibration complete at {CALIBRATION_DISTANCE_CM:.0f} cm.",
        "calibrated": True,
    })


@app.post("/api/stop")
def stop_session():
    with lock:
        if not state["active"]:
            return jsonify({"ok": True, "message": "No active session."})
        data = snapshot()
        started = state["session_start"] or time.time()
        average_distance = state["last_distance"]
        started_text = datetime.fromtimestamp(started, timezone.utc).isoformat()
        with sqlite3.connect(DB_PATH) as con:
            con.execute("""
                INSERT INTO sessions
                (started_at, duration_seconds, blink_count, average_blink_rate,
                 final_esi, average_distance, unsafe_distance_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                started_text, data["elapsed_seconds"], data["blink_count"],
                data["blink_rate"], data["esi"], average_distance,
                state["time_at_unsafe_distance"]
            ))
            con.commit()
        state["active"] = False
    return jsonify({"ok": True, "summary": data})


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


init_db()

if __name__ == "__main__":
    # Local development only. Render uses Gunicorn via Procfile.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=True, threaded=True)
