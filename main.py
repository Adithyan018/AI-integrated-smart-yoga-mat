import os
import threading
import time

import cv2
import configparser
import mediapipe as mp
import serial  # Added for serial communication
from flask import (
    Flask, Response, render_template, session,
    redirect, url_for, request, jsonify
)
from pose_detection import YogaPoseRecognizer

# Mediapipe landmark enum shortcut
PoseLandmark = mp.solutions.pose.PoseLandmark

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'replace_with_secure_random_key'

# Initialize pose recognizer
recognizer = YogaPoseRecognizer(config_path='pose_details.ini')

# Serial port handle (will be set in serial_thread)
ser = None

def init_serial():
    """Attempt to open the serial port once."""
    global ser
    try:
        ser = serial.Serial('COM3', 9600, timeout=1)
        print('✅ Serial port opened successfully')
    except serial.SerialException as e:
        ser = None
        print(f'❌ Warning: could not open serial port: {e}')

def serial_thread():
    """Background thread: open serial and optionally read."""
    init_serial()
    while True:
        if ser and ser.is_open and ser.in_waiting:
            data = ser.readline()
            print(f"← From serial: {data!r}")
        time.sleep(0.1)

# Angle‐checking setup
ANGLE_KEYS = [
    'left_elbow', 'right_elbow',
    'left_shoulder', 'right_shoulder',
    'left_knee', 'right_knee'
]
LANDMARK_MAP = {
    'left_elbow': PoseLandmark.LEFT_ELBOW.value,
    'right_elbow': PoseLandmark.RIGHT_ELBOW.value,
    'left_shoulder': PoseLandmark.LEFT_SHOULDER.value,
    'right_shoulder': PoseLandmark.RIGHT_SHOULDER.value,
    'left_knee': PoseLandmark.LEFT_KNEE.value,
    'right_knee': PoseLandmark.RIGHT_KNEE.value
}

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        all_poses = list(recognizer.config.sections())
        session['today_poses']        = all_poses[:3]
        session['current_pose_index'] = 0
        session['correct_poses']      = 0
        session['wrong_poses']        = 0
        return redirect(url_for('pose', index=0))
    return render_template('start.html')

@app.route('/pose/<int:index>')
def pose(index):
    poses = session.get('today_poses', [])
    if index >= len(poses):
        return redirect(url_for('summary'))
    pose_name = poses[index]
    ref_img = url_for('static', filename=f'poses/{pose_name.lower()}.png')
    return render_template('pose.html', pose=pose_name, reference_image=ref_img)

@app.route('/video_feed')
def video_feed():
    idx = session.get('current_pose_index', 0)
    pose_name = session['today_poses'][idx]
    return Response(
        stream_frames(pose_name),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def stream_frames(pose_name):
    cap = cv2.VideoCapture(0)
    thresholds = recognizer.config[pose_name]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = recognizer.pose.process(rgb)
        all_green = True

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            recognizer.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, recognizer.mp_pose.POSE_CONNECTIONS
            )

            for key in ANGLE_KEYS:
                side, joint = key.split('_')
                side = side.upper()
                b = lm[LANDMARK_MAP[key]]

                # Select A and C landmarks for angle calculation
                if joint == 'elbow':
                    a = lm[getattr(PoseLandmark, f"{side}_SHOULDER")]
                    c = lm[getattr(PoseLandmark, f"{side}_WRIST")]
                elif joint == 'shoulder':
                    if side == 'LEFT':
                        a = lm[PoseLandmark.LEFT_ELBOW.value]
                        c = lm[PoseLandmark.LEFT_HIP.value]
                    else:
                        a = lm[PoseLandmark.RIGHT_HIP.value]
                        c = lm[PoseLandmark.RIGHT_ELBOW.value]
                else:  # knee
                    a = lm[getattr(PoseLandmark, f"{side}_HIP")]
                    c = lm[getattr(PoseLandmark, f"{side}_ANKLE")]

                angle = recognizer.calculate_angle(a, b, c)
                lo = float(thresholds[f"{key}_min"])
                hi = float(thresholds[f"{key}_max"])
                is_ok = (lo <= angle <= hi)
                if not is_ok:
                    all_green = False

                color = (0, 255, 0) if is_ok else (0, 0, 255)
                x, y = int(b.x * frame.shape[1]), int(b.y * frame.shape[0])
                cv2.putText(
                    frame, f"{key}:{int(angle)}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

            if all_green:
                recognizer.last_pose = pose_name

        ret2, buffer = cv2.imencode('.jpg', frame)
        if not ret2:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )

    cap.release()

@app.route('/pose_complete', methods=['POST'])
def pose_complete():
    idx = session.get('current_pose_index', 0)
    pose_name = session['today_poses'][idx]
    correct = (recognizer.last_pose == pose_name)

    if correct:
        session['correct_poses'] += 1
    else:
        session['wrong_poses'] += 1
        # send over serial on wrong pose
        if ser and ser.is_open:
            try:
                data = b'*0#'
                print(f"→ Writing to serial: {data}")
                ser.write(data)
                ser.flush()
            except Exception as e:
                print(f"❌ Serial write failed: {e}")
        else:
            print("⚠️ Serial port not open; skipping write")

    session['current_pose_index'] = idx + 1
    if session['current_pose_index'] < len(session['today_poses']):
        next_url = url_for('pose', index=session['current_pose_index'])
    else:
        next_url = url_for('summary')

    return jsonify({'next_url': next_url, 'correct': correct})

@app.route('/summary')
def summary():
    total   = len(session['today_poses'])
    correct = session['correct_poses']
    wrong   = session['wrong_poses']
    calories = correct * 5

    if   correct == 0: diet = 'High-protein, low-carb diet plan.'
    elif correct == 1: diet = 'Balanced diet with moderate calories.'
    elif correct == 2: diet = 'High-fiber, moderate-protein diet.'
    else:               diet = 'Low-fat, high-protein diet plan.'

    poses_all = list(recognizer.config.sections())
    next_pre  = poses_all[3:]

    return render_template(
        'summary.html',
        total=total, correct=correct, wrong=wrong,
        calories=calories, diet=diet, next_day_poses=next_pre
    )

if __name__ == '__main__':
    # Only start our serial thread in the actual running process
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        t = threading.Thread(target=serial_thread, daemon=True)
        t.start()

    # If you’d rather never fork for reloads, disable it:
    # app.run(debug=True, use_reloader=False)
    app.run(debug=True)
