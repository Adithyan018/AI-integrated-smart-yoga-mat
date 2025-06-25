import cv2
import mediapipe as mp
import math
import configparser
import os

# Key angle names
ANGLE_KEYS = [
    'left_elbow', 'right_elbow',
    'left_shoulder', 'right_shoulder',
    'left_knee', 'right_knee'
]

# Utility to calculate angle between three landmarks
def calculate_angle(a, b, c):
    x1, y1 = a.x, a.y
    x2, y2 = b.x, b.y
    x3, y3 = c.x, c.y
    angle = math.degrees(
        math.atan2(y3 - y2, x3 - x2) -
        math.atan2(y1 - y2, x1 - x2)
    )
    return angle + 360 if angle < 0 else angle

# Capture angles + frame for new pose via camera feed
def measure_pose():
    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose.Pose(static_image_mode=False,
                                     min_detection_confidence=0.3,
                                     model_complexity=1)
    mp_draw = mp.solutions.drawing_utils

    captured_angles = None
    captured_clean_frame = None
    print("Press 's' to capture angles when you're in the desired pose, or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # keep a copy of the clean frame before drawing
        clean_frame = frame.copy()

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(image_rgb)
        angles = {}

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # compute each key angle
            angles['left_elbow'] = calculate_angle(
                lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value],
                lm[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value],
                lm[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
            )
            angles['right_elbow'] = calculate_angle(
                lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value],
                lm[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value],
                lm[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
            )
            angles['left_shoulder'] = calculate_angle(
                lm[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value],
                lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value],
                lm[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
            )
            angles['right_shoulder'] = calculate_angle(
                lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value],
                lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value],
                lm[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]
            )
            angles['left_knee'] = calculate_angle(
                lm[mp.solutions.pose.PoseLandmark.LEFT_HIP.value],
                lm[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value],
                lm[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]
            )
            angles['right_knee'] = calculate_angle(
                lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value],
                lm[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value],
                lm[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]
            )

            # Draw landmarks
            mp_draw.draw_landmarks(frame, results.pose_landmarks,
                                   mp.solutions.pose.POSE_CONNECTIONS)

            # Overlay each angle on its joint
            for key, val in angles.items():
                part = key.split('_')[0].upper()
                lm_idx = getattr(mp.solutions.pose.PoseLandmark, f"{part}_{key.split('_')[1].upper()}").value
                x = int(lm[lm_idx].x * frame.shape[1])
                y = int(lm[lm_idx].y * frame.shape[0])
                cv2.putText(frame, f"{key}:{int(val)}", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            # Guidance message
            cv2.putText(frame, "Press 's' to save pose or 'q' to quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Measure New Pose', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and angles:
            captured_angles = angles.copy()
            captured_clean_frame = clean_frame
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_angles, captured_clean_frame

def add_pose_to_ini(angles, clean_frame):
    if not angles:
        print("No angles captured, exiting.")
        return

    pose_name = input("Enter new pose name: ").strip()
    if not pose_name:
        print("Pose name cannot be empty.")
        return

    # Write to pose_details.ini
    config_path = 'pose_details.ini'
    config = configparser.ConfigParser()
    if os.path.exists(config_path):
        config.read(config_path)

    if pose_name in config.sections():
        print(f"Pose '{pose_name}' already exists.")
        return

    config[pose_name] = {}
    for key, val in angles.items():
        config[pose_name][f"{key}_min"] = str(val - 10)
        config[pose_name][f"{key}_max"] = str(val + 10)

    with open(config_path, 'w') as f:
        config.write(f)
    print(f"Pose '{pose_name}' added to {config_path}.")

    # Save captured frame to static/poses/
    img_dir = os.path.join('static', 'poses')
    os.makedirs(img_dir, exist_ok=True)
    filename = f"{pose_name.lower().replace(' ', '_')}.png"
    img_path = os.path.join(img_dir, filename)
    cv2.imwrite(img_path, clean_frame)
    print(f"Clean pose image saved to {img_path}.")

if __name__ == '__main__':
    angles, clean_frame = measure_pose()
    add_pose_to_ini(angles, clean_frame)
