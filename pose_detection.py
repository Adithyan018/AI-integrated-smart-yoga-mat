# pose_detection.py
import cv2
import mediapipe as mp
import math
import configparser

class YogaPoseRecognizer:
    def __init__(self, config_path='pose_details.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.pose = mp.solutions.pose.Pose(static_image_mode=False,
                                           min_detection_confidence=0.3,
                                           model_complexity=1)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.last_pose = ""

    def detect_pose(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            self.last_pose = self.classify_pose(results.pose_landmarks)
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return image

    def calculate_angle(self, a, b, c):
        x1, y1 = a.x, a.y
        x2, y2 = b.x, b.y
        x3, y3 = c.x, c.y
        angle = math.degrees(
            math.atan2(y3 - y2, x3 - x2) -
            math.atan2(y1 - y2, x1 - x2)
        )
        return angle + 360 if angle < 0 else angle

    def classify_pose(self, landmarks):
        lm = landmarks.landmark
        angles = {}
        angles['left_elbow'] = self.calculate_angle(
            lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            lm[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
            lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        )
        angles['right_elbow'] = self.calculate_angle(
            lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            lm[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        )
        angles['left_shoulder'] = self.calculate_angle(
            lm[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
            lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            lm[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        )
        angles['right_shoulder'] = self.calculate_angle(
            lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
            lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        )
        angles['left_knee'] = self.calculate_angle(
            lm[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            lm[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
            lm[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        )
        angles['right_knee'] = self.calculate_angle(
            lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
            lm[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
            lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        )

        for pose_name in self.config.sections():
            thresholds = self.config[pose_name]
            match = True
            for key, angle in angles.items():
                min_key = f"{key}_min"
                max_key = f"{key}_max"
                low = float(thresholds.get(min_key, -1e6))
                high = float(thresholds.get(max_key, 1e6))
                if not (low <= angle <= high):
                    match = False
                    break
            if match:
                return pose_name

        return 'Unknown Pose'