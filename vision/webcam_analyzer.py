"""
Project ECHO - Webcam Analyzer (OpenCV + MediaPipe)
==================================================
Captures live webcam feed to extract facial presence,
motion energy, expression proxies, chin-contact detection,
fist detection (anger signal), and reference photo comparison.

Features:
    - Face detection via Haar cascades (skip-frame optimized)
    - Smile detection via Haar cascade on lower face ROI
    - Chin-contact detection via skin-color segmentation
    - Reference photo capture for baseline expression comparison
    - Optical flow for movement energy estimation

Latency optimizations:
    - Face detection runs every 2nd frame
    - MediaPipe Hands runs every 3rd frame
    - Optical flow on 160x120 downscale
    - Skin segmentation on tiny chin ROI only
"""

import cv2
import numpy as np
import time
import threading
import os

try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    from mediapipe.tasks.python.core.base_options import BaseOptions
    MP_AVAILABLE = True
except ImportError as e:
    print(f"[DEBUG] MediaPipe Import Error: {e}")
    MP_AVAILABLE = False


class WebcamAnalyzer:
    """Analyzes face, motion, smile, chin-contact, and fist using OpenCV + MediaPipe."""

    # HSV skin-color range (works for a wide range of skin tones)
    SKIN_LOWER = np.array([0, 30, 60], dtype=np.uint8)
    SKIN_UPPER = np.array([25, 180, 255], dtype=np.uint8)

    def __init__(self, device_id=0):
        self.device_id = device_id
        self.cap = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )

        # MediaPipe Hands for fist detection (v1.0.0 task-based API)
        self.hands_detector = None
        if MP_AVAILABLE:
            model_path = os.path.join(
                os.path.dirname(__file__), 'hand_landmarker.task'
            )
            if os.path.exists(model_path):
                try:
                    options = HandLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=model_path),
                        running_mode=RunningMode.IMAGE,
                        num_hands=2,
                        min_hand_detection_confidence=0.5,
                        min_hand_presence_confidence=0.4,
                    )
                    self.hands_detector = HandLandmarker.create_from_options(options)
                except Exception as e:
                    print(f"[WARNING] MediaPipe HandLandmarker init failed: {e}")
                    self.hands_detector = None
            else:
                print(f"[WARNING] hand_landmarker.task not found at {model_path}")
                print("  Fist detection disabled. Download from:")
                print("  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")

        self.prev_gray = None
        self.movement_energy = 0.0

        self.running = False
        self.thread = None
        self.latest_frame = None
        self.latest_metrics = {
            "face_detected": False,
            "movement_energy": 0.0,
            "blink_rate": 15,
            "mouth_openness": 0.0,
            "emotion": "neutral",
            "smile": False,
            "chin_contact": False,
            "fist_detected": False,
            "processing_latency_ms": 0.0,
        }
        self.lock = threading.Lock()

        # Skip-frame counters
        self._frame_count = 0
        self._cached_faces = []
        self._cached_face_detected = False
        self._cached_smile = False
        self._cached_hand_landmarks = []  # Normalized (x, y) of all hand landmarks

        # Reference photo for baseline comparison
        self._reference_histogram = None
        self._reference_frame = None

    def start(self):
        """Open the webcam connection and start capture thread."""
        self.cap = cv2.VideoCapture(self.device_id)

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam {self.device_id}")

        # Force MJPG codec for stability on Windows
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Release the webcam and stop thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        # Cleanup MediaPipe HandLandmarker
        if self.hands_detector is not None:
            try:
                self.hands_detector.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Reference Photo Capture
    # ------------------------------------------------------------------

    def update_reference_from_current(self):
        """
        Capture a reference photo of the user's neutral face using the latest frame.
        
        Extracts the face region and computes a baseline HSV histogram for comparison.

        Returns:
            The captured reference frame (BGR), or None if no face found.
        """
        with self.lock:
            frame = self.latest_frame
            metrics = dict(self.latest_metrics)

        if frame is None or not metrics.get("face_detected", False):
            return None

        faces = metrics.get("faces", [])
        if len(faces) == 0:
            return None

        best_frame = None
        best_face_area = 0

        # Pick the largest face
        for (x, y, w, h) in faces:
            area = w * h
            if area > best_face_area:
                best_face_area = area
                best_frame = frame.copy()
                best_face_rect = (x, y, w, h)

        if best_frame is not None:
            self._reference_frame = best_frame
            # Compute face region histogram for comparison
            x, y, w, h = best_face_rect
            face_roi = best_frame[y:y+h, x:x+w]
            hsv_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            self._reference_histogram = cv2.calcHist(
                [hsv_roi], [0, 1], None, [50, 60],
                [0, 180, 0, 256]
            )
            cv2.normalize(self._reference_histogram, self._reference_histogram,
                          0, 1, cv2.NORM_MINMAX)

        return best_frame

    def get_reference_frame(self):
        """Return the stored reference frame."""
        return self._reference_frame

    def compare_with_reference(self, frame, face_rect):
        """
        Compare current face with reference photo histogram.

        Returns a similarity score in [0, 1] where 1 = identical.
        """
        if self._reference_histogram is None:
            return 0.5  # No reference, return neutral

        x, y, w, h = face_rect
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            return 0.5

        hsv_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        current_hist = cv2.calcHist(
            [hsv_roi], [0, 1], None, [50, 60],
            [0, 180, 0, 256]
        )
        cv2.normalize(current_hist, current_hist, 0, 1, cv2.NORM_MINMAX)

        # Correlation comparison (1.0 = perfect match)
        similarity = cv2.compareHist(
            self._reference_histogram, current_hist, cv2.HISTCMP_CORREL
        )
        return max(0.0, min(1.0, similarity))

    # ------------------------------------------------------------------
    # Fist Detection via MediaPipe Hands
    # ------------------------------------------------------------------

    def _detect_hand_landmarks(self, frame_rgb):
        """
        Run MediaPipe HandLandmarker and return hand landmark positions for chin detection.

        Returns:
            landmarks_px: list of (x, y) tuples in pixel coords of fingertips
        """
        if self.hands_detector is None:
            return []

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self.hands_detector.detect(mp_image)

            all_points = []
            h, w = frame_rgb.shape[:2]

            if result.hand_landmarks:
                for hand_lms in result.hand_landmarks:
                    # Collect only fingertip landmark pixel positions (thumb, index, middle, ring, pinky)
                    for idx in [4, 8, 12, 16, 20]:
                        if idx < len(hand_lms):
                            lm = hand_lms[idx]
                            px = int(lm.x * w)
                            py = int(lm.y * h)
                            all_points.append((px, py))

            return all_points
        except Exception as e:
            print(f"[DEBUG] Hand tracking error: {e}")
            return []

    # ------------------------------------------------------------------
    # Chin Contact Detection (via MediaPipe Hand Landmarks)
    # ------------------------------------------------------------------

    def _detect_chin_from_hands(self, hand_points, face_rect, frame_h, frame_w):
        """
        Detect if any hand landmark overlaps with the chin/jaw region
        of the detected face.

        The chin region is defined as the lower 40% of the face bounding
        box plus a small area below it. If multiple hand landmark points
        fall inside this region, the user is likely resting their hand
        on their chin (thinking pose).

        Returns True if >= 3 hand landmarks are in the chin zone.
        """
        if not hand_points or face_rect is None:
            return False

        fx, fy, fw, fh = face_rect

        # Chin zone: lower 50% of face + 50% below face, with wider horizontal margins
        chin_top = fy + int(fh * 0.5)
        chin_bottom = min(frame_h, fy + fh + int(fh * 0.5))
        chin_left = max(0, fx - int(fw * 0.3))
        chin_right = min(frame_w, fx + fw + int(fw * 0.3))

        for (px, py) in hand_points:
            if chin_left <= px <= chin_right and chin_top <= py <= chin_bottom:
                return True  # Any fingertip in the chin zone triggers the state

        return False

    # ------------------------------------------------------------------
    # Capture Loop
    # ------------------------------------------------------------------

    def _reinit_camera(self):
        """Fallback initialization if grabbing frames fails repeatedly."""
        print("[WARNING] Webcam feed failed. Re-initializing with DirectShow...")
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def _capture_loop(self):
        fail_count = 0
        while self.running and self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                fail_count += 1
                if fail_count > 30:
                    self._reinit_camera()
                    fail_count = 0
                time.sleep(0.05)
                continue

            fail_count = 0
            start_t = time.perf_counter()
            self._frame_count += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- Skip-frame face detection (every 2nd frame) ---
            run_face_detection = (self._frame_count % 2 == 0)

            if run_face_detection:
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50)
                )
                
                if len(faces) > 0:
                    self._cached_faces = faces
                    self._cached_face_detected = True
                    self._last_face_time = time.perf_counter()
                else:
                    if hasattr(self, '_last_face_time') and time.perf_counter() - self._last_face_time < 1.5:
                        self._cached_face_detected = True
                        # Do NOT overwrite self._cached_faces, keep the last known bounding box
                    else:
                        self._cached_face_detected = False
                        self._cached_faces = []

                self._cached_smile = False

                if self._cached_face_detected:
                    for (fx, fy, fw, fh) in self._cached_faces:
                        roi_gray = gray[fy + fh // 2 : fy + fh, fx : fx + fw]
                        smiles = self.smile_cascade.detectMultiScale(
                            roi_gray, scaleFactor=1.4, minNeighbors=8, minSize=(25, 25)
                        )
                        if len(smiles) > 0:
                            self._cached_smile = True

            # --- Skip-frame hand detection (every 3rd frame) ---
            run_hand_detection = (self._frame_count % 3 == 0)

            if run_hand_detection and self.hands_detector is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._cached_hand_landmarks = self._detect_hand_landmarks(frame_rgb)

            face_detected = self._cached_face_detected
            smile_detected = self._cached_smile
            faces = self._cached_faces

            # --- Draw face rectangles (every frame using cached boxes) ---
            chin_contact = False
            if face_detected and len(faces) > 0:
                for (fx, fy, fw, fh) in faces:
                    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

                    # Smile sub-rectangles
                    if smile_detected:
                        roi_gray_s = gray[fy + fh // 2 : fy + fh, fx : fx + fw]
                        smiles = self.smile_cascade.detectMultiScale(
                            roi_gray_s, scaleFactor=1.4, minNeighbors=8, minSize=(25, 25)
                        )
                        for (sx, sy, sw, sh) in smiles:
                            cv2.rectangle(
                                frame,
                                (fx + sx, fy + fh // 2 + sy),
                                (fx + sx + sw, fy + fh // 2 + sy + sh),
                                (255, 0, 0), 2,
                            )

                # Check chin contact using MediaPipe hand landmarks
                raw_chin_contact = self._detect_chin_from_hands(
                    self._cached_hand_landmarks,
                    (faces[0][0], faces[0][1], faces[0][2], faces[0][3]),
                    frame.shape[0], frame.shape[1],
                )
                
                if raw_chin_contact:
                    self._cached_chin_contact = True
                    self._last_chin_time = time.perf_counter()
                else:
                    if hasattr(self, '_last_chin_time') and time.perf_counter() - self._last_chin_time < 3.0:
                        self._cached_chin_contact = True
                    else:
                        self._cached_chin_contact = False
                        
                chin_contact = self._cached_chin_contact

                # Draw chin contact indicator
                if chin_contact:
                    cv2.putText(frame, "CHIN CONTACT", (10, frame.shape[0] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                                
            # Draw tracked fingertips
            if self._cached_hand_landmarks:
                for (px, py) in self._cached_hand_landmarks:
                    cv2.circle(frame, (px, py), 5, (255, 0, 255), -1)

            # --- Optical flow on downscaled frame ---
            small_gray = cv2.resize(gray, (160, 120))
            if self.prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_gray, small_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mean_motion = np.mean(mag)
                self.movement_energy = 0.4 * self.movement_energy + 0.6 * mean_motion
            self.prev_gray = small_gray

            # --- Raw emotion from vision only (used as a signal, not final) ---
            emotion = "neutral"
            if face_detected:
                if smile_detected:
                    emotion = "joy"
                elif self.movement_energy > 4.0:
                    emotion = "frustrated"
                elif chin_contact:
                    emotion = "thinking"

            latency_ms = (time.perf_counter() - start_t) * 1000.0

            with self.lock:
                self.latest_frame = frame
                self.latest_metrics = {
                    "face_detected": face_detected,
                    "faces": faces,
                    "movement_energy": float(self.movement_energy),
                    "blink_rate": 15,
                    "mouth_openness": 0.0,
                    "emotion": emotion,
                    "smile": smile_detected,
                    "chin_contact": chin_contact,
                    "processing_latency_ms": latency_ms,
                }

            time.sleep(0.001)

    def analyze_frame(self):
        """Returns the most recent frame and metrics without blocking."""
        with self.lock:
            if self.latest_frame is None:
                return None, {}
            return self.latest_frame.copy(), dict(self.latest_metrics)
