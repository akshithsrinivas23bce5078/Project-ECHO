import sys
import os
import cv2
import numpy as np
import pytest
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vision.webcam_analyzer import WebcamAnalyzer

class TestWebcamAnalyzer:
    def test_aspect_ratio_computation(self):
        analyzer = WebcamAnalyzer()
        
        # Fake a perfect square (EAR)
        landmarks = np.array([
            [0, 50],   # left
            [50, 0],   # top left
            [100, 0],  # top right
            [150, 50], # right
            [100, 100],# bottom right
            [50, 100]  # bottom left
        ])
        
        ear = analyzer._compute_aspect_ratio(landmarks, [0, 1, 2, 3, 4, 5])
        # |p2-p6| + |p3-p5| = 100 + 100 = 200
        # 2 * |p1-p4| = 2 * 150 = 300
        # EAR = 200 / 300 = 0.666...
        assert abs(ear - 0.666) < 0.01

        # Fake MAR (Mouth Aspect Ratio)
        landmarks_mar = np.array([
            [0, 50],   # left
            [100, 50], # right
            [50, 0],   # top
            [50, 100]  # bottom
        ])
        mar = analyzer._compute_aspect_ratio(landmarks_mar, [0, 1, 2, 3])
        # width = 100, height = 100
        # MAR = 100 / 100 = 1.0
        assert abs(mar - 1.0) < 0.01
