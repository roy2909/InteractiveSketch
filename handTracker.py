import mediapipe as mp
import numpy as np
import cv2
import math

class HandTracker():
    def __init__(self, mode=False, maxHands=2, detection_confidence=0.5, tracking_confidence=0.5):
        """
        Initialize the HandTracker object.
        
        Parameters:
        - mode: Static image mode or not.
        - maxHands: Maximum number of hands to detect.
        - detection_confidence: Minimum detection confidence.
        - tracking_confidence: Minimum tracking confidence.
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detection_confidence, self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
