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

    def findHands(self, img, draw=True):
        """
        Detect hands in the provided image and draw landmarks if draw is True.
        
        Parameters:
        - img: The input image.
        - draw: Whether to draw the hand landmarks on the image.
        
        Returns:
        - The image with or without hand landmarks.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handlandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlandmarks, self.mpHands.HAND_CONNECTIONS)
        return img
