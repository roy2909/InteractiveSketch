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
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.detection_confidence, self.tracking_confidence)
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
                    self.mpDraw.draw_landmarks(
                        img, handlandmarks, self.mpHands.HAND_CONNECTIONS)
        return img

    def getPostion(self, img, num=0, draw=True):
        """
        Get the positions of hand landmarks.

        Parameters:
        - img: The input image.
        - num: Index of the hand to process (default is 0).
        - draw: Whether to draw the hand landmarks on the image.

        Returns:
        - A list of landmark positions (x, y).
        """
        landmarklist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[num]
            for lm in myHand.landmark:
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarklist.append((cx, cy))

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return landmarklist

    def getUpFingers(self, img):
        """
        Determine which fingers are up.

        Parameters:
        - img: The input image.

        Returns:
        - A list of booleans indicating which fingers are up.
        """
        pos = self.getPostion(img, draw=False)
        self.upfingers = []
        if pos:
            # Thumb
            self.upfingers.append(
                (pos[4][1] < pos[3][1] and (pos[5][0] - pos[4][0] > 10)))
            # Index finger
            self.upfingers.append(
                (pos[8][1] < pos[7][1] and pos[7][1] < pos[6][1]))
            # Middle finger
            self.upfingers.append(
                (pos[12][1] < pos[11][1] and pos[11][1] < pos[10][1]))
            # Ring finger
            self.upfingers.append(
                (pos[16][1] < pos[15][1] and pos[15][1] < pos[14][1]))
            # Pinky finger
            self.upfingers.append(
                (pos[20][1] < pos[19][1] and pos[19][1] < pos[18][1]))
        return self.upfingers

    def get_distance(self, point1_idx, point2_idx, frame, draw=False):
        """
        Calculate the distance between two landmarks.

        Parameters:
        - point1_idx: Index of the first landmark.
        - point2_idx: Index of the second landmark.
        - frame: The input image.
        - draw: Whether to draw the distance measurement on the image.

        Returns:
        - The distance between the two points.
        """
        landmarklist = self.getPostion(frame, draw=False)
        if landmarklist:
            x1, y1 = landmarklist[point1_idx]
            x2, y2 = landmarklist[point2_idx]
            distance = math.hypot(x2 - x1, y2 - y1)
            if draw:
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(frame, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 5, (255, 0, 255), cv2.FILLED)
            return distance
        return None
