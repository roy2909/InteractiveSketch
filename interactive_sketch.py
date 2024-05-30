import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict
from handTracker import HandTracker

class Canvas:
    def __init__(self, x, y, w, h, color, text='', transparency=0.5):
        """
        Initialize the Canvas object.
        
        Parameters:
        - x, y: Top left corner coordinates of the canvas.
        - w, h: Width and height of the canvas.
        - color: Color of the canvas.
        - text: Optional text to be displayed on the canvas.
        - transparency: Transparency level of the canvas.
        """
        self.x, self.y, self.w, self.h = x, y, w, h
        self.color, self.text, self.transparency = color, text, transparency
