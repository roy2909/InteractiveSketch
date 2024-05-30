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

    def draw(self, img, text_color=(255, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, thickness=2):
        """
        Draw the canvas on the image.
        
        Parameters:
        - img: The image on which to draw the canvas.
        - text_color: Color of the text on the canvas.
        - font: Font type for the text.
        - font_scale: Font scale for the text.
        - thickness: Thickness of the text.
        """
        bg_rec = img[self.y: self.y + self.h, self.x: self.x + self.w].astype(np.float32)
        bg_rec_color = np.ones(bg_rec.shape, dtype=np.float32) * np.array(self.color, dtype=np.float32)
        res = cv2.addWeighted(bg_rec, self.transparency, bg_rec_color, 1-self.transparency, 0, dtype=cv2.CV_32F)
        if res is not None:
            img[self.y: self.y + self.h, self.x: self.x + self.w] = res.astype(np.uint8)
            text_size = cv2.getTextSize(self.text, font, font_scale, thickness)
            text_pos = (int(self.x + self.w / 2 - text_size[0][0] / 2), int(self.y + self.h / 2 + text_size[0][1] / 2))
            cv2.putText(img, self.text, text_pos, font, font_scale, text_color, thickness)
    def contains(self, x, y):
        """
        Check if the point (x, y) is inside the canvas.
        
        Parameters:
        - x, y: Coordinates of the point.
        
        Returns:
        - True if the point is inside the canvas, False otherwise.
        """
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h

def initialize_hand_tracker():
    """
    Initialize the hand tracker.
    
    Returns:
    - HandTracker object with a detection confidence threshold of 0.8.
    """
    return HandTracker(detection_confidence=int(0.8))

def create_color_buttons():
    """
    Create a list of color buttons.
    
    Returns:
    - List of Canvas objects representing the color buttons.
    """
    return [
        Canvas(300, 0, 100, 100, (0, 0, 255)),
        Canvas(400, 0, 100, 100, (255, 0, 0)),    
        Canvas(500, 0, 100, 100, (0, 255, 0)),    
        Canvas(600, 0, 100, 100, (255, 255, 0)),  
        Canvas(700, 0, 100, 100, (255, 165, 0)),  
        Canvas(800, 0, 100, 100, (128, 0, 128)),  
        Canvas(900, 0, 100, 100, (255, 255, 255)),
        Canvas(1000, 0, 100, 100, (0, 0, 0), 'Eraser'), 
        Canvas(1100, 0, 100, 100, (100, 100, 100), 'Clear'), 
        Canvas(1200, 0, 100, 100, (255, 0, 0), 'Fill') 
    ]

def create_shape_buttons():
    """
    Create a list of shape buttons.
    
    Returns:
    - List of Canvas objects representing the shape buttons.
    """
    return [
        Canvas(1100, 100, 100, 100, (255, 255, 255), 'Circle'), # Circle shape
        Canvas(1100, 200, 100, 100, (255, 255, 255), 'Square')  # Square shape
    ]

def initialize_reference_shapes():
    """
    Initialize the reference shapes for template matching.
    
    Returns:
    - Dictionary of reference shapes with their template images and matching threshold.
    """
    return {
        'House': (cv2.imread('/home/rahulroy/comp_vision/src/virtual_painter/images_refernce/house.png', 0), 0.8),
        'Heart': (cv2.imread('/home/rahulroy/comp_vision/src/virtual_painter/images_refernce/heart.png', 0), 0.8),
        'Star': (cv2.imread('/home/rahulroy/comp_vision/src/virtual_painter/images_refernce/star.png', 0), 0.8),
        'Tree': (cv2.imread('/home/rahulroy/comp_vision/src/virtual_painter/images_refernce/tree.png', 0), 0.8),
        'Cloud': (cv2.imread('/home/rahulroy/comp_vision/src/virtual_painter/images_refernce/cloud.png', 0), 0.8),
        'Circle': (cv2.imread('/home/rahulroy/comp_vision/src/virtual_painter/images_refernce/circle.png', 0), 0.8)
    }
