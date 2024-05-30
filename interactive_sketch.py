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
