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
        bg_rec = img[self.y: self.y + self.h,
                     self.x: self.x + self.w].astype(np.float32)
        bg_rec_color = np.ones(bg_rec.shape, dtype=np.float32) * \
            np.array(self.color, dtype=np.float32)
        res = cv2.addWeighted(bg_rec, self.transparency,
                              bg_rec_color, 1-self.transparency, 0, dtype=cv2.CV_32F)
        if res is not None:
            img[self.y: self.y + self.h, self.x: self.x +
                self.w] = res.astype(np.uint8)
            text_size = cv2.getTextSize(self.text, font, font_scale, thickness)
            text_pos = (int(self.x + self.w / 2 - text_size[0][0] / 2), int(
                self.y + self.h / 2 + text_size[0][1] / 2))
            cv2.putText(img, self.text, text_pos, font,
                        font_scale, text_color, thickness)

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
        Canvas(1100, 100, 100, 100, (255, 255, 255), 'Circle'),  # Circle shape
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


def main():
    """
    Main function to run the virtual painter application.
    """
    # Initialize the hand tracker
    hand_tracker = initialize_hand_tracker()

    # Create UI elements
    color_button = Canvas(200, 0, 100, 100, (120, 255, 0), 'Colors')
    canvas_button = Canvas(50, 0, 100, 100, (0, 0, 255), 'Draw')
    canvas = Canvas(50, 120, 1020, 580, (255, 255, 255), transparency=0.6)
    colors = create_color_buttons()
    shapes = create_shape_buttons()

    # Initialize drawing canvas
    drawing_canvas = np.zeros((720, 1280, 3), np.uint8)

    # Initialize drawing parameters
    brush_color = (255, 0, 0)
    brush_size = 5
    eraser_size = 20
    previous_x, previous_y = 0, 0
    selected_shape, shape_locked, shape_position = None, False, None
    initial_distance, initial_size = 0, 0
    hide_canvas, hide_colors = True, True
    cooldown_counter = 20

    # Open the camera
    camera = cv2.VideoCapture(0)
    camera.set(3, 1920)
    camera.set(4, 1080)

    while True:
        # Manage cooldown counter to prevent rapid toggling
        if cooldown_counter:
            cooldown_counter -= 1

        # Read a frame from the camera
        ret, frame = camera.read()
        if not ret:
            break

        # Resize and flip the frame
        frame = cv2.resize(frame, (1280, 720))
        frame = cv2.flip(frame, 1)

        # Find hands and their positions
        hand_tracker.findHands(frame)
        positions = hand_tracker.getPostion(frame, draw=False)
        up_fingers = hand_tracker.getUpFingers(frame)

        if up_fingers:
            x, y = positions[8][0], positions[8][1]

            if up_fingers[1] and not canvas.contains(x, y):
                previous_x, previous_y = 0, 0

                # Handle shape selection
                if not hide_colors:
                    for shape_button in shapes:
                        if shape_button.contains(x, y):
                            selected_shape = shape_button.text
                            shape_button.transparency = 0
                            shape_locked = False
                            shape_position = None
                        else:
                            shape_button.transparency = 0.5

                # Handle color selection
                if not hide_colors:
                    for color_box in colors:
                        if color_box.contains(x, y):
                            brush_color = color_box.color
                            color_box.transparency = 0
                        else:
                            color_box.transparency = 0.5

                # Toggle color buttons visibility
                if color_button.contains(x, y) and not cooldown_counter:
                    cooldown_counter = 10
                    color_button.transparency = 0
                    hide_colors = not hide_colors
                    color_button.text = 'Colors' if hide_colors else 'Close'
                else:
                    color_button.transparency = 0.5

                # Toggle canvas visibility
                if canvas_button.contains(x, y) and not cooldown_counter:
                    cooldown_counter = 10
                    canvas_button.transparency = 0
                    hide_canvas = not hide_canvas
                    canvas_button.text = 'Canvas' if hide_canvas else 'Close'

                # Fill the drawn shapes with selected color
                if colors[-1].contains(x, y) and not cooldown_counter:
                    cooldown_counter = 10
                    line_mask = np.zeros(
                        drawing_canvas.shape[:2], dtype=np.uint8)
                    gray_canvas = cv2.cvtColor(
                        drawing_canvas, cv2.COLOR_BGR2GRAY)
                    _, line_mask = cv2.threshold(
                        gray_canvas, 10, 255, cv2.THRESH_BINARY)

                    contours, _ = cv2.findContours(cv2.cvtColor(
                        drawing_canvas, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if cv2.contourArea(contour) > 500:
                            mask = np.zeros(
                                drawing_canvas.shape[:2], dtype=np.uint8)
                            cv2.drawContours(
                                mask, [contour], -1, (255, 255, 255), -1)
                            mask = cv2.subtract(mask, line_mask)
                            color_image = np.zeros_like(drawing_canvas)
                            color_image[:] = brush_color
                            filled_shape = cv2.bitwise_and(
                                color_image, color_image, mask=mask)
                            mask_inv = cv2.bitwise_not(mask)
                            drawing_canvas_bg = cv2.bitwise_and(
                                drawing_canvas, drawing_canvas, mask=mask_inv)
                            drawing_canvas = cv2.add(
                                drawing_canvas_bg, filled_shape)

                # Clear the drawing canvas
                if colors[-2].contains(x, y) and not cooldown_counter:
                    cooldown_counter = 10
                    drawing_canvas = np.zeros((720, 1280, 3), np.uint8)

                # Set eraser mode
                if colors[-3].contains(x, y):
                    brush_color = (0, 0, 0)
                    colors[-3].transparency = 0
                else:
                    colors[-3].transparency = 0.5

            elif up_fingers[1] and not up_fingers[2]:
                # Draw on the canvas with brush or eraser
                if canvas.contains(x, y) and not hide_canvas:
                    if previous_x == 0 and previous_y == 0:
                        previous_x, previous_y = positions[8]
                    if brush_color == (0, 0, 0):
                        cv2.line(drawing_canvas, (previous_x, previous_y),
                                 positions[8], brush_color, eraser_size)
                    else:
                        cv2.line(drawing_canvas, (previous_x, previous_y),
                                 positions[8], brush_color, brush_size)
                    previous_x, previous_y = positions[8]

            elif up_fingers[1] and up_fingers[2]:
                # Draw shapes (circle or square)
                if canvas.contains(x, y) and not hide_canvas:
                    if selected_shape == 'Circle' or selected_shape == 'Square':
                        if not shape_locked:
                            shape_position = positions[8]
                            shape_locked = True
                            initial_distance = np.sqrt(
                                (positions[8][0] - positions[12][0])**2 + (positions[8][1] - positions[12][1])**2)
                            initial_size = 30
                        else:
                            current_distance = np.sqrt(
                                (positions[8][0] - positions[12][0])**2 + (positions[8][1] - positions[12][1])**2)
                            zoom_factor = current_distance / initial_distance
                            current_size = int(initial_size * zoom_factor)
                            if selected_shape == 'Circle':
                                cv2.circle(drawing_canvas, shape_position,
                                           current_size, brush_color, -1)
                            elif selected_shape == 'Square':
                                shape_corner2 = (
                                    shape_position[0] + current_size, shape_position[1] + current_size)
                                cv2.rectangle(
                                    drawing_canvas, shape_position, shape_corner2, brush_color, -1)
                    else:
                        if previous_x == 0 and previous_y == 0:
                            previous_x, previous_y = positions[8]
                        if brush_color == (0, 0, 0):
                            cv2.line(drawing_canvas, (previous_x, previous_y),
                                     positions[8], brush_color, eraser_size)
                        else:
                            cv2.line(drawing_canvas, (previous_x, previous_y),
                                     positions[8], brush_color, brush_size)
                        previous_x, previous_y = positions[8]
            else:
                previous_x, previous_y = 0, 0
                shape_locked = False

        # Draw UI elements on the frame
        color_button.draw(frame)
        cv2.rectangle(frame, (color_button.x, color_button.y), (color_button.x +
                      color_button.w, color_button.y + color_button.h), (255, 255, 255), 2)
        canvas_button.draw(frame)
        cv2.rectangle(frame, (canvas_button.x, canvas_button.y), (canvas_button.x +
                      canvas_button.w, canvas_button.y + canvas_button.h), (255, 255, 255), 2)

        # Display the canvas if not hidden
        if not hide_canvas:
            canvas.draw(frame)
            canvas_gray = cv2.cvtColor(drawing_canvas, cv2.COLOR_BGR2GRAY)
            _, inv_img = cv2.threshold(
                canvas_gray, 20, 255, cv2.THRESH_BINARY_INV)
            inv_img = cv2.cvtColor(inv_img, cv2.COLOR_GRAY2BGR)
            frame = cv2.bitwise_and(frame, inv_img)
            frame = cv2.bitwise_or(frame, drawing_canvas)

        # Display the color and shape buttons if not hidden
        if not hide_colors:
            for color_box in colors:
                color_box.draw(frame)
                cv2.rectangle(frame, (color_box.x, color_box.y), (color_box.x +
                              color_box.w, color_box.y + color_box.h), (255, 255, 255), 2)
            for shape_button in shapes:
                shape_button.draw(frame)
                cv2.rectangle(frame, (shape_button.x, shape_button.y), (shape_button.x +
                              shape_button.w, shape_button.y + shape_button.h), (255, 255, 255), 2)

        # Show the final frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the camera and close all windows
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
