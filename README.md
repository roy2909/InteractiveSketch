# Interactive Sketch

The Interactive Sketch is an interactive application that allows users to draw and sketch on a virtual canvas using hand gestures captured by a camera. It leverages computer vision techniques to detect and track hand movements, enabling a hands-free drawing experience.

## Features

- **Hand tracking using the MediaPipe library**
- **Drawing on a virtual canvas using hand gestures**
- **Selection of different colors and brush sizes**
- **Eraser functionality to remove drawn lines**
- **Shape drawing (circles and squares) with adjustable sizes**
- **Filling drawn shapes with selected colors**
- **Clearing the canvas**
- **User-friendly interface with buttons for colors, shapes, and canvas control**

## Requirements

- Python 3.x
- OpenCV (cv2)
- MediaPipe
- NumPy

## Installation

1. **Clone the repository:**
    ```sh
    git clone git@github.com:roy2909/InteractiveSketch.git
    ```

2. **Install the required dependencies:**
    ```sh
    pip install opencv-python
    pip install mediapipe
    ```

## Usage

1. **Run the interactive_sketch.py script:**
    ```sh
    python interactive_sketch.py
    ```

2. **Interact with the virtual canvas:**
    - The application will open a window displaying the camera feed.
    - Use your hand gestures to interact with the virtual canvas:
        - Raise your index finger to draw or select colors/shapes.
        - Raise your index and middle fingers together to draw shapes or adjust their size.
        - Use the color buttons to select different colors for drawing.
        - Use the shape buttons to select the shape to draw (circle or square).
        - Use the eraser button to switch to eraser mode and remove drawn lines.
        - Use the clear button to clear the entire canvas.
        - Use the fill button to fill the drawn shapes with the selected color.

3. **Press 'q' to quit the application.**
   

## Demo

https://github.com/roy2909/InteractiveSketch/assets/144197977/4acf1d77-cb58-4987-aea4-4465e783143c