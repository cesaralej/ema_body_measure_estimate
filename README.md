# Chest Measurement using YOLO and SAM

This project calculates the perimeter of the chest of a person using computer vision models YOLO (You Only Look Once) for detecting key points and SAM (Segment Anything Model) for segmenting the body. The perimeter is calculated by approximating the oval shape formed within the front and side width measurements.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Description

The project involves detecting key points of a person using YOLO, segmenting the body using SAM, and calculating the perimeter of the chest. The chest measurement is derived from the oval shape formed within the front and side width measurements of the person.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/chest-measurement.git
    cd chest-measurement
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Load and Preprocess Image**:
    Load the image and resize it if necessary.
    ```python
    import cv2

    demo_image = 'path_to_image.jpg'
    loaded_image = cv2.imread(demo_image)
    small_cropped_img = cv2.resize(loaded_image, (512, 512), interpolation=cv2.INTER_LINEAR)
    ```

2. **Detect Key Points and Segments**:
    Use YOLO to detect key points and SAM to segment the image.
    ```python
    from ultralytics import YOLO
    from segment_anything import SamPredictor, sam_model_registry

    model = YOLO('yolo/yolov8n-pose.pt')
    results = model.predict(source=small_cropped_img)
    ```

3. **Calculate Perimeter**:
    Calculate the perimeter of the chest using the ellipse approximation formula.
    ```python
    def ellipse_perimeter(front_width, side_width):
        import math
        a = front_width / 2
        b = side_width / 2
        return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))

    front_width = 100  # Example value, replace with actual measurement
    side_width = 50    # Example value, replace with actual measurement
    perimeter = ellipse_perimeter(front_width, side_width)
    print(f"The approximate perimeter of the oval is: {perimeter:.2f}")
    ```

4. **Display Results**:
    Visualize the key points and segmentation on the image using OpenCV and matplotlib.

## Project Structure

