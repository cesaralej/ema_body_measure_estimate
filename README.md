# Ema: e-fitting

Ema is an AI-powered plugin for clothing stores that uses computer vision to calculate precise body measurements from a photo, ensuring customers get the perfect fit every time. This project calculates the measurements using computer vision models YOLO (You Only Look Once) for detecting key points and SAM (Segment Anything Model) for segmenting the body.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/cesaralej/ema_body_measure_estimate.git
    cd ema_body_measure_estimate
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

4. Make sure you install Ultralytics and Segment Anything libraries to your environment. The Ema notebook includes instructions to install them.

## Usage

Run Ema by following the Jupyter notebook or by running the ema script.

