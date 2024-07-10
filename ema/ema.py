import cv2 #cv2 is the OpenCV library which is used for computer vision tasks like reading images, drawing shapes on images, etc.
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode

# Load the Haar cascade for full body detection
full_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml') # Haar Cascade algorithm is used for object detection in real-time

def detect_body(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converts the image to grayscale
    bodies = full_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3) #detects the full body in the image

    for (x, y, w, h) in bodies: #(x, y): The top-left corner of the detected rectangle. (w, h): The width and height of the detected rectangle.
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2) #draws a rectangle around the detected body in blue
        body_img = img[y:y+h, x:x+w] #Crops the detected body region from the original image
        return body_img
    
    return None

def segment_body(body_img, num_segments=40):
    height, width = body_img.shape[:2] #gets the height and width of the body image
    segment_width = width // num_segments #Calculates the width of each segment by dividing the image width by the number of segments.
    
    segments = [] #A loop iterates over the number of segments, calculates the start and end x-coordinates for each segment, extracts the segment from the image, and appends it to the list of segments.
    for i in range(num_segments):
        start_x = i * segment_width
        end_x = (i + 1) * segment_width if i != num_segments - 1 else width
        segment = body_img[:, start_x:end_x]
        segments.append((start_x, end_x, segment))
        
    return segments

def extract_focal_points(segments, gender='female'):
    focal_points = {
        'shoulders': [],
        'bust': [],
        'waist': [],
        'hips': []
    }
    
    # Define the ranges for each body part based on the gender
    if gender == 'female':
        ranges = {
            'shoulders': (11, 18, 29, 37),
            'bust': (12, 19, 28, 35),
            'waist': (12, 19, 28, 34),
            'hips': (11, 18, 28, 36)
        }
    else:
        ranges = {
            'shoulders': (11, 16, 28, 34),
            'bust': (14, 17, 27, 32),
            'waist': (13, 17, 27, 32),
            'hips': (13, 17, 28, 31)
        }
    
    def find_edges(segment):
        edges = cv2.Canny(segment, 100, 200)
        left_points = np.where(edges.any(axis=0))[0]
        if left_points.size > 0:
            leftmost = left_points[0]
            rightmost = left_points[-1]
            return leftmost, rightmost
        return None, None
    
    for part, (left_start, left_end, right_start, right_end) in ranges.items():
        left_segments = segments[left_start:left_end]
        right_segments = segments[right_start:right_end]
        
        for start_x, end_x, segment in left_segments:
            leftmost, _ = find_edges(segment)
            if leftmost is not None:
                focal_points[part].append((start_x + leftmost, None))
        
        for start_x, end_x, segment in right_segments:
            _, rightmost = find_edges(segment)
            if rightmost is not None:
                focal_points[part].append((None, start_x + rightmost))
    
    # Average the focal points for each part
    for part in focal_points:
        left_points = [p[0] for p in focal_points[part] if p[0] is not None]
        right_points = [p[1] for p in focal_points[part] if p[1] is not None]
        
        if left_points and right_points:
            focal_points[part] = (int(np.mean(left_points)), int(np.mean(right_points)))
        else:
            focal_points[part] = (None, None)
    
    return focal_points

# estimate measurements
def estimate_measurements(body_img, focal_points, height_cm):
    # get image height with cv2 after body detection
    image_height = body_img.shape[0]
    conversion_factor = height_cm / image_height  # Convert pixel measurements to cm
    measurements = {}
    
    for part, (left, right) in focal_points.items():
        if left is not None and right is not None:
            pixel_width = right - left
            cm_width = pixel_width * conversion_factor
            measurements[part] = cm_width
    
    return measurements

# function to get inputs from streamlit
def getMeasurements(image_path, height_cm, gender):
    image = Image.open(image_path)
    img = np.array(image)  # Convert PIL image to NumPy array
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV compatibility
    body_img = detect_body(img)

    
    if body_img is not None:    
        segments = segment_body(body_img)
        focal_points = extract_focal_points(segments, gender)
        measurements = estimate_measurements(body_img, focal_points, height_cm)
        return measurements
    else:
        return "No body detected."
    