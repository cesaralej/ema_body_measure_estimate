import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
import math

# Set variables
model_path = 'yolo/yolov8n-pose.pt'
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

def yolo_inference(model_path, image):
    model = YOLO(model_path)
    results = model(image)
    return results

def crop_body(model_path, demo_image):
    results = yolo_inference(model_path, demo_image)

    # Get the first detected person (assuming single person)
    person = results[0].boxes.xyxy.cpu().numpy()[0]
    xmin, ymin, xmax, ymax = int(person[0]), int(person[1]), int(person[2]), int(person[3])

    loaded_image = cv2.imread(demo_image)
    # Crop the image using the bounding box
    cropped_img = loaded_image[ymin:ymax, xmin:xmax] 
    return cropped_img

def get_keypoints(results_crop):
    result_keypoint = results_crop[0].keypoints.xy.cpu().numpy()[0]
    return result_keypoint

def segment(image, sam_model, boxes):
  sam_model.set_image(image)
  H, W, _ = image.shape
  boxes_xyxy = boxes.xyxyn * torch.Tensor([W, H, W, H])

  transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.cpu(), image.shape[:2])
  masks, _, _ = sam_model.predict_torch(
      point_coords = None,
      point_labels = None,
      boxes = transformed_boxes,
      multimask_output = False,
      )
  return masks.cpu()

def sam_segmentation(image, results_crop):
    sam = sam_model_registry["default"](checkpoint="legacy/segment-anything/sam_vit_h_4b8939.pth")
    predictor = SamPredictor(sam)
    segmented_frame_masks = segment(image, predictor, boxes=results_crop[0].boxes)
    return segmented_frame_masks

def get_binary_mask(mask):
    """
    Generate a black and white mask image.
    
    :param mask: The mask to be processed (PyTorch tensor or NumPy array).
    :return: The binary mask image (NumPy array).
    """
    # Convert the tensor to a NumPy array if it is a tensor
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    # Ensure the mask is 2D
    if mask.ndim > 2:
        mask = mask[0]

    # Create a binary mask (white for the mask area, black otherwise)
    binary_mask = np.zeros_like(mask, dtype=np.uint8)
    binary_mask[mask > 0] = 255
    
    return binary_mask

def find_edge(image, start_x, start_y, left=True):
    """
    Given a specific coordinate (start_x, start_y), go left until hitting a black pixel.
    :param image: The mask image (numpy array).
    :param start_x: The starting x-coordinate.
    :param start_y: The starting y-coordinate.
    :return: The x-coordinate of the edge.
    """

    # Check if the starting coordinate is within the image bounds
    if start_y >= image.shape[0] or start_x >= image.shape[1]:
        raise ValueError("Starting coordinates are outside the image bounds")

    if left:
        # Start from the given coordinate and move left until a black pixel is found
        for x in range(start_x, image.shape[1]):
            if image[start_y, x] == 0:  # Assuming black pixel has value 0
                return x - 1  # Return the first non-black pixel's x-coordinate
    else:
        # Start from the given coordinate and move right until a black pixel is found
        for x in range(start_x, -1, -1):
            if image[start_y, x] == 0:  # Assuming black pixel has value 0
                return x + 1  # Return the first non-black pixel's x-coordinate

    # If no black pixel is found, return 0 (the leftmost edge)
    return 0

def get_pixels_per_cm(image, keypoints, height_cm, relative=False):
    """
    Calculate the number of pixels per centimeter in the image.
    
    :param image: The mask image (NumPy array).
    :param height_cm: The height in centimeters.
    :return: The number of pixels per centimeter.
    """

    ymax, _ = image.shape
    # Get the y-coordinate of the top of the head
    y_top_head = int(keypoints[LEFT_EYE][1])
    # Get the y-coordinate of the bottom of the foot
    y_bottom_foot = int(keypoints[LEFT_ANKLE][1])

    # Calculate the height of the person in pixels
    relative_height_pixels = y_bottom_foot- y_top_head
    relative_height_cm = relative_height_pixels / height_cm

    pixels_per_cm = ymax / height_cm

    bias = 1.3

    if relative:
        return relative_height_cm * bias
    else:
        return pixels_per_cm
    
def ellipse_perimeter(front_width, side_width):
    """
    Calculate the circumference of an ellipse that circumscribes a rectangle.

    Parameters:
    front_width (float): The width of the rectangle.
    side_width (float): The height of the rectangle.

    Returns:
    float: The circumference of the ellipse.
    """
    # Calculate the semi-major and semi-minor axes
    a = math.sqrt((front_width / 2)**2 + (side_width / 2)**2)
    b = min(front_width / 2, side_width / 2)

    # Calculate the perimeter using Ramanujan's approximation formula for an ellipse
    perimeter = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))

    return perimeter

def get_shoulder_length(image, keypoints):
    left_shoulder_x, left_shoulder_y = keypoints[LEFT_SHOULDER]
    ls_x, ls_y = int(left_shoulder_x), int(left_shoulder_y)
    right_shoulder_x, right_shoulder_y = keypoints[RIGHT_SHOULDER]
    rs_x, rs_y = int(right_shoulder_x), int(right_shoulder_y)
    # Find the edge
    left_edge = find_edge(image, ls_x, ls_y)
    distance_to_edge = left_edge - ls_x
    right_edge = rs_x - distance_to_edge
    difference =  left_edge - right_edge
    shoulder_length = difference

    return shoulder_length

def measure_waist(focuspoints, focuspoints_side, image, mask_side):
    left_hip_x, left_hip_y = focuspoints[LEFT_HIP]
    lh_x, lh_y  = int(left_hip_x), int(left_hip_y)
    right_hip_x, right_hip_y = focuspoints[RIGHT_HIP]
    rh_x, rh_y = int(right_hip_x), int(right_hip_y)

    left_hip_side_x, left_hip_side_y = focuspoints_side[LEFT_HIP]
    lhs_x, lhs_y = int(left_hip_side_x), int(left_hip_side_y)

    # Get the measurement from the front of the chest
    left_edge = find_edge(image, lh_x, lh_y)
    right_edge = find_edge(image, rh_x, rh_y, left=False)
    distance_to_left_edge = left_edge - lh_x
    distance_to_right_edge = right_edge - rh_x
    average_distance = (distance_to_left_edge + distance_to_right_edge) / 2
    max_left = lh_x + average_distance
    max_right = rh_x - average_distance
    difference =  max_left - max_right
    waist_front = difference
    #print(f"The person's waist width is {waist_front:.2f} cm")

    #Get the measurement from the side of the chest
    right_side = find_edge(mask_side, lhs_x, lhs_y)
    left_side = find_edge(mask_side, lhs_x, lhs_y, left=False)

    waist_side = right_side - left_side
    #print(f"The person's Chest width is {shoulder_width:.2f} pixels")

    perimeter = ellipse_perimeter(waist_front, waist_side)
    #print(f"The approximate perimeter of the oval is: {perimeter:.2f}")
    return perimeter

def measure_chest(focuspoints, focuspoints_side, mask_side):
    left_shoulder_x = focuspoints[LEFT_SHOULDER][0]
    ls_x = int(left_shoulder_x)
    right_shoulder_x = focuspoints[RIGHT_SHOULDER][0]
    rs_x = int(right_shoulder_x)

    left_shoulder_side_x, left_shoulder_side_y = focuspoints_side[LEFT_SHOULDER]
    lss_x, lss_y = int(left_shoulder_side_x), int(left_shoulder_side_y)
    left_elbow_y = focuspoints_side[LEFT_ELBOW][1]
    le_y = int(left_elbow_y)

    # Get the measurement from the front of the chest
    shoulder_front =  ls_x - rs_x
    #print(f"The person's shoulder width is {shoulder_front:.2f} cm")
    
    # Get the height between the shoulder and elbow
    mid_shoulder_elbow_height = int(lss_y-((lss_y - le_y)/2))

    #Get the measurement from the side of the chest
    right_side = find_edge(mask_side, lss_x, mid_shoulder_elbow_height)
    left_side = find_edge(mask_side, lss_x, mid_shoulder_elbow_height, left=False)

    shoulder_width = right_side - left_side
    #print(f"The person's Chest width is {shoulder_width:.2f} pixels")

    perimeter = ellipse_perimeter(shoulder_front, shoulder_width)
    #print(f"The approximate perimeter of the oval is: {perimeter:.2f}")
    return perimeter

def measure_arm(focuspoints):
    left_shoulder_x, left_shoulder_y = focuspoints[LEFT_SHOULDER]
    ls_x, ls_y = int(left_shoulder_x), int(left_shoulder_y)
    right_shoulder_x, right_shoulder_y = focuspoints[RIGHT_SHOULDER]
    rs_x, rs_y = int(right_shoulder_x), int(right_shoulder_y)
    left_wrist_x, left_wrist_y = focuspoints[LEFT_WRIST]
    lw_x, lw_y = int(left_wrist_x), int(left_wrist_y)
    right_wrist_x, right_wrist_y = focuspoints[RIGHT_WRIST]
    rw_x, rw_y = int(right_wrist_x), int(right_wrist_y)

    # Get distance from shoulder to wrist
    left_distance = math.sqrt((lw_x - ls_x)**2 + (lw_y - ls_y)**2)
    right_distance = math.sqrt((rw_x - rs_x)**2 + (rw_y - rs_y)**2) 
    distance = (left_distance + right_distance) / 2
    return distance

def measure_length(focuspoints):
    left_hip_x, left_hip_y = focuspoints[LEFT_HIP]
    lh_x, lh_y = int(left_hip_x), int(left_hip_y)
    right_hip_x, right_hip_y = focuspoints[RIGHT_HIP]
    rh_x, rh_y = int(right_hip_x), int(right_hip_y)
    left_shoulder_x, left_shoulder_y = focuspoints[LEFT_SHOULDER]
    ls_x, ls_y = int(left_shoulder_x), int(left_shoulder_y)
    right_shoulder_x, right_shoulder_y = focuspoints[RIGHT_SHOULDER]
    rs_x, rs_y = int(right_shoulder_x), int(right_shoulder_y)

    # Get distance from hip to ankle
    left_distance = math.sqrt((ls_x - lh_x)**2 + (ls_y - lh_y)**2)
    right_distance = math.sqrt((rs_x - rh_x)**2 + (rs_y - rh_y)**2) 
    distance = (left_distance + right_distance) / 2
    bias = 1.2
    return distance * bias

def measure_wrist(focuspoints, image_front):
    left_wrist_x, left_wrist_y = focuspoints[LEFT_WRIST]
    lw_x, lw_y = int(left_wrist_x), int(left_wrist_y)
    right_wrist_x, right_wrist_y = focuspoints[RIGHT_WRIST]
    rw_x, rw_y = int(right_wrist_x), int(right_wrist_y)

    # Left wrist circumference
    lw_left_edge = find_edge(image_front, lw_x, lw_y)
    lw_right_edge = find_edge(image_front, lw_x, lw_y, left=False)
    left_diameter = lw_left_edge - lw_right_edge

    # Right wrist circumference
    rw_left_edge = find_edge(image_front, rw_x, rw_y)
    rw_right_edge = find_edge(image_front, rw_x, rw_y, left=False)
    right_diameter = rw_left_edge - rw_right_edge

    avr_diameter = (right_diameter + left_diameter) / 2

    circumference = math.pi * avr_diameter
    return circumference

def ema_inference(model_path, image_front):
    print("Starting the EMA process...")
    print("Cropping the body...")
    cropped_img = crop_body(model_path, image_front)
    print("Detecting pose focus points...")
    results_crop = yolo_inference(model_path, cropped_img)
    result_keypoint = get_keypoints(results_crop)
    print("Segmenting the body...")
    segmented_frame_masks = sam_segmentation(cropped_img, results_crop)
    binary_mask = get_binary_mask(segmented_frame_masks[0][0])

    return binary_mask, result_keypoint

def save_mask(mask, focalpoints, file_name):
    body_coords = {part: focalpoints[part] for part in range(17)}

    # make body_coords a list of tuples
    body_coords_t = [(x, y) for x, y in body_coords.values()]


    # Draw a point at the left shoulder coordinates
    point_color = (255, 0, 0)  # Red color in RGB
    point_radius = 5
    point_thickness = -1  # Thickness of -1 px will fill the circle

    people_mask = cv2.convertScaleAbs(mask)
    mask_image_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Draw the point on the image
    for x, y in body_coords_t:
        cv2.circle(mask_image_bgr, (int(x), int(y)), point_radius, point_color, point_thickness)

    #save the result image
    cv2.imwrite(file_name, cv2.cvtColor(mask_image_bgr, cv2.COLOR_RGB2BGR))

def ema(model_path, image_front, image_side, height_cm, size_chart):
    print("Starting the EMA process...")
    print("Calculating front focal points...")
    binary_mask, result_keypoint = ema_inference(model_path, image_front)
    print("Calculating side focal points...")
    binary_mask_side, result_keypoint_side = ema_inference(model_path, image_side)
    pixels_per_cm = get_pixels_per_cm(binary_mask, result_keypoint, height_cm, relative=True)
    chest = round(measure_chest(result_keypoint, result_keypoint_side, binary_mask_side)/pixels_per_cm,2)
    waist = round(measure_waist(result_keypoint, result_keypoint_side, binary_mask, binary_mask_side)/pixels_per_cm,2)
    arm = round(measure_arm(result_keypoint)/pixels_per_cm,2)
    length = round(measure_length(result_keypoint)/pixels_per_cm,2)
    wrist = round(measure_wrist(result_keypoint, binary_mask)/pixels_per_cm,2)
    shoulders = round(get_shoulder_length(binary_mask, result_keypoint)/pixels_per_cm,2)
    body_measurements = {"shoulders": shoulders, "waist": waist, "length": length, "arm": arm, "chest": chest, "wrist": wrist, "Px/cm": pixels_per_cm}
    shirt_size = find_shirt_size(size_chart, body_measurements)
    print(f"The number of pixels per centimeter is: {pixels_per_cm:.2f}")
    print(f"Shoulders: {shoulders:.2f} cm")
    print(f"Upper body length: {length:.2f} cm")
    print(f"Waist: {waist:.2f} cm")
    print(f"Arm length: {arm:.2f} cm")
    print(f"Chest: {chest:.2f} cm")
    #print(f"wrist: {wrist:.2f} cm\n")
    print(f"The recommended shirt size for {demo_image.split('/')[-1].split('.')[0]} is: {shirt_size}")
    result_file_name = demo_image.split('/')[-1].split('.')[0] + '_result.jpg'
    save_mask(binary_mask, result_keypoint, result_file_name)
    save_mask(binary_mask_side, result_keypoint_side, result_file_name.split('.')[0] + '_side.jpg')
    body_measurements = {"shoulders": shoulders, "waist": waist, "length": length, "arm": arm, "chest": chest, "wrist": wrist, "Px/cm": pixels_per_cm, "shirt_size": shirt_size}
    return body_measurements

demo=1
demo_height = 175
demo_image = f'data/{demo}.jpeg'
demo_image_side = f'data/{demo}b.jpeg'
size_chart = [
    {'size': 'S', 'chest': 88, 'waist': 88, 'length': 70, 'arm': 62},
    {'size': 'M', 'chest': 94, 'waist': 94, 'length': 71, 'arm': 63},
    {'size': 'L', 'chest': 101, 'waist': 101, 'length': 72, 'arm': 64},
    {'size': 'XL', 'chest': 107, 'waist': 107, 'length': 75 , 'arm': 65}
]

test_results = ema(model_path, demo_image, demo_image_side, demo_height, size_chart)
print(test_results)