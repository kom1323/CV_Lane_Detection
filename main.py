import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

roi_coordinates_focused = (480, 720, 100,1000)
can_change_lines=True
switch_direction=0


def detect_crosswalk(image):

    """Uses template matchin to detect crosswalks and display a bounding box on the crosswalk"""

    subfolder_path = 'croswalk'
    os.makedirs(subfolder_path, exist_ok=True)
    template_file_path = os.path.join(subfolder_path, 'crosswalk_template.jpg')
    template = cv2.imread(template_file_path)   
    main_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    threshold = 0.48
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val >= threshold:
        #draw rectangle around the template
        h, w = template_gray.shape
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

def enhance_lane_visibility(image):
   
    """Apply histogram equalizers to the image to get a better detection of the lane lines"""
   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_output = clahe.apply(gray)

    alpha = 1.1  # Contrast control 
    beta = 1    # Brightness control 
    enhanced = cv2.convertScaleAbs(clahe_output, alpha=alpha, beta=beta)
    enhanced[enhanced<180] = 0
    return enhanced

def image_manipulation(image):

    "Extracts the valuable pixels from the image as white and not valuable as black"

    temp_clipped_frame = image.copy()
    temp_clipped_frame = cv2.bilateralFilter(temp_clipped_frame, d=5, sigmaColor=75, sigmaSpace=150)
    temp_clipped_frame=enhance_lane_visibility(temp_clipped_frame)
        
    kernel_size = 8
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    edges = cv2.dilate(temp_clipped_frame, kernel, iterations=1)

    edges = cv2.Canny(edges, 50, 100)
    
    #creating a triangle mask for better roi
    mask_corners = np.zeros_like(edges)
    height, width = mask_corners.shape[:2]
    vertices = np.array([[(0, int(height)), (int(width*0.55), 0), (int(width*0.65), 0) ,(width,int(height))]], dtype=np.int32)
    cv2.fillPoly(mask_corners, vertices, 255)
    masked_corner_edges = cv2.bitwise_and(edges, mask_corners)

    return masked_corner_edges

def average_lines(lines):
    count = len(lines)
    if count > 0:
        avg_rho = np.mean(lines[:, 0, 0])
        avg_theta = np.mean(lines[:, 0, 1])
        return np.array([[avg_rho, avg_theta]])
    return None

def filter_lines(lines_left,lines_right):

    """combines left lines into one line and right lines into one line and handles the lane switching states"""

    global switch_direction
    global can_change_lines
    left_line=None
    right_line=None
    change_lanes=0
    filtered_lines = [None,None]

    #avg lines for each side
    if lines_left is not None:
        left_line = average_lines(lines_left)
        filtered_lines[0]=left_line 
    if lines_right is not None:
        right_line = average_lines(lines_right)
        filtered_lines[1]=right_line
    
    #changing lanes logic
    change_lanes= check_lane_change(filtered_lines)
    if left_line is not None and right_line is not None:
        can_change_lines=True
        switch_direction=0
    if change_lanes!=0 and can_change_lines==True:
        can_change_lines=False
        switch_direction=change_lanes

    return filtered_lines

def check_lane_change(lines):
        dirc_val=0       
        if lines[0] is not None and lines[1] is None:
            dirc_val= -1
        elif lines[1] is not None and lines[0] is None:
            dirc_val= 1
        return dirc_val
       
def region(original_frame,type,image=None):
    #focus on region of interest
    if type=="crop":
        roi=original_frame[roi_coordinates_focused[0]: roi_coordinates_focused[1], roi_coordinates_focused[2]:roi_coordinates_focused[3]]
        clipped_frame = roi.copy()
        return clipped_frame
    else:
        #paste the roi back into the original frame
        original_frame[roi_coordinates_focused[0]: roi_coordinates_focused[1], roi_coordinates_focused[2]:roi_coordinates_focused[3]] = image
        return original_frame
    
def drawLines(image,lines):
    if lines is None:
        return image
    for line in lines:
        if line is None:
            continue
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 5)
    return image

def collectLines(image):

    """Uses cv2.houghlines to collect all lines of the left and right lanes sperately"""

    l_lines =cv2.HoughLines(image, rho=1,theta=np.pi / 180, threshold=37 ,min_theta=0.1,max_theta=1.07)
    r_lines = cv2.HoughLines(image, rho=1, theta=np.pi / 180, threshold=37 ,min_theta=2.2,max_theta=2.9)        
    return l_lines,r_lines

def lane_notifier(original_frame,side):

    """displays on frame the side the car is switching lanes to"""

    text = f"Taking {side} lane"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 2
    font_color = (0, 255, 0)  # White color in BGR format

    # Get text size to determine the position
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_position = ((original_frame.shape[1] - text_size[0]) // 4, 200)

    # Create a black background for the text
    text_background = np.zeros_like(original_frame)
    cv2.putText(text_background, text, text_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    # Blend the text background with the original image
    alpha = 0.7  # Adjust the transparency of the text background
    result = cv2.addWeighted(original_frame, 1, text_background, alpha, 0)
    return result

def process_image(original_frame):
    
    """Finds lane lines and crosswalks, displays them and returns the frame"""
    
    global switch_direction
    global can_change_lines

    #focusing on region of interest   
    cropped_frame= region(original_frame,"crop")
    
    #image manipulation
    manipulated_image = image_manipulation(cropped_frame)

    #finding crosswalks and displaying bounding box
    detect_crosswalk(cropped_frame)

    #extracting lines
    lines_left, lines_right = collectLines(manipulated_image)
    lines = filter_lines(lines_left,lines_right)
    
    #creating result
    cropped_image_with_lines = drawLines(cropped_frame,lines)
    frame_with_warning = region(original_frame,"paste",cropped_image_with_lines)

    #check and implement lane_notifier
    if can_change_lines==False:
        if switch_direction==-1:
            return lane_notifier(frame_with_warning,"left")
        elif switch_direction==1:
            return lane_notifier(frame_with_warning,"right")
    return frame_with_warning

if __name__ == "__main__":
    cap = cv2.VideoCapture('Driving-crosswalk.mp4')
    
    while(cap.isOpened() ): 
        ret, frame = cap.read()
        if ret :
            processed_frame = process_image(frame)
            cv2.imshow('Lane Detection',processed_frame)
        else:
            break
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
