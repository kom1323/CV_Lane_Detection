import cv2
import numpy as np
import matplotlib.pyplot as plt

import math

########### Importing Notice ###########
prev_lines = np.array([None,None])
roi_coordinates_focused = (500, 720, 125,900)
can_change_lines=True
switch_direction=0




def filter_white_yellow_and_gray(image):

    
    # gamma = 0.25    
    # image = np.power(image/255.0, gamma) * 255.0
    # image = np.uint8(image)


    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_range = (108, 122)
    saturation_range = (15, 85)
    value_range = (90, 130)
    lower_custom = np.array([hue_range[0], saturation_range[0], value_range[0]], dtype=np.uint8)
    upper_custom = np.array([hue_range[1], saturation_range[1], value_range[1]], dtype=np.uint8)
    # Define the lower and upper bounds for white color in HSV
    lower_white = np.array([0, 0, 180], dtype=np.uint8)
    upper_white = np.array([255, 100, 255], dtype=np.uint8)

   
    # Create masks for white, yellow, and gray regions
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    #yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    custom_mask = cv2.inRange(hsv_image, lower_custom, upper_custom)
   
    final_mask = cv2.bitwise_or(white_mask, custom_mask)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=final_mask)
    cv2.imshow('before equ',result)

    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    equalized_result = cv2.equalizeHist(result)
    cv2.imshow('after equ',equalized_result)
    equalized_result[equalized_result>0] = 255



    return equalized_result

def show_image(img):

    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.colorbar()
    plt.show()

def image_manipulation(image):

    temp_clipped_frame = image.copy()
    temp_clipped_frame = cv2.bilateralFilter(temp_clipped_frame, d=5, sigmaColor=75, sigmaSpace=150)
    temp_clipped_frame=filter_white_yellow_and_gray(temp_clipped_frame)

    kernel_size = 8
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    edges = cv2.dilate(temp_clipped_frame, kernel, iterations=1)
    

    kernel_size = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    edges = cv2.erode(edges, kernel, iterations=1)

    edges = cv2.Canny(edges, 10, 20)
    
    
    ################################################################
    #masking corners of roi
    mask_corners = np.zeros_like(edges) #mask for corners
    height, width = mask_corners.shape[:2]
    vertices = np.array([[(0, int(height)), (int(width*0.55), 0), (int(width*0.82), 0) ,(width,int(height))]], dtype=np.int32)

    # Fill the triangles in the mask
    cv2.fillPoly(mask_corners, vertices, 255)
    masked_corner_edges = cv2.bitwise_and(edges, mask_corners)

    cv2.imshow("help", masked_corner_edges)
    return masked_corner_edges

def average_lines(lines):
    count = len(lines)
    if count > 0:
        avg_rho = np.mean(lines[:, 0, 0])
        avg_theta = np.mean(lines[:, 0, 1])
        return np.array([[avg_rho, avg_theta]])
    return None

def filter_lines(lines_left,lines_right):
    global prev_lines
    global switch_direction
    global can_change_lines
    left_line=None
    right_line=None
    change_lanes=0
    count_left=0
    count_right=0

    filtered_lines = [None,None]



    if lines_left is not None:
        left_line = average_lines(lines_left)
        filtered_lines[0]=left_line 
    if lines_right is not None:
        right_line = average_lines(lines_right)
        filtered_lines[1]=right_line
    
#  
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
        original_frame[roi_coordinates_focused[0]: roi_coordinates_focused[1], roi_coordinates_focused[2]:roi_coordinates_focused[3]] = image
        return original_frame
    
#let's define it for start to 50%
def drawLines(image,lines,length_lines):
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
    l_lines =cv2.HoughLines(image, rho=1,theta=np.pi / 180, threshold=30 ,min_theta=0.1,max_theta=1.07)##min_theta =math.pi/4, max_theta = math.pi/3.5)#return to -math.pi/2
    r_lines = cv2.HoughLines(image, rho=1, theta=np.pi / 180, threshold=30 ,min_theta=2.2,max_theta=3) ##=-math.pi/4, max_theta = -math.pi/4.5)#return to -math.pi/2        
    return l_lines,r_lines

def calculate_real_distance(pixel_size, focal_length):
    # Assuming a simple linear relationship between pixel size and real-world distance
    # You may need a more sophisticated calibration for accurate results
    real_size = 3.0  # Adjust based on the actual size of a vehicle in the scene
    real_distance = focal_length * real_size / pixel_size
    return real_distance

def draw_proximity_warning(frame, vehicles, focal_length):
    min_distance = float('inf')  # Initialize with positive infinity
    warning_color = (0, 0, 255)  # Default color for the warning text

    for (x, y, w, h) in vehicles:
        # Draw bounding boxes around detected vehicles
        pixel_size = (w + h) / 2 
        

        # Calculate pixel size (average of width and height)

        # Calculate real-world distance
        real_distance = calculate_real_distance(pixel_size, focal_length)
        if real_distance > 50:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Update minimum distance if the current distance is smaller
        min_distance = min(min_distance, real_distance)

        # Add real-world distance to the bounding box
        text = f"{real_distance:.2f} meters"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Check if close cars are found before adding the warning text
    if min_distance < 50.0:  # Adjust the threshold as needed
        # Update warning text color based on the minimum distance
        if min_distance < 20.0:
            warning_color = (255, 0, 0)  # Change color to green if minimum distance is less than 2.0 meters
        elif min_distance < 30.0:
            warning_color = (0, 255, 0)  # Change color to yellow if minimum distance is less than 5.0 meters
        else:
            warning_color = (0, 0, 255)

        # Add a warning message with the updated color
        warning_text = f"Proximity Warning: Vehicles Nearby! Min Distance: {min_distance:.2f} meters"
        cv2.putText(frame, warning_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, warning_color, 2, cv2.LINE_AA)

    return frame

def lane_notifier(original_frame,side):
    text = f"Taking Lane {side}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 2
    font_color = (0, 255, 0)  # White color in BGR format
    background_color = (original_frame.shape[0]*0.5, 0, 0)  # Black color in BGR format

    # Get text size to determine the position
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_position = ((original_frame.shape[1] - text_size[0]) // 4, 200)  # Adjust the Y-coordinate as needed

    # Create a black background for the text
    text_background = np.zeros_like(original_frame)
    cv2.putText(text_background, text, text_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    # Blend the text background with the original image
    alpha = 0.7  # Adjust the transparency of the text background
    result = cv2.addWeighted(original_frame, 1, text_background, alpha, 0)
    return result
def process_image(original_frame):
    global switch_direction
    global can_change_lines
    global is_night
    focal_length = 500.0
    ################################################################
    #focusing on region of interest   
    cropped_frame= region(original_frame,"crop")
    
    ################################################################
    #image manipulation
    manipulated_image = image_manipulation(cropped_frame)

    #################################################################
    #extracting lines
    lines_left, lines_right = collectLines(manipulated_image)
    lines = filter_lines(lines_left,lines_right)
    # Detect vehicles in the frame
    #vehicles = detect_vehicles(original_frame)

    # Draw proximity warning on the frame with real-world distance information
    

    #################################################################
    #creating result
    length_lines=1
    cropped_image_with_lines = drawLines(cropped_frame,lines,length_lines)
    frame_with_warning = region(original_frame,"paste",cropped_image_with_lines)
    #frame_with_warning = draw_proximity_warning(result, vehicles, focal_length)
    #################################################################
    #print(f'can_change_lines:{can_change_lines}, switch_direction:{switch_direction}')
    #frame_with_warning
    if can_change_lines==False:
        if switch_direction==-1:
            return lane_notifier(frame_with_warning,"left")
        elif switch_direction==1:
            return lane_notifier(frame_with_warning,"right")
    return frame_with_warning

def detect_vehicles(frame):
    # Load the pre-trained vehicle detection Haarcascades classifier
    cascade_src = 'cars.xml'

    car_cascade = cv2.CascadeClassifier(cascade_src)
    # Convert the frame to grayscale for Haarcascades
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect vehicles in the frame
    vehicles = car_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=6)#,minSize=)
    print(vehicles)

    return vehicles

if __name__ == "__main__":

    cap = cv2.VideoCapture('Driving-pass.mp4')
    
    counter=1
    
    while(cap.isOpened() ): #and cap2.isOpened()):
        ret, frame = cap.read()
        output_path = f'C:\\temp\\frames\\frame{counter}.jpg'
        # Save the frame as an image file

       #cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        print(f"frame {counter}")

        counter+=1
        if ret :
            processed_frame = process_image(frame)
            cv2.imshow('Lane Detection',processed_frame)
        #     if counter%5==0:
        #         cv2.imwrite(output_path, processed_frame)
        # else:
        #     break
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
