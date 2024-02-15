import cv2
import numpy as np
import matplotlib.pyplot as plt

import math

########### Importing Notice ###########
    # original_frame.shape is (360, 640, 3)#
prev_lines = None
roi_coordinates_focused = (500, 720, 200,900)

def filter_white_and_yellow(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for white color in HSV
    lower_white = np.array([0, 0, 180], dtype=np.uint8)
    upper_white = np.array([255, 30, 255], dtype=np.uint8)

    # Define the lower and upper bounds for yellow color in HSV
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)

    # Create masks for white and yellow regions
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Combine the masks to get the final mask
    final_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=final_mask)

    return result

def show_image(img):

    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.colorbar()
    plt.show()

def image_manipulation(image):
    temp_clipped_frame = image.copy()
    temp_clipped_frame=filter_white_and_yellow(temp_clipped_frame)
    edges = cv2.cvtColor(temp_clipped_frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.bilateralFilter(edges, d=9, sigmaColor=75, sigmaSpace=150)
    edges = cv2.Canny(temp_clipped_frame, 100, 150)
    
    #edges=cv2.dilate(edges,np.array([[1,0,1],[0,1,0],[1,0,1]],dtype=np.uint8),iterations=4)
    #edges=cv2.erode(edges,np.array([[1,0,1],[0,1,0],[1,0,1]],dtype=np.uint8),iterations=4)

    #edges[edges>10]=255
    ################################################################
    #masking corners of roi
    mask_corners = np.zeros_like(edges) #mask for corners
    height, width = mask_corners.shape[:2]
    print(width,height)
    vertices = np.array([[(0, int(height)), (int(width*0.45), 0), (int(width*0.65), 0) ,(width,int(height))]], dtype=np.int32)

    # Fill the triangles in the mask
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
    global prev_lines
    left_line=None
    right_line=None
    change_lanes=0
    count_left=0
    count_right=0

    filtered_lines = []
    if lines_left is None and lines_right is None:
        left_line,right_line = prev_lines[0],prev_lines[1]
    if lines_left is not None:
        count_left=len(lines_left) 
        left_line = average_lines(lines_left)
    if lines_right is not None:
        count_right=len(lines_right)
        right_line = average_lines(lines_right)

    if prev_lines is not None: 
        if count_left == 0:
            left_line = prev_lines[0]
        if count_right == 0:
            right_line = prev_lines[1]
        

    if prev_lines is None and count_left > 0 and count_right > 0:
        filtered_lines = np.array([left_line,right_line])
        prev_lines = filtered_lines
    elif prev_lines is not None:
        filtered_lines = np.array([left_line,right_line])
        prev_lines = filtered_lines
    change_lanes= check_lane_change(filtered_lines)
            #return None,change_lanes
        # print("rho diff: ", prev_lines[0][0][0] -  prev_lines[1][0][0])
        # print("theta diff: ", prev_lines[0][0][1] -  prev_lines[1][0][1])
    return filtered_lines,change_lanes
def check_lane_change(lines):
        if lines is not None and len(lines) == 2:
            theta_diff = lines[0][0][1] - lines[1][0][1]
            print(theta_diff)
            if theta_diff < -2:
                print("Turn right")
                return 1
            elif theta_diff > 2:
                print("Turn left")
                return -1
        return 0



def region(original_frame,type,image=None):
    #focus on region of interest
    if type=="crop":
        roi=original_frame[roi_coordinates_focused[0]: roi_coordinates_focused[1], roi_coordinates_focused[2]:roi_coordinates_focused[3]]
        clipped_frame = roi.copy()
        return clipped_frame
    else:
        original_frame[roi_coordinates_focused[0]: roi_coordinates_focused[1], roi_coordinates_focused[2]:roi_coordinates_focused[3]] = image
        return original_frame
    

#function that caulatres length based on how full is the triangle
#let's define it for start to 50%
def drawLines(image,lines,length_lines):
    for line in lines:
        rho, theta = line[0]
        m = -1/np.tan(theta)
        b = rho /np.sin(theta)
        y1 = int(image.shape[0])
        x1 = int((y1-b)/m)
        y2 = int(image.shape[0]-image.shape[0]*length_lines)
        x2 = int((y2-b)/m)
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return image


def collectLines(image):
    l_lines =cv2.HoughLines(image, rho=1, theta=np.pi / 90, threshold=50 ,min_theta=0,max_theta=1)##min_theta =math.pi/4, max_theta = math.pi/3.5)#return to -math.pi/2
    r_lines = cv2.HoughLines(image, rho=1, theta=np.pi / 90, threshold=50 ,min_theta=2.2,max_theta=4) ##=-math.pi/4, max_theta = -math.pi/4.5)#return to -math.pi/2        
    return l_lines,r_lines
    # if l_lines is not None and r_lines is not None:
    #     lines=np.concatenate((l_lines,r_lines))
    # elif l_lines is not None:
    #     lines=l_lines
    # elif r_lines is not None:
    #     lines=r_lines
    # else:
    #     return None
    # return lines

def lane_notifier(original_frame,side):
    text = f"Taking {side} Lane"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 2
    font_color = (255, 255, 255)  # White color in BGR format
    background_color = (0, 0, 0)  # Black color in BGR format

    # Get text size to determine the position
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_position = ((original_frame.shape[1] - text_size[0]) // 2, 50)  # Adjust the Y-coordinate as needed

    # Create a black background for the text
    text_background = np.zeros_like(original_frame)
    cv2.putText(text_background, text, text_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    # Blend the text background with the original image
    alpha = 0.7  # Adjust the transparency of the text background
    result = cv2.addWeighted(original_frame, 1, text_background, alpha, 0)
    return result
def process_image(original_frame):
    
    ################################################################
    #focusing on region of interest   
    cropped_frame= region(original_frame,"crop")

    ################################################################
    #image manipulation
    manipulated_image = image_manipulation(cropped_frame)
    cv2.imshow('manipulated image',manipulated_image)

    #################################################################
    #extracting lines
    lines_left, lines_right = collectLines(manipulated_image)
    if lines_left is None and lines_right is None:
        return original_frame
    # print(f'how many lines found {len(lines)}')
    # print(lines)
    lines,change_lanes = filter_lines(lines_left,lines_right)
    

    #################################################################
    #creating result
    length_lines=1#proximity(manipulated_image)
    #print(f'length_lines: {length_lines}')
    cropped_image_with_lines = drawLines(cropped_frame,lines,length_lines)
    result = region(original_frame,"paste",cropped_image_with_lines)
    #################################################################
    if change_lanes==1:
        return lane_notifier(result,"left")
    elif change_lanes==-1:
        return lane_notifier(result,"right")
    return result


if __name__ == "__main__":

    cap = cv2.VideoCapture('Driving-pass.mp4')

    
    counter=1
    
    while(cap.isOpened() ): #and cap2.isOpened()):
        ret, frame = cap.read()
        output_path = 'output_frame.jpg'

        # Save the frame as an image file

        cv2.imwrite(output_path, cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        #print(f"frame {counter}")
        counter+=1
        if ret :

            processed_frame = process_image(frame)
            cv2.imshow('Lane Detection',processed_frame)

        else:
            break
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
