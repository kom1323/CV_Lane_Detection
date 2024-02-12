import cv2
import numpy as np
import matplotlib.pyplot as plt

import math


prev_lines = None


def filter_white_and_yellow(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for white color in HSV
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
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



def filter_lines(lines):
    global prev_lines
   
   
    lines = [line for line in lines if abs(line[0][1] - np.pi / 2) > np.radians(30) and abs(line[0][1] - np.pi / 2) < np.radians(90)]

    old_theta = []
    old_rho = []
    epsilon_theta = np.pi / 20
    epsilon_rho = 20
    count_left, avg_rho_left, avg_theta_left = (0,0,0)
    count_right, avg_rho_right, avg_theta_right = (0,0,0)
    filtered_lines = []


    # Draw the two longest lines on the image
    for line in lines:
        rho, theta = line[0]
        # #check if two lines have almost the same angle
        # if any(abs(theta - x) <= epsilon_theta for x in old_theta):
        #     continue
        # #check if two lines have almost the same distance from 0,0
        # if any(abs(rho - x) <= epsilon_rho for x in old_rho):
        #     continue
        # old_rho.append(rho)
        # old_theta.append(theta)
        

        if theta < 1.5:
            # Update average
            avg_rho_left += rho
            avg_theta_left += theta
            count_left += 1
        else:
            avg_rho_right += rho
            avg_theta_right += theta
            count_right += 1

    if count_left > 0:
        avg_rho_left /= count_left
        avg_theta_left /= count_left


    if count_right > 0:
        avg_rho_right /= count_right
        avg_theta_right /= count_right



    left_line = np.array([[avg_rho_left, avg_theta_left]])
    right_line = np.array([[avg_rho_right, avg_theta_right]])


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
        print("rho diff: ", prev_lines[0][0][0] -  prev_lines[1][0][0])
        print("theta diff: ", prev_lines[0][0][1] -  prev_lines[1][0][1])

    


    return filtered_lines


def region(image):

    """MAYBE USE"""
    height, width = image.shape
    triangle = np.array([
                       [(int(width*0.2), height), (int(width*0.5), int(height*0.2)), (int(width*0.8), height)]
                       ])
    
    mask = np.zeros_like(image)
    
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask


def process_image(original_frame):
    


    
    #roi2_coordinates_focused = (220, 290, 100, 300)
    #roi1_coordinates_focused = (600,1080,250,1500)
    roi_coordinates_focused = (200, 350, 150,500)

    #focus on region of interest
    roi=original_frame[roi_coordinates_focused[0]: roi_coordinates_focused[1], roi_coordinates_focused[2]:roi_coordinates_focused[3]]
    clipped_frame = roi.copy()
    temp_clipped_frame = clipped_frame.copy()
    temp_clipped_frame=filter_white_and_yellow(temp_clipped_frame)
    temp_clipped_frame = cv2.cvtColor(temp_clipped_frame, cv2.COLOR_RGB2GRAY)

    """USE DILATE AND ERODE"""
    d = 5  # edge size of neighborhood perimeter
    sigma_r = 150  # sigma range
    sigma_s = 100  # sigma spatial
    edges = cv2.Canny(temp_clipped_frame, 10, 150)

    mask_corners = np.ones_like(edges) #mask for corners
    mask_corners[:,:]=255
    height, width = mask_corners.shape[:2]
   


    vertices = np.array([[(0, 0), (width*0.4, 0), (0, int(height * 0.6))],[(width, 0), (width-width*0.4, 0), (width, int(height * 0.6))]], dtype=np.int32)

    # Fill the triangles in the mask
    cv2.fillPoly(mask_corners, vertices, 0)
    masked_corner_edges = cv2.bitwise_and(edges, mask_corners)
    # Apply the mask to the edges



    lines = cv2.HoughLines(masked_corner_edges, rho=0.5, theta=np.pi / 180, threshold=30 ,min_theta = -math.pi, max_theta = math.pi)
    


    if lines is None:
        return original_frame
    

    

    lines = filter_lines(lines)
  
    for line in lines:
        
        rho, theta = line[0]
        
        
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(clipped_frame, (x1, y1), (x2, y2), (255, 0, 0), 4)

    alpha = 0.1  # Adjust the alpha value for blending
    beta = 1.0 - alpha
    result = cv2.addWeighted(roi, alpha, clipped_frame, beta, 0.0)

    # Replace the ROI in the larger image with the blended result
    original_frame[roi_coordinates_focused[0]: roi_coordinates_focused[1], roi_coordinates_focused[2]:roi_coordinates_focused[3]] = result
    return original_frame


if __name__ == "__main__":

    cap = cv2.VideoCapture('Driving4.mp4')
    plt.figure(figsize=(20, 20))

    
    counter=1
    
    while(cap.isOpened() ): #and cap2.isOpened()):
        ret, frame = cap.read()
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
