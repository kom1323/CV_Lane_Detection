import cv2
import numpy as np
import matplotlib.pyplot as plt

import math

########### Importing Notice ###########
    # original_frame.shape is (360, 640, 3)#
prev_lines = None
roi_coordinates_focused = (170, 360, 120,510)

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

def image_manipulation(image):
    temp_clipped_frame = image.copy()
    temp_clipped_frame=filter_white_and_yellow(temp_clipped_frame)
    edges = cv2.cvtColor(temp_clipped_frame, cv2.COLOR_RGB2GRAY)
    edges=cv2.dilate(edges,np.array([[1,0,1],[0,1,0],[1,0,1]],dtype=np.uint8),iterations=4)
    edges=cv2.erode(edges,np.array([[1,0,1],[0,1,0],[1,0,1]],dtype=np.uint8),iterations=4)
    
    edges = cv2.GaussianBlur(edges, (9, 9), 2)
    edges = cv2.Canny(edges, 50, 150)
    cv2.imshow("edges", edges)
    #masking corners of roi
    mask_corners = np.zeros_like(edges) #mask for corners
    height, width = mask_corners.shape[:2]
    vertices = np.array([[(0, height), (int(width*0.5), int(height*0.2)), (width,height)]], dtype=np.int32)

    # Fill the triangles in the mask
    cv2.fillPoly(mask_corners, vertices, 255)
    masked_corner_edges = cv2.bitwise_and(edges, mask_corners)
    return masked_corner_edges

def filterLines(lines):
    global prev_lines
    count_left, avg_rho_left, avg_theta_left = (0,0,0)
    count_right, avg_rho_right, avg_theta_right = (0,0,0)
    filtered_lines = []

    if lines is None:
        lines = prev_lines
    else:
        # Draw the two longest lines on the image
        for line in lines:
            rho, theta = line[0]
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

    #handle no lines found on frame
    if prev_lines is not None: 
        if count_left == 0:
            left_line = prev_lines[0]
        if count_right == 0:
            right_line = prev_lines[1]
        
    #save lines for the next frame if they are new
    if prev_lines is None and count_left > 0 and count_right > 0:
        filtered_lines = np.array([left_line,right_line])
        prev_lines = filtered_lines
    elif prev_lines is not None:
        filtered_lines = np.array([left_line,right_line])
        prev_lines = filtered_lines
    return filtered_lines
    
def collectLines(image):
    l_lines =cv2.HoughLines(image, rho=1, theta=np.pi / 90, threshold=60 ,min_theta=0.5,max_theta=1)##min_theta =math.pi/4, max_theta = math.pi/3.5)#return to -math.pi/2
    r_lines = cv2.HoughLines(image, rho=1, theta=np.pi / 90, threshold=60 ,min_theta=2.2,max_theta=3) ##=-math.pi/4, max_theta = -math.pi/4.5)#return to -math.pi/2
    if l_lines is not None and r_lines is not None:
        lines=np.concatenate((l_lines,r_lines))
    elif l_lines is not None:
        lines=l_lines
    elif r_lines is not None:
        lines=r_lines
    else:
        return None
    return lines
    
def region(original_frame,type,image=None):
    #focus on region of interest
    if type=="crop":
        roi=original_frame[roi_coordinates_focused[0]: roi_coordinates_focused[1], roi_coordinates_focused[2]:roi_coordinates_focused[3]]
        clipped_frame = roi.copy()
        return clipped_frame
    else:
        original_frame[roi_coordinates_focused[0]: roi_coordinates_focused[1], roi_coordinates_focused[2]:roi_coordinates_focused[3]] = image
        return original_frame

def drawLines(image,lines):
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
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return image

def collectPolynomial(image):
    # Find edge coordinates
    edge_coordinates_left = np.column_stack(np.where(image[:, :image.shape[1] // 2] > 0))
    edge_coordinates_right = np.column_stack(np.where(image[:, image.shape[1] // 2 :] > 0))

    

    cv2.imshow("help",image[:, image.shape[1] // 2 :])
    # Fit a polynomial (you can adjust the degree)
    degree_of_polynomial = 2
    x_values_left, y_values_left = edge_coordinates_left[:, 1], edge_coordinates_left[:, 0]
    x_values_right, y_values_right = edge_coordinates_right[:, 1], edge_coordinates_right[:, 0]
    coefficients_left = np.polyfit(x_values_left, y_values_left, degree_of_polynomial)
    coefficients_right = np.polyfit(x_values_right, y_values_right, degree_of_polynomial)


    # Generate points along the fitted polynomial curve
    x_curve_left = np.linspace(min(x_values_left), max(x_values_left), 1000)
    y_curve_left = np.polyval(coefficients_left, x_curve_left)
    x_curve_right = np.linspace(min(x_values_right), max(x_values_right), 1000)
    y_curve_right = np.polyval(coefficients_right, x_curve_right)
   
    poly_left = (x_curve_left, y_curve_left)
    poly_right = (x_curve_right, y_curve_right)

    return (poly_left, poly_right)

def drawPoly(image,poly):
    poly_left = poly[0]
    poly_right = poly[1]

    x_curve = poly_left[0]
    y_curve = poly_left[1]
    for i in range(len(x_curve) - 1):
        cv2.line(image, (int(x_curve[i]), int(y_curve[i])), (int(x_curve[i + 1]), int(y_curve[i + 1])), (0, 0, 255), 2)

    x_curve = poly_right[0]
    y_curve = poly_right[1]
    for i in range(len(x_curve) - 1):
        cv2.line(image, (int(x_curve[i]), int(y_curve[i])), (int(x_curve[i + 1]), int(y_curve[i + 1])), (0, 0, 255), 2)

    return image


def process_image(original_frame):
    

    ################################################################
    #focusing on region of interest   
    cropped_frame= region(original_frame,"crop")

    ################################################################
    #image manipulation
    manipulated_image = image_manipulation(cropped_frame)

    #################################################################
    #extracting lines
    lines = collectLines(manipulated_image)
    lines = filterLines(lines)

    poly = collectPolynomial(manipulated_image)
  

    
    #################################################################
    #creating result
    #cropped_image_with_lines = drawLines(cropped_frame, lines)
    cropped_image_with_lines = drawPoly(cropped_frame, poly)
    result = region(original_frame,"paste",cropped_image_with_lines)
    #################################################################

    return result


if __name__ == "__main__":

    cap = cv2.VideoCapture('Driving4.mp4')

    
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
