import cv2
import numpy as np
import matplotlib.pyplot as plt

import math
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

def display_lines(image, lines):
 lines_image = np.zeros_like(image)
 if lines is not None:
   for line in lines:
     x1, y1, x2, y2 = line
     cv2.line(lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
 return lines_image

def process_image(original_frame):
    copy = np.copy(original_frame)
    filtered_colors = filter_white_and_yellow(original_frame)
    gray_image= cv2.cvtColor(filtered_colors,cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image,(5,5),0)
    edges = cv2.Canny(blurred_image,50,150)
    region1_image=region(edges)
    lines = cv2.HoughLinesP(region1_image, rho=1, theta=np.pi/180, threshold=5, minLineLength=15, maxLineGap=3)
    print(lines)
    averaged_lines=average(original_frame,lines)
    print(averaged_lines)
    black_lines = display_lines(copy, averaged_lines)
    lanes = cv2.addWeighted(copy, 1, black_lines, 1, 1)
    return lanes;

def average(image, lines):
    left = []
    right = []
    avg_lines=[]
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_int = parameters[1]
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))
    if len(right)>0:
        right_avg = np.average(right, axis=0)
        avg_lines.append(make_points(image, right_avg))
    if len(left)>0:
        left_avg = np.average(left, axis=0)
        avg_lines.append(make_points(image, left_avg))

    return np.array(avg_lines)

def make_points(image, average): 
    slope, y_int = average 
    y1 = image.shape[0]
    y2 = int(np.maximum(0,y1 * (3/5)))
    x1 = int(np.maximum(0,int((y1 - y_int) // slope)))
    x2 = int(np.maximum(0,np.minimum(image.shape[1],(y2 - y_int) // slope)))
    return np.array([x1, y1, x2, y2])

def region(image):
    height, width = image.shape
    print()
    triangle = np.array([
                       [(int(width*0.2), height), (int(width*0.5), int(height*0.2)), (int(width*0.8), height)]
                       ])
    
    mask = np.zeros_like(image)
    
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask
if __name__ == "__main__":

    cap = cv2.VideoCapture('Driving3.mp4')
    # cap2 = cv2.VideoCapture('Driving.mp4')
    plt.figure(figsize=(20, 20))

    
    counter=1
    
    while(cap.isOpened() ): #and cap2.isOpened()):
        ret, frame = cap.read()
        # ret2, frame2 = cap2.read()
        print(f"frame {counter}")
        counter+=1
        if ret : #and ret2:
            processed_frame = process_image(frame)
            #cv2.resize(processed_frame,(processed_frame.shape[0]*2,processed_frame.shape[1]*2))
            cv2.imshow('Lane Detection',processed_frame)
            #cv2.imshow('Lane Detection', processed_frame)
            #processed_frame2 = process_image(frame2, 2)
            #show_image('Lane Detection', processed_frame)
        else:
            break
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
