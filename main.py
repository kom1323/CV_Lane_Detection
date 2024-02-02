import cv2
import numpy as np
import matplotlib.pyplot as plt



def show_image(img):

    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.colorbar()
    plt.show()


def process_image(frame, kind):
    focused_frame = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    roi_coordinates_focused = (600, 1080, 300, 1600)
    roi_coordinates_left = (0, 480, 300, 900)
    roi_coordinates_right = (0, 480, 1000, 1600)
    
    #focus on region of interest
    focused_frame = focused_frame[roi_coordinates_focused[0]: roi_coordinates_focused[1], roi_coordinates_focused[2]:roi_coordinates_focused[3]]
    original_focused = focused_frame.copy()
    
    focused_frame_left = focused_frame[roi_coordinates_left[0]: roi_coordinates_left[1], roi_coordinates_left[2]:roi_coordinates_left[3]]
    focused_frame_right = focused_frame[roi_coordinates_right[0]: roi_coordinates_right[1], roi_coordinates_right[2]:roi_coordinates_right[3]]

    
    focused_frame_left = cv2.cvtColor(focused_frame_left, cv2.COLOR_BGR2GRAY)
    focused_frame_right = cv2.cvtColor(focused_frame_right, cv2.COLOR_BGR2GRAY)


    d = 5  # edge size of neighborhood perimeter
    sigma_r = 100  # sigma range
    sigma_s = 100  # sigma spatial


    focused_frame_left = cv2.bilateralFilter(focused_frame_left , d, sigma_r, sigma_s)    
    edges_left = cv2.Canny(focused_frame_left , 100, 150, apertureSize=3)
    lines_left = cv2.HoughLines(edges_left, 1, np.pi / 180, threshold=200)


    focused_frame_right = cv2.bilateralFilter(focused_frame_right , d, sigma_r, sigma_s)    
    edges_right = cv2.Canny(focused_frame_right , 100, 150, apertureSize=3)
    
    lines_right = cv2.HoughLines(edges_right, 1, np.pi / 180, threshold=200)


    lines = np.concatenate((lines_left, lines_right), axis=0)

    if lines is None:
        return frame
    

    
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
        cv2.line(original_focused, (x1, y1), (x2, y2), (255, 0, 0), 2)


    frame[roi_coordinates_focused[0]: roi_coordinates_focused[1], roi_coordinates_focused[2]:roi_coordinates_focused[3]] = original_focused
    return frame


if __name__ == "__main__":

    cap = cv2.VideoCapture('Driving.mp4')
    cap2 = cv2.VideoCapture('Driving.mp4')

    
    
    
    while(cap.isOpened() and cap2.isOpened()):
        ret, frame = cap.read()
        ret2, frame2 = cap2.read()
        
        if ret and ret2:
            processed_frame = process_image(frame, 1)
            cv2.imshow('Lane Detection', processed_frame)
            # processed_frame2 = process_image(frame2, 2)
            # cv2.imshow('Lane Detection FOR 2', processed_frame2)
        else:
            break
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




    pass