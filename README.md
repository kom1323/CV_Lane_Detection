# CV_Lane_Detection
This repository contains a lane detection app that uses classical computervision techniques in order to find road lanes from a given
dashboard video and display them while also showing whether the driver is passing lanes. The app can also identify crosswalks and draws a bounding box around them.

# Example Videos


https://github.com/kom1323/CV_Lane_Detection/assets/28003020/0d6ce332-0fad-4544-9733-b92d651c39d4


https://github.com/kom1323/CV_Lane_Detection/assets/28003020/2fa27587-34a5-48b0-b4e5-4e19baee452d



https://github.com/kom1323/CV_Lane_Detection/assets/28003020/2bd6c0cc-e5cc-4ff6-b28c-dfbca2f6fcbd



# Try It Yourself

1. Clone this repo
2. Setup a Python virtual environment with all the Python dependencies based on [requirements.txt](requirements.txt).
3. Change the video file path to your video's path in line 223.
   
   `cap = cv2.VideoCapture(YOUR VIDEO PATH)`
4. Run **main.py**.

# ComputerVision Techniques 
Houghlines

Histogram equalizers - CLAHE

Template matching

BilateralFilter

Canny edge detection
