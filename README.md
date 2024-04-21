# CV_Lane_Detection
This repository contains a lane detection app that uses classical computervision techniques in order to find road lanes from a given
dashboard video and display them while also showing whether the driver is passing lanes. The app can also identify crosswalks and draws a bounding box around them.

# Example Videos
<p align="Left">
  <img src="Driving-passDay-output.mp4" alt="Video">
</p>


https://github.com/kom1323/CV_Lane_Detection/assets/28003020/1d19f5a9-da39-4979-9240-92ca39eed785

https://github.com/kom1323/CV_Lane_Detection/assets/28003020/57fc453a-6b57-4f7e-afc6-8052462a6d81

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
