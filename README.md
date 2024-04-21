# CV_Lane_Detection
This repository contains a lane detection app that uses classical computervision techniques in order to find road lanes from a given
dashboard video and display them while also showing whether the driver is passing lanes. The app can also identify crosswalks and draws a bounding box around them.

# Example Videos

https://github.com/kom1323/CV_Lane_Detection/assets/28003020/30da8501-6c78-4cd7-bd66-b04e533afe65

https://github.com/kom1323/CV_Lane_Detection/assets/28003020/21e1b866-86d9-4a97-9c29-12455c5e586e

https://github.com/kom1323/CV_Lane_Detection/assets/28003020/ed00314b-284e-48b6-af7b-707885832f62








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
