# Import the required libraries
import cv2
import time
import numpy as np

# Create the classifier
car_classifier = cv2.CascadeClassifier('haarcascade_car.xml')

# Initiate video capture for the video file
capture = cv2.VideoCapture('cars.avi')

# Loop once video is successfully loaded
while capture.isOpened():

    time.sleep(.05)
    # Read the first frame
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass the frame to our classifier
    cars = car_classifier.detectMultiScale(gray, 1.4, 2)

    # Extract bounding boxes for any bodies identified
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.imshow('Cars', frame)

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

capture.release()
cv2.destroyAllWindows()