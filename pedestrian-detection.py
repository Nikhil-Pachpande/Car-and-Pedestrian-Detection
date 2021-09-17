# import the required libraries
import cv2
import numpy as np

# Creating the Classifier
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Initiate video capture for the video file
capture = cv2.VideoCapture('walking.avi')

# Loop once the video is successfully loaded
while capture.isOpened():

    # Read the first frame
    ret, frame = capture.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Pass the frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

    # Extract the bounding boxes for any bodies identified
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.imshow('Pedestrians', frame)

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

capture.release()
cv2.destroyAllWindows()