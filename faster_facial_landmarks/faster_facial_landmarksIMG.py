# USAGE
# python faster_facial_landmarks.py --shape-predictor shape_predictor_5_face_landmarks.dat

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2

# construct the argument parser and parse the arguments


# initialize dlib's face detector (HOG-based) and then create the
# facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# initialize the video stream and sleep for a bit, allowing the
# camera sensor to warm up

# loop over the frames from the video stream

        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 400 pixels, and convert it to
        # grayscale
        #frame = vs.read()

        #frame = imutils.resize(frame, width=400)
frame=cv2.imread("123.png")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
        # detect faces in the grayscale frame
rects = detector(gray, 0)

        # check to see if a face was detected, and if so, draw the total
        # number of faces on the frame
        #if len(rects) > 0:
        #       text = "{} face(s) found".format(len(rects))
        #       cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
        #               0.5, (0, 0, 255), 2)

        # loop over the face detections
for rect in rects:
        
        
                # compute the bounding box of the face and draw it on the
                # frame
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),
                        (0, 255, 0), 1)

                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
 
                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw each of them
        for (i, (x, y)) in enumerate(shape):
                print("["+str(x)+","+str(y)+"]")
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # show the frame
cv2.imshow("te1.jpg", frame)
frame=cv2.imread("test2.jpg")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
        # detect faces in the grayscale frame
rects = detector(gray, 0)

        # check to see if a face was detected, and if so, draw the total
        # number of faces on the frame
        #if len(rects) > 0:
        #       text = "{} face(s) found".format(len(rects))
        #       cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
        #               0.5, (0, 0, 255), 2)

        # loop over the face detections
for rect in rects:
        
                # compute the bounding box of the face and draw it on the
                # frame
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),
                        (0, 255, 0), 1)

                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
 
                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw each of them
        for (i, (x, y)) in enumerate(shape):
                print("["+str(x)+","+str(y)+"]")
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # show the frame
cv2.imshow("te2.jpg", frame)
