import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
from tensorflow.keras.models import model_from_json
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib

# Get root directory path
root_dir = os.getcwd()

# Load Model from Disk
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
json_file = open('antispoofing_models/antispoofing_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('antispoofing_models/antispoofing_model.h5')
print("Model loaded from disk")


def eye_aspect_ratio(eye):
    vertical_dist = dist.euclidean(eye[1], eye[5]) + dist.euclidean(eye[2], eye[4])
    horizontal_dist = dist.euclidean(eye[0], eye[3])
    ear = vertical_dist / (2.0 * horizontal_dist)
    return ear


BLINK_THRESHOLD = 0.25  # the threshold of the ear below which we assume that the eye is closed
CONSEC_FRAMES_NUMBER = 2  # minimal number of consecutive frames with a low enough ear value for a blink to be detected

ap = argparse.ArgumentParser(description='Eye blink detection')

# initialize dlib's face detector (HOG-based) and facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# choose indexes for the left and right eye
(left_s, left_e) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_s, right_e) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

counter = 0
total = 0
alert = False
start_time = 0

# Liveness Detection
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    try:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fr1 = frame.copy()
        fr2 = frame.copy()
        gr1 = cv2.cvtColor(fr1, cv2.COLOR_BGR2GRAY)
        gr2 = cv2.cvtColor(fr1, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gr2, 1.3, 5)
        rects = detector(gr1, 0)
        ear = 0
        # loop over the face detections:
        # determine the facial landmarks,
        # convert the facial landmark (x, y)-coordinates to a numpy array,
        # then extract the left and right eye coordinates,
        # and use them to compute the average eye aspect ratio for both eyes
        for rect in rects:
            shape = predictor(gr1, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[left_s:left_e]
            rightEye = shape[right_s:right_e]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # if the eye aspect ratio is below the threshold, increment counter
            # if the eyes are closed longer than for 2 secs, raise an alert
            if ear < BLINK_THRESHOLD:
                counter += 1
                if start_time == 0:
                    start_time = time.time()
                else:
                    end_time = time.time()
                    if end_time - start_time > 2: alert = True
            else:
                if counter >= CONSEC_FRAMES_NUMBER:
                    total += 1
                counter = 0
                start_time = 0
                alert = False

        # draw the total number of blinks and EAR value
        cv2.putText(frame, "Blinks: {}".format(total), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (500, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if alert:
            cv2.putText(frame, "ALERT!", (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for (x, y, w, h) in faces:
            face = fr2[y - 5:y + h + 5, x - 5:x + w + 5]
            resized_face = cv2.resize(face, (160, 160))
            resized_face = resized_face.astype("float") / 255.0
            # resized_face = img_to_array(resized_face)
            resized_face = np.expand_dims(resized_face, axis=0)
            # pass the face ROI through the trained liveness detector
            # model to determine if the face is "real" or "fake"
            preds = model.predict(resized_face)[0]
            color = (0, 0, 0)
            if preds > 0.5:
                label = 'spoof'
                color = (0, 0, 255)
            else:
                label = 'real'
                color = (0, 255, 0)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    except Exception as e:
        pass

video.release()
cv2.destroyAllWindows()
