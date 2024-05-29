from scipy.spatial import distance as dist  #for calculating euclidean distance
from imutils.video import VideoStream       #for video stream handling and utility functions.
from imutils import face_utils
from threading import Thread                #for running tasks concurrently.
import numpy as np                          
import argparse                             #for parsing command-line arguments.
import imutils
import time
import dlib                                   #for facial landmark detection 
import cv2                                     # and image processing.
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client
import pyttsx3  # Import pyttsx3 for text-to-speech synthesis

# Function to send an email
def send_email(subject, body):
    sender_email = "sayanpaul369@gmail.com"
    receiver_email = "rajp68354@gmail.com"
    password = "vqhj yksi wkob ndvt"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Drowsiness Alert!!!"

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print("Failed to send email")
        print(e)

# Function to make a call using Twilio
def make_call():
    account_sid = 'ACcbab10afe34b813e5a377b1d552e8fd0'
    auth_token = '03931dbb676c88a7dcdff6bc7f4f2d2b'
    client = Client(account_sid, auth_token)

    call = client.calls.create(
        twiml='<Response><Say>Your attention is needed, drowsiness detected!</Say></Response>',
        to='+919804957595',
        from_='+15515505746'
    )

    print(f"Call initiated: {call.sid}")

def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    while alarm_status:
        # print('call')
        engine.say(msg)  # Speak the message
        engine.runAndWait()

    if alarm_status2:
        # print('call')
        saying = True
        engine.say(msg)  # Speak the message
        engine.runAndWait()
        saying = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0
# drowsiness_count = 0  # Counter for drowsiness detection events
CLOSED_EYE_CONSEC_FRAMES = 50


print("-> Loading the predictor and detector...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    # Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

email_sent = False  # To ensure email is sent only once per drowsiness event

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= CLOSED_EYE_CONSEC_FRAMES:
                if not alarm_status:
                    alarm_status = True
                    # drowsiness_count += 1  # Increment the drowsiness count
                    t = Thread(target=alarm, args=('wake up sir',))
                    t.daemon = True
                    t.start()

                if COUNTER >= 120:  # Make a call when drowsiness is detected thrice
                    make_call()
                    if not email_sent:
                        send_email("Drowsiness Alert", "The system has detected drowsiness.")
                        email_sent = True
                    COUNTER = 0
                      # Reset the count after making the call

                cv2.putText(frame, f"DROWSINESS ALERT! : {COUNTER}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            alarm_status = False
            email_sent = False  # Reset email flag when no drowsiness is detected

        if distance > YAWN_THRESH:
            cv2.putText(frame, "Yawn Alert!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not alarm_status2 and not saying:
                alarm_status2 = True
                t = Thread(target=alarm, args=('take some fresh air sir',))
                t.daemon = True
                t.start()
                  
        else:
            alarm_status2 = False

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()

