import cv2
import numpy as np
import poseEstimationModule as pm
import time
import os

path = os.path.dirname(os.path.abspath(__file__)) +"\\Videos\\08.mp4"
# print(path)

cap = cv2.VideoCapture(path)

detector = pm.poseDetector()

pTime = 0
count = 0
dir = 0 

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findPose(img,draw=False)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        # print(lmList[0]) #0 to 32 
        angleLA = detector.findAngle(img, 11, 13, 15, draw=True)  # Left Arm

        # angleRA =  detector.findAngle(img, 12, 14, 16, draw=True)  # Right Arm        
        # angleLL = detector.findAngle(img, 23, 25, 27, draw=True)  # Left Leg
        # angleRL = detector.findAngle(img, 24, 26, 28, draw=True)  # Right Leg

        per = np.interp(angleLA, (210, 310), (0, 100))
        bar = np.interp(angleLA, (220, 310), (240, 140))

        # print(angleLA ,per)
        
        # check for the duble curls

        color = (255,0,255)
        if per == 100:
            if dir == 0:
                color = (0, 255, 0)
                count += 0.5
                dir = 1
        
        if per == 0:
            if dir == 1:
                color = (0, 0, 255)
                count += 0.5
                dir = 0

        print(count)


        cv2.rectangle(img, (85, 240), (100, 140), color, 2)
        cv2.rectangle(img, (85, 340), (100, int(bar)), color, -1)

        cv2.putText(img, f'{int(per)}%', (83, 135), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

        
        cv2.rectangle(img, (0, 240), (80, 140), (0, 255, 0), -1)
        cv2.putText(img,f'Count', (8, 160), cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 0, 0), 2)
        cv2.putText(img,f'{int(count)}', (20, 230), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 10)
        
        # cv2.circle(img, (lmList[0][1], lmList[0][2]), 10, (255,0 , 255), -1)

    cTime = time.time()
    fps = 1 / (cTime - pTime)   
    pTime = cTime
    cv2.putText(img, f'FPS : {int(fps)}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1)  & 0xFF == ord('q'):
        break