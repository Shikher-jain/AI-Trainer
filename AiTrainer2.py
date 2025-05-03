import cv2
import numpy as np
import poseEstimationModule as pm
import time
import os

path = os.path.dirname(os.path.abspath(__file__)) +"\\Videos\\08.mp4"
cap = cv2.VideoCapture(path)

detector = pm.poseDetector()

pTime = 0

countLA = 0
dirLA = 0 

countRA = 0
dirRA = 0 

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findPose(img,draw=False)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        # print(lmList[0]) #0 to 32 

        # angleLL = detector.findAngle(img, 23, 25, 27, draw=True)  # Left Leg
        # angleRL = detector.findAngle(img, 24, 26, 28, draw=True)  # Right Leg

# LEFT ARM

        angleLA = detector.findAngle(img, 11, 13, 15, draw=True)  # Left Arm

        perLA = np.interp(angleLA, (210, 310), (0, 100))
        barLA = np.interp(angleLA, (220, 310), (240, 140))

        # print(angleLA ,perLA)
        
        # check for the duble curls

        color = (255,0,255)
        if perLA == 100:
            if dirLA == 0:
                color = (0, 255, 0)
                countLA += 0.5
                dirLA = 1
        
        if perLA == 0:
            if dirLA == 1:
                color = (0, 0, 255)
                countLA += 0.5
                dirLA = 0

        print(countLA)

        cv2.rectangle(img, (85, 240), (100, 140), color, 2)
        cv2.rectangle(img, (85, 340), (100, int(barLA)), color, -1)

        cv2.putText(img, f'{int(perLA)}%', (83, 135), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

        cv2.rectangle(img, (0, 240), (80, 140), (0, 255, 0), -1)
        cv2.putText(img,f'LEFT', (8, 160), cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 0, 0), 2)
        cv2.putText(img,f'{int(countLA)}', (20, 230), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 10)
        

# RIGHT ARM

        angleRA =  detector.findAngle(img, 12, 14, 16, draw=True)  # Right Arm        

        perRA = np.interp(angleRA, (210, 310), (0, 100))
        barRA = np.interp(angleRA, (220, 310), (240, 140))

        # print(angleRA ,perRA)
        
        # check for the duble curls

        color = (255,0,255)
        if perRA == 100:
            if dirRA == 0:
                color = (0, 255, 0)
                countRA += 0.5
                dirRA = 1
        
        if perRA == 0:
            if dirRA == 1:
                color = (0, 0, 255)
                countRA += 0.5
                dirRA = 0

        print(countRA)


        
        cv2.rectangle(img, (345, 240), (330, 140), color, 2)
        cv2.rectangle(img, (345, 340), (330, int(barRA)), color, -1)

        cv2.putText(img, f'{int(perRA)}%', (300, 135), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

        cv2.rectangle(img, (350, 240), (426, 140), (0, 255, 0), -1)
        cv2.putText(img,f'{int(countLA)}', (370, 230), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 10)
        cv2.putText(img,f'RIGHT', (360, 160), cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 0, 0), 2)
        

    cTime = time.time()
    fps = 1 / (cTime - pTime)   
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("AI Trainer", img)
    if cv2.waitKey(1)  & 0xFF == ord('q'):
        break