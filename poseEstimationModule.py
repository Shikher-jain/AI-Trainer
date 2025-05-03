import cv2
import mediapipe as mp
import time
import math

class poseDetector():
    
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.75, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth    
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        # self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,self.detectionCon, self.trackCon)
        self.pose = self.mpPose.Pose( static_image_mode=self.mode, model_complexity=1, smooth_landmarks=self.smooth, enable_segmentation=False, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList
    
    def findAngle(self, img, p1, p2, p3, draw=True):
        
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        
        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        
        if angle < 0:
            angle += 360
        
        print(angle)
        
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.line(img, (x3, y3), (x2, y2), (0, 0, 255), 4)
            cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 2)
            cv2.line(img, (x3, y3), (x2, y2), (255,255,255), 2)
            
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)

            # cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            
        return angle

def main():
    path = "F:/SHIKHER-VS/Regular/Advance-Python-SJ/Open CV/Advance/Projects/AI Trainer/Videos/01.mp4"
    # path = "F:/SHIKHER-VS/Advance-Python-SJ/Open CV/Advance/Pose Estimation/Videos/4.mp4"
    cap = cv2.VideoCapture(path)

    # cap = cv2.VideoCapture(0)

    pTime = 0
    detector = poseDetector()
    
    while True:
        success, img = cap.read()
        
        if not success:
            print("Error: Could not read frame from video.")
            break

        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        
        if len(lmList) != 0:
            print(lmList[0]) #0 to 32 
            cv2.circle(img, (lmList[0][1], lmList[0][2]), 10, (255,0 , 255), -1)
        
            detector.findAngle(img, 11, 13, 15, draw=True)  # Left Arm
            # detector.findAngle(img, 12, 14, 16, draw=True)  # Right Arm
            # detector.findAngle(img, 23, 25, 27, draw=True)  # Left Leg  
            # detector.findAngle(img, 24, 26, 28, draw=True)  # Right Leg

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 0), 3)
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
