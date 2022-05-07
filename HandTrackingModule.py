import enum
import cv2 as cv
import numpy as np 
import time 
import mediapipe as mp
from torch import det 



class handDetector():
    def __init__(self, mode = False, maxHands=2, modelComplexity=1,
                detectionConfidence = 0.5, trackConfidence = 0.5):
        
        self.mode = mode 
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity,
                                        self.detectionConfidence, self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils    

    def findHands(self, img, draw =True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw == True:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
            
    def findPosition(self, img, handNo =0, draw =True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx , cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)

                lmList.append([id,cx,cy])
                if draw:
                    cv.circle(img, (cx,cy),5,(255,0,255), cv.FILLED)

        return lmList

def main():
    pTime = 0
    cTime = 0
    cap =  cv.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        img = cv.flip(img,1)
        # print(results.multi_hand_landmarks)
        lmList = detector.findPosition(img,draw=False)
        if len(lmList)!=0:
            print(lmList[4])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (10,70), 
                    cv.FONT_HERSHEY_COMPLEX, 1, (255,0,255), 2)
        cv.imshow("Image",img) 
        if cv.waitKey(1)  & 0xff == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()



if __name__=="__main__":
    main()


