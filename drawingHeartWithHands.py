import cv2
from cv2 import sqrt

import mediapipe as mp
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            arr = []
            for id, lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                if(id==4 or id==8): 
                    arr.append((cx,cy))
            if(len(arr)==2):
                d = math.sqrt((arr[0][0]-arr[1][0])**2+(arr[0][1]-arr[1][1])**2)
                if(d<=50):
                    ctX = (arr[0][0]+arr[1][0])//2
                    ctY = (arr[0][1]+arr[1][1])//2
                    r = 13
                    distance = 10
                    p1 = (ctX-distance-r+1,ctY+5)
                    p2 = (ctX+distance+r-1,ctY+5)
                    p3 = ((p1[0]+p2[0])//2,ctY+30)
                    triangle_forDraw = np.array([p1,p2,p3])
                    cv2.circle(img,(ctX-distance,ctY),r,(206, 95, 254), thickness = -1)
                    cv2.circle(img,(ctX+distance,ctY),r,(206, 95, 254), thickness = -1)
                    cv2.drawContours(img,[triangle_forDraw],0,(206, 95, 254),-1)
            # mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
    img = cv2.flip(img,1)
    cv2.imshow("Image",img)
    if cv2.waitKey(10) & 0xFF==ord('s'):
        break
    if not success:
        break