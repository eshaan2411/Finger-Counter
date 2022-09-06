import cv2
import time
import os
import hand_tracking as ht

folder = "Fingers"
myList = os.listdir(folder)
overlayList = []

for imgpath in myList:
    img = cv2.imread(f"{folder}/{imgpath}")
    overlayList.append(img)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = ht.HandDetect(detectionConfidence=0.75)

# ID = [thumb, index, middle, ring, pinky]
fingerTipId = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.getHands(img)
    landmarkList = detector.getPosition(img, draw=False)

    # Example - If the index finger (8) is below mid-index finger (6), then consider it closed
    if len(landmarkList)!=0:
        fingers = []
        
        # Checking for thumb
        if landmarkList[4][1] > landmarkList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Checking for other 4 fingers
        for id in range(1, 5):
            if landmarkList[fingerTipId[id]][2] < landmarkList[fingerTipId[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)

        totalFingers = fingers.count(1)
        img[0:200, 0:200] = cv2.resize(overlayList[totalFingers], (200, 200))
        
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break