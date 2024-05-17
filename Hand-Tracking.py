import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
ctime = 0
while True:
    success, img = cap.read()

    convertImg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(convertImg)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks :
        for handMarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handMarks,mphands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img,"fps:" + str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
