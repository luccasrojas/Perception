#!/usr/bin/env python3
import cv2
import os
import imutils
'''
importa in video y guarda en una carpeta(con el nobre de la cara) en escala de grises los recuadros donde est'e una cara

'''
name ="Santiago"
dataPath = "/home/luccas/Documents/sinfonIA/perception/Data"
personPath = dataPath + "/"+ name

if not os.path.exists(personPath):
    os.makedirs(personPath)

cap = cv2.VideoCapture("/home/luccas/Documents/sinfonIA/perception/Prueba/Santiago.mp4")
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count = 654

while True:
    ret,frame = cap.read()
    if ret == False:
        break
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150,150), interpolation=cv2.INTER_CUBIC)
        count += 1
        cv2.imwrite(personPath + "/" + name + str(count) + ".jpg", rostro)
    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)
    if k == ord('q') or k == 27 or count >800:
        break
cap.release()
cv2.destroyAllWindows()


