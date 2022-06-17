#!/usr/bin/env python3
import cv2
import os
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

'''
Este script abre la camara y por cada frame que recibe, extrae la foto de la cara y la compara con el modelo
'''

dataPath = "/home/luccas/Documents/sinfonIA/perception/Data"
imagePaths = os.listdir(dataPath)
print(imagePaths)

face_recognizer = cv2.face.EigenFaceRecognizer_create()

face_recognizer.read('/home/luccas/Documents/sinfonIA/perception/trained_model_Luccas_Santiago.xml')

cap = cv2.VideoCapture(0)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret,frame = cap.read()
    if ret == False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150,150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)
        if result[1] < 6000:
            cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x+20,y-25),1,1.3,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        else:
            cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or k == 27:
        break
cap.release()
cv2.destroyAllWindows()
