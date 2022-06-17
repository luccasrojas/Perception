#!/usr/bin/env python
import cv2
import os
import numpy as np
'''
Entrenamiento:
Todas las carpetas (con los nombres) est'an guardadas en /data, este script entrena TODAS las carpetas dentro, asocia cada foto dentro de
la carpeta al nombre de la carpeta. Este es el que genera el train model.xml (modelo entrenado)
'''
dataPath = "/home/luccas/Documents/sinfonIA/perception/Data"
peopleList = os.listdir(dataPath)
print('People in the database: ', peopleList)


labels = []
faceData = []
label = 0
for nameDir in peopleList:
    personPath = dataPath + "/" + nameDir
    for img in os.listdir(personPath):
        imgPath = personPath + "/" + img
        image = cv2.imread(imgPath, 0)
        faceData.append(image)
        labels.append(label)
    label += 1
cv2.destroyAllWindows()

face_recognizer = cv2.face.EigenFaceRecognizer_create()

print("Training the model...")
face_recognizer.train(faceData, np.array(labels))
face_recognizer.write("trained_model.xml")
print("Model trained successfully...")