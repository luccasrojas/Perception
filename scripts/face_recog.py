#!/usr/bin/env python3

import face_recognition
import imutils
import pickle
import time
import cv2
import copy
import os

class Face_Recognition:
    
    def __init__(self, encodings_file="face_enc"):
        self.cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml" #find path of xml file containing haarcascade file     
        self.faceCascade = cv2.CascadeClassifier(self.cascPathface) # load the harcaascade in the cascade classifier
        self.data = pickle.loads(open(encodings_file, "rb").read()) # load the known faces and embeddings saved in last file
        #self.data = pickle.dump(encodings_file, open("encodings_2","wb"), protocol=2)

    def recognition(self, image): 
        print("Streaming started")
        # loop over frames from the video file stream
        cont = True
        while cont:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(gray,
                                                 scaleFactor=1.1,
                                                 minNeighbors=5,
                                                 minSize=(60, 60),
                                                 flags=cv2.CASCADE_SCALE_IMAGE) 
            # convert the input frame from BGR to RGB 
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #cv2.imshow("image",rgb)
            #cv2.waitKey(0)
            # the facial embeddings for face in input
            encodings = face_recognition.face_encodings(rgb)
            names = []
            #breakpoint()
            # loop over the facial embeddings incase we have multiple embeddings for multiple fcaes
            for encoding in encodings:
               #Compare encodings with encodings in data["encodings"]
               #Matches contain array with boolean values and True for the embeddings it matches closely and False for rest
                matches = face_recognition.compare_faces(self.data["encodings"], encoding)
                #set name = Unknown if no encoding matches
                name = "Unknown"
                # check to see if we have found a match
                if True in matches:
                    #Find positions at which we get True and store them
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    # loop over the matched indexes and maintain a count for each recognized face face
                    for i in matchedIdxs:
                        #Check the names at respective indexes we stored in matchedIdxs
                        name = self.data["names"][i]
                        #increase count for the name we got
                        counts[name] = counts.get(name, 0) + 1
                    #set name which has highest count
                    name = max(counts, key=counts.get)
                  
                # update the list of names
                names.append(name)
                cont = False

        print(names)

        return faces, names

    def draw_face_detection(self, recognition_output, image):

        image_ = copy.deepcopy(image)
        faces, names = recognition_output
        # loop over the recognized faces
        for ((x, y, w, h), name) in zip(faces, names):
        # rescale the face coordinates and draw the predicted face name on the image
            cv2.rectangle(image_, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image_, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        return image_, names

