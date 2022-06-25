#!/usr/bin/env python3
import rospkg
import cv2
import os
import numpy as np
import mxnet as mx

from mxnet_moon.lightened_moon import lightened_moon_feature

import ConsoleFormatter

class HAD:
    def __init__(self) -> None:
        self.consoleFormatter=ConsoleFormatter.ConsoleFormatter()

        #Constants
        self.PATH_PERCEPTION_UTLITIES = rospkg.RosPack().get_path('perception_utilities')

        self.PATH_FACE_PROTO = self.PATH_PERCEPTION_UTLITIES+'/resources/model/facenet/opencv_face_detector.pbtxt'
        self.PATH_FACE_MODEL = self.PATH_PERCEPTION_UTLITIES+'/resources/model/facenet/opencv_face_detector_uint8.pb'
        
        self.PATH_AGE_PROTO = self.PATH_PERCEPTION_UTLITIES+'/resources/model/age/age_deploy.prototxt'
        self.PATH_AGE_MODEL = self.PATH_PERCEPTION_UTLITIES+'/resources/model/age/age_net.caffemodel'

        self.PATH_GENDER_PROTO = self.PATH_PERCEPTION_UTLITIES+'/resources/model/gender/gender_deploy.prototxt'
        self.PATH_GENDER_MODEL = self.PATH_PERCEPTION_UTLITIES+'/resources/model/gender/gender_net.caffemodel'

        self.PATH_DATA = self.PATH_PERCEPTION_UTLITIES+'/resources/data/'

        #Models
        self.faceNet=cv2.dnn.readNet(self.PATH_FACE_MODEL, self.PATH_FACE_PROTO)
        self.ageNet=cv2.dnn.readNet(self.PATH_AGE_MODEL, self.PATH_AGE_PROTO)
        self.genderNet=cv2.dnn.readNet(self.PATH_GENDER_MODEL, self.PATH_GENDER_PROTO)

    def getHumanAttributes(self, file_name):
        self.PATH_IMAGE = self.PATH_DATA+file_name
        res = {"status": None, "gender": None, "age": None, "attributes":None}
        if os.path.exists(self.PATH_IMAGE):
            
            symbol = lightened_moon_feature(num_classes=40, use_fuse=True)
            devs = mx.cpu() 
            print('\033[95m')
            _, arg_params, aux_params = mx.model.load_checkpoint(self.PATH_PERCEPTION_UTLITIES+'/resources/model/lightened_moon/lightened_moon_fuse', 82)
            print('\033[0m')

            print(self.consoleFormatter.format('Image being processed: '+self.PATH_IMAGE+"\n", 'WARNING'))
            # read img and drat face rect
            image = cv2.imread(self.PATH_IMAGE)
            img = cv2.imread(self.PATH_IMAGE, -1)
            
            resultImg,faceBoxes=self.getFaceBox(self.faceNet,image)
            if not faceBoxes:
                print("No face detected")
                res["status"] = "failure"
            else:
                res["status"] = "success"

            # Loop throuth the coordinates
            for faceBox in faceBoxes:               

                gender,age = self.genderAge(image,faceBox)

                print(self.consoleFormatter.format('Gender: '+gender, 'OKBLUE'))
                res["gender"] = gender

                print(self.consoleFormatter.format('Age: '+age, 'OKBLUE'))
                res["age"] = age

                # Detect the facial attributes using mxnet
                # crop face area
                left = faceBox[0]
                width = faceBox[2] - faceBox[0]
                top = faceBox[1]
                height =  faceBox[3] - faceBox[1]
                right = faceBox[2]
                bottom = faceBox[3]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                pad = [0.25, 0.25, 0.25, 0.25]
                left = int(max(0, left - width*float(pad[0])))
                top = int(max(0, top - height*float(pad[1])))
                right = int(min(gray.shape[1], right + width*float(pad[2])))
                bottom = int(min(gray.shape[0], bottom + height*float(pad[3])))
                gray = gray[left:right, top:bottom]
                # resizing image and increasing the image size
                gray = cv2.resize(gray, (128, 128))/255.0
                img = np.expand_dims(np.expand_dims(gray, axis=0), axis=0)
                # get image parameter from mxnet
                arg_params['data'] = mx.nd.array(img, devs)
                
                print('\033[95m')
                exector = symbol.bind(devs, arg_params ,args_grad=None, grad_req="null", aux_states=aux_params)
                print('\033[0m')

                exector.forward(is_train=False)
                exector.outputs[0].wait_to_read()
                output = exector.outputs[0].asnumpy()
                # 40 facial attributes
                text = ["5_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes","Bald", "Bangs","Big_Lips","Big_Nose",
                        "Black_Hair","Blond_Hair","Blurry","Brown_Hair","Bushy_Eyebrows","Chubby","Double_Chin","Eyeglasses","Goatee",
                        "Gray_Hair", "Heavy_Makeup","High_Cheekbones","Male","Mouth_Slightly_Open","Mustache","Narrow_Eyes","No_Beard",
                        "Oval_Face","Pale_Skin","Pointy_Nose","Receding_Hairline","Rosy_Cheeks","Sideburns","Smiling","Straight_Hair",
                        "Wavy_Hair","Wearing_Earrings","Wearing_Hat","Wearing_Lipstick","Wearing_Necklace","Wearing_Necktie","Young"]
                
                #Predict the results
                pred = np.ones(40)
                # create a list based on the attributes generated.
                attrDict = {}
                detectedAttributeList = []
                for i in range(40):
                    attr = text[i].rjust(20)
                    if output[0][i] < 0:
                        attrDict[attr] = 'No'
                    else:
                        attrDict[attr] = 'Yes'
                        detectedAttributeList.append(text[i])

                res["attributes"] = detectedAttributeList
                for attribute in detectedAttributeList:
                    print(self.consoleFormatter.format('Attribute: '+attribute, 'OKBLUE'))
                
                # Write images into the results directory
                cv2.imwrite(self.PATH_PERCEPTION_UTLITIES+'/resources/results/'+str(file_name), resultImg) 
                print(self.consoleFormatter.format('Human Attribute Detection of '+str(file_name)+" was executed successfully", 'OKGREEN'))
        else:
            res["status"] = "failure"
            print(self.consoleFormatter.format('Get person description service rejected: '+str(file_name)+' not found', 'FAIL'))
        return res
    
    """ Detects face and extracts the coordinates"""
    def getFaceBox(self, net, image, conf_threshold=0.7):
        image=image.copy()
        imageHeight=image.shape[0]
        imageWidth=image.shape[1]
        blob=cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], True, False)
        net.setInput(blob)
        detections=net.forward()
        faceBoxes=[]
        for i in range(detections.shape[2]):
            confidence=detections[0,0,i,2]
            if confidence>conf_threshold:
                x1=int(detections[0,0,i,3]*imageWidth)
                y1=int(detections[0,0,i,4]*imageHeight)
                x2=int(detections[0,0,i,5]*imageWidth)
                y2=int(detections[0,0,i,6]*imageHeight)
                faceBoxes.append([x1,y1,x2,y2])
                cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), int(round(imageHeight/150)), 8)
        return image,faceBoxes

    """ Detects age and gender """
    def genderAge(self, image,faceBox):

        MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
        ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        genderList=['Male','Female']
        
        padding=20
        face=image[max(0,faceBox[1]-padding):
            min(faceBox[3]+padding,image.shape[0]-1),max(0,faceBox[0]-padding)
            :min(faceBox[2]+padding, image.shape[1]-1)]
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict the gender
        self.genderNet.setInput(blob)
        genderPreds=self.genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        # Predict the age
        self.ageNet.setInput(blob)
        agePreds=self.ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        # Return
        return gender,age