#!/usr/bin/env python3
from tokenize import String
import rospy
import rospkg
import cv2
import os
import threading
import time
import random
import shutil
import subprocess
import rosservice
import face_recognition
import numpy as np
from statistics import mean

from cv_bridge import CvBridge
from threading import Thread

from robot_toolkit_msgs.srv import vision_tools_srv
from robot_toolkit_msgs.msg import vision_tools_msg


from sensor_msgs.msg import Image, CompressedImage
from perception_msgs.srv import start_recognition_srv, get_labels_srv, save_image_srv, turn_camera_srv, turn_camera_srvRequest, get_person_description_srv, look_for_object_srv, recognize_face_srv,recognize_face_srvResponse,save_face_srv, set_model_recognition_srv, object_recognition_photo_srv, object_recognition_photo_srvResponse,crop_photo_object_recognition_srv,crop_photo_object_recognition_srvResponse
from perception_msgs.msg import get_labels_msg

from std_msgs.msg import Int32

import ConsoleFormatter
import HAD
import YoloDetection


class PerceptionUtilities:
    def __init__(self) -> None:
        local=False
        hayRos=False
        try:
            availableServices = rosservice.get_service_list()  
            hayRos=True
        except:
            hayRos=False
        #Correr en pepper
        if hayRos and '/robot_toolkit/vision_tools_srv' in availableServices:
            rospy.init_node('perception_utilities')
            print(consoleFormatter.format('RUNNING PERCEPTION IN PEPPER', 'OKGREEN'))
            #Service Clients
            print(consoleFormatter.format("Waiting for vision_tools service", "WARNING"))
            rospy.wait_for_service('/robot_toolkit/vision_tools_srv')
            print(consoleFormatter.format("vision_tools service connected!", "OKGREEN"))
            self.visionToolsServiceClient = rospy.ServiceProxy('/robot_toolkit/vision_tools_srv', vision_tools_srv)
            #Subscribers
            self.frontCameraRawSubscriber = rospy.Subscriber('/robot_toolkit_node/camera/front/image_raw', Image, self.callback_front_camera_raw_subscriber)
            self.bottomCameraRawSubscriber = rospy.Subscriber('/robot_toolkit_node/camera/bottom/image_raw', Image, self.callback_bottom_camera_raw_subscriber)
            self.depthCameraRawSubscriber = rospy.Subscriber('/robot_toolkit_node/camera/depth/image_raw', Image, self.callback_depth_camera_raw_subscriber)
            self.frontCameraCompressedSubsciber = rospy.Subscriber("/robot_toolkit_node/camera/front/image_raw/compressed", CompressedImage, self.callback_front_camera_compressed_subscriber)
            self.bottomCameraCompressedSubscriber = rospy.Subscriber("/robot_toolkit_node/camera/bottom/image_raw/compressed", CompressedImage, self.callback_bottom_camera_compressed_subscriber)
            #Call Functions
            local=False
            
        #Correr en pc
        else:
            print(consoleFormatter.format('RUNNING PERCEPTION IN PC', 'OKGREEN'))
            #Init ROS
            roscore = subprocess.Popen('roscore')
            time.sleep(1)
            #Init node
            rospy.init_node('perception_utilities')
            #Publisher
            x = threading.Thread(target=self.publish_image)
            x.start()
            #Subscribers
            self.frontCameraRawSubscriber = rospy.Subscriber('/camera/image_raw', Image, self.callback_local_camera)
            local=True
            


        #Service Servers
        self.startRecognitionServer = rospy.Service('perception_utilities/start_recognition_srv', start_recognition_srv, self.callback_start_recognition_srv)
        print(consoleFormatter.format('start_recognition_srv on!', 'OKGREEN'))

        self.getLabelsServer = rospy.Service('perception_utilities/get_labels_srv', get_labels_srv, self.callback_get_labels_srv)
        print(consoleFormatter.format('get_labels_srv on!', 'OKGREEN'))

        self.lookForObjectServer = rospy.Service('perception_utilities/look_for_object_srv', look_for_object_srv, self.callback_look_for_object_srv)
        print(consoleFormatter.format('look_for_object_srv on!', 'OKGREEN'))

        self.turnCameraServer = rospy.Service('perception_utilities/turn_camera_srv', turn_camera_srv, self.callback_turn_camera_srv)
        print(consoleFormatter.format('turn_camera_srv on!', 'OKGREEN'))

        self.saveImageServer = rospy.Service('perception_utilities/save_image_srv', save_image_srv, self.callback_save_image_srv)
        print(consoleFormatter.format('save_image_srv on!', 'OKGREEN'))

        self.getPersonDescriptionServer = rospy.Service('perception_utilities/get_person_description_srv', get_person_description_srv, self.callback_get_person_description_srv)
        print(consoleFormatter.format('get_person_description_srv on!', 'OKGREEN'))

        self.compareFacesServer = rospy.Service('perception_utilities/recognize_face_srv', recognize_face_srv, self.callback_recognize_face_srv)
        print(consoleFormatter.format('recognize_face_srv on!', 'OKGREEN'))

        self.saveFaceServer = rospy.Service('perception_utilities/save_face_srv', save_face_srv, self.callback_save_face_srv)
        print(consoleFormatter.format('save_face_srv on!', 'OKGREEN'))

        self.setModelRecognition = rospy.Service('perception_utilities/set_model_recognition_srv',set_model_recognition_srv, self.callback_set_model_recognition_srv)
        print(consoleFormatter.format('set_model_recognition_srv on!', 'OKGREEN'))

        self.objRecognitionPhoto = rospy.Service('perception_utilities/object_recognition_photo_srv',object_recognition_photo_srv, self.callback_object_recog_photo_srv)
        print(consoleFormatter.format('object_recongition_photo_srv on!', 'OKGREEN'))

        self.cropPhotoRecognition = rospy.Service('perception_utilities/crop_photo_object_recognition_srv',crop_photo_object_recognition_srv, self.callback_crop_photo_object_recognition_srv)


        #Publishers
        self.yoloPublisher = rospy.Publisher('/perception_utilities/yolo_publisher', Image, queue_size=1)
        print(consoleFormatter.format("Yolo_publisher topic is up!","OKGREEN"))

        self.lookForObjectPublisher = rospy.Publisher('/perception_utilities/look_for_object_publisher', Int32, queue_size=10)
        print(consoleFormatter.format("Look_for_object topic is up!","OKGREEN"))

        self.getLabelsPublisher = rospy.Publisher('/perception_utilities/get_labels_publisher', get_labels_msg, queue_size=10)
        print(consoleFormatter.format("Look_for_object topic is up!","OKGREEN"))

        
        #Constants
        self.CAMERA_FRONT = "front_camera"
        self.CAMERA_BOTTOM = "bottom_camera"
        self.CAMERA_DEPTH = "depth_camera"

        self.PATH_PERCEPTION_UTLITIES = rospkg.RosPack().get_path('perception_utilities')
        self.PATH_DATA = self.PATH_PERCEPTION_UTLITIES+'/resources/data/'

        #Attributes
        self.bridge = CvBridge()

        self.isFrontCameraUp = False
        self.isBottomCameraUp = False
        self.isDepthCameraUp = False

        self.front_image_raw = None
        self.bottom_image_raw = None
        self.depth_image_raw = None
        
        self.labels = dict()
        self.getLabelsOn = False

        self.isLooking4Object = False
        self.objectBeingSearched = ''


        self.frontItsRunning = False
        self.bottomItsRunning = False
        self.faceClassif = cv2.CascadeClassifier(self.PATH_PERCEPTION_UTLITIES+"/resources/model/haarcascade_frontalface_default.xml")

        self.lookingLabels = list()
        

        #External File
        self.HAD=HAD.HAD()
        self.yoloDetect = YoloDetection.YoloDetection(n_cores=-1, confThreshold=0.35, nmsThreshold=0.6, inpWidth=416, use_gpu=True)

        
    def publish_image(self):
        cap=cv2.VideoCapture(0)
        pub = rospy.Publisher("camera/image_raw", Image, queue_size = 1)
        rate = rospy.Rate(10)
        bridge = CvBridge()
        while not rospy.is_shutdown():
            ret, frame = cap.read()
            if not ret:
                break

            msg = bridge.cv2_to_imgmsg(frame, "bgr8")
            pub.publish(msg)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if rospy.is_shutdown():
                cap.release()

    def callback_start_recognition_srv(self, req):
        print(consoleFormatter.format("\nRequested start recognition service", "WARNING"))
        if req.camera_name == 'bottom':
            self.frontItsRunning = False
            self.bottomItsRunning = True
            print(consoleFormatter.format('The recognition service with bottom camera was started successfully', 'OKGREEN'))

        elif req.camera_name == 'front':
            self.frontItsRunning = True
            self.bottomItsRunning = False
            print(consoleFormatter.format('The recognition service with front camera was started successfully', 'OKGREEN'))
        else:
            self.frontItsRunning = False
            self.bottomItsRunning = False
            print(consoleFormatter.format('The recognition service was stopped successfully', 'OKGREEN'))
        return 'approved'
    
    def callback_get_labels_srv(self, req):
        print(consoleFormatter.format("\nRequested get labels service", "WARNING"))
        if req.start:
            self.labels = {}
            self.getLabelsOn=True
            print(consoleFormatter.format("The get labels service was started successfully", 'OKGREEN'))
            return "approved"
        else:
            self.getLabelsOn=False
            obtainedLabels = []
            for label in self.labels:
                obtainedLabels.append(label)
            print(consoleFormatter.format("The labels detected were: "+",".join(obtainedLabels), 'OKBLUE'))
            print(consoleFormatter.format("The get labels service was stopped successfully", 'OKGREEN'))
            return ",".join(obtainedLabels)
    
    def callback_look_for_object_srv(self, req):
        print(consoleFormatter.format("\nRequested look for object service", "WARNING"))
        if req.object != "":
            self.isLooking4Object = True
            self.objectBeingSearched = req.object
            self.labels = {}
            self.getLabelsOn = True
            print(consoleFormatter.format("The look for object service was started successfully", 'OKGREEN'))
        else:
            self.isLooking4Object = False
        return "approved"

    def turn_on_front_and_bottom_camera(self):
        turn_camera_front = turn_camera_srvRequest()
        turn_camera_front.camera_name = self.CAMERA_FRONT
        turn_camera_front.enable = "enable"
        turn_camera_bottom = turn_camera_srvRequest()
        turn_camera_bottom.camera_name = self.CAMERA_BOTTOM
        turn_camera_bottom.enable = "enable"
        self.callback_turn_camera_srv(turn_camera_front)
        self.callback_turn_camera_srv(turn_camera_bottom)
        

    def callback_turn_camera_srv(self, req):
        print(consoleFormatter.format("\nRequested turn camera service", "WARNING"))
        if req.camera_name in [self.CAMERA_FRONT, self.CAMERA_BOTTOM, self.CAMERA_DEPTH]:
            vision_request = vision_tools_msg()
            vision_request.camera_name = req.camera_name
            vision_request.command = req.enable
            if req.camera_name == self.CAMERA_BOTTOM:
                #vision_request.camera_name+="_face_detector"
                self.isBottomCameraUp = req.enable
                print(consoleFormatter.format("The "+req.camera_name+" was "+req.enable+"d", "OKBLUE"))
            elif req.camera_name == self.CAMERA_FRONT:
                #vision_request.camera_name+="_face_detector"
                self.isFrontCameraUp = req.enable
                print(consoleFormatter.format("The "+req.camera_name+" was "+req.enable+"d", "OKBLUE"))
            elif req.camera_name == self.CAMERA_DEPTH:
                self.isDepthCameraUp = req.enable
                print(consoleFormatter.format("The "+req.camera_name+" was "+req.enable+"d", "OKBLUE"))
            self.visionToolsServiceClient(vision_request)
            print(consoleFormatter.format('Turn camera service was executed successfully', 'OKGREEN'))
            return 'approved'
        else:
            print(consoleFormatter.format("The camera "+req.camera_name+" is not known.", "FAIL"))
            return 'not-approved'
        
    def callback_save_image_srv(self, req):
        print(consoleFormatter.format("\nRequested save image service", "WARNING"))
        if req.camera_name in [self.CAMERA_BOTTOM, self.CAMERA_FRONT]:
            if req.camera_name == self.CAMERA_FRONT:
                image_raw = self.front_image_raw
                if not self.isFrontCameraUp:
                    turn_camera = turn_camera_srvRequest()
                    turn_camera.camera_name = self.CAMERA_FRONT
                    turn_camera.enable = "enable"
                    self.callback_turn_camera_srv(turn_camera)
            elif req.camera_name == self.CAMERA_BOTTOM:
                image_raw = self.bottom_image_raw
                if not self.isBottomCameraUp:
                    turn_camera = turn_camera_srvRequest()
                    turn_camera.camera_name = self.CAMERA_BOTTOM
                    turn_camera.enable = "enable"
                    self.callback_turn_camera_srv(turn_camera)
                    print(consoleFormatter.format("Will try to save a photo to the path:", "WARNING"))
            cv2_img = self.bridge.imgmsg_to_cv2(image_raw, 'bgr8')
            cv2.imwrite(self.PATH_DATA+req.file_name, cv2_img)
            print(consoleFormatter.format("The image has been saved successfully", "OKGREEN"))
            return 'approved'
        else:
            print(consoleFormatter.format("Can't take a photo with camera: "+req.camera_name, "FAIL"))
            return 'not-approved'

    def callback_get_person_description_srv(self, req):
        print(consoleFormatter.format('\nRequested get person description service', 'WARNING'))
        return self.HAD.getHumanAttributes(req.file_name)

    def callback_recognize_face_srv(self, req):
        print(consoleFormatter.format("\nRequested recognize_face_service", "WARNING"))
        threshold =req.threshold
        face_test_path= self.PATH_DATA+req.photo_name
        face_response = recognize_face_srvResponse()
        try:
            if os.path.exists(face_test_path):

                folder_face = self.PATH_DATA+'faces'

                for folders in os.listdir(folder_face):
                    person_len_photo = len(os.listdir(folder_face+'/'+folders))
                    if person_len_photo == 0:
                        os.rmdir(folder_face+'/'+folders)
                        print("The face "+folders+"was delete succesfuly")

                known_face_names = [None]*len(os.listdir(folder_face))
                cont = 0
                for names in known_face_names:
                    known_face_names[cont] = os.listdir(folder_face)[cont]
                    cont+=1

                current_face = 0
                nFaces = [None]*len(os.listdir(folder_face))

                photos_face = []
                known_faces = []
                know_face_encodings = []
                           
                for face_list in os.listdir(folder_face):

                    specific_face_path= folder_face+'/'+face_list
                    for img_list in os.listdir(folder_face+'/'+face_list):
                       photo_face=img_list
                       photos_face.append(photo_face)

                    for current_photo in photos_face:
                        known_faces.append(face_recognition.load_image_file(specific_face_path+'/'+current_photo))


                    for current_photo in known_faces:
                        #print(current_photo)
                        know_face_encodings.append(face_recognition.face_encodings(current_photo)[0])


                    test_image = face_recognition.load_image_file(face_test_path)

                    face_locations = face_recognition.face_locations(test_image)
                    face_encodings = face_recognition.face_encodings(test_image,face_locations)

                    matches = None
                    name_of_the_face = "Unknow Person"
                    for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        matches = face_recognition.compare_faces(know_face_encodings,face_encoding)
                        #print(str(matches)+'->'+str(face_list))
                        n_trues = 0
                        for trues in matches:
                            if trues:
                                n_trues+=1
                        percent_true = n_trues/len(os.listdir(folder_face+'/'+face_list))
                        nFaces[current_face]= percent_true

                    current_face+=1
                    photos_face = []
                    known_faces = []    
                    know_face_encodings = []


                max_accert = 0
                if (max(nFaces)*100)>= threshold:
                    max_accert = max(nFaces)
                    index_max = nFaces.index(max_accert)
                    name_of_the_face = known_face_names[index_max]
                    summary = ""
                    contador=-1
                    for fresult in known_face_names:
                        contador +=1  
                        summary += " ["+ str(known_face_names[contador])+','+str(nFaces[contador])+']'
                        
                #print(consoleFormatter.format(summary,"OKBLUE"))
                print(consoleFormatter.format("Face name detected: "+ name_of_the_face+', '+str(max_accert*100)+'%',"OKBLUE"))
                print(consoleFormatter.format("Face recognition has been done successfully","OKGREEN"))



                face_response.person = name_of_the_face
                face_response.result = 'approved'
            else:
                print(consoleFormatter.format("The photo "+req.photo_name+" does not exist.","FAIL"))
                face_response.person = "NONE"
                face_response.result = 'not-approved'
        except:
            face_response.person = 'NONE'
            face_response.result = 'not-approved'
            print(consoleFormatter.format("It was not possible to recognize a person","FAIL"))

        return face_response


    def callback_save_face_srv(self, req):
        print(consoleFormatter.format("\nRequested save face service", "WARNING"))
        try:
            umbral=10
            record_time = req.record_time
            name= req.name
            nPics = req.n_pics
            picsPath = self.PATH_DATA+"pics"
            facePath = self.PATH_DATA+"/faces"
            stop_record = False
            picsPersonPath = picsPath+"/"+name
            facePersonPath = facePath+"/"+name

            #Creates the folder where all the pictures are going to be saved
            if not os.path.exists(picsPersonPath):
                    os.makedirs(picsPersonPath) 
            #Creates the folder of the faces in case it doesnt exist        
            if not os.path.exists(facePersonPath):
                            os.makedirs(facePersonPath)  
            #Turns on the camer ain case it not on          
            # if not self.isFrontCameraUp:
            #     turn_camera = turn_camera_srvRequest()
            #     turn_camera.camera_name = self.CAMERA_FRONT
            #     turn_camera.enable = "enable"
            #     self.callback_turn_camera_srv(turn_camera)

            siguiente = 0

            start =time.time()
            firstTime = True
            while stop_record!=True:
                actual = time.time()
                if not os.path.exists(picsPersonPath):
                    os.makedirs(picsPersonPath) 
                if(len(os.listdir(picsPersonPath)))>999 or start+record_time<actual:
                    stop_record = True
                #Image from the front camera
                #Posibilidad de cambiar esto a la imagen comprimida
                image_raw = self.front_image_raw
                cv2_img = self.bridge.imgmsg_to_cv2(image_raw, 'bgr8')
                largo = len(cv2_img[0])
                ancho = len(cv2_img)
                centro = largo//2

                faces = self.faceClassif.detectMultiScale(cv2_img, 1.3, 5)
                copy = cv2_img.copy()
                menor = 10000
                centinela = False
                outOfBounds =0
                centers = []
                for (x,y,w,h) in faces:
                    if x<20 or y<20 or x+w>largo-20 or y+h>ancho-20:
                        outOfBounds +=1
                        print("La cara se salió de la foto")
                        continue
                    x-=20
                    w+=40
                    y-=20
                    h+=40
                    centroCara = (x+(x+w))//2
                    if firstTime:
                        centroCaraAnterior = centroCara
                        firstTime= False
                    #print("El centro de la cara es",centroCara)
                    #diferencia con el centro
                    dif = centro-centroCara
                    centers.append(dif)
                    if centroCara > centroCaraAnterior-umbral and centroCara<centroCaraAnterior+umbral:
                        centroCaraAnterior=centroCara
                        centinela = True
                        # cv2.rectangle(cv2_img, (x,y), (x+w, y+h), (0,255,0), 2)
                        rostro = copy[y:y+h, x:x+w]
                        rostro = cv2.resize(rostro, (150,150), interpolation=cv2.INTER_CUBIC)
                if centinela:                    
                    siguiente+=1
                    cv2.imwrite(picsPersonPath+"/"+name+str(siguiente)+".jpg",rostro)
                    if centroCaraAnterior-centro<0:
                        menor = centro-centroCaraAnterior
                    else:
                        menor = centroCaraAnterior-centro
                    for diferencia in centers:
                        if diferencia-centro<0:
                            absDif = centro-diferencia
                        else:
                            absDif = diferencia-centro
                        #Se encontro una cara que se encuentra mas al centro que la anterior cara guardada    
                        if absDif < menor:
                            siguiente = 0
                            print(consoleFormatter.format("A face located nearer the center was found, restarting the save face process", "WARNING"))
                            centroCaraAnterior = diferencia
                            if os.path.isdir(picsPersonPath):
                                shutil.rmtree(picsPersonPath)
                        
            numSaved = nPics
            numPics = len(os.listdir(picsPersonPath))
            picsPerRange = numPics//numSaved

            #Saves random pictures from the pics file
            print(consoleFormatter.format("Saving random pictures from the saved ones", "WARNING"))
            for i in range(numSaved):
                pic = str(i*picsPerRange+random.randint(1,picsPerRange))
                shutil.copy2(picsPersonPath+"/{}{}.jpg".format(name,pic),facePersonPath+"/{}{}".format(name,pic)+".jpg")
            
            #Cleans the pictures for optimizing memory    
            if os.path.isdir(picsPersonPath):
                shutil.rmtree(picsPersonPath)
            print(consoleFormatter.format("The person: {} has been saved".format(name), "OKGREEN"))
            return True

        except:
            if os.path.isdir(picsPersonPath):
                shutil.rmtree(picsPersonPath)
                shutil.rmtree(facePersonPath)      
            print(consoleFormatter.format("The image coudn't be saved", "FAIL"))
            return False   

    # def callback_save_face_srv(self, req):
    #     print(consoleFormatter.format("\nRequested save face service", "WARNING"))
    #     #try:
    #     picsPath = self.PATH_DATA+"pics"
    #     if not os.path.exists(picsPath+"/{}".format(req.name)):
    #             os.makedirs(picsPath+"/{}".format(req.name))    
    #     # if not self.isFrontCameraUp:
    #     #     turn_camera = turn_camera_srvRequest()
    #     #     turn_camera.camera_name = self.CAMERA_FRONT
    #     #     turn_camera.enable = "enable"
    #         # self.callback_turn_camera_srv(turn_camera)
    #     stop_record = False
    #     facePath = self.PATH_DATA+"/faces"
    #     if not os.path.exists(facePath+"/{}".format(req.name)):
    #             os.makedirs(facePath+"/{}".format(req.name))    
    #     personPath = picsPath+"/"+req.name
    #     personPath1 = facePath+"/"+req.name
    #     if not os.path.exists(personPath):
    #             os.makedirs(personPath)
    #     siguiente = 0
    #     if len(os.listdir(personPath))!=0:
    #         for imageName in os.listdir(personPath):
    #             if int(imageName.replace(".jpg","")[-1])>siguiente:
    #                 siguiente = int(imageName.replace(".jpg","")[-1])  
    #     if len(os.listdir(personPath1))!=0:
    #         for imageName in os.listdir(personPath1):
    #             if int(imageName.replace(".jpg","")[-1])>siguiente:
    #                 siguiente = int(imageName.replace(".jpg","")[-1])  
    #     start_time = time.time()
    #     promedio = []            
    #     while len(promedio)<100 and not stop_record:
    #         stop_time = time.time()
    #         image_raw = self.front_image_raw
    #         cv2_img = self.bridge.imgmsg_to_cv2(image_raw, 'bgr8')
    #         largo = len(cv2_img[0])
    #         centro = largo//2
    #         faces = self.faceClassif.detectMultiScale(cv2_img, 1.3, 5)
    #         copy = cv2_img.copy()
    #         menor = 10000
    #         dif = None
    #         difabs = None
    #         for (x,y,w,h) in faces:
    #             if x<20 or y<20 or x+w>largo-20 or y+h>centro-20:
    #                 continue
    #             x-=20
    #             w+=40
    #             y-=20
    #             h+=40
    #             centroCara = (x+(x+w))//2
    #             print("El centro de la cara es",centroCara)
    #             #diferencia con el centro
    #             difabs = centro-centroCara
    #             if centroCara<centro:
    #                 difabs = centro-centroCara
    #             else:
    #                 difabs = centroCara-centro
    #         if len(faces)!=0 or difabs is not None:
    #             promedio.append(difabs)        
    #         if(stop_time-start_time)>10.0:
    #             stop_record = True   
    #     if stop_record:
    #         print(consoleFormatter.format("No face detected ", "FAIL"))
    #         if os.path.isdir(picsPath+"/{}".format(req.name)):
    #             shutil.rmtree(picsPath+"/{}".format(req.name))
    #             shutil.rmtree(facePath+"/{}".format(req.name))  
    #         return False    
    #     avg = mean(promedio)
    #     start_time = time.time()
    #     while stop_record!=True:
    #         stop_time = time.time()
    #         image_raw = self.front_image_raw
    #         cv2_img = self.bridge.imgmsg_to_cv2(image_raw, 'bgr8')
    #         largo = len(cv2_img[0])
    #         centro = largo//2
    #         print("EL centro de la imagen es:",centro)
    #         faces = self.faceClassif.detectMultiScale(cv2_img, 1.3, 5)
    #         copy = cv2_img.copy()
    #         menor = 10000
    #         centinela = False
    #         for (x,y,w,h) in faces:
    #             if x<20 or y<20 or x+w>largo-20 or y+h>centro-20:
    #                 print("La cara se salió de la foto")
    #                 continue
    #             x-=20
    #             w+=40
    #             y-=20
    #             h+=40
    #             centroCara = (x+(x+w))//2
    #             #print("El centro de la cara es",centroCara)
    #             #diferencia con el centro

    #             dif = centro-centroCara

    #             if centroCara<centro:
    #                 absdif = centro-centroCara
    #             else:
    #                 absdif = centroCara-centro
    #             #print("La diferencia es:",dif)
    #             if absdif<menor and dif > avg-40 and dif < avg+40:
    #                 centinela = True
    #                 cv2.rectangle(cv2_img, (x,y), (x+w, y+h), (0,255,0), 2)
    #                 rostro = copy[y:y+h, x:x+w]
    #                 rostro = cv2.resize(rostro, (150,150), interpolation=cv2.INTER_CUBIC)
    #                 menor = absdif
    #         if centinela:
    #             siguiente+=1
    #             cv2.imwrite(personPath+"/"+req.name+str(siguiente)+".jpg",rostro)
    #         if(stop_time-start_time)>req.record_time:
    #             stop_record = True
    #     numSaved = req.n_pics
    #     numPics = len(os.listdir(picsPath+"/{}".format(req.name)))
    #     picsPerRange = numPics//numSaved
    #     for i in range(numSaved):
    #         pic = str(i*picsPerRange+random.randint(1,picsPerRange))
    #         shutil.copy2(personPath+"/{}{}.jpg".format(req.name,pic),personPath1+"/{}{}".format(req.name,pic)+".jpg")
    #     print(consoleFormatter.format("The person: {} has been saved".format(req.name), "OKGREEN"))
    #     print(consoleFormatter.format("El promedi fue de: {}".format(avg), "OKGREEN"))
    #     return True
    #     #except:
    #         # if os.path.isdir(picsPath+"/{}".format(req.name)):
    #         #     shutil.rmtree(picsPath+"/{}".format(req.name))
    #         #     shutil.rmtree(facePath+"/{}".format(req.name))      
    #         # print(consoleFormatter.format("The image coudn't be saved", "FAIL"))
    #         # return False


    def callback_save_image_srv(self, req):
        print(consoleFormatter.format("\nRequested save image service", "WARNING"))
        if req.camera_name in [self.CAMERA_BOTTOM, self.CAMERA_FRONT]:
            if req.camera_name == self.CAMERA_FRONT:
                image_raw = self.front_image_raw
                if not self.isFrontCameraUp:
                    turn_camera = turn_camera_srvRequest()
                    turn_camera.camera_name = self.CAMERA_FRONT
                    turn_camera.enable = "enable"
                    self.callback_turn_camera_srv(turn_camera)
            elif req.camera_name == self.CAMERA_BOTTOM:
                image_raw = self.bottom_image_raw
                if not self.isBottomCameraUp:
                    turn_camera = turn_camera_srvRequest()
                    turn_camera.camera_name = self.CAMERA_BOTTOM
                    turn_camera.enable = "enable"
                    self.callback_turn_camera_srv(turn_camera)
                    print(consoleFormatter.format("Will try to save a photo to the path:", "WARNING"))
            cv2_img = self.bridge.imgmsg_to_cv2(image_raw, 'bgr8')
            cv2.imwrite(self.PATH_DATA+req.file_name, cv2_img)
            print(consoleFormatter.format("The image has been saved successfully", "OKGREEN"))
            return 'approved'
        else:
            print(consoleFormatter.format("Can't take a photo with camera: "+req.camera_name, "FAIL"))
            return 'not-approved'



    def callback_set_model_recognition_srv(self,req):
        print(consoleFormatter.format("\nRequested set model recognition service", "WARNING"))
        recog_model = req.model_name
        cfgFilepath = '/resources/yolo/yolov3-tiny_obj.cfg'
        weightFilepath = '/resources/yolo/yolov3-tiny_obj_last.weights'
        nameFilepath = '/resources/yolo/obj.names'
        if len(recog_model) !=0:
            cfgFilepath = '/resources/yolo/yolov3-tiny_'+recog_model+'.cfg'
            weightFilepath = '/resources/yolo/yolov3-tiny_'+recog_model+'.weights'
            nameFilepath = '/resources/yolo/obj_'+recog_model+'.names'
        self.yoloDetect= YoloDetection.YoloDetection(n_cores=-1, confThreshold=0.35, nmsThreshold=0.6, inpWidth=416, use_gpu=True,cfgFile=cfgFilepath, weightsFile =weightFilepath,namesFile=nameFilepath)
        print(consoleFormatter.format("The model has been changed successfully", "OKGREEN"))
        return 'approved'

    def callback_object_recog_photo_srv(self,req):
        print(consoleFormatter.format("\nRequested object recognition photo service", "WARNING"))
        photo_path = self.PATH_DATA+'/'+req.photo_name
        response = object_recognition_photo_srvResponse()
        try:
            photo = cv2.imread(photo_path)
            prediction = self.yoloDetect.Detection(photo)
            print(prediction)
            print(prediction[0][0][5])
            print(len(prediction[0]))
            result, labels, x_coordinates, y_coordinates = self.yoloDetect.Draw_detection(prediction,photo) 
            ############# ver para otro servicio (hacer recorte de una subfoto)#######
            print(labels)
            #cv2.imshow('frame',result)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            response.status = "approved"
            response.labels = ",".join(labels)
            print(consoleFormatter.format("The photo object recognition works successfully","OKGREEN"))
        except:
            response.status ="not-approved"
            response.status = "NONE"
            print(consoleFormatter.format("It was not possible to recognize objects in the photo","FAIL"))
        return response

    def callback_crop_photo_object_recognition_srv(self, req):
        print(consoleFormatter.format("\nRequested crop photo object recognition service","WARNING"))
        photo_path = self.PATH_DATA+'/'+req.photo_name
        object_2_recog = req.object
        response = crop_photo_object_recognition_srvResponse()
        #try:
        photo = cv2.imread(photo_path)
        predictions = self.yoloDetect.Detection(photo)
        img_obj = []
        for objects in predictions[0]:
            img_obj.append(objects[5])
        print(predictions[0])
        if object_2_recog in img_obj:
            folder_photo = self.PATH_DATA+req.photo_name.replace('.jpg','')
            if not os.path.exists(folder_photo):
                os.makedirs(folder_photo)                
            
            if img_obj.count(object_2_recog)==1:
                pos = img_obj.index(object_2_recog)
                xmin = int(predictions[0][pos][0])
                ymin = int(predictions[0][pos][1])
                ymax = int(predictions[0][pos][2])
                xmax = int(predictions[0][pos][3])
                crop_img = photo[xmin:xmax,ymin:ymax]
                print(folder_photo)
                cv2.imwrite(folder_photo+'/'+req.object+'.jpg',crop_img)
                response.result = "success"




            else:
                contador = 0
                for reqobj in img_obj:
                    if reqobj == object_2_recog:
                        pos = img_obj.index(object_2_recog)
                        xmin = int(predictions[0][pos][0])
                        ymin = int(predictions[0][pos][1])
                        xmax = int(predictions[0][pos][2])
                        ymax = int(predictions[0][pos][3])
                        crop_img = photo[xmin:xmax,ymin:ymax]
                        print(folder_photo)
                        cv2.imwrite(folder_photo+'/'+req.object+str(contador)+'.jpg',crop_img)
                    contador+=1

                print(predictions[0][pos])


            response.result = "success"



        else:
            response.result = "not-approved"
            print(consoleFormatter.format("The object you requested is not in the photo","FAIL"))
        
        return response





    def callback_front_camera_raw_subscriber(self, msg):
        self.front_image_raw = msg
        
    def callback_local_camera(self, msg):
        self.front_image_raw = msg
        #np_arr = np.fromstring(imageMsg.data, np.uint8)
        #self.imageData = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)            
        self.imageData = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.imageData = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        self.Predictions = self.yoloDetect.Detection(self.imageData)
        result, labels, x_coordinates, y_coordinates = self.yoloDetect.Draw_detection(self.Predictions, self.imageData)
        # print("Labels finales: ", labels)
        self.yoloPublisher.publish(self.bridge.cv2_to_imgmsg(result,'bgr8'))
        

    
    def callback_front_camera_compressed_subscriber(self, msg):
        if self.frontItsRunning == True:
            np_arr = np.frombuffer(msg.data, np.uint8)
            imageData = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)            
            predictions = self.yoloDetect.Detection(imageData)
            result, labels, x_coordinates, y_coordinates = self.yoloDetect.Draw_detection(predictions, imageData)
            self.yoloPublisher.publish(self.bridge.cv2_to_imgmsg(result, 'bgr8'))

            if len(labels) > 0 and self.getLabelsOn:   
                msgLabels = get_labels_msg()
                msgLabels.labels = list(map(str, labels))
                msgLabels.x_coordinates = x_coordinates
                msgLabels.y_coordinates = y_coordinates
                self.getLabelsPublisher.publish(msgLabels) 
                for i in range(len(labels)):
                    self.labels[labels[i]]=1
                    if self.isLooking4Object and labels[i]==self.objectBeingSearched:
                        print(consoleFormatter.format("The object: " +self.objectBeingSearched+" has been found", 'OKBLUE'))
                        thread = Thread(target=self.publish_look_for_object, args=(1,1,))
                        thread.start()

    def callback_bottom_camera_compressed_subscriber(self, msg):
        if self.bottomItsRunning == True:    
            np_arr = np.frombuffer(msg.data, np.uint8)
            imageData = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)            
            predictions = self.yoloDetect.Detection(imageData)
            result, labels, x_coordinates, y_coordinates = self.yoloDetect.Draw_detection(predictions, imageData)
            self.yoloPublisher.publish(self.bridge.cv2_to_imgmsg(result, 'bgr8'))

            if len(labels) > 0 and self.getLabelsOn:   
                msgLabels = get_labels_msg()
                msgLabels.labels = list(map(str, labels))
                msgLabels.x_coordinates = x_coordinates
                msgLabels.y_coordinates = y_coordinates
                self.getLabelsPublisher.publish(msgLabels) 
                for i in range(len(labels)):
                    self.labels[labels[i]]=1
                    if self.isLooking4Object and labels[i]==self.objectBeingSearched:
                        print(consoleFormatter.format("The object: " +self.objectBeingSearched+" has been found", 'OKBLUE'))
                        thread = Thread(target=self.publish_look_for_object, args=(1,1,))
                        thread.start()

    def callback_bottom_camera_raw_subscriber(self, msg):
        self.bottom_image_raw = msg

    def callback_depth_camera_raw_subscriber(self, msg):
        self.depth_image_raw = msg
        

    def publish_look_for_object(self, time, data):
        duration = rospy.Duration(time)
        beginTime = rospy.Time.now()
        endTime = beginTime + duration
        while rospy.Time.now() < endTime:
            self.lookForObjectPublisher.publish(data)
            rospy.sleep(0.2)

if __name__ == '__main__':
    consoleFormatter=ConsoleFormatter.ConsoleFormatter()
    perceptionUtilities = PerceptionUtilities()
    print(consoleFormatter.format(" \n ----------------------------------------------------------", "OKGREEN"))  
    print(consoleFormatter.format(" --- perception_utilities node successfully initialized --- ", "OKGREEN"))
    print(consoleFormatter.format(" ----------------------------------------------------------\n", "OKGREEN")) 
    rospy.spin()