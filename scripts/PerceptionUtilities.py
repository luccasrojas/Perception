#!/usr/bin/env python3
from tokenize import String
import rospy
import rospkg
import cv2
import os
import threading
import time
import subprocess
import rosservice
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

#Import subTools
import face


class PerceptionUtilities:
    def __init__(self) -> None:
        self.local=False
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
            self.local=False
            
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
            self.frontCameraRawSubscriber = rospy.Subscriber('/camera/image_raw', Image, self.callback_local_camera_subscriber)
            self.local=True
            


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
        self.CAMERA_FRONT = "image1"
        self.CAMERA_BOTTOM = "image2"
        self.CAMERA_DEPTH = "depth"

        self.PATH_PERCEPTION_UTLITIES = rospkg.RosPack().get_path('perception_utilities')
        self.PATH_DATA = self.PATH_PERCEPTION_UTLITIES+'/resources/data/'

        #Attributes
        self.bridge = CvBridge()

        self.image1up = False
        self.image2up = False
        self.depthup = False


        #CameraImageVariables
        self.image1 = None
        self.image2 = None
        self.depth = None

        
        self.labels = dict()
        self.getLabelsOn = False

        self.isLooking4Object = False
        self.objectBeingSearched = ''


        self.image1detection = False
        self.image2detection = False

        self.lookingLabels = list()
        

        #External File
        self.HAD=HAD.HAD()
        self.yoloDetect = YoloDetection.YoloDetection(n_cores=-1, confThreshold=0.35, nmsThreshold=0.6, inpWidth=416, use_gpu=True)
    
####################### CALLBACKS #######################
    
    def callback_front_camera_raw_subscriber(self, msg):
        self.image1 = msg

    #Callback for local pc camera    

    #TODO

    def callback_local_camera_subscriber(self, msg):
        self.image1 = msg
        if self.image1detection == True:
            self.imageData = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.imageData = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            self.Predictions = self.yoloDetect.Detection(self.imageData)
            result, labels, x_coordinates, y_coordinates = self.yoloDetect.Draw_detection(self.Predictions, self.imageData)
            self.yoloPublisher.publish(self.bridge.cv2_to_imgmsg(result,'bgr8'))

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
    

    def callback_front_camera_compressed_subscriber(self, msg):
        if self.image1detection == True:
            self.camera_compressed_subscriber(msg)

    def callback_bottom_camera_compressed_subscriber(self, msg):
        if self.image2detection == True:    
            self.camera_compressed_subscriber(msg)

    def camera_compressed_subscriber(self, msg):
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
        self.image2 = msg

    def callback_depth_camera_raw_subscriber(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        self.depth = cv2.cvtColor(self.depth,cv2.COLOR_BGR2GRAY)



####################### PUBLISHERS #########################

    #Local image publisher
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

    def publish_look_for_object(self, time, data):
        duration = rospy.Duration(time)
        beginTime = rospy.Time.now()
        endTime = beginTime + duration
        while rospy.Time.now() < endTime:
            self.lookForObjectPublisher.publish(data)
            rospy.sleep(0.2)

####################### TOOLKIT CAMERA CONTROL #########################
    #Turn on/off robot cameras via Toolkit
    def turn_on_front_and_bottom_camera(self):
        turn_camera_front = turn_camera_srvRequest()
        turn_camera_front.camera_name = self.CAMERA_FRONT
        turn_camera_front.enable = "enable"
        turn_camera_bottom = turn_camera_srvRequest()
        turn_camera_bottom.camera_name = self.CAMERA_BOTTOM
        turn_camera_bottom.enable = "enable"
        self.callback_turn_camera_srv(turn_camera_front)
        self.callback_turn_camera_srv(turn_camera_bottom)

#TODO
    def callback_turn_camera_srv(self, req):
        print(consoleFormatter.format("\nRequested turn camera service", "WARNING"))
        if req.camera_name in [self.CAMERA_FRONT, self.CAMERA_BOTTOM, self.CAMERA_DEPTH]:
            vision_request = vision_tools_msg()
            
            vision_request.command = req.enable
            # vision_request.camera_name = req.camera_name
            if req.camera_name == self.CAMERA_BOTTOM:
                #vision_request.camera_name+="_face_detector"
                vision_request.camera_name = 'bottom_camera'
                self.image2up = req.enable
                print(consoleFormatter.format("The "+req.camera_name+" was "+req.enable+"d", "OKBLUE"))
            elif req.camera_name == self.CAMERA_FRONT:
                #vision_request.camera_name+="_face_detector"
                vision_request.camera_name = 'front_camera'
                self.image1up = req.enable
                print(consoleFormatter.format("The "+req.camera_name+" was "+req.enable+"d", "OKBLUE"))
            elif req.camera_name == self.CAMERA_DEPTH:
                vision_request.camera_name = "depth_camera"
                self.depthup = req.enable
                print(consoleFormatter.format("The "+req.camera_name+" was "+req.enable+"d", "OKBLUE"))
            self.visionToolsServiceClient(vision_request)
            print(consoleFormatter.format('Turn camera service was executed successfully', 'OKGREEN'))
            return 'approved'
        else:
            print(consoleFormatter.format("The camera "+req.camera_name+" is not known.", "FAIL"))
            return 'not-approved'

################################# PREPROCESSING SERVICES #################################

    def callback_save_image_srv(self, req):
            print(consoleFormatter.format("\nRequested save image service", "WARNING"))
            if req.camera_name in [self.CAMERA_BOTTOM, self.CAMERA_FRONT]:
                if req.camera_name == self.CAMERA_FRONT:
                    image_raw = self.image1
                    if not self.image1up:
                        turn_camera = turn_camera_srvRequest()
                        turn_camera.camera_name = self.CAMERA_FRONT
                        turn_camera.enable = "enable"
                        self.callback_turn_camera_srv(turn_camera)
                elif req.camera_name == self.CAMERA_BOTTOM:
                    image_raw = self.image2
                    if not self.image2up:
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


##################################### FACE SERVICES #####################################
    """ All services offered by face recognition lay in the face.py module,
     all of them use only the front camera."""

    #RECOGNIZE FACE: Search the most similar face in the database and return the name of the person.
    def callback_recognize_face_srv(self, req):
        """ 
        INPUT: threshold(accuracy of the recognition), photo_name(name of the photo to be recognized(path))
        OUTPUT: person(name of the person recognized), result("approved" or "not-approved)
        """
        print(consoleFormatter.format("\nRequested recognize_face_service", "WARNING"))
        face_response = recognize_face_srvResponse()
        face_response.person, face_response.result = face.recognize_face(self,req)
        return face_response

    #SAVE FACE: Save a face in the database.
    def callback_save_face_srv(self, req):
        """ 
        INPUT: record_time(time to take pictures of the face), name(name of the person), n_pics(number of pictures to select from all the pictures taken)
        OUTPUT: save_face_response(True if the face was saved successfully or False if not)
        """
        print(consoleFormatter.format("\nRequested save face service", "WARNING"))
        return face.save_face(self,req)

    #GET PERSON DESCRIPTION: Given a picture, return the description of the person.
    def callback_get_person_description_srv(self, req):
        """ 
        INPUT: file_name(name of the photo to be recognized(path))
        OUTPUT: person_description_response(array of characteristics of the person)
        """
        print(consoleFormatter.format('\nRequested get person description service', 'WARNING'))
        return face.faceAttributes(self,req)
        # return self.HAD.getHumanAttributes(req.file_name)

##################################### OBJECT SERVICES #####################################
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
    
    def callback_start_recognition_srv(self, req):
        print(consoleFormatter.format("\nRequested start recognition service", "WARNING"))
        if req.camera_name == 'image2':
            self.image1detection = False
            self.image2detection = True
            print(consoleFormatter.format('The recognition service with bottom camera was started successfully', 'OKGREEN'))
        elif req.camera_name == 'image1':
            self.image1detection = True
            self.image2detection = False
            print(consoleFormatter.format('The recognition service with front camera was started successfully', 'OKGREEN'))
        else:
            self.image1detection = False
            self.image2detection = False
            print(consoleFormatter.format('The recognition service was stopped successfully', 'OKGREEN'))
        return 'approved'
    
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

if __name__ == '__main__':
    consoleFormatter=ConsoleFormatter.ConsoleFormatter()
    perceptionUtilities = PerceptionUtilities()
    print(consoleFormatter.format(" \n ----------------------------------------------------------", "OKGREEN"))  
    if perceptionUtilities.local:
        print(consoleFormatter.format(" --- local perception_utilities node initialized --- ", "OKGREEN"))
    else:
        print(consoleFormatter.format(" --- pepper perception_utilities node initialized --- ", "OKGREEN"))
    print(consoleFormatter.format(" ----------------------------------------------------------\n", "OKGREEN")) 

    rospy.spin()
