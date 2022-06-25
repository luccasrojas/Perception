#!/usr/bin/env python3
import rospy
import rospkg
import os
import cv2
import time
import random
import shutil
import numpy as np
import face_recognition

from cv_bridge import CvBridge
from threading import Thread

from robot_toolkit_msgs.srv import vision_tools_srv
from robot_toolkit_msgs.msg import vision_tools_msg


from sensor_msgs.msg import Image, CompressedImage
from perception_msgs.srv import start_recognition_srv, get_labels_srv, save_image_srv, turn_camera_srv, turn_camera_srvRequest, get_person_description_srv, look_for_object_srv, recognize_face_srv,recognize_face_srvResponse,save_face_srv
from std_msgs.msg import Int32

import ConsoleFormatter
import HAD
import YoloDetection


class PerceptionUtilities:
    def __init__(self) -> None:

        #Service Clients
        print(consoleFormatter.format("Waiting for vision_tools service", "WARNING"))
        rospy.wait_for_service('/robot_toolkit/vision_tools_srv')
        print(consoleFormatter.format("vision_tools service connected!", "OKGREEN"))
        self.visionToolsServiceClient = rospy.ServiceProxy('/robot_toolkit/vision_tools_srv', vision_tools_srv)

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

        #Publishers
        self.yoloPublisher = rospy.Publisher('/perception_utilities/yolo_publisher', Image, queue_size=1)
        print(consoleFormatter.format("Yolo_publisher topic is up!","OKGREEN"))

        self.lookForObjectPublisher = rospy.Publisher('/perception_utilities/look_for_object_publisher', Int32, queue_size=10)
        print(consoleFormatter.format("Look_for_object topic is up!","OKGREEN"))

        #Subscribers
        self.frontCameraRawSubscriber = rospy.Subscriber('/robot_toolkit_node/camera/front/image_raw', Image, self.callback_front_camera_raw_subscriber)
        self.bottomCameraRawSubscriber = rospy.Subscriber('/robot_toolkit_node/camera/bottom/image_raw', Image, self.callback_bottom_camera_raw_subscriber)
        self.depthCameraRawSubscriber = rospy.Subscriber('/robot_toolkit_node/camera/depth/image_raw', Image, self.callback_depth_camera_raw_subscriber)
        self.frontCameraCompressedSubsciber = rospy.Subscriber("/robot_toolkit_node/camera/front/image_raw/compressed", CompressedImage, self.callback_front_camera_compressed_subscriber)

        #Constants
        self.CAMERA_FRONT = "front_camera"
        self.CAMERA_BOTTOM = "bottom_camera"
        self.CAMERA_DEPTH = "depth_camera"

        self.PATH_PERCEPTION_UTLITIES = rospkg.RosPack().get_path('perception_utilities')
        self.PATH_DATA = self.PATH_PERCEPTION_UTLITIES+'/resources/data/'

        #Attributes
        self.faceClassif = cv2.CascadeClassifier(self.PATH_PERCEPTION_UTLITIES+'/resources/model/'+'haarcascade_frontalface_default.xml')
        self.bridge = CvBridge()

        self.isFrontCameraUp = False
        self.isBottomCameraUp = False
        self.isDepthCameraUp = False

        self.front_image_raw = None
        self.bottom_image_raw = None
        self.depth_image_raw = None
        self.front_image_compressed = None
        
        self.labels = dict()
        self.getLabelsOn = False

        self.isLooking4Object = False
        self.objectBeingSearched = ''


        self.itsRunning = False

        self.lookingLabels = list()
        
        #Call Functions
        self.turn_on_front_and_bottom_camera()

        #External File
        self.HAD=HAD.HAD()
        self.yoloDetect = YoloDetection.YoloDetection(n_cores=-1, confThreshold=0.35, nmsThreshold=0.6, inpWidth=416, use_gpu=False)

        
    
    def callback_start_recognition_srv(self, req):
        print(consoleFormatter.format("\nRequested start recognition service", "WARNING"))
        self.itsRunning = req.state
        if self.itsRunning:
            print(consoleFormatter.format('The recognition service was started successfully', 'OKGREEN'))
        else:
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
        self.isLooking4Object = True
        self.objectBeingSearched = req.object
        self.labels = {}
        self.getLabelsOn = True
        print(consoleFormatter.format("The look for object service was started successfully", 'OKGREEN'))
        return "approved"

    def turn_on_front_and_bottom_camera(self):
        turn_camera_front = turn_camera_srvRequest()
        turn_camera_front.camera_name = self.CAMERA_FRONT
        turn_camera_front.enable = "enable"
        #turn_camera_bottom = turn_camera_srvRequest()
        #turn_camera_bottom.camera_name = self.CAMERA_BOTTOM
        #turn_camera_bottom.enable = "enable"
        self.callback_turn_camera_srv(turn_camera_front)
        #self.callback_turn_camera_srv(turn_camera_bottom)
        

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
                self.isFrontCameraUp = req.enable
                print(consoleFormatter.format("The "+req.camera_name+" was "+req.enable+"d", "OKBLUE"))
            else:
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

        if os.path.exists(face_test_path):
            folder_face = self.PATH_DATA+'faces'
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
            face_response.person = ""
            face_response.result = 'not-approved'

        return face_response

    def callback_save_face_srv(self, req):
            print(consoleFormatter.format("\nRequested save image service", "WARNING"))
            try:
                picsPath = self.PATH_DATA+"/pics"
                if not os.path.exists(picsPath+"/{}".format(req.name)):
                        os.makedirs(picsPath+"/{}".format(req.name))    
                if not self.isFrontCameraUp:
                    turn_camera = turn_camera_srvRequest()
                    turn_camera.camera_name = self.CAMERA_FRONT
                    turn_camera.enable = "enable"
                    self.callback_turn_camera_srv(turn_camera)
                start_time = time.time()
                stop_record = False
                facePath = self.PATH_DATA+"/faces"
                if not os.path.exists(facePath+"/{}".format(req.name)):
                        os.makedirs(facePath+"/{}".format(req.name))    
                personPath = picsPath+"/"+req.name
                personPath1 = facePath+"/"+req.name
                if not os.path.exists(personPath):
                        os.makedirs(personPath)
                siguiente = 1
                if len(os.listdir(personPath))!=0:
                    for imageName in os.listdir(personPath):
                        if int(imageName.replace(".jpg","")[-1])>siguiente:
                            siguiente = int(imageName.replace(".jpg","")[-1])  
                if len(os.listdir(personPath1))!=0:
                    for imageName in os.listdir(personPath1):
                        if int(imageName.replace(".jpg","")[-1])>siguiente:
                            siguiente = int(imageName.replace(".jpg","")[-1])  
                while stop_record!=True:
                    stop_time = time.time()
                    image_raw = self.front_image_raw
                    cv2_img = self.bridge.imgmsg_to_cv2(image_raw, 'bgr8')
                    faces = self.faceClassif.detectMultiScale(cv2_img, 1.3, 5)
                    copy = cv2_img.copy()
                    for (x,y,w,h) in faces:
                        cv2.rectangle(cv2_img, (x,y), (x+w, y+h), (0,255,0), 2)
                        rostro = copy[y:y+h, x:x+w]
                        rostro = cv2.resize(rostro, (150,150), interpolation=cv2.INTER_CUBIC)
                        cv2.imwrite(personPath+"/"+req.name+str(siguiente)+".jpg",rostro)
                        siguiente+=1
                    if(stop_time-start_time)>req.record_time:
                        stop_record = True
                numSaved = req.n_pics
                numPics = len(os.listdir(picsPath+"/{}".format(req.name)))
                picsPerRange = numPics//numSaved
                for i in range(numSaved):
                    pic = str(i*picsPerRange+random.randint(1,picsPerRange))
                    shutil.copy2(personPath+"/{}{}.jpg".format(req.name,pic),personPath1+"/{}{}".format(req.name,pic))
                print(consoleFormatter.format("The person: {} has been saved".format(req.name), "OKGREEN"))
                return True
            except:
                print(consoleFormatter.format("The image coudn't be saved", "FAIL"))
                return False






    def callback_front_camera_raw_subscriber(self, msg):
        self.front_image_raw = msg
    
    def callback_front_camera_compressed_subscriber(self, msg):
        self.front_image_compressed = msg
        if self.itsRunning == True:
            np_arr = np.frombuffer(msg.data, np.uint8)
            imageData = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)            
            predictions = self.yoloDetect.Detection(imageData)
            result, labels = self.yoloDetect.Draw_detection(predictions, imageData)
            self.yoloPublisher.publish(self.bridge.cv2_to_imgmsg(result, 'bgr8'))
            if len(labels) > 0 and self.getLabelsOn:
                for i in labels:
                    self.labels[i]=1
                    if self.isLooking4Object and i==self.objectBeingSearched:
                        print(consoleFormatter.format("The object: " +self.objectBeingSearched+" has been found", 'OKBLUE'))
                        thread = Thread(target=self.publish_look_for_object(3,1))
                        thread.start()
                        self.isLooking4Object = False
                        self.getLabelsOn = False

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
            rospy.sleep(0.5)

if __name__ == '__main__':
    consoleFormatter=ConsoleFormatter.ConsoleFormatter()
    rospy.init_node('perception_utilities')
    perceptionUtilities = PerceptionUtilities()
    print(consoleFormatter.format(" \n ----------------------------------------------------------", "OKGREEN"))  
    print(consoleFormatter.format(" --- perception_utilities node successfully initialized --- ", "OKGREEN"))
    print(consoleFormatter.format(" ----------------------------------------------------------\n", "OKGREEN")) 
    rospy.spin()
