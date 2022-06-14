#!/usr/bin/env python
import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image,CompressedImage
from robot_toolkit_msgs.msg import vision_tools_msg
from robot_toolkit_msgs.srv import vision_tools_srv
from perception_package_msgs.srv import init_recog_srv , get_labels_srv, get_labels_srvResponse
from yolo_prueba import Yolo_detection



class Recognition:

    def __init__(self):
        """This is the main class of the monocular depth estimation."""
        self.yolo_prueba = Yolo_detection(n_cores=1, confThreshold=0.4, nmsThreshold=0.6, inpWidth=416, use_gpu=True)
        # CvBridge to handle Image messages
        self.bridge = CvBridge()
        # Final labels
        self.labels = {}
        self.getLabelsOn = False
        # This is the service client to communicate with the robot toolkit
        self.visionToolsService = rospy.ServiceProxy('/robot_toolkit/vision_tools_srv', vision_tools_srv)

        # This is the message that will be sent trough the vision tools service
        self.visionToolsMessage = vision_tools_msg()

        # Send a service to enable
        self.visionToolsMessage.camera_name = "front_camera"
        self.visionToolsMessage.command = "custom"
        self.visionToolsMessage.resolution = 2
        self.visionToolsMessage.frame_rate = 24
        self.visionToolsMessage.color_space = 11
        serviceResponse = self.visionToolsService(self.visionToolsMessage)

        # Create a service to start/stop recongnition
        #self.itsRunning = False
        #self.startRecog = rospy.Service('perception_utilites/init_ recognition_srv',init_recog_srv, self.setRecogCallback)


        self.yoloPub=rospy.Publisher('yoloPub', Image, queue_size=1)
        # Subscribe to front camera topic
        rospy.Subscriber("/robot_toolkit_node/camera/front/image_raw/compressed", CompressedImage, self.newImageCallback)
        # This is the service server to handle prediction requests
        #s = rospy.Service('perception_recognition_srv', perception_srv, self.handlePeceptionRecognition)
        # Keep node alive
        self.startRecog = rospy.Service('perception_utilites/init_recognition_srv',init_recog_srv, self.callback_setRecog_srv)

        self.getObjLabels = rospy.Service('perception_utilites/get_labels_srv', get_labels_srv, self.callback_getLabels_srv)
        rospy.spin()


    def callback_setRecog_srv(self, req):
        self.itsRunning = req.state
        return("Start recognition: "+str(self.itsRunning))

    def callback_getLabels_srv(self,req):
        if req.start:
            self.labels = {}
            self.getLabelsOn=True
            return get_labels_srvResponse("Running get labels")
        else:
            self.getLabelsOn=False
            obtainedLabels = []
            # while (rospy.get_rostime()).secs<timeOut:
            #     if self.finalLabel != None:
            #         for i in self.finalLabel:
            #             if i not in obtainedLabels:
            #                 obtainedLabels.append(i)   
            for label in self.labels:
                obtainedLabels.append(label)
            return get_labels_srvResponse(",".join(obtainedLabels))
    def handlePeceptionRecognition(self, req):
        return ""+ " " + self.finalLabel
        #self.finalLabelyolo 

    def newImageCallback(self, imageMsg):
        #if self.itsRunning == True:
            np_arr = np.fromstring(imageMsg.data, np.uint8)
            self.imageData = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            #self.imageData = self.bridge.imgmsg_to_cv2(imageMsg, "bgr8")
            #self.imageData = np.frombuffer(imageMsg.data, dtype=np.uint8).reshape(imageMsg.height, imageMsg.width, -1)
            self.Predictions = self.yolo_prueba.Detection(self.imageData)
            result, labels = self.yolo_prueba.Draw_detection(self.Predictions, self.imageData)
            print("Labels finales: ", labels)
            self.yoloPub.publish(self.bridge.cv2_to_imgmsg(result,'bgr8'))
            if len(labels) > 0:
                self.finalLabel = labels[0]
        

        

        #cv2.imshow("result", result)
        #cv2.imwrite('prueba.jpg',result)

if __name__ == "__main__":
    rospy.init_node('object_recognition', anonymous=False)
    # Depth Estimation class initialization
    Recognition()