#!/usr/bin/env python
import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from perception_package_msgs.srv import init_recog_srv , get_labels_srv, get_labels_srvResponse
from yolo_prueba import Yolo_detection



class Recognition:

    def __init__(self):
        """This is the main class of the monocular depth estimation."""
        self.yolo_prueba = Yolo_detection(n_cores=1, confThreshold=0.35, nmsThreshold=0.6, inpWidth=416, use_gpu=False)
        # CvBridge to handle Image messages
        self.bridge = CvBridge()
        # Final labels
        self.labels = {}
        self.getLabelsOn = False
        """    Create all service Servers       """
        # Create a service to start/stop recongnition
        self.itsRunning = False
        self.startRecog = rospy.Service('perception_utilites/init_recognition_srv',init_recog_srv, self.callback_setRecog_srv)

        self.getObjLabels = rospy.Service('perception_utilites/get_labels_srv', get_labels_srv, self.callback_getLabels_srv)

        self.yoloPub=rospy.Publisher('yoloPub', Image, queue_size=1)
        # Subscribe to front camera topic
        rospy.Subscriber("/robot_toolkit_node/camera/front/image_raw", Image, self.newImageCallback)
        # Keep node alive
        rospy.spin()

    # Services callbacks
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


    def newImageCallback(self, imageMsg):
        if self.itsRunning == True:  
            self.imageData = self.bridge.imgmsg_to_cv2(imageMsg, "bgr8")
            #self.imageData = np.frombuffer(imageMsg.data, dtype=np.uint8).reshape(imageMsg.height, imageMsg.width, -1)
            self.Predictions = self.yolo_prueba.Detection(self.imageData)
            result, labels = self.yolo_prueba.Draw_detection(self.Predictions, self.imageData)
            print("Labels finales: ", labels)
            self.yoloPub.publish(self.bridge.cv2_to_imgmsg(result,'bgr8'))
            if len(labels) > 0 and self.getLabelsOn:
                for i in labels:
                    self.labels[i]=1



        

        #cv2.imshow("result", result)
        #cv2.imwrite('prueba.jpg',result)

if __name__ == "__main__":
    rospy.init_node('object_recognition', anonymous=False)
    # Depth Estimation class initialization
    Recognition()