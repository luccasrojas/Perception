#!/usr/bin/env python

# //======================================================================//
# //  This software is free: you can redistribute it and/or modify        //
# //  it under the terms of the GNU General Public License Version 3,     //
# //  as published by the Free Software Foundation.                       //
# //  This software is distributed in the hope that it will be useful,    //
# //  but WITHOUT ANY WARRANTY; without even the implied warranty of      //
# //  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE..  See the      //
# //  GNU General Public License for more details.                        //
# //  You should have received a copy of the GNU General Public License   //
# //  Version 3 in the file COPYING that came with this distribution.     //
# //  If not, see <http://www.gnu.org/licenses/>                          //
# //======================================================================//
# //                                                                      //
# //      Copyright (c) 2020 SinfonIA Pepper RoboCup Team                 //
# //      Sinfonia - Colombia                                             //
# //      https://sinfoniateam.github.io/sinfonia/index.html              //
# //                                                                      //
import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from yolo import Yolo_detection
from face_recog import Face_Recognition


class Recognition:

    def __init__(self):
        """This is the main class of the monocular depth estimation."""
        self.face = Face_Recognition(encodings_file="./src/perception_package/scripts/face_enc")
        # CvBridge to handle Image messages
        self.bridge = CvBridge()
        # Final labels
        self.finalLabelface = ""
        # This is the service client to communicate with the robot toolkit
        #self.visionToolsService = rospy.ServiceProxy('/robot_toolkit/vision_tools_srv', vision_tools_srv)

        # This is the message that will be sent trough the vision tools service
        #self.visionToolsMessage = vision_tools_msg()

        # Send a service to enable
        #self.visionToolsMessage.camera_name = "front_camera"
        #self.visionToolsMessage.command = "custom"
        #self.visionToolsMessage.resolution = 2
        #self.visionToolsMessage.frame_rate = 10
        #self.visionToolsMessage.color_space = 11
        #serviceResponse = self.visionToolsService(self.visionToolsMessage)
        self.FacePub=rospy.Publisher('FacePub', Image, queue_size=1)
        # Subscribe to front camera topic
        rospy.Subscriber("/robot_toolkit_node/camera/front/image_raw", Image, self.newImageCallback)
        # This is the service server to handle prediction requests
        #s = rospy.Service('perception_recognition_srv', perception_srv, self.handlePeceptionRecognition)
        # Keep node alive
        rospy.spin()

    def handlePeceptionRecognition(self, req):
        return self.finalLabelyolo + " " + self.finalLabelface

    def newImageCallback(self, imageMsg):
        self.imageData = self.bridge.imgmsg_to_cv2(imageMsg, "bgr8")
        self.FACEPredictions = self.face.recognition(self.imageData)
        result_face, labels_face = self.face.draw_face_detection(self.FACEPredictions, self.imageData)
        print("Labels finales (Face): ", labels_face)
        self.FacePub.publish(self.bridge.cv2_to_imgmsg(result_face,'bgr8'))

        if len(labels_face) > 0:
            self.finalLabelface = labels_face[0]
    

if __name__ == "__main__":
    rospy.init_node('object_recognition', anonymous=False)
    # Depth Estimation class initialization
    Recognition()