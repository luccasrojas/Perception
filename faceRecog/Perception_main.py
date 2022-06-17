import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from robot_toolkit_msgs.msg import vision_tools_msg
from robot_toolkit_msgs.srv import vision_tools_srv
from perception_package_msgs.srv import perception_srv, perception_srvResponse
from yolo_prueba import Yolo_detection



class Recognition:

    def __init__(self):
        """This is the main class of the monocular depth estimation."""
        self.yolo_prueba = Yolo_detection(n_cores=1, confThreshold=0.4, nmsThreshold=0.6, inpWidth=416, use_gpu=False)
        # CvBridge to handle Image messages
        self.bridge = CvBridge()
        # Final labels
        self.finalLabel = ""
        # This is the service client to communicate with the robot toolkit
        self.visionToolsService = rospy.ServiceProxy('/robot_toolkit/vision_tools_srv', vision_tools_srv)

        # This is the message that will be sent trough the vision tools service
        self.visionToolsMessage = vision_tools_msg()

        # Send a service to enable
        self.visionToolsMessage.camera_name = "front_camera"
        self.visionToolsMessage.command = "custom"
        self.visionToolsMessage.resolution = 2
        self.visionToolsMessage.frame_rate = 10
        self.visionToolsMessage.color_space = 11
        serviceResponse = self.visionToolsService(self.visionToolsMessage)

        # Create a service to start/stop recongnition  

        self.yoloPub=rospy.Publisher('yoloPub', Image, queue_size=1)
        # Subscribe to front camera topic
        rospy.Subscriber("/robot_toolkit_node/camera/front/image_raw", Image, self.newImageCallback)
        # This is the service server to handle prediction requests
        s = rospy.Service('perception_recognition_srv', perception_srv, self.handlePeceptionRecognition)
        # Keep node alive
        rospy.spin()

    def handlePeceptionRecognition(self, req):
        resp = req.request
        if resp:
            return " "+self.finalLabel
        return ""+ " " + self.finalLabel
        #self.finalLabelyolo 

    def newImageCallback(self, imageMsg):
        self.imageData = self.bridge.imgmsg_to_cv2(imageMsg, "bgr8")
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