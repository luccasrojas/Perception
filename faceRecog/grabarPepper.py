#!/usr/bin/env python
import cv2
import rospy 
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from robot_toolkit_msgs.msg import vision_tools_msg
from robot_toolkit_msgs.srv import vision_tools_srv

'''
Grabas video con la camara de pepper, termonar de grabar, press 's'
'''

salida = 0


print('-------------')

class Main:

    def __init__(self):
        self.salida = cv2.VideoWriter('videoSalida.mp4',cv2.VideoWriter_fourcc(*'mp4v'),20.0,(640,480))
        self.bridge = CvBridge()
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
        print('holis')
        rospy.Subscriber("/robot_toolkit_node/camera/front/image_raw", Image, self.newImageCallback)
        #while not rospy.is_shutdown():
        rospy.spin()

    def newImageCallback(self,imgmsg):
        print('holii')
        print("Object bla bla"+imgmsg)
        self.img= self.bridge.imgmsg_to_cv2(imgmsg,"bgr8")
        self.salida.write(self.img)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            self.salida.release()
            cv2.destroyAllWindows()

    
    

if __name__ == "__main__":
    print("entro")
    rospy.init_node('object_recognition', anonymous=False)
    # Depth Estimation class initialization
    Main()
    # Depth Estimation class initialization

