#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

cap = cv2.VideoCapture(0)
print(cap.isOpened())

def pubImage():
	pub = rospy.Publisher("/robot_toolkit_node/camera/front/image_raw", Image, queue_size = 1)
	rospy.init_node('webcam_image', anonymous = False)
	rate = rospy.Rate(5)
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
		rate.sleep()

if __name__ == '__main__':
	try:
		pubImage()
	except rospy.ROSInterruptException:
		pass