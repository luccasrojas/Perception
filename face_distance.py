#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import math

class FaceDistance:
	def __init__(self):

		self.bridge = CvBridge()

		self.camera_info_sub = message_filters.Subscriber('/robot_toolkit_node/camera/depth/camera_info', CameraInfo)

		self.image_sub = message_filters.Subscriber('/robot_toolkit_node/camera/front/image_raw', Image)
		self.depth_sub = message_filters.Subscriber('/robot_toolkit_node/camera/depth/image_raw', Image)


		self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub, self.camera_info_sub], queue_size=1, slop=0.5)
		self.ts.registerCallback(self.getFaceDistanceCallback)

		self.pub = rospy.Publisher('/perception_utilities/face_distance', Image, queue_size = 1)
		rospy.spin()

	def getFaceDistanceCallback(self, rgb_data, depth_data, camera_info):
		print('hola si hola')
		try:


			camera_info_k = np.array(camera_info.K)

			# Intrinsic camera matrix for the raw (distorted) images.
			#     [fx  0 cx]
			# K = [ 0 fy cy]
			#     [ 0  0  1]
    
			m_fx = camera_info.K[0]
			m_fy = camera_info.K[4]
			m_cx = camera_info.K[2]
			m_cy = camera_info.K[5]
			inv_fx = 1. / m_fx;
			inv_fy = 1. / m_fy;



			cv_rgb = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")			
			cv_rgb = cv2.pyrDown(cv_rgb)
			print(len(cv_rgb[0]))
			depth_image = self.bridge.imgmsg_to_cv2(depth_data, "32FC1")
			depth_array = np.array(depth_image, dtype=np.float32)
			#print('depth_array'+ str(len(depth_array[0])))
			cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
			depth_8 = (depth_array * 255).round().astype(np.uint8)
			#print(depth_8)
			cv_depth = np.zeros_like(cv_rgb)
			cv_depth[:, :, 0] = depth_8
			cv_depth[:, :, 1] = depth_8
			cv_depth[:, :, 2] = depth_8

			face_cascade = cv2.CascadeClassifier('./src/vision/resources/haarcascade_frontalface_default.xml')
			gray = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2GRAY)
			faces = face_cascade.detectMultiScale(gray, 1.3, 5)
			rgb_height, rgb_width, rgb_channels = cv_rgb.shape

			for(x,y,w,h) in faces:
				cv2.rectangle(cv_rgb,(x,y),(x+w,y+h),(255,0,0),2)
				cv2.rectangle(cv_depth,(x,y),(x+w,y+h),(255,0,0),2)
				cv2.rectangle(cv_rgb,(x+30,y+30),(x+w-30,y+h-30),(0,0,255),2)
				cv2.rectangle(cv_depth,(x+30,y+30),(x+w-30,y+h-30),(0,0,255),2)
				roi_depth = depth_image[y+30:y+h-30, x+30:x+w-30]

				n = 30
				suma = 0
				for i in range(0,roi_depth.shape[0]):
					for j in range(0, roi_depth.shape[1]):
						value = roi_depth.item(i, j)
						if value > 0:
							n = n+1
							suma = suma+value

				mean_z = suma/n

				point_z = mean_z * 0.001 #Distance in meters
				point_x = ((x + w/2) - m_cx) * point_z * inv_fx
				point_y = ((y + h/2) - m_cy) * point_z * inv_fy

				x_str = "X: "+str(format(point_x, '.2f'))
				y_str = "Y: "+str(format(point_y, '.2f'))
				z_str = "Z: "+str(format(point_z, '.2f'))

				cv2.putText(cv_rgb, x_str, (x+w, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)
				cv2.putText(cv_rgb, y_str, (x+w, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)
				cv2.putText(cv_rgb, z_str, (x+w, y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)
				

				dist = math.sqrt(point_x*point_x + point_y*point_y + point_z*point_z)
				dist_str = "Dist: "+str(format(dist, '.2f'))+ "m"

				cv2.putText(cv_rgb, dist_str,(x+w, y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv2.LINE_AA)


		except CvBridgeError as e:
			print(e)

		#rgdb = np.concatenate((cv_rgb, cv_depth), axis = 1)

		#Convert opencv format back to ROS format and publish result

		try:
			#faces_message = self.bridge.cv2_to_imgmsg(rgdb, "bgr8")
			faces_message = self.bridge.cv2_to_imgmsg(cv_rgb,"bgr8")
			self.pub.publish(faces_message)

		except CvBridgeError as e:
			print(e)


def main():
	rospy.init_node('face_distance')
	faced = FaceDistance()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")

if __name__ == '__main__':
	rospy.init_node('face_distance')
	FaceDistance()






