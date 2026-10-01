import cv2
import cv2.aruco as aruco

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
marker_id = 0
marker_size_px = 200  # print resolution, not physical size

marker_img = aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)
cv2.imwrite("sync_marker.png", marker_img)