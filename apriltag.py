import cv2
import numpy as np

# ## Generate a aruco marker.
# tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50), id=5, sidePixels=200,borderBits = 1)

# ## Save the marker to a file.
# cv2.imwrite("marker.png", tag)
# cv2.imshow("marker", tag)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## Load Image
camera_output = np.load('camera_output.npy', allow_pickle=True).item()
color_image = camera_output['color']
intrinsics_matrix = camera_output['intrinsics']

print("Intrinsics Matrix: \n", intrinsics_matrix)
ARUCO_DICT = cv2.aruco.DICT_6X6_50
ARUCO_PARAMETERS = cv2.aruco.DetectorParameters()
print(ARUCO_DICT)
print(ARUCO_PARAMETERS)

## Detect Aruco Marker in the image
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_parameters = ARUCO_PARAMETERS
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(color_image, aruco_dict, parameters=aruco_parameters)

## Numver of markers detected
print("Number of markers detected: ", len(corners))


## Draw the detected markers
color_image = cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

# get the pose of the markers.
rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, intrinsics_matrix, None)
print("rvecs: ", rvecs)
print("tvecs: ", tvecs)

## Draw the pose of the markers
for i in range(len(rvecs)):
    color_image = cv2.drawFrameAxes(color_image, intrinsics_matrix, None, rvecs[i], tvecs[i], 0.01)

## Convert Rvecs to Euler angles.
euler_angles = cv2.Rodrigues(rvecs[0])[0]
print("Euler Angles: ", euler_angles)


# Display the image
cv2.imshow("image", color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def get_aruco_pose():
    ARUCO_DICT = cv2.aruco.DICT_6X6_50
    ARUCO_PARAMETERS = cv2.aruco.DetectorParameters()

    ## Detect Aruco Marker in the image
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    aruco_parameters = ARUCO_PARAMETERS
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(color_image, aruco_dict, parameters=aruco_parameters)

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, intrinsics_matrix, None)
    return rvecs, tvecs