import cv2
import numpy as np

class ArucoDetector:
    def __init__(self, color_image, intrinsics_matrix, distortion_matrix = None, visualization=False):
        self.color_image = color_image
        self.intrinsics_matrix = intrinsics_matrix
        self.distortion_matrix = distortion_matrix
        self.visualization = visualization
        self.aruco_size = 0.05 ## Meters.

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

    def detect_aruco(self):
        ## Detect Aruco Marker in the image
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(self.color_image, self.aruco_dict, parameters=self.aruco_params)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.aruco_size, self.intrinsics_matrix, self.distortion_matrix)
        print(f"Detected {len(corners)} markers in frame, with {len(rejectedImgPoints)} rejected frames.")

        if self.visualization:
            ## Draw the detected markers
            self.color_image = cv2.aruco.drawDetectedMarkers(self.color_image, corners, ids)
            for i in range(len(rvecs)):
                self.color_image = cv2.drawFrameAxes(self.color_image, self.intrinsics_matrix, self.distortion_matrix,
                                                      rvecs[i], tvecs[i], length=0.01, thickness=1) ## Last parameter is the size of the axis.

            ## Display the image
            cv2.imshow("Image with Detection : ", self.color_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return rvecs, tvecs

def marker_generator():
    ## Generate a aruco marker.
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50), id=5, sidePixels=200,borderBits = 1)

    ## Save the marker to a file.
    cv2.imwrite("marker.png", tag)
    cv2.imshow("marker", tag)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":

    ## Read From last Saved File.
    image_data = np.load('camera_output.npy', allow_pickle=True).item()
    color_image = image_data['color']
    intrinsics_matrix = image_data['intrinsics']

    aruco_detector = ArucoDetector(color_image, intrinsics_matrix, visualization=True)
    rvecs, tvecs = aruco_detector.detect_aruco()
