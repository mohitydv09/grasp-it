import numpy as np
import cv2

def make_matrix_from_tvec_and_rvec(tvec : list, rvec : list):
    """
    Create a 4x4 matrix from a list of translation vectors and a list of rotation vectors.
    """
    T = np.eye(4)
    T[:3, 3] = tvec
    T[:3,:3] = cv2.Rodrigues(np.array(rvec))[0]
    return T

def make_matrix_from_tvec_axis_angle(tvec : list, axis : str, angle : float):
    """
    Create a 4x4 matrix from an angle and a translation vector.
    """
    T = np.eye(4)
    if axis == "x":
        T[:3,:3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0]
    elif axis == "y":
        T[:3,:3] = cv2.Rodrigues(np.array([0, angle, 0]))[0]
    elif axis == "z":
        T[:3,:3] = cv2.Rodrigues(np.array([0, 0, angle]))[0]
    T[:3, 3] = tvec
    return T