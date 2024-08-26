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

def move_pose_back(target: list[6], distance = 0.01) -> list[6]:
    pose_matrix = make_matrix_from_tvec_and_rvec(target[:3], target[3:])
    approach_vector = pose_matrix[:3, 2]
    norm_approach_vector = approach_vector / np.linalg.norm(approach_vector)
    translate_vector = - distance * norm_approach_vector
    pose_matrix[:3, 3] = pose_matrix[:3, 3] + translate_vector

    target = pose_matrix[:3, 3].tolist() + cv2.Rodrigues(pose_matrix[:3, :3])[0].reshape(-1).tolist()
    return target

def jitter_pose(target: list[6], distance = 0.01) -> list[6]:
    random_jitter_x = np.random.uniform(-distance, distance)
    random_jitter_y = np.random.uniform(-distance, distance)
    random_jitter_z = np.random.uniform(-distance, distance)
    target[0] += random_jitter_x
    target[1] += random_jitter_y
    target[2] += random_jitter_z
    return target

def predict_target_from_prev(prev_target_configs: list[list[6]]) -> list[6]:
    """
    Predict the next target configuration from the previous target configurations.
    """
    diff = [prev_target_configs[1][i]-prev_target_configs[0][i] for i in range(6)]
    pred_target = [prev_target_configs[1][i]+diff[i] for i in range(6)]
    return pred_target

# def predict_target_from_prev(prev_target_configs: list[list[6]]) -> list[6]:
#     """
#     Predict the next target configuration from the previous target configurations.
#     """
#     pred_target = [0 for i in range(6)]
#     data_np = np.array(prev_target_configs)
#     for i in range(6):
#         model = ARIMA(data_np[:,i], order=(10,5,10))
#         model_fit = model.fit()
#         prediction = model_fit.forecast(steps=1)
#         pred_target[i] = prediction[0]
#     return pred_target

if __name__ == "__main__":
    print(np.random.uniform(-0.01, 0.01))