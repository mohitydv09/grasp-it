## To read predictions
import numpy as np
import cv2


# Reading the predictions from the model.
data = np.load('/home/rpmdt05/Code/break-it/mohit/contact_graspnet_pytorch/results/predictions_output_depth_image_new.npz', allow_pickle=True)

print("Content of Data file : ",data.files)

# Dict of datas.
pred_grasps_cam = data['pred_grasps_cam'].item()
pc_full = data['pc_full']
scores = data['scores'].item()
contact_pts = data['contact_pts'].item()
# pc_colors = data['pc_colors'].item()

print("Keys in pred_grasps_cam : ",pred_grasps_cam.keys())
print("Shape of pc_full : ",pc_full.shape)
print("Keys in scores : ",scores.keys())
print("Keys in contact_pts : ",contact_pts.keys())
# print("Keys in pc_colors : ",pc_colors.keys())


# First item in.
print("First item in pred_grasps_cam : \n",pred_grasps_cam[1.0][0])
print("Second item in pred_grasps_cam : \n",pred_grasps_cam[1.0][1])
print("First item in scores : ",scores[1.0][0])
print("Second item in scores : ",scores[1.0][1])
print("First item in contact_pts : ",contact_pts[1.0][0])

# Print the content of the file.
# # print("Point cloud : \n",data['pc_full'])
# print(data['pred_grasps_cam'].item())
# print(data['scores'].shape)
# # print(data['contact_pts'])
# print(data['pc_colors'])