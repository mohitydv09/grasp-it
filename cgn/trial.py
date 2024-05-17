import numpy as np
import cv2

data = np.load('/home/rpmdt05/Code/break-it/mohit/contact_graspnet_pytorch/test_data/output_depth_image.npy', allow_pickle=True)
# data = np.load('/home/rpmdt05/Code/break-it/mohit/contact_graspnet_pytorch/test_data/pb_sim.npy', allow_pickle=True)
# data = np.load('/home/rpmdt05/Code/break-it/mohit/contact_graspnet_pytorch/test_data/niru_input.npy', allow_pickle=True)

# Reading the predictions from the model.
# data = np.load('/home/rpmdt05/Code/break-it/mohit/contact_graspnet_pytorch/results/predictions_niru.npz', allow_pickle=True)

# For Pybullet images.
# data = np.load('/home/rpmdt05/Code/break-it/mohit/contact_graspnet_pytorch/test_data/pb_depth_opengl.npy', allow_pickle=True)
# data = np.load('/home/rpmdt05/Code/break-it/mohit/contact_graspnet_pytorch/test_data/pb_depth_tiny.npy', allow_pickle=True)
# print("Data : ", data)
# print("Max of data : ",np.max(data))
# print("Min of data : ",np.min(data))


# For reading input.
# print(data)
print(type(data))
data = data.item()
keys = data.keys()
# print(data)

# print("Len of data shape :",len(data.shape))
# print("Shape of data : ", data.shape)

# print("Keys",keys)
# print(data['xyz'][150000:150010])
# # print(data['seg'].shape)
# print(data['K'].shape)
print(data['rgb'].shape)
print(data['depth'].shape)
print(data['seg'].shape)
print(data['K'])

np.set_printoptions(threshold=np.inf)
print(data['seg'][300])

data['seg'][data['seg']==1] = 0
data['seg'][data['seg']==2] = 1
data['seg'][data['seg']==3] = 2


# np.save('/home/rpmdt05/Code/break-it/mohit/contact_graspnet_pytorch/test_data/output_depth_image_newfile.npy',data)
np.save('output_depth_image_newfile.npy',data)

# # np.set_printoptions(threshold=np.inf)
# # print(data['depth'][100])
# print('max of depth : ',np.max(data['depth']))
# print( 'min of depth : ',np.min(data['depth']))
# # For reading predictions.
# keys = data.files
# print("Keys : ",keys)

# # print("Type of PC Full : ",type(data["pc_full"]))
# # print("PC Full : ",data["pc_full"])
# # print("Shape of PC Full : ",data["pc_full"].shape)

# print("Type of Pred Grasps Cam : ",type(data["pred_grasps_cam"]))
# print("shape of Pred Grasps Cam : ",data["pred_grasps_cam"].item)
# print("Pred Grasps Cam : ",type(data["pred_grasps_cam"]))

# print("Type of Scores : ",type(data["scores"]))
# print("Shape of Scores : ",data["scores"].shape)
# print("Scores : ",data["scores"])

# print("Type of Contact Points : ",type(data["contact_pts"]))
# print("Shape of Contact Points : ",data["contact_pts"].shape)
# print("Contact Points : ",data["contact_pts"])

# print("Type of PC Colors : ",type(data["pc_colors"]))
# print("Shape of PC Colors : ",data["pc_colors"].shape)
# print("PC Colors : ",data["pc_colors"])