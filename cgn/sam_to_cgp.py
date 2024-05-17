import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

# Load the rgb and depth images
rgb_image = cv2.imread('/home/rpmdt05/Code/break-it/mohit/contact_graspnet_pytorch/test_data/intel_l515/color_image.png')

print("Shape of RGB Image : ",rgb_image.shape)
# print("Shape of Depth Image : ",depth_image.shape)

with open('/home/rpmdt05/Code/break-it/mohit/contact_graspnet_pytorch/test_data/intel_l515/depth_image_actual', 'rb') as f:
    depth_image = pickle.load(f)

with open('/home/rpmdt05/Code/break-it/mohit/contact_graspnet_pytorch/test_data/intel_l515/intrinsic_matrix', 'rb') as f:
    K = pickle.load(f)

with open('/home/rpmdt05/Code/break-it/mohit/contact_graspnet_pytorch/test_data/intel_l515/segmentation_masks', 'rb') as f:
    seg_mask = pickle.load(f)

print(len(seg_mask))
print(seg_mask[0].shape)

# Segmentation mask
seg = np.zeros((rgb_image.shape[0],rgb_image.shape[1]))
for i,mask in enumerate(seg_mask):
    indices = np.where(mask[0]==True)
    seg[indices] = i+1
# indices = np.where(seg==False)
# seg[indices] = len(seg_mask)+1



depth_image = depth_image*0.00025

#Visualize the images.
# cv2.imshow("RGB Image",seg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # Make a dictionary to store the images.
# image = {'rgb':rgb_image, 'depth':depth_image, 'K' : K ,'seg' : seg, 'segmap_id' : 2}
image = {'rgb':rgb_image, 'depth':depth_image, 'K' : K ,'seg' : seg}

print("Shape rgb : ",image['rgb'].shape)
print("Shape depth : ",image['depth'].shape)
print("Shape K : ",image['K'].shape)
print("Shape seg : ",image['seg'].shape)

# print("row of depth : ",image['seg'][200])

#Write a npy file
np.save('/home/rpmdt05/Code/break-it/mohit/contact_graspnet_pytorch/test_data/niru_input.npy',image)



