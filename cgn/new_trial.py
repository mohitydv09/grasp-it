import numpy as np
import cv2

data = cv2.imread('/home/rpmdt05/Code/break-it/mohit/contact_graspnet_pytorch/test_data/0231_depth.png', cv2.IMREAD_UNCHANGED)

print("Shape of data : ",data.shape)
print("Max of data : ",np.max(data))
print("Min of data : ",np.min(data))

# np.set_printoptions(threshold=np.inf)
# print("Data :", data[:,:])

