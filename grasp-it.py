import numpy as np
from cgn.contact_graspnet_pytorch.inference import CGN
# contact_graspnet_pytorch/contact_graspnet_pytorch/inference.py
## Get input from realsence to get the depth image.


## Get Segmentation from Grounded SAM
image_is = np.load('camera_output.npy', allow_pickle=True).item()
print(image_is.keys())

mask_from_grounded_SAM = np.load('mask_from_grounded_SAM.npy')/255
print(mask_from_grounded_SAM.shape)
print(mask_from_grounded_SAM.max(), mask_from_grounded_SAM.min())


cgn_input = {}
cgn_input['rgb'] = image_is['color']
cgn_input['depth'] = image_is['depth']*0.00025
cgn_input['seg'] = mask_from_grounded_SAM
cgn_input['K'] = image_is['intrinsics']

np.save('cgn_input.npy', cgn_input)

## Get Grasps from CGN.
CGN_obj = CGN(input_path="cgn_input.npy", K=None, z_range = [0.8,1.5],visualize = True,forward_passes=6)
pred_grasps, grasp_scores, contact_pts, gripper_openings = CGN_obj.inference()