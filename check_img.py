import numpy as np
import matplotlib.pyplot as plt

camera_output = np.load('camera_output.npy', allow_pickle=True).item()

color_image = camera_output['color']
depth_image = camera_output['depth']
intrinsics_matrix = camera_output['intrinsics']

plt.figure(figsize=(10, 10))
plt.imshow(color_image)
plt.show()