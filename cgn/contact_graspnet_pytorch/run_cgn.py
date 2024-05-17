from inference import inference
import config_utils

# ckpt_dir = 'checkpoints/contact_graspnet'
# input_paths = "test_data/output_depth_image_new.npy"
# local_regions = True
# filter_grasps = True
# skip_border_objects = False
# z_range = [0.2,1.8]
# forward_passes = 5
# K = None

# global_config = config_utils.load_config('checkpoints/contact_graspnet', batch_size=forward_passes, arg_configs=[])

# print(str(global_config))
# # print('pid: %s'%(str(os.getpid())))

### Contact GraspNet
global_config_cgn = config_utils.load_config('checkpoints/contact_graspnet')
pred_grasps, grasp_scores, contact_pts = inference(global_config_cgn,
                                        ckpt_dir = 'checkpoints/contact_graspnet',
                                        input_paths = "test_data/output_depth_image_new.npy",
                                        local_regions=True,
                                        filter_grasps=True,
                                        skip_border_objects=False,
                                        z_range = [0.2,1.8],
                                        forward_passes=1,
                                        K=None)

print(pred_grasps, grasp_scores, contact_pts)