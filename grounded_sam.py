from utils import *
from typing import List, Dict, Any
from transformers import pipeline, AutoModelForMaskGeneration, AutoProcessor

## Models.
DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM_MODEL  = "facebook/sam-vit-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def dino(image : Image.Image,
         labels : List[str],
         model_name : str = DINO_MODEL,
         threshold : float = 0.5,
          ) -> List[Dict[str, Any]]:
    
    object_detector = pipeline(model = model_name, task ='zero-shot-object-detection', device = DEVICE)
    results = object_detector(image, candidate_labels = labels, threshold = threshold)
    results = [DetectionResult.from_dict(result) for result in results]
    print("Number of detections by Dino : ", len(results))

    return results

def sam(
        image : Image.Image,
        detection_results : List[Dict[str, Any]],
        model_name : str = SAM_MODEL,
        polygon_refinement : bool = False) -> List[DetectionResult]:

    segmentor = AutoModelForMaskGeneration.from_pretrained(model_name).to(DEVICE)
    processor = AutoProcessor.from_pretrained(model_name)
    
    boxes = get_boxes(detection_results)
    inputs = processor(image, input_boxes=boxes, return_tensors="pt").to(DEVICE)
    outputs = segmentor(**inputs)
    masks = processor.post_process_masks(
        masks = outputs.pred_masks,
        original_sizes = inputs.original_sizes,
        reshaped_input_sizes = inputs.reshaped_input_sizes,
    )[0]

    masks = refine_mask(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

def grounded_sam(
        image : Image.Image,
        labels : List[str],
        threshold : float = 0.5,
        polygon_refinement : bool = False,
        detect_model : str = DINO_MODEL,
        segment_model : str = SAM_MODEL) -> Tuple[np.ndarray, List[DetectionResult]]:
    
    ## Detection.
    detections = dino(image, labels, detect_model, threshold)
    detections = sam(image, detections, segment_model, polygon_refinement)

    return np.array(image), detections

## Should be picked by by a LLM Think about combining the two.
image_from_camera = np.load('camera_output.npy', allow_pickle=True).item()

image_rgb = cv2.cvtColor(image_from_camera['color'], cv2.COLOR_BGR2RGB)
image = Image.fromarray(image_rgb)

# image = load_image('/home/rpmdt05/Code/grasp_it/output_1.png')
labels = ["red cube.",
           "green cube."]
threshold = 0.5

## Load a sample Image

## Make detections.
image, detections = grounded_sam(image,
                                labels,
                                threshold,
                                polygon_refinement=True,
                                detect_model=DINO_MODEL, 
                                segment_model=SAM_MODEL)

### Get the mask from detection results.
mask = np.zeros_like(detections[0].mask)
for detection in detections:
    mask[:,:] += detection.mask


mask = (mask != 0).astype(np.uint8)
print(mask.shape)
print(mask.max(), mask.min())
np.save('mask_from_grounded_SAM.npy', mask)
plot_detections(image, detections, "Final Output")



## Get the mask from the detections.
## Mask is filled mask can be directly fed into CGN.
# print(len(detections))
# mask = detections[0].mask
# print(mask.max(), mask.min())
# np.set_printoptions(threshold=np.inf)
# print(mask[200:250, 150:500])