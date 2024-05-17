import numpy as np
import torch
import cv2
from PIL import Image
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

## Class to store the results.
@dataclass
class BoundingBox:
    xmin:int
    ymin:int
    xmax:int
    ymax:int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]
    
@dataclass
class DetectionResult:
    score:float
    label:str
    box:BoundingBox
    mask:Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(detection_dict['score'],
                   detection_dict['label'],
                    box=BoundingBox(xmin = detection_dict['box']['xmin'],
                                     ymin = detection_dict['box']['ymin'],
                                     xmax = detection_dict['box']['xmax'],
                                     ymax = detection_dict['box']['ymax']))
    
def annotate(image : Image.Image, detection_resutls : List[DetectionResult]) -> np.ndarray:
    ## Convert image to cv2 format.
    image_cv2 = np.array(image)
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    ## Iterate over detection and draw BB on image.
    for detection in detection_resutls:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        ## Random color for each class.
        color = np.random.randint(0, 256, size=3).tolist()

        ## Draw BB.
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color, 2)
        cv2.putText(image_cv2,
                    f"{label} {score:.2f}", (box.xmin, box.ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        ## Apply Mask.
        if mask is not None:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color, 2)

    ## Convert image back to RGB format.
    converted_image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    return converted_image

def plot_detections(image : Image.Image, detections : List[DetectionResult], save_name : Optional[str] = None) -> None:
    annotated_image = annotate(image, detections)

    plt.imshow(annotated_image)
    plt.axis('off')
    if save_name:
        plt.savefig(save_name)
    plt.show()

def random_named_css_color(num_colors : int) -> List[str]:
    named_css_colors = [
        'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond',
        'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
        'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey',
        'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
        'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
        'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite',
        'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory',
        'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
        'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray',
        'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
        'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
        'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive',
        'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip',
        'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
        'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey',
        'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white',
        'whitesmoke', 'yellow', 'yellowgreen'
    ]
    return random.sample(named_css_colors, min(num_colors, len(named_css_colors)))

def mask_to_polygon(mask : np.ndarray) -> List[List[int]]:
    ## Find contours.
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ## Find the largest contour.
    largest_contour = max(contours, key=cv2.contourArea)

    ## Get vertices of the polygon.
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon : List[Tuple[int, int]], image_shape : Tuple[int, int]) -> np.ndarray:
    ## Intialize mask.
    mask = np.zeros(image_shape, dtype=np.uint8)

    ## Convert polygon to np array.
    pts = np.array(polygon, np.int32)

    ## Fill the polygon with white color.
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def load_image(image_path : str) -> Image.Image:
    return Image.open(image_path).convert('RGB')

def get_boxes(results : DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)
    return [boxes]

def refine_mask(masks : torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0,2,3,1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask
    
    return masks
