from PIL import Image
import cv2
import numpy as np

def verify_customization(initial_image, customized_image):
    """
    Verifies whether a third pair of wings was added to the bee by counting distinct light-colored wing blobs in the customized image
    compared to the initial image.

    Returns a dict with counts and a score between 0.0 and 1.0.
    """
    def count_wing_blobs(pil_img):
        # Convert to OpenCV image (BGR)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        # Focus on central circular area if present
        # Convert to HSV to detect light/white wing shapes
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # White-ish mask: high V, low saturation
        lower = np.array([0, 0, 180])
        upper = np.array([180, 60, 255])
        mask = cv2.inRange(hsv, lower, upper)
        # Remove background circular blue by masking with brightness, then morphological ops
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (w*h)*0.002:  # ignore tiny specks relative to image size
                continue
            # bounding box
            x,y,ww,hh = cv2.boundingRect(cnt)
            blobs.append({'area': area, 'bbox': (x,y,ww,hh)})
        # Sort by area desc
        blobs = sorted(blobs, key=lambda b: -b['area'])
        return blobs, mask

    blobs_init, mask_init = count_wing_blobs(initial_image)
    blobs_cust, mask_cust = count_wing_blobs(customized_image)

    count_init = len(blobs_init)
    count_cust = len(blobs_cust)

    # Heuristic scoring:
    # If customized has at least 5-6 wing blobs -> likely added third pair (3 pairs -> 6 wings)
    # If it has 4 -> maybe two pairs (partial)
    # If <=2 -> not applied
    if count_cust >= 5:
        score = 1.0
    elif count_cust == 4:
        score = 0.6
    elif count_cust == 3:
        score = 0.5
    elif count_cust == 2:
        score = 0.0
    else:
        score = 0.0

    return {
        'initial_wing_blobs': count_init,
        'custom_wing_blobs': count_cust,
        'score': score,
        'mask_init': mask_init,
        'mask_custom': mask_cust
    }

from datasets import load_dataset
ds = load_dataset("CharlyR/VeriTikz", "full", split="train")
selected_row = ds[3]
original_code = selected_row["original_image"]
customized_code = selected_row["solution_image"]

print(verify_customization(original_code,customized_code))

