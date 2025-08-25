import base64
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from io import BytesIO
import math
from typing import Sequence
from PIL import Image
import numpy as np


def write_base64_to_image(base64_str: str, output_path: str) -> None:
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]
    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(base64_str))


def get_used_pixels(image: Image.Image):
    used_pixels = 0
    for wp in range(image.width):
        for hp in range(image.height):
            used_pixels = used_pixels + (image.getpixel((wp, hp)) != (0, 0, 0))
    return used_pixels


def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_image(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def se(image1: Image.Image, image2: Image.Image):
    """Compute Mean Squared Error between two PIL images."""
    arr1 = np.array(image1, dtype=np.float32)
    arr2 = np.array(image2, dtype=np.float32)

    if arr1.shape != arr2.shape:
        raise ValueError("Images must have the same dimensions")

    max_val = max(arr1.max(), arr2.max())
    if max_val == 0:
        return 0.0  # Avoid divide-by-zero

    # Normalize to [0, 1]
    arr1 /= max_val
    arr2 /= max_val

    return np.sum((arr1 - arr2) ** 2)


def mse(image1: Image.Image, image2: Image.Image):
    return se(image1, image2) / (math.prod(image1.size))


def adjust_bbox(box, image: Image.Image):
    adjust = lambda box_k, cursize: (box_k / 1000) * cursize
    box_2d = box["box_2d"]
    new_box = (
        int(adjust(box_2d[1], image.width)),
        int(adjust(box_2d[0], image.height)),
        int(adjust(box_2d[3], image.width)),
        int(adjust(box_2d[2], image.height)),
    )
    box["box_2d"] = new_box
    return box


import cv2 as cv
import numpy.typing as npt





### Template matching methods(useless at the end of the day)
def get_template_matches_multiscale(
    template: Image.Image,
    image: npt.NDArray,
    threshold: float = 0.1,
    scales: list[float] = [i / 100 for i in range(10, 300, 1)],
) -> list[tuple[int, int, int, int]]:
    """Identifies templates in an image using multiscale matching."""
    method = cv.TM_SQDIFF_NORMED
    image = cv.Laplacian(image, 0)
    imageB, imageG, imageR = cv.split(image)

    template_np = np.array(template)
    template_np = cv.cvtColor(template_np, cv.COLOR_RGB2BGR)
    template_np = cv.Laplacian(template_np, 0)

    boxes = []

    for scale in scales:
        # Resize template
        scaled_template = cv.resize(
            template_np, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR
        )
        th, tw = scaled_template.shape[:2]
        if th > image.shape[0] or tw > image.shape[1]:
            continue  # Skip oversized templates

        # Split channels
        tB, tG, tR = cv.split(scaled_template)

        # Match on each channel
        resB = cv.matchTemplate(imageB, tB, method)
        resG = cv.matchTemplate(imageG, tG, method)
        resR = cv.matchTemplate(imageR, tR, method)
        res = resB + resG + resR

        loc = np.where(res <= threshold)
        for pt in zip(*loc[::-1]):
            boxes.append((pt[0], pt[1], pt[0] + tw, pt[1] + th))

    return box_similarity_removal(boxes)


def get_template_matches(
    template: Image.Image, image: npt.NDArray, threshold: float = 0.1
) -> list[tuple[int, int, int, int]]:
    """Identifies templates in an image

    Args:
        template (Image.Image): the template to identify in the image.
        image (npt.NDArray): the image to identify the template in. In a opencv NDArray BGR format.
    """
    w = template.width
    h = template.height
    method = cv.TM_SQDIFF_NORMED  # best for exact pattern match
    template = cv.cvtColor(np.array(template), cv.COLOR_RGB2BGR)
    template = cv.Laplacian(template, 0)
    image = cv.Laplacian(image, 0)

    # getting each color component of the image
    imageB, imageG, imageR = cv.split(image)
    TemplateB, TemplateG, TemplateR = cv.split(template)

    # Apply template Matching on all components
    resB = cv.matchTemplate(imageB, TemplateB, method)
    resG = cv.matchTemplate(imageG, TemplateG, method)
    resR = cv.matchTemplate(imageR, TemplateR, method)

    res = resB + resG + resR

    loc = np.where(res <= threshold)

    detected_boxes: list[tuple[int, int, int, int]] = []
    for pt in zip(*loc[::-1]):
        detected_boxes.append((pt[0], pt[1], pt[0] + w, pt[1] + h))

    return box_similarity_removal(detected_boxes)


def get_feature_matches(
    template: Image.Image, image: npt.NDArray, threshold: float = 100
) -> tuple[Sequence[cv.DMatch], Sequence[cv.KeyPoint], Sequence[cv.KeyPoint]]:
    """Identifies templates in an image

    Args:
        template (Image.Image): the template to identify in the image.
        image (npt.NDArray): the image to identify the template in. In a opencv NDArray BGR format.
    """

    template_cv = cv.cvtColor(np.array(template), cv.COLOR_RGB2GRAY)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    orb = cv.ORB.create()

    kp1, des1 = orb.detectAndCompute(template_cv, None)
    kp2, des2 = orb.detectAndCompute(image_gray, None)

    # create BFMatcher object
    bf = cv.BFMatcher()
    matches = bf.match(des1, des2)
    print(matches)
    matches = sorted(matches, key=lambda m: m.distance)
    good_matches = [m for m in matches if m.distance < threshold]

    return good_matches, kp1, kp2, image_gray, template_cv


def get_feature_matches(
    modified_image: Image.Image,
    template_box: tuple[int, int, int, int],
    image: npt.NDArray,
    threshold: float = 100,
) -> tuple[Sequence[cv.DMatch], Sequence[cv.KeyPoint], Sequence[cv.KeyPoint]]:
    """Identifies templates in an image

    Args:
        template (Image.Image): the template to identify in the image.
        image (npt.NDArray): the image to identify the template in. In a opencv NDArray BGR format.
    """

    modified_cv = cv.cvtColor(np.array(modified_image), cv.COLOR_RGB2GRAY)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # template_mask = np.zeros(image.shape[:2])
    # template_mask[
    #     template_box[0] : template_box[2], template_box[1] : template_box[3]
    # ] = 255
    # template_mask = template_mask.astype(np.uint8)

    orb = cv.ORB.create()
    kp1, des1 = orb.detectAndCompute(image_gray, None)
    kp2, des2 = orb.detectAndCompute(
        modified_cv, None
    )  # not using mask on modified image, because shape could be different

    # create BFMatcher object
    bf = cv.BFMatcher()
    matches = bf.match(des1, des2, None)
    matches = sorted(matches, key=lambda m: m.distance)
    good_matches = [m for m in matches if m.distance < threshold]

    x1, y1, x2, y2 = template_box

    """ DEBUG """
    # final_matches = []
    # for m in good_matches:
    #     if x1 <= kp2[m.trainIdx].pt[0] <= x2 and y1 <= kp2[m.trainIdx].pt[1] <= y2:
    #         final_matches.append(m)
    #     debug = cv.drawKeypoints(image_gray, [kp2[m.trainIdx]], None, color=(0,255,0), flags=0)
    #     debug = cv.rectangle(debug, (x1, y1), (x2, y2), (36, 255, 12), 1)
    #     cv.imwrite(".tmp/debug/delete.png",debug)
    """"""

    good_matches = [
        m
        for m in good_matches
        if x1 <= kp1[m.queryIdx].pt[0] <= x2 and y1 <= kp1[m.queryIdx].pt[1] <= y2
    ]

    good_matches = get_repr_cluster(good_matches, kp1, kp2)

    return good_matches, kp1, kp2, image_gray, modified_cv



def get_repr_cluster(matches: list, kp1: list, kp2: list) -> list:
    """Takes as input a list of feature matches, returns a cluster containing only the features that remain shape between eachother in the train image

    Args:
        matches (list): original match list
        kp1 (list): feature query keypoints
        kp2 (list): feature train keypoints

    Returns:
        list: filtered match list with the cluster only
    """


    coords = lambda m: (
        round(kp1[m.queryIdx].pt[0],0),
        round(kp1[m.queryIdx].pt[1],0),
        round(kp2[m.trainIdx].pt[0],0),
        round(kp2[m.trainIdx].pt[1],0),
    )

    clusters = defaultdict(set)
    i = 0
    while i < len(matches):
        m1 = matches[i]
        x1, y1, x1p, y1p = coords(m1)
        for j in range(i + 1, len(matches)):
            m2 = matches[j]
            x2, y2, x2p, y2p = coords(m2)

            # computing Rx and Ry ratios only if x1!=x2 and y1 != y2
            if x1p != x2p and y1p != y2p:
                Rx = abs(x1 - x2) / abs(x1p - x2p)
                Ry = abs(y1 - y2) / abs(y1p - y2p)
                key = tuple(round(x, 5) for x in (Rx, Ry))
                if any(k==0.0 for k in key):
                    continue
                clusters[key].add(m1)  # redundant
                clusters[key].add(m2)

                # """ DEBUG """
                # debug = cv.drawMatches(
                #     image_original_cv,
                #     kp1,
                #     image_modified_wide_cv,
                #     kp2,
                #     list(clusters[key]),
                #     None,
                #     flags=2,
                # )
                # cv.imwrite(".tmp/debug/delete.png", debug)
        i += 1

    return list(max(clusters.values(), key=lambda x: len(x)))


def get_template_matches_perfect(
    template: Image.Image, image: npt.NDArray
) -> list[tuple[int, int, int, int]]:
    """Same as get_template_matching(...), but with exact pixel equality"""
    template = cv.cvtColor(np.array(template), cv.COLOR_RGB2BGR)
    if template.shape[2] == 4:
        template = template[:, :, :3]  # remove alpha if present

    h, w, _ = template.shape

    if image.shape[2] == 4:
        image = image[:, :, :3]

    # slide template over image and compare slices
    matches = []
    for y in range(image.shape[0] - h + 1):
        for x in range(image.shape[1] - w + 1):
            patch = image[y : y + h, x : x + w]
            if np.array_equal(patch, template):
                matches.append((x, y, x + w, y + h))

    return box_similarity_removal(matches)


def parallel_template_match(
    templates: dict[str, Image.Image], image: Image.Image
) -> dict[str, tuple[int, int, int, int]]:
    """Does parallel computation of template match on an image, given a dictionnary of "feature" to image

    Returns:
        dict[str, tuple[int, int, int, int]]: Mapping from the feature to a list of template matches
    """
    image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    with ThreadPoolExecutor() as executor:
        tasks: list[Future] = []
        for template in templates.values():
            tasks.append(executor.submit(get_template_matches, template, image))
        results = []
        for task in tasks:
            results.append(task.result())
    return {feat_name: res for feat_name, res in zip(templates, results)}


def box_similarity_removal(boxes: list[tuple[int, int, int, int]], iou_threshold=0.5):
    """Removes boxes that have too much overlap using IoU

    Args:
        boxes (list[tuple[int, int, int, int]]): The list of boxes to filter

    """
    i = len(boxes) - 1
    while i > 0:
        boxA = boxes[i]
        for j in range(i - 1, -1, -1):
            boxB = boxes[j]
            if IoU(boxA, boxB) > iou_threshold:
                boxes.pop(i)
                i -= 1
                break
        else:
            i -= 1
    return boxes


def IoU(boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def box_size_increase(
    box: tuple[int, int, int, int], max_image: Image.Image, ratio: float = 1.3
) -> tuple[int, int, int, int]:
    """image-dependent box size increase

    Args:
        box (tuple[int, int, int, int]): box to increase
        max_image (Image.Image): reference image
        ratio (float, optional): ratio to use for increasing the box sie 1 is unchanged,2 is the image size. Defaults to 1.3.

    Returns:
        tuple[int, int, int, int]: _description_
    """
    ratio = min(max(ratio, 1), 2)
    w = max_image.width
    h = max_image.height
    ratio = ratio - 1
    return (
        (box[0] - (box[0]) * ratio),
        (box[1] - (box[1]) * ratio),
        (box[2] + (w - box[2]) * ratio),
        (box[3] + (h - box[3]) * ratio),
    )


import numpy as np
from skimage.transform import rotate

def crop_with_box(mask, box):
    """Crop mask using box = (x0,x1,y0,y1)."""
    x0,x1,y0,y1 = box
    return mask[y0:y1, x0:x1]

def rotate_mask(mask, angle):
    """Rotate mask around its center, preserving size."""
    return rotate(mask, angle=angle, resize=True, preserve_range=True).astype(mask.dtype)

def compute_overlap(rotated_mask1, mask2):
    """Center-align smaller mask to larger one, compute overlap."""
    h1, w1 = rotated_mask1.shape
    h2, w2 = mask2.shape

    # Compute canvas size
    H = max(h1, h2)
    W = max(w1, w2)

    def pad_center(mask, H, W):
        h, w = mask.shape
        top = (H - h) // 2
        bottom = H - h - top
        left = (W - w) // 2
        right = W - w - left
        return np.pad(mask, ((top, bottom), (left, right)))

    m1_p = pad_center(rotated_mask1, H, W)
    m2_p = pad_center(mask2, H, W)
    
    # Binary masks assumed: overlap = logical AND
    intersection = np.logical_and(m1_p, m2_p)
    union = np.logical_or(m1_p, m2_p)
    
    return intersection.sum() / union.sum() 