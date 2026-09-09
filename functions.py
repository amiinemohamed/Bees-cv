import cv2
import numpy as np
from PIL import Image
from skimage import morphology
from skimage.morphology import skeletonize
from skimage.feature import corner_harris, corner_peaks


def detect_junction_points(
    image,
    max_dimension=800,
    min_distance=10,
    threshold_rel=0.3,
    return_intermediate=False,
):
    # ---- 1. Normalize input to an RGB numpy array -----------------------
    if isinstance(image, str):
        pil_img = Image.open(image)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        img_array = np.array(pil_img)
    elif isinstance(image, Image.Image):
        pil_img = image
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        img_array = np.array(pil_img)
    elif isinstance(image, np.ndarray):
        img_array = image
        if img_array.ndim == 2:
            # Grayscale array -> make 3-channel so downstream code is uniform
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    else:
        raise TypeError(
            "image must be a file path (str), a PIL.Image.Image, or a numpy.ndarray"
        )
 
    # ---- 2. Grayscale + resize -------------------------------------------
    gray_full = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    height, width = gray_full.shape
    scale = min(max_dimension / height, max_dimension / width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    image_gray = cv2.resize(
        gray_full, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
    )
 
    # ---- 3. DoG edge map -> Otsu binary -> cleanup -----------------------
    blur1 = cv2.GaussianBlur(image_gray, (0, 0), sigmaX=7)
    blur2 = cv2.GaussianBlur(image_gray, (0, 0), sigmaX=0.5)
    dog = cv2.subtract(blur1, blur2)
    dog_abs = cv2.convertScaleAbs(dog)
 
    _, thresh = cv2.threshold(
        dog_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
 
    bilateral = cv2.bilateralFilter(thresh, d=30, sigmaColor=75, sigmaSpace=75)
    median = cv2.medianBlur(bilateral, 3)
 
    # ---- 4. Dilate to strengthen/connect edges ---------------------------
    kernel = np.ones((5, 5), np.uint8)
    dilated_img = cv2.dilate(median, kernel, iterations=2)
 
    # ---- 5. Auto-crop rows with no significant structure -----------------
    rect_width = max(1, dilated_img.shape[1] // 10)
    if hasattr(morphology, "footprint_rectangle"):
        selem = morphology.footprint_rectangle((1, rect_width))
    else:
        selem = morphology.rectangle(1, rect_width)
    img_opened = morphology.opening(dilated_img, selem)
 
    hist = np.sum(img_opened, axis=1)
    non_zero = np.where(hist > 0)[0]
 
    if non_zero.size > 0:
        top_bound = int(np.round(np.min(non_zero) * 0.95))
        bottom_bound = int(np.round(np.max(non_zero) * 1.05))
        bottom_bound = min(bottom_bound, dilated_img.shape[0])
    else:
        top_bound, bottom_bound = 0, dilated_img.shape[0]
 
    img_cropped = dilated_img[top_bound:bottom_bound, :]
 
    # ---- 6. Skeletonize ---------------------------------------------------
    binary = (img_cropped > 0).astype(np.uint8)
    skeleton = skeletonize(binary)
 
    # ---- 7. Junction/corner detection on the skeleton ----------------------
    coords = corner_peaks(
        corner_harris(skeleton),
        min_distance=min_distance,
        threshold_rel=threshold_rel,
    )
 
    junction_points = [tuple(coord) for coord in coords]
 
    if return_intermediate:
        intermediates = {
            "gray": image_gray,
            "binary_mask": dilated_img,
            "cropped_mask": img_cropped,
            "skeleton": skeleton,
            "crop_offset": (top_bound, 0),
        }
        return junction_points, intermediates
 
    return junction_points








