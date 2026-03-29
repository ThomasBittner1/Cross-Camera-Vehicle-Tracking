import cv2
import numpy as np
import general_utils



def calculate_histograms_multiple(crops):
    distributed_crops = general_utils.get_distributed_items(crops)
    histograms = []
    for crop in distributed_crops:
        histogram = compute_vehicle_color_histogram(crop)
        if histogram is not None:
            histograms.append(histogram)
    return histograms



def compute_vehicle_color_histogram(crop, h_bins=24, s_bins=16):
    if crop is None or crop.size == 0:
        return None

    h, w = crop.shape[:2]
    center_crop = crop[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
    if center_crop.size == 0:
        center_crop = crop

    hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 40, 30]), np.array([180, 255, 255]))
    hist = cv2.calcHist([hsv], [0, 1], mask, [h_bins, s_bins], [0, 180, 0, 256])

    if hist is None or float(hist.sum()) == 0.0:
        return None

    hist = cv2.normalize(hist, None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist.astype(np.float32)


def compare_histograms(query_hist, gallery_hists):
    if query_hist is None or gallery_hists is None:
        return None, None

    best_idx = None
    best_score = None
    for idx, gallery_hist in enumerate(gallery_hists):
        if gallery_hist is None:
            continue

        distance = cv2.compareHist(
            query_hist.astype(np.float32),
            gallery_hist.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA,
        )
        similarity = 1.0 - float(distance)

        if best_score is None or similarity > best_score:
            best_idx = idx
            best_score = similarity

    return best_idx, best_score




def blur_score(img):
    """
    Returns a blur score for an image using variance of Laplacian.
    Higher = sharper, lower = blurrier.
    Works with BGR or grayscale images.
    """
    import cv2
    import numpy as np

    if img is None or img.size == 0:
        return 0.0

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())