import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np


def l2_normalize(x, axis=1, eps=1e-10):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norms + eps)


def cosine_similarity(query, gallery):
    query = np.asarray(query, dtype=np.float32)
    gallery = np.asarray(gallery, dtype=np.float32)

    if query.ndim == 1:
        query = query[None, :]
    if gallery.ndim == 1:
        gallery = gallery[None, :]

    query = l2_normalize(query)
    gallery = l2_normalize(gallery)

    return query @ gallery.T


def cosine_distance(query, gallery):
    return 1.0 - cosine_similarity(query, gallery)


def find_closest_embedding(query, gallery):
    gallery = np.asarray(gallery, dtype=np.float32)
    if gallery.ndim == 1:
        gallery = gallery[None, :]
    if len(gallery) == 0:
        return None, None

    scores = cosine_similarity(query, gallery)
    if scores.ndim == 2:
        scores = scores[0]

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    return best_idx, best_score


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


def compare_color_histograms(query_hist, gallery_hists):
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


class EmbeddingGenerator:
    def __init__(self):
        # 1. Setup Device (Gaming Laptop GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"EmbeddingGenerator is using: {self.device}")

        # 2. Load and Prepare Model
        # Using ResNet18 is often better for real-time tracking on laptops than ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        self.model.to(self.device)
        self.model.eval()

        # 3. Preprocessing
        self.preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def get_embeddings(self, crops):
        """
        frame: Full BGR image
        bboxes: List or array of [x1, y1, x2, y2]
        """
        if len(crops) == 0:
            return None

        batch_tensors = []

        for crop in crops:

            if crop.size == 0:
                continue

            # Convert to RGB -> PIL -> Tensor
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            batch_tensors.append(self.preprocess(pil_img))

        if not batch_tensors:
            return None

        # 4. Create a single batch and move it to GPU
        # Shape becomes: [N, 3, 224, 224] where N is number of cars
        input_batch = torch.stack(batch_tensors).to(self.device)

        # 5. Run inference on all crops simultaneously
        embeddings = self.model(input_batch)

        # Flatten and return as a numpy array
        # Shape: [N, 2048]
        return embeddings.view(embeddings.size(0), -1).cpu().numpy()

    @staticmethod
    def normalize_embeddings(embeddings):
        return l2_normalize(embeddings)

    @staticmethod
    def compare_embeddings(query, gallery):
        return cosine_similarity(query, gallery)

    @staticmethod
    def compare_embedding_distance(query, gallery):
        return cosine_distance(query, gallery)

    @staticmethod
    def find_closest(query, gallery):
        return find_closest_embedding(query, gallery)
