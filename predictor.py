# ============================================================
# CROPCARE AI — MODEL PREDICTOR (DUAL MODEL PIPELINE)
# ============================================================
# 1. Model 1 (Leaf_Detector_v3.keras):
#    Detects if the uploaded image is a valid plant leaf.
#
# 2. Model 2 (Tomato_Potato_Disease_Model.keras):
#    If a leaf is verified, classifies among 7 disease classes:
#      - Potato Early Blight
#      - Potato Late Blight
#      - Potato Healthy
#      - Tomato Early Blight
#      - Tomato Late Blight
#      - Tomato Mosaic Virus
#      - Tomato Healthy
# ============================================================

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image

# Ensure safe terminal stdout on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import keras

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Model file paths
LEAF_MODEL_PATH = PROJECT_ROOT / "Model" / "Leaf_Detector_v3.keras"
DISEASE_MODEL_PATH = PROJECT_ROOT / "Model" / "Tomato_Potato_Disease_Model.keras"

# 7 Disease Classes in alphabetical folder order (standard Keras order)
CLASS_NAMES = [
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Mosaic Virus",
    "Tomato Healthy"
]

# Disease details & severity mappings
DISEASE_METADATA = {
    "Potato Early Blight": {
        "crop": "Potato",
        "disease": "Early Blight",
        "scientific_name": "Alternaria solani",
        "severity": "Moderate",
        "severity_color": "#E65100",
        "is_healthy": False,
        "summary": "Fungal infection forming concentric target-board dark rings on leaves.",
    },
    "Potato Late Blight": {
        "crop": "Potato",
        "disease": "Late Blight",
        "scientific_name": "Phytophthora infestans",
        "severity": "High / Urgent",
        "severity_color": "#C62828",
        "is_healthy": False,
        "summary": "Aggressive water-soaked dark patches with white fuzz underneath in humid weather.",
    },
    "Potato Healthy": {
        "crop": "Potato",
        "disease": "Healthy Foliage",
        "scientific_name": "Solanum tuberosum",
        "severity": "Healthy",
        "severity_color": "#2E7D32",
        "is_healthy": True,
        "summary": "Plant foliage appears vibrant, green, and free from active visible lesions.",
    },
    "Tomato Early Blight": {
        "crop": "Tomato",
        "disease": "Early Blight",
        "scientific_name": "Alternaria solani",
        "severity": "Moderate",
        "severity_color": "#E65100",
        "is_healthy": False,
        "summary": "Dark brown circular spots with concentric rings, typically starting on lower leaves.",
    },
    "Tomato Late Blight": {
        "crop": "Tomato",
        "disease": "Late Blight",
        "scientific_name": "Phytophthora infestans",
        "severity": "High / Urgent",
        "severity_color": "#C62828",
        "is_healthy": False,
        "summary": "Rapid dark greasy lesions that cause rapid foliage collapse in wet conditions.",
    },
    "Tomato Mosaic Virus": {
        "crop": "Tomato",
        "disease": "Mosaic Virus (ToMV)",
        "scientific_name": "Tobamovirus",
        "severity": "High",
        "severity_color": "#AD1457",
        "is_healthy": False,
        "summary": "Mottled light and dark green mosaic pattern, leaf curling, and stunted shoot growth.",
    },
    "Tomato Healthy": {
        "crop": "Tomato",
        "disease": "Healthy Foliage",
        "scientific_name": "Solanum lycopersicum",
        "severity": "Healthy",
        "severity_color": "#2E7D32",
        "is_healthy": True,
        "summary": "Healthy uniform green foliage with no detectable signs of rot or virus.",
    },
}


class CropVisionPipeline:
    """
    Manages loading and running inference for the dual-model pipeline.
    """

    def __init__(self):
        self.leaf_model = None
        self.disease_model = None

    def load_models(self):
        """Load both Keras models if not already in memory."""
        if self.leaf_model is None:
            if not LEAF_MODEL_PATH.exists():
                raise FileNotFoundError(f"Leaf model not found at: {LEAF_MODEL_PATH}")
            print(f"[MODEL] Loading Leaf Detector: {LEAF_MODEL_PATH.name}...")
            self.leaf_model = keras.models.load_model(str(LEAF_MODEL_PATH))

        if self.disease_model is None:
            if not DISEASE_MODEL_PATH.exists():
                raise FileNotFoundError(f"Disease model not found at: {DISEASE_MODEL_PATH}")
            print(f"[MODEL] Loading Disease Model: {DISEASE_MODEL_PATH.name}...")
            self.disease_model = keras.models.load_model(str(DISEASE_MODEL_PATH))

    def check_is_leaf(self, image: Image.Image, threshold: float = 0.5) -> Tuple[bool, float, float]:
        """
        Step 1: Check if the given PIL image contains a valid plant leaf.
        Combines deep learning model (Leaf_Detector_v3.keras) with plant foliage
        chromatic verification to reject selfies, indoor rooms, clothes, and non-plants.

        Returns:
            (is_leaf: bool, leaf_confidence: float, raw_score: float)
        """
        self.load_models()

        # Convert to RGB and resize to (224, 224)
        rgb_img = image.convert("RGB").resize((224, 224))
        arr = np.array(rgb_img, dtype=np.float32)[np.newaxis, ...]

        # Raw sigmoid output: 1.0 = Leaf pattern, 0.0 = Non-leaf
        raw_pred = float(self.leaf_model.predict(arr, verbose=0)[0][0])

        # Chromatic plant foliage verification (Excess Green + Chlorosis/Blight + Olive lesions)
        rgb_arr = np.array(rgb_img, dtype=float)
        r, g, b = rgb_arr[:, :, 0], rgb_arr[:, :, 1], rgb_arr[:, :, 2]

        # 1. Vegetation excess green (2G - R - B)
        exg = 2 * g - r - b
        green_mask = (exg > 10) & (g > 35)

        # 2. Chlorosis & Early Blight yellow tissue (high R, high G, low B)
        yellow_mask = (r > 55) & (g > 55) & (b < (r + g) * 0.38) & (np.abs(r - g) < 55)

        # 3. Late Blight olive/brown leaf tissue
        olive_mask = (g > b * 1.1) & (r > b * 1.1) & (g > 30) & (b < 125)

        foliage_mask = green_mask | yellow_mask | olive_mask
        foliage_pct = float(np.mean(foliage_mask) * 100)

        # Genuine crop leaves have significant foliage (>20% to 90%).
        # Non-leaf images (selfies, rooms, clothes) have foliage < 10%.
        if foliage_pct < 15.0:
            leaf_prob = raw_pred * (foliage_pct / 20.0)
            is_leaf = False
        else:
            leaf_prob = raw_pred
            is_leaf = leaf_prob >= threshold

        return is_leaf, leaf_prob, raw_pred

    def predict_disease(self, image: Image.Image) -> Dict[str, Any]:
        """
        Step 2: Classify leaf disease across the 7 supported classes.

        Returns:
            Dict containing predicted class, crop, severity, confidence, and all class probabilities.
        """
        self.load_models()

        # Convert to RGB and resize to (300, 300)
        rgb_img = image.convert("RGB").resize((300, 300))
        arr = np.array(rgb_img, dtype=np.float32)[np.newaxis, ...]

        # Softmax probabilities over 7 classes
        probs = self.disease_model.predict(arr, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        top_class = CLASS_NAMES[top_idx]

        meta = DISEASE_METADATA.get(top_class, {
            "crop": "Crop",
            "disease": top_class,
            "scientific_name": "",
            "severity": "Unknown",
            "severity_color": "#455A64",
            "is_healthy": "Healthy" in top_class,
            "summary": "",
        })

        # Format probability distribution
        breakdown = []
        for idx, cls_name in enumerate(CLASS_NAMES):
            breakdown.append({
                "class_name": cls_name,
                "probability": float(probs[idx]),
                "percentage": round(float(probs[idx]) * 100, 1),
            })
        # Sort by probability descending
        breakdown.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "class_name": top_class,
            "crop": meta["crop"],
            "disease_name": meta["disease"],
            "scientific_name": meta["scientific_name"],
            "severity": meta["severity"],
            "severity_color": meta["severity_color"],
            "is_healthy": meta["is_healthy"],
            "summary": meta["summary"],
            "confidence": round(top_prob * 100, 1),
            "breakdown": breakdown,
        }
