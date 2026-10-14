"""Dual-stage vision pipeline for leaf detection and disease classification."""


import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
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
CROP_MODEL_PATH = PROJECT_ROOT / "Model" / "Crop_Feature_Extractor.keras"
DISEASE_MODEL_PATH = PROJECT_ROOT / "Model" / "Tomato_Potato_Disease_Model.keras"

# 7 Disease Classes in true model training order
CLASS_NAMES = [
    "Potato Early Blight",
    "Potato Healthy",
    "Potato Late Blight",
    "Tomato Early Blight",
    "Tomato Healthy",
    "Tomato Late Blight",
    "Tomato Mosaic Virus"
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
    "Potato Healthy": {
        "crop": "Potato",
        "disease": "Healthy Foliage",
        "scientific_name": "Solanum tuberosum",
        "severity": "Healthy",
        "severity_color": "#2E7D32",
        "is_healthy": True,
        "summary": "Plant foliage appears vibrant, green, and free from active visible lesions.",
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
    "Tomato Early Blight": {
        "crop": "Tomato",
        "disease": "Early Blight",
        "scientific_name": "Alternaria solani",
        "severity": "Moderate",
        "severity_color": "#E65100",
        "is_healthy": False,
        "summary": "Dark brown circular spots with concentric rings, typically starting on lower leaves.",
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
}


class CropVisionPipeline:
    """
    Manages loading and running inference for the multi-stage vision pipeline:
    1. Leaf Detector (Leaf vs Not Leaf)
    2. Crop Identifier (Potato vs Tomato)
    3. Disease Classifier (7 Disease Classes)
    """

    def __init__(self):
        self.leaf_model = None
        self.crop_model = None
        self.disease_model = None

    def load_models(self):
        """Load Keras models if not already in memory."""
        if self.leaf_model is None:
            if not LEAF_MODEL_PATH.exists():
                raise FileNotFoundError(f"Leaf model not found at: {LEAF_MODEL_PATH}")
            print(f"[MODEL] Loading Leaf Detector: {LEAF_MODEL_PATH.name}...")
            self.leaf_model = keras.models.load_model(str(LEAF_MODEL_PATH))

        if self.crop_model is None and CROP_MODEL_PATH.exists():
            print(f"[MODEL] Loading Crop Identifier: {CROP_MODEL_PATH.name}...")
            self.crop_model = keras.models.load_model(str(CROP_MODEL_PATH))

        if self.disease_model is None:
            if not DISEASE_MODEL_PATH.exists():
                raise FileNotFoundError(f"Disease model not found at: {DISEASE_MODEL_PATH}")
            print(f"[MODEL] Loading Disease Model: {DISEASE_MODEL_PATH.name}...")
            self.disease_model = keras.models.load_model(str(DISEASE_MODEL_PATH))

    def check_is_leaf(self, image: Image.Image, threshold: float = 0.5) -> Tuple[bool, float, float]:
        """Validates if image contains a plant leaf, rejecting non-plant surfaces."""
        self.load_models()

        # Convert to RGB, resize to (224, 224), normalize to [0, 1]
        rgb_img = image.convert("RGB").resize((224, 224))
        arr = (np.array(rgb_img, dtype=np.float32) / 255.0)[np.newaxis, ...]

        # Raw sigmoid: 0.0 = Leaf, 1.0 = Non-leaf ("Others")
        raw_pred = float(self.leaf_model.predict(arr, verbose=0)[0][0])
        model_leaf_prob = float(np.clip(1.0 - raw_pred, 0.0, 1.0))

        # 1. Botanical Chromatic Analysis (Excess Green, Chlorosis Halo, Olive Lesions)
        rgb_sample = image.convert("RGB").resize((150, 150))
        rgb_arr = np.array(rgb_sample, dtype=float)
        r, g, b = rgb_arr[:, :, 0], rgb_arr[:, :, 1], rgb_arr[:, :, 2]

        exg = 2 * g - r - b
        green_mask = (exg > 5) & (g > 30) & (g > r * 0.85)
        green_pct = float(np.mean(green_mask) * 100)

        yellow_mask = (r > 50) & (g > 50) & (b < (r + g) * 0.45) & (np.abs(r - g) < 55)
        olive_mask = (g > b * 1.05) & (r > b * 1.05) & (g > 25) & (b < 145)

        foliage_mask = green_mask | yellow_mask | olive_mask
        foliage_pct = float(np.mean(foliage_mask) * 100)

        # Human skin detector (faces, arms holding objects):
        skin_mask = (r > 80) & (g > 40) & (b > 20) & (r > g + 15) & (g > b) & (exg < -15)
        skin_pct = float(np.mean(skin_mask) * 100)

        # 2. Botanical Texture Variance (veins, margins vs flat walls/cloth/paper):
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        lap_var = float(np.var(gray[1:, :] - gray[:-1, :]))

        # 3. Model Consensus (Disease Model):
        arr2 = np.array(image.convert("RGB").resize((300, 300), Image.Resampling.NEAREST), dtype=np.float32)[np.newaxis, ...]
        probs2 = self.disease_model.predict(arr2, verbose=0)[0]
        top_dis_conf = float(np.max(probs2))

        # Rejection guards for non-leaf images
        if lap_var < 3.0:
            return False, 0.05, raw_pred
        if skin_pct > 25.0 and green_pct < 5.0:
            return False, 0.02, raw_pred
        if foliage_pct < 5.0:
            return False, 0.03, raw_pred

        # Positive leaf verification
        if raw_pred <= 0.60:
            return True, model_leaf_prob, raw_pred
        if foliage_pct >= 12.0 and top_dis_conf >= 0.35:
            computed_conf = float(min(0.96, (foliage_pct / 100.0) * 0.35 + top_dis_conf * 0.65))
            return True, max(0.72, computed_conf), raw_pred
        if green_pct >= 18.0 and lap_var >= 8.0:
            return True, 0.75, raw_pred
        if foliage_pct >= 10.0 and raw_pred <= 0.85:
            return True, 0.65, raw_pred

        return False, model_leaf_prob, raw_pred

    def predict_disease(self, image: Image.Image, crop_filter: Optional[str] = None) -> Dict[str, Any]:
        """Classify leaf disease across 7 classes with optional crop filter."""
        self.load_models()

        # Preprocess image to (300, 300) matching disease detector model
        rgb_img = image.convert("RGB").resize((300, 300), Image.Resampling.NEAREST)
        arr = np.array(rgb_img, dtype=np.float32)[np.newaxis, ...]

        probs = self.disease_model.predict(arr, verbose=0)[0]
        raw_top_idx = int(np.argmax(probs))
        raw_top_class = CLASS_NAMES[raw_top_idx]

        pot_prob_sum = float(sum(probs[i] for i, c in enumerate(CLASS_NAMES) if c.startswith("Potato")))
        tom_prob_sum = float(sum(probs[i] for i, c in enumerate(CLASS_NAMES) if c.startswith("Tomato")))

        # Crop identification and disease class determination
        if crop_filter in ["Potato", "Tomato"]:
            effective_crop = crop_filter
            crop_conf_val = 100.0
            is_auto = False
            matching_indices = [i for i, c in enumerate(CLASS_NAMES) if c.startswith(effective_crop)]
            sub_probs = [probs[i] for i in matching_indices]
            best_sub_rel_idx = int(np.argmax(sub_probs))
            best_sub_idx = matching_indices[best_sub_rel_idx]
            top_class = CLASS_NAMES[best_sub_idx]
            sub_sum = float(sum(sub_probs))
            top_prob = float(probs[best_sub_idx] / sub_sum) if sub_sum > 0 else float(probs[best_sub_idx])
        else:
            top_class = raw_top_class
            top_prob = float(probs[raw_top_idx])
            effective_crop = "Potato" if top_class.startswith("Potato") else "Tomato"
            is_auto = True

            if self.crop_model is not None:
                crop_img = image.convert("RGB").resize((224, 224), Image.Resampling.NEAREST)
                crop_arr = np.array(crop_img, dtype=np.float32)[np.newaxis, ...]
                crop_preds = self.crop_model.predict(crop_arr, verbose=0)[0]
                pot_crop_prob = float(crop_preds[0])
                tom_crop_prob = float(crop_preds[1])
                crop_conf_val = round(max(pot_crop_prob, tom_crop_prob) * 100, 1)
            else:
                crop_conf_val = round((pot_prob_sum if effective_crop == "Potato" else tom_prob_sum) * 100, 1)

        meta = DISEASE_METADATA.get(top_class, {
            "crop": effective_crop,
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
            "crop": effective_crop,
            "auto_detected_crop": effective_crop,
            "crop_confidence": crop_conf_val,
            "is_auto_crop": is_auto,
            "disease_name": meta["disease"],
            "scientific_name": meta["scientific_name"],
            "severity": meta["severity"],
            "severity_color": meta["severity_color"],
            "is_healthy": meta["is_healthy"],
            "summary": meta["summary"],
            "confidence": round(top_prob * 100, 1),
            "breakdown": breakdown,
        }
