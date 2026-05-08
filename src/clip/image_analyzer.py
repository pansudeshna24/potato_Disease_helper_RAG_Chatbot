"""
Stable CLIP Disease Analyzer
"""

import os
import json
import numpy as np
from typing import List, Dict
from PIL import Image
import time

from src.core.logging_utils import setup_logger, log_timing

logger = setup_logger('image_analyzer')

# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

REFERENCE_IMAGES_DIR = os.path.join(
    PROJECT_ROOT,
    "reference_images"
)

CLIP_INDEX_PATH = os.path.join(
    REFERENCE_IMAGES_DIR,
    "clip_reference_index.faiss"
)

CLIP_METADATA_PATH = os.path.join(
    REFERENCE_IMAGES_DIR,
    "clip_reference_metadata.json"
)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# =========================================================
# DISEASE PROMPTS
# =========================================================

DISEASE_TEXT_PROMPTS = {
    "Late_Blight": [
        "Potato leaf infected with late blight showing dark lesions",
        "Potato plant with Phytophthora infestans infection",
    ],

    "Early_Blight": [
        "Potato leaf with concentric ring brown lesions",
        "Potato infected with Alternaria solani",
    ],

    "Healthy": [
        "Healthy green potato plant",
        "Healthy potato leaf without disease",
    ],
}

DISPLAY_NAMES = {
    "Late_Blight": "Late Blight",
    "Early_Blight": "Early Blight",
    "Healthy": "Healthy",
}


class CLIPDiseaseAnalyzer:

    def __init__(self, load_faiss_index=True):

        import torch
        from transformers import CLIPProcessor, CLIPModel

        logger.info("Initializing CLIPDiseaseAnalyzer...")

        self.torch = torch

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = CLIPModel.from_pretrained(
            CLIP_MODEL_NAME
        ).to(self.device)

        self.processor = CLIPProcessor.from_pretrained(
            CLIP_MODEL_NAME
        )

        self.model.eval()

        self.disease_names = list(
            DISEASE_TEXT_PROMPTS.keys()
        )

        # SAFE PROMPT ENCODING
        self.disease_text_embeddings = (
            self._encode_disease_prompts()
        )

        self.ref_index = None
        self.ref_metadata = []

        if load_faiss_index:
            self._load_reference_index()

        logger.info(
            "CLIPDiseaseAnalyzer initialized successfully"
        )

    # =====================================================
    # SAFE NORMALIZATION
    # =====================================================

    def _normalize_embedding(self, emb):

        if not isinstance(emb, self.torch.Tensor):

            if hasattr(emb, "pooler_output"):
                emb = emb.pooler_output

            elif isinstance(emb, tuple):
                emb = emb[0]

        emb = emb.float()

        # FORCE 2D
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)

        emb = emb / (
            emb.norm(dim=-1, keepdim=True) + 1e-8
        )

        return emb

    # =====================================================
    # ENCODE TEXT PROMPTS
    # =====================================================

    def _encode_disease_prompts(self):

        logger.info("Encoding disease text prompts...")

        embeddings = []

        for disease_name in self.disease_names:

            prompts = DISEASE_TEXT_PROMPTS[disease_name]

            inputs = self.processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
            }

            with self.torch.no_grad():

                text_features = (
                    self.model.get_text_features(**inputs)
                )

                text_features = (
                    self._normalize_embedding(
                        text_features
                    )
                )

            # AVERAGE PROMPTS
            avg_emb = text_features.mean(dim=0)

            avg_emb = avg_emb.squeeze()

            avg_emb = avg_emb.cpu().numpy()

            avg_emb = avg_emb.astype(np.float32)

            # FINAL NORMALIZATION
            avg_emb = avg_emb / (
                np.linalg.norm(avg_emb) + 1e-8
            )

            embeddings.append(avg_emb)

        # FORCE CONSISTENT SHAPE
        embeddings = [
            e.reshape(-1)
            for e in embeddings
        ]

        return np.stack(embeddings).astype(np.float32)

    # =====================================================
    # ENCODE IMAGE
    # =====================================================

    def encode_image(self, image: Image.Image):

        image = image.convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with self.torch.no_grad():

            image_features = (
                self.model.get_image_features(**inputs)
            )

            image_features = (
                self._normalize_embedding(
                    image_features
                )
            )

        image_features = (
            image_features
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        image_features = image_features.reshape(-1)

        image_features = image_features / (
            np.linalg.norm(image_features) + 1e-8
        )

        return image_features

    # =====================================================
    # LOAD FAISS
    # =====================================================

    def _load_reference_index(self):

        import faiss

        if (
            not os.path.exists(CLIP_INDEX_PATH)
            or
            not os.path.exists(CLIP_METADATA_PATH)
        ):

            logger.warning(
                "Reference image index not found"
            )

            return

        self.ref_index = faiss.read_index(
            CLIP_INDEX_PATH
        )

        with open(
            CLIP_METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            self.ref_metadata = json.load(f)

        logger.info(
            f"Loaded reference index with "
            f"{self.ref_index.ntotal} images"
        )

    # =====================================================
    # ZERO SHOT
    # =====================================================

    def zero_shot_classify(
        self,
        image: Image.Image,
        top_k=5
    ):

        image_emb = self.encode_image(image)

        similarities = (
            image_emb @ self.disease_text_embeddings.T
        )

        ranked = np.argsort(-similarities)

        results = []

        for idx in ranked[:top_k]:

            disease = self.disease_names[idx]

            results.append({
                "disease": disease,
                "display_name": DISPLAY_NAMES.get(
                    disease,
                    disease
                ),
                "similarity": float(similarities[idx]),
            })

        return results

    # =====================================================
    # MAIN ANALYSIS
    # =====================================================

    def analyze_image(
        self,
        image: Image.Image
    ):

        results = self.zero_shot_classify(image)

        top = results[0]

        rag_query = (
            f"Potato plant appears affected by "
            f"{top['display_name']}. "
            f"Provide symptoms, causes, and treatment."
        )

        return {
            "prediction": top["disease"],
            "display_name": top["display_name"],
            "confidence": round(top["similarity"], 4),
            "all_candidates": results,
            "rag_query": rag_query,
        }