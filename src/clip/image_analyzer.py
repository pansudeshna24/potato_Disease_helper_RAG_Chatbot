# =========================================================
# Stable CLIP Disease Analyzer (Improved)
# =========================================================

import os
import json
import numpy as np
from typing import List, Dict
from PIL import Image
import time

from src.core.logging_utils import setup_logger, log_timing
from src.cnn.cnn_classifier import CNNDiseaseClassifier


logger = setup_logger('image_analyzer')

# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

REFERENCE_IMAGES_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "image_reference_index"
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

    "Potato___Late_blight": [
        "Potato leaf infected with late blight showing dark lesions",
        "Potato plant with Phytophthora infestans infection",
    ],

    "Potato___Early_blight": [
        "Potato leaf with concentric ring brown lesions",
        "Potato infected with Alternaria solani",
    ],

    "Potato___healthy": [
        "Healthy green potato plant",
        "Healthy potato leaf without disease",
    ],

    "Dry Rot": [
        "Potato tuber infected with dry rot disease",
        "Dry rotting potato tuber",
    ],

    "Pink Rot": [
        "Potato tuber affected by pink rot",
        "Pink rot disease in potato tuber",
    ],

    "Black Scurf": [
        "Potato tuber infected with black scurf",
        "Black fungal crust on potato tuber",
    ],

    "Blackleg": [
        "Potato stem affected by blackleg disease",
        "Blackleg symptoms in potato plant",
    ],

    "Common Scab": [
        "Potato tuber with common scab disease",
        "Rough scab lesions on potato",
    ],
}

DISPLAY_NAMES = {

    "Potato___Late_blight": "Late Blight",
    "Potato___Early_blight": "Early Blight",
    "Potato___healthy": "Healthy",

    "Dry Rot": "Dry Rot",
    "Pink Rot": "Pink Rot",
    "Black Scurf": "Black Scurf",
    "Blackleg": "Blackleg",
    "Common Scab": "Common Scab",
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

        # =====================================
        # LOAD CNN CLASSIFIER
        # =====================================

        self.cnn_classifier = CNNDiseaseClassifier()

        # =====================================
        # LOAD CLIP MODEL
        # =====================================

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

        # =====================================================
        # ENCODE TEXT PROMPTS
        # =====================================================

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

            avg_emb = text_features.mean(dim=0)

            avg_emb = avg_emb.squeeze()

            avg_emb = avg_emb.cpu().numpy()

            avg_emb = avg_emb.astype(np.float32)

            avg_emb = avg_emb / (
                np.linalg.norm(avg_emb) + 1e-8
            )

            embeddings.append(avg_emb)

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
    # IMPROVED CLASSIFICATION
    # =====================================================

    def zero_shot_classify(
        self,
        image: Image.Image,
        top_k=5
    ):

        image_emb = self.encode_image(image)

        # =====================================================
        # TEXT SIMILARITY
        # =====================================================

        text_similarities = (
            image_emb @ self.disease_text_embeddings.T
        )

        # =====================================================
        # IMAGE RETRIEVAL FROM FAISS
        # =====================================================

        faiss_scores = {}

        if self.ref_index is not None:

            D, I = self.ref_index.search(
                np.array([image_emb]).astype(np.float32),
                top_k
            )

            for score, idx in zip(D[0], I[0]):

                if idx >= len(self.ref_metadata):
                    continue

                metadata = self.ref_metadata[idx]

                disease = metadata.get(
                    "disease",
                    "Unknown"
                )

                if disease not in faiss_scores:
                    faiss_scores[disease] = []

                faiss_scores[disease].append(
                    float(score)
                )

        # =====================================================
        # COMBINE TEXT + IMAGE SCORES
        # =====================================================

        combined_results = []

        for idx, disease in enumerate(self.disease_names):

            text_score = float(
                text_similarities[idx]
            )

            image_score = 0.0

            if disease in faiss_scores:
                image_score = np.mean(
                    faiss_scores[disease]
                )

            # weighted fusion
            final_score = (
                0.35 * text_score
                +
                0.65 * image_score
            )

            combined_results.append({

                "disease": disease,

                "display_name": DISPLAY_NAMES.get(
                    disease,
                    disease
                ),

                "similarity": final_score,

                "text_score": round(
                    text_score,
                    4
                ),

                "image_score": round(
                    image_score,
                    4
                ),
            })

        combined_results = sorted(
            combined_results,
            key=lambda x: x["similarity"],
            reverse=True
        )

        return combined_results[:top_k]

    # =====================================================
    # MAIN ANALYSIS
    # =====================================================

    def analyze_image(self, image):

    # =====================================
    # CNN PREDICTION
    # =====================================

        cnn_result = self.cnn_classifier.predict(image)

        cnn_prediction = cnn_result["prediction"]
        cnn_confidence = cnn_result["confidence"]

        # =====================================
        # CLIP RETRIEVAL
        # =====================================

        clip_results = self.zero_shot_classify(
            image,
            top_k=5
        )

        top_clip = clip_results[0]

        clip_prediction = top_clip["display_name"]

        clip_confidence = round(
            top_clip["similarity"] * 100,
            2
        )

        # =====================================
        # HYBRID DECISION
        # =====================================

        final_prediction = cnn_prediction

        final_confidence = cnn_confidence

        # boost confidence if both agree
        if clip_prediction.lower() in cnn_prediction.lower():

            final_confidence = min(
                99,
                cnn_confidence + 10
            )

        # uncertainty handling
        if final_confidence < 45:

            final_prediction = "Uncertain"

        rag_query = (
            f"Potato plant appears affected by "
            f"{final_prediction}. "
            f"Provide symptoms, causes, and treatment."
        )

        return {
            "prediction": final_prediction,
            "display_name": final_prediction,
            "confidence": final_confidence,
            "cnn_prediction": cnn_prediction,
            "cnn_confidence": cnn_confidence,
            "clip_prediction": clip_prediction,
            "clip_confidence": clip_confidence,
            "top_candidates": clip_results,
            "rag_query": rag_query
        }