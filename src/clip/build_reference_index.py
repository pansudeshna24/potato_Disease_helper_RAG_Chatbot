"""
Reference Image Index Builder

Scans data/image_reference_index/{Category}/ folders,
encodes each image with CLIP,
and builds a FAISS index for fast image-to-image similarity search.

Usage:
    python -m src.clip.build_reference_index

Output:
    data/image_reference_index/clip_reference_index.faiss
    data/image_reference_index/clip_reference_metadata.json
"""

import os
import sys
import json
import numpy as np
from typing import Dict
from PIL import Image

# ============================================================
# Ensure project root is on path
# ============================================================

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ============================================================
# Configuration
# ============================================================

REFERENCE_IMAGES_DIR = os.path.join(
    project_root,
    "data",
    "image_reference_index"
)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp"
}

MIN_IMAGE_SIZE = 50

INDEX_OUTPUT_PATH = os.path.join(
    REFERENCE_IMAGES_DIR,
    "clip_reference_index.faiss"
)

METADATA_OUTPUT_PATH = os.path.join(
    REFERENCE_IMAGES_DIR,
    "clip_reference_metadata.json"
)

# ============================================================
# Main Builder Function
# ============================================================

def build_reference_index():

    import torch
    import faiss
    from transformers import CLIPProcessor, CLIPModel

    print("\n" + "=" * 70)
    print("      CLIP REFERENCE IMAGE INDEX BUILDER")
    print("=" * 70)

    print(f"\nReference image folder:")
    print(f"{REFERENCE_IMAGES_DIR}")

    # ========================================================
    # Validate dataset folder
    # ========================================================

    if not os.path.exists(REFERENCE_IMAGES_DIR):
        print("\nERROR:")
        print(f"Folder not found:\n{REFERENCE_IMAGES_DIR}")
        return

    # ========================================================
    # Find disease categories
    # ========================================================

    categories = []

    for item in sorted(os.listdir(REFERENCE_IMAGES_DIR)):

        item_path = os.path.join(REFERENCE_IMAGES_DIR, item)

        # Skip hidden/system files
        if item.startswith("."):
            continue

        # Skip generated files
        if item.endswith(".faiss") or item.endswith(".json"):
            continue

        if os.path.isdir(item_path):
            categories.append(item)

    if not categories:
        print("\nERROR: No disease category folders found.")
        return

    print(f"\nFound {len(categories)} disease categories:\n")

    total_images = 0

    for category in categories:

        category_path = os.path.join(
            REFERENCE_IMAGES_DIR,
            category
        )

        image_count = len([
            f for f in os.listdir(category_path)
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
        ])

        total_images += image_count

        status = "✓"

        if image_count < 5:
            status = "⚠ LOW IMAGE COUNT"

        print(f"{category:<30} {image_count:>4} images   {status}")

    print(f"\nTotal Images: {total_images}")

    if total_images == 0:
        print("\nERROR: No valid images found.")
        return

    # ========================================================
    # Load CLIP
    # ========================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading CLIP model on {device}...")

    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)

    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    model.eval()

    print("CLIP model loaded successfully.")

    # ========================================================
    # Encode Images
    # ========================================================

    embeddings = []
    metadata = []

    processed = 0
    skipped = 0

    print("\nEncoding images...\n")

    for category in categories:

        category_path = os.path.join(
            REFERENCE_IMAGES_DIR,
            category
        )

        image_files = sorted([
            f for f in os.listdir(category_path)
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
        ])

        for image_file in image_files:

            image_path = os.path.join(
                category_path,
                image_file
            )

            try:

                image = Image.open(image_path).convert("RGB")

                # Skip tiny/corrupt images
                if (
                    image.size[0] < MIN_IMAGE_SIZE or
                    image.size[1] < MIN_IMAGE_SIZE
                ):
                    print(f"Skipping tiny image: {image_file}")
                    skipped += 1
                    continue

                # Prepare CLIP input
                inputs = processor(
                    images=image,
                    return_tensors="pt"
                )

                inputs = {
                    k: v.to(device)
                    for k, v in inputs.items()
                }

                # Generate embedding
                with torch.no_grad():

                    features = model.get_image_features(**inputs)

                    features = features / features.norm(
                        dim=-1,
                        keepdim=True
                    )



                embedding = (
                    features.squeeze(0)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                embeddings.append(embedding)

                metadata.append({
                    "image_path": image_path,
                    "image_name": image_file,
                    "disease": category
                })

                processed += 1

                if processed % 25 == 0:
                    print(f"Processed {processed}/{total_images}")

            except Exception as e:

                print(f"\nERROR processing:")
                print(image_path)
                print(str(e))

                skipped += 1

    # ========================================================
    # Final Validation
    # ========================================================

    print(f"\nSuccessfully encoded: {len(embeddings)}")
    print(f"Skipped images: {skipped}")

    if len(embeddings) == 0:
        print("\nERROR: No embeddings generated.")
        return

    # ========================================================
    # Build FAISS Index
    # ========================================================

    print("\nBuilding FAISS index...")

    embedding_matrix = np.stack(embeddings).astype(np.float32)

    embedding_dim = embedding_matrix.shape[1]

    index = faiss.IndexFlatIP(embedding_dim)

    index.add(embedding_matrix)

    print(
        f"FAISS index created with "
        f"{index.ntotal} vectors"
    )

    # ========================================================
    # Save Files
    # ========================================================

    print("\nSaving files...")

    faiss.write_index(index, INDEX_OUTPUT_PATH)

    with open(
        METADATA_OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nSUCCESSFULLY SAVED:")
    print(f"\nFAISS INDEX:")
    print(INDEX_OUTPUT_PATH)

    print(f"\nMETADATA:")
    print(METADATA_OUTPUT_PATH)

    # ========================================================
    # Distribution Summary
    # ========================================================

    print("\n" + "=" * 70)
    print("CATEGORY DISTRIBUTION")
    print("=" * 70)

    label_counts: Dict[str, int] = {}

    for item in metadata:

        disease = item["disease"]

        label_counts[disease] = (
            label_counts.get(disease, 0) + 1
        )

    for disease, count in sorted(
        label_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(f"{disease:<30} {count:>4}")

    print("\n" + "=" * 70)
    print("REFERENCE IMAGE INDEX BUILD COMPLETE")
    print("=" * 70)

# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    build_reference_index()