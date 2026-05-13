"""
Generate CLIP text embeddings for VOC categories and save as .pt files.

Usage:
    python tools/generate_clip_text_features.py --split 1
    python tools/generate_clip_text_features.py --split 1 2 3   # all splits at once
"""

import os
import argparse
import torch
import clip

PASCAL_VOC_ALL_CATEGORIES = {
    1: [
        "aeroplane", "bicycle", "boat", "bottle", "car",
        "cat", "chair", "diningtable", "dog", "horse",
        "person", "pottedplant", "sheep", "train", "tvmonitor",
        "bird", "bus", "cow", "motorbike", "sofa",
    ],
    2: [
        "bicycle", "bird", "boat", "bus", "car",
        "cat", "chair", "diningtable", "dog", "motorbike",
        "person", "pottedplant", "sheep", "train", "tvmonitor",
        "aeroplane", "bottle", "cow", "horse", "sofa",
    ],
    3: [
        "aeroplane", "bicycle", "bird", "bottle", "bus",
        "car", "chair", "cow", "diningtable", "dog",
        "horse", "person", "pottedplant", "train", "tvmonitor",
        "boat", "cat", "motorbike", "sheep", "sofa",
    ],
}

PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a photograph of a {}.",
    "an image of a {}.",
    "a picture of a {}.",
    "a {} in a scene.",
]


def generate_text_features(split_id, clip_model_name, device):
    model, _ = clip.load(clip_model_name, device=device)
    model.eval()

    categories = PASCAL_VOC_ALL_CATEGORIES[split_id]
    all_features = []

    for cat_name in categories:
        texts = [t.format(cat_name) for t in PROMPT_TEMPLATES]
        tokens = clip.tokenize(texts).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.mean(dim=0)
            feats = feats / feats.norm()
        all_features.append(feats)

    text_features = torch.stack(all_features).cpu().float()
    return text_features, categories, text_features.shape[1]


def main():
    parser = argparse.ArgumentParser(
        description="Generate CLIP text features for VOC categories"
    )
    parser.add_argument(
        "--split", type=int, nargs="+", default=[1, 2, 3],
        choices=[1, 2, 3], help="VOC split IDs to generate"
    )
    parser.add_argument(
        "--clip-model", type=str, default="ViT-L/14@336px",
        help="CLIP model name (default: ViT-L/14@336px, 768-dim)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="text_features",
        help="Output directory"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, CLIP model: {args.clip_model}")

    for split_id in args.split:
        text_features, categories, feat_dim = generate_text_features(
            split_id, args.clip_model, device
        )

        output_path = os.path.join(args.output_dir, f"voc_split{split_id}.pt")
        torch.save({
            "text_features": text_features,
            "feature_dim": feat_dim,
            "num_classes": len(categories),
            "class_names": categories,
            "clip_model": args.clip_model,
            "prompt_templates": PROMPT_TEMPLATES,
        }, output_path)

        print(f"[Split {split_id}] Saved {len(categories)} classes "
              f"(dim={feat_dim}) → {output_path}")
        print(f"  Classes: {categories}")

    print("\nDone! Use these paths in TCC_TEXT_FEATURES_PATH config.")


if __name__ == "__main__":
    main()
