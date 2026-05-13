"""
Generate CLIP text embeddings for COCO 80 categories and save as .pt file.

Usage:
    python tools/generate_coco_text_features.py
    python tools/generate_coco_text_features.py --clip-model ViT-L/14@336px
"""

import os
import argparse
import torch
import clip

COCO_ALL_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a photograph of a {}.",
    "an image of a {}.",
    "a picture of a {}.",
    "a {} in a scene.",
]


def main():
    parser = argparse.ArgumentParser(
        description="Generate CLIP text features for COCO 80 categories"
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
    print(f"Generating features for {len(COCO_ALL_CATEGORIES)} COCO categories...")

    model, _ = clip.load(args.clip_model, device=device)
    model.eval()

    all_features = []
    for cat_name in COCO_ALL_CATEGORIES:
        texts = [t.format(cat_name) for t in PROMPT_TEMPLATES]
        tokens = clip.tokenize(texts).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.mean(dim=0)
            feats = feats / feats.norm()
        all_features.append(feats)
        print(f"  ✓ {cat_name}")

    text_features = torch.stack(all_features).cpu().float()
    feat_dim = text_features.shape[1]

    output_path = os.path.join(args.output_dir, "coco_all_classes.pt")
    torch.save({
        "text_features": text_features,
        "feature_dim": feat_dim,
        "num_classes": len(COCO_ALL_CATEGORIES),
        "class_names": COCO_ALL_CATEGORIES,
        "clip_model": args.clip_model,
        "prompt_templates": PROMPT_TEMPLATES,
    }, output_path)

    print(f"\nDone! Saved {len(COCO_ALL_CATEGORIES)} classes (dim={feat_dim}) → {output_path}")
    print("Now you can set TFE_ENABLE=True in run_coco_gfsod_finetuning.sh")


if __name__ == "__main__":
    main()
