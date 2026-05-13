"""
Extract CLIP text features for VOC/COCO class names and save as .pt file.
These pre-extracted features are used by the TFE (Text-guided Feature Enhancement) module.

Usage:
    python tools/extract_text_features.py --dataset voc --clip-model ViT-B/32 --save-dir ./text_features
    python tools/extract_text_features.py --dataset coco --clip-model ViT-B/32 --save-dir ./text_features
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdcc.evaluation.archs import clip


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

COCO_CLASSES = [
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
    "a {} in the wild.",
    "a clear photo of a {}.",
]


def extract_features(class_names, clip_model_name, device, use_multi_prompt=True):
    model, _ = clip.load(clip_model_name, device=device)
    model.eval()

    all_features = []
    for cls_name in class_names:
        if use_multi_prompt:
            prompts = [t.format(cls_name) for t in PROMPT_TEMPLATES]
        else:
            prompts = [f"a photo of a {cls_name}."]

        tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            text_feat = model.encode_text(tokens).float()  # [P, dim]
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat.mean(dim=0)  # [dim]
            text_feat = text_feat / text_feat.norm()
        all_features.append(text_feat.cpu())

    text_features = torch.stack(all_features, dim=0)  # [C, dim]
    return text_features


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP text features")
    parser.add_argument("--dataset", type=str, required=True, choices=["voc", "coco"])
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--save-dir", type=str, default="./text_features")
    parser.add_argument("--single-prompt", action="store_true",
                        help="Use single prompt instead of multi-prompt ensemble")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset == "voc":
        class_names = VOC_CLASSES
    else:
        class_names = COCO_CLASSES

    print(f"Dataset: {args.dataset} ({len(class_names)} classes)")
    print(f"CLIP model: {args.clip_model}")
    print(f"Multi-prompt: {not args.single_prompt}")
    print(f"Device: {device}")

    text_features = extract_features(
        class_names, args.clip_model, device,
        use_multi_prompt=not args.single_prompt,
    )

    print(f"Text features shape: {text_features.shape}")

    os.makedirs(args.save_dir, exist_ok=True)
    clip_tag = args.clip_model.replace("/", "-")
    save_path = os.path.join(args.save_dir, f"{args.dataset}_{clip_tag}_text_features.pt")

    torch.save({
        "text_features": text_features,
        "class_names": class_names,
        "clip_model": args.clip_model,
        "num_classes": len(class_names),
        "feature_dim": text_features.shape[1],
    }, save_path)

    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
