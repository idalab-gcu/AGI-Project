"""
Visualizes anatomical bounding boxes on Chest X-ray images based on MIMIC-CXR data.
"""

import cv2
import json
import re
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set

# ==========================================
# [CONSTANTS] Colors & Labels
# ==========================================

FIXED_29 = {
    "right lung", "left lung", "right clavicle", "left clavicle",
    "right apical zone", "left apical zone", "mediastinum", "upper mediastinum",
    "abdomen", "spine", "aortic arch", "right atrium", "trachea", "carina",
    "svc", "superior vena cava structure", "right hemidiaphragm", "left hemidiaphragm",
    "right costophrenic angle", "left costophrenic angle",
    "right mid lung zone", "left mid lung zone", "right upper lung zone", "left upper lung zone",
    "right lower lung zone", "left lower lung zone",
    "right hilar structures", "left hilar structures", "cardiac silhouette", "cavoatrial junction"
}
FIXED_29_LOWER = {s.lower() for s in FIXED_29}

COL = {
    "RED": (60, 60, 255), "ORANGE": (32, 165, 255), "GREEN": (80, 200, 120),
    "TEAL": (180, 200, 200), "PURPLE": (200, 100, 255), "PINK": (255, 120, 200),
    "GRAY": (190, 190, 190)
}

GROUP_COLOR = {
    "RIGHT": COL["ORANGE"], "LEFT": COL["GREEN"], 
    "CARDIAC": COL["PINK"], "OTHER": COL["GRAY"]
}

KEYWORD_TO_LABELS = {
    "central venous catheter": ["right atrium", "svc", "superior vena cava structure", "cavoatrial junction"],
    "cvc": ["right atrium", "svc", "superior vena cava structure", "cavoatrial junction"],
    "opacity": ["right lung", "left lung", "right upper lung zone", "left upper lung zone", "right mid lung zone", "left mid lung zone", "right lower lung zone", "left lower lung zone"],
    "pneumothorax": ["right lung", "left lung", "right costophrenic angle", "left costophrenic angle"],
}

def pick_group(label_lower: str) -> str:
    if "right" in label_lower: return "RIGHT"
    if "left" in label_lower: return "LEFT"
    if any(x in label_lower for x in ["cardiac", "atrium", "svc", "aortic"]): return "CARDIAC"
    return "OTHER"

def labels_from_report(text: str) -> Set[str]:
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    keep = set()
    for kw, labels in KEYWORD_TO_LABELS.items():
        if kw in text:
            keep.update([lb for lb in labels if lb in FIXED_29_LOWER])
    return keep

def main(args):
    # 1. Setup Paths
    base_path = Path(args.base_dir)
    img_path = base_path / args.img_rel_path
    
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found at: {img_path}")

    # 2. Load Data
    print(f"Loading image: {img_path.name}")
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]

    # Load Bbox CSV
    print("Loading BBox data...")
    df = pd.read_csv(args.bbox_csv)
    rows = df[(df["study_id"] == args.study_id) & (df["image_id"] == args.image_id)].copy()
    rows["label_lower"] = rows["object_label"].str.lower()

    # Load Report (JSON)
    keep_labels = set()
    if args.report_json:
        print("Filtering labels based on report...")
        with open(args.report_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Handle list or dict format
        recs = [data] if isinstance(data, dict) else data
        rec = next((r for r in recs if int(r.get("study_id", 0)) == args.study_id), {})
        
        report_text = f"{rec.get('gt_report','')} {rec.get('generated_report','')}"
        keep_labels = labels_from_report(report_text)

    # Filter Rows
    target_rows = rows[rows["label_lower"].isin(keep_labels)]
    if target_rows.empty:
        print("No matching labels found in report, showing all bboxes.")
        target_rows = rows

    # 3. Draw
    # Scale logic: MIMIC-CXR original JPGs are usually larger than 224x224.
    # Adjust this logic based on your BBox coordinate system (224 vs Original)
    sx, sy = W/224.0, H/224.0 if rows["x2"].max() <= 260 else (1.0, 1.0)
    
    for _, r in target_rows.iterrows():
        x1, y1 = int(r.x1 * sx), int(r.y1 * sy)
        x2, y2 = int(r.x2 * sx), int(r.y2 * sy)
        color = GROUP_COLOR.get(pick_group(r.label_lower), COL["GRAY"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Optional: Add text label
        # cv2.putText(img, r.object_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 4. Save
    save_path = img_path.parent / f"{args.image_id}_bbox_vis.jpg"
    cv2.imwrite(str(save_path), img)
    print(f" Saved visualization to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw Bounding Boxes on MIMIC-CXR Images")
    
    # Required/Optional Arguments for flexibility
    parser.add_argument("--base_dir", type=str, default="F:/", help="Root directory of dataset")
    parser.add_argument("--study_id", type=int, required=True, help="Study ID (e.g., 59116935)")
    parser.add_argument("--image_id", type=str, required=True, help="DICOM Image ID")
    parser.add_argument("--img_rel_path", type=str, required=True, help="Relative path to the image from base_dir")
    parser.add_argument("--bbox_csv", type=str, required=True, help="Path to bbox coordinates CSV")
    parser.add_argument("--report_json", type=str, default=None, help="Path to report JSON for filtering")

    # Example Usage for Testing (User can uncomment or run via CLI)
    # args = parser.parse_args([
    #     "--study_id", "59116935",
    #     "--image_id", "00005197-869d72f3-66210bf4-fa2c9d83-b613c4e7",
    #     ...
    # ])
    
    args = parser.parse_args()
    main(args)
