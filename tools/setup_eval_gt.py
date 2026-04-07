import argparse
import csv
import os
import sys

import cv2


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(ROOT_DIR, "eval_inputs", "manifest.csv")
DEFAULT_GT_DIR = os.path.join(ROOT_DIR, "eval_gt")
DEFAULT_LABELS_DIR = os.path.join(ROOT_DIR, "eval_labels")
DEFAULT_LABELS_CSV = os.path.join(DEFAULT_LABELS_DIR, "labels.csv")


def load_manifest(manifest_path):
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def crop_gt_from_source(source_path, x, y, w, h):
    img = cv2.imread(source_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read source image: {source_path}")
    crop = img[y : y + h, x : x + w]
    return crop


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def build_labels_csv(gt_dir, manifest_rows, output_path):
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "gt_text", "source_image", "notes"])
        for row in manifest_rows:
            crop_name = row["crop_name"]
            gt_img_path = os.path.join(gt_dir, crop_name)
            exists = "✅" if os.path.exists(gt_img_path) else "❌ missing"
            writer.writerow([crop_name, "", row["source_image"], exists])
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate eval_gt/ ground-truth crops from source images per manifest.csv.")
    parser.add_argument("--manifest", type=str, default=MANIFEST_PATH, help="Path to eval_inputs/manifest.csv")
    parser.add_argument("--gt_dir", type=str, default=DEFAULT_GT_DIR, help="Output directory for GT crops")
    parser.add_argument("--labels_csv", type=str, default=DEFAULT_LABELS_CSV, help="Output path for labels.csv template")
    parser.add_argument("--dry_run", action="store_true", help="Print plan without writing files")
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"[ERROR] Manifest not found: {args.manifest}")
        sys.exit(1)

    manifest_rows = load_manifest(args.manifest)
    if not manifest_rows:
        print("[ERROR] Manifest is empty.")
        sys.exit(1)

    print(f"[setup_eval_gt] Found {len(manifest_rows)} entries in manifest.")
    gt_dir = ensure_dir(args.gt_dir)

    success_count = 0
    fail_count = 0
    for row in manifest_rows:
        crop_name = row["crop_name"]
        source = row["source_image"]
        try:
            x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
        except (ValueError, KeyError) as e:
            print(f"  [SKIP] {crop_name}: invalid coords in manifest: {e}")
            fail_count += 1
            continue

        out_path = os.path.join(gt_dir, crop_name)
        if os.path.exists(out_path) and not args.dry_run:
            print(f"  [EXISTS] {crop_name} -> {out_path}")
            success_count += 1
            continue

        if args.dry_run:
            print(f"  [PLAN] {crop_name}: crop ({x},{y},{w},{h}) from {source}")
            success_count += 1
            continue

        try:
            crop = crop_gt_from_source(source, x, y, w, h)
            cv2.imwrite(out_path, crop)
            print(f"  [OK] {crop_name} -> {out_path} ({crop.shape[1]}x{crop.shape[0]})")
            success_count += 1
        except Exception as e:
            print(f"  [FAIL] {crop_name}: {e}")
            fail_count += 1

    labels_path = build_labels_csv(gt_dir, manifest_rows, args.labels_csv)
    print(f"\n[setup_eval_gt] Done: {success_count} ok, {fail_count} failed.")
    print(f"  GT dir:      {gt_dir}")
    print(f"  Labels CSV:  {labels_path}")
    print(f"\n[IMPORTANT] Please edit {labels_path} and fill in the 'gt_text' column with correct text content.")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
