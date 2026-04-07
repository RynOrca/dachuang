import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_step(description: str, cmd: list, cwd=None, critical=True) -> int:
    print(f"\n{'='*60}")
    print(f"[WEEK_1] {description}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=cwd or str(ROOT_DIR))
    rc = result.returncode
    status = "OK" if rc == 0 else f"FAILED (rc={rc})"
    print(f"[{status}]")
    if critical and rc != 0:
        print(f"\n[ABORT] Critical step failed: {description}")
        sys.exit(rc)
    return rc


def main():
    parser = argparse.ArgumentParser(
        description="WEEK_1 full evaluation pipeline: quality baseline & data closed-loop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full automatic run (assumes eval_gt already populated):
  python scripts/run_week1_eval.py --skip_gt_setup

  # Run everything including GT generation from source photos:
  python scripts/run_week1_eval.py

  # Dry-run to see what would execute:
  python scripts/run_week1_eval.py --dry_run
""",
    )
    parser.add_argument("--input_dir", type=str, default="eval_inputs", help="Eval input folder")
    parser.add_argument("--output_dir", type=str, default="eval_outputs/week1_baseline", help="Output folder")
    parser.add_argument("--gt_dir", type=str, default="eval_gt", help="Ground-truth image folder")
    parser.add_argument("--labels_csv", type=str, default="eval_labels/labels.csv", help="OCR labels CSV path")
    parser.add_argument("--model_path", type=str, default=None, help="Diffusion model path (auto-resolved if omitted)")
    parser.add_argument("--methods", type=str, default="bicubic,realesrgan,diffusion,diffusion_realesrgan", help="Methods to compare")
    parser.add_argument("--preset", type=str, default="text-balanced", help="Quality preset")
    parser.add_argument("--skip_gt_setup", action="store_true", help="Skip GT image generation (if already done)")
    parser.add_argument("--skip_ocr", action="store_true", help="Skip OCR evaluation step")
    parser.add_argument("--lpips", action="store_true", default=True, help="Enable LPIPS metric")
    parser.add_argument("--no_lpips", dest="lpips", action="store_false", help="Disable LPIPS metric")
    parser.add_argument("--dry_run", action="store_true", help="Print commands only without executing")
    parser.add_argument("--fail_fast", action="store_true", help="Stop on first failure")
    args = parser.parse_args()

    steps_executed = 0
    steps_passed = 0

    print("=" * 60)
    print("  WEEK_1: 建立质量基线与数据闭环")
    print("  Quality Baseline & Data Closed-Loop Pipeline")
    print("=" * 60)
    print(f"  input_dir:   {args.input_dir}")
    print(f"  output_dir:  {args.output_dir}")
    print(f"  gt_dir:      {args.gt_dir}")
    print(f"  labels_csv:  {args.labels_csv}")
    print(f"  methods:     {args.methods}")
    print(f"  preset:      {args.preset}")

    if args.dry_run:
        print("\n[DRY RUN] Commands below will be executed:")

    STEP1 = "Generate eval_gt/ from source images (tools/setup_eval_gt.py)"
    if not args.skip_gt_setup:
        cmd = [PYTHON, "tools/setup_eval_gt.py"]
        if not args.dry_run:
            rc = run_step(STEP1, cmd, critical=False)
            steps_executed += 1
            if rc == 0:
                steps_passed += 1
        else:
            print(f"\n  [WOULD RUN] {STEP1}")
            print(f"    {' '.join(cmd)}")
    else:
        print(f"\n  [SKIP] {STEP1} (--skip_gt_setup)")

    STEP2 = "Validate labels.csv gt_text is filled"
    if not args.dry_run:
        import csv
        labels_path = ROOT_DIR / args.labels_csv
        empty_rows = []
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not (row.get("gt_text") or "").strip():
                        empty_rows.append(row.get("image_name", "?"))
        if empty_rows:
            print(f"\n{'='*60}")
            print(f"[{STEP2}] WARNING")
            print(f"  {len(empty_rows)} rows in {args.labels_csv} have empty gt_text:")
            for name in empty_rows[:5]:
                print(f"    - {name}")
            print(f"  Please edit {args.labels_csv} and fill in correct text.")
            print(f"  Then re-run with --skip_gt_setup")
            print(f"{'='*60}")
            if not args.skip_gt_setup:
                sys.exit(1)
        else:
            print(f"\n  [OK] {STEP2} - all gt_text fields are populated")
            steps_passed += 1
        steps_executed += 1
    else:
        print(f"\n  [WOULD CHECK] {STEP2}")

    STEP3 = "Precheck: validate all paths and data consistency"
    precheck_cmd = [
        PYTHON, "run_all.py", "precheck",
        "--input_dir", args.input_dir,
        "--model_path", args.model_path or "model/diffusion_fix_sanity_best.pth",
        "--gt_dir", args.gt_dir,
        "--gt_csv", args.labels_csv,
        "--require_gt",
        "--require_ocr_csv",
        "--strict_eval_set",
    ]
    if not args.dry_run:
        rc = run_step(STEP3, precheck_cmd, critical=True)
        steps_executed += 1
        if rc == 0:
            steps_passed += 1
    else:
        print(f"\n  [WOULD RUN] {STEP3}")
        print(f"    {' '.join(precheck_cmd)}")

    model_path_arg = ["--model_path", args.model_path] if args.model_path else []
    lpips_arg = ["--lpips"] if args.lpips else []

    STEP4 = f"Batch evaluation: {args.methods}"
    batch_cmd = [
        PYTHON, "run_all.py", "batch",
        "--input_dir", args.input_dir,
        "--output_dir", args.output_dir,
        "--methods", args.methods,
        "--preset", args.preset,
        *model_path_arg,
        "--gt_dir", args.gt_dir,
        *lpips_arg,
        "--strict_precheck",
        "--auto_report",
    ] + (["--fail_fast"] if args.fail_fast else [])
    if not args.dry_run:
        rc = run_step(STEP4, batch_cmd, critical=True)
        steps_executed += 1
        if rc == 0:
            steps_passed += 1
    else:
        print(f"\n  [WOULD RUN] {STEP4}")
        print(f"    {' '.join(batch_cmd)}")

    ocr_steps = 0
    ocr_passed = 0
    if not args.skip_ocr:
        STEP5 = "OCR evaluation (CER/WER)"
        pred_dir = f"{args.output_dir}/diffusion"
        ocr_cmd = [
            PYTHON, "run_all.py", "ocr-eval",
            "--pred_dir", pred_dir,
            "--gt_csv", args.labels_csv,
            "--ocr_backend", "paddle",
            "--lang", "ch",
            "--output_csv", f"{args.output_dir}/ocr_detail.csv",
            "--output_json", f"{args.output_dir}/ocr_summary.json",
        ]
        if not args.dry_run:
            rc = run_step(STEP5, ocr_cmd, critical=False)
            ocr_steps += 1
            if rc == 0:
                ocr_passed += 1
        else:
            print(f"\n  [WOULD RUN] {STEP5}")
            print(f"    {' '.join(ocr_cmd)}")

        STEP6 = "Final report generation (HTML + JSON)"
        report_cmd = [
            PYTHON, "run_all.py", "report",
            "--output_dir", args.output_dir,
            "--summary_json", f"{args.output_dir}/batch_summary.json",
            "--report_html", f"{args.output_dir}/week1_report.html",
            "--report_json", f"{args.output_dir}/week1_report.json",
            "--title", "WEEK_1 Baseline Report",
            "--ocr_summary_json", f"{args.output_dir}/ocr_summary.json",
            "--ocr_detail_csv", f"{args.output_dir}/ocr_detail.csv",
        ]
        if not args.dry_run:
            rc = run_step(STEP6, report_cmd, critical=False)
            ocr_steps += 1
            if rc == 0:
                ocr_passed += 1
        else:
            print(f"\n  [WOULD RUN] {STEP6}")
            print(f"    {' '.join(report_cmd)}")

    print("\n" + "=" * 60)
    total = steps_executed + ocr_steps
    passed = steps_passed + ocr_passed
    print(f"WEEK_1 PIPELINE SUMMARY")
    print(f"  Total steps:  {total}")
    print(f"  Passed:       {passed}")
    print(f"  Failed:       {total - passed}")
    print(f"  Output dir:   {args.output_dir}/")
    print(f"  Artifacts:")
    print(f"    - batch_summary.json  (visual metrics: PSNR/SSIM/LPIPS)")
    print(f"    - metrics.csv         (per-image per-method metrics)")
    print(f"    - top_*_samples.json  (TopN worst samples)")
    print(f"    - ocr_summary.json    (CER/WER aggregates)")
    print(f"    - ocr_detail.csv      (per-image CER/WER)")
    if not args.skip_ocr:
        print(f"    - week1_report.html   (full visual report)")
    print("=" * 60)

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
