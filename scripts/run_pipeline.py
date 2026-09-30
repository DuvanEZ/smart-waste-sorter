"""Run the complete analysis + modelling pipeline from the command line.

Examples
--------
    # Kaggle zip downloaded to your computer
    python scripts/run_pipeline.py --zip ~/Downloads/archive.zip --out outputs

    # already extracted folder, quick test run (few epochs, no small CNN)
    python scripts/run_pipeline.py --data-dir data/raw --out outputs --quick
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waste_classifier.config import BaselineConfig, SplitConfig, TrainConfig  # noqa: E402
from waste_classifier.pipeline import run_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", help="path to the Kaggle archive.zip")
    ap.add_argument("--data-dir", help="folder that already contains the extracted images")
    ap.add_argument("--out", default="outputs", help="output folder (figures, metrics, model)")
    ap.add_argument("--workers", type=int, default=2, help="CPU processes for data loading/analysis")
    ap.add_argument("--no-cnn", action="store_true", help="skip the small CNN baseline")
    ap.add_argument("--quick", action="store_true", help="very short training (pipeline test)")
    ap.add_argument("--head-epochs", type=int)
    ap.add_argument("--finetune-epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    args = ap.parse_args()

    train_cfg, base_cfg = TrainConfig(num_workers=args.workers), BaselineConfig()
    if args.quick:
        train_cfg.head_epochs, train_cfg.finetune_epochs = 1, 1
        base_cfg.cnn_epochs = 1
    if args.head_epochs is not None:
        train_cfg.head_epochs = args.head_epochs
    if args.finetune_epochs is not None:
        train_cfg.finetune_epochs = args.finetune_epochs
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    run_all(zip_path=args.zip, data_dir=args.data_dir, out_dir=args.out, workers=args.workers,
            run_cnn=not args.no_cnn, train_cfg=train_cfg, base_cfg=base_cfg, split_cfg=SplitConfig())


if __name__ == "__main__":
    main()
