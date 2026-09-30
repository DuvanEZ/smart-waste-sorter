"""Smart Waste Sorter - command-line version (works without a browser).

Interactive menu:
    python app/cli.py
One-off use:
    python app/cli.py photo.jpg another.png some_folder --csv results.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from predictor import WasteClassifier  # noqa: E402
from recycling_guide import GENERAL_NOTE, advice_for  # noqa: E402
from validation import list_folder_images, validate_image_file  # noqa: E402

LINE = "-" * 78


def classify_paths(model: WasteClassifier, paths: list[Path], threshold: float) -> list[dict]:
    """Validate and classify a list of files; returns one result row per file."""
    rows, valid = [], []
    for p in paths:
        v = validate_image_file(p)
        rows.append({"file": str(p), "validation": v})
        if v.ok:
            valid.append(rows[-1])
    predictions = model.predict_batch([r["validation"].image for r in valid], threshold) if valid else []
    for r, pred in zip(valid, predictions):
        r["prediction"] = pred
    return rows


def print_rows(rows: list[dict]) -> None:
    print(LINE)
    print(f"{'File':<34}{'Prediction':<12}{'Confidence':>11}  {'Status':<10}Second choice")
    print(LINE)
    for r in rows:
        name = Path(r["file"]).name
        name = name if len(name) <= 32 else name[:29] + "..."
        if "prediction" not in r:
            print(f"{name:<34}{'-':<12}{'-':>11}  {'REJECTED':<10}{r['validation'].error}")
            continue
        p = r["prediction"]
        status = "OK" if p.is_confident else "UNCERTAIN"
        second = f"{p.top_k[1][0]} ({p.top_k[1][1]:.1%})"
        print(f"{name:<34}{p.label:<12}{p.confidence:>10.1%}  {status:<10}{second}")
        for w in r["validation"].warnings:
            print(f"{'':<34}note: {w}")
    print(LINE)


def print_detail(row: dict) -> None:
    """Full explanation for a single image."""
    if "prediction" not in row:
        print(f"\n  REJECTED: {row['validation'].error}\n")
        return
    p = row["prediction"]
    g = advice_for(p.label)
    print(f"\n  Prediction : {p.label.upper()}  ({p.confidence:.1%} confidence)")
    print("  Status     : " + ("OK" if p.is_confident else
                               f"UNCERTAIN (below {p.threshold:.0%}) - it could also be {p.top_k[1][0]}"))
    print(f"  Dispose in : {g['bin']}")
    for tip in g["tips"]:
        print(f"               - {tip}")
    for w in row["validation"].warnings:
        print(f"  Note       : {w}")
    print("  All classes:")
    for cls, prob in p.top_k:
        print(f"     {cls:<10} {prob:6.1%} {'#' * int(round(prob * 40))}")
    print(f"  ({GENERAL_NOTE})\n")


def save_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "prediction", "confidence", "status", "second_choice", "disposal", "message"])
        for r in rows:
            if "prediction" not in r:
                w.writerow([r["file"], "", "", "REJECTED", "", "", r["validation"].error])
            else:
                p = r["prediction"]
                w.writerow([r["file"], p.label, f"{p.confidence:.4f}", "OK" if p.is_confident else "UNCERTAIN",
                            p.top_k[1][0], advice_for(p.label)["bin"], " ".join(r["validation"].warnings)])
    print(f"Results saved to {out.resolve()}")


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip().strip('"').strip("'")
    except (EOFError, KeyboardInterrupt):
        print()
        return "q"


def interactive(model: WasteClassifier) -> None:
    threshold = model.default_threshold
    acc = f" (test accuracy {model.test_accuracy:.1%})" if model.test_accuracy else ""
    print(f"\n{'=' * 78}\n  SMART WASTE SORTER - command-line version\n  Model: {model.info.get('model_name')}{acc}\n{'=' * 78}")
    while True:
        print(f"\n  1) Classify one image\n  2) Classify every image in a folder\n"
              f"  3) Change the confidence threshold (now {threshold:.0%})\n  4) Help\n  5) Quit")
        choice = ask("Choose an option [1-5]: ")
        if choice in ("5", "q", "Q", "quit", "exit"):
            print("Goodbye!")
            return
        if choice == "1":
            path = ask("Path of the image file: ")
            if not path:
                print("  No path entered.")
                continue
            print_detail(classify_paths(model, [Path(path).expanduser()], threshold)[0])
        elif choice == "2":
            files, error = list_folder_images(ask("Path of the folder: "))
            if error:
                print(f"  {error}")
                continue
            print(f"  Classifying {len(files)} image(s) ...")
            rows = classify_paths(model, files, threshold)
            print_rows(rows)
            out = ask("Save the results as CSV? Enter a file name (or press Enter to skip): ")
            if out and out.lower() != "q":
                out_path = Path(out).expanduser()
                if out_path.suffix.lower() != ".csv":
                    out_path = out_path.with_suffix(".csv")
                try:
                    save_csv(rows, out_path)
                except OSError as exc:
                    print(f"  Could not write the file: {exc.strerror}")
        elif choice == "3":
            value = ask("New threshold between 30 and 95 (%): ").rstrip("%")
            try:
                v = float(value)
                if not 30 <= v <= 95:
                    raise ValueError
                threshold = v / 100
                print(f"  Threshold set to {threshold:.0%}.")
            except ValueError:
                print("  Please enter a number between 30 and 95.")
        elif choice == "4":
            print(f"\n  Supported files: jpg, jpeg, png, bmp, webp, tif, tiff (max 20 MB, at least 32x32 px).\n"
                  f"  The model recognises: {', '.join(model.class_names)}.\n"
                  f"  Confidence = calibrated probability of the predicted class; below the threshold the\n"
                  f"  result is marked UNCERTAIN. Invalid files are REJECTED with the reason.")
        else:
            print("  Please type a number from 1 to 5.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Waste Sorter (command line)")
    parser.add_argument("inputs", nargs="*", help="image files and/or folders (omit for the interactive menu)")
    parser.add_argument("--csv", help="save the results to this CSV file")
    parser.add_argument("--threshold", type=float, help="confidence threshold between 0.3 and 0.95")
    args = parser.parse_args()
    try:
        model = WasteClassifier()
    except Exception as exc:
        sys.exit(f"Could not load the model: {exc}")
    if not args.inputs:
        interactive(model)
        return
    threshold = model.default_threshold
    if args.threshold is not None:
        if not 0.3 <= args.threshold <= 0.95:
            sys.exit("--threshold must be between 0.3 and 0.95")
        threshold = args.threshold
    paths: list[Path] = []
    for item in args.inputs:
        p = Path(item).expanduser()
        if p.is_dir():
            files, error = list_folder_images(p)
            if error:
                print(f"{p}: {error}")
            paths += files
        else:
            paths.append(p)
    rows = classify_paths(model, paths, threshold)
    if len(rows) == 1:
        print_detail(rows[0])
    else:
        print_rows(rows)
    if args.csv:
        save_csv(rows, Path(args.csv))


if __name__ == "__main__":
    main()
