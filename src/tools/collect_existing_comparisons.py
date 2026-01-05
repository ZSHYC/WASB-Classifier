import csv
import shutil
from pathlib import Path
import argparse


ALLOWED_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def find_in_dir_by_stem(dirpath: Path, stem: str):
    if not dirpath.exists():
        return None
    for ext in ALLOWED_EXTS:
        p = dirpath / (stem + ext)
        if p.exists():
            return p
    # try case-insensitive glob
    for p in dirpath.iterdir():
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS and p.stem.lower() == stem.lower():
            return p
    return None


def find_wasb_image(wasb_root: Path, dataset: str, game: str, stem: str):
    ds_dir = wasb_root / dataset
    if not ds_dir.exists():
        return None
    # folders may include Clip suffix like game_28_Clip_1; match prefix
    for sub in ds_dir.iterdir():
        if not sub.is_dir():
            continue
        if sub.name.lower().startswith(game.lower()):
            found = find_in_dir_by_stem(sub, stem)
            if found:
                return found
    # fallback: search recursively in ds_dir
    for p in ds_dir.rglob(stem + ".*"):
        if p.suffix.lower() in ALLOWED_EXTS:
            return p
    return None


def find_yolo_image(yolo_root: Path, dataset: str, game: str, stem: str):
    comp_dir = yolo_root / dataset / game / "comparison"
    if comp_dir.exists() and comp_dir.is_dir():
        return find_in_dir_by_stem(comp_dir, stem)
    # fallback: try game folder itself
    game_dir = yolo_root / dataset / game
    return find_in_dir_by_stem(game_dir, stem)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--comparison-csv", default=Path("reports/detection_comparison.csv"), type=Path)
    p.add_argument("--wasb-root", default=Path("wasb_outputs"), type=Path)
    p.add_argument("--yolo-root", default=Path("yolo_output"), type=Path)
    p.add_argument("--out-dir", default=Path("reports/comparisons"), type=Path)
    args = p.parse_args()

    comp_csv = args.comparison_csv
    wasb_root = args.wasb_root
    yolo_root = args.yolo_root
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    with comp_csv.open("r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            status = (row.get("status") or "").strip()
            if status not in ("only_wasb", "only_yolo"):
                continue

            ds = row.get("dataset")
            game = row.get("game")
            img = row.get("image")
            if not (ds and game and img):
                skipped += 1
                continue

            stem = Path(img).stem

            src = None
            if status == "only_wasb":
                src = find_wasb_image(wasb_root, ds, game, stem)
            else:
                src = find_yolo_image(yolo_root, ds, game, stem)

            if not src:
                skipped += 1
                continue

            dest_dir = out_dir / ds / game
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / (stem + "_" + status + src.suffix)
            try:
                shutil.copy2(src, dest)
                copied += 1
            except Exception:
                skipped += 1

    print(f"Copied: {copied}, Skipped: {skipped}")


if __name__ == "__main__":
    main()


# python d:\Personal\Desktop\WASB-SBDT\src\tools\collect_existing_comparisons.py --comparison-csv "D:\Personal\Desktop\WASB-SBDT\reports\detection_comparison.csv" --wasb-root "D:\Personal\Desktop\WASB-SBDT\wasb_outputs" --yolo-root "D:\Personal\Desktop\yolo\yolo_output" --out-dir "D:\Personal\Desktop\WASB-SBDT\reports\comparisons_existing"