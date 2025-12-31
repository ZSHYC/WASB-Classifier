import os
import shutil
from pathlib import Path
import math

# Script: 为 reports/yolo_only_games.txt 中的每个 dataset_game 建立目录并复制素材

REPORTS_DIR = Path("reports")
YOLO_ONLY_LIST = REPORTS_DIR / "yolo_only_games.txt"
DEST_ROOT = REPORTS_DIR / "yolo_only_review"

# Source roots (可按需修改)
DATASETS_ROOT = Path("datasets")
YOLO_ROOT = Path("yolo_output")
WASB_ROOT = Path("src/wasb_outputs/main")

# thresholds
SKIP_VIDEO_SIZE_BYTES = 200 * 1024 * 1024  # 200 MB, 超过则跳过并记录


def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, stats: dict):
    try:
        mkdir(dst.parent)
        shutil.copy2(src, dst)
        stats['copied'] += 1
    except Exception:
        stats['errors'] += 1


def find_images_in(dirpath: Path):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    files = []
    if not dirpath.exists():
        return files
    for p in dirpath.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def find_videos_in(dirpath: Path):
    exts = {'.mp4', '.avi', '.mov', '.mkv'}
    files = []
    if not dirpath.exists():
        return files
    for p in dirpath.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def find_yolo_results(dataset: str, game: str):
    # typical dir: yolo_output/output_<dataset>/<game>/
    base = YOLO_ROOT / f"output_{dataset}" / game
    # if exists, collect images and videos
    imgs = find_images_in(base)
    vids = find_videos_in(base)
    # also look for comparison subfolders
    for sub in (base.glob('*comparison*')):
        if sub.is_dir():
            imgs += find_images_in(sub)
            vids += find_videos_in(sub)
    return list(set(imgs)), list(set(vids))


def find_wasb_results(dataset: str, game: str):
    base = WASB_ROOT / dataset
    imgs = []
    vids = []
    if not base.exists():
        return imgs, vids
    # files starting with game (e.g., game_2_Clip_1_predictions.csv or images)
    for p in base.rglob(f"{game}*"):
        if p.is_file():
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                imgs.append(p)
            if p.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}:
                vids.append(p)
    # also search any images under directories named game
    for d in base.rglob(game):
        if d.is_dir():
            imgs += find_images_in(d)
            vids += find_videos_in(d)
    return list(set(imgs)), list(set(vids))


def find_originals(dataset: str, game: str):
    base = DATASETS_ROOT / dataset / game
    imgs = find_images_in(base)
    vids = find_videos_in(base)
    return imgs, vids


def process_entry(entry: str, stats_overall: dict):
    entry = entry.strip()
    if not entry:
        return
    # expected format: <dataset>_game_<n>
    # dataset may contain underscores (e.g., maibo_land, maibo_serve_left)
    if '_game_' in entry:
        idx = entry.index('_game_')
        dataset = entry[:idx]
        game = entry[idx+1:]  # keep leading 'game_...'
    else:
        parts = entry.split('_')
        if len(parts) < 2:
            return
        dataset = parts[0]
        game = '_'.join(parts[1:])

    dest_base = DEST_ROOT / entry
    yolo_dest = dest_base / 'yolo'
    wasb_dest = dest_base / 'wasb'

    # create subfolders
    for d in [yolo_dest, wasb_dest]:
        for sub in ['orig', 'results', 'videos']:
            mkdir(d / sub)

    stats = {'copied': 0, 'errors': 0, 'skipped_videos': 0}

    # originals (copy to both orig folders)
    orig_imgs, orig_vids = find_originals(dataset, game)
    for img in orig_imgs:
        dst1 = yolo_dest / 'orig' / img.name
        dst2 = wasb_dest / 'orig' / img.name
        copy_file(img, dst1, stats)
        copy_file(img, dst2, stats)

    for vid in orig_vids:
        if vid.stat().st_size > SKIP_VIDEO_SIZE_BYTES:
            stats['skipped_videos'] += 1
        else:
            dst1 = yolo_dest / 'videos' / vid.name
            dst2 = wasb_dest / 'videos' / vid.name
            copy_file(vid, dst1, stats)
            copy_file(vid, dst2, stats)

    # yolo results
    yolo_imgs, yolo_vids = find_yolo_results(dataset, game)
    for img in yolo_imgs:
        dst = yolo_dest / 'results' / img.name
        copy_file(img, dst, stats)
    for vid in yolo_vids:
        if vid.stat().st_size > SKIP_VIDEO_SIZE_BYTES:
            stats['skipped_videos'] += 1
        else:
            dst = yolo_dest / 'videos' / vid.name
            copy_file(vid, dst, stats)

    # wasb results
    wasb_imgs, wasb_vids = find_wasb_results(dataset, game)
    for img in wasb_imgs:
        dst = wasb_dest / 'results' / img.name
        copy_file(img, dst, stats)
    for vid in wasb_vids:
        if vid.stat().st_size > SKIP_VIDEO_SIZE_BYTES:
            stats['skipped_videos'] += 1
        else:
            dst = wasb_dest / 'videos' / vid.name
            copy_file(vid, dst, stats)

    stats_overall['entries'] += 1
    stats_overall['files_copied'] += stats['copied']
    stats_overall['errors'] += stats['errors']
    stats_overall['skipped_videos'] += stats['skipped_videos']


def main():
    if not YOLO_ONLY_LIST.exists():
        print(f"Missing {YOLO_ONLY_LIST}")
        return
    mkdir(DEST_ROOT)
    lines = [l.strip() for l in YOLO_ONLY_LIST.read_text(encoding='utf-8').splitlines() if l.strip()]
    stats_overall = {'entries': 0, 'files_copied': 0, 'errors': 0, 'skipped_videos': 0}
    for e in lines:
        process_entry(e, stats_overall)

    print(f"Processed {stats_overall['entries']} entries")
    print(f"Files copied: {stats_overall['files_copied']}")
    print(f"Errors: {stats_overall['errors']}")
    print(f"Skipped large videos: {stats_overall['skipped_videos']}")


if __name__ == '__main__':
    main()
