import os
import csv
from pathlib import Path

"""
按 game 聚合比较 YOLO 与 WASB 的检测结果。
输出：
 - reports/game_compare.csv (dataset,game,yolo_has,wasb_has,only)
 - reports/yolo_only_games.txt (dataset_game 每行)
 - reports/wasb_only_games.txt (dataset_game 每行)

假设：
 - YOLO 输出目录结构: yolo_output/output_<dataset>/game_<n>/ (存在 comparison/ 或 任何文件则视为有结果)
 - WASB 输出目录: src/wasb_outputs/main/<dataset>/ 下可能包含以 game_<n> 开头的文件/文件夹或 predictions CSV
 - 也会以 datasets/<dataset>/game_<n>/ 作为游戏列表的来源
"""


def list_games_from_datasets(root_datasets, datasets_to_check=None):
    games = {}
    root = Path(root_datasets)
    if not root.exists():
        return games
    for ds in sorted(root.iterdir()):
        if not ds.is_dir():
            continue
        ds_name = ds.name
        if datasets_to_check and ds_name not in datasets_to_check:
            continue
        gs = []
        for g in sorted(ds.iterdir()):
            if g.is_dir() and g.name.startswith("game"):
                gs.append(g.name)
        if gs:
            games[ds_name] = gs
    return games


def yolo_game_has(yolo_root, dataset, game_name):
    # try path: yolo_output/output_<dataset>/<game_name>
    out_root = Path(yolo_root)
    dir1 = out_root / f"output_{dataset}" / game_name
    if dir1.exists() and dir1.is_dir():
        # if any file or subdir exists inside (non-empty) -> has results
        for p in dir1.rglob("*"):
            if p.is_file():
                return True
    # fallback: check output_{dataset}.zip unpack or other naming - not implemented
    return False


def wasb_game_has(wasb_root, dataset, game_name):
    # Rule per user: find CSV files under wasb_root/<dataset>/ matching game_name (not game folder)
    # e.g. game_2_Clip_1_predictions.csv
    root = Path(wasb_root) / dataset
    if not root.exists():
        return False

    # collect candidate csv files that start with game_name
    csvs = list(root.glob(f"{game_name}*.csv"))
    if not csvs:
        return False

    # For each csv (each corresponds to a game/clip), determine fraction of imgs missing
    # If >=50% images missing => this game considered "no detection" for WASB
    # Otherwise considered has detection
    game_has_any = False
    for csvf in csvs:
        try:
            with csvf.open("r", encoding='utf-8', errors='ignore') as fh:
                reader = csv.reader(fh)
                headers = None
                rows = []
                for r in reader:
                    if not r:
                        continue
                    if headers is None:
                        # detect header row if non-numeric
                        is_header = any(not _looks_like_number(x) for x in r[:3])
                        if is_header:
                            headers = [c.strip() for c in r]
                            continue
                    rows.append(r)
                if not rows:
                    continue
                # find x,y column indices
                x_idx, y_idx = _find_xy_indices(headers, rows)
                if x_idx is None or y_idx is None:
                    # cannot find xy -> skip this csv
                    continue
                total = 0
                missing = 0
                for r in rows:
                    total += 1
                    xv = r[x_idx].strip() if x_idx < len(r) else ''
                    yv = r[y_idx].strip() if y_idx < len(r) else ''
                    if not _is_number_str(xv) or not _is_number_str(yv):
                        missing += 1
                if total == 0:
                    continue
                if missing / total < 0.5:
                    # less than half missing -> this csv indicates detections present
                    game_has_any = True
                    break
        except Exception:
            continue
    return game_has_any


def _looks_like_number(s):
    try:
        float(str(s).strip())
        return True
    except Exception:
        return False


def _is_number_str(s):
    # treat '-inf' or 'inf' or non-numeric as not a normal number
    try:
        v = float(str(s))
        if v == float('inf') or v == float('-inf'):
            return False
        return True
    except Exception:
        return False


def _find_xy_indices(headers, rows):
    # headers may be None; try to find column indices for x and y
    if headers:
        lx = None
        ly = None
        for i, h in enumerate(headers):
            hn = h.lower()
            if 'x' in hn and lx is None:
                lx = i
            if 'y' in hn and ly is None:
                ly = i
        if lx is not None and ly is not None:
            return lx, ly
    # fallback: try to detect numeric columns in rows
    # examine first row to find numeric columns
    first = rows[0]
    num_idxs = [i for i, v in enumerate(first) if _is_number_str(v)]
    if len(num_idxs) >= 2:
        # assume last two numeric cols are x,y
        return num_idxs[-2], num_idxs[-1]
    # else cannot determine
    return None, None


def ensure_reports_dir():
    Path("reports").mkdir(exist_ok=True)


def main():
    # config - 可按需改路径
    datasets_root = Path("datasets")
    yolo_root = Path("yolo_output")
    wasb_root = Path("src/wasb_outputs/main")

    games_map = list_games_from_datasets(datasets_root)
    ensure_reports_dir()

    csv_path = Path("reports") / "game_compare.csv"
    yolo_only_path = Path("reports") / "yolo_only_games.txt"
    wasb_only_path = Path("reports") / "wasb_only_games.txt"

    yolo_only = []
    wasb_only = []

    with csv_path.open("w", newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(["dataset", "game", "yolo_has", "wasb_has", "only"])
        for ds, games in sorted(games_map.items()):
            for g in sorted(games):
                y_has = yolo_game_has(yolo_root, ds, g)
                w_has = wasb_game_has(wasb_root, ds, g)
                only = "none"
                if y_has and not w_has:
                    only = "yolo"
                    yolo_only.append(f"{ds}_{g}")
                elif w_has and not y_has:
                    only = "wasb"
                    wasb_only.append(f"{ds}_{g}")
                writer.writerow([ds, g, int(y_has), int(w_has), only])

    with yolo_only_path.open("w", encoding='utf-8') as f:
        for it in yolo_only:
            f.write(it + "\n")

    with wasb_only_path.open("w", encoding='utf-8') as f:
        for it in wasb_only:
            f.write(it + "\n")

    print(f"Wrote {csv_path}")
    print(f"YOLO-only games: {len(yolo_only)} -> {yolo_only_path}")
    print(f"WASB-only games: {len(wasb_only)} -> {wasb_only_path}")


if __name__ == '__main__':
    main()
